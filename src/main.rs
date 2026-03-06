use std::sync::Arc;
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::RenderPassBeginInfo;
use vulkano::command_buffer::SubpassBeginInfo;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::Pipeline;
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass};
use vulkano_util::context::{VulkanoConfig, VulkanoContext};
use vulkano_util::window::{VulkanoWindows, WindowDescriptor};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::WindowId;

// --- 核心模块声明 ---
// 必须在这里声明 mod，编译器才能找到文件夹里的代码
mod math;
mod renderer;
mod shaders;

// 使用项目新定义的结构
use crate::renderer::buffer::MyVertex;
use crate::shaders::vs;

struct App {
    context: Arc<VulkanoContext>,
    windows: VulkanoWindows,
    vertex_buffer: Option<Subbuffer<[MyVertex]>>,
    pipeline: Option<Arc<GraphicsPipeline>>,
    render_pass: Option<Arc<RenderPass>>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // 1. 创建窗口
        self.init_window(event_loop);
        // 2. 准备数据（使用各模块封装好的逻辑）
        self.init_resources();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                if let Err(e) = self.draw() {
                    eprintln!("渲染出错: {:?}", e);
                }
            }
            _ => (),
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = self.windows.get_primary_window() {
            window.request_redraw();
        }
    }
}

impl App {
    fn init_window(&mut self, event_loop: &ActiveEventLoop) {
        if self.windows.get_primary_window().is_none() {
            let descriptor = WindowDescriptor {
                width: 1024.0,
                height: 768.0,
                title: "Rust Vulkano Modular Engine".to_string(),
                ..WindowDescriptor::default()
            };
            self.windows
                .create_window(event_loop, &self.context, &descriptor, |_| {});
        }
    }

    fn init_resources(&mut self) {
        let device = self.context.device().clone();

        // --- 1. 必须初始化顶点数据，否则 draw() 会崩溃 ---
        let vertices = vec![
            MyVertex {
                position: [0.0, -0.5, 0.0],
                color: [1.0, 0.0, 0.0],
            },
            MyVertex {
                position: [-0.5, 0.5, 0.0],
                color: [0.0, 1.0, 0.0],
            },
            MyVertex {
                position: [0.5, 0.5, 0.0],
                color: [0.0, 0.0, 1.0],
            },
        ];
        // 调用 renderer::buffer 模块
        self.vertex_buffer = Some(renderer::buffer::create_vertex_buffer(
            self.context.memory_allocator().clone(),
            vertices,
        ));

        // --- 2. 创建 Render Pass ---
        let render_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    format: self.windows.get_primary_renderer().unwrap().swapchain_format(),
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                }
            },
            pass: { color: [color], depth_stencil: {} }
        )
        .unwrap();

        // --- 3. 加载着色器并创建管线 ---
        let vs_entry = shaders::vs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let fs_entry = shaders::fs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();

        self.pipeline = Some(renderer::pipeline::create_graphics_pipeline(
            device,
            render_pass.clone(),
            vs_entry,
            fs_entry,
        ));

        self.render_pass = Some(render_pass);
        println!("GPU 资源模块化初始化成功！");
    }

    fn draw(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        static START_TIME: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        let start_time = START_TIME.get_or_init(std::time::Instant::now);
        let elapsed = start_time.elapsed().as_secs_f32();

        let (image_view, w, h) = {
            let renderer = self.windows.get_primary_renderer().ok_or("找不到渲染器")?;
            let extent = renderer.swapchain_image_view().image().extent();
            (
                renderer.swapchain_image_view().clone(),
                extent[0] as f32,
                extent[1] as f32,
            )
        };

        // 调用 math/camera.rs 计算 MVP
        let (model, view, proj) = math::camera::compute_mvp(w, h, elapsed);

        let framebuffer = Framebuffer::new(
            self.render_pass.as_ref().unwrap().clone(),
            FramebufferCreateInfo {
                attachments: vec![image_view],
                ..Default::default()
            },
        )?;

        // 这里的录制命令建议保留在 App 里或者移到 renderer/mod.rs
        let command_buffer = self.record_command_buffer(framebuffer, w, h, model, view, proj)?;

        let renderer_mut = self
            .windows
            .get_primary_renderer_mut()
            .ok_or("找不到渲染器")?;
        let acquire_future = renderer_mut.acquire(None, |_| {})?;

        use vulkano::sync::GpuFuture;
        let future = acquire_future
            .then_execute(self.context.graphics_queue().clone(), command_buffer)?
            .boxed();

        renderer_mut.present(future, true);
        Ok(())
    }

    fn record_command_buffer(
        &self,
        fb: Arc<Framebuffer>,
        w: f32,
        h: f32,
        model: cgmath::Matrix4<f32>,
        view: cgmath::Matrix4<f32>,
        proj: cgmath::Matrix4<f32>,
    ) -> Result<Arc<vulkano::command_buffer::PrimaryAutoCommandBuffer>, Box<dyn std::error::Error>>
    {
        let mut builder = vulkano::command_buffer::AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.context.graphics_queue().queue_family_index(),
            vulkano::command_buffer::CommandBufferUsage::OneTimeSubmit,
        )?;

        // 对应 shaders/vert.glsl 里的布局
        let push_constants = vs::PushConstants {
            model: model.into(),
            view: view.into(),
            proj: proj.into(),
        };

        unsafe {
            builder
                .begin_render_pass(
                    RenderPassBeginInfo {
                        clear_values: vec![Some([0.1, 0.2, 0.3, 1.0].into())],
                        ..RenderPassBeginInfo::framebuffer(fb)
                    },
                    SubpassBeginInfo::default(),
                )? // 1. 先开启 Render Pass
                .set_viewport(
                    0,
                    [vulkano::pipeline::graphics::viewport::Viewport {
                        offset: [0.0, 0.0],
                        extent: [w, h],
                        depth_range: 0.0..=1.0,
                    }]
                    .into_iter()
                    .collect(),
                )? // 2. 设置视口
                .set_scissor(
                    0,
                    [vulkano::pipeline::graphics::viewport::Scissor {
                        offset: [0, 0],
                        extent: [w as u32, h as u32],
                    }]
                    .into_iter()
                    .collect(),
                )? // 3. 设置剪裁矩形（修复上一个 VUID 错误）
                .bind_pipeline_graphics(self.pipeline.as_ref().unwrap().clone())?
                .push_constants(
                    self.pipeline.as_ref().unwrap().layout().clone(),
                    0,
                    push_constants,
                )?
                .bind_vertex_buffers(0, self.vertex_buffer.as_ref().unwrap().clone())?
                .draw(3, 1, 0, 0)? // 4. 此时 Draw 指令才有效
                .end_render_pass(Default::default())?; // 5. 最后关闭
        }
        Ok(builder.build()?)
    }
}

fn main() {
    let event_loop = EventLoop::new().expect("创建事件循环失败");
    let context = VulkanoContext::new(VulkanoConfig::default());
    let context_arc = Arc::new(context);

    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        context_arc.device().clone(),
        Default::default(),
    ));

    let mut app = App {
        context: context_arc,
        windows: VulkanoWindows::default(),
        vertex_buffer: None,
        pipeline: None,
        render_pass: None,
        command_buffer_allocator,
    };

    event_loop.run_app(&mut app).expect("运行失败");
}

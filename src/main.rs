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
    index_buffer: Option<Subbuffer<[u32]>>, // 新增此行
    pipeline: Option<Arc<GraphicsPipeline>>,
    render_pass: Option<Arc<RenderPass>>,
    depth_view: Option<Arc<vulkano::image::view::ImageView>>,
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

    // 专门用于创建与当前窗口大小匹配的深度图
    fn create_depth_view(&self, extent: [u32; 3]) -> Arc<vulkano::image::view::ImageView> {
        let depth_image = vulkano::image::Image::new(
            self.context.memory_allocator().clone(),
            vulkano::image::ImageCreateInfo {
                image_type: vulkano::image::ImageType::Dim2d,
                format: vulkano::format::Format::D16_UNORM, // 必须与 RenderPass 一致
                extent,
                usage: vulkano::image::ImageUsage::DEPTH_STENCIL_ATTACHMENT,
                ..Default::default()
            },
            vulkano::memory::allocator::AllocationCreateInfo {
                memory_type_filter: vulkano::memory::allocator::MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .unwrap();

        vulkano::image::view::ImageView::new_default(depth_image).unwrap()
    }

    fn init_resources(&mut self) {
        let device = self.context.device().clone();
        let allocator = self.context.memory_allocator().clone();

        // 1. 定义立方体的 8 个顶点（不再是 36 个）
        let vertices = vec![
            MyVertex {
                position: [-0.5, -0.5, 0.5],
                color: [1.0, 0.0, 0.0],
            }, // 0
            MyVertex {
                position: [0.5, -0.5, 0.5],
                color: [0.0, 1.0, 0.0],
            }, // 1
            MyVertex {
                position: [0.5, 0.5, 0.5],
                color: [0.0, 0.0, 1.0],
            }, // 2
            MyVertex {
                position: [-0.5, 0.5, 0.5],
                color: [1.0, 1.0, 0.0],
            }, // 3
            MyVertex {
                position: [-0.5, -0.5, -0.5],
                color: [1.0, 0.0, 1.0],
            }, // 4
            MyVertex {
                position: [0.5, -0.5, -0.5],
                color: [0.0, 1.0, 1.0],
            }, // 5
            MyVertex {
                position: [0.5, 0.5, -0.5],
                color: [1.0, 1.0, 1.0],
            }, // 6
            MyVertex {
                position: [-0.5, 0.5, -0.5],
                color: [0.5, 0.5, 0.5],
            }, // 7
        ];

        // 2. 定义索引（每 3 个索引构成一个三角形，共 12 个三角形）
        let indices = vec![
            0, 1, 2, 2, 3, 0, // 前
            1, 5, 6, 6, 2, 1, // 右
            7, 6, 5, 5, 4, 7, // 后
            4, 0, 3, 3, 7, 4, // 左
            4, 5, 1, 1, 0, 4, // 下
            3, 2, 6, 6, 7, 3, // 上
        ];

        self.vertex_buffer = Some(renderer::buffer::create_vertex_buffer(
            allocator.clone(),
            vertices,
        ));
        self.index_buffer = Some(renderer::buffer::create_index_buffer(
            allocator.clone(),
            indices,
        ));

        // 3. 创建 Render Pass
        let render_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    format: self.windows.get_primary_renderer().unwrap().swapchain_format(),
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
                depth: {
                    format: vulkano::format::Format::D16_UNORM,
                    samples: 1,
                    load_op: Clear,
                    store_op: DontCare,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {depth},
            }
        )
        .unwrap();

        // 4. 加载着色器
        let vs_entry = shaders::vs::load(device.clone())
            .expect("顶点着色器加载失败")
            .entry_point("main")
            .unwrap();
        let fs_entry = shaders::fs::load(device.clone())
            .expect("片元着色器加载失败")
            .entry_point("main")
            .unwrap();

        self.pipeline = Some(renderer::pipeline::create_graphics_pipeline(
            device,
            render_pass.clone(),
            vs_entry,
            fs_entry,
        ));

        self.render_pass = Some(render_pass);
        let extent = self
            .windows
            .get_primary_renderer()
            .unwrap()
            .swapchain_image_view()
            .image()
            .extent();
        self.depth_view = Some(self.create_depth_view(extent));
    }

    fn draw(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        static START_TIME: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        let start_time = START_TIME.get_or_init(std::time::Instant::now);
        let elapsed = start_time.elapsed().as_secs_f32();

        // 1. 获取当前交换链图像及其尺寸
        let (image_view, extent_u32) = {
            let renderer = self.windows.get_primary_renderer().ok_or("找不到渲染器")?;
            let ev = renderer.swapchain_image_view();
            (ev.clone(), ev.image().extent())
        };
        let w = extent_u32[0] as f32;
        let h = extent_u32[1] as f32;

        // 2. 核心优化：检查并更新深度图
        // 如果当前窗口尺寸与深度图尺寸不符（比如缩放了窗口），则重新创建
        let current_depth_view = if self.depth_view.as_ref().unwrap().image().extent() != extent_u32
        {
            let new_view = self.create_depth_view(extent_u32);
            self.depth_view = Some(new_view.clone());
            new_view
        } else {
            self.depth_view.as_ref().unwrap().clone()
        };

        // 3. 计算 MVP 矩阵
        let (model, view, proj) = math::camera::compute_mvp(w, h, elapsed);

        // 4. 创建 Framebuffer：传入颜色附件和深度附件
        // 这里的顺序 [image_view, current_depth_view] 必须与 RenderPass 的定义一致
        let framebuffer = Framebuffer::new(
            self.render_pass.as_ref().unwrap().clone(),
            FramebufferCreateInfo {
                attachments: vec![image_view, current_depth_view],
                ..Default::default()
            },
        )?;

        // 5. 录制并提交指令
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

        let push_constants = vs::PushConstants {
            model: model.into(),
            view: view.into(),
            proj: proj.into(),
        };

        unsafe {
            builder
                .begin_render_pass(
                    RenderPassBeginInfo {
                        clear_values: vec![Some([0.1, 0.2, 0.3, 1.0].into()), Some(1f32.into())],
                        ..RenderPassBeginInfo::framebuffer(fb)
                    },
                    SubpassBeginInfo::default(),
                )?
                .set_viewport(
                    0,
                    [vulkano::pipeline::graphics::viewport::Viewport {
                        offset: [0.0, 0.0],
                        extent: [w, h],
                        depth_range: 0.0..=1.0,
                    }]
                    .into_iter()
                    .collect(),
                )?
                .set_scissor(
                    0,
                    [vulkano::pipeline::graphics::viewport::Scissor {
                        offset: [0, 0],
                        extent: [w as u32, h as u32],
                    }]
                    .into_iter()
                    .collect(),
                )?
                .bind_pipeline_graphics(self.pipeline.as_ref().unwrap().clone())?
                .push_constants(
                    self.pipeline.as_ref().unwrap().layout().clone(),
                    0,
                    push_constants,
                )?
                // 核心修改：绑定顶点缓冲
                .bind_vertex_buffers(0, self.vertex_buffer.as_ref().unwrap().clone())?
                // 核心修改：绑定索引缓冲
                .bind_index_buffer(self.index_buffer.as_ref().unwrap().clone())?
                // 核心修改：使用 draw_indexed。参数 36 是索引数量（12个三角形 * 3）
                .draw_indexed(36, 1, 0, 0, 0)?
                .end_render_pass(Default::default())?;
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
        index_buffer: None, // 新增此行
        pipeline: None,
        render_pass: None,
        depth_view: None,
        command_buffer_allocator,
    };

    event_loop.run_app(&mut app).expect("运行失败");
}

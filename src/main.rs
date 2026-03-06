use std::sync::Arc;
use vulkano::buffer::Buffer;
use vulkano::buffer::BufferContents;
use vulkano::buffer::BufferCreateInfo;
use vulkano::buffer::BufferUsage;
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::CommandBufferUsage;
use vulkano::command_buffer::PrimaryAutoCommandBuffer;
use vulkano::command_buffer::RenderPassBeginInfo;
use vulkano::command_buffer::SubpassBeginInfo;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::memory::allocator::AllocationCreateInfo;
use vulkano::memory::allocator::MemoryTypeFilter;
use vulkano::pipeline::DynamicState;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::Pipeline;
use vulkano::pipeline::PipelineLayout;
use vulkano::pipeline::PipelineShaderStageCreateInfo;
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::color_blend::ColorBlendAttachmentState;
use vulkano::pipeline::graphics::color_blend::ColorBlendState;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::vertex_input::Vertex;
use vulkano::pipeline::graphics::viewport::ViewportState;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::render_pass::Framebuffer;
use vulkano::render_pass::FramebufferCreateInfo;
use vulkano::render_pass::RenderPass;
use vulkano::render_pass::Subpass;
use vulkano_util::context::{VulkanoConfig, VulkanoContext};
use vulkano_util::window::{VulkanoWindows, WindowDescriptor};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::event_loop::EventLoop;
use winit::window::WindowId;
// 确保有这一行，它定义了顶点如何与管线对接
use vulkano::pipeline::graphics::vertex_input::VertexDefinition;

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 450
            layout(location = 0) in vec3 position;
            layout(location = 1) in vec3 color;

            layout(location = 0) out vec3 v_color;

            layout(push_constant) uniform PushConstants {
                mat4 model;
                mat4 view;
                mat4 proj;
            } push;

            void main() {
                // 矩阵相乘：投影 * 视图 * 模型 * 顶点坐标
                gl_Position = push.proj * push.view * push.model * vec4(position, 1.0);
                v_color = color;
            }
        ",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 450
            layout(location = 0) in vec3 v_color;
            layout(location = 0) out vec4 f_color;
            void main() {
                f_color = vec4(v_color, 1.0);
            }
        ",
    }
}

// 使用 3D 坐标和 RGB 颜色
#[derive(BufferContents, Vertex)]
#[repr(C)]
struct MyVertex {
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    color: [f32; 3],
}

// 定义 Push Constants，包含我们的 MVP 矩阵
#[repr(C)]
#[derive(BufferContents)] // 必须派生这个才能传给 Vulkan
struct PushConstants {
    model: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
}

#[repr(C)]
struct SceneConstants {
    view_proj: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
}

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
        // 2. 准备数据（顶点和管线）
        self.init_resources();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                // 将画图逻辑独立出去，如果失败了在这里打印即可
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
    /// 抽屉 1：创建窗口逻辑
    fn init_window(&mut self, event_loop: &ActiveEventLoop) {
        if self.windows.get_primary_window().is_none() {
            let descriptor = WindowDescriptor {
                width: 1024.0,
                height: 768.0,
                title: "Rust Vulkano Window".to_string(),
                ..WindowDescriptor::default()
            };
            self.windows
                .create_window(event_loop, &self.context, &descriptor, |_| {});
        }
    }

    /// 抽屉 2：准备 GPU 资源（顶点缓存和渲染管线）
    fn init_resources(&mut self) {
        let device = self.context.device().clone();
        let allocator = self.context.memory_allocator();

        // --- 创建顶点缓存 ---
        let vertices = [
            MyVertex {
                position: [0.0, -0.5, 0.0],
                color: [1.0, 0.0, 0.0],
            }, // 红色顶点
            MyVertex {
                position: [-0.5, 0.5, 0.0],
                color: [0.0, 1.0, 0.0],
            }, // 绿色顶点
            MyVertex {
                position: [0.5, 0.5, 0.0],
                color: [0.0, 0.0, 1.0],
            }, // 蓝色顶点
        ];
        self.vertex_buffer = Some(
            Buffer::from_iter(
                allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::VERTEX_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                vertices,
            )
            .unwrap(),
        );

        // --- 创建渲染通路 (Render Pass) ---
        let render_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: { color: { format: self.windows.get_primary_renderer().unwrap().swapchain_format(), samples: 1, load_op: Clear, store_op: Store } },
            pass: { color: [color], depth_stencil: {} }
        ).unwrap();

        // --- 创建管线 (Pipeline) ---
        let vs = vs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let fs = fs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let stages = [
            PipelineShaderStageCreateInfo::new(vs.clone()),
            PipelineShaderStageCreateInfo::new(fs.clone()),
        ];
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();

        let pipeline = GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(MyVertex::per_vertex().definition(&vs).unwrap()),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState::default()),
                dynamic_state: [DynamicState::Viewport, DynamicState::Scissor]
                    .into_iter()
                    .collect(),
                rasterization_state: Some(RasterizationState {
                    cull_mode: vulkano::pipeline::graphics::rasterization::CullMode::None,
                    ..RasterizationState::default()
                }),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    Subpass::from(render_pass.clone(), 0)
                        .unwrap()
                        .num_color_attachments(),
                    ColorBlendAttachmentState::default(),
                )),
                subpass: Some(Subpass::from(render_pass.clone(), 0).unwrap().into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .unwrap();

        self.render_pass = Some(render_pass);
        self.pipeline = Some(pipeline);
    }

    /// 抽屉 3：每一帧的渲染逻辑
    fn draw(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // --- 1. 计算旋转角度 ---
        // 使用程序运行以来的时间来驱动旋转
        static START_TIME: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        let start_time = START_TIME.get_or_init(std::time::Instant::now);
        let elapsed = start_time.elapsed().as_secs_f32();

        // --- 2. 准备数据（保持原来的逻辑，但增加对 w, h 的获取） ---
        let (image_view, w, h) = {
            let renderer = self
                .windows
                .get_primary_renderer()
                .ok_or("Renderer not found")?;
            let extent = renderer.swapchain_image_view().image().extent();
            (
                renderer.swapchain_image_view().clone(),
                extent[0] as f32,
                extent[1] as f32,
            )
        };

        // --- 3. 计算 3D 矩阵 ---
        use cgmath::{Matrix4, Point3, Rad, Vector3};

        // 模型矩阵：绕 Y 轴旋转
        let model = Matrix4::from_angle_y(Rad(elapsed));

        // 视图矩阵：相机放在 (0, 0, -2)，看向原点
        let view = Matrix4::look_at_rh(
            Point3::new(0.0, 0.0, -2.0),
            Point3::new(0.0, 0.0, 0.0),
            Vector3::unit_y(),
        );

        // 投影矩阵：处理透视和宽高比
        let proj = cgmath::perspective(Rad(std::f32::consts::FRAC_PI_2), w / h, 0.01, 100.0);

        // 注意：Vulkan 的 Y 轴是向下的，而 cgmath 默认向上，我们需要翻转一下投影矩阵的 Y
        let mut proj = proj;
        proj[1][1] *= -1.0;

        // --- 4. 录制与呈现 (将矩阵传给 record_command_buffer) ---
        let framebuffer = Framebuffer::new(
            self.render_pass.as_ref().unwrap().clone(),
            FramebufferCreateInfo {
                attachments: vec![image_view],
                ..Default::default()
            },
        )?;

        // 修改你的 record_command_buffer 签名以接收矩阵
        let command_buffer = self.record_command_buffer(framebuffer, w, h, model, view, proj)?;

        // --- 第三阶段：重新获取可变借用来呈现 ---
        let renderer_mut = self
            .windows
            .get_primary_renderer_mut()
            .ok_or("Renderer not found")?;
        let acquire_future = renderer_mut.acquire(None, |_| {})?;

        use vulkano::sync::GpuFuture;
        let future = acquire_future
            .then_execute(self.context.graphics_queue().clone(), command_buffer)?
            .boxed();

        renderer_mut.present(future, true);
        Ok(())
    }

    /// 抽屉 4：专门负责录制“画画”的动作 (解决超级缩进的关键)
    fn record_command_buffer(
        &self,
        fb: Arc<Framebuffer>,
        w: f32,
        h: f32,
        model: cgmath::Matrix4<f32>,
        view: cgmath::Matrix4<f32>,
        proj: cgmath::Matrix4<f32>,
    ) -> Result<Arc<PrimaryAutoCommandBuffer>, Box<dyn std::error::Error>> {
        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.context.graphics_queue().queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
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
                        clear_values: vec![Some([0.1, 0.2, 0.3, 1.0].into())],
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
                    push_constants, // 现在这个变量被找到了！
                )?
                .bind_vertex_buffers(0, self.vertex_buffer.as_ref().unwrap().clone())?
                .draw(3, 1, 0, 0)?
                .end_render_pass(Default::default())?;
        }

        Ok(builder.build()?)
    }
}

fn main() {
    let event_loop = EventLoop::new().expect("创建事件循环失败");
    let context = VulkanoContext::new(VulkanoConfig::default());
    let context_arc = Arc::new(context); // 提前 Arc 化方便后续共享

    // 初始化指令分配器
    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        context_arc.device().clone(),
        Default::default(),
    ));

    println!(
        "正在使用的显卡: {}",
        context_arc
            .device()
            .physical_device()
            .properties()
            .device_name
    );

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

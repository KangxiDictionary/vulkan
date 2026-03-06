use std::sync::Arc;
use vulkano::buffer::Buffer;
use vulkano::buffer::BufferContents;
use vulkano::buffer::BufferCreateInfo;
use vulkano::buffer::BufferUsage;
use vulkano::buffer::Subbuffer;
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
            layout(location = 0) in vec2 position;
            // 接收一个动态传入的宽高比
            layout(push_constant) uniform PushConstants {
                float aspect_ratio;
            } push_constants;

            void main() {
                // 如果窗口很宽，我们就缩小 X 轴的显示比例
                gl_Position = vec4(position.x / push_constants.aspect_ratio, position.y, 0.0, 1.0);
            }
        ",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 450
            layout(location = 0) out vec4 f_color;
            void main() {
                f_color = vec4(1.0, 0.5, 0.2, 1.0); // 橘色
            }
        ",
    }
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
struct MyVertex {
    #[format(R32G32_SFLOAT)] // 代表两个 32 位浮点数 (x, y)
    position: [f32; 2],
}

impl App {
    // 将复杂的渲染逻辑抽离到这里
    fn draw(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let renderer = self
            .windows
            .get_primary_renderer_mut()
            .ok_or("无法获取渲染器")?;

        // 1. 准备数据（减少缩进的第一步：提前计算并处理 Result）
        let image_view = renderer.swapchain_image_view();
        let image_extent = image_view.image().extent();
        let width = image_extent[0] as f32;
        let height = image_extent[1] as f32;
        let aspect_ratio = width / height;

        let acquire_future = match renderer.acquire(None, |_| {}) {
            Ok(res) => res,
            Err(_) => return Ok(()), // 交换链过期时直接跳过本帧
        };

        let framebuffer = vulkano::render_pass::Framebuffer::new(
            self.render_pass.as_ref().unwrap().clone(),
            vulkano::render_pass::FramebufferCreateInfo {
                attachments: vec![image_view.clone()],
                ..Default::default()
            },
        )?;

        // 2. 录制指令（使用这种风格可以避免深度缩进）
        let mut builder = vulkano::command_buffer::AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.context.graphics_queue().queue_family_index(),
            vulkano::command_buffer::CommandBufferUsage::OneTimeSubmit,
        )?;

        // 将 unsafe 块限制在最小范围，或者直接包裹链式调用
        unsafe {
            builder
                .begin_render_pass(
                    vulkano::command_buffer::RenderPassBeginInfo {
                        clear_values: vec![Some([0.1, 0.2, 0.3, 1.0].into())],
                        ..vulkano::command_buffer::RenderPassBeginInfo::framebuffer(framebuffer)
                    },
                    vulkano::command_buffer::SubpassBeginInfo::default(),
                )?
                .set_viewport(
                    0,
                    [vulkano::pipeline::graphics::viewport::Viewport {
                        offset: [0.0, 0.0],
                        extent: [width, height],
                        depth_range: 0.0..=1.0,
                    }]
                    .into_iter()
                    .collect(),
                )?
                .set_scissor(
                    0,
                    [vulkano::pipeline::graphics::viewport::Scissor {
                        offset: [0, 0],
                        extent: [width as u32, height as u32],
                    }]
                    .into_iter()
                    .collect(),
                )?
                .bind_pipeline_graphics(self.pipeline.as_ref().unwrap().clone())?
                .push_constants(
                    self.pipeline.as_ref().unwrap().layout().clone(),
                    0,
                    vs::PushConstants { aspect_ratio },
                )?
                .bind_vertex_buffers(0, self.vertex_buffer.as_ref().unwrap().clone())?
                .draw(3, 1, 0, 0)?
                .end_render_pass(Default::default())?;
        }

        let command_buffer = builder.build()?;

        // 3. 提交与呈现
        use vulkano::sync::GpuFuture;
        let execution_future = acquire_future
            .then_execute(self.context.graphics_queue().clone(), command_buffer)?
            .boxed();

        renderer.present(execution_future, true);
        Ok(())
    }
}

struct App {
    context: Arc<VulkanoContext>,
    windows: VulkanoWindows,
    vertex_buffer: Option<Subbuffer<[MyVertex]>>,
    // 新增这两个字段
    pipeline: Option<Arc<GraphicsPipeline>>,
    render_pass: Option<Arc<RenderPass>>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
}

impl ApplicationHandler for App {
    // 只要程序一开始运行，就会触发这个函数
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // 在新版 winit 中，窗口必须在 resumed 阶段创建
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

        let memory_allocator = self.context.memory_allocator();

        // 定义三个点：左下、右下、顶部
        // 修改顶点坐标顺序，确保它是逆时针（Counter-Clockwise）
        let vertices = [
            MyVertex {
                position: [0.0, -0.5],
            }, // 顶部
            MyVertex {
                position: [-0.5, 0.5],
            }, // 左下
            MyVertex {
                position: [0.5, 0.5],
            }, // 右下
        ];

        // 申请显存并存入数据
        let buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER, // 告诉 GPU：这是顶点数据
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices,
        )
        .expect("创建顶点缓存失败");

        self.vertex_buffer = Some(buffer);
        println!("顶点数据已成功发送至 GPU");
        // 获取设备句柄
        let device = self.context.device().clone();

        // 1. 创建渲染通路 (Render Pass)：定义“我们要往屏幕上画画”
        let render_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    // 从窗口渲染器获取当前屏幕的颜色格式
                    format: self.windows.get_primary_renderer().unwrap().swapchain_format(),
                    samples: 1,
                    load_op: Clear, // 每次画之前先清空屏幕
                    store_op: Store,
                },
            },
            pass: { color: [color], depth_stencil: {} }
        )
        .expect("创建 RenderPass 失败");

        // 2. 加载着色器入口点
        let vs = vs::load(device.clone())
            .expect("加载 VS 失败")
            .entry_point("main")
            .unwrap();
        let fs = fs::load(device.clone())
            .expect("加载 FS 失败")
            .entry_point("main")
            .unwrap();

        // 3. 定义管线布局
        let stages = [
            PipelineShaderStageCreateInfo::new(vs.clone()),
            PipelineShaderStageCreateInfo::new(fs.clone()),
        ];
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())
                .expect("创建 PipelineLayout 失败"),
        )
        .unwrap();

        // 4. 正式创建图形管线
        let pipeline = GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(MyVertex::per_vertex().definition(&vs).unwrap()),
                input_assembly_state: Some(InputAssemblyState::default()),

                // --- 重点修改 1: 保持 ViewportState 为空，因为我们要动态设置 ---
                viewport_state: Some(ViewportState::default()),

                // --- 重点修改 2: 显式声明 Viewport 和 Scissor 是动态的 ---
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
        .expect("创建 GraphicsPipeline 失败");

        self.render_pass = Some(render_pass);
        self.pipeline = Some(pipeline);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                // 这里的逻辑变得非常清爽
                if let Err(e) = self.draw() {
                    eprintln!("渲染失败: {:?}", e);
                }
            }
            _ => (),
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // 相当于之前的 MainEventsCleared，用于触发重绘
        if let Some(window) = self.windows.get_primary_window() {
            window.request_redraw();
        }
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

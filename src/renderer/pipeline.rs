use std::sync::Arc;
use vulkano::pipeline::graphics::vertex_input::Vertex; // 必须引入这个 Trait

// src/renderer/pipeline.rs
use crate::renderer::buffer::MyVertex;
use vulkano::{
    pipeline::{
        DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::VertexDefinition,
            viewport::ViewportState,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    render_pass::{RenderPass, Subpass},
};

// 1. 增加 pub 让 main.rs 可以访问
// 2. 移除 &self，改传显式参数 device
pub fn create_graphics_pipeline(
    device: Arc<vulkano::device::Device>,
    render_pass: Arc<RenderPass>,
    vs: vulkano::shader::EntryPoint,
    fs: vulkano::shader::EntryPoint,
) -> Arc<GraphicsPipeline> {
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

    GraphicsPipeline::new(
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
            subpass: Some(Subpass::from(render_pass, 0).unwrap().into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .expect("创建 GraphicsPipeline 失败")
}

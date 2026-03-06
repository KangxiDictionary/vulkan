use crate::renderer::buffer::MyVertex;
use std::sync::Arc;
use vulkano::{
    pipeline::{
        DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{ColorBlendAttachmentState, ColorBlendState}, // 必须包含
            depth_stencil::{DepthState, DepthStencilState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState, // 必须包含
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::ViewportState,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    render_pass::{RenderPass, Subpass},
};

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

            // 1. 光栅化状态
            rasterization_state: Some(RasterizationState::default()),

            // 2. 深度测试状态 (使用非弃用接口)
            depth_stencil_state: Some(DepthStencilState {
                depth: Some(DepthState::simple()),
                ..Default::default()
            }),

            // 3. 多重采样状态 (解决你现在的报错)
            multisample_state: Some(MultisampleState::default()),

            // 4. 颜色混合状态 (渲染到颜色附件所必须)
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                Subpass::from(render_pass.clone(), 0)
                    .unwrap()
                    .num_color_attachments(),
                ColorBlendAttachmentState::default(),
            )),

            dynamic_state: [DynamicState::Viewport, DynamicState::Scissor]
                .into_iter()
                .collect(),
            subpass: Some(Subpass::from(render_pass, 0).unwrap().into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .expect("创建 GraphicsPipeline 失败")
}

use crate::renderer::buffer::MyVertex;
use cgmath::Matrix4;
use std::sync::Arc;
use vulkano::buffer::Subbuffer;
use vulkano::pipeline::GraphicsPipeline;

pub trait Renderable {
    fn get_pipeline(&self) -> Arc<GraphicsPipeline>;
    fn get_vertex_buffer(&self) -> Subbuffer<[MyVertex]>;
    fn get_model_matrix(&self, elapsed: f64) -> Matrix4<f64>;
    fn vertex_count(&self) -> u32;
}

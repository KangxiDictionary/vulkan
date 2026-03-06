// src/renderer/buffer.rs
use std::sync::Arc;
use vulkano::buffer::BufferContents;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::vertex_input::Vertex;

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct MyVertex {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub color: [f32; 3],
}

pub fn create_vertex_buffer(
    allocator: Arc<StandardMemoryAllocator>,
    vertices: Vec<MyVertex>,
) -> Subbuffer<[MyVertex]> {
    Buffer::from_iter(
        allocator,
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
    .expect("创建顶点缓存失败")
}

// 新增：创建索引缓存函数
pub fn create_index_buffer(
    allocator: Arc<StandardMemoryAllocator>,
    indices: Vec<u32>,
) -> Subbuffer<[u32]> {
    Buffer::from_iter(
        allocator,
        BufferCreateInfo {
            usage: BufferUsage::INDEX_BUFFER, // 核心区别：声明为 INDEX_BUFFER
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        indices,
    )
    .expect("创建索引缓存失败")
}

#version 450

// 来自顶点缓冲区的输入
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;

// 输出到片元着色器
layout(location = 0) out vec3 v_color;

// 推送常量：用于接收来自 Rust 的 MVP 矩阵
layout(push_constant) uniform PushConstants {
    mat4 model;
    mat4 view;
    mat4 proj;
} push;

void main() {
    // 矩阵运算顺序：投影 * 视图 * 模型 * 局部坐标
    gl_Position = push.proj * push.view * push.model * vec4(position, 1.0);
    v_color = color;
}
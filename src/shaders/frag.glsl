#version 450

// 接收来自顶点着色器的插值颜色 (location 必须对应)
layout(location = 0) in vec3 v_color;

// 输出到屏幕的颜色
layout(location = 0) out vec4 f_color;

void main() {
    f_color = vec4(v_color, 1.0);
}
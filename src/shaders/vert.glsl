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
    gl_Position = push.proj * push.view * push.model * vec4(position, 1.0);
    v_color = color;
}
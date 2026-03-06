// src/math/camera.rs
use cgmath::{Deg, Matrix4, Point3, Rad, Vector3};

pub fn compute_mvp(
    width: f32,
    height: f32,
    elapsed: f32,
) -> (Matrix4<f32>, Matrix4<f32>, Matrix4<f32>) {
    // 1. 逻辑层：全部使用 f64 运算
    let elapsed_f64 = elapsed as f64;
    let aspect_ratio = (width / height) as f64;

    // 模型矩阵：让三角形随时间旋转
    let model_f64 =
        Matrix4::from_angle_y(Rad(elapsed_f64)) * Matrix4::from_angle_x(Rad(elapsed_f64 * 0.5));

    // 视图矩阵（相机）：
    // 即使未来你的玩家坐标到了 1,000,000.0 米外，这里的 f64 也能保证相机平滑
    let view_f64 = Matrix4::look_at_rh(
        Point3::new(0.0, 0.0, 2.0),  // 相机位置
        Point3::new(0.0, 0.0, 0.0),  // 目标点
        Vector3::new(0.0, 1.0, 0.0), // 上方向
    );

    // 投影矩阵：
    let proj_f64 = cgmath::perspective(
        Deg(45.0),
        aspect_ratio,
        0.1,    // 近平面
        1000.0, // 远平面
    );

    // 2. 转换层：将 f64 矩阵转换为 f32 传给 GPU
    // cgmath 的 cast() 会尝试将内部元素类型转换
    let model: Matrix4<f32> = model_f64.cast().expect("Model matrix cast failed");
    let view: Matrix4<f32> = view_f64.cast().expect("View matrix cast failed");

    // 注意：Vulkan 的 NDC 坐标系中 Y 轴是朝下的，而 cgmath 默认朝上
    // 我们在这里直接对投影矩阵做一次修正
    let mut proj: Matrix4<f32> = proj_f64.cast().expect("Projection matrix cast failed");
    proj.y.y *= -1.0;

    (model, view, proj)
}

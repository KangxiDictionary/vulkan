// src/math/camera.rs
use cgmath::{Deg, Matrix4, Point3, Rad, Vector3};

pub fn compute_mvp(
    width: f32,
    height: f32,
    elapsed: f32,
) -> (Matrix4<f32>, Matrix4<f32>, Matrix4<f32>) {
    // 这里保留你之前的 f64 逻辑构思，返回 f32 给 GPU
    let model = Matrix4::from_angle_y(Rad(elapsed));
    let view = Matrix4::look_at_rh(
        Point3::new(0.0, 0.0, -2.0),
        Point3::new(0.0, 0.0, 0.0),
        Vector3::unit_y(),
    );
    let mut proj = cgmath::perspective(Deg(90.0), width / height, 0.01, 100.0);
    proj[1][1] *= -1.0;

    (model, view, proj)
}

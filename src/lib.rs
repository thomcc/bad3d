#![allow(dead_code)]

#[macro_use]
pub mod util;

#[macro_use]
mod asserts;

pub mod math;
pub mod hull;

pub mod support;
pub mod gjk;
pub mod phys;

pub mod wingmesh;

pub mod bsp;

pub use wingmesh::WingMesh;
pub use math::{
    V2, V3, V4,
    M2x2, M3x3, M4x4,
    Quat,
    Pose,
    Plane,
    VecType,
    vec2, vec3, vec4,
    mat2, mat3, mat4,
};

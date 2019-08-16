#![warn(rust_2018_idioms)]
#![allow(clippy::float_cmp, clippy::many_single_char_names, clippy::cast_lossless)]

#[macro_use]
pub mod util;
pub use t3m as math;
pub mod prelude;

pub mod hull;

pub mod gjk;
pub mod phys;
pub mod shape;
pub mod support;
pub mod wingmesh;

pub mod bsp;
pub use prelude::*;

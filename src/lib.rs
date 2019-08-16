#![warn(rust_2018_idioms)]
#![allow(clippy::float_cmp, clippy::many_single_char_names, clippy::cast_lossless)]

#[macro_use]
pub mod util;
mod core;
pub use t3m as math;
pub mod prelude;

pub use crate::core::*;

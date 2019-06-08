#![allow(dead_code, unstable_name_collision)]
// XXX unstable_name_collision is for `clamp` :(

#[macro_use]
extern crate log;

#[macro_use]
extern crate more_asserts;

#[macro_use]
pub mod util;
mod core;
pub mod math;
pub mod prelude;

pub use core::*;

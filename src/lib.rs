#![allow(dead_code)]

#[macro_use]
extern crate more_asserts;

#[macro_use]
pub mod util;
mod core;
pub mod math;
pub mod prelude;

pub use crate::core::*;

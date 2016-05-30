#![allow(dead_code)]
mod math;
mod geom;
mod pose;
mod hull;
use math::{V3, M3x3};
use std::mem;



fn main() {
    let v = V3{x:1f32, y:2f32, z:3f32};
    println!("Hello, world: {:?}, {}", v, mem::size_of::<M3x3>());
}

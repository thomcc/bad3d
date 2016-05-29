#![allow(dead_code)]
mod linalg;
use std::mem;

fn main() {
    let v = linalg::F3{x:1f32, y:2f32, z:3f32};
    println!("Hello, world: {:?}, {}", v, mem::size_of::<linalg::F3x3>());
}

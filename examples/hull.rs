#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(clippy::float_cmp, clippy::many_single_char_names, clippy::cast_lossless)]
#[macro_use]
extern crate glium;
use rand;

#[macro_use]
extern crate bad3d;

#[macro_use]
extern crate failure;
#[global_allocator]
static GLOBAL: mimallocator::Mimalloc = mimallocator::Mimalloc;

use imgui;

mod shared;
use crate::shared::{object, DemoMesh, DemoOptions, DemoWindow, Result};
use failure::Error;

use bad3d::prelude::*;
use std::{cell::RefCell, rc::Rc};

fn main() -> Result<()> {
    let mut win = DemoWindow::new(
        DemoOptions {
            title: "Hull test",
            camera: Pose::new_look_at(vec3(0.0, 0.0, 2.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0)).inverse(),
            clear_color: vec4(0.5, 0.6, 1.0, 1.0),
            near_far: (0.01, 50.0),
            ..Default::default()
        },
        Rc::new(RefCell::new(imgui::Context::create())),
    )?;

    let (vertices, triangles) = object::random_point_cloud(64);
    let mesh = DemoMesh::new(&win.display, vertices, triangles, object::random_color())?;
    let mut pose = Pose::identity();

    while win.is_up() {
        if win.input.mouse.down.0 {
            let q = pose.orientation;
            pose.orientation = Quat::virtual_track_ball(
                vec3(0.0, 0.0, 2.0),
                vec3(0.0, 0.0, 0.0),
                win.input.mouse_prev.vec,
                win.input.mouse.vec,
            ) * q;
        }

        win.draw_lit_mesh(pose.to_mat4(), &mesh)?;

        win.end_frame()?;
    }
    Ok(())
}

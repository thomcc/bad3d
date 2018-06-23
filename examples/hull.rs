#![allow(dead_code)]
#![allow(unused_imports)]

#[macro_use]
extern crate glium;
extern crate rand;

#[macro_use]
extern crate bad3d;

#[macro_use]
extern crate failure;

extern crate imgui;
extern crate imgui_glium_renderer;

mod shared;
use failure::Error;
use shared::{DemoWindow, DemoOptions, Result, object, DemoMesh};

use bad3d::prelude::*;
use std::{rc::Rc, cell::RefCell};

fn main() -> Result<()> {
    let mut win = DemoWindow::new(DemoOptions {
        title: "Hull test",
        view: M4x4::look_at(vec3(0.0, 0.0, 2.0),
                            vec3(0.0, 0.0, 0.0),
                            vec3(0.0, 1.0, 0.0)),
        clear_color: vec4(0.5, 0.6, 1.0, 1.0),
        near_far: (0.01, 50.0),
        .. Default::default()
    }, Rc::new(RefCell::new(imgui::ImGui::init())))?;

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
                win.input.mouse.vec
            ) * q;
        }

        win.draw_lit_mesh(pose.to_mat4(), &mesh)?;

        win.end_frame()?;
    }
    Ok(())
}

#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::cast_lossless
)]
#[macro_use]
extern crate glium;
use rand;

#[macro_use]
extern crate bad3d;

#[macro_use]
extern crate failure;
#[global_allocator]
static GLOBAL: mimallocator::Mimalloc = mimallocator::Mimalloc;
use imgui_glium_renderer;

use imgui::{self, im_str};

mod shared;
use crate::shared::{object, DemoMesh, DemoObject, DemoOptions, DemoWindow, Result};

use bad3d::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;

struct GjkTestState {
    vert_count: usize,
    all_verts: Vec<V3>,

    a_verts: Vec<V3>,
    b_verts: Vec<V3>,

    a_tris: Vec<[u16; 3]>,
    b_tris: Vec<[u16; 3]>,
}

impl GjkTestState {
    fn new() -> GjkTestState {
        let mut res = GjkTestState {
            vert_count: 6,
            all_verts: Vec::new(),
            a_verts: Vec::new(),
            a_tris: Vec::new(),
            b_verts: Vec::new(),
            b_tris: Vec::new(),
        };
        res.reinit();
        res
    }

    fn reinit(&mut self) {
        self.all_verts.clear();
        for _ in 0..2 {
            let pos = (object::rand_v3() - V3::splat(0.5)) * 1.0;
            for _ in 0..self.vert_count {
                self.all_verts
                    .push(pos + (object::rand_v3() - V3::splat(0.5)).normalize().unwrap() / 2.0);
            }
        }
        let com =
            self.all_verts.iter().fold(V3::zero(), |a, b| a + *b) / (self.all_verts.len() as f32);
        for v in self.all_verts.iter_mut() {
            *v -= com;
        }
        self.regen();
    }

    fn set_count(&mut self, vert_count: usize) {
        if vert_count >= 4 {
            self.vert_count = vert_count;
        }
        self.reinit();
        self.regen();
    }

    fn regen(&mut self) {
        loop {
            self.a_verts.clear();
            self.a_tris.clear();
            self.b_verts.clear();
            self.b_tris.clear();

            for v in &self.all_verts[0..self.all_verts.len() / 2] {
                self.a_verts.push(*v);
            }
            for v in &self.all_verts[self.all_verts.len() / 2..self.all_verts.len()] {
                self.b_verts.push(*v);
            }
            let a_hull = hull::compute_hull(&mut self.a_verts[..]);
            let b_hull = hull::compute_hull(&mut self.b_verts[..]);
            if let (Some((a, a_max_idx)), Some((b, b_max_idx))) = (a_hull, b_hull) {
                self.a_tris = a;
                self.a_verts.truncate(a_max_idx);
                self.b_tris = b;
                self.b_verts.truncate(b_max_idx);
                return;
            } else {
                self.reinit();
            }
        }
    }
}

fn main() -> Result<()> {
    env_logger::init();
    let gui = Rc::new(RefCell::new(imgui::Context::create()));
    let mut win = DemoWindow::new(
        DemoOptions {
            title: "Hull test",
            view: M4x4::look_at(
                vec3(0.0, 0.0, 2.0),
                vec3(0.0, 0.0, 0.0),
                vec3(0.0, 1.0, 0.0),
            ),
            clear_color: vec4(0.1, 0.1, 0.2, 1.0),
            near_far: (0.01, 50.0),
            light_pos: vec3(1.4, 0.4, 0.7),
            ..Default::default()
        },
        gui.clone(),
    )?;

    let mut a_pos = V3::zero();
    let mut b_pos = V3::zero();

    let mut test_state = GjkTestState::new();
    let mut show_mink = false;

    use glium::index::PrimitiveType::*;

    let mut model_orientation = Quat::identity();
    let mut gui_renderer =
        imgui_glium_renderer::Renderer::init(&mut *gui.borrow_mut(), &win.display).unwrap();
    while win.is_up() {
        for &(key, down) in win.input.key_changes.iter() {
            if !down {
                continue;
            }
            match key {
                glium::glutin::VirtualKeyCode::Key1 => {
                    test_state.reinit();
                    a_pos = V3::zero();
                    b_pos = V3::zero();
                }
                glium::glutin::VirtualKeyCode::Key2 => {
                    let t = !show_mink;
                    show_mink = t;
                }
                _ => {}
            }
        }

        if win.input.mouse.down.0 {
            let q = model_orientation;
            model_orientation = Quat::virtual_track_ball(
                vec3(0.0, 0.0, 2.0),
                vec3(0.0, 0.0, 0.0),
                win.input.mouse_prev.vec,
                win.input.mouse.vec,
            ) * q;
        }

        // let mouse_ray = model_orientation * win.input.mouse.vec.must_norm();

        // let on_a = geom::convex_hit_check_posed(
        //     test_state.a_tris.iter().map(|&tri| {
        //         let (v0, v1, v2) = tri.tri_verts(&test_state.a_verts);
        //         Plane::from_tri(v0, v1, v2)
        //     }),
        //     a_pos.into(),
        //     V3::zero(),
        //     mouse_ray * 100.0,
        // );

        // let on_b = geom::convex_hit_check_posed(
        //     test_state.b_tris.iter().map(|&tri| {
        //         let (v0, v1, v2) = tri.tri_verts(&test_state.b_verts);
        //         Plane::from_tri(v0, v1, v2)
        //     }),
        //     b_pos.into(),
        //     V3::zero(),
        //     mouse_ray * 100.0,
        // );
        // let a_dist = on_a.map(|hi| hi.impact.length_sq());
        // let b_dist = on_b.map(|hi| hi.impact.length_sq());

        let scene_matrix = Pose::from_rotation(model_orientation).to_mat4();

        let va = test_state
            .a_verts
            .iter()
            .map(|&v| v + a_pos)
            .collect::<Vec<_>>();
        let vb = test_state
            .b_verts
            .iter()
            .map(|&v| v + b_pos)
            .collect::<Vec<_>>();

        let hit = gjk::separated(&va[..], &vb[..], true);

        let did_hit = hit.separation <= 0.0;

        let hit_p = if show_mink {
            let mut mink_vertices = Vec::with_capacity(va.len() * vb.len());

            for a in &va {
                for b in &vb {
                    mink_vertices.push(*a - *b);
                }
            }
            let (mink_tris, len) = hull::compute_hull(&mut mink_vertices[..]).unwrap();
            win.draw_tris(
                scene_matrix,
                vec4(1.0, 0.5, 0.5, 0.8),
                &mink_vertices[..len],
                Some(&mink_tris),
                false,
            )?;
            win.draw_solid(
                scene_matrix,
                vec4(1.0, 1.0, 1.0, 1.0),
                &[V3::zero()],
                Points,
                false,
            )?;

            for i in 0..3 {
                let mut v = V3::zero();
                v[i] = 1.0;
                win.draw_solid(
                    scene_matrix,
                    vec4(v[0], v[1], v[2], 1.0),
                    &[-v, v],
                    LinesList,
                    false,
                )?;
            }

            win.draw_solid(
                scene_matrix,
                vec4(1.0f32, 1.0, 1.0, 1.0),
                &[V3::zero(), hit.plane.normal * hit.separation],
                LinesList,
                false,
            )?;

            hit.plane.normal * hit.separation
        } else {
            win.draw_tris(
                scene_matrix,
                vec4(1.0, 0.5, 0.5, 0.8),
                &va,
                Some(&test_state.a_tris[..]),
                false,
            )?;
            win.draw_tris(
                scene_matrix,
                vec4(0.5, 0.5, 1.0, 0.8),
                &vb,
                Some(&test_state.b_tris[..]),
                false,
            )?;

            let points = [hit.points.0, hit.points.1];

            win.draw_solid(
                scene_matrix,
                vec4(1.0, 0.5, 0.5, 1.0),
                &points,
                Points,
                false,
            )?;
            win.draw_solid(
                scene_matrix,
                vec4(1.0, 0.0, 1.0, 1.0),
                &hit.simplex,
                Points,
                false,
            )?;
            win.draw_solid(
                scene_matrix,
                vec4(1.0, 0.0, 0.0, 1.0),
                &points,
                LinesList,
                false,
            )?;
            hit.impact
        };

        let q = Quat::shortest_arc(vec3(0.0, 0.0, 1.0), hit.plane.normal);

        let q0v0 = hit_p + q.x_dir();
        let q0v1 = hit_p + q.y_dir();
        let q0v2 = hit_p - q.x_dir();
        let q0v3 = hit_p - q.y_dir();

        let q1v0 = hit_p - q.y_dir();
        let q1v1 = hit_p - q.x_dir();
        let q1v2 = hit_p + q.y_dir();
        let q1v3 = hit_p + q.x_dir();

        let quads0 = [q0v0, q0v1, q0v2, q0v0, q0v2, q0v3];
        let quads1 = [q1v0, q1v1, q1v2, q1v0, q1v2, q1v3];
        let rc = if did_hit { 0.6f32 } else { 0.0f32 };
        win.draw_tris(scene_matrix, vec4(rc, 0.0, 1.0, 0.5), &quads0, None, false)?;
        win.draw_tris(scene_matrix, vec4(rc, 1.0, 0.0, 0.5), &quads1, None, false)?;

        let mut imgui = gui.borrow_mut();
        let ui = imgui.frame();

        ui.window(im_str!("GJK"))
            .position([20.0, 20.0], imgui::Condition::Appearing)
            .always_auto_resize(true)
            .build(|| {
                ui.checkbox(im_str!("show minkowski hull"), &mut show_mink);
                let mut count = test_state.vert_count as i32;
                if ui
                    .slider_int(im_str!("vert count"), &mut count, 4, 50)
                    .build()
                {
                    if count >= 4 {
                        test_state.vert_count = count as usize;
                        a_pos = V3::zero();
                        b_pos = V3::zero();
                        test_state.reinit();
                    }
                }
                ui.drag_float3(im_str!("a pos"), a_pos.as_mut())
                    .speed(0.01)
                    .build();
                ui.drag_float3(im_str!("b pos"), b_pos.as_mut())
                    .speed(0.01)
                    .build();

                if ui.button(im_str!("new shapes"), [0.0; 2]) {
                    test_state.reinit();
                    a_pos = V3::zero();
                    b_pos = V3::zero();
                }
                ui.separator();
                ui.text(im_str!("did_hit: {}", hit.separation <= 0.0));
                ui.text(im_str!("separation: {}", hit.separation));
                ui.text(im_str!("full: {:#?}", hit));
            });
        win.end_frame_and_ui(&mut gui_renderer, ui)?;
    }
    Ok(())
}

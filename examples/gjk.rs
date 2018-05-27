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
use shared::{DemoWindow, DemoOptions, Result, object, DemoMesh, DemoObject};

use bad3d::{hull, gjk};
use bad3d::math::*;
use std::rc::Rc;
use std::cell::RefCell;

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
                self.all_verts.push(pos + object::rand_v3() - V3::splat(0.5));
            }
        }
        let com = self.all_verts.iter()
            .fold(V3::zero(), |a, b| a + *b) / (self.all_verts.len() as f32);
        for v in self.all_verts.iter_mut() {
            *v -= com;
        }
    }

    fn set_count(&mut self, vert_count: usize) {
        if vert_count >= 4 {
            self.vert_count = vert_count;
        }
        self.reinit();
    }

    fn regen(&mut self) {
        loop {
            self.a_verts.clear();
            self.a_tris.clear();
            self.b_verts.clear();
            self.b_tris.clear();

            for v in &self.all_verts[0..self.all_verts.len()/2] {
                self.a_verts.push(*v);
            }
            for v in &self.all_verts[self.all_verts.len()/2..self.all_verts.len()] {
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
    let mut win = DemoWindow::new(DemoOptions {
        title: "Hull test",
        view: M4x4::look_at(vec3(0.0, 0.0, 2.0),
                            vec3(0.0, 0.0, 0.0),
                            vec3(0.0, 1.0, 0.0)),
        clear_color: vec4(0.1, 0.1, 0.2, 1.0),
        near_far: (0.01, 50.0),
        light_pos: vec3(1.4, 0.4, 0.7),
        .. Default::default()
    })?;

    let mut test_state = GjkTestState::new();
    let mut show_mink = false;

    use glium::index::PrimitiveType::*;

    let mut model_orientation = Quat::identity();
    let mut print_hit_info = true;

    while win.is_up() {

        for &(key, down) in win.input.key_changes.iter() {
            if !down { continue; }
            match key {
                glium::glutin::VirtualKeyCode::Key1 => {
                    test_state.reinit();
                    print_hit_info = true;
                },
                glium::glutin::VirtualKeyCode::Key2 => {
                    let t = !show_mink;
                    show_mink = t;
                    print_hit_info = true;
                },
                _ => {},
            }
        }

        if win.input.mouse_down {
            let q = model_orientation;
            model_orientation = Quat::virtual_track_ball(
                vec3(0.0, 0.0, 2.0),
                vec3(0.0, 0.0, 0.0),
                win.input.mouse_vec_prev,
                win.input.mouse_vec) * q;
        }

        let scene_matrix = pose::Pose::from_rotation(model_orientation).to_mat4();

        test_state.regen();
        let hit = gjk::separated(&&test_state.a_verts[..],
                                 &&test_state.b_verts[..],
                                 true);
        let did_hit = hit.separation <= 0.0;
        if print_hit_info {
            println!(r#"
                did hit? {}\n
                separation: {}\n
                full info: {:?}
            "#, did_hit, hit.separation, hit);
        }

        let hit_p = if show_mink {
            let mut mink_vertices = Vec::with_capacity(
                test_state.a_verts.len() * test_state.b_verts.len());

            for a in &test_state.a_verts {
                for b in &test_state.b_verts {
                    mink_vertices.push(*a - *b);
                }
            }
            let mink_tris = hull::compute_hull(&mut mink_vertices[..]).unwrap().0;
            win.draw_tris(scene_matrix, vec4(1.0, 0.5, 0.5, 0.8), &mink_vertices, Some(&mink_tris), false)?;
            win.draw_solid(scene_matrix, vec4(1.0, 1.0, 1.0, 1.0), &[V3::zero()], Points)?;

            for i in 0..3 {
                let mut v = V3::zero();
                v[i] = 1.0;
                win.draw_solid(scene_matrix, vec4(v[0], v[1], v[2], 1.0), &[-v, v], LinesList)?;
            }

            win.draw_solid(scene_matrix,
                vec4(1.0f32, 1.0, 1.0, 1.0),
                &[V3::zero(), hit.plane.normal * hit.separation],
                LinesList)?;

            hit.plane.normal * hit.separation

        } else {
            win.draw_tris(scene_matrix, vec4(1.0, 0.5, 0.5, 0.8), &test_state.a_verts[..], Some(&test_state.a_tris[..]), false)?;
            win.draw_tris(scene_matrix, vec4(0.5, 0.5, 1.0, 0.8), &test_state.b_verts[..], Some(&test_state.b_tris[..]), false)?;

            let points = [hit.points.0, hit.points.1];

            win.draw_solid(scene_matrix, vec4(1.0, 0.5, 0.5, 1.0), &points, Points)?;
            win.draw_solid(scene_matrix, vec4(0.5, 0.0, 0.0, 1.0), &hit.simplex, Points)?;
            win.draw_solid(scene_matrix, vec4(1.0, 0.0, 0.0, 1.0), &points, LinesList)?;
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

        win.end_frame()?;
    }
    Ok(())
}

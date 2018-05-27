#![allow(dead_code)]
#![allow(unused_imports)]
#[macro_use]
extern crate glium;
extern crate rand;

extern crate imgui;
extern crate imgui_glium_renderer;

#[macro_use]
extern crate bad3d;

#[macro_use]
extern crate failure;

mod shared;
use shared::{DemoWindow, DemoOptions, Result, object, DemoMesh, DemoObject, input::InputState};

use bad3d::{hull, bsp, gjk, wingmesh::WingMesh, phys::{self, Shape, RigidBody, RigidBodyRef}};
use bad3d::math::*;
use std::rc::Rc;
use std::f32;
use std::cell::RefCell;

pub fn main() -> Result<()> {
    let mut win = DemoWindow::new(DemoOptions {
        title: "BSP (CSG) test",
        fov: 45.0,
        light_pos: vec3(-1.0, 0.5, 0.5),
        near_far: (0.01, 10.0),
        .. Default::default()
    })?;

    win.input.view_angle = 45.0;
    let mut draw_mode = 0;
    let mut drag_mode = 1;

    let mut cam = Pose::from_rotation(Quat::from_axis_angle(vec3(1.0, 0.0, 0.0), 60f32.to_radians()));

    let cam_dist = 5_f32;
    let mut hit_dist = 0_f32;

    let mut bpos = vec3(0.0, 0.0, 0.5);
    let mut cpos = vec3(0.8, 0.0, 0.45);

    let ac = WingMesh::new_cube(1.0);
    let bc = WingMesh::new_box(vec3(-0.5, -0.5, -1.2), vec3(0.5, 0.5, 1.2));
    let co = WingMesh::new_cube(1.0).dual_r(0.85);

    let af = ac.faces();
    let bf = bc.faces();
    let cf = co.faces();

    let mut bsp = None;

    let mut faces: Vec<bsp::Face> = Vec::new();

    while win.is_up() {
        if win.input.key_changes.iter().any(|&(a, b)| b && a == glium::glutin::VirtualKeyCode::D) {
            draw_mode = (draw_mode + 1) % 2;
        }

        if win.input.mouse.down.0 {
            match drag_mode {
                1 => {
                    cam.orientation *= Quat::virtual_track_ball(vec3(0.0, 0.0, 2.0), V3::zero(),
                        win.input.mouse_prev.vec, win.input.mouse.vec).conj();
                },
                0 => {
                    drag_mode = 1;
                    let v0 = cam.position;
                    let v1 = cam.position + cam.orientation * (win.input.mouse.vec*100.0);
                    let bhit = geom::convex_hit_check_posed(
                        bc.faces.iter().cloned(), Pose::from_translation(bpos), v0, v1);
                    let v1 = bhit.impact;
                    let chit = geom::convex_hit_check_posed(
                        co.faces.iter().cloned(), Pose::from_translation(cpos), v0, v1);
                    hit_dist = v0.dist(chit.impact);
                    if bhit.did_hit {
                        drag_mode = 2
                    }
                    if chit.did_hit {
                        drag_mode = 3;
                    }
                    if draw_mode == 2 {
                        drag_mode = 1;
                    }
                    println!("DRAG MODE => {}", drag_mode);
                },
                n => {
                    let pos = if n == 2 { &mut bpos } else { &mut cpos };
                    *pos += (cam.orientation * win.input.mouse.vec -
                             cam.orientation * win.input.mouse_prev.vec) * hit_dist;
                    bsp = None;
                },
            }
        } else {
            drag_mode = 0;
        }

        cam.position = cam.orientation.z_dir() * cam_dist;
        if bsp.is_none() {
            let bsp_a = Box::new(bsp::compile(af.clone(), WingMesh::new_cube(2.0)));
            let mut bsp_b = Box::new(bsp::compile(bf.clone(), WingMesh::new_cube(2.0)));
            let mut bsp_c = Box::new(bsp::compile(cf.clone(), WingMesh::new_cube(2.0)));

            bsp_b.translate(bpos);
            bsp_c.translate(cpos);

            bsp_b.negate();
            bsp_c.negate();

            let mut bspres = bsp::intersect(bsp_c, bsp::intersect(bsp_b, bsp_a));

            let brep = bspres.rip_brep();
            bspres.make_brep(brep, 0);

            faces = bspres.rip_brep();
            bsp = bsp::clean(bspres);
            assert!(bsp.is_some());
        }

        win.view = cam.inverse().to_mat4();

        win.wm_draw_wireframe(M4x4::identity(), vec4(0.0, 1.0, 0.5, 1.0), &ac)?;
        win.wm_draw_wireframe(M4x4::from_translation(bpos), vec4(0.0, 0.5, 1.0, 1.0), &bc)?;
        win.wm_draw_wireframe(M4x4::from_translation(cpos), vec4(0.5, 0.0, 1.0, 1.0), &co)?;

        match draw_mode {
            0 => {
                // faces (boundary)
                win.draw_faces(M4x4::identity(), &faces[..])?;
            },
            1 => {
                // cells
                let mut stack = vec![bsp.as_ref().unwrap().as_ref()];
                while let Some(n) = stack.pop() {
                    if n.leaf_type == bsp::LeafType::Under {
                        let c = n.convex.verts.iter().fold(V3::zero(), |a, &b| a+b) / (n.convex.verts.len() as f32);
                        let mut m = M4x4::from_translation(c);
                        m *= M4x4::from_scale(V3::splat(0.95));
                        m *= M4x4::from_translation(-c);
                        win.draw_tris(m, V4::splat(1.0), &n.convex.verts[..],
                            Some(&n.convex.generate_tris()[..]), false)?;
                    }
                    if let Some(ref r) = n.under {
                        stack.push(r.as_ref());
                    }
                    if let Some(ref r) = n.over {
                        stack.push(r.as_ref());
                    }
                }
            },
            _ => {
                unreachable!("bad draw_mode {}", draw_mode);
            }
        }



        win.end_frame()?;
    }
    Ok(())
}

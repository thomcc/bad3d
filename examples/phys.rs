#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(clippy::float_cmp, clippy::many_single_char_names, clippy::cast_lossless)]
#[macro_use]
extern crate glium;
use rand;

#[macro_use]
extern crate bad3d;

#[macro_use]
extern crate imgui;
use imgui_glium_renderer;

#[macro_use]
extern crate failure;
#[global_allocator]
static GLOBAL: mimallocator::Mimalloc = mimallocator::Mimalloc;

mod shared;
use crate::shared::{input::InputState, object, DemoMesh, DemoObject, DemoOptions, DemoWindow, Result};
use bad3d::prelude::*;
use glium::glutin::VirtualKeyCode;
use std::cell::RefCell;
use std::f32;
use std::rc::Rc;
use std::time::Instant;

struct DemoCamera {
    pub head_tilt: f32,
    pub head_turn: f32,
    pub position: V3,
    pub mouse_sensitivity: f32,
    pub movement_speed: f32,
}

impl DemoCamera {
    pub fn new(tilt: f32, turn: f32, pos: V3) -> DemoCamera {
        DemoCamera {
            head_tilt: tilt,
            head_turn: turn,
            position: pos,
            mouse_sensitivity: 0.15,
            movement_speed: 0.1,
        }
    }

    pub fn new_at(pos: V3) -> DemoCamera {
        DemoCamera::new(45_f32.to_radians(), 0.0, pos)
    }

    pub fn view_matrix(&self) -> M4x4 {
        self.pose().to_mat4().inverse().unwrap()
    }

    pub fn pose(&self) -> Pose {
        Pose::new(self.position, self.orientation())
    }

    pub fn orientation(&self) -> Quat {
        Quat::from_yaw_pitch_roll(self.head_turn, self.head_tilt + f32::consts::PI / 4.0, 0.0)
    }

    pub fn handle_input(&mut self, is: &InputState) {
        let impulse = {
            use crate::VirtualKeyCode::*;
            vec3(is.keys_dir(A, D), is.keys_dir(Q, E), is.keys_dir(W, S))
        };
        let (move_turn, move_tilt) = if is.mouse.down.0 && !is.shift_down() {
            let dm = is.mouse_delta();
            (
                (-dm.x * self.mouse_sensitivity * is.view_angle).to_radians() / 100.0,
                (-dm.y * self.mouse_sensitivity * is.view_angle).to_radians() / 100.0,
            )
        } else {
            (0.0, 0.0)
        };

        self.head_turn += move_turn;
        self.head_tilt += move_tilt;

        let ht = clamp(self.head_tilt, -f32::consts::PI, f32::consts::PI);
        self.head_tilt = ht;

        self.position += self.orientation() * impulse * self.movement_speed;
    }
}

fn body_hit_check(body: &RigidBody, p0: V3, p1: V3) -> Option<HitInfo> {
    let pose = body.pose;
    for shape in &body.shapes {
        let hit = geom::convex_hit_check_posed(
            shape.tris.iter().map(|&tri| {
                let (v0, v1, v2) = tri.tri_verts(&shape.vertices);
                Plane::from_tri(v0, v1, v2)
            }),
            pose,
            p0,
            p1,
        );

        if hit.is_some() {
            return hit;
        }
    }
    None
}

#[inline]
fn duration_ms(d: std::time::Duration) -> f32 {
    1000.0 * (d.as_secs() as f32 + d.subsec_nanos() as f32 / 1_000_000_000.0)
}
#[allow(clippy::cognitive_complexity)]
fn main() -> Result<()> {
    env_logger::init();
    let gui = Rc::new(RefCell::new(imgui::Context::create()));
    gui.borrow_mut().set_ini_filename(None);

    let mut win = DemoWindow::new(
        DemoOptions {
            title: "Physics engine test",
            clear_color: vec4(0.5, 0.6, 1.0, 1.0),
            near_far: (0.01, 100.0),
            light_pos: vec3(0.0, 1.2, 1.0),
            ..Default::default()
        },
        gui.clone(),
    )?;

    let ground = Shape::new_aabb(vec3(-40.0, -40.0, -5.0), vec3(40.0, 40.0, -2.0));

    let ground_mesh = DemoMesh::from_shape(&win.display, &ground, Some(vec4(0.25, 0.75, 0.25, 1.0)))?;

    let mut demo_objects: Vec<DemoObject> = Vec::new();

    let jack_push_pos = vec3(0.0, 0.0, 0.0);
    let jack_momentum = vec3(4.0, -0.8, 5.0);
    let jack_push_pos_2 = vec3(0.0, 0.5, 0.0);
    let jack_momentum_2 = vec3(0.3, 0.4, 1.0);
    let seesaw_start = vec3(0.0, -4.0, 0.25);

    demo_objects.push(DemoObject::from_wingmesh(
        &win.display,
        WingMesh::new_cone(10, 0.5, 1.0),
        vec3(1.5, 0.0, 1.5),
    )?);

    demo_objects.push(DemoObject::new_box(
        &win.display,
        V3::splat(1.0),
        vec3(-1.5, 0.0, 1.5),
        Some(quat(0.1, 0.01, 0.3, 1.0).must_norm()),
    )?);

    demo_objects.push(DemoObject::new_box(
        &win.display,
        vec3(4.0, 0.5, 0.1),
        seesaw_start,
        None,
    )?);

    let seesaw = demo_objects[demo_objects.len() - 1].body.clone();
    let log = bad3d::util::PerfLog::new();

    {
        let mut wm = WingMesh::new_cylinder(30, 1.0, 2.0);
        wm.translate(V3::splat(-1.0));
        wm.rotate(Quat::shortest_arc(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 0.0)));

        let l = DemoObject::from_wingmesh(&win.display, wm, seesaw_start + vec3(3.5, 0.0, 50.0))?;
        l.body.borrow_mut().scale_mass(6.0);

        let l2 = DemoObject::new_box(&win.display, V3::splat(0.25), seesaw_start + vec3(3.0, 0.0, 0.4), None)?;
        l2.body.borrow_mut().scale_mass(0.75);

        let r = DemoObject::new_box(&win.display, V3::splat(0.5), seesaw_start + vec3(-2.5, 0.0, 8.0), None)?;
        r.body.borrow_mut().scale_mass(2.0);

        demo_objects.push(l);
        demo_objects.push(l2);
        demo_objects.push(r);
    }

    let jack = RigidBody::new_ref(
        vec![
            Shape::new_box(vec3(1.0, 0.2, 0.2)),
            Shape::new_box(vec3(0.2, 1.0, 0.2)),
            Shape::new_box(vec3(0.2, 0.2, 1.0)),
        ],
        vec3(-5.5, 0.5, 7.5),
        RbMass::FromVolume,
    );
    jack.borrow_mut().apply_impulse(jack_push_pos, jack_momentum);
    jack.borrow_mut().apply_impulse(jack_push_pos_2, jack_momentum_2);
    demo_objects.push(DemoObject::from_body(&win.display, jack.clone())?);

    {
        let mut z = 8.5;
        while z < 14.0 {
            demo_objects.push(DemoObject::new_box(
                &win.display,
                V3::splat(0.5),
                vec3(0.0, 0.0, z),
                None,
            )?);
            z += 3.0;
        }
    }

    {
        let mut z = 15.0;
        while z < 20.0 {
            demo_objects.push(DemoObject::new_octa(
                &win.display,
                V3::splat(0.5),
                vec3(0.0, 0.0, z),
                None,
            )?);
            z += 3.0;
        }
    }
    let excessive = true;
    if excessive {
        for z in 0..2 {
            for i in 0..10 {
                for k in 0..10 {
                    demo_objects.push(DemoObject::new_cloud(
                        &win.display,
                        vec3(i as f32 - 5.0, k as f32 - 5.0, 5.0 + (z as f32)),
                        None,
                    )?);
                }
            }
        }
    }
    for i in 0..4 {
        let fi = i as f32;
        demo_objects.push(DemoObject::new_cloud(
            &win.display,
            vec3(3.0 + fi * 2.0, -3.0, 4.0 + fi * 3.0),
            None,
        )?);
    }

    demo_objects.push(DemoObject::new_box(
        &win.display,
        vec3(2.0, 0.1, 0.1),
        vec3(0.0, 0.0, -0.5),
        None,
    )?);
    demo_objects.push(DemoObject::new_box(
        &win.display,
        vec3(2.0, 0.4, 0.1),
        vec3(0.0, 1.0, -0.5),
        None,
    )?);

    {
        let mut wm = WingMesh::new_cone(30, 0.5, 2.0);
        wm.rotate(Quat::shortest_arc(vec3(0.0, 0.0, 1.0), vec3(0.0, -0.5, -0.5)));
        demo_objects.push(DemoObject::from_wingmesh(&win.display, wm, vec3(-4.0, -4.0, 4.0))?);
    }

    let mut cam = DemoCamera::new_at(vec3(0.2, -20.6, 6.5));

    let world_geom = [ground.vertices.clone()];

    let mini_cube = DemoMesh::from_shape(
        &win.display,
        &Shape::new_box(vec3(0.025, 0.025, 0.025)),
        Some(vec4(1.0, 0.0, 1.0, 1.0)),
    )?;
    let mut cube_pos = cam.pose() * vec3(0.0, 0.0, -10.0);

    let mut selected: Option<RigidBodyRef> = None;
    let mut rb_pos = V3::zero();

    let mut gui_renderer = imgui_glium_renderer::Renderer::init(&mut *gui.borrow_mut(), &win.display).unwrap();

    let mut running = false;
    let mut params = bad3d::phys::PhysParams::default();
    let mut dt_scale = 1.0;
    let mut fix_dt: bool = false;
    let mut fix_phys_fps: f32 = 60.0;
    while win.is_up() {
        let mut imgui = gui.borrow_mut();
        let ui = imgui.frame();
        for &(key, down) in win.input.key_changes.iter() {
            if !down {
                continue;
            }
            match key {
                glium::glutin::VirtualKeyCode::Space => {
                    let r = running;
                    running = !r;
                }
                glium::glutin::VirtualKeyCode::R => {
                    for &mut DemoObject { body: ref b, .. } in demo_objects.iter_mut() {
                        let mut body = b.borrow_mut();
                        body.pose = body.start_pose;
                        body.linear_momentum = V3::zero();
                        body.angular_momentum = V3::zero();
                    }
                    seesaw.borrow_mut().pose.orientation = Quat::identity();

                    jack.borrow_mut().apply_impulse(jack_push_pos, jack_momentum);
                    jack.borrow_mut().apply_impulse(jack_push_pos_2, jack_momentum_2);
                }
                _ => {}
            }
        }
        let mut targ_id = selected.as_ref().map(|rb| rb.borrow().id);
        let mouse_ray = (cam.orientation() * win.input.mouse.vec).must_norm();
        let targ_pos = if selected.is_none() {
            let mut picked = None;
            let mut best_dist = 10_000_000.0;
            let v1 = cam.position + mouse_ray * 500.0;
            for obj in &demo_objects {
                if let Some(hit) = body_hit_check(&obj.body.borrow(), cam.position, v1) {
                    let dist = hit.impact.dist(cam.position);
                    if dist < best_dist {
                        rb_pos = obj.body.borrow().pose.inverse() * hit.impact;
                        targ_id = Some(obj.body.borrow().id);
                        picked = Some((obj.body.clone(), hit));
                        best_dist = dist;
                    }
                }
            }
            if let Some((obj, hit)) = picked {
                selected = Some(obj.clone());
                hit.impact
            } else {
                cube_pos
            }
        } else {
            cube_pos
        };
        if !win.input.mouse.down.0 {
            selected = None;
        }
        cube_pos =
            cam.position + mouse_ray * (targ_pos.dist(cam.position) * 1.025_f32.powf(win.input.mouse.wheel / 30.0));

        let dt = if fix_dt {
            safe_div0(1.0, fix_phys_fps)
        } else {
            win.last_frame_time
        } * dt_scale;
        if running && dt > 0.0 {
            let mut cs = phys::ConstraintSet::new_with_params(dt, params);

            if win.input.shift_down() && win.input.mouse.down.0 {
                if let Some(body) = &selected {
                    cs.nail(None, cube_pos, Some(body.clone()), rb_pos);
                }
            }

            cs.nail(None, seesaw_start, Some(seesaw.clone()), V3::zero());
            cs.range(
                None,
                Some(seesaw.clone()),
                Quat::identity(),
                vec3(0.0, -20.0, 0.0),
                vec3(0.0, 20.0, 0.0),
            );

            let mut bodies = demo_objects
                .iter()
                .map(|item| item.body.clone())
                .collect::<Vec<phys::RigidBodyRef>>();
            log.sections.lock().unwrap().clear();
            phys::update_physics(&mut bodies[..], &mut cs, &world_geom[..], dt, &log);
        }

        cam.handle_input(&win.input);
        win.view = cam.view_matrix();

        win.draw_lit_mesh(M4x4::identity(), &ground_mesh)?;
        win.draw_lit_mesh(M4x4::from_translation(cube_pos), &mini_cube)?;

        for obj in &demo_objects {
            let model_mat = obj.body.borrow().pose.to_mat4();
            let is_hovered = Some(obj.body.borrow().id) == targ_id;
            for mesh in &obj.meshes {
                win.draw_lit_mesh(model_mat, mesh)?;
                if is_hovered {
                    win.draw_wire_mesh(model_mat, mesh, vec4(1.0, 0.0, 0.0, 1.0), false)?;
                }
            }
        }

        let framerate = ui.io().framerate;
        ui.window(im_str!("test"))
            .position([20.0, 20.0], imgui::Condition::Appearing)
            .always_auto_resize(true)
            // .size([200.0, 300.0], )
            .build(|| {
                ui.text(im_str!("fps: {:.3}", framerate));
                ui.separator();
                ui.text(im_str!("Keys:"));
                ui.text(im_str!("  reset: [R]"));
                ui.text(im_str!("  pause/unpause: [Space]"));
                ui.separator();
                if ui
                    .collapsing_header(im_str!("tunable parameters"))
                    .default_open(true)
                    .build()
                {
                    ui.checkbox(im_str!("use rk4 instead of euler?"), &mut params.use_rk4);

                    ui.slider_float(im_str!("time scale"), &mut dt_scale, 0.0, 3.0).build();
                    ui.checkbox(im_str!("fix timestep"), &mut fix_dt);
                    if fix_dt {
                        ui.input_float(im_str!("fixed fps value: "), &mut fix_phys_fps).build();
                    }

                    ui.slider_float(im_str!("gravity z"), &mut params.gravity.z, -30.0, 0.0)
                        .build();

                    ui.input_float(im_str!("restitution"), &mut params.restitution).build();
                    ui.input_float(im_str!("ballistic r"), &mut params.ballistic_response)
                        .build();
                    ui.input_float(im_str!("pos bias"), &mut params.pos_bias).build();
                    ui.input_float(im_str!("neg bias"), &mut params.neg_bias).build();
                    ui.input_float(im_str!("joint bias"), &mut params.joint_bias).build();
                    ui.input_float(im_str!("max drift"), &mut params.max_drift).build();
                    ui.input_float(im_str!("damping"), &mut params.damping).build();

                    let mut its = params.solver_iterations as i32;
                    if ui.slider_int(im_str!("solve iters"), &mut its, 0, 32).build() {
                        params.solver_iterations = its as usize
                    }

                    let mut its = params.post_solver_iterations as i32;
                    if ui.slider_int(im_str!("post solve iters"), &mut its, 0, 32).build() {
                        params.post_solver_iterations = its as usize
                    }
                }
                ui.separator();
                if ui
                    .collapsing_header(im_str!("time breakdown"))
                    .default_open(true)
                    .build()
                {
                    let mut s = log.sections.lock().unwrap();
                    s.sort_by_key(|a| a.2);

                    for (s, t, _) in s.iter() {
                        ui.text(im_str!("{}: {:.3}ms", s, duration_ms(*t)));
                    }
                }
            });
        win.end_frame_and_ui(&mut gui_renderer, ui)?;
    }
    Ok(())
}

#![allow(dead_code)]
#![allow(unused_imports)]
#[macro_use]
extern crate glium;
extern crate rand;

#[macro_use]
extern crate bad3d;

#[macro_use]
extern crate failure;

mod shared;
use shared::{DemoWindow, DemoOptions, Result, object, DemoMesh, DemoObject, input::InputState};

use bad3d::{hull, gjk, wingmesh::WingMesh, phys::{self, Shape, RigidBody, RigidBodyRef}};
use bad3d::math::*;
use std::rc::Rc;
use std::f32;
use std::cell::RefCell;


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

    pub fn matrix(&self) -> M4x4 {
        Pose::new(self.position, self.orientation())
            .to_mat4().inverse().unwrap()
    }

    pub fn orientation(&self) -> Quat {
        Quat::from_yaw_pitch_roll(
            self.head_turn, self.head_tilt+f32::consts::PI/4.0, 0.0)
    }

    pub fn handle_input(&mut self, is: &InputState) {
        let impulse = {
            use glium::glutin::VirtualKeyCode::*;
            vec3(is.keys_dir(A, D), is.keys_dir(Q, E), is.keys_dir(W, S))
        };
        let mut move_turn = 0.0;
        let mut move_tilt = 0.0;
        if is.mouse_down {
            let dm = is.mouse_delta();
            move_turn = (-dm.x*self.mouse_sensitivity*is.view_angle).to_radians()/100.0;
            move_tilt = (-dm.y*self.mouse_sensitivity*is.view_angle).to_radians()/100.0;
        }

        self.head_turn += move_turn;
        self.head_tilt += move_tilt;

        let ht = self.head_tilt.clamp(-f32::consts::PI, f32::consts::PI);
        self.head_tilt = ht;

        self.position += self.orientation() * impulse * self.movement_speed;
    }
}

fn main() -> Result<()> {
    let mut win = DemoWindow::new(DemoOptions {
        title: "Physics engine test",
        clear_color: vec4(0.5, 0.6, 1.0, 1.0),
        near_far: (0.01, 100.0),
        light_pos: vec3(0.0, 1.2, 1.0),
        .. Default::default()
    })?;

    let ground = Shape::new_aabb(vec3(-10.0, -10.0, -5.0), vec3(10.0, 10.0, -2.0));

    let ground_mesh = DemoMesh::from_shape(&win.display, &ground,
        Some(vec4(0.25, 0.75, 0.25, 1.0)))?;

    let mut demo_objects: Vec<DemoObject> = Vec::new();

    let jack_push_pos   = vec3(0.0, 0.0, 0.0);
    let jack_momentum   = vec3(4.0, -0.8, 5.0);
    let jack_push_pos_2 = vec3(0.0, 0.5, 0.0);
    let jack_momentum_2 = vec3(0.3, 0.4, 1.0);
    let seesaw_start    = vec3(0.0, -4.0, 0.25);

    demo_objects.push(DemoObject::from_wingmesh(&win.display,
                                                WingMesh::new_cone(10, 0.5, 1.0),
                                                vec3(1.5, 0.0, 1.5))?);

    demo_objects.push(DemoObject::new_box(&win.display, V3::splat(1.0), vec3(-1.5, 0.0, 1.5),
                      Some(quat(0.1, 0.01, 0.3, 1.0).must_norm()))?);

    demo_objects.push(DemoObject::new_box(&win.display, vec3(4.0, 0.5, 0.1),
                                          seesaw_start, None)?);

    let seesaw = demo_objects[demo_objects.len() - 1].body.clone();

    {
        let mut wm = WingMesh::new_cylinder(30, 1.0, 2.0);
        wm.translate(V3::splat(-1.0));
        wm.rotate(Quat::shortest_arc(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 0.0)));

        let l = DemoObject::from_wingmesh(&win.display, wm, seesaw_start + vec3(3.5, 0.0, 50.0))?;
        l.body.borrow_mut().scale_mass(6.0);

        let l2 = DemoObject::new_box(&win.display, V3::splat(0.25), seesaw_start + vec3(3.0, 0.0, 0.4), None)?;
        l2.body.borrow_mut().scale_mass(0.75);

        let r = DemoObject::new_box(&win.display, V3::splat(0.5), seesaw_start + vec3(-2.5, 0.0, 5.0), None)?;
        r.body.borrow_mut().scale_mass(2.0);

        demo_objects.push(l);
        demo_objects.push(l2);
        demo_objects.push(r);
    }

    let jack = RigidBody::new_ref(vec![
        Shape::new_box(vec3(1.0, 0.2, 0.2)),
        Shape::new_box(vec3(0.2, 1.0, 0.2)),
        Shape::new_box(vec3(0.2, 0.2, 1.0)),
    ], vec3(-5.5, 0.5, 7.5), 1.0);
    jack.borrow_mut().apply_impulse(jack_push_pos, jack_momentum);
    jack.borrow_mut().apply_impulse(jack_push_pos_2, jack_momentum_2);
    demo_objects.push(DemoObject::from_body(&win.display, jack.clone())?);

    {
        let mut z = 5.5;
        while z < 14.0 {
            demo_objects.push(DemoObject::new_box(&win.display, V3::splat(0.5), vec3(0.0, 0.0, z), None)?);
            z += 3.0;
        }
    }

    {
        let mut z = 15.0;
        while z < 20.0 {
            demo_objects.push(DemoObject::new_octa(&win.display, V3::splat(0.5), vec3(0.0, 0.0, z), None)?);
            z += 3.0;
        }
    }

    for i in 0..4 {
        let fi = i as f32;
        demo_objects.push(DemoObject::new_cloud(&win.display,
            vec3(3.0+fi*2.0, -3.0, 4.0+fi*3.0), None)?);
    }

    demo_objects.push(DemoObject::new_box(&win.display, vec3(2.0, 0.1, 0.1), vec3(0.0, 0.0, -0.5), None)?);
    demo_objects.push(DemoObject::new_box(&win.display, vec3(2.0, 0.4, 0.1), vec3(0.0, 1.0, -0.5), None)?);

    {
        let mut wm = WingMesh::new_cone(30, 0.5, 2.0);
        wm.rotate(Quat::shortest_arc(vec3(0.0, 0.0, 1.0), vec3(0.0, -0.5, -0.5)));
        demo_objects.push(DemoObject::from_wingmesh(&win.display, wm, vec3(-4.0, -4.0, 4.0))?);
    }

    let mut cam = DemoCamera::new_at(vec3(0.2, -20.6, 6.5));

    let world_geom = [ground.vertices.clone()];

    let mut running = false;
    while win.is_up() {
        for &(key, down) in win.input.key_changes.iter() {
            if !down { continue; }
            match key {
                glium::glutin::VirtualKeyCode::Space => {
                    let r = running;
                    running = !r;
                },
                glium::glutin::VirtualKeyCode::R => {
                    for &mut DemoObject{ body: ref b, .. } in demo_objects.iter_mut() {
                        let mut body = b.borrow_mut();
                        body.pose = body.start_pose;
                        body.linear_momentum = V3::zero();
                        body.angular_momentum = V3::zero();
                    }
                    seesaw.borrow_mut().pose.orientation = Quat::identity();

                    jack.borrow_mut().apply_impulse(jack_push_pos, jack_momentum);
                    jack.borrow_mut().apply_impulse(jack_push_pos_2, jack_momentum_2);
                },
                _ => {},
            }
        }

        if running {
            let dt = 1.0 / 60.0;
            let mut cs = phys::ConstraintSet::new(dt);

            cs.nail(None, seesaw_start, Some(seesaw.clone()), V3::zero());
            cs.range(None, Some(seesaw.clone()), Quat::identity(),
                vec3(0.0, -20.0, 0.0), vec3(0.0, 20.0, 0.0));

            let mut bodies = demo_objects.iter()
                .map(|item| item.body.clone())
                .collect::<Vec<phys::RigidBodyRef>>();


            phys::update_physics(&mut bodies[..], &mut cs, &world_geom[..], dt);
        }

        cam.handle_input(&win.input);
        win.view = cam.matrix();

        win.draw_lit_mesh(M4x4::identity(), &ground_mesh)?;

        for obj in &demo_objects {
            let model_mat = obj.body.borrow().pose.to_mat4();
            for mesh in &obj.meshes {
                win.draw_lit_mesh(model_mat, mesh)?;
            }
        }
        win.end_frame()?;
    }
    Ok(())
}


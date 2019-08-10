#![allow(dead_code)]
#![allow(unused_imports)]
#[macro_use]
extern crate glium;
use rand;

#[macro_use]
extern crate bad3d;

#[macro_use]
extern crate failure;

use imgui;

mod shared;
use crate::shared::{object, DemoMesh, DemoObject, DemoOptions, DemoWindow, Result};

use bad3d::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;

fn main() -> Result<()> {
    env_logger::init();
    let body_sizes = [
        vec3(0.25, 0.50, 0.10), // torso
        vec3(0.25, 0.05, 0.05), // limb upper bones
        vec3(0.25, 0.05, 0.05),
        vec3(0.25, 0.05, 0.05),
        vec3(0.25, 0.05, 0.05),
        vec3(0.25, 0.05, 0.05),
        vec3(0.25, 0.05, 0.05),
        vec3(0.05, 0.05, 0.25), // limb lower bones
        vec3(0.05, 0.05, 0.25),
        vec3(0.05, 0.05, 0.25),
        vec3(0.05, 0.05, 0.25),
        vec3(0.05, 0.05, 0.25),
        vec3(0.05, 0.05, 0.25),
    ];

    let joints = [
        (0, 1, 0.25_f32, vec3(0.25, -0.5, 0.0), vec3(-0.25, 0.0, 0.0)), // upper limbs to torso
        (0, 2, -0.25_f32, vec3(0.25, 0.0, 0.0), vec3(-0.25, 0.0, 0.0)),
        (0, 3, 0.25_f32, vec3(0.25, 0.5, 0.0), vec3(-0.25, 0.0, 0.0)),
        (0, 4, 0.25_f32, vec3(-0.25, -0.5, 0.0), vec3(0.25, 0.0, 0.0)),
        (0, 5, -0.25_f32, vec3(-0.25, 0.0, 0.0), vec3(0.25, 0.0, 0.0)),
        (0, 6, 0.25_f32, vec3(-0.25, 0.5, 0.0), vec3(0.25, 0.0, 0.0)),
        (1, 7, 0.0_f32, vec3(0.25, 0.0, 0.0), vec3(0.0, 0.0, 0.25)), // lower limb to upper limb
        (2, 8, 0.0_f32, vec3(0.25, 0.0, 0.0), vec3(0.0, 0.0, 0.25)),
        (3, 9, 0.0_f32, vec3(0.25, 0.0, 0.0), vec3(0.0, 0.0, 0.25)),
        (4, 10, 0.0_f32, vec3(-0.25, 0.0, 0.0), vec3(0.0, 0.0, 0.25)),
        (5, 11, 0.0_f32, vec3(-0.25, 0.0, 0.0), vec3(0.0, 0.0, 0.25)),
        (6, 12, 0.0_f32, vec3(-0.25, 0.0, 0.0), vec3(0.0, 0.0, 0.25)),
    ];

    let mut window = DemoWindow::new(
        DemoOptions {
            title: "Powered ragdoll physics test",
            view: M4x4::look_at(
                vec3(0.0, -8.0, 0.0),
                vec3(0.0, 0.0, 0.0),
                vec3(0.0, 0.0, 1.0),
            ),
            light_pos: vec3(5.0, 1.2, 1.0),
            clear_color: vec4(0.5, 0.6, 1.0, 1.0),
            ..Default::default()
        },
        Rc::new(RefCell::new(imgui::Context::create())),
    )?;
    let Shape {
        vertices: ground_verts,
        tris: ground_tris,
    } = Shape::new_aabb(vec3(-5.0, -5.0, -3.0), vec3(5.0, 5.0, -2.0));

    let ground_mesh = Box::new(DemoMesh::new(
        &window.display,
        ground_verts.clone(),
        ground_tris,
        vec4(0.25, 0.75, 0.25, 1.0),
    )?);

    let mut demo_objects = Vec::with_capacity(body_sizes.len() + 2);

    for size in body_sizes.iter() {
        let obj = DemoObject::new_box(&window.display, *size, V3::zero(), None)?;
        {
            // Mass is based on volume by default on DemoObjects, but for this
            // demo that's the wrong call. Hackily fix it up.
            let mut body = obj.body.borrow_mut();
            let mass = body.mass;
            body.scale_mass(1.0 / mass);
        }
        demo_objects.push(obj);
        for m in demo_objects.last_mut().unwrap().meshes.iter_mut() {
            m.color = vec4(0.8, 0.4, 0.2, 1.0);
        }
    }

    demo_objects[0].body.borrow_mut().scale_mass(5.0);

    for joint in joints.iter() {
        let mut body0 = demo_objects[joint.0].body.borrow_mut();
        let mut body1 = demo_objects[joint.1].body.borrow_mut();

        body0.ignored.insert(body1.id);
        body1.ignored.insert(body0.id);
        let pos = body0.pose * joint.3 - body1.pose.orientation * joint.4;
        body1.pose.position = pos;
        body1.start_pose.position = pos;
    }

    demo_objects.push(DemoObject::new_box(
        &window.display,
        vec3(2.0, 0.1, 0.1),
        vec3(0.0, 0.0, -0.5),
        None,
    )?);
    demo_objects.push(DemoObject::new_box(
        &window.display,
        vec3(2.0, 0.4, 0.1),
        vec3(0.0, 1.0, -0.5),
        None,
    )?);

    let torque_limit = 38.0;
    let mut time = 0.0;

    let world_geom = [ground_verts];

    while window.is_up() {
        let dt = 1.0 / 60.0f32;
        time += 0.06f32;

        let mut cs = phys::ConstraintSet::new(dt);

        for joint in joints.iter() {
            cs.nail(
                Some(demo_objects[joint.0].body.clone()),
                joint.3,
                Some(demo_objects[joint.1].body.clone()),
                joint.4,
            );

            cs.powered_angle(
                Some(demo_objects[joint.0].body.clone()),
                Some(demo_objects[joint.1].body.clone()),
                quat(
                    0.0,
                    joint.2 * time.cos(),
                    joint.2 * time.sin(),
                    (1.0 - joint.2 * joint.2).sqrt(),
                ),
                torque_limit,
            );
        }

        let mut bodies = demo_objects
            .iter()
            .map(|item| item.body.clone())
            .collect::<Vec<phys::RigidBodyRef>>();

        for body in bodies[body_sizes.len()..].iter_mut() {
            cs.under_plane(
                body.clone(),
                Plane::from_norm_and_point(phys::GRAVITY.must_norm(), vec3(5.0, 5.0, -10.0)),
                None,
            );
        }

        phys::update_physics(&mut bodies[..], &mut cs, &world_geom[..], dt);

        window.draw_lit_mesh(M4x4::identity(), &*ground_mesh)?;
        for obj in demo_objects.iter() {
            let pose = obj.body.borrow().pose.to_mat4();
            for mesh in &obj.meshes {
                window.draw_lit_mesh(pose, mesh)?;
            }
        }

        let need_reset = bodies[0..body_sizes.len()]
            .iter()
            .find(|b| b.borrow().pose.position.length() > 25.0)
            .is_some();
        if need_reset {
            for body in bodies[0..body_sizes.len()].iter_mut() {
                let momentum = body.borrow().linear_momentum;
                let start_pose = body.borrow_mut().start_pose;
                body.borrow_mut().linear_momentum = -momentum;
                body.borrow_mut().pose = start_pose;
            }
            time = 0.0;
        }
        window.end_frame()?;
    }
    Ok(())
}

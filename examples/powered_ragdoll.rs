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
            view: M4x4::look_at(vec3(0.0, -8.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0)),
            light_pos: vec3(5.0, 1.2, 1.0),
            clear_color: vec4(0.5, 0.6, 1.0, 1.0),
            ..Default::default()
        },
        Rc::new(RefCell::new(imgui::Context::create())),
    )?;
    let mut scene = bad3d::phys::PhysScene::default();
    let ground = Shape::new_aabb(vec3(-5.0, -5.0, -3.0), vec3(5.0, 5.0, -2.0));

    let ground_mesh = Box::new(DemoMesh::from_shape(
        &window.display,
        &ground,
        vec4(0.25, 0.75, 0.25, 1.0).into(),
    )?);
    scene.world.push(ground);

    let handles = body_sizes
        .iter()
        .enumerate()
        .map(|(i, size)| {
            let mass = if i == 0 { 5.0 } else { 1.0 };
            scene
                .add()
                .box_collider(*size)
                .build_with(|b| b.scale_mass(mass / b.mass))
        })
        .collect::<Vec<_>>();

    for joint in joints.iter() {
        scene.bodies[handles[joint.0]].ignored.push(handles[joint.1]);
        scene.bodies[handles[joint.1]].ignored.push(handles[joint.0]);
        let pos =
            scene.bodies[handles[joint.0]].pose * joint.3 - scene.bodies[handles[joint.1]].pose.orientation * joint.4;
        scene.bodies[handles[joint.1]].pose.position = pos;
        scene.bodies[handles[joint.1]].start_pose.position = pos;
    }
    scene
        .add()
        .box_collider(vec3(2.0, 0.1, 0.1))
        .at(vec3(0.0, 0.0, -0.5))
        .build();
    scene
        .add()
        .box_collider(vec3(2.0, 0.4, 0.1))
        .at(vec3(0.0, 1.0, -0.5))
        .build();

    use std::collections::HashMap;
    let mut body_meshes: HashMap<handy::Handle, Vec<DemoMesh>> = HashMap::with_capacity(scene.bodies.len());
    for (h, b) in scene.bodies.iter_with_handles() {
        let meshes: Vec<DemoMesh> = b
            .shapes
            .iter()
            .map(|s| {
                let mut m = DemoMesh::from_shape(&window.display, s, None).unwrap();
                m.color = vec4(0.8, 0.4, 0.2, 1.0);
                m
            })
            .collect::<Vec<_>>();
        body_meshes.insert(h, meshes);
    }

    let torque_limit = 38.0;
    let mut time = 0.0;

    // let world_geom = [ground_verts];

    while window.is_up() {
        let dt = 1.0 / 60.0f32;
        time += 0.06f32;
        scene.begin(dt);

        // let mut cs = phys::ConstraintSet::new(dt);

        for joint in joints.iter() {
            scene.constraints.nail(
                Some(&scene.bodies[handles[joint.0]]),
                joint.3,
                Some(&scene.bodies[handles[joint.1]]),
                joint.4,
            );

            scene.constraints.powered_angle(
                Some(&scene.bodies[handles[joint.0]]),
                Some(&scene.bodies[handles[joint.1]]),
                quat(
                    0.0,
                    joint.2 * time.cos(),
                    joint.2 * time.sin(),
                    (1.0 - joint.2 * joint.2).sqrt(),
                ),
                torque_limit,
            );
        }

        // let mut bodies = demo_objects
        //     .iter()
        //     .map(|item| item.body.clone())
        //     .collect::<Vec<phys::RigidBodyRef>>();

        // for &body in &handles {
        // scene.constraints.under_plane(
        //     &scene.bodies[body],
        //     Plane::from_norm_and_point(scene.params.gravity.must_norm(), vec3(5.0, 5.0, -10.0)),
        //     None,
        // );
        // }

        // perf_log.sections.lock().unwrap().clear();
        scene.simulate();
        // phys::update_physics(&mut bodies[..], &mut cs, &world_geom[..], dt, &perf_log);

        window.draw_lit_mesh(M4x4::identity(), &*ground_mesh)?;
        for (k, v) in &body_meshes {
            let pose = scene.bodies[*k].pose.to_mat4();
            for mesh in v {
                window.draw_lit_mesh(pose, mesh)?;
            }
        }

        let need_reset = handles.iter().any(|b| scene.bodies[*b].pose.position.length() > 25.0);
        if need_reset {
            for &b in &handles {
                let momentum = scene.bodies[b].linear_momentum;
                let start_pose = scene.bodies[b].start_pose;
                scene.bodies[b].linear_momentum = -momentum;
                scene.bodies[b].pose = start_pose;
            }
            time = 0.0;
        }
        window.end_frame()?;
    }
    Ok(())
}

#![allow(dead_code)]
#![allow(unused_imports)]
#[macro_use]
extern crate glium;
extern crate rand;

#[macro_use]
extern crate bad3d;

#[macro_use]
extern crate imgui;
extern crate imgui_glium_renderer;

#[macro_use]
extern crate more_asserts;

#[macro_use]
extern crate failure;

mod shared;

use shared::{DemoWindow, DemoOptions, Result, object, DemoMesh, DemoObject, input::InputState};

use glium::glutin::VirtualKeyCode;
use bad3d::{hull, gjk, wingmesh::WingMesh, phys::{self, Shape, RigidBody, RigidBodyRef}, bsp::{self, BspNode}};
use bad3d::math::*;
use std::rc::Rc;
use std::{f32, u16};
use std::time::Instant;
use std::cell::RefCell;

const GRAVITY: V3 = phys::GRAVITY;

const DAMP_AIR: f32 = 1.0;
const DAMP_GROUND: f32 = 10.0;
const JUMP_SPEED: f32 = 5.0;
const MAX_SPEED: f32 = 10.0;
const MOUSE_SENSITIVITY: f32 = 2.0;
const TIMESTEP: f32 = 0.016;

#[derive(Debug, Clone)]
struct Player {
    pub pose: Pose,
    pub pos_new: V3,
    pub pos_old: V3,
    pub vel: V3,

    pub height: f32,
    pub radius: f32,
    pub head_tilt: f32,

    pub ground_norm: Option<V3>,
}

impl Player {
    pub fn new() -> Self {
        Self {
            pose: Pose::from_translation(vec3(0.1, 0.1, 0.1)),
            pos_new: V3::zero(),
            pos_old: V3::zero(),
            vel: V3::zero(),
            height: 1.2,
            radius: 0.2,
            head_tilt: 0.0,
            ground_norm: None,
        }
    }

    pub fn eye_pose(&self) -> Pose {
        self.pose * Pose::new(
            vec3(0.0, 0.0, self.height * 0.9),
            Quat::from_axis_angle(vec3(1.0, 0.0, 0.0),
                                  (90.0 + self.head_tilt).to_radians())
        )
    }

    #[inline]
    fn check_sweep_if(&self, cnd: bool, node: &BspNode, v0: V3, v1: V3) -> Option<geom::HitInfo> {
        if !cnd {
            return None;
        }
        node.hit_check_cylinder(self.radius, self.height, v0, v1, V3::zero(), false)
    }

    pub fn area_check(&mut self, node: &BspNode, dt: f32) {
        let mut ground_dist = 0.0;
        let mut wall_contact = false;
        let target = self.pos_new;
        let mut impact: V3;
        let mut hit = 0usize;
        while let Some(hi) = self.check_sweep_if(hit < 5, node, self.pose.position, self.pos_new) {
            hit += 1;

            let norm = hi.normal;
            impact = hi.impact;
            if norm.z > 0.0 {
                self.ground_norm = Some(norm);
                ground_dist = -dot(norm, impact);
            } else if norm.z < -0.5 {
                wall_contact = true;
            }
            self.pos_new = Plane::from_norm_and_point(norm, impact).project(self.pos_new) + (norm * 0.00001);
            if let Some(ground_normal) = self.ground_norm {
                if dot(norm, ground_normal) < 0.0 && dot(ground_normal, self.pos_new) + ground_dist <= 0.0 {
                    let mut slide = Plane::new(norm, 0.0).project(ground_normal);
                    slide = slide * -(dot(ground_normal, self.pos_new) + ground_dist) / (dot(slide, ground_normal));
                    slide = slide + slide.norm_or_unit() * 0.000001;
                    self.pos_new += slide;
                }
            }
        }

        if hit == 5 {
            self.pos_new = self.pose.position;
        }

        if self.ground_norm.is_some() && wall_contact {
            let mut pos_up = self.pos_new + vec3(0.0, 0.0, 0.4);
            hit = 1;
            while let Some(hi) = self.check_sweep_if(hit < 5, node, self.pos_new, pos_up) {
                hit += 1;
                pos_up = hi.impact;
                pos_up.z -= 0.00001;
            }

            let mut target_up = target + (pos_up - self.pos_new);
            while let Some(hi) = self.check_sweep_if(hit < 5, node, pos_up, target_up) {
                hit += 1;
                impact = hi.impact;
                // slide along plane of impact
                target_up = Plane::from_norm_and_point(hi.normal, impact).project(target_up) + hi.normal * 0.00001;
            }

            let mut pos_drop = target_up - (pos_up - self.pos_new);
            while let Some(hi) = self.check_sweep_if(hit < 5, node, target_up, pos_drop) {
                hit += 1;
                pos_drop = hi.impact;
                pos_drop.z += 0.00001;
            }
            if hit != 5 {
                self.pos_new = pos_drop;
            }
        }
        if hit != 0 {
            self.vel = (self.pos_new - self.pose.position) / dt;
        }
    }

    pub fn update(&mut self, mut mouse: V2, thrust: V3, bsp: &BspNode, dt: f32) {
        mouse.y *= -1.0;
        let (damp, thrust_dom) = if let Some(gnorm) = self.ground_norm {
            let dom = Quat::shortest_arc(vec3(0.0, 0.0, 1.0), gnorm) * self.pose.orientation;
            (DAMP_GROUND, dom)
        } else {
            (DAMP_AIR, self.pose.orientation)
        };
        let contact_velocity = V3::zero(); // if ground is moving it goes here.
        let acc_damping = (self.vel - contact_velocity) * -damp;
        let mut micro_impulse = vec3(0.0, 0.0, 0.0);
        if thrust.dot(thrust) == 0.0 {
            if let Some(gnorm) = self.ground_norm {
                micro_impulse = gnorm * (GRAVITY.z * gnorm.z) - vec3(0.0, 0.0, GRAVITY.z);
            }
        }
        if self.ground_norm.is_some() {
            if self.vel.z < 0.0 {
                self.vel.z = 0.0;
            }
            if thrust.z > 0.0 {
                self.vel.z = JUMP_SPEED;
                self.ground_norm = None;
            }
        }
        let accel = GRAVITY + acc_damping + micro_impulse +
            (thrust_dom.y_dir() * thrust.y + thrust_dom.x_dir() * thrust.x) * MAX_SPEED * damp;

        self.vel += accel * dt;
        self.pos_new = self.pose.position + self.vel * dt;
        self.pose.orientation = (
            self.pose.orientation * Quat::from_axis_angle(vec3(0.0, 0.0, 1.0), -mouse.x * MOUSE_SENSITIVITY)
        ).must_norm();

        self.head_tilt = clamp(self.head_tilt + (mouse.y * MOUSE_SENSITIVITY).to_degrees(),
                               -90.0, 90.0);

        self.pos_old = self.pose.position;
        self.ground_norm = None;
        self.area_check(bsp, dt);
        self.pose.position = self.pos_new;
    }
}

const Q_CUBE_RADIUS: usize = 4;
const Q_SNAP: f32 = 0.5;

fn cube_projected_v(v: V3) -> V3 {
    v * safe_div0(Q_CUBE_RADIUS as f32, v.abs().max_elem())
}

fn cube_projected_p(p: Plane) -> Plane {
    let p4 = p.to_v4();
    Plane::from_v4(p4 * safe_div0(Q_CUBE_RADIUS as f32, p.normal.abs().max_elem()))
}

fn quantized_v(v: V3) -> V3 {
    cube_projected_v(v).round().must_norm()
}

fn quantum_dist(v: V3) -> f32 {
    Q_SNAP / cube_projected_v(v).round().length()
}

fn quantized_p(p: Plane) -> Plane {
    let n = cube_projected_v(p.normal);
    let mag = n.length();
    Plane::new(n / mag, round_to(p.offset, Q_SNAP / mag))
}

struct Blaster {
    mashers: Vec<Box<BspNode>>,
    idx: usize,
}

impl Blaster {
    pub fn new() -> Self {
        Self { mashers: Blaster::build_mashers(), idx: 0 }
    }

    pub fn blast(&mut self, bsp: Box<BspNode>, p: V3, s: f32) -> Box<BspNode> {
        if self.mashers.len() == 0 {
            self.mashers = Blaster::build_mashers();
            self.idx = 0;
        }
        self.idx = (self.idx + 1) % self.mashers.len();
        let mut b = self.mashers[self.idx].clone();
        b.scale3(vec3(s, s, s));
        b.translate(p);
        bsp::intersect(b, bsp)
    }

    fn build_mashers() -> Vec<Box<BspNode>> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut mashers = Vec::with_capacity(10);
        for _ in 0..10 {
            let mut mash_box = WingMesh::new_cube(0.5);
            for _ in 0..8 {
                let rn = vec3(
                    ((rng.gen_range(0, 9) as f32) - 4.0) / 4.0,
                    ((rng.gen_range(0, 9) as f32) - 4.0) / 4.0,
                    ((rng.gen_range(0, 9) as f32) - 4.0) / 4.0,
                ).norm_or_v(vec3(0.0, 0.0, 1.0));
                let rd = -(rng.gen::<f32>() * 0.5 + 0.125);
                mash_box = mash_box.crop(quantized_p(Plane::new(rn, rd)));
            }
            let mut b = bsp::compile(mash_box.faces(), WingMesh::new_cube(16.0));
            b.negate();
            b.derive_convex(WingMesh::new_cube(30.51));
            b.make_boundary(vec![], 0);
            mashers.push(b);
        }
        mashers
    }
}

fn make_bsp(wm0: &WingMesh, wm1: WingMesh) -> Box<BspNode> {
    bsp::compile(wm0.faces(), wm1)
}

fn build_scene_bsp() -> Box<BspNode> {
    let arena = WingMesh::new_box(vec3(-20.0, -20.0, -10.0), vec3(20.0, 20.0, 10.0));

    let mut bsp_geom = bsp::compile(arena.faces(), WingMesh::new_cube(64.0));
    bsp_geom.negate();

    let boxes = [
        (vec3(-11.0, -11.0,-0.25), vec3(11.0, 11.0, 0.0)),
        (vec3(  4.0, -11.0, -6.0), vec3( 4.5, 11.0, 6.0)),
        (vec3( -4.5, -11.0, -6.0), vec3(-4.0, 11.0, 6.0)),
        (vec3(-11.0,   4.0, -6.0), vec3(11.0,  4.5, 6.0)),
        (vec3(-11.0,  -4.5, -6.0), vec3(11.0, -4.0, 6.0)),
        (vec3(  2.5,   1.5,  2.0), vec3( 3.5,  3.5, 4.5)),
    ];

    for (min, max) in boxes.iter() {
        bsp_geom = bsp::union(
            bsp::compile(WingMesh::new_box(*min, *max).faces(),
                         WingMesh::new_cube(32.0)),
            bsp_geom);
    }

    for door_x in &[-7.0f32, 0.0, 7.0] {
        let mut dx = bsp::compile(
            WingMesh::new_box(vec3(door_x - 1.0, 9.0, 0.1),
                              vec3(door_x + 1.0, 9.0, 2.5)).faces(),
            WingMesh::new_cube(32.0));
        dx.negate();
        bsp_geom = bsp::intersect(dx, bsp_geom);
    }

    for y_door in &[-7.0f32, 7.0] {
        let mut dy = bsp::compile(
            WingMesh::new_box(vec3(-9.0, y_door - 1.0, 0.1),
                              vec3( 9.0, y_door + 1.0, 2.5)).faces(),
            WingMesh::new_cube(32.0));
        dy.negate();
        bsp_geom = bsp::intersect(dy, bsp_geom);
    }
    bsp_geom = bsp::clean(bsp_geom).unwrap();

    bsp_geom.rebuild_boundary();
    bsp_geom
}

fn bsp_meshes(f: &glium::Display, bsp: &mut BspNode, color: V4) -> Result<Vec<DemoMesh>> {
    let boundary = bsp.rip_boundary();
    let mut vs = Vec::new();
    let mut ts = Vec::new();
    let mut ms = vec![];
    for face in &boundary {
        let mut offset = vs.len();
        let vertices = face.vertex.len();
        if vertices == 0 {
            continue;
        }
        assert_lt!(vertices, u16::MAX as usize);
        if offset + vertices >= u16::MAX as usize {
            ms.push(DemoMesh::new(f, vs.drain(..).collect(), ts.drain(..).collect(), color)?);
            assert_eq!(vs.len(), 0);
            assert_eq!(ts.len(), 0);
            offset = 0;
        }
        vs.extend(&face.vertex);

        ts.reserve(vertices - 2);
        for i in 1..(vertices - 1) {
            ts.push([offset as u16, (i + offset) as u16, (i + offset + 1) as u16]);
        }
    }
    if vs.len() > 0 || ts.len() > 0 {
        ms.push(DemoMesh::new(f, vs, ts, color)?);
    }
    bsp.make_boundary(boundary, 0);
    Ok(ms)
}

fn bsp_cell_meshes(f: &glium::Display, bsp: &BspNode, color: V4) -> Result<Vec<DemoMesh>> {

    let mut vs = Vec::new();
    let mut ts = Vec::new();
    let mut ms = vec![];
    let mut stack = vec![bsp];
    while let Some(n) = stack.pop() {
        if n.leaf_type == bsp::LeafType::Under {
            let mut offset = vs.len();
            let vertices = n.convex.verts.len();
            if vertices == 0 {
                continue;
            }
            if offset + vertices >= u16::MAX as usize {
                ms.push(DemoMesh::new(f, vs.drain(..).collect(), ts.drain(..).collect(), color)?);
                assert_eq!(vs.len(), 0);
                assert_eq!(ts.len(), 0);
                offset = 0;
            }
            vs.extend(&n.convex.verts);
            let new_tris = n.convex.generate_tris();
            ts.reserve(new_tris.len());
            ts.extend(new_tris.into_iter().map(|t|
                [t[0] + (offset as u16), t[1] + (offset as u16), t[2] + (offset as u16)]));
        }
        if let Some(ref r) = n.under {
            stack.push(r.as_ref());
        }
        if let Some(ref r) = n.over {
            stack.push(r.as_ref());
        }
    }
    if vs.len() > 0 || ts.len() > 0 {
        ms.push(DemoMesh::new(f, vs, ts, color)?);
    }
    Ok(ms)
}

fn main() -> Result<()> {
    let gui = Rc::new(RefCell::new(imgui::ImGui::init()));
    gui.borrow_mut().set_ini_filename(None);

    let mut win = DemoWindow::new(DemoOptions {
        title: "FPS (bsp test)",
        clear_color: vec4(0.5, 0.6, 1.0, 1.0),
        near_far: (0.01, 100.0),
        light_pos: vec3(0.0, 1.2, 1.0),
        fov: 45.0,
        .. Default::default()
    }, gui.clone())?;

    let mut gui_renderer = imgui_glium_renderer::Renderer::init(
        &mut *gui.borrow_mut(), &win.display).unwrap();

    let mut fly_cam = false;
    let mut paused = false;

    let mut player = Player::new();
    let mut camera = player.eye_pose();// Pose::new(vec3(0.0, 0.0, 20.0), Quat::identity());
    let mut blaster = Blaster::new();

    let mut bsp_geom = build_scene_bsp();

    let scene_color = vec4(0.4, 0.4, 0.4, 1.0);

    let mut scene_meshes = bsp_cell_meshes(&win.display, &mut bsp_geom, scene_color)?;

    while win.is_up() {
        let mut imgui = gui.borrow_mut();
        let ui = win.get_ui(&mut *imgui);
        let mut exit = false;
        for &(key, down) in win.input.key_changes.iter() {
            if !down { continue; }
            match key {
                glium::glutin::VirtualKeyCode::Escape => {
                    exit = true;
                }
                glium::glutin::VirtualKeyCode::P => {
                    paused = !paused;
                }
                glium::glutin::VirtualKeyCode::F => {
                    fly_cam = !fly_cam;
                }
                _ => {}
            }
        }
        if exit {
            win.end_frame()?;
            return Ok(());
        }
        if paused {
             win.ungrab_cursor();
        } else {
             win.grab_cursor();
        }

        if !paused {
            let mut do_blast = None;
            let mut blast_sz = 1.0;
            if win.input.mouse.down.0 {
                let h = bsp_geom.hit_check(camera.position, camera * vec3(0.0, 0.0, -100.0));
                let dist = if !win.input.mouse_prev.down.0 && win.input.shift_down() {
                    blast_sz = 4.0;
                    14.0
                } else {
                    3.0
                };
                if let Some(hit) = h {
                    if hit.impact.length() < dist {
                        do_blast = Some(hit.impact);
                    }
                }
            }
            if let Some(impact) = do_blast {
                bsp_geom = blaster.blast(bsp_geom, impact, blast_sz);
                scene_meshes = bsp_cell_meshes(&win.display, &mut bsp_geom, scene_color)?;
            }

            let thrust = {
                use glium::glutin::VirtualKeyCode::*;
                vec3(win.input.keys_dir(A, D),
                     win.input.keys_dir(S, W),
                     win.input.keys_dir(Z, Space))
            };

            player.update(win.input.scaled_mouse_delta(), thrust, &bsp_geom, 1.0 / 60.0);
            if !fly_cam {
                camera = player.eye_pose();
            } else {
                use glium::glutin::VirtualKeyCode::*;
                let fly_thrust = vec3(-win.input.keys_dir(A, D), win.input.keys_dir(E, Q), win.input.keys_dir(W, S));
                let dmouse = win.input.scaled_mouse_delta();
                camera.position += camera.orientation * fly_thrust;
                camera.orientation = (camera.orientation *
                    Quat::shortest_arc(vec3(0.0, 0.0, 1.0), vec3(dmouse.x, dmouse.y, 1.0).must_norm())).must_norm();
            }
        }

        win.view = camera.to_mat4().inverse().unwrap();
        for m in &scene_meshes {
            win.draw_lit_mesh(M4x4::identity(), m)?;
        }
        let input_size = win.input.size;
        let mouse_pos = win.input.mouse.pos;
        let mdelta = win.input.scaled_mouse_delta();
        let framerate = ui.framerate();
        ui.window(im_str!("test"))
            .position((20.0, 20.0), imgui::ImGuiCond::Appearing)
            .build(|| {
                ui.text(im_str!("fps: {:.3}", framerate));
                ui.text(im_str!("size: {:?}", input_size));
                ui.separator();
                ui.text(im_str!("pos: {}", player.pose.position));
                ui.text(im_str!("rot: {}", player.pose.orientation));
                ui.separator();
                ui.text(im_str!("mouse: {}", mouse_pos));
                ui.text(im_str!("mdelta: {}", mdelta));
            });
        win.end_frame_and_ui(&mut gui_renderer, ui)?;
    }

    Ok(())
}



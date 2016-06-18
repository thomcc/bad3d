#![allow(dead_code)]

#[macro_use]
extern crate glium;

extern crate rand;

#[macro_use]
extern crate more_asserts;

#[macro_use]
mod util;

mod math;
mod hull;

mod support;
mod gjk;
mod phys;

mod wingmesh;

mod bsp;

use bsp::{BspNode, Face};
use wingmesh::WingMesh;
use math::*;
use math::pose::Pose;
use glium::{DisplayBuild, Surface};
use std::collections::{HashSet};
use std::cell::{RefCell};
use std::rc::Rc;
use std::f32;
use glium::backend::glutin_backend::GlutinFacade;

// flat phong
static VERT_SRC: &'static str = include_str!("../shaders/phong-vs.glsl");
static FRAG_SRC: &'static str = include_str!("../shaders/phong-fs.glsl");

// solid color
static SOLID_VERT_SRC: &'static str = include_str!("../shaders/solid-vs.glsl");
static SOLID_FRAG_SRC: &'static str = include_str!("../shaders/solid-fs.glsl");

impl rand::Rand for V3 {
    fn rand<R: rand::Rng>(rng: &mut R) -> V3 {
        let x = rng.next_f32();
        let y = rng.next_f32();
        let z = rng.next_f32();
        vec3(x, y, z)
    }
}

fn unpack_arrays<'a>(arrays: &'a [[u16; 3]]) -> &'a [u16] {
    unsafe {
        std::slice::from_raw_parts(arrays.as_ptr() as *const u16,
                                   arrays.len() * 3)
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
struct Vertex {
    position: [f32; 3]
}

implement_vertex!(Vertex, position);


fn vertex_slice<'a>(v3s: &'a [V3]) -> &'a [Vertex] {
    unsafe {
        std::slice::from_raw_parts(v3s.as_ptr() as *const Vertex, v3s.len())
    }
}

fn random_point_cloud(size: usize) -> (Vec<V3>, Vec<[u16; 3]>) {
    assert!(size >= 4);
    let mut vs = vec![V3::zero(); size];
    loop {
        for item in vs.iter_mut() {
            *item = rand::random::<V3>() + V3::splat(-0.5);
        }
        if let Some(indices) = hull::compute_hull(&mut vs[..]) {
            return (vs, indices)
        } else {
            vs.push(V3::zero());
        }
    }
}

struct InputState {
    pub mouse_pos: (i32, i32),
    pub mouse_pos_prev: (i32, i32),
    pub mouse_vec: V3,
    pub mouse_vec_prev: V3,
    pub view_angle: f32,
    pub size: (u32, u32),
    pub mouse_down: bool,
    pub keys_down: HashSet<glium::glutin::VirtualKeyCode>,
    pub key_changes: Vec<(glium::glutin::VirtualKeyCode, bool)>
}

impl InputState {
    fn new(w: u32, h: u32, view_angle: f32) -> InputState {
        InputState {
            mouse_pos: (0, 0),
            mouse_pos_prev: (0, 0),
            mouse_vec: V3::zero(),
            mouse_vec_prev: V3::zero(),
            view_angle: view_angle,
            size: (w, h),
            mouse_down: false,
            keys_down: HashSet::new(),
            key_changes: Vec::new(),
        }
    }

    fn get_projection_matrix(&self, near: f32, far: f32) -> M4x4 {
        M4x4::perspective(self.view_angle.to_radians(), self.size.0 as f32 / self.size.1 as f32, near, far)
    }

    pub fn mouse_delta(&self) -> V2 {
        let now = vec2(self.mouse_pos.0 as f32, self.mouse_pos.1 as f32);
        let prev = vec2(self.mouse_pos_prev.0 as f32, self.mouse_pos_prev.1 as f32);
        (now - prev)
    }

    #[inline]
    pub fn mouse_pos_v(&self) -> V2 {
        vec2(self.mouse_pos.0 as f32, self.mouse_pos.1 as f32)
    }

    #[inline]
    pub fn mouse_prev_v(&self) -> V2 {
        vec2(self.mouse_pos_prev.0 as f32, self.mouse_pos_prev.1 as f32)
    }

    #[inline]
    pub fn dims(&self) -> V2 {
        vec2(self.size.0 as f32, self.size.1 as f32)
    }

    #[inline]
    pub fn scaled_mouse_delta(&self) -> V2 {
        (self.mouse_pos_v() - self.mouse_prev_v()) / self.dims()*0.5
    }

    #[inline]
    pub fn keys_dir(&self, k1: glium::glutin::VirtualKeyCode, k2: glium::glutin::VirtualKeyCode) -> f32 {
        let v1 = if self.keys_down.contains(&k1) { 1 } else { 0 };
        let v2 = if self.keys_down.contains(&k2) { 1 } else { 0 };
        (v2 - v1) as f32
    }

    fn update(&mut self, display: &glium::backend::glutin_backend::GlutinFacade) -> bool {
        use glium::glutin::{Event, ElementState, MouseButton};
        let mouse_pos = self.mouse_pos;
        let mouse_vec = self.mouse_vec;
        self.key_changes.clear();
        self.mouse_pos_prev = mouse_pos;
        self.mouse_vec_prev = mouse_vec;
        for ev in display.poll_events() {
            match ev {
                Event::Closed => return false,
                Event::Resized(w, h) => {
                    self.size = (w, h);
                },
                Event::Focused(true) => {
                    self.keys_down.clear()
                },
                Event::KeyboardInput(pressed, _, Some(vk)) => {
                    let was_pressed = match pressed {
                        ElementState::Pressed => {
                            self.keys_down.insert(vk);
                            true
                        },
                        ElementState::Released => {
                            self.keys_down.remove(&vk);
                            false
                        }
                    };
                    self.key_changes.push((vk, was_pressed));
                },
                Event::MouseMoved(x, y) => {
                    self.mouse_pos = (x, y);
                },
                Event::MouseInput(state, MouseButton::Left) => {
                    self.mouse_down = state == ElementState::Pressed;
                }
                _ => (),
            }
        }
        self.mouse_vec = {
            let spread = (self.view_angle.to_radians() * 0.5).to_radians().tan();
            let (w, h) = (self.size.0 as f32, self.size.1 as f32);
            let (mx, my) = (self.mouse_pos.0 as f32, self.mouse_pos.1 as f32);
            let hh = h * 0.5;
            let y = spread * (h - my - hh) / hh;
            let x = spread * (mx - w * 0.5) / hh;
            vec3(x, y, -1.0).normalize().unwrap()
        };
        true
    }
}

pub fn glmat(inp: M4x4) -> [[f32; 4]; 4] { inp.into() }

struct DemoWindow {
    pub display: GlutinFacade,
    pub input: InputState,
    pub view: M4x4,
    pub lit_shader: glium::Program,
    pub solid_shader: glium::Program,
    pub clear_color: V4,
    pub light_pos: [f32; 3],
    pub targ: Option<glium::Frame>,
    pub near_far: (f32, f32),
}

impl DemoWindow {
    pub fn new() -> DemoWindow {
        let display = glium::glutin::WindowBuilder::new()
                            .with_depth_buffer(24)
                            .with_vsync()
                            .build_glium()
                            .unwrap();
        let input_state = {
            let (win_w, win_h) = display.get_window().unwrap()
                .get_inner_size_pixels().unwrap();
            InputState::new(win_w, win_h, 75.0)
        };

        let phong_program = glium::Program::from_source(&display,
            include_str!("../shaders/phong-vs.glsl"),
            include_str!("../shaders/phong-fs.glsl"), None).unwrap();

        let solid_program = glium::Program::from_source(&display,
            include_str!("../shaders/solid-vs.glsl"),
            include_str!("../shaders/solid-fs.glsl"), None).unwrap();

        DemoWindow {
            display: display,
            input: input_state,
            view: M4x4::identity(),
            lit_shader: phong_program,
            solid_shader: solid_program,
            clear_color: vec4(0.5, 0.6, 1.0, 1.0),
            light_pos: [1.4, 0.4, 0.7f32],
            near_far: (0.01, 500.0),
            targ: None,
        }
    }

    pub fn up(&mut self) -> bool {
        assert!(self.targ.is_none());
        if !self.input.update(&self.display) {
            false
        } else {
            self.targ = Some(self.display.draw());
            self.targ.as_mut().unwrap()
                .clear_color_and_depth(self.clear_color.into(), 1.0);
            true
        }
    }

    pub fn draw_lit_mesh(&mut self, mat: M4x4, mesh: &DemoMesh) {
        self.targ.as_mut().unwrap().draw((&mesh.vbo,), &mesh.ibo, &self.lit_shader,
                &uniform! {
                    model: glmat(mat),
                    u_color: <[f32; 4]>::from(mesh.color),
                    view: glmat(self.view),
                    perspective: glmat(self.input.get_projection_matrix(
                        self.near_far.0, self.near_far.1)),
                    u_light: self.light_pos,
                },
                &glium::DrawParameters {
                    blend: glium::Blend::alpha_blending(),
                    depth: glium::Depth {
                        test: glium::draw_parameters::DepthTest::IfLess,
                        write: true,
                        .. Default::default()
                    },
                    .. Default::default()
                }).unwrap();
    }

    pub fn draw_lit_tris(&mut self, mat: M4x4, color: V4, verts: &[V3], maybe_tris: Option<&[[u16; 3]]>) {
        let vbo = glium::VertexBuffer::new(&self.display, vertex_slice(verts)).unwrap();
        let params = glium::DrawParameters {
            blend: glium::Blend::alpha_blending(),
            depth: glium::Depth {
                test: glium::draw_parameters::DepthTest::IfLess,
                write: true,
                .. Default::default()
            },
            .. Default::default()
        };
        let uniforms = uniform! {
            model: glmat(mat),
            u_color: <[f32; 4]>::from(color),
            view: glmat(self.view),
            perspective: glmat(self.input.get_projection_matrix(self.near_far.0, self.near_far.1)),
            u_light: self.light_pos,
        };
        if let Some(tris) = maybe_tris {
            let ibo = glium::IndexBuffer::new(&self.display,
                glium::index::PrimitiveType::TrianglesList, unpack_arrays(tris)).unwrap();
            self.targ.as_mut().unwrap().draw((&vbo,), &ibo, &self.lit_shader, &uniforms, &params).unwrap();
        } else {
            let ibo = glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList);
            self.targ.as_mut().unwrap().draw((&vbo,), &ibo, &self.lit_shader, &uniforms, &params).unwrap();
        }
    }

    pub fn draw_solid(&mut self, mat: M4x4, color: V4, verts: &[V3], prim_type: glium::index::PrimitiveType) {
        let vbo = glium::VertexBuffer::new(&self.display, vertex_slice(verts)).unwrap();
        let ibo = glium::index::NoIndices(prim_type);

        self.targ.as_mut().unwrap().draw((&vbo,), &ibo, &self.solid_shader,
                &uniform! {
                    model: glmat(mat),
                    u_color: <[f32; 4]>::from(color),
                    view: glmat(self.view),
                    perspective: glmat(self.input.get_projection_matrix(
                        self.near_far.0, self.near_far.1)),
                },
                &glium::DrawParameters {
                    point_size: Some(5.0),
                    blend: glium::Blend::alpha_blending(),
                    .. Default::default()
                }).unwrap();
    }

    pub fn wm_draw_wireframe(&mut self, mat: M4x4, color: V4, wm: &WingMesh) {
        let mut verts = Vec::with_capacity(wm.edges.len()*2);
        for e in &wm.edges {
            verts.push(wm.verts[e.vert_idx()]);
            verts.push(wm.verts[wm.edges[e.next_idx()].vert_idx()]);
        }
        self.draw_solid(mat, color, &verts[..], glium::index::PrimitiveType::LinesList);
    }

    pub fn draw_face(&mut self, mat: M4x4, color: V4, f: &Face) {
        self.draw_lit_tris(mat, color, &f.vertex, Some(&f.gen_tris()[..]));
    }

    pub fn new_mesh(&self, verts: &[V3], tris: &[[u16; 3]], color: V4) -> DemoMesh {
        DemoMesh::new(&self.display, verts, tris, color)
    }

    pub fn end_frame(&mut self) {
        self.targ.take().unwrap().finish().unwrap();
    }
}


fn run_hull_test() {
    let display = glium::glutin::WindowBuilder::new()
                        .with_depth_buffer(24)
                        .with_vsync()
                        .build_glium()
                        .unwrap();
    let mut input_state = {
        let (win_w, win_h) = display.get_window().unwrap()
            .get_inner_size_pixels().unwrap();
        InputState::new(win_w, win_h, 75.0)
    };

    let (vertices, triangles) = random_point_cloud(64);

    let positions = glium::VertexBuffer::new(&display,
        vertex_slice(&vertices[..])).unwrap();

    let indices = glium::IndexBuffer::new(&display,
        glium::index::PrimitiveType::TrianglesList,
        unpack_arrays(&triangles[..])).unwrap();

    let program = glium::Program::from_source(&display,
        VERT_SRC, FRAG_SRC, None).unwrap();

    let mut model_orientation = Quat::identity();
    while input_state.update(&display) {
        if input_state.mouse_down {
            let q = model_orientation;
            model_orientation = Quat::virtual_track_ball(
                vec3(0.0, 0.0, 2.0),
                vec3(0.0, 0.0, 0.0),
                input_state.mouse_vec_prev,
                input_state.mouse_vec) * q;
        }
        let mut target = display.draw();
        target.clear_color_and_depth((0.5, 0.6, 1.0, 1.0), 1.0);

        let model_matrix = pose::Pose::from_rotation(model_orientation).to_mat4();

        let view_matrix = M4x4::look_at(vec3(0.0, 0.0, 2.0),
                                        vec3(0.0, 0.0, 0.0),
                                        vec3(0.0, 1.0, 0.0));


        let proj_matrix = input_state.get_projection_matrix(0.01, 50.0);
        let params = glium::DrawParameters {
            blend: glium::Blend::alpha_blending(),
            depth: glium::Depth {
                test: glium::draw_parameters::DepthTest::IfLess,
                write: true,
                .. Default::default()
            },
            .. Default::default()
        };

        target.draw((&positions,), &indices, &program,
            &uniform! {
                model: <[[f32; 4]; 4]>::from(model_matrix),
                view: <[[f32; 4]; 4]>::from(view_matrix),
                perspective: <[[f32; 4]; 4]>::from(proj_matrix),
                u_light: [1.4, 0.4, 0.7f32],
                u_color: [0.6, 0.6, 0.6f32, 1.0f32],
            },
            &params).unwrap();

        target.finish().unwrap();
    }
}

struct DemoMesh {
    verts: Vec<V3>,
    tris: Vec<[u16; 3]>,
    color: V4,
    ibo: glium::IndexBuffer<u16>,
    vbo: glium::VertexBuffer<Vertex>,
}

impl DemoMesh {
    pub fn new(display: &GlutinFacade, verts: &[V3], tris: &[[u16; 3]], color: V4) -> DemoMesh {
        DemoMesh {
            color: color,
            verts: verts.iter().map(|x| *x).collect::<Vec<V3>>(),
            tris: tris.iter().map(|x| *x).collect::<Vec<[u16; 3]>>(),
            vbo: glium::VertexBuffer::new(display, vertex_slice(&verts[..])).unwrap(),
            ibo: glium::IndexBuffer::new(display, glium::index::PrimitiveType::TrianglesList,
                unpack_arrays(&tris[..])).unwrap()
        }
    }
}

struct DemoObject {
    pub body: phys::RigidBodyRef,
    pub meshes: Vec<Box<DemoMesh>>
}

fn create_box_verts(min: V3, max: V3) -> Vec<V3> {
    let mut v = Vec::new();
    for z in &[min.z, max.z] {
        for y in &[min.y, max.y] {
            for x in &[min.x, max.x] {
                v.push(vec3(*x, *y, *z))
            }
        }
    }
    v
}

fn rand_color() -> V4 {
    let mut c = vec4(rand::random::<f32>(), rand::random::<f32>(), rand::random::<f32>(), 1.0);
    c[rand::random::<usize>() % 3] = rand::random::<f32>()*0.5;
    c
}

fn create_box(display: &GlutinFacade, r: V3, com: V3) -> DemoObject {
    let mut v = Vec::new();
    for x in &[-1.0f32, 1.0] {
        for y in &[-1.0f32, 1.0] {
            for z in &[-1.0f32, 1.0] {
                v.push(vec3(*x, *y, *z) * r)
            }
        }
    }

    let tris = hull::compute_hull(&mut v[..]).unwrap();
    v.truncate((tris.iter().flat_map(|n| n.iter()).max().unwrap() + 1) as usize);

    println!(r#"
        made box with dim: {:?},
        volume: {},
        com: {:?}
    "#, r, geom::volume(&v[..], &tris[..]), geom::center_of_mass(&v[..], &tris[..]));
    let body = Rc::new(RefCell::new(phys::RigidBody::new(vec![phys::Shape::new(v, tris)], com, 1.0)));
    let meshes = body.borrow().shapes.iter().map(|s|
        Box::new(DemoMesh::new(&display, &s.vertices[..], &s.tris[..], rand_color())))
        .collect::<Vec<_>>();
    DemoObject {
        body: body,
        meshes: meshes,
    }
}

fn create_cube_shape(r: V3) -> phys::Shape {
    let mut v = Vec::new();
    for x in &[-1.0f32, 1.0] {
        for y in &[-1.0f32, 1.0] {
            for z in &[-1.0f32, 1.0] {
                v.push(vec3(*x, *y, *z) * r)
            }
        }
    }

    let tris = hull::compute_hull(&mut v[..]).unwrap();
    v.truncate((tris.iter().flat_map(|n| n.iter()).max().unwrap() + 1) as usize);
    phys::Shape::new(v, tris)
}

fn create_octa(display: &GlutinFacade, r: V3, com: V3) -> DemoObject {
    let mut v = Vec::new();
    for x in &[-r.x, r.x] { v.push(vec3(*x, 0.0, 0.0)) }
    for y in &[-r.y, r.y] { v.push(vec3(0.0, *y, 0.0)) }
    for z in &[-r.z, r.z] { v.push(vec3(0.0, 0.0, *z)) }

    let tris = hull::compute_hull(&mut v[..]).unwrap();
    v.truncate((tris.iter().flat_map(|n| n.iter()).max().unwrap() + 1) as usize);;
    let body = Rc::new(RefCell::new(phys::RigidBody::new(vec![phys::Shape::new(v, tris)], com, 1.0)));
    body.borrow_mut().pose.orientation = quat(0.1, 0.2, 0.2, 0.8).must_norm();
    let meshes = body.borrow().shapes.iter().map(|s|
        Box::new(DemoMesh::new(&display, &s.vertices[..], &s.tris[..], rand_color())))
        .collect::<Vec<_>>();
    DemoObject {
        body: body,
        meshes: meshes,
    }
}


fn run_joint_test() {

    let body_sizes = [
        vec3(0.25, 0.50, 0.10),   // torso
        vec3(0.25, 0.05, 0.05),   // limb upper bones
        vec3(0.25, 0.05, 0.05),
        vec3(0.25, 0.05, 0.05),
        vec3(0.25, 0.05, 0.05),
        vec3(0.25, 0.05, 0.05),
        vec3(0.25, 0.05, 0.05),
        vec3(0.05, 0.05, 0.25),   // limb lower bones
        vec3(0.05, 0.05, 0.25),
        vec3(0.05, 0.05, 0.25),
        vec3(0.05, 0.05, 0.25),
        vec3(0.05, 0.05, 0.25),
        vec3(0.05, 0.05, 0.25),
    ];

    let joints = [
        (0,  1,  0.25_f32, vec3( 0.25, -0.5, 0.0), vec3(-0.25, 0.0, 0.0)), // upper limbs to torso
        (0,  2, -0.25_f32, vec3( 0.25,  0.0, 0.0), vec3(-0.25, 0.0, 0.0)),
        (0,  3,  0.25_f32, vec3( 0.25,  0.5, 0.0), vec3(-0.25, 0.0, 0.0)),
        (0,  4,  0.25_f32, vec3(-0.25, -0.5, 0.0), vec3( 0.25, 0.0, 0.0)),
        (0,  5, -0.25_f32, vec3(-0.25,  0.0, 0.0), vec3( 0.25, 0.0, 0.0)),
        (0,  6,  0.25_f32, vec3(-0.25,  0.5, 0.0), vec3( 0.25, 0.0, 0.0)),
        (1,  7,  0.0_f32, vec3( 0.25,  0.0, 0.0), vec3( 0.0,  0.0, 0.25)), // lower limb to upper limb
        (2,  8,  0.0_f32, vec3( 0.25,  0.0, 0.0), vec3( 0.0,  0.0, 0.25)),
        (3,  9,  0.0_f32, vec3( 0.25,  0.0, 0.0), vec3( 0.0,  0.0, 0.25)),
        (4, 10,  0.0_f32, vec3(-0.25,  0.0, 0.0), vec3( 0.0,  0.0, 0.25)),
        (5, 11,  0.0_f32, vec3(-0.25,  0.0, 0.0), vec3( 0.0,  0.0, 0.25)),
        (6, 12,  0.0_f32, vec3(-0.25,  0.0, 0.0), vec3( 0.0,  0.0, 0.25))
    ];

    let display = glium::glutin::WindowBuilder::new()
                        .with_depth_buffer(24)
                        .with_vsync()
                        .build_glium()
                        .unwrap();
    let mut input_state = {
        let (win_w, win_h) = display.get_window().unwrap()
            .get_inner_size_pixels().unwrap();
        InputState::new(win_w, win_h, 75.0)
    };

    let mut ground_verts = create_box_verts(vec3(-5.0, -5.0, -3.0), vec3(5.0, 5.0, -2.0));
    let ground_tris = hull::compute_hull(&mut ground_verts[..]).unwrap();
    ground_verts.truncate((ground_tris.iter().flat_map(|n| n.iter()).max().unwrap() + 1) as usize);

    let ground_mesh = Box::new(DemoMesh::new(&display,
        &ground_verts[..],
        &ground_tris[..],
        vec4(0.25, 0.75, 0.25, 1.0)));

    let mut demo_objects: Vec<DemoObject> = Vec::new();

    for size in body_sizes.iter() {
        demo_objects.push(create_box(&display, *size, V3::zero()));
        // demo_objects.last_mut().unwrap().body.borrow_mut().pose.orientation = Quat::from_axis_angle(rand::random::<V3>().must_norm(), 1f32.to_radians());
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
        let pos = body0.pose*joint.3 - body1.pose.orientation*joint.4;
        body1.pose.position = pos;
        body1.start_pose.position = pos;
    }

    demo_objects.push(create_box(&display, vec3(2.0, 0.1, 0.1), vec3(0.0, 0.0, -0.5)));
    demo_objects.push(create_box(&display, vec3(2.0, 0.4, 0.1), vec3(0.0, 1.0, -0.5)));

    let program = glium::Program::from_source(&display,
        VERT_SRC, FRAG_SRC, None).unwrap();

    let torque_limit = 38.0;
    let mut time = 0.0;

    // let camera = pose::Pose::new(vec3(0.0, -8.0, 0.0), quat(0.9, 0.0, 0.0, 1.0).must_norm());
    let world_geom = [ground_verts];
    while input_state.update(&display) {
        let target = RefCell::new(display.draw());


        target.borrow_mut().clear_color_and_depth((0.5, 0.6, 1.0, 1.0), 1.0);

        let proj_matrix = input_state.get_projection_matrix(0.01, 500.0);
        let dt = 1.0 / 60.0f32;
        time += 0.06f32;

        let mut cs = phys::ConstraintSet::new(dt);

        for joint in joints.iter() {
            cs.nail(Some(demo_objects[joint.0].body.clone()), joint.3,
                    Some(demo_objects[joint.1].body.clone()), joint.4);
            cs.powered_angle(Some(demo_objects[joint.0].body.clone()),
                             Some(demo_objects[joint.1].body.clone()),
                             quat(0.0, joint.2*time.cos(), joint.2*time.sin(),
                                  (1.0 - joint.2*joint.2).sqrt()),
                             torque_limit);
        }

        let mut bodies = demo_objects.iter()
            .map(|item| item.body.clone())
            .collect::<Vec<phys::RigidBodyRef>>();

        for body in bodies[body_sizes.len()..].iter_mut() {
            cs.under_plane(body.clone(), geom::Plane::from_norm_and_point(phys::GRAVITY.must_norm(), vec3(5.0, 5.0, -10.0)), None);
        }

        phys::update_physics(&mut bodies[..], &mut cs, &world_geom[..], dt);


        let params = glium::DrawParameters {
            blend: glium::Blend::alpha_blending(),
            depth: glium::Depth {
                test: glium::draw_parameters::DepthTest::IfLess,
                write: true,
                .. Default::default()
            },
            .. Default::default()
        };
        {

            let light = [5.0, 1.2, 1.0f32];

            let camera = M4x4::look_at(vec3(0.0, -8.0, 0.0),
                                       vec3(0.0, 0.0, 0.0),
                                       vec3(0.0, 0.0, 1.0));
            let cam_info = <[[f32; 4]; 4]>::from(camera);
            let proj = <[[f32; 4]; 4]>::from(proj_matrix);

            target.borrow_mut().draw((&ground_mesh.vbo,), &ground_mesh.ibo, &program,
                &uniform! {
                    model: <[[f32; 4]; 4]>::from(M4x4::identity()),
                    u_color: <[f32; 4]>::from(ground_mesh.color),
                    view: cam_info,
                    perspective: proj,
                    u_light: light,
                },
                &params).unwrap();

            for obj in demo_objects.iter() {
                let pose = obj.body.borrow().pose;
                let model_mat = <[[f32; 4]; 4]>::from(pose.to_mat4());
                for mesh in obj.meshes.iter() {
                    target.borrow_mut().draw((&mesh.vbo,), &mesh.ibo, &program,
                        &uniform! {
                            model: model_mat,
                            u_color: <[f32; 4]>::from(mesh.color),
                            view: cam_info,
                            perspective: proj,
                            u_light: light,
                        },
                        &params).unwrap();
                }
            }
        }

        target.into_inner().finish().unwrap();

        let need_reset = bodies[0..body_sizes.len()].iter().find(|b| b.borrow().pose.position.length() > 25.0).is_some();
        if need_reset {
            for body in bodies[0..body_sizes.len()].iter_mut() {
                let momentum = body.borrow().linear_momentum;
                let start_pose = body.borrow_mut().start_pose;
                body.borrow_mut().linear_momentum = -momentum;
                body.borrow_mut().pose = start_pose;
            }
            time = 0.0;
        }
    }
}

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
            let pos = (rand::random::<V3>() - V3::splat(0.5)) * 1.0;
            for _ in 0..self.vert_count {
                self.all_verts.push(pos + rand::random::<V3>() - V3::splat(0.5));
            }
        }
        let com = self.all_verts.iter().fold(V3::zero(), |a, b| a + *b)
                  / (self.all_verts.len() as f32);
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
            let a_tris = hull::compute_hull(&mut self.a_verts[..]);
            let b_tris = hull::compute_hull(&mut self.b_verts[..]);
            if let (Some(a), Some(b)) = (a_tris, b_tris) {
                self.a_tris = a;
                self.a_verts.truncate((self.a_tris.iter().flat_map(|n| n.iter()).max().unwrap() + 1) as usize);;
                self.b_tris = b;
                self.b_verts.truncate((self.b_tris.iter().flat_map(|n| n.iter()).max().unwrap() + 1) as usize);;
                return
            } else {
                self.reinit();
            }
        }
    }
}

fn run_gjk_test() {
    let display = glium::glutin::WindowBuilder::new()
                        .with_depth_buffer(24)
                        .with_vsync()
                        .build_glium()
                        .unwrap();

    let mut test_state = GjkTestState::new();
    let mut show_mink = false;

    let mut input_state = {
        let (win_w, win_h) = display.get_window().unwrap()
            .get_inner_size_pixels().unwrap();
        InputState::new(win_w, win_h, 75.0)
    };

    let program = glium::Program::from_source(&display,
        VERT_SRC, FRAG_SRC, None).unwrap();

    let solid_program = glium::Program::from_source(&display,
        SOLID_VERT_SRC, SOLID_FRAG_SRC, None).unwrap();

    let params = glium::DrawParameters {
        blend: glium::Blend::alpha_blending(),
        depth: glium::Depth {
            test: glium::draw_parameters::DepthTest::IfLess,
            write: true,
            .. Default::default()
        },
        .. Default::default()
    };

    use glium::index::PrimitiveType::*;

    let mut model_orientation = Quat::identity();
    let mut print_hit_info = true;
    while input_state.update(&display) {
        let proj_matrix = input_state.get_projection_matrix(0.01, 500.0);

        for &(key, down) in input_state.key_changes.iter() {
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

        let view_matrix = M4x4::look_at(vec3(0.0, 0.0, 2.0),
                                        vec3(0.0, 0.0, 0.0),
                                        vec3(0.0, 1.0, 0.0));

        if input_state.mouse_down {
            let q = model_orientation;
            model_orientation = Quat::virtual_track_ball(
                vec3(0.0, 0.0, 2.0),
                vec3(0.0, 0.0, 0.0),
                input_state.mouse_vec_prev,
                input_state.mouse_vec) * q;
        }
        let scene_matrix = pose::Pose::from_rotation(model_orientation).to_mat4();

        let target = RefCell::new(display.draw());
        {
            let draw_model_no_tris = |verts: &[V3], color, prim_type, solid| {
                let vbo = glium::VertexBuffer::new(&display,
                    vertex_slice(verts)).unwrap();

                let ibo = glium::index::NoIndices(prim_type);
                if solid {
                    target.borrow_mut().draw((&vbo,), &ibo, &solid_program,
                        &uniform! {
                            model: <[[f32; 4]; 4]>::from(scene_matrix),
                            view: <[[f32; 4]; 4]>::from(view_matrix),
                            perspective: <[[f32; 4]; 4]>::from(proj_matrix),
                            u_color: color,
                        },
                        &glium::DrawParameters {
                            point_size: Some(5.0),
                            depth: Default::default(),
                            .. params.clone()
                        }).unwrap();
                } else {
                    target.borrow_mut().draw((&vbo,), &ibo, &program,
                        &uniform! {
                            model: <[[f32; 4]; 4]>::from(scene_matrix),
                            view: <[[f32; 4]; 4]>::from(view_matrix),
                            perspective: <[[f32; 4]; 4]>::from(proj_matrix),
                            u_light: [1.4, 0.4, 0.7f32],
                            u_color: color,
                        },
                        &params).unwrap();
                }
            };

            let draw_model = |verts: &[V3], tris: &[[u16; 3]], color| {
                let vbo = glium::VertexBuffer::new(&display,
                    vertex_slice(verts)).unwrap();

                let ibo = glium::IndexBuffer::new(&display,
                    TrianglesList, unpack_arrays(tris)).unwrap();

                target.borrow_mut().draw((&vbo,), &ibo, &program,
                    &uniform! {
                        model: <[[f32; 4]; 4]>::from(scene_matrix),
                        view: <[[f32; 4]; 4]>::from(view_matrix),
                        perspective: <[[f32; 4]; 4]>::from(proj_matrix),
                        u_light: [1.4, 0.4, 0.7f32],
                        u_color: color,
                    },
                    &params).unwrap();
            };

            target.borrow_mut().clear_color_and_depth((0.1, 0.1, 0.2, 1.0), 1.0);
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
                print_hit_info = false;
            }


            if show_mink {
                let mut mink_vertices = Vec::new();
                for a in &test_state.a_verts {
                    for b in &test_state.b_verts {
                        mink_vertices.push(*a - *b);
                    }
                }
                let mink_tris = hull::compute_hull(&mut mink_vertices[..]).unwrap();

                draw_model(&mink_vertices[..], &mink_tris[..], [1.0f32, 0.5, 0.5, 0.8]);

                draw_model_no_tris(&[V3::zero()], [1.0f32, 1.0, 1.0, 1.0], Points, true);
                for i in 0..3 {
                    let mut v = V3::zero();
                    v[i] = 1.0;
                    draw_model_no_tris(&[-v, v], [v[0], v[1], v[2], 1.0], LinesList, true);
                }

                draw_model_no_tris(&[V3::zero(), hit.plane.normal*hit.separation], [1.0f32, 1.0, 1.0, 1.0], LinesList, true);

                {
                    let q = Quat::shortest_arc(vec3(0.0, 0.0, 1.0), hit.plane.normal);

                    let q0v0 = hit.plane.normal * hit.separation + q.x_dir();
                    let q0v1 = hit.plane.normal * hit.separation + q.y_dir();
                    let q0v2 = hit.plane.normal * hit.separation - q.x_dir();
                    let q0v3 = hit.plane.normal * hit.separation - q.y_dir();

                    let q1v0 = hit.plane.normal * hit.separation - q.y_dir();
                    let q1v1 = hit.plane.normal * hit.separation - q.x_dir();
                    let q1v2 = hit.plane.normal * hit.separation + q.y_dir();
                    let q1v3 = hit.plane.normal * hit.separation + q.x_dir();

                    let quads0 = [q0v0, q0v1, q0v2, q0v0, q0v2, q0v3];
                    let quads1 = [q1v0, q1v1, q1v2, q1v0, q1v2, q1v3];
                    let rc = if did_hit { 0.6f32 } else { 0.0f32 };
                    draw_model_no_tris(&quads0, [rc, 0.0, 1.0, 0.5], TrianglesList, false);
                    draw_model_no_tris(&quads1, [rc, 1.0, 0.0, 0.5], TrianglesList, false);
                }

            } else {


                draw_model(&test_state.a_verts[..], &test_state.a_tris[..], [1.0f32, 0.5, 0.5, 0.8]);
                draw_model(&test_state.b_verts[..], &test_state.b_tris[..], [0.5f32, 0.5, 1.0, 0.8]);

                let points = [hit.points.0, hit.points.1];

                draw_model_no_tris(&points, [1.0f32, 0.5, 0.5, 1.0], Points, true);
                draw_model_no_tris(&hit.simplex, [0.5f32, 0.0, 0.0, 1.0], Points, true);
                draw_model_no_tris(&points, [1.0f32, 0.0, 0.0, 1.0], LinesList, true);
                {
                    let q = Quat::shortest_arc(vec3(0.0, 0.0, 1.0), hit.plane.normal);

                    let q0v0 = hit.impact + q.x_dir();
                    let q0v1 = hit.impact + q.y_dir();
                    let q0v2 = hit.impact - q.x_dir();
                    let q0v3 = hit.impact - q.y_dir();

                    let q1v0 = hit.impact - q.y_dir();
                    let q1v1 = hit.impact - q.x_dir();
                    let q1v2 = hit.impact + q.y_dir();
                    let q1v3 = hit.impact + q.x_dir();

                    let quads0 = [q0v0, q0v1, q0v2, q0v0, q0v2, q0v3];
                    let quads1 = [q1v0, q1v1, q1v2, q1v0, q1v2, q1v3];
                    let rc = if did_hit { 0.6f32 } else { 0.0f32 };
                    draw_model_no_tris(&quads0, [rc, 0.0, 1.0, 0.5], TrianglesList, false);
                    draw_model_no_tris(&quads1, [rc, 1.0, 0.0, 0.5], TrianglesList, false);
                }

            }
        }

        target.into_inner().finish().unwrap();
    }
}

fn wm_shape(m: wingmesh::WingMesh) -> phys::Shape {
    phys::Shape::new(m.verts.clone(), m.generate_tris())
}

fn wm_object(display: &GlutinFacade, m: wingmesh::WingMesh, com: V3) -> DemoObject {
    let body = Rc::new(RefCell::new(phys::RigidBody::new(vec![wm_shape(m)], com, 1.0)));
    let c = rand_color();
    let meshes = body.borrow().shapes.iter().map(|s|
        Box::new(DemoMesh::new(display, &s.vertices[..], &s.tris[..], c))).collect::<Vec<_>>();
    DemoObject { body: body, meshes: meshes }
}

fn create_demo_blob(display: &GlutinFacade, com: V3) -> DemoObject {
    let (verts, tris) = random_point_cloud(15);
    let body = Rc::new(RefCell::new(phys::RigidBody::new(vec![phys::Shape::new(verts, tris)], com, 1.0)));
    let c = rand_color();
    let meshes = body.borrow().shapes.iter().map(|s|
        Box::new(DemoMesh::new(display, &s.vertices[..], &s.tris[..], c))).collect::<Vec<_>>();
    DemoObject { body: body, meshes: meshes }
}

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


fn run_phys_test() {
    let display = glium::glutin::WindowBuilder::new()
                        .with_depth_buffer(24)
                        .with_vsync()
                        .build_glium()
                        .unwrap();
    let mut input_state = {
        let (win_w, win_h) = display.get_window().unwrap()
            .get_inner_size_pixels().unwrap();
        InputState::new(win_w, win_h, 75.0)
    };

    let mut ground_verts = create_box_verts(vec3(-10.0, -10.0, -5.0), vec3(10.0, 10.0, -2.0));
    let ground_tris = hull::compute_hull(&mut ground_verts[..]).unwrap();
    ground_verts.truncate((ground_tris.iter().flat_map(|n| n.iter()).max().unwrap() + 1) as usize);

    let ground_mesh = Box::new(DemoMesh::new(&display,
        &ground_verts[..],
        &ground_tris[..],
        vec4(0.25, 0.75, 0.25, 1.0)));

    let mut demo_objects: Vec<DemoObject> = Vec::new();


    let jack_push_pos   = vec3(0.0, 0.0, 0.0);
    let jack_momentum   = vec3(4.0, -0.8, 5.0);
    let jack_push_pos_2 = vec3(0.0, 0.5, 0.0);
    let jack_momentum_2 = vec3(0.3, 0.4, 1.0);

    let seesaw_start    = vec3(0.0, -4.0, 0.25);
    demo_objects.push(wm_object(&display, wingmesh::WingMesh::new_cone(10, 0.5, 1.0), vec3(1.5, 0.0, 1.5)));
    // demo_objects.push(create_box(&display, V3::splat(1.0), vec3(1.5, 0.0, 1.5)));
    {
        let o = create_box(&display, V3::splat(1.0), vec3(-1.5, 0.0, 1.5));
        let r = quat(0.1, 0.01, 0.3, 1.0).must_norm();
        o.body.borrow_mut().pose.orientation = r;
        o.body.borrow_mut().start_pose.orientation = r;
        demo_objects.push(o);
    }

    let seesaw;
    {
        let o = create_box(&display, vec3(4.0, 0.5, 0.1), seesaw_start);
        seesaw = o.body.clone();
        demo_objects.push(o);
    }

    {
        let mut wm = wingmesh::WingMesh::new_cylinder(30, 1.0, 2.0);
        wm.translate(V3::splat(-1.0));
        wm.rotate(Quat::shortest_arc(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 0.0)));
        let l = wm_object(&display, wm, seesaw_start + vec3(3.5, 0.0, 50.0));
        l.body.borrow_mut().scale_mass(6.0);

        let l2 = create_box(&display, V3::splat(0.25), seesaw_start + vec3(3.0, 0.0, 0.4));
        l2.body.borrow_mut().scale_mass(0.75);

        let r = create_box(&display, V3::splat(0.5), seesaw_start + vec3(-2.5, 0.0, 5.0));
        r.body.borrow_mut().scale_mass(2.0);

        demo_objects.push(l);
        demo_objects.push(l2);
        demo_objects.push(r);
    }
    let jack = Rc::new(RefCell::new(phys::RigidBody::new(vec![
        create_cube_shape(vec3(1.0, 0.2, 0.2)),
        create_cube_shape(vec3(0.2, 1.0, 0.2)),
        create_cube_shape(vec3(0.2, 0.2, 1.0)),
    ], vec3(-5.5, 0.5, 7.5), 1.0)));

    {
        jack.borrow_mut().apply_impulse(jack_push_pos, jack_momentum);
        jack.borrow_mut().apply_impulse(jack_push_pos_2, jack_momentum_2);
        let c = rand_color();
        let meshes = jack.borrow().shapes.iter().map(|s|
            Box::new(DemoMesh::new(&display, &s.vertices[..], &s.tris[..], c))).collect::<Vec<_>>();
        demo_objects.push(DemoObject { body: jack.clone(), meshes: meshes });
    }

    {
        let mut z = 5.5;
        while z < 14.0 {
            demo_objects.push(create_box(&display, V3::splat(0.5), vec3(0.0, 0.0, z)));
            z += 3.0;
        }
    }

    {
        let mut z = 15.0;
        while z < 20.0 {
            demo_objects.push(create_octa(&display, V3::splat(0.5), vec3(0.0, 0.0, z)));
            z += 3.0;
        }
    }

    for i in 0..4 {
        let fi = i as f32;
        demo_objects.push(create_demo_blob(&display, vec3(3.0+fi*2.0, -3.0, 4.0+fi*3.0)));
    }

    demo_objects.push(create_box(&display, vec3(2.0, 0.1, 0.1), vec3(0.0, 0.0, -0.5)));
    demo_objects.push(create_box(&display, vec3(2.0, 0.4, 0.1), vec3(0.0, 1.0, -0.5)));
    {
        let mut wm = wingmesh::WingMesh::new_cone(30, 0.5, 2.0);
        wm.rotate(Quat::shortest_arc(vec3(0.0, 0.0, 1.0), vec3(0.0, -0.5, -0.5)));
        demo_objects.push(wm_object(&display, wm, vec3(-4.0, -4.0, 4.0)));
    }

    let program = glium::Program::from_source(&display,
        VERT_SRC, FRAG_SRC, None).unwrap();

    let world_geom = [ground_verts];

    let mut cam = DemoCamera::new(0.76, 0.0, vec3(0.2, -20.6, 6.5));

    let mut running = false;
    while input_state.update(&display) {
        for &(key, down) in input_state.key_changes.iter() {
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

        cam.handle_input(&input_state);

        let target = RefCell::new(display.draw());

        target.borrow_mut().clear_color_and_depth((0.5, 0.6, 1.0, 1.0), 1.0);

        let proj_matrix = input_state.get_projection_matrix(0.01, 100.0);
        let dt = 1.0 / 60.0f32;

        if running {
            let mut cs = phys::ConstraintSet::new(dt);

            cs.nail(None, seesaw_start, Some(seesaw.clone()), V3::zero());
            cs.range(None, Some(seesaw.clone()), Quat::identity(),
                vec3(0.0, -20.0, 0.0), vec3(0.0, 20.0, 0.0));

            let mut bodies = demo_objects.iter()
                .map(|item| item.body.clone())
                .collect::<Vec<phys::RigidBodyRef>>();


            phys::update_physics(&mut bodies[..], &mut cs, &world_geom[..], dt);
        }


        let params = glium::DrawParameters {
            blend: glium::Blend::alpha_blending(),
            depth: glium::Depth {
                test: glium::draw_parameters::DepthTest::IfLess,
                write: true,
                .. Default::default()
            },
            .. Default::default()
        };
        {

            let light = [0.0, 1.2, 1.0f32];

            let cam_info = <[[f32; 4]; 4]>::from(cam.matrix());
            let proj = <[[f32; 4]; 4]>::from(proj_matrix);

            target.borrow_mut().draw((&ground_mesh.vbo,), &ground_mesh.ibo, &program,
                &uniform! {
                    model: <[[f32; 4]; 4]>::from(M4x4::identity()),
                    u_color: <[f32; 4]>::from(ground_mesh.color),
                    view: cam_info,
                    perspective: proj,
                    u_light: light,
                },
                &params).unwrap();

            for obj in demo_objects.iter() {
                let pose = obj.body.borrow().pose;
                let model_mat = <[[f32; 4]; 4]>::from(pose.to_mat4());
                for mesh in obj.meshes.iter() {
                    target.borrow_mut().draw((&mesh.vbo,), &mesh.ibo, &program,
                        &uniform! {
                            model: model_mat,
                            u_color: <[f32; 4]>::from(mesh.color),
                            view: cam_info,
                            perspective: proj,
                            u_light: light,
                        },
                        &params).unwrap();
                }
            }
        }

        target.into_inner().finish().unwrap();

    }
}

/*
pub fn run_bsp_test() {
    let mut win = DemoWindow::new();
    let mut draw_mode = 0;
    let mut drag_mode = 0;
    let mut cam = Pose::from_rotation(Quat::from_axis_angle(vec3(1.0, 0.0, 0.0), 60.0.to_radians()));

    while win.up() {
        if win.input.key_changes.any(|(a,b)| b && a == glium::glutin::VirtualKeyCode::D) {
            draw_mode = (draw_mode+1)%3;
        }
        if win.input.mouse_down {
            match drag_mode {
                1 => {
                    cam.orientation *= Quat::virtual_track_ball(vec3(0.0, 0.0, 2.0), V3::zero(), win.input.mouse_vec_prev, win.input.mouse_vec).conj();
                },
                0 => {

                },
                n => {

                },
            }
        }


    float4 cameraorientation = normalize(float4(sinf(60.0f*3.14f/180.0f/2),0,0,cosf(60.0f*3.14f/180.0f/2)));
    float  cameradist = 5;
    float3 camerapos;
    int    dragmode = 0;   // for mouse motion.  1: orbit camera   2,3: move corresponding object
    float  hitdist  = 0;   // if we select an operand this is how far it is from the viewpoint, so we know how much to translate for subsequent lateral mouse movement
    float3 mousevec_prev;  // direction of mouse vector from previous frame



        win.end_frame();
    }
}


*/


// https://gfycat.com/ElaborateHarshHyrax
fn main() {
    // run_hull_test();
    // run_joint_test();
    // run_gjk_test();
    run_phys_test();
}

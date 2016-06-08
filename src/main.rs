#![allow(dead_code)]

#[macro_use]
extern crate glium;

extern crate rand;

#[macro_use]
mod util;

mod math;
mod hull;

mod support;
mod gjk;
mod phys;

use math::*;

use glium::{DisplayBuild, Surface};
use std::collections::{HashSet};
use std::cell::{RefCell};
use std::rc::Rc;

use glium::backend::glutin_backend::GlutinFacade;

// flat phong
static VERT_SRC: &'static str = r#"
    #version 140
    in vec3 position;
    out vec3 v_position;
    out vec3 v_viewpos;
    uniform mat4 perspective;
    uniform mat4 view;
    uniform mat4 model;

    void main() {
        mat4 modelview = view * model;
        vec4 mpos = modelview * vec4(position, 1.0);
        gl_Position = perspective * mpos;
        v_position = gl_Position.xyz / gl_Position.w;
        v_viewpos = -mpos.xyz;
    }
    "#;

static FRAG_SRC: &'static str = r#"
    #version 140

    in vec3 v_position;
    in vec3 v_viewpos;

    out vec4 color;
    uniform vec3 u_light;
    uniform vec4 u_color;

    const vec3 ambient_color = vec3(0.1, 0.1, 0.1);
    // const vec3 diffuse_color = vec3(0.5, 0.5, 0.5);
    const vec3 specular_color = vec3(1.0, 1.0, 1.0);

    void main() {
        vec3 normal = normalize(cross(dFdx(v_viewpos), dFdy(v_viewpos)));
        float diffuse = max(dot(normal, normalize(u_light)), 0.0);
        vec3 camera_dir = normalize(-v_position);
        vec3 half_direction = normalize(normalize(u_light) + camera_dir);
        float specular = pow(max(dot(half_direction, normal), 0.0), 16.0);
        color = vec4(ambient_color + diffuse*u_color.rgb + specular * specular_color, u_color.a);
    }"#;

// solid color
static SOLID_VERT_SRC: &'static str = r#"
    #version 140
    in vec3 position;
    uniform mat4 perspective, view, model;
    void main() {
        mat4 modelview = view * model;
        vec4 mpos = modelview * vec4(position, 1.0);
        gl_Position = perspective * mpos;
    }"#;
static SOLID_FRAG_SRC: &'static str = r#"
    #version 140
    out vec4 color;
    uniform vec4 u_color;
    void main() {
        color = u_color;
    }"#;


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
        M4x4::perspective(self.view_angle, self.size.0 as f32 / self.size.1 as f32, near, far)
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
            let spread = (self.view_angle * 0.5).to_radians().tan();
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


fn run_hull_test() {
    let display = glium::glutin::WindowBuilder::new()
                        .with_depth_buffer(24)
                        .build_glium()
                        .unwrap();
    let mut input_state = {
        let (win_w, win_h) = display.get_window().unwrap()
            .get_inner_size_pixels().unwrap();
        InputState::new(win_w, win_h, 90.0)
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
    v.truncate((tris.iter().flat_map(|n| n.iter()).max().unwrap() + 1) as usize);;
    DemoObject {
        body: Rc::new(RefCell::new(phys::RigidBody::new(
            vec![phys::Shape::new(v.clone(), tris.clone())], com, 1.0))),
        meshes: vec![ Box::new(DemoMesh::new(&display, &v[..], &tris[..], rand_color())) ]
    }
}

fn create_octa(display: &GlutinFacade, r: V3, com: V3) -> DemoObject {
    let mut v = Vec::new();
    for x in &[-r.x, r.x] { v.push(vec3(*x, 0.0, 0.0)) }
    for y in &[-r.y, r.y] { v.push(vec3(0.0, *y, 0.0)) }
    for z in &[-r.z, r.z] { v.push(vec3(0.0, 0.0, *z)) }

    let tris = hull::compute_hull(&mut v[..]).unwrap();
    v.truncate((tris.iter().flat_map(|n| n.iter()).max().unwrap() + 1) as usize);;
    let body = Rc::new(RefCell::new(phys::RigidBody::new(vec![phys::Shape::new(v, tris)], com, 1.0)));
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
        (0,  1,  0.2_f32, vec3( 0.25, -0.5, 0.0), vec3(-0.25, 0.0, 0.0)), // upper limbs to torso
        (0,  2, -0.2_f32, vec3( 0.25,  0.0, 0.0), vec3(-0.25, 0.0, 0.0)),
        (0,  3,  0.2_f32, vec3( 0.25,  0.5, 0.0), vec3(-0.25, 0.0, 0.0)),
        (0,  4,  0.2_f32, vec3(-0.25, -0.5, 0.0), vec3( 0.25, 0.0, 0.0)),
        (0,  5, -0.2_f32, vec3(-0.25,  0.0, 0.0), vec3( 0.25, 0.0, 0.0)),
        (0,  6,  0.2_f32, vec3(-0.25,  0.5, 0.0), vec3( 0.25, 0.0, 0.0)),
        (1,  7,  0.0_f32, vec3( 0.25,  0.0, 0.0), vec3( 0.0,  0.0, 0.25)), // lower limb to upper limb
        (2,  8,  0.0_f32, vec3( 0.25,  0.0, 0.0), vec3( 0.0,  0.0, 0.25)),
        (3,  9,  0.0_f32, vec3( 0.25,  0.0, 0.0), vec3( 0.0,  0.0, 0.25)),
        (4, 10,  0.0_f32, vec3(-0.25,  0.0, 0.0), vec3( 0.0,  0.0, 0.25)),
        (5, 11,  0.0_f32, vec3(-0.25,  0.0, 0.0), vec3( 0.0,  0.0, 0.25)),
        (6, 12,  0.0_f32, vec3(-0.25,  0.0, 0.0), vec3( 0.0,  0.0, 0.25))
    ];

    let display = glium::glutin::WindowBuilder::new()
                        .with_depth_buffer(24)
                        .build_glium()
                        .unwrap();
    let mut input_state = {
        let (win_w, win_h) = display.get_window().unwrap()
            .get_inner_size_pixels().unwrap();
        InputState::new(win_w, win_h, 90.0)
    };

    let mut ground_verts = create_box_verts(vec3(-5.0, -5.0, -2.0), vec3(5.0, 5.0, -1.0));
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
        let b1p = body1.pose;
        body1.pose.position = body0.pose*joint.3 - b1p.orientation*joint.4;
    }

    let program = glium::Program::from_source(&display,
        VERT_SRC, FRAG_SRC, None).unwrap();
    let torque_limit = 38.0;
    let mut time = 0.0;

    // let camera = pose::Pose::new(vec3(0.0, -8.0, 0.0), quat(0.9, 0.0, 0.0, 1.0).must_norm());
    let world_geom = [ground_verts];
    while input_state.update(&display) {
        let mut target = display.draw();
        target.clear_color_and_depth((0.5, 0.6, 1.0, 1.0), 1.0);

        let proj_matrix = input_state.get_projection_matrix(0.01, 50.0);
        let dt = 1.0 / 60.0f32;
        time += dt;

        let mut cs = phys::ConstraintSet::new(dt);

        for joint in joints.iter() {
            cs.nail(Some(demo_objects[joint.0].body.clone()), joint.3,
                    Some(demo_objects[joint.1].body.clone()), joint.4);
            cs.powered_angle(Some(demo_objects[joint.0].body.clone()),
                             Some(demo_objects[joint.1].body.clone()),
                             quat(0.0, joint.2*time.cos(), joint.2*time.sin(),
                                  (1.0 - joint.2*joint.2).sqrt()).must_norm(),
                             torque_limit);
        }

        let mut bodies = demo_objects.iter()
            .map(|item| item.body.clone())
            .collect::<Vec<phys::RigidBodyRef>>();

        phys::update_physics(&mut bodies[..], &mut cs, &world_geom[..], dt);

        let params = glium::DrawParameters {
            depth: glium::Depth {
                test: glium::draw_parameters::DepthTest::IfLess,
                write: true,
                .. Default::default()
            },
            .. Default::default()
        };

        let light = [1.4, 0.4, 0.7f32];

        let camera = M4x4::look_at(vec3(0.0, -8.0, 0.0),
                                        vec3(0.0, 0.0, 0.0),
                                        vec3(0.0, 0.0, 1.0));
        let cam_info = <[[f32; 4]; 4]>::from(camera);
        let proj = <[[f32; 4]; 4]>::from(proj_matrix);

        target.draw((&ground_mesh.vbo,), &ground_mesh.ibo, &program,
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
                target.draw((&mesh.vbo,), &mesh.ibo, &program,
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

        target.finish().unwrap();
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
            // for v in self.all_verts[..].chunks(2) {
            //     self.a_verts.push(v[0]);
            //     self.b_verts.push(v[1]);
            // }
            for v in &self.all_verts[0..self.all_verts.len()/2] {
                self.a_verts.push(*v);
            }
            for v in &self.all_verts[self.all_verts.len()/2..self.all_verts.len()] {
                self.b_verts.push(*v);
            }
            let a_tris = hull::compute_hull(&mut self.a_verts[..]);
//            v.truncate((tris.iter().flat_map(|n| n.iter()).max().unwrap() + 1) as usize);;
            let b_tris = hull::compute_hull(&mut self.b_verts[..]);
//            v.truncate((tris.iter().flat_map(|n| n.iter()).max().unwrap() + 1) as usize);;
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
                        .build_glium()
                        .unwrap();

    let mut test_state = GjkTestState::new();
    let mut show_mink = false;

    let mut input_state = {
        let (win_w, win_h) = display.get_window().unwrap()
            .get_inner_size_pixels().unwrap();
        InputState::new(win_w, win_h, 90.0)
    };

    let program = glium::Program::from_source(&display,
        VERT_SRC, FRAG_SRC, None).unwrap();

    let solid_program = glium::Program::from_source(&display,
        SOLID_VERT_SRC, SOLID_FRAG_SRC, None).unwrap();

    // let mut camera = Pose::identity();
    // let mut proj_matrix = M4x4::identity();

    // let default_color = [0.6, 0.6, 0.6f32, 1.0f32];

    // let mut scene_matrix = M4x4::identity();

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
        let proj_matrix = input_state.get_projection_matrix(0.01, 50.0);

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



fn main() {
    // run_hull_test();
    run_joint_test();
    // run_gjk_test();
}

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

use math::*;
use std::mem;

use glium::{DisplayBuild, Surface};
use std::collections::{HashSet};


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

    const vec3 ambient_color = vec3(0.2, 0.2, 0.2);
    const vec3 diffuse_color = vec3(0.6, 0.6, 0.6);
    const vec3 specular_color = vec3(1.0, 1.0, 1.0);

    void main() {
        vec3 normal = normalize(cross(dFdx(v_viewpos), dFdy(v_viewpos)));
        float diffuse = max(dot(normal, normalize(u_light)), 0.0);
        vec3 camera_dir = normalize(-v_position);
        vec3 half_direction = normalize(normalize(u_light) + camera_dir);
        float specular = pow(max(dot(half_direction, normal), 0.0), 16.0);
        color = vec4(ambient_color + diffuse * diffuse_color + specular * specular_color, 1.0);
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
    pub on_key: Option<&'static Fn(glium::glutin::VirtualKeyCode, bool)>
}

impl InputState {
    fn new(w: u32, h: u32, view_angle: f32,
           key_func: Option<&'static Fn(glium::glutin::VirtualKeyCode, bool)>)
            -> InputState {
        InputState {
            mouse_pos: (0, 0),
            mouse_pos_prev: (0, 0),
            mouse_vec: V3::zero(),
            mouse_vec_prev: V3::zero(),
            view_angle: view_angle,
            size: (w, h),
            mouse_down: false,
            keys_down: HashSet::new(),
            on_key: key_func,
        }
    }

    fn get_projection_matrix(&self, near: f32, far: f32) -> M4x4 {
        M4x4::perspective(self.view_angle, self.size.0 as f32 / self.size.1 as f32, near, far)
    }

    fn update(&mut self, display: &glium::backend::glutin_backend::GlutinFacade) -> bool {
        use glium::glutin::{Event, ElementState, MouseButton};
        let mouse_pos = self.mouse_pos;
        let mouse_vec = self.mouse_vec;
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
                    if let Some(key_cb) = self.on_key {
                        key_cb(vk, was_pressed);
                    }
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
        InputState::new(win_w, win_h, 90.0, None)
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
        target.clear_color_and_depth((0.0, 0.0, 1.0, 1.0), 1.0);

        let model_matrix = pose::Pose::from_rotation(model_orientation).to_mat4();

        let view_matrix = M4x4::look_at(vec3(0.0, 0.0, 2.0),
                                        vec3(0.0, 0.0, 0.0),
                                        vec3(0.0, 1.0, 0.0));


        let proj_matrix = input_state.get_projection_matrix(0.01, 50.0);
        let params = glium::DrawParameters {
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
                u_light: [1.4, 0.4, 0.7f32]
            },
            &params).unwrap();

        target.finish().unwrap();
    }

}


fn main() {
    run_hull_test();
}

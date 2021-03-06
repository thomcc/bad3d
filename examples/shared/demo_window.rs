use crate::shared::{
    input::InputState,
    object::{unpack_arr3, vertex_slice, DemoMesh},
};
use failure::Error;

use imgui_glium_renderer::Renderer;
use std::cell::{Ref, RefCell, RefMut};
use std::rc::Rc;
use std::time::Instant;

use super::shader_lib::{ShaderConf, ShaderLib};
use bad3d::{bsp::Face, prelude::*};
use glium::{
    self,
    backend::Facade,
    glutin::{ContextBuilder, EventsLoop, WindowBuilder},
    Display, Surface,
};
use imgui::{self, Ui};
use std::fmt::Write;

// lit flat shader
// static PREAMBLE_SRC: &'static str = include_str!("shaders/common.glsl");

// static VERT_SRC: &'static str = include_str!("shaders/lit-vs.glsl");
// static FRAG_SRC: &'static str = include_str!("shaders/lit-fs.glsl");

// solid color
// static SOLID_VERT_SRC: &'static str = include_str!("shaders/solid-vs.glsl");
// static SOLID_FRAG_SRC: &'static str = include_str!("shaders/solid-fs.glsl");

//static TEX_VERT_SRC: &'static str = include_str!("shaders/tex-vs.glsl");
//static TEX_FRAG_SRC: &'static str = include_str!("shaders/tex-fs.glsl");

// pub struct LightsUniform {
//     pub array: [PointLight; 4],
//     pub light_count: i32,
//     pub lights_ubo: glium::uniforms::UniformBuffer<[PointLight; 4]>,
//     pub last_array: [PointLight; 4],
// }
pub struct DemoWindow {
    pub events: EventsLoop,
    pub display: Display,
    pub input: InputState,
    view: M4x4,
    pub camera: Pose,
    pub hemilight_intensity: f32,
    // pub lit_shader: glium::Program,
    // pub solid_shader: glium::Program,
    //    pub tex_shader: glium::Program,
    pub clear_color: V4,
    pub light_dir: [f32; 3],
    pub targ: Option<RefCell<glium::Frame>>,
    pub near_far: (f32, f32),
    // pub gui: Rc<RefCell<imgui::ImGui>>,
    // pub gui_renderer: RefCell<Renderer>,
    pub last_frame: Instant,
    pub last_frame_time: f32,
    pub grabbed_cursor: bool,
    pub fog_amount: f32,
    pub shaders: ShaderLib,
    pub frame_num: u64,
}

#[derive(Clone, Debug)]
pub struct DemoOptions<'a> {
    pub clear_color: V4,
    pub title: &'a str,
    pub window_size: (u32, u32),
    pub near_far: (f32, f32),
    pub light_dir: V3,
    pub fov: f32,
    pub camera: Pose,
    pub fog_amount: f32,
    pub hemilight_intensity: f32,
}

impl<'a> Default for DemoOptions<'a> {
    fn default() -> DemoOptions<'a> {
        DemoOptions {
            title: "Bad3d Demo",
            window_size: (1024, 768),
            clear_color: vec4(0.5, 0.6, 1.0, 1.0),
            light_dir: vec3(1.4, 0.4, 0.7),
            near_far: (0.01, 500.0),
            fov: 75.0,
            // view: M4x4::identity(),
            fog_amount: 0.0,
            camera: Pose::identity(),
            hemilight_intensity: 0.2,
        }
    }
}

pub fn init_testing() {
    if cfg!(debug_assertions) {
        std::env::set_var("RUST_BACKTRACE", "1");
        env_logger::init_from_env(env_logger::Env::default().filter_or("RUST_LOG", "trace"));
    } else {
        env_logger::init_from_env(env_logger::Env::default().filter_or("RUST_LOG", "warn"));
    }
}
impl DemoWindow {
    pub fn new(opts: DemoOptions<'_>, gui: Rc<RefCell<imgui::Context>>) -> Result<DemoWindow, Error> {
        init_testing();
        let context = ContextBuilder::new()
            .with_depth_buffer(24)
            .with_srgb(false)
            .with_vsync(true);

        let window = WindowBuilder::new()
            .with_title(opts.title)
            .with_min_dimensions(glium::glutin::dpi::LogicalSize {
                width: opts.window_size.0 as f64,
                height: opts.window_size.1 as f64,
            });

        let mut shaders = ShaderLib::new(ShaderConf {
            repo_relative_path: "examples/shared/shaders",
            override_env_var: None,
            preamble_file: Some("common.glsl"),
            fs_suffix: "-fs.glsl",
            vs_suffix: "-vs.glsl",
            fs_define: None,
            vs_define: None,
        })?;

        let events = EventsLoop::new();

        let display = Display::new(window, context, &events).unwrap();

        let mut input_state = InputState::new(display.get_framebuffer_dimensions(), opts.fov, gui);

        // let phong_program = compile_shader(&display, "lit", VERT_SRC, FRAG_SRC)?;
        // let solid_program = compile_shader(&display, "solid", SOLID_VERT_SRC, SOLID_FRAG_SRC)?;
        //        let tex_program = compile_shader(&display, "tex", TEX_VERT_SRC, TEX_FRAG_SRC)?;
        {
            let glw = display.gl_window();
            let mut gui = input_state.gui.borrow_mut();
            input_state
                .platform
                .attach_window(gui.io_mut(), glw.window(), imgui_winit_support::HiDpiMode::Default);
        }
        shaders.add(&display, &["lit", "solid"])?;
        Ok(DemoWindow {
            display,
            events,
            input: input_state,
            view: opts.camera.inverse().to_mat4(),
            shaders,
            camera: opts.camera,
            frame_num: 0,
            //            tex_shader: tex_program,
            clear_color: opts.clear_color,
            light_dir: opts.light_dir.into(),
            near_far: opts.near_far,
            targ: None,
            last_frame: Instant::now(),
            last_frame_time: 0.0,
            grabbed_cursor: false,
            fog_amount: opts.fog_amount,
            hemilight_intensity: opts.hemilight_intensity,
        })
    }
    #[allow(clippy::wrong_self_convention)]
    pub fn is_up(&mut self) -> bool {
        let now = Instant::now();
        let delta = now - self.last_frame;
        let delta_s = delta.as_secs() as f32 + delta.subsec_nanos() as f32 / 1_000_000_000.0;
        self.last_frame = now;
        self.last_frame_time = delta_s;
        self.frame_num = self.frame_num.wrapping_add(1);
        if self.frame_num % 60 == 0 {
            let _ = self.shaders.maybe_reload_shaders(&self.display);
        }
        assert!(self.targ.is_none());
        if !self.input.update(&mut self.events, &mut self.display, delta_s) {
            false
        } else {
            let mut targ = self.display.draw();
            targ.clear_color_and_depth(self.clear_color.into(), 1.0);
            self.targ = Some(RefCell::new(targ));
            true
        }
    }

    pub fn grab_cursor(&mut self) {
        // use glium::glutin::CursorState;
        let gl_window = self.display.gl_window();
        let window = gl_window.window();
        if !self.grabbed_cursor {
            self.grabbed_cursor = true;
            self.input.mouse_grabbed = true;
            window.hide_cursor(true); //.ok().expect("Could not grab mouse cursor");
        }
        // let dpi = gl_window.hidpi_factor();
        let dims = self.input.dims(); // / dpi;
        window
            .set_cursor_position(glium::glutin::dpi::LogicalPosition {
                x: (dims.x / 2.0).trunc() as f64,
                y: (dims.y / 2.0).trunc() as f64,
            })
            .expect("Could not set mouse cursor position");
    }

    #[inline]
    pub fn target(&self) -> Ref<'_, glium::Frame> {
        self.targ.as_ref().unwrap().borrow()
    }

    #[inline]
    pub fn target_mut(&self) -> RefMut<'_, glium::Frame> {
        self.targ.as_ref().unwrap().borrow_mut()
    }

    pub fn ungrab_cursor(&mut self) {
        // use glium::glutin::CursorState;
        self.grabbed_cursor = false;
        self.input.mouse_grabbed = false;
        self.display.gl_window().window().hide_cursor(false);
    }

    pub fn draw_lit_mesh(&self, mat: M4x4, mesh: &DemoMesh) -> Result<(), Error> {
        let mut target = self.target_mut();
        target.draw(
            (&mesh.vbo,),
            &mesh.ibo,
            &self.shaders["lit"],
            &uniform! {
                model: mat.to_arr(),
                u_color: (mesh.color * mesh.color).to_arr(),
                view: self.view.to_arr(),
                u_light: self.light_dir,
                perspective: self.input.get_projection_matrix(
                    self.near_far.0, self.near_far.1).to_arr(),
                u_hl_intensity: self.hemilight_intensity,
                u_fog: self.fog_amount,
                u_camera_pos: self.camera.position.to_arr(),
                u_camera_q: self.camera.orientation.to_arr(),
                u_nearfar_dist: (self.near_far.1 - self.near_far.0).abs(),
            },
            &glium::DrawParameters {
                blend: glium::Blend::alpha_blending(),
                smooth: Some(glium::draw_parameters::Smooth::Nicest),
                backface_culling: glium::draw_parameters::BackfaceCullingMode::CullingDisabled,
                depth: glium::Depth {
                    test: glium::draw_parameters::DepthTest::IfLess,
                    write: true,
                    ..Default::default()
                },
                ..Default::default()
            },
        )?;
        Ok(())
    }

    pub fn draw_wire_mesh(&self, mat: M4x4, mesh: &DemoMesh, color: V4, use_depth: bool) -> Result<(), Error> {
        let mut wire = Vec::with_capacity(mesh.tris.len() * 6);
        for tri in &mesh.tris {
            let (v0, v1, v2) = Idx3::from(*tri).tri_verts(&mesh.verts);
            wire.extend([v0, v1, v1, v2, v2, v0].iter().cloned())
        }
        self.draw_solid(mat, color, &wire, glium::index::PrimitiveType::LinesList, use_depth)?;
        Ok(())
    }

    pub fn draw_tris(
        &self,
        mat: M4x4,
        color: V4,
        verts: &[V3],
        maybe_tris: Option<&[[u16; 3]]>,
        solid: bool,
    ) -> Result<(), Error> {
        let vbo = glium::VertexBuffer::new(&self.display, vertex_slice(verts))?;
        let params = glium::DrawParameters {
            point_size: if solid { Some(5.0) } else { None },
            line_width: Some(5.0),
            blend: glium::Blend::alpha_blending(),
            smooth: if solid {
                None
            } else {
                Some(glium::draw_parameters::Smooth::Nicest)
            },
            //            blend: glium::Blend::alpha_blending(),
            backface_culling: glium::draw_parameters::BackfaceCullingMode::CullingDisabled,
            depth: glium::Depth {
                test: glium::draw_parameters::DepthTest::IfLess,
                write: true,
                ..Default::default()
            },
            ..Default::default()
        };
        let shader = if solid {
            &self.shaders["solid"]
        } else {
            &self.shaders["lit"]
        };
        let uniforms = uniform! {
            model: mat.to_arr(),
            u_color: color.to_arr(),
            view: self.view.to_arr(),
            u_light: self.light_dir,
            u_camera_pos: self.camera.position.to_arr(),
            u_camera_q: self.camera.orientation.to_arr(),
            u_hl_intensity: self.hemilight_intensity,
            perspective: self.input.get_projection_matrix(self.near_far.0, self.near_far.1).to_arr(),
            u_fog: self.fog_amount,
        };
        let mut target = self.target_mut();
        if let Some(tris) = maybe_tris {
            let ibo = glium::IndexBuffer::new(
                &self.display,
                glium::index::PrimitiveType::TrianglesList,
                unpack_arr3(tris),
            )?;
            target.draw((&vbo,), &ibo, &shader, &uniforms, &params)?;
        } else {
            let ibo = glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList);
            target.draw((&vbo,), &ibo, &shader, &uniforms, &params)?;
        }
        Ok(())
    }

    pub fn set_camera(&mut self, p: Pose) {
        self.camera = p;
        self.view = p.inverse().to_mat4();
    }

    pub fn draw_solid(
        &self,
        mat: M4x4,
        color: V4,
        verts: &[V3],
        prim_type: glium::index::PrimitiveType,
        depth: bool,
    ) -> Result<(), Error> {
        let vbo = glium::VertexBuffer::new(&self.display, vertex_slice(verts)).unwrap();
        let ibo = glium::index::NoIndices(prim_type);
        let mut target = self.target_mut();
        let depth_test = if depth {
            glium::Depth {
                test: glium::draw_parameters::DepthTest::IfLess,
                write: true,
                ..Default::default()
            }
        } else {
            Default::default()
        };
        target.draw(
            (&vbo,),
            &ibo,
            &self.shaders["solid"],
            &uniform! {
                model: mat.to_arr(),
                u_color: color.to_arr(),
                view: self.view.to_arr(),
                u_hl_intensity: self.hemilight_intensity,
                u_camera_pos: self.camera.position.to_arr(),
                u_camera_q: self.camera.orientation.to_arr(),
                perspective: self.input.get_projection_matrix(
                    self.near_far.0, self.near_far.1).to_arr(),
            },
            &glium::DrawParameters {
                point_size: Some(5.0),
                line_width: Some(5.0),
                backface_culling: glium::draw_parameters::BackfaceCullingMode::CullingDisabled,
                blend: glium::Blend::alpha_blending(),
                depth: depth_test,
                ..Default::default()
            },
        )?;
        Ok(())
    }

    pub fn wm_draw_wireframe(&self, mat: M4x4, color: V4, wm: &WingMesh, use_depth: bool) -> Result<(), Error> {
        let mut verts = Vec::with_capacity(wm.edges.len() * 2);
        for e in &wm.edges {
            verts.push(wm.verts[e.vert_idx()]);
            verts.push(wm.verts[wm.edges[e.next_idx()].vert_idx()]);
        }
        self.draw_solid(
            mat,
            color,
            &verts[..],
            glium::index::PrimitiveType::LinesList,
            use_depth,
        )?;
        Ok(())
    }

    pub fn wm_draw_lit(&self, mat: M4x4, color: V4, wm: &WingMesh) -> Result<(), Error> {
        self.draw_tris(mat, color, &wm.verts, Some(&wm.generate_tris()), false)
    }

    pub fn draw_face(&self, mat: M4x4, color: V4, f: &Face) -> Result<(), Error> {
        self.draw_tris(mat, color, &f.vertex, Some(&f.gen_tris()[..]), false)
    }

    pub fn draw_faces(&self, mat: M4x4, faces: &[Face]) -> Result<(), Error> {
        // TODO: we could do this in 1 draw call if we offset the indices and abandon the color
        for face in faces.iter() {
            self.draw_face(mat, V4::expand(face.plane.normal, 1.0), face)?;
        }
        Ok(())
    }

    pub fn new_mesh(&self, verts: &[V3], tris: &[[u16; 3]], color: V4) -> Result<DemoMesh, Error> {
        DemoMesh::new(&self.display, verts.to_vec(), tris.to_vec(), color)
    }

    pub fn end_frame(&mut self) -> Result<(), Error> {
        let target = self.targ.take().unwrap().into_inner();
        target.finish()?;
        Ok(())
    }

    fn end_ui<'a>(&self, ui_render: &mut Renderer, ui: Ui<'a>) {
        let mut targ = self.target_mut();
        let glw = self.display.gl_window();
        self.input.platform.prepare_render(&ui, glw.window());

        let data = ui.render();
        let result = ui_render.render(&mut *targ, &data);
        if let Err(e) = result {
            println!("Error: {}", e);
        }
    }

    pub fn end_frame_and_ui<'a>(&mut self, ui_render: &mut Renderer, ui: Ui<'a>) -> Result<(), Error> {
        self.end_ui(ui_render, ui);
        self.end_frame()
    }
}

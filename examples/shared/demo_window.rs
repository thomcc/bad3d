
use bad3d::{WingMesh, bsp::Face, util::unpack_arr3};
use shared::{object::{DemoMesh, vertex_slice}, input::InputState};
use failure::Error;

use std::time::Instant;
use std::rc::Rc;
use std::cell::RefCell;
use imgui_glium_renderer::Renderer;

use bad3d::math::*;
use imgui::{self, Ui};

use glium::{
    self,
    Display,
    Surface,
    glutin::{
        EventsLoop,
        ContextBuilder,
        WindowBuilder,
    },
    backend::Facade,
};

// lit flat shader
static VERT_SRC: &'static str = include_str!("shaders/lit-vs.glsl");
static FRAG_SRC: &'static str = include_str!("shaders/lit-fs.glsl");

// solid color
static SOLID_VERT_SRC: &'static str = include_str!("shaders/solid-vs.glsl");
static SOLID_FRAG_SRC: &'static str = include_str!("shaders/solid-fs.glsl");

pub fn glmat(inp: M4x4) -> [[f32; 4]; 4] {
    inp.into()
}

pub struct DemoWindow {
    pub events: EventsLoop,
    pub display: Display,
    pub input: InputState,
    pub view: M4x4,
    pub lit_shader: glium::Program,
    pub solid_shader: glium::Program,
    pub clear_color: V4,
    pub light_pos: [f32; 3],
    pub targ: Option<glium::Frame>,
    pub near_far: (f32, f32),
    pub gui: Rc<RefCell<imgui::ImGui>>,
    pub gui_renderer: Renderer,
    pub last_frame: Instant,
    pub last_frame_time: f32,
}

#[derive(Clone, Debug)]
pub struct DemoOptions<'a> {
    pub clear_color: V4,
    pub title: &'a str,
    pub window_size: (u32, u32),
    pub near_far: (f32, f32),
    pub light_pos: V3,
    pub fov: f32,
    pub view: M4x4,
}

impl<'a> Default for DemoOptions<'a> {
    fn default() -> DemoOptions<'a> {
        DemoOptions {
            title: "Bad3d Demo",
            window_size: (1024, 768),
            clear_color: vec4(0.5, 0.6, 1.0, 1.0),
            light_pos: vec3(1.4, 0.4, 0.7),
            near_far: (0.01, 500.0),
            fov: 75.0,
            view: M4x4::identity(),
        }
    }
}

impl DemoWindow {
    pub fn new(opts: DemoOptions) -> Result<DemoWindow, Error> {
        let context = ContextBuilder::new()
            .with_depth_buffer(24)
            .with_vsync(true);

        let window = WindowBuilder::new()
            .with_title(opts.title)
            .with_min_dimensions(opts.window_size.0, opts.window_size.1);

        let events = EventsLoop::new();

        let display = Display::new(window, context, &events).unwrap();

        let gui = Rc::new(RefCell::new(imgui::ImGui::init()));
        gui.borrow_mut().set_ini_filename(None);
        let gui_renderer = Renderer::init(&mut gui.borrow_mut(), &display).unwrap();

        let input_state = InputState::new(
            display.get_framebuffer_dimensions(), opts.fov, gui.clone());

        let phong_program = glium::Program::from_source(&display, VERT_SRC, FRAG_SRC, None)?;
        let solid_program = glium::Program::from_source(&display, SOLID_VERT_SRC, SOLID_FRAG_SRC, None)?;

        Ok(DemoWindow {
            display,
            events,
            input: input_state,
            view: opts.view,
            lit_shader: phong_program,
            solid_shader: solid_program,
            clear_color: opts.clear_color,
            light_pos: opts.light_pos.into(),
            near_far: opts.near_far,
            targ: None,
            gui,
            gui_renderer,
            last_frame: Instant::now(),
            last_frame_time: 0.0,
        })
    }

    pub fn is_up(&mut self) -> bool {
        let now = Instant::now();
        let delta = now - self.last_frame;
        let delta_s = delta.as_secs() as f32 +
                      delta.subsec_nanos() as f32 / 1_000_000_000.0;
        self.last_frame = now;
        self.last_frame_time = delta_s;
        assert!(self.targ.is_none());
        if !self.input.update(&mut self.events) {
            false
        } else {
            self.targ = Some(self.display.draw());
            self.targ.as_mut().unwrap().clear_color_and_depth(self.clear_color.into(), 1.0);
            true
        }
    }

    pub fn draw_lit_mesh(&mut self, mat: M4x4, mesh: &DemoMesh) -> Result<(), Error> {
        self.targ.as_mut().unwrap().draw(
            (&mesh.vbo,),
            &mesh.ibo,
            &self.lit_shader,
            &uniform! {
                model: glmat(mat),
                u_color: <[f32; 4]>::from(mesh.color),
                view: glmat(self.view),
                u_light: <[f32; 3]>::from(self.light_pos),
                perspective: glmat(self.input.get_projection_matrix(
                    self.near_far.0, self.near_far.1)),
            },
            &glium::DrawParameters {
                blend: glium::Blend::alpha_blending(),
                depth: glium::Depth {
                    test: glium::draw_parameters::DepthTest::IfLess,
                    write: true,
                    .. Default::default()
                },
                .. Default::default()
            })?;
        Ok(())
    }

    pub fn draw_tris(
        &mut self,
        mat: M4x4,
        color: V4,
        verts: &[V3],
        maybe_tris: Option<&[[u16; 3]]>,
        solid: bool
    ) -> Result<(), Error> {
        let vbo = glium::VertexBuffer::new(&self.display, vertex_slice(verts))?;
        let params = glium::DrawParameters {
            point_size: if solid { Some(5.0) } else { None },
            blend: glium::Blend::alpha_blending(),
            depth: glium::Depth {
                test: glium::draw_parameters::DepthTest::IfLess,
                write: true,
                .. Default::default()
            },
            .. Default::default()
        };
        let shader = if solid { &self.solid_shader } else { &self.lit_shader };
        let uniforms = uniform! {
            model: glmat(mat),
            u_color: <[f32; 4]>::from(color),
            view: glmat(self.view),
            u_light: <[f32; 3]>::from(self.light_pos),
            perspective: glmat(self.input.get_projection_matrix(self.near_far.0, self.near_far.1)),
        };
        if let Some(tris) = maybe_tris {
            let ibo = glium::IndexBuffer::new(
                &self.display,
                glium::index::PrimitiveType::TrianglesList,
                unpack_arr3(tris)
            )?;
            self.targ.as_mut().unwrap().draw((&vbo,), &ibo, &shader, &uniforms, &params)?;
        } else {
            let ibo = glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList);
            self.targ.as_mut().unwrap().draw((&vbo,), &ibo, &shader, &uniforms, &params)?;
        }
        Ok(())
    }

    pub fn draw_solid(&mut self, mat: M4x4, color: V4, verts: &[V3], prim_type: glium::index::PrimitiveType) -> Result<(), Error> {
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
                })?;
        Ok(())
    }

    pub fn wm_draw_wireframe(&mut self, mat: M4x4, color: V4, wm: &WingMesh) -> Result<(), Error> {
        let mut verts = Vec::with_capacity(wm.edges.len()*2);
        for e in &wm.edges {
            verts.push(wm.verts[e.vert_idx()]);
            verts.push(wm.verts[wm.edges[e.next_idx()].vert_idx()]);
        }
        self.draw_solid(mat, color, &verts[..], glium::index::PrimitiveType::LinesList)?;
        Ok(())
    }

    pub fn draw_face(&mut self, mat: M4x4, color: V4, f: &Face) -> Result<(), Error> {
        self.draw_tris(mat, color, &f.vertex, Some(&f.gen_tris()[..]), false)?;
        Ok(())
    }

    pub fn draw_faces(&mut self, mat: M4x4, faces: &[Face]) -> Result<(), Error> {
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
        self.targ.take().unwrap().finish()?;
        Ok(())
    }

    pub fn ui<F: FnMut(&Ui) -> Result<(), Error>>(&mut self, mut ui_callback: F) -> Result<(), Error> {
        let gl_window = self.display.gl_window();
        // let size_points = gl_window.get_inner_size_points().unwrap();
        // let size_pixels = gl_window.get_inner_size_pixels().unwrap();
        let dpi = gl_window.hidpi_factor();
        let size = gl_window.get_inner_size().unwrap();
        let size_points = ((size.0 as f32 * dpi) as u32,
                           (size.1 as f32 * dpi) as u32);
        let mut gui = self.gui.borrow_mut();
        let ui = gui.frame(size, size_points, self.last_frame_time);
        ui_callback(&ui)?;
        let mut targ = self.targ.take().unwrap();
        let result = self.gui_renderer.render(&mut targ, ui);
        self.targ = Some(targ);
        if let Err(e) = result {
            println!("Error: {}", e);
        }
        Ok(())
    }
}

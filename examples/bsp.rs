#![allow(dead_code)]
#![allow(unused_imports)]
#[macro_use]
extern crate glium;
use rand;

#[macro_use]
extern crate imgui;
use imgui_glium_renderer;

#[macro_use]
extern crate bad3d;

#[macro_use]
extern crate failure;
#[macro_use]
extern crate more_asserts;

#[macro_use]
mod shared;

use shared::{input::InputState, object, DemoMesh, DemoObject, DemoOptions, DemoWindow, Result};

use bad3d::prelude::*; //{hull, bsp, gjk, wingmesh::WingMesh, phys::{self, RigidBody, RigidBodyRef}};
                       // use bad3d::math::*;
use std::cell::RefCell;
use std::f32;
use std::fmt::Debug;
use std::rc::Rc;
use std::time::{Duration, Instant};
#[global_allocator]
static GLOBAL: mimallocator::Mimalloc = mimallocator::Mimalloc;

pub trait PickableEnum: Debug + Clone + PartialEq {
    fn variants() -> Vec<Self>;
}

macro_rules! implement_pickable {
    ($enum_name:ident { $($field:ident),* $(,)* }) => {
        impl PickableEnum for $enum_name {
            fn variants() -> Vec<Self> {
                use $enum_name :: *;
                vec![ $($field),* ]
            }
        }
    };
}

trait ImguiExt {
    fn enum_picker<E>(&mut self, label: &str, cur_choice: &mut E) -> (bool, E)
    where
        E: PickableEnum + Debug + PartialEq + Clone;
}

impl<'a> ImguiExt for imgui::Ui<'a> {
    fn enum_picker<E: PickableEnum>(&mut self, label: &str, cur_choice: &mut E) -> (bool, E) {
        let choices = E::variants();
        let items_storage = choices
            .iter()
            .map(|choice| im_str!("{:?}", choice).clone())
            .collect::<Vec<imgui::ImString>>();
        let items = items_storage
            .iter()
            .map(|r| r.as_ref())
            .collect::<Vec<&imgui::ImStr>>();

        let mut cur_item = choices
            .iter()
            .enumerate()
            .find(|(_, x)| x == &cur_choice)
            .unwrap()
            .0 as i32;

        let res = self.combo(&im_str!("{}", label), &mut cur_item, &items, -1);
        if cur_item < 0 || cur_item as usize > choices.len() {
            (false, cur_choice.clone())
        } else {
            *cur_choice = choices[cur_item as usize].clone();
            (res, cur_choice.clone())
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum DrawMode {
    Faces,
    ConvexCells,
}

implement_pickable!(DrawMode { Faces, ConvexCells });

#[derive(Debug, Clone, Copy)]
struct UiOptions {
    pub draw_mode: DrawMode,
    pub cell_wireframe: bool,
    pub cell_wires_respect_depth: bool,
    pub cell_scale: f32,

    pub collider_solids: bool,
    pub collider_wires: bool,
}

#[derive(Copy, Debug, Clone)]
enum Shape {
    Rect(V3),
    Octahedron(V3),
    Cylinder { sides: usize, r: f32, h: f32 },
    Cone { sides: usize, r: f32, h: f32 },
    Sphere { lat: usize, lng: usize, r: f32 },
}

impl Shape {
    fn name(&self) -> &'static str {
        match self {
            Shape::Rect(_) => "Rect",
            Shape::Octahedron(_) => "Octahedron",
            Shape::Cylinder { .. } => "Cylinder",
            Shape::Cone { .. } => "Cone",
            Shape::Sphere { .. } => "Sphere",
        }
    }
}

#[derive(Copy, Debug, Clone)]
enum CsgOp {
    Intersect,
    Union,
    Subtract,
}

#[derive(Clone)]
struct SceneObj {
    shape: Shape,
    mesh: WingMesh,
    faces: Vec<bsp::Face>,
    pos: V3,
    color: V4,
}

impl SceneObj {
    pub fn new(shape: Shape, pos: V3) -> SceneObj {
        let mesh = match shape {
            Shape::Rect(size) => WingMesh::new_box(size * -0.5, size * 0.5),
            Shape::Octahedron(V3 { x, y, z }) => {
                let s = vec3(1.0 / x, 1.0 / y, 1.0 / z);
                WingMesh::new_box(-s, s).dual()
            }
            Shape::Cylinder { sides, r, h } => WingMesh::new_cylinder(sides, r, h),
            Shape::Cone { sides, r, h } => WingMesh::new_cone(sides, r, h),
            Shape::Sphere { lat, lng, r } => WingMesh::new_sphere(r, (lat, lng)),
        };
        let faces = mesh.faces();
        SceneObj {
            shape,
            pos,
            mesh,
            faces,
            color: object::random_color(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum DragMode {
    Scene,
    Obj(usize),
}

pub struct BspScene {
    bsp: Option<Box<BspNode>>,
    root_obj: SceneObj,
    objects: Vec<(SceneObj, CsgOp)>,
    drag_mode: Option<DragMode>,
    faces: Vec<bsp::Face>,
    cam: Pose,
    cam_dist: f32,
    world_size: f32,
    hit_dist: f32,
}

impl BspScene {
    pub fn new() -> BspScene {
        BspScene {
            drag_mode: None,
            bsp: None,
            cam_dist: 5.0,
            world_size: 2.0,
            hit_dist: 0.0,
            faces: vec![],
            cam: Pose::from_rotation(Quat::from_axis_angle(
                vec3(1.0, 0.0, 0.0),
                60f32.to_radians(),
            )),
            objects: vec![
                (
                    SceneObj::new(Shape::Rect(vec3(1.0, 1.0, 2.4)), vec3(0.0, 0.0, 0.5)),
                    CsgOp::Subtract,
                ),
                (
                    SceneObj::new(
                        Shape::Cylinder {
                            sides: 20,
                            r: 0.6,
                            h: 1.0,
                        },
                        vec3(0.6, 0.8, 0.2),
                    ),
                    CsgOp::Union,
                ),
                (
                    SceneObj::new(
                        Shape::Octahedron(vec3(0.85, 0.85, 0.85)),
                        vec3(0.8, 0.0, 0.45),
                    ),
                    CsgOp::Subtract,
                ),
                (
                    SceneObj::new(
                        Shape::Sphere {
                            lat: 8,
                            lng: 8,
                            r: 1.0,
                        },
                        vec3(0.2, 0.1, 0.2),
                    ),
                    CsgOp::Subtract,
                ),
            ],
            root_obj: SceneObj::new(Shape::Rect(V3::splat(2.0)), V3::zero()),
        }
    }

    pub fn handle_input(&mut self, input: &InputState) {
        if !input.mouse.down.0 {
            self.drag_mode = None;
            return;
        }
        if input.shift_down() {
            self.drag_mode = Some(DragMode::Scene);
        }
        let dm = self.drag_mode.clone();
        match dm {
            Some(DragMode::Scene) => {
                self.cam.orientation *= Quat::virtual_track_ball(
                    vec3(0.0, 0.0, 2.0),
                    V3::zero(),
                    input.mouse_prev.vec,
                    input.mouse.vec,
                )
                .conj();
            }
            Some(DragMode::Obj(which)) => {
                assert_lt!(which, self.objects.len());
                let pos_offset = (self.cam.orientation * input.mouse.vec
                    - self.cam.orientation * input.mouse_prev.vec)
                    * self.hit_dist;
                self.objects[which].0.pos += pos_offset;
                self.bsp = None;
            }
            None => {
                let mut next_mode = DragMode::Scene;
                let v0 = self.cam.position;
                let mut v1 = self.cam.position + self.cam.orientation * (input.mouse.vec * 100.0);

                for (i, o) in self.objects.iter().enumerate() {
                    let hit_info = geom::convex_hit_check_posed(
                        o.0.mesh.faces.iter().cloned(),
                        Pose::from_translation(o.0.pos),
                        v0,
                        v1,
                    );
                    if let Some(hit) = hit_info {
                        v1 = hit.impact;
                        next_mode = DragMode::Obj(i);
                    }
                }
                self.hit_dist = v0.dist(v1);
                self.drag_mode = Some(next_mode);
            }
        }
    }

    pub fn maybe_update_bsp(&mut self) -> Option<Duration> {
        if self.bsp.is_some() {
            return None;
        }
        let now = Instant::now();
        let world = WingMesh::new_cube(self.world_size);
        let mut bspres = bsp::compile(self.root_obj.faces.clone(), world.clone());
        for (obj, op) in &self.objects {
            let mut b = bsp::compile(obj.faces.clone(), world.clone());
            b.translate(obj.pos);
            match op {
                CsgOp::Union => bspres = bsp::union(b, bspres),
                CsgOp::Intersect => bspres = bsp::intersect(b, bspres),
                CsgOp::Subtract => {
                    b.negate();
                    bspres = bsp::intersect(b, bspres);
                }
            }
        }
        bspres.rebuild_boundary();
        self.faces = bspres.rip_boundary();
        self.bsp = bsp::clean(bspres);
        assert!(
            self.bsp.is_some(),
            "Somehow our compiled BSP ended up as a leaf?"
        );
        Some(now.elapsed())
    }
}

#[inline]
fn duration_ms(d: Duration) -> f32 {
    1000.0 * (d.as_secs() as f32 + d.subsec_nanos() as f32 / 1_000_000_000.0)
}

pub fn main() -> Result<()> {
    env_logger::init();
    let gui = Rc::new(RefCell::new(imgui::Context::create()));
    gui.borrow_mut().set_ini_filename(None);

    let mut win = DemoWindow::new(
        DemoOptions {
            title: "BSP (CSG) test",
            fov: 45.0,
            light_pos: vec3(-1.0, 0.5, 0.5),
            near_far: (0.01, 10.0),
            ..Default::default()
        },
        gui.clone(),
    )?;

    let mut gui_renderer =
        imgui_glium_renderer::Renderer::init(&mut *gui.borrow_mut(), &win.display).unwrap();

    win.input.view_angle = 45.0;

    let mut scene = BspScene::new();

    let mut ui_opts = UiOptions {
        draw_mode: DrawMode::ConvexCells,
        cell_wireframe: false,
        cell_wires_respect_depth: false,
        cell_scale: 0.95,
        collider_wires: true,
        collider_solids: false,
    };
    let mut last_rebuild_ms: f32 = 0.0;

    while win.is_up() {
        let mut imgui = gui.borrow_mut();
        let mut ui = imgui.frame();
        use glium::glutin::VirtualKeyCode as Key;
        if win.input.key_hit(Key::Q) {
            win.end_frame()?;
            break;
        }
        if !ui.io().want_capture_mouse {
            scene.handle_input(&win.input);
        }
        // XXX hack
        scene.cam.position = scene.cam.orientation.z_dir() * scene.cam_dist;
        if let Some(dur) = scene.maybe_update_bsp() {
            last_rebuild_ms = duration_ms(dur);
            println!("Rebuilt bsp in {:.3}ms", last_rebuild_ms);
        }

        win.view = scene.cam.inverse().to_mat4();

        let render_start = Instant::now();

        if ui_opts.collider_wires {
            win.wm_draw_wireframe(
                M4x4::identity(),
                vec4(0.0, 1.0, 0.5, 1.0),
                &scene.root_obj.mesh,
                false,
            )?;
            for (obj, _) in scene.objects.iter() {
                win.wm_draw_wireframe(
                    M4x4::from_translation(obj.pos),
                    obj.color,
                    &obj.mesh,
                    false,
                )?;
            }
        }

        if ui_opts.collider_solids {
            for (obj, _) in scene.objects.iter() {
                win.wm_draw_lit(M4x4::from_translation(obj.pos), obj.color, &obj.mesh)?;
            }
        }

        match ui_opts.draw_mode {
            DrawMode::Faces => {
                win.draw_faces(M4x4::identity(), &scene.faces)?;
            }
            DrawMode::ConvexCells => {
                let mut stack = vec![scene.bsp.as_ref().unwrap().as_ref()];
                while let Some(n) = stack.pop() {
                    if n.leaf_type == bsp::LeafType::Under {
                        let c = n.convex.verts.iter().fold(V3::zero(), |a, &b| a + b)
                            / (n.convex.verts.len() as f32);
                        let m = M4x4::from_translation(c)
                            * M4x4::from_scale(V3::splat(ui_opts.cell_scale))
                            * M4x4::from_translation(-c);
                        win.draw_tris(
                            m,
                            V4::splat(1.0),
                            &n.convex.verts,
                            Some(&n.convex.generate_tris()),
                            false,
                        )?;
                        if ui_opts.cell_wireframe {
                            win.wm_draw_wireframe(
                                M4x4::identity(),
                                vec4(1.0, 0.0, 1.0, 1.0),
                                &n.convex,
                                ui_opts.cell_wires_respect_depth,
                            )?;
                        }
                    }
                    if let Some(ref r) = n.under {
                        stack.push(r.as_ref());
                    }
                    if let Some(ref r) = n.over {
                        stack.push(r.as_ref());
                    }
                }
            }
        }

        let render_time = duration_ms(render_start.elapsed());

        let framerate = ui.io().framerate;
        ui.window(im_str!("test"))
            .position([20.0, 20.0], imgui::Condition::Appearing)
            .build(|| {
                ui.text(im_str!("fps: {:.3}", framerate));
                if cfg!(debug_assertions) {
                    ui.text(im_str!("  debug_assertions enabled (slow)"));
                }
                ui.text(im_str!("render_ms: {:.3}", render_time));
                ui.text(im_str!("last_rebuild_ms: {:.3}", last_rebuild_ms));
                ui.separator();
                let e = ui.enum_picker("draw mode", &mut ui_opts.draw_mode).1;
                match e {
                    DrawMode::Faces => {
                        // No special options
                    }
                    DrawMode::ConvexCells => {
                        ui.checkbox(im_str!("cell wireframe?"), &mut ui_opts.cell_wireframe);
                        if ui_opts.cell_wireframe {
                            ui.checkbox(
                                im_str!("cell wires respect depth?"),
                                &mut ui_opts.cell_wires_respect_depth,
                            );
                        }
                        ui.slider_float(im_str!("cell scale"), &mut ui_opts.cell_scale, 0.1, 1.0)
                            .build();
                    }
                }
                ui.separator();
                ui.text(im_str!("intersected objects"));
                ui.checkbox(im_str!("show solid"), &mut ui_opts.collider_solids);
                ui.checkbox(im_str!("show wires"), &mut ui_opts.collider_wires);

                ui.separator();
                ui.text(im_str!("click and drag to move items or camera"));
                ui.text(im_str!("  hold shift to force camera"));
                ui.separator();
                ui.text(im_str!("Scene:"));
                ui.text(im_str!("  2x2x2 Box"));
                for (obj, op) in &scene.objects {
                    ui.text(im_str!("  {:?} {}", op, obj.shape.name()));
                }
            });

        win.end_frame_and_ui(&mut gui_renderer, ui)?;
    }
    Ok(())
}

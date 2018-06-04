
use bad3d::{hull, wingmesh, util::unpack_arr3, phys};
use bad3d::math::*;
use glium::{self, backend::Facade};

use std::rc::Rc;
use std::cell::RefCell;
use rand;

use failure::Error;
use std::{result, slice};

type Result<T> = result::Result<T, Error>;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Vertex {
    pub position: [f32; 3]
}

implement_vertex!(Vertex, position);
/*
#[repr(C)]
#[derive(Copy, Clone)]
pub struct TVertex {
    pub position: [f32; 3],
    pub texcoord: [u16; 2],
    pub color: [u8; 4],
}

implement_vertex!(TVertex, position, texcoord normalize(true), color normalize(true));
*/
pub fn vertex_slice<'a>(v3s: &'a [V3]) -> &'a [Vertex] {
    unsafe { slice::from_raw_parts(v3s.as_ptr() as *const Vertex, v3s.len()) }
}

pub struct DemoMesh {
    pub verts: Vec<V3>,
    pub tris: Vec<[u16; 3]>,
    pub color: V4,
    pub ibo: glium::IndexBuffer<u16>,
    pub vbo: glium::VertexBuffer<Vertex>,
}

impl DemoMesh {
    pub fn new<F: Facade>(
        display: &F,
        verts: Vec<V3>,
        tris: Vec<[u16; 3]>,
        color: V4
    ) -> Result<DemoMesh> {
        let vbo = glium::VertexBuffer::new(display, vertex_slice(&verts[..]))?;
        let ibo = glium::IndexBuffer::new(
            display,
            glium::index::PrimitiveType::TrianglesList,
            unpack_arr3(&tris[..])
        )?;
        Ok(DemoMesh { color, verts, tris, vbo, ibo })
    }

    pub fn from_shape<F: Facade>(display: &F, s: &phys::Shape, color: Option<V4>) -> Result<DemoMesh> {
        DemoMesh::new(display, s.vertices.clone(), s.tris.clone(), color.unwrap_or_else(random_color))
    }
}

#[derive(Debug, Clone)]
pub struct PtLight {
    pub pos: V3,
    pub radius: f32,
    pub color: V4,
}
/*
pub struct MeshScene {
    pub meshes: Vec<TDemoMesh>,
    pub materials: Vec<Rc<MaterialData>>,
    pub lights: Vec<PtLight>,
}

#[derive(Clone)]
pub struct MaterialData {
    pub diffuse: Rc<glium::Texture2d>,
    pub emissive: V3,
    pub diffuse_tint: V4,
    pub diffuse_coef: f32,
    pub gloss: f32,
    pub specular_intensity: f32,
}

#[derive(Clone)]
pub struct TDemoMesh {
    pub ibo: glium::IndexBuffer<u16>,
    pub vbo: glium::VertexBuffer<TVertex>,
    pub material: Rc<MaterialData>,
    pub matrix: M4x4,
}
*/
pub fn random_color() -> V4 {
    let mut c = vec4(rand::random::<f32>(), rand::random::<f32>(), rand::random::<f32>(), 1.0);
    c[rand::random::<usize>() % 3] = rand::random::<f32>()*0.5;
    c
}

pub fn rand_v3() -> V3 {
    vec3(rand::random::<f32>(),
         rand::random::<f32>(),
         rand::random::<f32>())
}

pub struct DemoObject {
    pub body: phys::RigidBodyRef,
    pub meshes: Vec<Box<DemoMesh>>
}

pub fn convex_parts(mut v: Vec<V3>) -> (Vec<V3>, Vec<[u16; 3]>) {
    let tris = hull::compute_hull_trunc(&mut v, None).expect("Planar or linear vertices...");
    (v, tris)
}

pub fn convex_shape(v: Vec<V3>) -> phys::Shape {
    let (v, tris) = convex_parts(v);
    phys::Shape::new(v, tris)
}

pub fn random_point_cloud(size: usize) -> (Vec<V3>, Vec<[u16; 3]>) {
    assert!(size > 4);
    let mut vs = vec![V3::zero(); size];
    loop {
        for item in vs.iter_mut() {
            *item = rand_v3() + V3::splat(-0.5);
        }
        if let Some((indices, _)) = hull::compute_hull(&mut vs[..]) {
            return (vs, indices)
        } else {
            vs.push(V3::zero());
        }
    }
}

impl DemoObject {
    pub fn new_box<F: Facade>(facade: &F, r: V3, com: V3, orient: Option<Quat>) -> Result<DemoObject> {
        DemoObject::from_shape(facade, phys::Shape::new_box(r), com, orient)
    }

    pub fn new_octa<F: Facade>(facade: &F, r: V3, com: V3, orient: Option<Quat>) -> Result<DemoObject> {
        DemoObject::from_shape(facade, phys::Shape::new_octa(r), com, orient)
    }

    pub fn new_cloud<F: Facade>(facade: &F, com: V3, orient: Option<Quat>) -> Result<DemoObject> {
        let (verts, tris) = random_point_cloud(15);
        DemoObject::from_shape(facade, phys::Shape::new(verts, tris), com, orient)
    }

    pub fn from_shape<F: Facade>(facade: &F, s: phys::Shape, com: V3, orient: Option<Quat>) -> Result<DemoObject> {
        let body = phys::RigidBody::new_ref(vec![s], com, 1.0);
        if let Some(p) = orient {
            body.borrow_mut().pose.orientation = p;
            body.borrow_mut().start_pose.orientation = p;
        }
        DemoObject::from_body(facade, body)
    }

    pub fn from_body<F: Facade>(facade: &F, body: phys::RigidBodyRef) -> Result<DemoObject> {
        let color = random_color();
        let meshes = body.borrow().shapes.iter()
            .map(|s| DemoMesh::new(facade, s.vertices.clone(), s.tris.clone(), color).map(Box::new))
            .collect::<Result<Vec<_>>>()?;
        Ok(DemoObject { body, meshes })
    }

    pub fn from_wingmesh<F: Facade>(facade: &F, m: wingmesh::WingMesh, com: V3) -> Result<DemoObject> {
        let body = phys::RigidBody::new_ref(vec![m.into()], com, 1.0);
        let meshes = body.borrow().shapes.iter()
            .map(|s| DemoMesh::new(facade, s.vertices.clone(), s.tris.clone(), random_color()).map(Box::new))
            .collect::<Result<Vec<_>>>()?;
        Ok(DemoObject { body, meshes })
    }
}

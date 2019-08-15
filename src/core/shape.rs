use crate::core::hull;
use crate::core::wingmesh;
use crate::math::prelude::*;

#[derive(Debug, Clone, Default)]
pub struct Shape {
    pub vertices: Vec<V3>,
    pub tris: Vec<[u16; 3]>,
}

impl Shape {
    #[inline]
    pub fn new(vertices: Vec<V3>, tris: Vec<[u16; 3]>) -> Self {
        Self { vertices, tris }
    }

    #[inline]
    pub fn from_winged(wm: &wingmesh::WingMesh) -> Shape {
        let tris = wm.generate_tris();
        Shape::new(wm.verts.clone(), tris)
    }

    #[inline]
    pub fn new_hull(mut vertices: Vec<V3>) -> Option<Self> {
        if let Some(tris) = hull::compute_hull_trunc(&mut vertices, None) {
            Some(Self { vertices, tris })
        } else {
            None
        }
    }

    #[inline]
    pub fn new_box_at(radii: V3, com: V3) -> Self {
        let size = radii.abs().map(|x| x.max(1.0e-3));
        let mut vertices = Vec::with_capacity(8);
        for &z in &[-size.z(), size.z()] {
            for &y in &[-size.y(), size.y()] {
                for &x in &[-size.x(), size.x()] {
                    vertices.push(vec3(x, y, z) + com)
                }
            }
        }
        // This is so lazy...
        Shape::new_hull(vertices).unwrap()
    }

    #[inline]
    pub fn new_box(radii: V3) -> Self {
        Shape::new_box_at(radii, V3::zero())
    }

    #[inline]
    pub fn new_aabb(min: V3, max: V3) -> Self {
        Shape::new_box_at((max - min) / 2.0, (max + min) / 2.0)
    }

    pub fn new_sphere(radius: f32, bands: (usize, usize)) -> Self {
        use std::{f32, u16};
        chek::le!(bands.0 * bands.1 * 4, u16::MAX as usize);
        let mut vertices = Vec::with_capacity(bands.0 * bands.1 * 4);
        let mut tris = Vec::with_capacity(bands.0 * bands.1 * 2);
        let lat_step = f32::consts::PI / (bands.0 as f32);
        let lng_step = f32::consts::PI * 2.0 / (bands.1 as f32);

        for i in 0..bands.0 {
            let lat_angle = (i as f32) * lat_step;
            let (lat_sin, y1) = lat_angle.sin_cos();
            let (lat_sin2, y2) = (lat_angle + lat_step).sin_cos();

            for j in 0..bands.1 {
                let lng_angle = (j as f32) * lng_step;
                let (lng_sin, lng_cos) = lng_angle.sin_cos();
                let (lng_sin2, lng_cos2) = (lng_angle + lng_step).sin_cos();

                let x1 = lat_sin * lng_cos;
                let x2 = lat_sin * lng_cos2;
                let x3 = lat_sin2 * lng_cos;
                let x4 = lat_sin2 * lng_cos2;

                let z1 = lat_sin * lng_sin;
                let z2 = lat_sin * lng_sin2;
                let z3 = lat_sin2 * lng_sin;
                let z4 = lat_sin2 * lng_sin2;

                // texcoords
                // let u1 = 1.0 - (j as f32) / (bands.1 as f32);
                // let u2 = 1.0 - (j as f32 + 1.0) / (bands.1 as f32);

                // let v1 = 1.0 - (i as f32) / (bands.0 as f32);
                // let v2 = 1.0 - (i as f32 + 1.0) / (bands.0 as f32);

                let k = vertices.len() as u16;
                vertices.extend(&[
                    // normals are same without * radius
                    vec3(x1, y1, z1) * radius,
                    vec3(x2, y1, z2) * radius,
                    vec3(x3, y2, z3) * radius,
                    vec3(x4, y2, z4) * radius,
                ]);

                tris.extend(&[[k, k + 1, k + 2], [k + 2, k + 1, k + 3]]);
            }
        }
        Shape { vertices, tris }
    }

    #[inline]
    pub fn new_octa(radii: V3) -> Self {
        let size = radii.abs().map(|x| x.max(1.0e-3));
        let vertices = vec![
            vec3(-size.x(), 0.0, 0.0),
            vec3(size.x(), 0.0, 0.0),
            vec3(0.0, -size.y(), 0.0),
            vec3(0.0, size.y(), 0.0),
            vec3(0.0, 0.0, -size.z()),
            vec3(0.0, 0.0, size.z()),
        ];
        // This is so lazy...
        Shape::new_hull(vertices).unwrap()
    }

    #[inline]
    pub fn volume(&self) -> f32 {
        geom::volume(&self.vertices, &self.tris)
    }

    #[inline]
    pub fn center_of_mass(&self) -> V3 {
        geom::center_of_mass(&self.vertices, &self.tris)
    }

    #[inline]
    pub fn inertia(&self, com: V3) -> M3x3 {
        geom::inertia(&self.vertices, &self.tris, com)
    }
}

impl From<wingmesh::WingMesh> for Shape {
    #[inline]
    fn from(wm: wingmesh::WingMesh) -> Shape {
        let tris = wm.generate_tris();
        Shape::new(wm.verts, tris)
    }
}

impl From<Shape> for wingmesh::WingMesh {
    #[inline]
    fn from(s: Shape) -> Self {
        wingmesh::WingMesh::from_mesh(s.vertices, &s.tris)
    }
}

pub fn combined_volume(shapes: &[Shape]) -> f32 {
    shapes.iter().fold(0.0, |sum, shape| sum + shape.volume())
}

pub fn combined_center_of_mass(shapes: &[Shape]) -> V3 {
    let mut com = V3::zero();
    let mut vol = 0.0f32;
    for mesh in shapes {
        let v = mesh.volume();
        let c = mesh.center_of_mass();
        vol += v;
        com += c * v;
    }
    com / vol
}

pub fn combined_inertia(shapes: &[Shape], com: V3) -> M3x3 {
    let mut vol = 0.0f32;
    let mut inertia = mat3(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for mesh in shapes {
        let v = mesh.volume();
        let i = mesh.inertia(com);
        vol += v;
        inertia += i * v;
    }
    inertia / vol
}

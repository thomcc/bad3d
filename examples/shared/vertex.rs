use glium::vertex as gv;
use std::borrow::Cow;
use std::convert::TryFrom;
use t3m::prelude::*;
#[repr(C)]
#[derive(Copy, Clone)]
pub struct VertexP {
    pub position: [f32; 4],
}
implement_vertex!(VertexP, position);

#[repr(C)]
#[derive(Copy, Clone)]
pub struct VertexC {
    pub position: [f32; 4],
    pub color: [u8; 4],
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct FullVertex {
    pub position: [f32; 3],
    pub orientation: [f32; 4],
    pub texcoord: [f32; 2],
}

implement_vertex!(FullVertex, position, orientation, texcoord);

// static_assert!((std::mem::size_of::<FullVertex>() % 16) == 0);

pub fn flat_mesh(verts: &[V3], tris: &[[u16; 3]]) -> (Vec<FullVertex>, Vec<[u16; 3]>) {
    let mut vert_out = Vec::with_capacity(tris.len() * 3);
    let mut tris_out = Vec::with_capacity(tris.len());
    for t in tris.iter().map(Idx3::from) {
        let (v0, v1, v2) = t.tri_verts(verts);
        let n = t3m::geom::tri_normal(v0, v1, v2);
        let vn = n.abs();
        let k = vn.max_index();
        chek::debug_lt!(
            vert_out.len() + 2,
            u16::max_value() as usize,
            "Mesh vertex count overflow u16::max_value :("
        );
        let c = vert_out.len() as u16;
        tris_out.push([c, c + 1, c + 2]);
        let k1 = (k + 1) % 3;
        let k2 = (k + 2) % 3;
        let st = t3m::geom::gradient(v0, v1, v2, v0[k1], v1[k1], v2[k1]);
        let sb = t3m::geom::gradient(v0, v1, v2, v0[k2], v1[k2], v2[k2]);
        let flip_u = dot(sb, cross(n, st)).signum();
        let st = cross(sb, n);
        let q = Quat::from_basis(st, sb, n);
        const UVMAX: f32 = u16::max_value() as f32;
        vert_out.reserve(3);
        for j in t.iter() {
            let v = verts[j as usize];
            // let tcu = (t3m::repeat(v[k1] * flip_u, 1.0) * UVMAX) as u16;
            // let tcv = (t3m::repeat(v[k2], 1.0) * UVMAX) as u16;
            vert_out.push(FullVertex {
                position: v.arr(),
                texcoord: [v[k1] * flip_u, v[k2]],
                orientation: q.arr(),
            });
        }
    }
    (vert_out, tris_out)
}

pub fn smooth_verts<I>(verts: &[V3], tris: &[[u16; 3]]) -> Vec<FullVertex> {
    chek::lt!(verts.len(), u16::max_value() as usize, "Too many vertices!");

    let mut mesh_verts = verts
        .iter()
        .map(|&p| FullVertex {
            position: p.arr(),
            texcoord: [p.x(), p.y()],
            orientation: [0.0; 4],
        })
        .collect::<Vec<_>>();

    for t in tris.iter().map(Idx3::from) {
        let (a, b, c) = t.tri_verts(verts);
        let sn = cross(b - a, c - a).fast_norm();
        let st = t3m::geom::gradient(
            a,
            b,
            c,
            mesh_verts[t.0 as usize].texcoord[0],
            mesh_verts[t.1 as usize].texcoord[0],
            mesh_verts[t.2 as usize].texcoord[0],
        );
        let sb = cross(sn, st);
        let q = Quat::from_basis(st, sb, sn);
        for i in 0..3 {
            for e in 0..4 {
                mesh_verts[t[i] as usize].orientation[e] += q[e];
            }
        }
    }
    for v in mesh_verts.iter_mut() {
        let orientation = Quat::from(v.orientation).norm_or_identity();
        v.orientation = orientation.into();
        // let n = orientation * vec3(0.0, 0.0, 1.0);
        // v.texcoord[0] = atan2(n,y, n.x);
        // v.texcoord[1] =  0.5 + n.y / 2.0;
    }
    mesh_verts
}

pub fn vertex_slice(v3s: &[t3m::V3]) -> &[VertexP] {
    unsafe { std::slice::from_raw_parts(v3s.as_ptr() as *const VertexP, v3s.len()) }
}

// impl gv::Vertex for FullVertex {
//     #[inline]
//     fn build_bindings() -> gv::VertexFormat {
//         use gv::AttributeType::*;
//         use Cow::Borrowed as S;
//         Cow::Borrowed(&[
//             (S("position"), 0, F32F32F32, false),
//             (S("texcoord"), 3 * 4, U16U16, true),
//             (S("orientation"), 4 * 4, F32F32F32F32, false),
//         ])
//     }
// }

impl gv::Vertex for VertexC {
    #[inline]
    fn build_bindings() -> gv::VertexFormat {
        use gv::AttributeType::*;
        use Cow::Borrowed as S;
        Cow::Borrowed(&[
            (S("position"), 0, F32F32F32F32, false),
            (S("color"), 16, U8U8U8U8, true),
        ])
    }
}

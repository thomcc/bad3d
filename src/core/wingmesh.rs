use math::prelude::*;
use std::{i32, u16};
use std::f32;
use core::support;
use core::bsp;
// TODO: this is a gnarly mess

#[derive(Copy, Clone, Debug)]
pub struct HalfEdge {
    pub id: i32,
    pub v: i32,
    pub adj: i32,
    pub next: i32,
    pub prev: i32,
    pub face: i32
}

impl Default for HalfEdge {
    #[inline] fn default() -> HalfEdge {
        HalfEdge { id: -1, v: -1, adj: -1, next: -1, prev: -1, face: -1 }
    }
}

impl HalfEdge {
    pub fn new(id: usize, v: usize, adj: usize, next: usize, prev: usize, face: usize) -> HalfEdge {
        HalfEdge {
            id: int(id),
            v: int(v),
            adj: int(adj),
            next: int(next),
            prev: int(prev),
            face: int(face),
        }
    }

    #[inline] pub fn idx(&self) -> usize { debug_assert!(self.id >= 0); self.id as usize }
    #[inline] pub fn adj_idx(&self) -> usize { debug_assert!(self.adj >= 0); self.adj as usize }
    #[inline] pub fn vert_idx(&self) -> usize { debug_assert!(self.v >= 0); self.v as usize }
    #[inline] pub fn next_idx(&self) -> usize { debug_assert!(self.next >= 0); self.next as usize }
    #[inline] pub fn prev_idx(&self) -> usize { debug_assert!(self.prev >= 0); self.prev as usize }
    #[inline] pub fn face_idx(&self) -> usize { debug_assert!(self.face >= 0); self.face as usize }

    #[inline]
    pub fn cull(&mut self) {
        self.v = -1;
        self.adj = -1;
        self.next = -1;
        self.prev = -1;
        self.face = -1;
    }
}

#[inline]
fn int(u: usize) -> i32 {
    debug_assert_lt!(u, i32::MAX as usize);
    u as i32
}

#[derive(Clone, Debug)]
pub struct WingMesh {
    pub edges: Vec<HalfEdge>,
    pub verts: Vec<V3>,
    pub faces: Vec<Plane>,
    pub vback: Vec<i32>,
    pub fback: Vec<i32>,
    pub is_packed: bool,
}

impl Default for WingMesh {
    fn default() -> WingMesh { WingMesh::new() }
}

pub struct FaceViewIterator<'a> {
    pub wm: &'a WingMesh,
    pub start: i32,
    pub current: i32,
}

impl<'a> Iterator for FaceViewIterator<'a> {
    type Item = &'a HalfEdge;
    fn next(&mut self) -> Option<&'a HalfEdge> {
        if self.current == -1 {
            return None;
        }
        assert_eq!(self.current, self.wm.edges[self.current as usize].id);
        let result = &self.wm.edges[self.current as usize];
        self.current = result.next;
        if self.current == self.start {
            self.current = -1;
        }
        Some(result)
    }
}

impl WingMesh {

    #[inline]
    pub fn with_capacities(verts: impl Into<Option<usize>>,
                           faces: impl Into<Option<usize>>,
                           edges: impl Into<Option<usize>>) -> Self {
        let mut wm = WingMesh::new();
        wm.reserve(verts, faces, edges);
        wm
    }

    #[inline]
    pub fn new() -> WingMesh {
        WingMesh {
            edges: Vec::new(),
            verts: Vec::new(),
            faces: Vec::new(),
            vback: Vec::new(),
            fback: Vec::new(),
            is_packed: true,
        }
    }

    #[inline]
    pub fn reserve(&mut self,
                   verts: impl Into<Option<usize>>,
                   faces: impl Into<Option<usize>>,
                   edges: impl Into<Option<usize>>) {
        if let Some(vs) = verts.into() { self.verts.reserve(vs); self.vback.reserve(vs); }
        if let Some(fs) = faces.into() { self.faces.reserve(fs); self.fback.reserve(fs); }
        if let Some(es) = edges.into() { self.edges.reserve(es); }
    }

    #[inline]
    pub fn clear(&mut self) {
        self.edges.clear();
        self.verts.clear();
        self.faces.clear();
        self.vback.clear();
        self.fback.clear();
        self.is_packed = true;
    }

    #[inline]
    pub fn new_rect(bounds: V3) -> WingMesh {
        WingMesh::new_box(bounds * -0.5, bounds * 0.5)
    }

    #[inline]
    pub fn new_cube(r: f32) -> WingMesh {
        WingMesh::new_box(V3::splat(-r), V3::splat(r))
    }

    pub fn new_mesh(verts: &[V3], tris: &[[u16; 3]]) -> WingMesh {
        WingMesh::from_mesh(verts.into(), tris)
    }

    pub fn from_mesh(verts: Vec<V3>, tris: &[[u16; 3]]) -> WingMesh {
        let mut m = WingMesh::new();
        if tris.is_empty() {
            return m;
        }

        let vc = 1 + tris.iter().fold(0, |a, b| max!(a, b[0], b[1], b[2])) as usize;
        assert_le!(vc, verts.len());

        m.verts = verts;
        m.verts.truncate(vc);
        m.vback.reserve(m.verts.len());

        m.faces.reserve(tris.len());
        m.fback.reserve(tris.len());
        m.edges.reserve(tris.len() * 3);

        for (i, t) in tris.iter().enumerate() {
            let mut e0 = HalfEdge::default();
            let mut e1 = HalfEdge::default();
            let mut e2 = HalfEdge::default();
            let f = int(i);

            e0.face = f;
            e1.face = f;
            e2.face = f;

            e0.v = t[0] as i32;
            e1.v = t[1] as i32;
            e2.v = t[2] as i32;

            let k = m.edges.len();
            let k0 = int(k + 0);
            let k1 = int(k + 1);
            let k2 = int(k + 2);

            e0.id = k0; e1.prev = k0; e2.next = k0;
            e1.id = k1; e2.prev = k1; e0.next = k1;
            e2.id = k2; e0.prev = k2; e1.next = k2;

            m.edges.extend(&[e0, e1, e2]);

            m.faces.push(Plane::from_tri(m.verts[t[0] as usize],
                                         m.verts[t[1] as usize],
                                         m.verts[t[2] as usize]));
        }

        m.finish();
        m
    }

    pub fn new_cylinder(sides: usize, radius: f32, height: f32) -> WingMesh {
        let mut mesh = WingMesh::new();

        for i in 0..sides {
            let progress = (i as f32) / (sides as f32);
            let a = 2.0 * f32::consts::PI * progress;
            let (s, c) = (a.sin(), a.cos());
            mesh.verts.push(vec3(c * radius, s * radius, 0.0));
            mesh.verts.push(vec3(c * radius, s * radius, height));
        }

        for i in 0..sides {
            let indices = [i*2, ((i+1)%sides)*2, ((i+1)%sides)*2+1, i*2+1];
            mesh.add_face(&indices[..]);
        }

        let mut bottom = Vec::with_capacity(sides);
        let mut top = Vec::with_capacity(sides);

        for i in 0..sides {
            bottom.push((sides - i - 1) * 2);
            top.push(i * 2 + 1);
        }
        mesh.add_face(&bottom);
        mesh.add_face(&top);
        mesh.finish();
        mesh
    }

    pub fn new_cone(sides: usize, radius: f32, height: f32) -> WingMesh {
        let mut mesh = WingMesh::new();

        for i in 0..sides {
            let progress = (i as f32) / (sides as f32);
            let a = 2.0 * f32::consts::PI * progress;
            mesh.verts.push(vec3(a.cos() * radius, a.sin() * radius, 0.0));
        }

        mesh.verts.push(vec3(0.0, 0.0, height));

        for i in 0..sides {
            let indices = [i, (i + 1) % sides, sides];
            mesh.add_face(&indices[..]);
        }
        let mut bottom = Vec::new();

        for i in 0..sides {
            bottom.push(sides-i-1);
        }
        mesh.add_face(&bottom[..]);
        mesh.finish();
        mesh
    }

    pub fn new_sphere(radius: f32, bands: (usize, usize)) -> Self {
        use std::f32;
        assert_gt!(bands.0, 2);
        assert_gt!(bands.1, 2);

        let mut mesh = WingMesh::with_capacities(
            bands.0 * bands.1 * 4, bands.0 * bands.1, None);

        let lat_step = f32::consts::PI / (bands.0 as f32);
        let lng_step = f32::consts::PI * 2.0 / (bands.1 as f32);


        for j in 0..(bands.0 - 1) {
            let polar = ((j + 1) as f32) * lat_step;
            let (sp, cp) = polar.sin_cos();
            for i in 0..bands.1 {
                let az = lng_step * (i as f32);
                let (sa, ca) = az.sin_cos();
                mesh.verts.push(vec3(sp * ca, cp, sp * sa) * radius);
            }
        }

        let top = mesh.verts.len();
        mesh.verts.push(vec3(0.0, radius, 0.0));

        let bottom = mesh.verts.len();
        mesh.verts.push(vec3(0.0, -radius, 0.0));

        for i in 0..bands.1 {
            let a = i;
            let b = (i + 1) % bands.1;
            mesh.add_face(&[top, b, a]);
        }
        for j in 0..(bands.0 - 2) {
            let a_start = j * bands.1;
            let b_start = (j + 1) * bands.1;
            for i in 0..bands.1 {
                let a = a_start + i;
                let a1 = a_start + ((i + 1) % bands.1);
                let b = b_start + i;
                let b1 = b_start + ((i + 1) % bands.1);
                mesh.add_face(&[a, a1, b1, b]);
            }
        }

        for i in 0..bands.1 {
            let off = bands.1 * (bands.0 - 2);
            let a = i + off;
            let b = ((i + 1) % bands.1) + off;
            mesh.add_face(&[bottom, a, b]);
        }
        mesh.finish();
        mesh
    }


    pub fn vertex_degree(&self, v: usize) -> i32 {
        let e0 = self.vback[v];
        let mut result = 0;
        if e0 != -1 {
            let mut e = e0;
            loop {
                result += 1;
                e = self.edges[self.edges[e as usize].adj as usize].next;
                if e == e0 {
                    break;
                }
            }
        }
        result
    }

    #[inline]
    pub fn adj_edge(&self, eid: usize) -> &HalfEdge {
        &self.edges[self.edges[eid].adj_idx()]
    }

    #[inline]
    pub fn next_edge(&self, eid: usize) -> &HalfEdge {
        &self.edges[self.edges[eid].next_idx()]
    }

    #[inline]
    pub fn prev_edge(&self, eid: usize) -> &HalfEdge {
        &self.edges[self.edges[eid].prev_idx()]
    }

    #[inline]
    pub fn debug_assert_valid(&self) {
        if cfg!(debug_assertions) {
            self.assert_valid();
        }
    }

    pub fn assert_valid(&self) {
        for (e, edge) in self.edges.iter().enumerate() {
            if !self.is_packed && edge.v == -1 {
                assert_eq!(edge.face, -1);
                assert_eq!(edge.next, -1);
                assert_eq!(edge.prev, -1);
                continue;
            }
            let id = int(e);
            assert_eq!(edge.id, id);
            assert_ge!(edge.v, 0);
            assert_ge!(edge.face, 0);
            assert_eq!(self.edges[edge.next_idx()].prev, id);
            assert_eq!(self.edges[edge.prev_idx()].next, id);
            assert_ne!(edge.adj, id);
            assert_eq!(self.edges[edge.adj_idx()].adj, id);
            assert_eq!(edge.v, self.edges[self.edges[edge.adj_idx()].next_idx()].v);
            assert_ne!(edge.v, self.edges[edge.adj_idx()].v);
        }
        for (i, &vb) in self.vback.iter().enumerate() {
            assert!((!self.is_packed && vb == -1) || self.edges[vb as usize].v == int(i));
        }
        for (i, &fb) in self.fback.iter().enumerate() {
            assert!((!self.is_packed && fb == -1) || self.edges[fb as usize].face == int(i));
        }
    }

    #[inline]
    pub fn iter_face<'a>(&'a self, face: usize) -> FaceViewIterator<'a> {
        let fb = self.fback[face];
        FaceViewIterator { wm: self, start: fb, current: fb }
    }

    pub fn face_verts(&self, face: usize) -> Vec<V3> {
        self.iter_face_verts(face).collect()
    }

    pub fn iter_face_verts<'a>(&'a self, face: usize) -> impl Iterator<Item = V3> + 'a {
        self.iter_face(face).map(move |edge| self.verts[edge.vert_idx()])
    }

    #[inline]
    pub fn face_plane(&self, face: usize) -> Plane {
        let vs = self.face_verts(face);
        // TODO: we dont need to allocate a vec to compute this...
        Plane::from_points(&vs)
    }

    #[inline]
    pub fn update_face_plane(&mut self, face: usize) {
        let plane = self.face_plane(face);
        self.faces[face] = plane;
    }

    pub fn link_mesh(&mut self) {
        let mut edge_v: Vec<usize> = (0..self.edges.len()).collect();
        edge_v.sort_by(|&a, &b|
            self.edges[a].v.cmp(&self.edges[b].v));

        let mut ve_back = vec![0i32; self.verts.len()];
        for i in (0..self.edges.len()).rev() {
            ve_back[self.edges[edge_v[i]].vert_idx()] = int(i);
        }
        for i in 0..self.edges.len() {
            assert_eq!(self.edges[i].id, int(i));
            if self.edges[i].adj != -1 {
                continue;
            }
            let a = self.edges[i].v;
            let b = self.edges[self.edges[i].next_idx()].v;
            let mut k = ve_back[b as usize] as usize;
            while k < self.edges.len() && self.edges[edge_v[k]].v == b {
                if self.edges[self.edges[edge_v[k]].next_idx()].v == a {
                    self.edges[i].adj = int(edge_v[k]);
                    let id = self.edges[i].id;
                    self.edges[edge_v[k]].adj = id;
                    break;
                }
                k += 1;
            }
            assert_ne!(self.edges[i].adj, -1);
        }
    }

    pub fn init_back_lists(&mut self) {
        self.vback.clear();
        self.fback.clear();
        self.vback.resize(self.verts.len(), -1);
        self.fback.resize(self.faces.len(), -1);
        // for i in (0..self.edges.len()).rev() {
        // let mut i = self.edges.len();
        // while i != 0 {
            // i -= 1;
        for (i, edge) in self.edges.iter().enumerate().rev() {
            if !self.is_packed && edge.v == -1 {
                continue;
            }
            self.vback[edge.vert_idx()] = int(i);
            self.fback[edge.face_idx()] = int(i);
        }
    }

    pub fn build_edge(&mut self, ea: usize, eb: usize) -> usize {
        debug_assert_ne!(self.edges[ea].next, int(eb), "already an edge (ea: {}, eb: {})", ea, eb);
        debug_assert_ne!(self.edges[eb].next, int(ea), "already an edge (ea: {}, eb: {})", ea, eb);
        debug_assert_eq!(self.edges[ea].face, self.edges[eb].face, "can't build edge to different face (ea: {}, eb: {})", ea, eb);

        if cfg!(debug_assertions) {
            let mut e = ea;
            while e != eb {
                let next = self.edges[e].next as usize;
                e = next;
                debug_assert_ne!(e, ea, "ea and eb are on different polygons (eb: {})", eb);
            }
        }

        let new_face = int(self.faces.len());

        let id_a = int(self.edges.len() + 0);
        let id_b = int(self.edges.len() + 1);
        // Build two new half edges beween ea.v and eb.v
        let sa = HalfEdge {
            id: id_a,
            adj: id_b,
            next: int(eb),
            v: self.edges[ea].v,
            prev: self.edges[ea].prev,
            face: new_face
        };
        let sb = HalfEdge {
            id: id_b,
            adj: id_a,
            next: int(ea),
            v: self.edges[eb].v,
            prev: self.edges[eb].prev,
            face: self.edges[ea].face // yes, ea
        };

        // let sa = HalfEdge::new(id_a, self.edges[ea].vert_idx(), id_b, eb, self.edges[ea].prev_idx(), new_face);
        // let sb = HalfEdge::new(id_b, self.edges[eb].vert_idx(), id_a, ea, self.edges[eb].prev_idx(), self.edges[ea].face_idx());

        self.edges[sa.prev_idx()].next = sa.id;
        self.edges[sa.next_idx()].prev = sa.id;

        self.edges[sb.prev_idx()].next = sb.id;
        self.edges[sb.next_idx()].prev = sb.id;

        self.edges.push(sa);
        self.edges.push(sb);

        let face = self.faces[self.edges[ea].face_idx()];

        self.faces.push(face);
        if !self.fback.is_empty() {
            self.fback.push(sa.id);
            self.fback[sb.face_idx()] = sb.id;
        }
        {
            let mut e = self.edges[sa.idx()].next_idx();
            while e != sa.idx() {
                debug_assert_ne!(e, ea);
                self.edges[e].face = new_face;
                let next = self.edges[e].next_idx();
                e = next;
            }
        }

        self.debug_assert_valid();
        sa.idx()
    }

    /// Insert new vertex in edge at vpos between edge and edge.adj. Adds two new
    /// edges, one between edge.v and vpos, and one between vpos and edge.adj.v
    pub fn split_edge(&mut self, edge: usize, vpos: V3) {
        let e_adj = self.edges[edge].adj_idx();
        let new_vert_id = int(self.verts.len());

        let s0_id = int(self.edges.len() + 0);
        let sa_id = int(self.edges.len() + 1);

        let s0 = HalfEdge {
            id:  s0_id,
            adj: sa_id,
            prev: int(edge),
            v:    new_vert_id,
            next: self.edges[edge].next,
            face: self.edges[edge].face,
        };

        let sa = HalfEdge {
            id: sa_id,
            adj: s0_id,
            next: self.edges[edge].adj,

            v:    self.edges[e_adj].v,
            prev: self.edges[e_adj].prev,
            face: self.edges[e_adj].face,
        };

        self.edges[s0.prev_idx()].next = s0_id;
        self.edges[s0.next_idx()].prev = s0_id;

        self.edges[sa.prev_idx()].next = sa_id;
        self.edges[sa.next_idx()].prev = sa_id;

        self.edges[e_adj].v = new_vert_id;

        self.edges.push(s0);
        self.edges.push(sa);

        self.verts.push(vpos);

        if !self.vback.is_empty() {
            self.vback.push(s0_id);
            self.vback[sa.vert_idx()] = sa_id;
        }
    }

    /// Split all edges that intersect with plane.
    pub fn split_edges(&mut self, split: Plane) {
        let mut ei = 0;
        while ei < self.edges.len() {
            let ea = self.edges[ei].adj_idx();
            let vi = self.verts[self.edges[ei].vert_idx()];
            let va = self.verts[self.edges[ea].vert_idx()];
            if let Some(vpos) = split.split_line(vi, va) {
                assert_eq!(split.test(vpos), PlaneTestResult::Coplanar);
                self.split_edge(ei, vpos);
            }
            ei += 1;
        }
    }

    pub fn find_or_add_coplanar_edge(&mut self, v: usize, slice: Plane) -> usize {
        let mut e = self.vback[v] as usize;
        {
            let es = e;
            while slice.test(self.verts[self.adj_edge(e).vert_idx()]) != PlaneTestResult::Under {
                e = self.adj_edge(self.prev_edge(e).idx()).idx(); // next ccw edge
                debug_assert_ne!(e, es, "all edges point over...");
            }
        }
        {
            let es = e;
            while slice.test(self.verts[self.adj_edge(e).vert_idx()]) == PlaneTestResult::Under {
                e = self.next_edge(self.adj_edge(e).idx()).idx(); // next cw edge
                debug_assert_ne!(e, es, "all edges point under...");
            }
        }

        let mut ec = self.edges[e].next_idx();

        while slice.test(self.verts[self.edges[ec].vert_idx()]) != PlaneTestResult::Coplanar {
            ec = self.edges[ec].next_idx();
            debug_assert_ne!(ec, e);
        }

        if ec == self.edges[e].next_idx() {
            e
        } else {
            debug_assert_ne!(ec, self.edges[e].prev_idx());
            self.build_edge(e, ec)
        }
    }

    pub fn slice_mesh(&mut self, slice: Plane) -> Vec<usize> {
        self.split_edges(slice);
        let mut result = Vec::new();
        let v0 = {
            let find = self.verts.iter().enumerate().find(|&(_, &v)|
                slice.test(v) == PlaneTestResult::Coplanar);
            match find {
                Some((i, _)) => i,
                None => return result,
            }
        };
        let mut v = v0;
        loop {
            let e = self.find_or_add_coplanar_edge(v, slice);
            v = self.adj_edge(e).vert_idx();
            result.push(e);
            if v == v0 {
                break;
            }
        }
        result
    }

    pub fn pack_slot_edge(&mut self, s: usize) {
        let mut e = self.edges.pop().unwrap();
        if self.edges.len() == s {
            return;
        }
        debug_assert_lt!(s, self.edges.len());
        debug_assert_ge!(e.v, 0);

        if !self.vback.is_empty() && self.vback[e.vert_idx()] == e.id { self.vback[e.vert_idx()] = int(s); }
        if !self.fback.is_empty() && self.fback[e.face_idx()] == e.id { self.fback[e.face_idx()] = int(s); }

        e.id = int(s);
        self.edges[s] = e;
        self.edges[e.next_idx()].prev = e.id;
        self.edges[e.prev_idx()].next = e.id;
        self.edges[e.adj_idx()].adj = e.id;
    }

    pub fn pack_slot_vert(&mut self, s: usize) {
        debug_assert_eq!(self.vback.len(), self.verts.len());
        debug_assert_eq!(self.vback[s], -1);
        let last = self.verts.len() - 1;
        if s == last {
            self.verts.pop();
            self.vback.pop();
            return;
        }
        {
            let vbl = self.vback[last];
            let vel = self.verts[last];
            self.vback[s] = vbl;
            self.verts[s] = vel;
        }
        let mut e = self.vback[s];
        debug_assert_ne!(e, -1);
        loop {
            let eu = e as usize;
            debug_assert_eq!(self.edges[eu].vert_idx(), last);
            self.edges[eu].v = int(s);
            e = self.adj_edge(eu).next;// self.edges[self.edges[eu].adj_idx()].next;
            if e == self.vback[s] {
                break;
            }
        }
        self.verts.pop();
        self.vback.pop();
    }

    pub fn pack_slot_face(&mut self, s: usize) {
        debug_assert_eq!(self.fback.len(), self.faces.len());
        debug_assert_eq!(self.fback[s], -1);
        let last = self.faces.len() - 1;
        if s == last {
            self.faces.pop();
            self.fback.pop();
            return;
        }
        {
            let fbl = self.fback[last];
            let fel = self.faces[last];
            self.fback[s] = fbl;
            self.faces[s] = fel;
        }
        let mut e = self.fback[s];
        debug_assert_ne!(e, -1);
        loop {
            let eu = e as usize;
            debug_assert_eq!(self.edges[eu].face_idx(), last);
            self.edges[eu].face = int(s);
            e = self.edges[eu].next;
            if e == self.fback[s] {
                break;
            }
        }
        self.faces.pop();
        self.fback.pop();
    }

    pub fn swap_faces(&mut self, a: usize, b: usize) {
        self.faces.swap(a, b);
        self.fback.swap(a, b);
        if self.fback[a] != -1 {
            let mut e = self.fback[a];
            loop {
                let eu = e as usize;
                debug_assert_eq!(self.edges[eu].face, int(b));
                self.edges[eu].face = int(a);
                e = self.edges[eu].next;
                if e == self.fback[a] {
                    break;
                }
            }
        }
        if self.fback[b] != -1 {
            let mut e = self.fback[b];
            loop {
                let eu = e as usize;
                debug_assert_eq!(self.edges[eu].face, int(a));
                self.edges[eu].face = int(b);
                e = self.edges[eu].next;
                if e == self.fback[b] {
                    break;
                }
            }
        }
    }

    pub fn pack_faces(&mut self) {
        let face_count = self.faces.len();
        debug_assert_eq!(self.fback.len(), face_count);
        let mut s = 0;
        for i in 0..face_count {
            if self.fback[i] == -1 {
                continue;
            }
            if s < i {
                self.swap_faces(s, i);
                debug_assert_eq!(self.faces.len(), face_count);
            }
            s += 1;
        }
        self.fback.truncate(s);
        self.faces.truncate(s);
    }

    pub fn compress(&mut self) {
        assert_eq!(self.fback.len(), self.faces.len());
        assert_eq!(self.vback.len(), self.verts.len());
        for i in (0..self.edges.len()).rev() {
            if self.edges[i].v == -1 {
                self.pack_slot_edge(i);
            }
        }
        for i in (0..self.vback.len()).rev() {
            if self.vback[i] == -1 {
                self.pack_slot_vert(i);
            }
        }
        self.pack_faces();

        self.is_packed = true;
        self.debug_assert_valid();
        // self.assert_valid();
    }

    pub fn avoid_back_refs(&mut self, eid: usize) {
        let ei = int(eid);

        debug_assert_eq!(self.edges[eid].id, ei);
        debug_assert_ne!(self.edges[eid].prev, ei);

        debug_assert_eq!(self.edges[self.edges[self.edges[eid].prev_idx()].adj_idx()].v,
                         self.edges[eid].v);

        debug_assert_eq!(self.edges[self.edges[eid].prev_idx()].face,
                         self.edges[eid].face);

        if self.vback[self.edges[eid].vert_idx()] == ei {
            let nv = self.edges[self.edges[eid].prev_idx()].adj;
            self.vback[self.edges[eid].vert_idx()] = nv;
        }

        if self.fback[self.edges[eid].face_idx()] == ei {
            let nf = self.edges[eid].prev;
            self.fback[self.edges[eid].face_idx()] = nf;
        }
    }

    pub fn collapse_edge(&mut self, ea: usize, pack: bool) {
        let eb = self.edges[ea].adj_idx();

        assert_ge!(self.edges[ea].v, 0);
        assert_ge!(self.edges[eb].v, 0);

        let eap = self.edges[ea].prev_idx();
        let ean = self.edges[ea].next_idx();

        let ebp = self.edges[eb].prev_idx();
        let ebn = self.edges[eb].next_idx();

        { let eai = self.edges[ea].idx(); self.avoid_back_refs(eai); }
        { let ebi = self.edges[eb].idx(); self.avoid_back_refs(ebi); }

        let old_v = self.edges[ea].v as usize;
        let new_v = self.edges[eb].v as usize;

        assert_eq!(self.edges[ean].v, int(new_v));
        assert_eq!(self.edges[ean].face, self.edges[ea].face);
        assert_eq!(self.edges[ebn].face, self.edges[eb].face);

        if self.vback[new_v] == int(eb) {
            self.vback[new_v] = self.edges[ean].id;
        }
        if self.fback[self.edges[ea].face_idx()] == int(ea) {
            self.fback[self.edges[ea].face_idx()] = self.edges[ean].id;
        }
        if self.fback[self.edges[eb].face_idx()] == int(eb) {
            self.fback[self.edges[eb].face_idx()] = self.edges[ebn].id;
        }

        {
            let mut e = ea;
            loop {
                assert_eq!(self.edges[e].vert_idx(), old_v);
                self.edges[e].v = int(new_v);
                e = self.edges[self.edges[e].adj_idx()].next_idx();
                if e == ea {
                    break;
                }
            }
        }

        { let i = self.edges[ean].id; self.edges[eap].next = i; }
        { let i = self.edges[eap].id; self.edges[ean].prev = i; }
        { let i = self.edges[ebn].id; self.edges[ebp].next = i; }
        { let i = self.edges[ebp].id; self.edges[ebn].prev = i; }

        self.vback[old_v] = -1;

        self.edges[ea].cull();
        self.edges[eb].cull();

        if pack && self.is_packed {
            let edge0 = max!(self.edges[ea].idx(), self.edges[eb].idx());
            self.pack_slot_edge(edge0);

            let edge1 = min!(self.edges[ea].idx(), self.edges[eb].idx());
            self.pack_slot_edge(edge1);

            self.pack_slot_vert(old_v);
        } else {
            self.is_packed = false;
        }

        self.debug_assert_valid();
    }

    pub fn remove_edges(&mut self, to_cull: &[usize]) {
        self.fback.clear();
        self.vback.clear();
        self.is_packed = false;
        for &ea in to_cull.iter() {
            if self.edges[ea].v == -1 {
                continue;
            }
            let eb = self.edges[ea].adj_idx();

            {
                let ii = self.edges[self.edges[eb].next_idx()].id;
                let ui = self.edges[ea].prev_idx();
                self.edges[ui].next = ii;
            }
            {
                let ii = self.edges[self.edges[ea].next_idx()].id;
                let ui = self.edges[eb].prev_idx();
                self.edges[ui].next = ii;
            }
            {
                let ii = self.edges[self.edges[eb].prev_idx()].id;
                let ui = self.edges[ea].next_idx();
                self.edges[ui].prev = ii;
            }
            {
                let ii = self.edges[self.edges[ea].prev_idx()].id;
                let ui = self.edges[eb].next_idx();
                self.edges[ui].prev = ii;
            }

            self.edges[ea].cull();
            self.edges[eb].cull();
        }
        for i in 0..self.edges.len() {
            if self.edges[i].v == -1 {
                continue;
            }
            if self.edges[i].face == self.next_edge(i).face {
                continue;
            }
            let mut e = self.edges[i].next_idx();
            while e != i {
                let face = self.edges[i].face;
                self.edges[e].face = face;
                let n = self.edges[e].next_idx();
                e = n;
            }
        }
        self.init_back_lists();
        self.debug_assert_valid();
        self.compress();
    }

    pub fn crop_to_loop(&mut self, edge_loop: &[usize]) {
        let loop_len = edge_loop.len();
        for i in 0..loop_len {
            // detach verts on loop from edges
            let i1 = (i+1) % loop_len;
            let ec = edge_loop[i];
            let en = edge_loop[i1];

            assert_lt!(ec, self.edges.len());
            assert_lt!(en, self.edges.len());

            let mut e = self.edges[ec].next;

            while e != int(en) {
                let eu = e as usize;
                assert_eq!(self.edges[eu].v, self.edges[en].v);
                self.edges[eu].v = -1;
                e = self.adj_edge(eu).next;
            }

            let eni = int(en);

            self.vback[self.edges[en].vert_idx()] = eni;
            self.fback[self.edges[en].face_idx()] = -1;
            let lf = self.edges[edge_loop[0]].face;
            self.edges[en].face = lf;
        }

        let mut kill_stack = Vec::new();
        for i in 0..loop_len {
            let ec = edge_loop[i];
            let en = edge_loop[(i+1)%loop_len];
            let ep = edge_loop[(i+loop_len-1)%loop_len];
            if self.edges[ec].next_idx() != en {
                let nidx = self.edges[ec].next_idx();
                if self.edges[nidx].id >= 0 {
                    kill_stack.push(nidx);
                }
                self.edges[nidx].id = -1;
                assert_eq!(self.edges[nidx].v, -1);
                assert_eq!(self.edges[nidx].prev, self.edges[ec].id);
                self.edges[nidx].prev = -1;
                self.edges[ec].next = int(en);
            }
            if self.edges[ec].prev_idx() != ep {
                let pidx = self.edges[ec].prev_idx();
                if self.edges[pidx].id >= 0 {
                    kill_stack.push(pidx);
                }
                self.edges[pidx].id = -1;
                assert_eq!(self.edges[pidx].next, self.edges[ec].id);
                self.edges[pidx].next = -1;
                self.edges[ec].prev = int(ep);
            }
        }

        while let Some(k) = kill_stack.pop() {
            assert_eq!(self.edges[k].id, -1);
            if self.edges[k].next != -1 && self.edges[self.edges[k].next_idx()].id != -1 {
                let ei = self.edges[k].next_idx();
                kill_stack.push(ei);
                self.edges[ei].id = -1;
            }
            if self.edges[k].prev != -1 && self.edges[self.edges[k].prev_idx()].id != -1 {
                let ei = self.edges[k].prev_idx();
                kill_stack.push(ei);
                self.edges[ei].id = -1;
            }
            if self.edges[k].adj != -1 && self.edges[self.edges[k].adj_idx()].id  != -1 {
                let ei = self.edges[k].adj_idx();
                kill_stack.push(ei);
                self.edges[ei].id = -1;
            }

            if self.edges[k].v != -1 {
                self.vback[self.edges[k].vert_idx()] = -1;
            }
            if self.edges[k].face != -1 {
                self.fback[self.edges[k].face_idx()] = -1;
            }
            self.edges[k].cull();
        }

        assert_eq!(self.fback[self.edges[edge_loop[0]].face_idx()], -1);

        self.fback[self.edges[edge_loop[0]].face_idx()] = int(edge_loop[0]);

        let face_to_update = self.edges[edge_loop[0]].face_idx();
        self.update_face_plane(face_to_update);
        self.swap_faces(0, face_to_update);
        self.is_packed = false;
        self.debug_assert_valid();
        self.compress();
    }

    pub fn finish(&mut self) {
        self.link_mesh();
        self.init_back_lists();
        self.debug_assert_valid();
    }

    // need to finish() after calling this
    pub fn add_face(&mut self, indices: &[usize]) {
        let mut verts = Vec::with_capacity(indices.len());
        let fid = self.faces.len();
        let base_edge = self.edges.len();
        for (i, &index) in indices.iter().enumerate() {
            verts.push(self.verts[index]);
            self.edges.push(HalfEdge {
                id: int(base_edge + i),
                v: int(index),
                adj: -1,
                next: int(base_edge + (i + 1) % indices.len()),
                prev: int(base_edge + (i + indices.len() - 1) % indices.len()),
                face: int(fid)
            })
        }
        self.faces.push(Plane::from_points(&verts[..]));
    }

    pub fn generate_tris(&self) -> Vec<[u16; 3]> {
        assert!(self.edges.len() >= self.faces.len() * 2,
                "Have more edges than we should (edges = {}, faces = {})",
                self.edges.len(), self.faces.len());

        let mut tris = Vec::with_capacity(
            self.edges.len() - self.faces.len() * 2);

        for &e0 in self.fback.iter() {
            if e0 == -1 {
                continue;
            }
            let e0u = e0 as usize;
            let mut ea = e0u;
            let mut eb = self.edges[ea].next_idx();
            loop {
                ea = eb;
                eb = self.edges[ea].next_idx();
                if eb == e0u {
                    break;
                }
                let a = self.edges[e0u].v;
                let b = self.edges[ea].v;
                let c = self.edges[eb].v;
                assert!(a >= 0 && a < u16::MAX as i32, "Doesn't fit in u16 {}", a);
                assert!(b >= 0 && b < u16::MAX as i32, "Doesn't fit in u16 {}", a);
                assert!(c >= 0 && c < u16::MAX as i32, "Doesn't fit in u16 {}", a);
                tris.push([a as u16, b as u16, c as u16]);
            }
        }
        tris
    }

    pub fn split_test(&self, plane: Plane) -> PlaneTestResult {
        plane.split_test(&self.verts)
    }

    pub fn volume(&self) -> f32 {
        let tris = self.generate_tris();
        geom::volume(&self.verts, &tris)
    }

    pub fn translate(&mut self, t: V3) -> &mut WingMesh {
        for v in self.verts.iter_mut() {
            *v += t;
        }
        for p in self.faces.iter_mut() {
            let r = p.translate(t);
            *p = r;
        }
        self
    }

    pub fn rotate(&mut self, q: Quat) -> &mut WingMesh {
        for v in self.verts.iter_mut() {
            let r = q * *v;
            *v = r;
        }
        for p in self.faces.iter_mut() {
            let r = p.rotate(q);
            *p = r;
        }
        self
    }

    pub fn scale3(&mut self, s: V3) -> &mut WingMesh {
        for v in self.verts.iter_mut() {
            *v *= s;
        }
        for p in self.faces.iter_mut() {
            *p = p.scale3(s);
        }
        self
    }

    pub fn scale(&mut self, s: f32) -> &mut WingMesh {
        self.scale3(V3::splat(s))
    }

    pub fn cropped(&self, slice: Plane) -> WingMesh {
        self.clone().crop(slice)
    }

    pub fn crop(mut self, slice: Plane) -> WingMesh {
        if self.verts.is_empty() || self.faces.is_empty() {
            return self;
        }
        match self.split_test(slice) {
            PlaneTestResult::Over => {
                self.clear();
                return self;
            },
            PlaneTestResult::Under => {
                return self;
            },
            _ => {}
        }

        let coplanar = self.slice_mesh(slice);
        let mut l = Vec::with_capacity(coplanar.len());
        for &c in coplanar.iter().rev() {
            l.push(self.edges[c].adj_idx());
        }
        if !coplanar.is_empty() {
            self.crop_to_loop(&l);
        }
        self.debug_assert_valid();
        debug_assert_gt!(dot(self.faces[0].normal, slice.normal), 0.99);
        self.faces[0] = slice;
        self
    }

    pub fn dual_r(&self, r: f32) -> WingMesh {
        let mut d = WingMesh::new();
        d.faces.reserve(self.verts.len());
        d.fback.reserve(self.vback.len());
        for (i, &v) in self.verts.iter().enumerate() {
            d.faces.push(Plane::new(v.norm_or(0.0, 0.0, 1.0), safe_div0(-r*r, v.length())));
            d.fback.push(self.vback[i]);
        }

        d.verts.reserve(self.faces.len());
        d.vback.reserve(self.fback.len());
        for (i, f) in self.faces.iter().enumerate() {
            d.verts.push(f.normal * safe_div0(-r*r, f.offset));
            d.vback.push(self.edges[self.fback[i] as usize].adj);
        }

        d.edges.reserve(self.edges.len());
        for e in self.edges.iter() {
            d.edges.push(HalfEdge {
                face: e.v,
                v:    self.edges[e.adj_idx()].face,
                next: self.edges[e.prev_idx()].adj,
                prev: self.edges[e.adj_idx()].next,
                .. *e
            });
        }
        d.update_face_planes();
        d.debug_assert_valid();
        d
    }

    fn update_face_planes(&mut self) {
        for f in 0..self.faces.len() {
            self.update_face_plane(f);
        }
    }

    pub fn dual(&self) -> WingMesh {
        self.dual_r(1.0)
    }

    pub fn new_box(min: V3, max: V3) -> WingMesh {
        let mut result = WingMesh::new();
        result.verts = vec![
            vec3(min.x, min.y, min.z),
            vec3(min.x, min.y, max.z),
            vec3(min.x, max.y, min.z),
            vec3(min.x, max.y, max.z),
            vec3(max.x, min.y, min.z),
            vec3(max.x, min.y, max.z),
            vec3(max.x, max.y, min.z),
            vec3(max.x, max.y, max.z)
        ];
        result.faces = vec![
            plane(-1.0, 0.0, 0.0, min.x),
            plane( 1.0, 0.0, 0.0,-max.x),
            plane( 0.0,-1.0, 0.0, min.y),
            plane( 0.0, 1.0, 0.0,-max.y),
            plane( 0.0, 0.0,-1.0, min.z),
            plane( 0.0, 0.0, 1.0,-max.z)
        ];

        result.edges = vec![
            HalfEdge::new( 0,0,11, 1, 3,0), HalfEdge::new( 1,1,23, 2, 0,0),
            HalfEdge::new( 2,3,15, 3, 1,0), HalfEdge::new( 3,2,16, 0, 2,0),
            HalfEdge::new( 4,6,13, 5, 7,1), HalfEdge::new( 5,7,21, 6, 4,1),
            HalfEdge::new( 6,5, 9, 7, 5,1), HalfEdge::new( 7,4,18, 4, 6,1),
            HalfEdge::new( 8,0,19, 9,11,2), HalfEdge::new( 9,4, 6,10, 8,2),
            HalfEdge::new(10,5,20,11, 9,2), HalfEdge::new(11,1, 0, 8,10,2),
            HalfEdge::new(12,3,22,13,15,3), HalfEdge::new(13,7, 4,14,12,3),
            HalfEdge::new(14,6,17,15,13,3), HalfEdge::new(15,2, 2,12,14,3),
            HalfEdge::new(16,0, 3,17,19,4), HalfEdge::new(17,2,14,18,16,4),
            HalfEdge::new(18,6, 7,19,17,4), HalfEdge::new(19,4, 8,16,18,4),
            HalfEdge::new(20,1,10,21,23,5), HalfEdge::new(21,5, 5,22,20,5),
            HalfEdge::new(22,7,12,23,21,5), HalfEdge::new(23,3, 1,20,22,5),
        ];
        result.init_back_lists();
        // The fact that the validity check fails without update_face_planes
        // indicates there's a bug somewhere... I've double-checked that the
        // edge list match references, so it must be in our code elswhere D:
        result.update_face_planes();
        result.debug_assert_valid();
        result
    }

    pub fn faces(&self) -> Vec<bsp::Face> {
        assert!(self.is_packed);
        assert!(self.faces.len() == self.fback.len());
        let mut faces = Vec::with_capacity(self.faces.len());
        for (i, &face) in self.faces.iter().enumerate() {
            let mut f = bsp::Face::new();
            f.plane = face;
            f.vertex = self.face_verts(i);
            faces.push(f);
        }
        faces
    }

    pub fn convex_hit_check(&self, v0: V3, v1: V3) -> Option<geom::HitInfo> {
        let mut test_info = geom::SegmentTestInfo { w0: v0, w1: v1, nw0: V3::zero() };

        for &face in &self.faces {
            if let Some(next) = geom::segment_under(face, test_info.w0, test_info.w1, test_info.nw0) {
                test_info = next;
            } else {
                return None;
                // return geom::HitInfo::new(false, test_info.w0, test_info.nw0);
            }
        }
        Some(geom::HitInfo::new(test_info.w0, test_info.nw0))
    }
}

impl support::Support for WingMesh {
    #[inline]
    fn support(&self, dir: V3) -> V3 {
        self.verts.as_slice().support(dir)
    }
}




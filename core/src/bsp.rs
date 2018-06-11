use math::*;
use wingmesh::WingMesh;
use util::OrdFloat;
use std::{f32, mem};
use gjk;
use support::Support;
const Q_SNAP: f32 = 0.05;
const QUANTIZE_CHECK: f32 = Q_SNAP * (1.0 / 256.0 * 0.5);
const FUZZY_WIDTH: f32 = 100.0*DEFAULT_PLANE_WIDTH;

// const SOLID_BIAS: bool = false;
const ALLOW_AXIAL: u8 = 0b001;
const FACE_TEST_LIMIT: usize = 50;

const OVER: usize = PlaneTestResult::Over as usize;
const UNDER: usize = PlaneTestResult::Under as usize;
const SPLIT: usize = PlaneTestResult::Split as usize;
const COPLANAR: usize = PlaneTestResult::Coplanar as usize;

#[derive(Clone, Default)]
pub struct Face {
    pub plane: Plane,
    pub mat_id: usize,
    pub vertex: Vec<V3>,
    pub gu: V3,
    pub gv: V3,
    pub ot: V3,
}

#[repr(u8)]
#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
pub enum LeafType {
    NotLeaf = 0b00,
    Under = (PlaneTestResult::Under as u8), // 0b01
    Over = (PlaneTestResult::Over as u8), // 0b10
}

impl Default for LeafType {
    #[inline]
    fn default() -> LeafType {
        LeafType::NotLeaf
    }
}

#[derive(Clone, Default)]
pub struct BspNode {
    pub plane: Plane,
    pub under: Option<Box<BspNode>>,
    pub over: Option<Box<BspNode>>,
    pub leaf_type: LeafType,
    pub convex: WingMesh,
    pub boundary: Vec<Face>,
}

#[derive(Clone)]
pub struct BspPreorder<'a> {
    stack: Vec<&'a BspNode>
}

impl<'a> Iterator for BspPreorder<'a> {
    type Item = &'a BspNode;
    fn next(&mut self) -> Option<&'a BspNode> {
        let node = self.stack.pop()?;
        if let Some(ref b) = node.under {
            self.stack.push(b.as_ref()); // is &*b the same?
        }
        if let Some(ref b) = node.over {
            self.stack.push(b.as_ref());
        }
        Some(node)
    }
}

#[derive(Clone)]
pub struct BspBackToFront<'a> {
    pub p: V3,
    stack: Vec<&'a BspNode>,
}

impl<'a> Iterator for BspBackToFront<'a> {
    type Item = &'a BspNode;
    fn next(&mut self) -> Option<&'a BspNode> {
        let node = self.stack.pop()?;
        let plane = Plane::new(self.p, 1.0);
        let mut np = if plane.dot(node.plane) > 0.0 {
            &node.over
        } else {
            &node.under
        };
        while let &Some(ref n) = np {
            self.stack.push(n.as_ref());
            if plane.dot(n.plane) > 0.0 {
                np = &n.over;
            } else {
                np = &n.under;
            }
        }
        Some(node)
    }
}

fn plane_cost_c(input: &[Face], split: Plane, space: &WingMesh) -> (f32, [f32; 4]) {
    let mut counts = [0.0f32, 0.0, 0.0, 0.0];
    for face in input.iter() {
        counts[face.split_test_val(split, FUZZY_WIDTH)] += 1.0;
    }
    if space.verts.is_empty() {
        return (((counts[OVER] - counts[UNDER]).abs() + counts[SPLIT] - counts[COPLANAR]), counts)
    }

    // let mut vol_over = 1.0f32;
    // let mut vol_under = 1.0f32;

    let vol_total = space.volume();
    let space_under = space.cropped(split);
    let space_over = space.cropped(-split);

    let vol_over = space_over.volume();
    let vol_under = space_under.volume();

    debug_assert_ge!(vol_over / vol_total, -0.01);
    debug_assert_ge!(vol_under / vol_total, -0.01);
    // if ((vol_over+vol_under-vol_total) / vol_total).abs() >= 0.01 {
    //     println!("Warning: hacky plane cost calculation is happening");
    //     vol_total = vol_over + vol_under;
    // }

    (vol_over * (counts[OVER] + 1.5*counts[SPLIT]).powf(0.9) +
     vol_under * (counts[UNDER] + 1.5*counts[SPLIT]).powf(0.9),
     counts)
}

fn plane_cost(input: &[Face], split: Plane, space: &WingMesh) -> f32 {
    plane_cost_c(input, split, space).0
}

pub fn compile(faces: Vec<Face>, space: WingMesh) -> Box<BspNode> {
    compile_lt(faces, space, LeafType::NotLeaf)
}

pub fn compile_lt(mut faces: Vec<Face>, space: WingMesh, side: LeafType) -> Box<BspNode> {
    if faces.is_empty() {
        return Box::new(BspNode {
            convex: space,
            leaf_type: side,
            .. BspNode::new(Plane::zero())
        });
    }

    faces.sort_by_key(|a| OrdFloat(a.area()));

    let mut min_val = f32::MAX;

    let mut split = Plane::zero();

    for (i, face) in faces.iter().enumerate() {
        if i > FACE_TEST_LIMIT {
            break;
        }
        let val = plane_cost(&faces, face.plane, &space);
        if val < min_val {
            min_val = val;
            split = faces[i].plane;
        }
    }

    assert!(!split.normal.approx_zero());

    if ALLOW_AXIAL != 0 && faces.len() > 8 {
        for face in faces.iter() {
            for v in face.vertex.iter() {
                for c in 0..3 {
                    let mask = 1u8 << c;
                    if (ALLOW_AXIAL & mask) != 0 {
                        let mut n = V3::zero();
                        n[c] = 1.0;
                        let (val, count) = plane_cost_c(&faces[..], Plane::new(n, -v[c]), &space);
                        if val < min_val && (count[OVER] * count[UNDER] > 0.0 || count[SPLIT] > 0.0) {
                            min_val = val;
                            split = Plane::new(n, -v[c]);
                        }
                    }
                }
            }
        }
    }

    let mut node = Box::new(BspNode::new(split));

    node.convex = space.clone();

    let (under, over, _) = divide_polys(split, faces);

    for face in over.iter() {
        for v in face.vertex.iter() {
            debug_assert_ge!(dot(node.plane.normal, *v) + node.plane.offset, -FUZZY_WIDTH);
        }
    }

    for face in under.iter() {
        for v in face.vertex.iter() {
            debug_assert_le!(dot(node.plane.normal, *v) + node.plane.offset, FUZZY_WIDTH);
        }
    }

    node.under = Some(compile_lt(under, space.cropped(split), LeafType::Under));
    node.over  = Some(compile_lt(over,  space.crop(-split), LeafType::Over));
    node
}

fn gen_faces_rev(wm: &WingMesh, mat: usize, r: &mut Vec<Face>) {
    r.reserve(wm.faces.len());
    for (i, &mf) in wm.faces.iter().enumerate() {
        let mut f = Face::default();
        f.plane = -mf;
        f.mat_id = mat;
        let e0 = wm.fback[i] as usize;
        let mut e = e0;
        loop {
            let vert_idx = wm.edges[e].vert_idx();
            f.vertex.push(wm.verts[vert_idx]);
            e = wm.edges[e].prev_idx();
            if e == e0 {
                break;
            }
        }
        f.assign_tex();
        r.push(f);
    }
}

pub fn divide_polys(split: Plane, input: Vec<Face>) -> (Vec<Face>, Vec<Face>, Vec<Face>) {
    let mut under = Vec::new();
    let mut over = Vec::new();
    let mut coplanar = Vec::new();

    for face in input.into_iter().rev() {
        match face.split_test(split, FUZZY_WIDTH) {
            PlaneTestResult::Coplanar => { coplanar.push(face); },
            PlaneTestResult::Over => { over.push(face); },
            PlaneTestResult::Under => { under.push(face); },
            PlaneTestResult::Split => {
                over.push(face.clipped(-split));
                under.push(face.into_clipped(split));
            },
        }
    }
    (under, over, coplanar)
}

impl BspNode {
    #[inline]
    pub fn new(plane: Plane) -> BspNode {
        BspNode { plane, .. Default::default() }
    }

    #[inline]
    pub fn new_with_type(plane: Plane, leaf_type: LeafType) -> BspNode {
        BspNode { plane, leaf_type, .. Default::default() }
    }

    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.leaf_type != LeafType::NotLeaf
    }

    fn only_planes(&self) -> BspNode {
        BspNode {
            leaf_type: self.leaf_type,
            plane: self.plane,
            under: self.under.as_ref().map(|n| Box::new(n.only_planes())),
            over: self.over.as_ref().map(|n| Box::new(n.only_planes())),
            .. Default::default()
        }
    }

    pub fn count(&self) -> usize {
        1 + self.over.as_ref().map_or(0, |node| node.count())
          + self.under.as_ref().map_or(0, |node| node.count())
    }

    pub fn assign_tex(&mut self, mat_id: usize) -> &mut BspNode {
        for face in self.boundary.iter_mut() {
            face.mat_id = mat_id;
            face.assign_tex();
        }
        if let Some(ref mut u) = self.under {
            u.assign_tex(mat_id);
        }
        if let Some(ref mut o) = self.over {
            o.assign_tex(mat_id);
        }
        self
    }

    pub fn derive_convex(&mut self, m: WingMesh) {
        if !m.edges.is_empty() && !m.verts.is_empty() {
            assert!(!m.verts.is_empty());
            assert!(!m.edges.is_empty());
            assert!(!m.faces.is_empty());
        }
        self.convex = m;
        if self.is_leaf() {
            return;
        }
        let mut cu = WingMesh::new();
        let mut co = WingMesh::new();
        if !self.convex.verts.is_empty() {
            match self.convex.split_test(self.plane) {
                PlaneTestResult::Split => {
                    cu = self.convex.cropped( self.plane);
                    co = self.convex.cropped(-self.plane);
                },
                PlaneTestResult::Over => {
                    co = self.convex.clone();
                },
                PlaneTestResult::Under => {
                    cu = self.convex.clone();
                },
                PlaneTestResult::Coplanar => {
                    unreachable!("has 0 volume somehow?");
                }
            }
        }

        self.under_mut().derive_convex(cu);
        self.over_mut().derive_convex(co);
    }

    pub fn iter(&self) -> BspPreorder {
        BspPreorder { stack: vec![self] }
    }

    pub fn iter_back_to_front(&self, p: V3) -> BspBackToFront {
        BspBackToFront { stack: vec![self], p }
    }

    #[inline]
    fn under(&self) -> &BspNode {
        self.under.as_ref().unwrap()
    }

    #[inline]
    fn over(&self) -> &BspNode {
        self.over.as_ref().unwrap()
    }

    #[inline]
    fn under_mut(&mut self) -> &mut BspNode {
        self.under.as_mut().unwrap()
    }

    #[inline]
    fn over_mut(&mut self) -> &mut BspNode {
        self.over.as_mut().unwrap()
    }

    fn embed_face(&mut self, face: Face) {
        if self.leaf_type == LeafType::Over {
            return;
        }
        if self.leaf_type == LeafType::Under {
            self.boundary.push(face);
            return;
        }
        match face.split_test(self.plane, FUZZY_WIDTH) {
            PlaneTestResult::Under => { self.under_mut().embed_face(face); },
            PlaneTestResult::Over => { self.over_mut().embed_face(face); },
            PlaneTestResult::Coplanar => {
                if dot(self.plane.normal, face.plane.normal) > 0.0 {
                    self.under_mut().embed_face(face);
                } else {
                    self.over_mut().embed_face(face);
                }
            },
            PlaneTestResult::Split => {
                // TODO slice edge here...
                let p = self.plane;
                self.over_mut().embed_face(face.clipped(-p));
                self.under_mut().embed_face(face.into_clipped(p));
            }
        }
    }

    fn face_cutting(&self, faces: &mut Vec<Face>) {
        if self.leaf_type == LeafType::Over {
            return;
        }
        if self.leaf_type == LeafType::Under {
            faces.clear();
            return;
        }
        let mut faces_over = Vec::new();
        let mut faces_under = Vec::new();
        let mut faces_coplanar = Vec::new();

        while let Some(f) = faces.pop() {
            match f.split_test(self.plane, FUZZY_WIDTH) {
                PlaneTestResult::Coplanar => faces_coplanar.push(f),
                PlaneTestResult::Under => faces_under.push(f),
                PlaneTestResult::Over => faces_over.push(f),
                PlaneTestResult::Split => {
                    faces_under.push(f.clipped(self.plane));
                    faces_under.push(f.into_clipped(-self.plane));
                }
            }
        }
        self.under().face_cutting(&mut faces_under);
        self.over().face_cutting(&mut faces_over);
        faces.reserve(faces_under.len() + faces_over.len() + faces_coplanar.len());

        faces.append(&mut faces_under);
        faces.append(&mut faces_over);
        faces.append(&mut faces_coplanar);
    }

    pub fn translate(&mut self, offset: V3) -> &mut BspNode {
        self.plane = self.plane.translate(offset);
        self.convex.translate(offset);
        for face in &mut self.boundary {
            face.translate(offset);
        }
        if let Some(under) = self.under.as_mut() {
            under.translate(offset);
        }
        if let Some(over) = self.over.as_mut() {
            over.translate(offset);
        }
        self
    }

    pub fn rotate(&mut self, rot: Quat) -> &mut BspNode {
        self.plane = self.plane.rotate(rot);
        self.convex.rotate(rot);
        for face in self.boundary.iter_mut() {
            face.rotate(rot);
        }
        if let Some(under) = self.under.as_mut() {
            under.rotate(rot);
        }

        if let Some(over) = self.over.as_mut() {
            over.rotate(rot);
        }

        self
    }

    pub fn scale3(&mut self, s: V3) -> &mut BspNode {
        self.plane = self.plane.scale3(s);
        self.convex.scale3(s);
        for face in self.boundary.iter_mut() {
            face.scale3(s);
        }
        if let Some(under) = self.under.as_mut() {
            under.scale3(s);
        }
        if let Some(over) = self.over.as_mut() {
            over.scale3(s);
        }
        self
    }

    fn each_mut<F: FnMut(&mut BspNode)>(&mut self, mut f: F) {
        let mut stack: Vec<&mut BspNode> = vec![self];
        while let Some(n) = stack.pop() {
            f(n);
            if n.under.is_some() {
                stack.push(&mut *n.under.as_mut().unwrap());
            }
            if n.over.is_some() {
                stack.push(&mut *n.over.as_mut().unwrap())
            }
        }
    }

    fn splitify_edges(&mut self) -> usize {
        let mut split_count = 0;
        let root = self.only_planes();
        self.each_mut(|n| {
            for face in n.boundary.iter_mut() {
                for j in (0..face.vertex.len()).rev() {
                    split_count += face.edge_splicer(j, &root);
                }
            }
        });
        split_count
    }

    fn extract_mat(&mut self, face: &Face) {
        for f in self.boundary.iter_mut() {
            f.extract_mat(face);
        }
        if self.is_leaf() {
            return;
        }
        let f = face.split_test_val(self.plane, FUZZY_WIDTH);
        if f == COPLANAR {
            if dot(self.plane.normal, face.plane.normal) > 0.0 {
                self.under.as_mut().unwrap().extract_mat(face);
            } else {
                self.over.as_mut().unwrap().extract_mat(face);
            }
        } else {
            if (f & UNDER) != 0 {
                self.under.as_mut().unwrap().extract_mat(face);
            }
            if (f & OVER) != 0 {
                self.over.as_mut().unwrap().extract_mat(face);
            }
        }
    }

    pub fn get_solids(&self) -> Vec<&WingMesh> {
        let mut result = Vec::new();
        for n in self.iter() {
            if n.leaf_type == LeafType::Under {
                result.push(&n.convex);
            }
        }
        result
    }

    pub fn rebuild_boundary(&mut self) {
        let boundary = self.rip_boundary();
        self.make_boundary(boundary, 0);
    }

    pub fn make_boundary(&mut self, faces: Vec<Face>, mat_id: usize) {
        self.gen_faces(mat_id);
        self.splitify_edges();
        for face in faces {
            self.extract_mat(&face);
        }
    }

    pub fn rip_boundary(&mut self) -> Vec<Face> {
        let mut out = Vec::new();
        self.each_mut(|n| {
            out.extend(n.boundary.drain(..).rev());
        });
        out
    }

    pub fn negate_tree_planes(&mut self) {
        self.each_mut(|n| {
            for face in n.boundary.iter_mut() {
                face.negate();
            }
            if n.is_leaf() {
                n.leaf_type = if n.leaf_type == LeafType::Under {
                    LeafType::Over
                } else {
                    LeafType::Under
                };
            }
            n.plane = -n.plane;
            mem::swap(&mut n.under, &mut n.over);
        });
    }

    fn gen_faces(&mut self, mat_id: usize) {
        let mut to_embed = Vec::new();
        self.each_mut(|n| {
            if n.leaf_type == LeafType::Over {
                gen_faces_rev(&n.convex, mat_id, &mut to_embed);
            }
        });
        for face in to_embed {
            self.embed_face(face);
        }
    }

    pub fn negate(&mut self) -> &mut BspNode {
        self.negate_tree_planes();
        for f in self.rip_boundary() {
            self.embed_face(f);
        }
        self
    }

    pub fn face_hit(&self, plane: Plane, s: V3) -> Option<&Face> {
        for f in &self.boundary {
            if f.plane == plane || f.plane == -plane {
                continue;
            }
            if f.contains_point(s) {
                return Some(f);
            }
        }
        None
    }

    fn do_hit_check<'tree>(
        &'tree self, v0: V3, v1: V3,
        skip_solid: &mut bool,
        out_info: &mut BspHitInfo<'tree>
    ) -> bool {
        if self.is_leaf() {
            if self.leaf_type == LeafType::Under {
                out_info.impact = v0;
                out_info.leaf = Some(self);
            } else {
                out_info.over_leaf = Some(self);
                *skip_solid = false;
            }
            return self.leaf_type == LeafType::Under && !*skip_solid;
        }
        let f0 = self.plane.test_e(v0, 0.0) == PlaneTestResult::Over;
        let f1 = self.plane.test_e(v1, 0.0) == PlaneTestResult::Over;
        match (f0, f1) {
            (false, false) => {
                self.under().do_hit_check(v0, v1, skip_solid, out_info)
            }
            (true, true) => {
                self.over().do_hit_check(v0, v1, skip_solid, out_info)
            }
            (false, true) => {
                let vmid = self.plane.intersect_with_line(v0, v1);
                let mid_check_u = self.under().do_hit_check(v0, vmid, skip_solid, out_info);
                if mid_check_u {
                    return true;
                }
                out_info.normal = -self.plane.normal;
                out_info.node = self;
                self.over().do_hit_check(vmid, v1, skip_solid, out_info)
            }
            (true, false) => {
                let vmid = self.plane.intersect_with_line(v0, v1);
                let mid_check_o = self.over().do_hit_check(v0, vmid, skip_solid, out_info);
                if mid_check_o {
                    return true;
                }
                out_info.normal = self.plane.normal;
                out_info.node = self;
                self.under().do_hit_check(vmid, v1, skip_solid, out_info)
            }
        }
    }

    pub fn hit_check<'t>(&'t self, v0: V3, v1: V3) -> Option<BspHitInfo<'t>> {
        let mut info = BspHitInfo {
            normal: vec3(1.0, 0.0, 0.0),
            impact: vec3(0.0, 0.0, 0.0),
            vertex_hit: -1,
            node: self,
            leaf: None,
            over_leaf: None
        };
        let mut solid_reenter = false;
        if self.do_hit_check(v0, v1, &mut solid_reenter, &mut info) {
            Some(info)
        } else {
            None
        }
    }

    pub fn hit_check_solid_reenter<'t>(&'t self, v0: V3, v1: V3) -> Option<BspHitInfo<'t>> {
        let mut info = BspHitInfo {
            normal: vec3(1.0, 0.0, 0.0),
            impact: vec3(0.0, 0.0, 0.0),
            vertex_hit: -1,
            node: self,
            leaf: None,
            over_leaf: None
        };
        let mut solid_reenter = true;
        if self.do_hit_check(v0, v1, &mut solid_reenter, &mut info) {
            Some(info)
        } else {
            None
        }
    }

    pub fn hit_check_sphere(&self, r: f32, v0: V3, mut v1: V3, nv0: V3) -> Option<geom::HitInfo> {
        if self.is_leaf() {
            return geom::HitInfo::new_opt(self.leaf_type == LeafType::Under, v0, nv0);
        }
        let mut test_info = geom::SegmentTestInfo { w0: v0, w1: v1, nw0: nv0 };
        let mut hit = false;
        if let Some(ti) = geom::segment_under(self.plane.offset_by(-r), v0, v1, nv0) {
            test_info = ti;
            if let Some(hit_info) = self.under().hit_check_sphere(r, test_info.w0, test_info.w1, test_info.nw0) {
                v1 = hit_info.impact;
                hit = true;
            }
        }

        if let Some(ti) = geom::segment_over(self.plane.offset_by(r), v0, v1, nv0) {
            test_info = ti;
            if let Some(hit_info) = self.over().hit_check_sphere(r, test_info.w0, test_info.w1, test_info.nw0) {
                v1 = hit_info.impact;
                hit = true;
            }
        }

        geom::HitInfo::new_opt(hit, v1, test_info.nw0)
    }

    pub fn hit_check_cylinder(&self, r: f32, h: f32, v0: V3, mut v1: V3, nv0: V3, bevel: bool) -> Option<geom::HitInfo> {
        if self.is_leaf() {
            if bevel && self.leaf_type == LeafType::Under {
                return hit_check_bevel_cylinder(&self.convex, r, h, v0, v1, nv0);
            }
            return geom::HitInfo::new_opt(self.leaf_type == LeafType::Under, v0, nv0);
        }
        let offset_up   = -geom::tangent_point_on_cylinder(r, h, -self.plane.normal).dot(-self.plane.normal);
        let offset_down =  geom::tangent_point_on_cylinder(r, h,  self.plane.normal).dot( self.plane.normal);

        let mut test_info: geom::SegmentTestInfo;
        let mut hit = false;
        let mut norm_hit = nv0;

        if let Some(ti) = geom::segment_under(self.plane.offset_by(offset_up), v0, v1, nv0) {
            test_info = ti;
            if let Some(hit_info) = self.under().hit_check_cylinder(r, h, test_info.w0, test_info.w1, test_info.nw0, bevel) {
                v1 = hit_info.impact;
                norm_hit = hit_info.normal;
                hit = true;
            }
        }
        if let Some(ti) = geom::segment_over(self.plane.offset_by(offset_down), v0, v1, nv0) {
            test_info = ti;
            if let Some(hit_info) = self.over().hit_check_cylinder(r, h, test_info.w0, test_info.w1, test_info.nw0, bevel) {
                v1 = hit_info.impact;
                norm_hit = hit_info.normal;
                hit = true;
            }
        }
        geom::HitInfo::new_opt(hit, v1, norm_hit)
    }

    pub fn all_proximity_cells<'a, S: Support>(
        &'a self,
        collider: &S,
        padding: f32
    ) -> Vec<&'a WingMesh> {
        let mut v = vec![];
        self.proximity_cells(collider, padding, &mut v);
        v
    }

    pub fn proximity_cells<'a, S: Support>(
        &'a self,
        collider: &S,
        padding: f32,
        res: &mut Vec<&'a WingMesh>
    ) -> usize {
        match self.leaf_type {
            LeafType::Over => return 0,
            LeafType::Under => { res.push(&self.convex); return 1; },
            _ => {}
        }
        let start_len = res.len();
        let u = collider.support(-self.plane.normal);
        let o = collider.support( self.plane.normal);
        let t = (self.plane.offset_by(-padding).test_e(u, 0.0) as usize)
              | (self.plane.offset_by( padding).test_e(o, 0.0) as usize);
        if (t & UNDER) != 0 {
            self.under().proximity_cells(collider, padding, res);
        }
        if (t & OVER) != 0 {
            self.over().proximity_cells(collider, padding, res);
        }
        res.len() - start_len
    }

    pub fn hit_check_convex_gjk<S: Support>(&self, collider: &S) -> Option<gjk::ContactInfo> {
        match self.leaf_type {
            LeafType::Over => return None,
            LeafType::Under => {
                let s = gjk::separated(collider, &self.convex, true);
                return if s.separation <= 0.0 { Some(s) } else { None };
            },
            _ => {}
        }
        let t = (self.plane.test_e(collider.support(-self.plane.normal), 0.0) as usize)
              | (self.plane.test_e(collider.support( self.plane.normal), 0.0) as usize);
        if (t & UNDER) != 0 {
            if let Some(hit) = self.under().hit_check_convex_gjk(collider) {
                return Some(hit);
            }
        }
        if (t & OVER) != 0 {
            if let Some(hit) = self.over().hit_check_convex_gjk(collider) {
                return Some(hit);
            }
        }
        None
    }

}

fn hit_check_bevel_cylinder(
    convex: &WingMesh,
    r: f32, h: f32,
    mut v0: V3, mut v1: V3, mut nv0: V3
) -> Option<geom::HitInfo> {
    if convex.edges.len() == 0 {
        return None;
    }
    for (i, edge_0) in convex.edges.iter().enumerate() {
        if i as i32 > edge_0.adj {
            continue;
        }
        let edge_a = &convex.edges[edge_0.adj_idx()];
        let face_0 = convex.faces[edge_0.face_idx()];
        let face_a = convex.faces[edge_a.face_idx()];

        if dot(face_0.normal, face_a.normal) > -0.03 {
            continue;
        }
        let half_angle = (face_0.normal + face_a.normal).must_norm();
        let mut bev = Plane::from_norm_and_point(half_angle, convex.verts[edge_0.vert_idx()]);
        if cfg!(debug_assertions) {
            for &vert in &convex.verts {
                debug_assert_ne!(bev.test_e(vert, DEFAULT_PLANE_WIDTH * 10.0),
                                 PlaneTestResult::Over,
                                 "vert = {:?}, bev = {:?}", vert, bev);
            }
        }
        bev.offset += -dot(geom::tangent_point_on_cylinder(r, h, -bev.normal),
                           -bev.normal);
        if let Some(hit) = geom::segment_under(bev, v0, v1, nv0) {
            v0 = hit.w0;
            v1 = hit.w1;
            match (nv0.dot(nv0) == 0.0, hit.nw0.dot(hit.nw0) == 0.0) {
                (true, true) => nv0 = bev.normal,
                (false, true) => {},
                (_, false) => nv0 = hit.nw0,
            }
        } else {
            return None;
        }
    }
    Some(HitInfo::new(v0, nv0))
}

pub struct BspHitInfo<'tree> {
    pub normal: V3,
    pub impact: V3,
    pub vertex_hit: i32,
    pub node: &'tree BspNode,
    pub leaf: Option<&'tree BspNode>,
    pub over_leaf: Option<&'tree BspNode>,
}

fn do_union(ao: Option<Box<BspNode>>, mut b: Box<BspNode>) -> Box<BspNode> {
    if ao.is_none() || b.leaf_type == LeafType::Under || ao.as_ref().unwrap().leaf_type == LeafType::Over {
        if ao.is_some() && b.leaf_type == LeafType::Under {
            ao.as_ref().unwrap().face_cutting(&mut b.boundary);
        }
        return b;
    }
    let a = ao.unwrap();

    if a.leaf_type == LeafType::Under || b.leaf_type == LeafType::Over {
        return a;
    }

    assert!(!a.is_leaf());
    assert!(!b.is_leaf());
    let (a_under, a_over) = partition(a, b.plane);
    assert!(a_under.is_some() || a_over.is_some());
    b.under = Some(do_union(a_under, b.under.take().unwrap()));
    b.over = Some(do_union(a_over, b.over.take().unwrap()));
    b
}

pub fn union(a: Box<BspNode>, b: Box<BspNode>) -> Box<BspNode> {
    do_union(Some(a), b)
}

fn leaf_or(node: &Option<Box<BspNode>>, l: LeafType) -> LeafType {
    node.as_ref().map(|n| n.leaf_type).unwrap_or(l)
}

fn do_intersect(ao: Option<Box<BspNode>>, mut b: Box<BspNode>) -> Box<BspNode> {
    if b.leaf_type == LeafType::Over || leaf_or(&ao, LeafType::Under) == LeafType::Under {
        if leaf_or(&ao, LeafType::NotLeaf) == LeafType::Under {
            let mut a = ao.unwrap();
            while let Some(f) = a.boundary.pop() {
                b.embed_face(f);
            }
        }
        return b;
    }
    let mut a = ao.unwrap();
    if b.leaf_type == LeafType::Under || a.leaf_type == LeafType::Over {
        if b.leaf_type == LeafType::Under {
            while let Some(f) = b.boundary.pop() {
                a.embed_face(f);
            }
        }
        return a;
    }

    let (a_under, a_over) = partition(a, b.plane);

    let nbu = do_intersect(a_under, b.under.take().unwrap());
    let nbo = do_intersect(a_over,  b.over.take().unwrap());
    b.under = Some(nbu);
    b.over = Some(nbo);
    if same_type_leaf_children(&b) {
        consume_children(&mut b);
    }
    b
//    if nbo.is_leaf() && nbo.leaf_type == nbu.leaf_type {
//        b.boundary.reserve(nbo.boundary.len() + nbu.boundary.len());
//        while let Some(f) = nbo.boundary.pop() {
//            b.boundary.push(f);
//        }
//        while let Some(f) = nbu.boundary.pop() {
//            b.boundary.push(f);
//        }
//        b.leaf_type = nbo.leaf_type;
//        b.under = None;
//        b.over = None;
//    } else {
//        b.under = Some(nbu);
//        b.over = Some(nbo);
//    }
//    b
}

pub fn intersect(a: Box<BspNode>, b: Box<BspNode>) -> Box<BspNode> {
    do_intersect(Some(a), b)
}

pub fn clean(mut n: Box<BspNode>) -> Option<Box<BspNode>> {
    if n.convex.verts.len() == 0 {
        return None;
    }

    if n.is_leaf() {
        n.plane = Plane::zero();
        assert!(n.over.is_none());
        assert!(n.under.is_none());
        return Some(n);
    }

    n.over = n.over.take().and_then(clean);
    n.under = n.under.take().and_then(clean);
    match (n.over.is_some(), n.under.is_some()) {
        (false, false) => None,
        (true, false) => n.over,
        (false, true) => n.under,
        (true, true) => {
            let ltu = n.under.as_ref().unwrap().leaf_type;
            let lto = n.over.as_ref().unwrap().leaf_type;
            if lto != LeafType::NotLeaf && ltu == lto {
                n.leaf_type = lto;
                n.plane = Plane::zero();
                n.over = None;
                n.under = None;
            }
            assert!(!n.convex.verts.is_empty());
            Some(n)
        }
    }
}

fn same_type_leaf_children(bsp: &BspNode) -> bool {
    if bsp.is_leaf() {
        false
    } else {
        let over_type = bsp.over().leaf_type;
        over_type != LeafType::NotLeaf && over_type == bsp.under().leaf_type
    }
}

fn consume_children(bsp: &mut BspNode) {
    let mut under = bsp.under.take().unwrap();
    let mut over = bsp.over.take().unwrap();
    assert_eq!(over.leaf_type, under.leaf_type);

    bsp.leaf_type = over.leaf_type;

    assert_eq!(bsp.boundary.len(), 0);
    bsp.boundary.reserve(under.boundary.len() + over.boundary.len());

    bsp.boundary.extend(under.boundary.drain(..).rev());
    bsp.boundary.extend(over.boundary.drain(..).rev());
}

mod part {
    use super::*;

    fn delinearize(mut bsp: Box<BspNode>) -> Box<BspNode> {
        assert!(bsp.over.is_some() || bsp.under.is_some());
        if bsp.under.is_some() && bsp.over.is_some() {
            return bsp;
        }
        let result = if bsp.under.is_none() {
            bsp.over.take().unwrap()
        } else {
            assert!(bsp.over.is_none());
            bsp.under.take().unwrap()
        };
        result
    }


    pub fn partition(mut bsp: Box<BspNode>, split_plane: Plane)
        -> (Option<Box<BspNode>>, Option<Box<BspNode>>)
    {
        match bsp.convex.split_test(split_plane) {
            PlaneTestResult::Under => {
                return (Some(bsp), None);
            },
            PlaneTestResult::Over => {
                return (None, Some(bsp));
            },
            PlaneTestResult::Split => {},
            PlaneTestResult::Coplanar => {
                unreachable!("convex shape is flat?");
            }
        }

        let mut under = Box::new(BspNode::new_with_type(bsp.plane, bsp.leaf_type));
        let mut over = Box::new(BspNode::new_with_type(bsp.plane, bsp.leaf_type));

        under.convex = bsp.convex.cropped(split_plane);
        over.convex = bsp.convex.cropped(-split_plane);

        if bsp.leaf_type == LeafType::Under {
            let mut fake = BspNode {
                under: Some(under),
                over: Some(over),
                .. BspNode::new(split_plane)
            };
            let mut e = Vec::new();
            e.append(&mut bsp.boundary);
            for face in e.into_iter().rev() {
                fake.embed_face(face)
            }
            assert!(fake.under.is_some());
            assert!(fake.over.is_some());
            under = fake.under.take().unwrap();
            over = fake.over.take().unwrap();
        }

        if let Some(n_under) = bsp.under.take() {
            let (uu, ou) = partition(n_under, split_plane);
            under.under = uu;
            over.under = ou;
        }

        if let Some(n_over) = bsp.over.take() {
            let (uo, oo) = partition(n_over, split_plane);
            under.over = uo;
            over.over = oo;
        }

        if bsp.is_leaf() {
            assert!(under.is_leaf());
            assert!(over.is_leaf());
            return (Some(under), Some(over));
        }


        under = delinearize(under);
        over = delinearize(over);

        assert!(under.is_leaf() || (under.under.is_some() && under.over.is_some()));
        assert!(over.is_leaf() || (over.under.is_some() && over.over.is_some()));

        if same_type_leaf_children(&under) {
            consume_children(&mut under);
        }

        if same_type_leaf_children(&over) {
            consume_children(&mut over);
        }

        (Some(under), Some(over))
    }
}

pub use self::part::partition;

impl Face {
    pub fn new() -> Face { Default::default() }

    pub fn new_quad(v0: V3, v1: V3, v2: V3, v3: V3) -> Face {
        let mut f = Face::new();
        f.vertex = vec![v0, v1, v2, v3];
        let norm = (cross(v1-v0, v2-v1) + cross(v3-v2, v0-v3)).norm_or_unit();
        f.plane = Plane::from_norm_and_point(norm, (v0+v1+v2+v3)*0.25);
        for v in f.vertex.iter() {
            debug_assert_eq!(f.plane.test(*v), PlaneTestResult::Coplanar);
        }
        f.extract_mat_vals(v0, v1, v3, vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0));
        f.gu = f.gu.norm_or_unit();
        f.gv = f.gv.norm_or_v(cross(f.plane.normal, f.gu).norm_or_unit());
        f.ot = V3::zero();
        f
    }

    pub fn new_tri(v0: V3, v1: V3, v2: V3) -> Face {
        let mut f = Face::new();
        f.vertex = vec![v0, v1, v2];
        f.plane = Plane::from_norm_and_point(cross(v1-v0, v2-v1).norm_or_unit(), (v0+v1+v2)*0.25);
        f.gu = (v1-v0).norm_or_unit();
        f.gv = cross(f.plane.normal, f.gu).norm_or_unit();
        f
    }

    pub fn new_tri_tex(v0: V3, v1: V3, v2: V3, t0: V2, t1: V2, t2: V2) -> Face {
        let mut f = Face::new();
        f.vertex = vec![v0, v1, v2];
        f.plane = Plane::from_norm_and_point(cross(v1-v0, v2-v1).norm_or_unit(), (v0+v1+v2)*0.25);
        f.extract_mat_vals(v0, v1, v2, t0, t1, t2);
        f
    }

    pub fn extract_mat_vals(&mut self, v0: V3, v1: V3, v2: V3, t0: V2, t1: V2, t2: V2) {
        self.gu = geom::gradient(v0, v1, v2, t0.x, t1.x, t2.x);
        self.gv = geom::gradient(v0, v1, v2, t0.y, t1.y, t2.y);

        self.ot.x = t0.x - dot(v0, self.gu);
        self.ot.y = t0.y - dot(v0, self.gv);
    }

    pub fn area(&self) -> f32 {
        let mut area = 0.0;
        for i in 2..self.vertex.len() {
            let vb = self.vertex[0];
            let v1 = self.vertex[i - 1];
            let v2 = self.vertex[i];
            area += dot(self.plane.normal, cross(v1 - vb, v2 - v1)) * 0.5;
        }
        area
    }

    pub fn center(&self) -> V3 {
        self.vertex.iter().fold(V3::zero(), |a, &b| a + b) *
            safe_div0(1.0, self.vertex.len() as f32)
    }

    pub fn split_test(&self, plane: Plane, e: f32) -> PlaneTestResult {
        plane.split_test_e(&self.vertex, e)
    }

    pub fn split_test_val(&self, plane: Plane, e: f32) -> usize {
        plane.split_test_val_e(&self.vertex[..], e)
    }

    pub fn translate(&mut self, offset: V3) {
        self.plane.translate(offset);
        for v in self.vertex.iter_mut() {
            *v += offset;
        }
        self.ot.x -= dot(offset, self.gu);
        self.ot.y -= dot(offset, self.gv);
    }

    pub fn rotate(&mut self, rot: Quat) {
        self.plane.rotate(rot);
        for v in self.vertex.iter_mut() {
            let r = rot * *v;
            *v = r;
        }
        self.gu = rot * self.gu;
        self.gv = rot * self.gv;
        self.ot = rot * self.ot;
    }

    pub fn scale3(&mut self, s: V3) {
        self.plane.scale3(s);
        for v in self.vertex.iter_mut() {
            let r = s * *v;
            *v = r;
        }
    }

    // point must be interior
    pub fn closest_edge(&self, point: V3) -> usize {
        assert_ge!(self.vertex.len(), 3);
        let mut closest = -1;
        let mut min_d = 0.0;
        for (i, &v0) in self.vertex.iter().enumerate() {
            let i1 = (i+1)% self.vertex.len();
            let v1 = self.vertex[i1];
            let d = line_project(v0, v1, point).dist(point);
            if closest == -1 || d < min_d {
                closest = i as isize;
                min_d = d;
            }
        }
        assert_ge!(closest, 0);
        closest as usize
    }

    pub fn contains_point(&self, s: V3) -> bool {
        for (i, &pp1) in self.vertex.iter().enumerate() {
            let pp2 = self.vertex[(i+1)%self.vertex.len()];
            let side = cross(pp2-pp1, s-pp1);
            if dot(self.plane.normal, side) < 0.0 {
                return false;
            }
        }
        true
    }

    #[inline]
    pub fn vert_uv(&self, i: usize) -> V2 {
        vec2(self.ot.x + dot(self.vertex[i], self.gu),
             self.ot.y + dot(self.vertex[i], self.gv))
    }

    #[inline]
    pub fn uv_at(&self, v: V3) -> V2 {
        vec2(self.ot.x + dot(v, self.gu), self.ot.y + dot(v, self.gv))
    }

    pub fn assign_tex(&mut self) {
        let n = self.plane.normal;
        if n.x.abs() > n.y.abs() && n.x.abs() > n.z.abs() {
            self.gu = vec3(0.0, n.x.signum(), 0.0);
            self.gv = vec3(0.0, 0.0, 1.0);
        } else if n.y.abs() > n.z.abs() {
            self.gu = vec3(-n.y.signum(), 0.0, 0.0);
            self.gv = vec3(0.0, 0.0, 1.0);
        } else {
            self.gu = vec3(1.0, 0.0, 0.0);
            self.gv = vec3(0.0, n.z.signum(), 0.0);
        }
    }

    fn edge_splicer(&mut self, vi0: usize, n: &BspNode) -> usize {
        if n.is_leaf() {
            return 0;
        }
        let mut split_count = 0;
        let vi1 = (vi0 + 1) % self.vertex.len();
        let v0 = self.vertex[vi0];
        let v1 = self.vertex[vi1];
        if v0.dist(v1) <= QUANTIZE_CHECK {
            split_count += 1;
        }
        debug_assert_gt!(v0.dist(v1), QUANTIZE_CHECK);
        let f0 = n.plane.test_e(v0, QUANTIZE_CHECK);
        let f1 = n.plane.test_e(v1, QUANTIZE_CHECK);
        match (f0 as usize)|(f1 as usize) {
            COPLANAR => {
                let count = self.vertex.len();
                split_count += self.edge_splicer(vi0, n.under.as_ref().unwrap());
                let mut k = vi0 + (self.vertex.len() - count);
                while k >= vi0 {
                    split_count += self.edge_splicer(k, n.over.as_ref().unwrap());
                    if k == 0 {
                        break;
                    }
                    k -= 1;
                }
            },
            UNDER => {
                split_count += self.edge_splicer(vi0, n.under.as_ref().unwrap());
            },
            OVER => {
                split_count += self.edge_splicer(vi0, n.over.as_ref().unwrap())
            },
            SPLIT => {
                split_count += 1;
                assert_gt!(v0.dist(v1), QUANTIZE_CHECK);
                let v_mid = n.plane.intersect_with_line(v0, v1);
                assert_gt!(v_mid.dist(v1), QUANTIZE_CHECK);
                assert_gt!(v0.dist(v_mid), QUANTIZE_CHECK);
                assert_eq!(n.plane.test(v_mid), PlaneTestResult::Coplanar);

                self.vertex.insert(vi0 + 1, v_mid);
            },
            _ => {
                unreachable!("Bad plane test result combination? {}", (f0 as usize)|(f1 as usize));
            }
        }
        split_count
    }

    pub fn clipped(&self, clip: Plane) -> Face {
        self.clone().into_clipped(clip)
    }

    pub fn into_clipped(mut self, clip: Plane) -> Face {
        debug_assert_eq!(self.split_test(clip, FUZZY_WIDTH), PlaneTestResult::Split);
        self.slice(clip);
        self.vertex.retain(|&v| clip.test(v) != PlaneTestResult::Over);
        self
    }

    pub fn slice(&mut self, clip: Plane) -> usize {
        let mut c = 0;
        let mut i = 0;
        while i < self.vertex.len() {
            let i2 = (i+1) % self.vertex.len();
            match (clip.test(self.vertex[i]), clip.test(self.vertex[i2])) {
                (PlaneTestResult::Over, PlaneTestResult::Under) |
                (PlaneTestResult::Under, PlaneTestResult::Over) => {
                    let v_mid = clip.intersect_with_line(self.vertex[i], self.vertex[i2]);
                    assert_eq!(clip.test(v_mid), PlaneTestResult::Coplanar);
                    self.vertex.insert(i2, v_mid);
                    i = 0;
                    assert_lt!(c, 2);
                    c += 1;
                },
                _ => {}
            }
            i += 1;
        }
        c
    }

    pub fn negate(&mut self) {
        self.plane = -self.plane;
        self.vertex.reverse();
    }

    fn extract_mat(&mut self, face: &Face) {
        if dot(self.plane.normal, face.plane.normal) < 0.95 {
            return;
        }
        if self.split_test(face.plane, DEFAULT_PLANE_WIDTH) != PlaneTestResult::Coplanar {
            return;
        }
        let interior = face.center();
        if geom::poly_hit_check(&face.vertex, interior + face.plane.normal, interior - face.plane.normal).is_none() {
            return;
        }

        self.mat_id = face.mat_id;
        self.gu = face.gu;
        self.ot = face.ot;
    }

    pub fn gen_tris(&self) -> Vec<[u16; 3]> {
        let mut tris = Vec::with_capacity(self.vertex.len()-2);
        for i in 1..self.vertex.len()-1 {
            tris.push([0, i as u16, (i + 1) as u16]);
        }
        tris
    }
}

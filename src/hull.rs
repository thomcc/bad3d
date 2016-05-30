use math::*;
use math::geom::{Plane, tri_normal};
use std::ops::*;
use std::{isize, i32, f32, usize};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct I3 {
    pub at: [i32; 3],
}

#[inline]
fn int3(a: i32, b: i32, c: i32) -> I3 {
    I3 { at: [a, b, c] }
}

impl Index<usize> for I3 {
    type Output = i32;
    #[inline]
    fn index(&self, i: usize) -> &i32 {
        &self.at[i]
    }
}

impl IndexMut<usize> for I3 {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut i32 {
        &mut self.at[i]
    }
}

impl I3 {
    #[inline]
    fn rev(self) -> I3 {
        int3(self[2], self[1], self[0])
    }

    #[inline]
    fn roll(self) -> I3 {
        int3(self[1], self[2], self[0])
    }

    #[inline]
    fn is(self, o: I3) -> bool {
        self == o || self == o.roll() || self.roll() == o
    }

    #[inline]
    fn back_to_back(self, o: I3) -> bool {
        self.is(o.rev())
    }

    #[inline]
    fn has_edge(self, a: i32, b: i32) -> bool {
        (self[0] == a && self[1] == b) ||
        (self[1] == a && self[2] == b) ||
        (self[2] == a && self[0] == b)
    }

    #[inline]
    fn has_vert(self, a: i32) -> bool {
        self[0] == a || self[1] == a || self[2] == a
    }

    fn share_edge(self, o: I3) -> bool {
        self.has_edge(o[1], o[0]) ||
        self.has_edge(o[2], o[1]) ||
        self.has_edge(o[0], o[2])
    }
}

#[inline]
fn tri(verts: &[V3], t: I3) -> (V3, V3, V3) {
    (verts[t[0] as usize], verts[t[1] as usize], verts[t[2] as usize])
}

#[inline]
fn tri_ref(verts: &[V3], t: I3) -> (&V3, &V3, &V3) {
    (&verts[t[0] as usize], &verts[t[1] as usize], &verts[t[2] as usize])
}

fn above(verts: &[V3], t: I3, p: V3, epsilon: f32) -> bool {
    let (v0, v1, v2) = tri(verts, t);
    dot(tri_normal(v0, v1, v2), p - v0) > epsilon
}

struct Tri {
    vi: I3,
    ni: I3,
    id: i32,
    max_v: i32,
    rise: f32
}

fn next_mod3(i: usize) -> (usize, usize) {
    debug_assert!(i < 3);
    ((i + 1) % 3, (i + 2) % 3)
}

impl Tri {
    fn new(a: i32, b: i32, c: i32, id: i32, n: I3) -> Tri {
        Tri {
            vi: int3(a, b, c),
            ni: n,
            id: id,
            max_v: -1,
            rise: 0.0,
        }
    }

    fn new2(a: i32, b: i32, c: i32, id: i32) -> Tri {
        Tri::new(a, b, c, id, int3(-1, -1, -1))
    }

    fn dead(&self) -> bool {
        self.ni[0] == -1
    }

    fn neib_idx(&self, va: i32, vb: i32) -> i32 {
        for i in 0..3 {
            let (i1, i2) = next_mod3(i);
            if (self.vi[i] == va && self.vi[i1] == vb) ||
               (self.vi[i] == vb && self.vi[i1] == va) {
                return i2 as i32
            }
        }
        debug_assert!(false,
                      "Fell through neib loop v={:?} n={:?} va={} vb={}",
                      self.vi, self.ni, va, vb);
        -1
    }

    fn get_neib(&self, va: i32, vb: i32) -> i32 {
        let idx = self.neib_idx(va, vb) as usize;
        assert!(idx < 3);
        self.ni[idx]
    }

    fn set_neib(&mut self, va: i32, vb: i32, new_value: i32) {
        let idx = self.neib_idx(va, vb) as usize;
        assert!(idx < 3);
        self.ni[idx] = new_value;
    }

}

fn neib_fix(tris: &mut[Tri], k_id: i32) {
    let k = k_id as usize;
    if tris[k].id == -1 {
        return;
    }
    debug_assert!(tris[k].id == k_id);
    for i in 0..3 {
        let (i1, i2) = next_mod3(i);
        if tris[k].ni[i] != -1 {
            let va = tris[k].vi[i2];
            let vb = tris[k].vi[i1];
            let ni = tris[k].ni[i] as usize;
            let nni = tris[ni as usize].neib_idx(va, vb);
            if nni != -1 {
                tris[ni].ni[nni as usize] = k_id;
            }
        }
    }
}

fn swap_neib(tris: &mut[Tri], a: usize, b: usize) {
    tris.swap(a, b);
    {
        let id = tris[a].id;
        tris[a].id = tris[b].id;
        tris[b].id = id;
    }
    neib_fix(tris, a as i32);
    neib_fix(tris, b as i32);
}


fn fix_back_to_back(tris: &mut[Tri], s: usize, t: usize) {
    for i in 0..3 {
        let (i1, i2) = next_mod3(i);
        let (va, vb) = (tris[s].vi[i1], tris[s].vi[i2]);
        debug_assert!(tris[tris[s].get_neib(va, vb) as usize].get_neib(vb, va)
                      == tris[s].id);
        debug_assert!(tris[tris[t].get_neib(va, vb) as usize].get_neib(vb, va)
                      == tris[t].id);

        let t_neib = tris[t].get_neib(vb, va);
        tris[tris[s].get_neib(va, vb) as usize].set_neib(vb, va, t_neib);

        let s_neib = tris[s].get_neib(va, vb);
        tris[tris[t].get_neib(vb, va) as usize].set_neib(va, vb, s_neib);
    }
    // cleaned up later
    tris[s].ni = int3(-1, -1, -1);
    tris[t].ni = int3(-1, -1, -1);
}

fn check_tri(tris: &[Tri], t: &Tri) {
    debug_assert!(tris[t.id as usize].id == t.id);
    debug_assert!(tris[t.id as usize].id == t.id);
    for i in 0..3 {
        let (i1, i2) = next_mod3(i);
        let (a, b) = (t.vi[i1], t.vi[i2]);
        debug_assert!(a != b);
        debug_assert!(tris[t.ni[i] as usize].get_neib(b, a) == t.id);
    }
}

fn extrude(tris: &mut Vec<Tri>, t0: usize, v: usize) {
    let bu = tris.len();
    let b = bu as i32;

    let n = tris[t0].ni;
    let t = tris[t0].vi;

    let vi = v as i32;

    tris.push(Tri::new(vi, t[1], t[2], b+0, int3(n[0], b+1, b+2)));
    tris[n[0] as usize].set_neib(t[1], t[2], b+0);

    tris.push(Tri::new(vi, t[2], t[0], b+1, int3(n[1], b+2, b+0)));
    tris[n[1] as usize].set_neib(t[2], t[0], b+1);

    tris.push(Tri::new(vi, t[0], t[1], b+2, int3(n[2], b+0, b+1)));
    tris[n[2] as usize].set_neib(t[0], t[1], b+2);

    tris[t0].ni = int3(-1, -1, -1);

    // @@TODO: disable in debug?
    check_tri(&tris[..], &tris[bu + 0]);
    check_tri(&tris[..], &tris[bu + 1]);
    check_tri(&tris[..], &tris[bu + 2]);

    if tris[n[0] as usize].vi.has_vert(vi) { fix_back_to_back(&mut tris[..], bu+0, n[0] as usize); }
    if tris[n[1] as usize].vi.has_vert(vi) { fix_back_to_back(&mut tris[..], bu+1, n[1] as usize); }
    if tris[n[2] as usize].vi.has_vert(vi) { fix_back_to_back(&mut tris[..], bu+2, n[2] as usize); }
}

fn find_extrudable(tris: &[Tri], epsilon: f32) -> Option<usize> {
    assert!(tris.len() > 0);
    let mut best: Option<usize> = None;
    for (idx, tri) in tris.iter().enumerate() {
        debug_assert!(tri.id >= 0);
        debug_assert!(tri.id == (idx as i32));
        debug_assert!(!tri.dead());
        if best.is_none() || tris[best.unwrap()].rise < tri.rise {
            best = Some(idx)
        }
    }
    if tris[best.unwrap()].rise > epsilon { best } else { None }
}

fn find_simplex(verts: &[V3]) -> Option<(usize, usize, usize, usize)> {
    let b0 = vec3(0.01, 0.02, 1.0);

    let p0 = try_opt!(max_dir(verts,  b0));
    let p1 = try_opt!(max_dir(verts, -b0));

    let b0 = verts[p0] - verts[p1];

    if p0 == p1 || approx_zero(dot(b0, b0)) {
        return None;
    }

    let b1 = cross(vec3(1.0, 0.0, 0.0), b0);
    let b2 = cross(vec3(0.0, 1.0, 0.0), b0);

    let b1 = try_opt!(if b1.length_sq() > b2.length_sq() { b1 }
                      else { b2 }.normalize());

    let p2 = try_opt!(max_dir(verts, b1));

    let p2 = if p2 == p0 || p2 == p1 { try_opt!(max_dir(verts, -b1)) }
             else { p2 };

    if p2 == p0 || p2 == p1 {
        return None;
    }

    let b1 = verts[p2] - verts[p0];
    let b2 = cross(b1, b2);

    let p3 = try_opt!(max_dir(verts, b2));

    let p3 = if p3 == p0 || p3 == p1 || p3 == p2 { try_opt!(max_dir(verts, -b2)) }
             else { p3 };

    if p3 == p0 || p3 == p1 || p3 == p2 {
        return None;
    }

    debug_assert!(p0 != p1 && p0 != p2 && p0 != p3 &&
                  p1 != p2 && p1 != p3 && p2 != p3);

    if dot(verts[p3] - verts[p0],
           cross(verts[p1] - verts[p0], verts[p2] - verts[p0])) < 0.0 {
        Some((p0, p1, p3, p2))
    } else {
        Some((p0, p1, p2, p3))
    }
}


pub fn get_sep_plane_epa<F: Fn(V3) -> V3>(init_verts: [V3; 4], max_dir: F) -> Plane {
    let mut plane = Plane::new(V3::zero(), -f32::MAX);
    let epsilon = 0.001f32; // ...

    let mut verts = Vec::from(&init_verts[..]);
    let mut tris = Vec::new();

    let iv0 = init_verts[0];
    let iv1 = init_verts[1];
    let iv2 = init_verts[2];
    let iv3 = init_verts[3];
    if dot(cross(iv2 - iv0, iv1 - iv0), iv3 - iv0) > 0.0 {
        verts.swap(2, 3);
    }

    let center = (iv0 + iv1 + iv2 + iv3) * 0.25;
    tris.push(Tri::new(2_i32, 3_i32, 1_i32, 0, int3(2, 3, 1)));
    tris.push(Tri::new(3_i32, 2_i32, 0_i32, 1, int3(3, 2, 0)));
    tris.push(Tri::new(0_i32, 1_i32, 3_i32, 2, int3(0, 1, 3)));
    tris.push(Tri::new(1_i32, 0_i32, 2_i32, 3, int3(1, 0, 2)));

    loop {
        let mut face = Plane::new(V3::zero(), -f32::MAX);
        for t in tris.iter() {
            debug_assert!(t.id >= 0);
            debug_assert!(t.max_v < 0);
            let (v0, v1, v2) = tri(&verts[..], t.vi);
            let tri_plane = Plane::from_tri(v0, v1, v2);
            if tri_plane.offset > face.offset {
                face = tri_plane;
            }
        }
        let v = max_dir(face.normal);
        let p = Plane::from_norm_and_point(face.normal, v);

        if p.offset > plane.offset {
            plane = p;
        }

        if let Some(_) = verts.iter().find(|e| **e == v) {
            return plane;
        }

        if plane.offset >= face.offset - epsilon {
            return plane;
        }

        let vid = verts.len();
        verts.push(v);
        for j in (0..tris.len()).rev() {
            if tris[j].dead() { continue; }
            let t = tris[j].vi;
            if above(&verts[..], t, verts[vid], 0.01*epsilon) {
                extrude(&mut tris, j, vid)
            }
        }

        {
            let mut ii: isize = tris.len() as isize;
            loop {
                ii -= 1;
                if ii < 0 { break; }
                let iu = ii as usize;
                if tris[iu].dead() { continue; }
                if !tris[iu].vi.has_vert(vid as i32) { break; }

                let nt = tris[iu].vi;
                let (nv0, nv1, nv2) = tri(&verts[..], nt);
                if above(&verts[..], nt, center, 0.01*epsilon) ||
                        cross(nv1-nv0, nv2-nv0).length() < epsilon*epsilon*0.1 {
                    debug_assert!(tris[iu].ni[0] >= 0);
                    let nb = tris[iu].ni[0] as usize;
                    debug_assert!(!tris[nb].dead());
                    debug_assert!(!tris[nb].vi.has_vert(vid as i32));
                    debug_assert!(tris[nb].id < (ii as i32));
                    extrude(&mut tris, nb, vid);
                    ii = tris.len() as isize;
                }
            }
        }

        for j in (0..tris.len()).rev() {
            if !tris[j].dead() { continue; }

            // remove dead
            for j in (0..tris.len()).rev() {
                if !tris[j].dead() {
                    continue;
                }
                let last = tris.len()-1;
                swap_neib(&mut tris[..], j, last);
                tris.pop();
            }
        }
    }
}

// pub fn compute_hull(verts: &mut [V3]) -> Vec<[i32; 3]> {
//     compute_hull_bounded(verts, 0)
// }

pub fn compute_hull_bounded(verts: &mut [V3], vert_limit: usize) -> Option<Vec<[i32; 3]>> {
    assert!(verts.len() > 4);
    assert!(verts.len() < (i32::MAX as usize));

    let mut vert_limit: isize = if vert_limit == 0 { isize::MAX } else { vert_limit as isize };

    let vert_count = verts.len();

    let mut is_extreme = vec![false; vert_count];
    let (min_bound, max_bound) = try_opt!(compute_bounds(verts));

    let epsilon = max_bound.distance(min_bound) * 0.001_f32;

    let (p0, p1, p2, p3) = try_opt!(find_simplex(verts));

    let mut tris = Vec::new();

    let center = (verts[p0] + verts[p1] + verts[p2] + verts[p3]) * 0.25;


    tris.push(Tri::new(p2 as i32, p3 as i32, p1 as i32, 0, int3(2, 3, 1)));
    tris.push(Tri::new(p3 as i32, p2 as i32, p0 as i32, 1, int3(3, 2, 0)));
    tris.push(Tri::new(p0 as i32, p1 as i32, p3 as i32, 2, int3(0, 1, 3)));
    tris.push(Tri::new(p1 as i32, p0 as i32, p2 as i32, 3, int3(1, 0, 2)));

    is_extreme[p0] = true;
    is_extreme[p1] = true;
    is_extreme[p2] = true;
    is_extreme[p3] = true;

    check_tri(&tris[..], &tris[0]);
    check_tri(&tris[..], &tris[1]);
    check_tri(&tris[..], &tris[2]);
    check_tri(&tris[..], &tris[3]);

    for t in &mut tris {
        debug_assert!(t.id >= 0);
        debug_assert!(t.max_v < 0);
        let (v0, v1, v2) = tri(&verts[..], t.vi);
        let n = geom::tri_normal(v0, v1, v2);
        let vmax = max_dir(&verts[..], n).unwrap();
        t.max_v = vmax as i32;
        t.rise = dot(n, verts[vmax] - v0);
    }

    vert_limit -= 4;

    while vert_limit > 0 {
        let te = match find_extrudable(&tris[..], epsilon) {
            Some(e) => e,
            None => break,
        };

        let v = tris[te].max_v as usize;
        debug_assert!(!is_extreme[v]);

        is_extreme[v] = true;
        // wherever we find a vertex 'above' a face, extrude.
        for j in (0..tris.len()).rev() {
            if tris[j].dead() {
                continue;
            }
            let t = tris[j].vi;
            if above(&verts[..], t, verts[v], 0.01*epsilon) {
                extrude(&mut tris, j, v)
            }
        }

        // degenerate case fixup
        {
            let mut ii: isize = tris.len() as isize;
            loop {
                ii -= 1;
                if ii < 0 { break; }
                let iu = ii as usize;
                if tris[iu].dead() { continue; }
                if !tris[iu].vi.has_vert(v as i32) { break; }

                let nt = tris[iu].vi;
                if above(&verts[..], nt, center, 0.01*epsilon) {
                    debug_assert!(tris[iu].ni[0] >= 0);
                    let nb = tris[iu].ni[0] as usize;
                    debug_assert!(!tris[nb].dead());
                    debug_assert!(!tris[nb].vi.has_vert(v as i32));
                    debug_assert!(tris[nb].id < (ii as i32));
                    extrude(&mut tris, nb, v);
                    ii = tris.len() as isize;
                }
            }
        }

        for j in (0..tris.len()).rev() {
            if tris[j].dead() { continue; }
            if tris[j].max_v >= 0 { break; }
            let (v0, v1, v2) = tri(&verts[..], tris[j].vi);
            let n = geom::tri_normal(v0, v1, v2);
            let vmax = max_dir(&verts[..], n).unwrap();
            tris[j].max_v = vmax as i32;
            if is_extreme[vmax] {
                tris[j].max_v = -1; // already did this one
            } else {
                tris[j].rise = dot(n, verts[vmax] - v0);
            }
        }

        // remove dead
        for j in (0..tris.len()).rev() {
            if !tris[j].dead() {
                continue;
            }
            let last = tris.len()-1;
            swap_neib(&mut tris[..], j, last);
            tris.pop();
        }

        vert_limit -= 1;
    }

    // build result -- 4 steps
    // 1. convert to the result type (throw away Tri stuff)
    // 2. compute used vertices
    // 3. move used vertices to front of array
    // 4. update indices
    let mut res: Vec<[i32; 3]> = tris.iter()
        .map(|t| t.vi.at)
        .collect::<Vec<[i32; 3]>>();

    let mut used = vec![0usize; vert_count];
    let mut map = vec![0isize; vert_count];

    for t in &res {
        for ti in t.iter() {
            used[*ti as usize] += 1;
        }
    }

    let mut n = 0;
    for (i, use_count) in used.iter().enumerate() {
        if *use_count > 0 {
            map[i] = n;
            verts.swap(n as usize, i);
            n += 1;
        } else {
            map[i] = -1;
        }
    }

    for i in 0..res.len() {
        for j in 0..3 {
            let old_pos = res[i][j];
            res[i][j] = map[old_pos as usize] as i32;
        }
    }
    Some(res)
}



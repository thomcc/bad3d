use crate::math::prelude::*;
use crate::util;
use std::{f32, i32, isize, ops::*, usize};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct I3 {
    pub at: [i32; 3],
}

#[inline]
fn int3(a: i32, b: i32, c: i32) -> I3 {
    I3 { at: [a, b, c] }
}

#[inline]
fn int3u(a: usize, b: usize, c: usize) -> I3 {
    debug_assert!(
        a <= (i32::MAX as usize) && b <= (i32::MAX as usize) && c <= (i32::MAX as usize),
        "int3u: one of ({}, {}, {}) >= {}",
        a,
        b,
        c,
        i32::MAX
    );
    int3(a as i32, b as i32, c as i32)
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
    fn has_vert(self, a: i32) -> bool {
        self[0] == a || self[1] == a || self[2] == a
    }
}

#[inline]
fn tri(verts: &[V3], t: I3) -> (V3, V3, V3) {
    (verts[t[0] as usize], verts[t[1] as usize], verts[t[2] as usize])
}

fn above(verts: &[V3], t: I3, p: V3, epsilon: f32) -> bool {
    let (v0, v1, v2) = tri(verts, t);
    dot(geom::tri_normal(v0, v1, v2), p - v0) > epsilon
}

#[derive(Copy, Clone, Debug)]
struct HullTri {
    vi: I3,
    ni: I3,
    id: i32,
    max_v: i32,
    rise: f32,
}

fn next_mod3(i: usize) -> (usize, usize) {
    debug_assert!(i < 3);
    ((i + 1) % 3, (i + 2) % 3)
}

impl HullTri {
    #[inline]
    const fn new(vi: I3, ni: I3, id: i32) -> HullTri {
        HullTri {
            vi,
            ni,
            id,
            max_v: -1,
            rise: 0.0,
        }
    }

    #[inline]
    fn dead(&self) -> bool {
        self.ni[0] == -1
    }

    fn neib_idx(&self, va: i32, vb: i32) -> usize {
        for i in 0..3 {
            let (i1, i2) = next_mod3(i);
            if (self.vi[i] == va && self.vi[i1] == vb) || (self.vi[i] == vb && self.vi[i1] == va) {
                return i2;
            }
        }
        unreachable!(
            "Fell through neib loop v={:?} n={:?} va={} vb={}",
            self.vi, self.ni, va, vb
        );
    }

    #[inline]
    fn get_neib(&self, va: i32, vb: i32) -> i32 {
        let idx = self.neib_idx(va, vb);
        self.ni[idx]
    }

    #[inline]
    fn set_neib(&mut self, va: i32, vb: i32, new_value: i32) {
        let idx = self.neib_idx(va, vb);
        self.ni[idx] = new_value;
    }

    fn update(&mut self, verts: &[V3], extreme_map: Option<&[bool]>) {
        let (v0, v1, v2) = tri(verts, self.vi);
        let n = geom::tri_normal(v0, v1, v2);

        let vmax = geom::max_dir_index(n, verts).unwrap();
        self.max_v = vmax as i32;

        if extreme_map.is_some() && extreme_map.unwrap()[vmax] {
            self.max_v = -1; // already did this one
        } else {
            self.rise = dot(n, verts[vmax] - v0);
        }
    }
}

fn neib_fix(tris: &mut [HullTri], k_id: i32) {
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
            tris[ni].set_neib(va, vb, k_id);
        }
    }
}

#[allow(clippy::manual_swap)] // bug in clippy, can't use mem::swap
fn swap_neib(tris: &mut [HullTri], a: usize, b: usize) {
    tris.swap(a, b);
    {
        let id = tris[a].id;
        tris[a].id = tris[b].id;
        tris[b].id = id;
    }
    neib_fix(tris, a as i32);
    neib_fix(tris, b as i32);
}

fn fix_back_to_back(tris: &mut [HullTri], s: usize, t: usize) {
    for i in 0..3 {
        let (i1, i2) = next_mod3(i);
        let (va, vb) = (tris[s].vi[i1], tris[s].vi[i2]);
        debug_assert_eq!(tris[tris[s].get_neib(va, vb) as usize].get_neib(vb, va), tris[s].id);
        debug_assert_eq!(tris[tris[t].get_neib(va, vb) as usize].get_neib(vb, va), tris[t].id);

        let t_neib = tris[t].get_neib(vb, va);
        tris[tris[s].get_neib(va, vb) as usize].set_neib(vb, va, t_neib);

        let s_neib = tris[s].get_neib(va, vb);
        tris[tris[t].get_neib(vb, va) as usize].set_neib(va, vb, s_neib);
    }
    // cleaned up later
    tris[s].ni = int3(-1, -1, -1);
    tris[t].ni = int3(-1, -1, -1);
}

#[inline]
fn check_tri(tris: &[HullTri], t: &HullTri) {
    debug_assert!(tris[t.id as usize].id == t.id);
    debug_assert!(tris[t.id as usize].id == t.id);
    for i in 0..3 {
        let (i1, i2) = next_mod3(i);
        let (a, b) = (t.vi[i1], t.vi[i2]);
        debug_assert!(a != b);
        debug_assert!(tris[t.ni[i] as usize].get_neib(b, a) == t.id);
    }
}

fn extrude(tris: &mut Vec<HullTri>, t0: usize, v: usize) {
    let bu = tris.len();
    let b = bu as i32;

    let n = tris[t0].ni;
    let t = tris[t0].vi;

    let vi = v as i32;

    tris.push(HullTri::new(int3(vi, t[1], t[2]), int3(n[0], b + 1, b + 2), b));
    tris[n[0] as usize].set_neib(t[1], t[2], b);

    tris.push(HullTri::new(int3(vi, t[2], t[0]), int3(n[1], b + 2, b), b + 1));
    tris[n[1] as usize].set_neib(t[2], t[0], b + 1);

    tris.push(HullTri::new(int3(vi, t[0], t[1]), int3(n[2], b, b + 1), b + 2));
    tris[n[2] as usize].set_neib(t[0], t[1], b + 2);

    tris[t0].ni = int3(-1, -1, -1);

    // @@TODO: disable in debug?
    check_tri(&tris, &tris[bu]);
    check_tri(&tris, &tris[bu + 1]);
    check_tri(&tris, &tris[bu + 2]);

    if tris[n[0] as usize].vi.has_vert(vi) {
        fix_back_to_back(&mut tris[..], bu, n[0] as usize);
    }
    if tris[n[1] as usize].vi.has_vert(vi) {
        fix_back_to_back(&mut tris[..], bu + 1, n[1] as usize);
    }
    if tris[n[2] as usize].vi.has_vert(vi) {
        fix_back_to_back(&mut tris[..], bu + 2, n[2] as usize);
    }
}

fn find_extrudable(tris: &[HullTri], epsilon: f32) -> Option<usize> {
    assert_ne!(tris.len(), 0);
    let mut best = 0usize;
    for (idx, tri) in tris.iter().enumerate() {
        chek::debug_ge!(tri.id, 0);
        debug_assert_eq!(tri.id, (idx as i32));
        debug_assert!(!tri.dead());
        if best != idx && tris[best].rise < tri.rise {
            best = idx
        }
    }
    util::some_if(tris[best].rise > epsilon, best)
}

fn find_simplex(verts: &[V3]) -> Option<(usize, usize, usize, usize)> {
    let b0 = vec3(0.01, 0.02, 1.0);

    let p0 = geom::max_dir_index(b0, verts).unwrap();
    let p1 = geom::max_dir_index(-b0, verts).unwrap();

    let b0 = verts[p0] - verts[p1];

    if p0 == p1 || b0 == V3::zero() {
        return None;
    }

    let b1 = cross(vec3(1.0, 0.0, 0.0), b0);
    let b2 = cross(vec3(0.0, 1.0, 0.0), b0);

    let b1 = (if b1.length_sq() > b2.length_sq() { b1 } else { b2 }).must_norm();

    let p2 = geom::max_dir_index(b1, verts).unwrap();

    let p2 = if p2 == p0 || p2 == p1 {
        geom::max_dir_index(-b1, verts).unwrap()
    } else {
        p2
    };

    if p2 == p0 || p2 == p1 {
        return None;
    }

    let b1 = verts[p2] - verts[p0];

    let b2 = cross(b1, b0);

    let p3 = geom::max_dir_index(b2, verts).unwrap();

    let p3 = if p3 == p0 || p3 == p1 || p3 == p2 {
        geom::max_dir_index(-b2, verts).unwrap()
    } else {
        p3
    };

    if p3 == p0 || p3 == p1 || p3 == p2 {
        return None;
    }

    debug_assert!(p0 != p1 && p0 != p2 && p0 != p3 && p1 != p2 && p1 != p3 && p2 != p3);

    if dot(
        verts[p3] - verts[p0],
        cross(verts[p1] - verts[p0], verts[p2] - verts[p0]),
    ) < 0.0
    {
        Some((p0, p1, p3, p2))
    } else {
        Some((p0, p1, p2, p3))
    }
}

fn remove_dead(tris: &mut Vec<HullTri>) {
    for j in (0..tris.len()).rev() {
        if !tris[j].dead() {
            continue;
        }
        let last = tris.len() - 1;
        swap_neib(&mut tris[..], j, last);
        tris.pop();
    }
}

// fix flipped/skinny tris. vert_id is the id of the vertex most recently added to the hull.
// since we only need to consider those triangles (hopefully others won't be broken...)
fn fix_degenerate_tris(tris: &mut Vec<HullTri>, verts: &[V3], center: V3, vert_id: usize, epsilon: f32) {
    let mut i: isize = tris.len() as isize;
    loop {
        i -= 1;
        if i < 0 {
            break;
        }
        let iu = i as usize;
        if tris[iu].dead() {
            continue;
        }

        if !tris[iu].vi.has_vert(vert_id as i32) {
            break;
        }

        let nt = tris[iu].vi;
        let (nv0, nv1, nv2) = tri(verts, nt);

        let is_flipped = above(verts, nt, center, 0.01 * epsilon);
        if is_flipped || geom::tri_area(nv0, nv1, nv2) < epsilon * epsilon * 0.1 {
            debug_assert!(tris[iu].ni[0] >= 0);
            let nb = tris[iu].ni[0] as usize;

            debug_assert!(!tris[nb].dead());
            debug_assert!(!tris[nb].vi.has_vert(vert_id as i32));
            debug_assert!(tris[nb].id < (i as i32));

            extrude(tris, nb, vert_id);
            i = tris.len() as isize;
        }
    }
}

// extrude any triangles who would be below the hull containing the new vertex
fn grow_hull(tris: &mut Vec<HullTri>, verts: &[V3], vert_id: usize, epsilon: f32) {
    for j in (0..tris.len()).rev() {
        if tris[j].dead() {
            continue;
        }
        let t = tris[j].vi;
        if above(verts, t, verts[vert_id], 0.01 * epsilon) {
            extrude(tris, j, vert_id);
        }
    }
}

fn build_simplex(p0: usize, p1: usize, p2: usize, p3: usize) -> [HullTri; 4] {
    let tris = [
        HullTri::new(int3u(p2, p3, p1), int3(2, 3, 1), 0),
        HullTri::new(int3u(p3, p2, p0), int3(3, 2, 0), 1),
        HullTri::new(int3u(p0, p1, p3), int3(0, 1, 3), 2),
        HullTri::new(int3u(p1, p0, p2), int3(1, 0, 2), 3),
    ];

    check_tri(&tris, &tris[0]);
    check_tri(&tris, &tris[1]);
    check_tri(&tris, &tris[2]);
    check_tri(&tris, &tris[3]);

    tris
}

// simp is the terminating simplex for gjk, max_dir is the combined support fn
// runs epa to find the separating plane aka the closest plane to the origin in
// the mink sum
pub fn furthest_plane_epa<F: Fn(V3) -> V3>(simp: (V3, V3, V3, V3), max_dir: F) -> Plane {
    let mut plane = Plane::new(V3::zero(), -f32::MAX);
    let epsilon = 0.01f32; // ...

    let mut verts = vec![simp.0, simp.1, simp.2, simp.3];
    let mut tris = Vec::from(&build_simplex(0, 1, 2, 3)[..]);

    let center = 0.25 * (simp.0 + simp.1 + simp.2 + simp.3);

    // possibly fix simplex winding
    if dot(cross(verts[2] - verts[0], verts[1] - verts[0]), verts[3] - verts[0]) > 0.0 {
        verts.swap(2, 3);
    }

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

        if plane.offset >= face.offset - epsilon || verts.contains(&v) {
            return plane;
        }

        verts.push(v);
        let vert_id = verts.len() - 1;
        grow_hull(&mut tris, &verts[..], vert_id, epsilon);
        fix_degenerate_tris(&mut tris, &verts[..], center, vert_id, epsilon);
        remove_dead(&mut tris);
    }
}

fn finish_hull(tris: &mut [HullTri], verts: &mut [V3]) -> (Vec<[u16; 3]>, usize) {
    let mut used = vec![0usize; verts.len()];
    let mut map = vec![0isize; verts.len()];

    for t in tris.iter() {
        for ti in &t.vi.at {
            used[*ti as usize] += 1;
        }
    }

    let mut n = 0usize;
    for (i, use_count) in used.iter().enumerate() {
        if *use_count > 0 {
            map[i] = n as isize;
            verts.swap(n, i);
            n += 1;
        } else {
            map[i] = -1;
        }
    }

    for tri in tris.iter_mut() {
        for j in 0..3 {
            let old_pos = tri.vi[j];
            tri.vi[j] = map[old_pos as usize] as i32;
        }
    }

    let mut ts: Vec<[u16; 3]> = Vec::with_capacity(tris.len());
    let mut max_idx = 0;

    for t in tris.iter() {
        let tmaxi = max!(t.vi[0] as usize, t.vi[1] as usize, t.vi[2] as usize);
        if tmaxi > max_idx {
            max_idx = tmaxi;
        }
        ts.push([t.vi[0] as u16, t.vi[1] as u16, t.vi[2] as u16]);
    }

    (ts, max_idx + 1)
}

pub fn compute_hull(verts: &mut [V3]) -> Option<(Vec<[u16; 3]>, usize)> {
    compute_hull_bounded(verts, 0)
}

pub fn compute_hull_bounded(verts: &mut [V3], vert_limit: usize) -> Option<(Vec<[u16; 3]>, usize)> {
    assert!(verts.len() < (i32::MAX as usize));
    if verts.len() < 4 {
        return None;
    }

    let mut vert_limit: isize = if vert_limit == 0 {
        isize::MAX
    } else {
        vert_limit as isize
    };
    let vert_count = verts.len();

    let mut is_extreme = vec![false; vert_count];
    let (min_bound, max_bound) = geom::compute_bounds(verts)?;

    let epsilon = max_bound.dist(min_bound) * 0.001_f32;

    let (p0, p1, p2, p3) = find_simplex(verts)?;

    let mut tris = Vec::from(&build_simplex(p0, p1, p2, p3)[..]);

    let center = (verts[p0] + verts[p1] + verts[p2] + verts[p3]) * 0.25;

    is_extreme[p0] = true;
    is_extreme[p1] = true;
    is_extreme[p2] = true;
    is_extreme[p3] = true;

    for t in &mut tris {
        debug_assert!(t.id >= 0);
        debug_assert!(t.max_v < 0);
        t.update(&verts, None);
    }

    vert_limit -= 4;

    while vert_limit > 0 {
        let te = match find_extrudable(&tris, epsilon) {
            Some(e) => e,
            None => break,
        };

        let v = tris[te].max_v as usize;
        debug_assert!(!is_extreme[v]);

        is_extreme[v] = true;

        grow_hull(&mut tris, &verts, v, epsilon);

        fix_degenerate_tris(&mut tris, &verts, center, v, epsilon);

        for tri in tris.iter_mut().rev() {
            if tri.dead() {
                continue;
            }
            if tri.max_v >= 0 {
                break;
            }
            tri.update(&verts, Some(&is_extreme));
        }

        remove_dead(&mut tris);

        vert_limit -= 1;
    }

    Some(finish_hull(&mut tris, verts))
}

pub fn compute_hull_trunc(verts: &mut Vec<V3>, vert_limit: Option<usize>) -> Option<Vec<[u16; 3]>> {
    if let Some((tris, size)) = compute_hull_bounded(verts, vert_limit.unwrap_or(0)) {
        verts.truncate(size);
        Some(tris)
    } else {
        None
    }
}

use crate::math::mat::*;
use crate::math::plane::*;
use crate::math::pose::*;
use crate::math::scalar::*;
use crate::math::traits::*;
use crate::math::vec::*;

#[inline]
pub fn tri_face_dir(v0: V3, v1: V3, v2: V3) -> V3 {
    cross(v1 - v0, v2 - v1)
}

#[inline]
pub fn tri_normal(v0: V3, v1: V3, v2: V3) -> V3 {
    tri_face_dir(v0, v1, v2).norm_or(0.0, 0.0, 1.0)
}

#[inline]
pub fn tri_area(v0: V3, v1: V3, v2: V3) -> f32 {
    tri_face_dir(v0, v1, v2).length()
}

#[inline]
pub fn line_project_time(p0: V3, p1: V3, a: V3) -> f32 {
    let d = p1 - p0;
    safe_div0(dot(d, a - p0), dot(d, d))
}

#[inline]
pub fn line_project(p0: V3, p1: V3, a: V3) -> V3 {
    p0 + (p1 - p0) * line_project_time(p0, p1, a)
}

pub fn barycentric(a: V3, b: V3, c: V3, p: V3) -> V3 {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;
    let d00 = dot(v0, v0);
    let d01 = dot(v0, v1);
    let d11 = dot(v1, v1);

    let d20 = dot(v2, v0);
    let d21 = dot(v2, v1);

    let d = safe_div0(1.0, d00 * d11 - d01 * d01);
    let v = (d11 * d20 - d01 * d21) * d;
    let w = (d00 * d21 - d01 * d20) * d;
    vec3(1.0 - v - w, v, w)
}

pub fn tri_project(v0: V3, v1: V3, v2: V3, p: V3) -> V3 {
    let cp = cross(v2 - v0, v2 - v1);
    let dtcpm = -dot(cp, v0);
    let cpm2 = dot(cp, cp);
    if cpm2 == 0.0 {
        let end = if (v1 - v0).length_sq() > (v2 - v0).length_sq() {
            v1
        } else {
            v2
        };
        line_project(v0, end, p)
    } else {
        p - cp * (dot(cp, p) + dtcpm) / cpm2
    }
}

pub fn gradient(v0: V3, v1: V3, v2: V3, t0: f32, t1: f32, t2: f32) -> V3 {
    let e0 = v1 - v0;
    let e1 = v2 - v0;
    let d0 = t1 - t0;
    let d1 = t2 - t0;
    let pd = e1 * d0 - e0 * d1;
    if dot(pd, pd) == 0.0 {
        return vec3(0.0, 0.0, 1.0);
    }
    let pd = pd.must_norm();
    let e = if d0.abs() > d1.abs() {
        (e0 + pd * -dot(pd, e0)) / d0
    } else {
        (e1 + pd * -dot(pd, e1)) / d1
    };

    e * safe_div0(1.0, dot(e, e))
}

#[inline]
fn tri_matrix<T: TriIndices>(tri: T, verts: &[V3]) -> M3x3 {
    let (va, vb, vc) = tri.tri_verts(verts);
    M3x3::from_cols(va, vb, vc)
}

// does this belong here?
pub fn volume<Idx: TriIndices>(verts: &[V3], tris: &[Idx]) -> f32 {
    (1.0 / 6.0)
        * tris
            .iter()
            .fold(0.0, |acc, &tri| acc + tri_matrix(tri, verts).determinant())
}

pub fn center_of_mass<Idx: TriIndices>(verts: &[V3], tris: &[Idx]) -> V3 {
    let (com, vol) = tris.iter().fold((V3::zero(), 0.0), |(acom, avol), &tri| {
        let m = tri_matrix(tri, verts);
        let vol = m.determinant();
        (acom + vol * (m.x + m.y + m.z), avol + vol)
    });
    com / (vol * 4.0)
}

pub fn inertia<Idx: TriIndices>(verts: &[V3], tris: &[Idx], com: V3) -> M3x3 {
    let mut volume = 0.0f32;
    let mut diag = V3::zero();
    let mut offd = V3::zero();

    for tri in tris.iter() {
        let (a, b, c) = tri.tri_indices();
        let m = M3x3::from_cols(verts[a] - com, verts[b] - com, verts[c] - com);
        let d = m.determinant();
        volume += d;
        for j in 0..3 {
            let j1 = (j + 1) % 3;
            let j2 = (j + 2) % 3;
            diag[j] += (m.x[j] * m.y[j]
                + m.y[j] * m.z[j]
                + m.z[j] * m.x[j]
                + m.x[j] * m.x[j]
                + m.y[j] * m.y[j]
                + m.z[j] * m.z[j])
                * d;
            offd[j] += ((m.x[j1] * m.y[j2] + m.y[j1] * m.z[j2] + m.z[j1] * m.x[j2])
                + (m.x[j1] * m.z[j2] + m.y[j1] * m.x[j2] + m.z[j1] * m.y[j2])
                + (m.x[j1] * m.x[j2] + m.y[j1] * m.y[j2] + m.z[j1] * m.z[j2]) * 2.0)
                * d;
        }
    }

    diag /= volume * 10.0;
    offd /= volume * 20.0;

    mat3(
        diag.y + diag.z,
        -offd.z,
        -offd.y,
        -offd.z,
        diag.x + diag.z,
        -offd.x,
        -offd.y,
        -offd.x,
        diag.x + diag.y,
    )
}

/// returns (sq dist, (pt on line1, t for line1), (pt on line2, t for line2))
pub fn closest_points_on_lines(line1: (V3, V3), line2: (V3, V3)) -> (f32, (V3, f32), (V3, f32)) {
    let epsilon = DEFAULT_EPSILON;
    let d1 = line1.1 - line1.0;
    let d2 = line2.1 - line2.0;
    let r = line1.0 - line2.0;
    let a = dot(d1, d1);
    let e = dot(d2, d2);

    if a <= epsilon && e <= epsilon {
        return (line1.0.dist_sq(line2.0), (line1.0, 0.0), (line2.0, 0.0));
    }
    let f = dot(d2, r);
    let (s, t) = if a <= epsilon {
        (0.0, clamp01(f / e))
    } else if e <= epsilon {
        (clamp01(-dot(d1, r) / a), 0.0)
    } else {
        let b = dot(d1, d2);
        let c = dot(d1, r);
        let denom = a * e - b * b;

        let s = safe_div(b * f - c * e, denom).map(clamp01).unwrap_or(0.0);

        let t_nom = b * s + f;

        if t_nom < 0.0 {
            (clamp01(-c / a), 0.0)
        } else if t_nom > e {
            (clamp01((b - c) / a), 1.0)
        } else {
            (s, t_nom / e)
        }
    };
    let c1 = line1.0 + d1 * s;
    let c2 = line2.0 + d2 * t;

    (c1.dist_sq(c2), (c1, s), (c2, t))
}

pub fn closest_point_on_triangle(a: V3, b: V3, c: V3, p: V3) -> V3 {
    let ab = b - a;
    let ac = c - a;
    // Check closest to A
    let ap = p - a;
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 || d2 <= 0.0 {
        return a; // bary (1, 0, 0)
    }
    // Check closest to B
    let bp = p - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b; // bary (0, 1, 0)
    }
    // Check AB edge
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return a + v * ab; // bary (1 - v, v, 0)
    }
    // Check closest to C
    let cp = p - c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c; // bary (0, 0, 1)
    }
    // Check AC edge
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return a + w * ac; // bary (1-w, 0, w)
    }
    // Check BC edge
    let va = d3 * d6 - d5 * d4;
    let d43 = d4 - d3;
    let d56 = d5 - d6;
    if va <= 0.0 && d43 >= 0.0 && d56 >= 0.0 {
        let w = d43 / (d43 + d56);
        return b + w * (c - b); // bary (0, 1-w, w)
    }

    // Inside face
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    a + ab * v + ac * w
}

#[inline]
pub fn sq_dist_pt_segment(p: V3, s: (V3, V3)) -> f32 {
    let (a, b) = s;
    let ab = b - a;
    let ap = p - a;
    let bp = p - b;
    let e = dot(ap, ab);
    if e <= 0.0 {
        return dot(ap, ap);
    }
    let f = dot(ab, ab);
    if e >= f {
        dot(bp, bp)
    } else {
        dot(ap, ap) - e * e / f
    }
}

#[inline]
pub fn test_sphere_capsule(sphere: (V3, f32), capsule: (V3, V3, f32)) -> bool {
    let dist2 = sq_dist_pt_segment(sphere.0, (capsule.0, capsule.1));
    let rad = sphere.1 + capsule.2;
    dist2 <= rad * rad
}

#[inline]
pub fn test_capsule_capsule(c1: (V3, V3, f32), c2: (V3, V3, f32)) -> bool {
    let (dist2, ..) = closest_points_on_lines((c1.0, c1.1), (c2.0, c2.1));
    let radius = c1.2 + c2.2;
    dist2 <= radius * radius
}

#[derive(Copy, Clone, Default, Debug)]
pub struct HitInfo {
    pub impact: V3,
    pub normal: V3,
}

impl HitInfo {
    #[inline]
    pub fn new_opt(did_hit: bool, impact: V3, normal: V3) -> Option<Self> {
        if did_hit {
            Some(Self::new(impact, normal))
        } else {
            None
        }
    }

    #[inline]
    pub fn new(impact: V3, normal: V3) -> Self {
        Self { impact, normal }
    }
}

pub fn poly_hit_check_p(verts: &[V3], plane: Plane, v0: V3, v1: V3) -> Option<HitInfo> {
    let d0 = Plane::new(v0, 1.0).dot(plane);
    let d1 = Plane::new(v1, 1.0).dot(plane);
    let mut did_hit = d0 > 0.0 && d1 < 0.0;
    if !did_hit {
        return None;
    }

    let impact = v0 + (v1 - v0) * safe_div0(d0, d0 - d1);
    for (i, &v) in verts.iter().enumerate() {
        if !did_hit {
            break;
        }
        did_hit = M3x3::from_cols(verts[(i + 1) % verts.len()] - v0, v - v0, v1 - v0).determinant() >= 0.0;
    }
    HitInfo::new_opt(did_hit, impact, plane.normal)
}

pub fn poly_hit_check(verts: &[V3], v0: V3, v1: V3) -> Option<HitInfo> {
    poly_hit_check_p(verts, Plane::from_points(verts), v0, v1)
}

pub fn convex_hit_check(planes: impl Iterator<Item = Plane>, p0: V3, p1: V3) -> Option<HitInfo> {
    let mut n = V3::zero();
    let mut v0 = p0;
    let mut v1 = p1;
    for plane in planes {
        let d0 = Plane::new(v0, 1.0).dot(plane);
        let d1 = Plane::new(v1, 1.0).dot(plane);
        if d0 >= 0.0 && d1 >= 0.0 {
            return None;
        }
        if d0 <= 0.0 && d1 <= 0.0 {
            continue;
        }
        let c = v0 + (v1 - v0) * safe_div0(d0, d0 - d1);
        if d0 >= 0.0 {
            n = plane.normal;
            v0 = c;
        } else {
            v1 = c;
        }
    }
    Some(HitInfo::new(v0, n))
}

pub fn convex_hit_check_posed(planes: impl Iterator<Item = Plane>, pose: Pose, p0: V3, p1: V3) -> Option<HitInfo> {
    let inv_pose = pose.inverse();
    convex_hit_check(planes, inv_pose * p0, inv_pose * p1)
        .map(|hit| HitInfo::new(pose * hit.impact, pose.orientation * hit.normal))
}

#[derive(Copy, Clone, Debug)]
pub struct SegmentTestInfo {
    pub w0: V3,
    pub w1: V3,
    pub nw0: V3,
}

pub fn segment_under(p: Plane, v0: V3, v1: V3, nv0: V3) -> Option<SegmentTestInfo> {
    let d0 = p.normal.dot(v0) + p.offset;
    let d1 = p.normal.dot(v1) + p.offset;
    match (d0 > 0.0, d1 > 0.0) {
        (true, true) => None,
        (false, false) => Some(SegmentTestInfo {
            w0: v0,
            w1: v1,
            nw0: nv0,
        }),
        (true, false) => {
            let vmid = p.intersect_with_line(v0, v1);
            Some(SegmentTestInfo {
                w0: vmid,
                w1: v1,
                nw0: p.normal,
            })
        }
        (false, true) => {
            let vmid = p.intersect_with_line(v0, v1);
            Some(SegmentTestInfo {
                w0: v0,
                w1: vmid,
                nw0: nv0,
            })
        }
    }
}

pub fn segment_over(p: Plane, v0: V3, v1: V3, nv0: V3) -> Option<SegmentTestInfo> {
    segment_under(-p, v0, v1, nv0)
}

pub fn tangent_point_on_cylinder(r: f32, h: f32, n: V3) -> V3 {
    let xy_inv = safe_div1(1.0, (n.x * n.x + n.y * n.y).sqrt());
    vec3(
        r * n.x * xy_inv,
        r * n.y * xy_inv,
        // Reference point is at cyl. base. use h/2 and -h/2 for midpt
        if n.z > 0.0 { h } else { 0.0 },
    )
}

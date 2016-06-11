use math::vec::*;
use math::mat::*;
use math::scalar::*;
use math::traits::*;
use math::pose::*;
pub use math::plane::*;

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
    let v0 = b-a;
    let v1 = c-a;
    let v2 = p-a;
    let d00 = dot(v0, v0);
    let d01 = dot(v0, v1);
    let d11 = dot(v1, v1);

    let d20 = dot(v2, v0);
    let d21 = dot(v2, v1);

    let d = safe_div0(1.0, d00*d11 - d01*d01);
    let v = (d11*d20 - d01*d21) * d;
    let w = (d00*d21 - d01*d20) * d;
    vec3(1.0-v-w, v, w)
}

pub fn tri_project(v0: V3, v1: V3, v2: V3, p: V3) -> V3 {
    let cp = cross(v2 - v0, v2 - v1);
    let dtcpm = -dot(cp, v0);
    let cpm2 = dot(cp, cp);
    if cpm2 == 0.0 {
        let end = if (v1 - v0).length() > (v2 - v0).length() { v1 } else { v2 };
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
    let pd = e1*d0 - e0*d1;
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

// does this belong here?
pub fn volume<Tri: TriIndices>(verts: &[V3], tris: &[Tri]) -> f32 {
    (1.0 / 6.0) * tris.iter().fold(0.0, |acc, tri| {
        let (a, b, c) = tri.tri_indices();
        acc + M3x3::from_cols(verts[a], verts[b], verts[c]).determinant()
    })
}

pub fn center_of_mass<Tri: TriIndices>(verts: &[V3], tris: &[Tri]) -> V3 {
    let (com, vol) = tris.iter().fold((V3::zero(), 0.0), |(acom, avol), tri| {
        let (a, b, c) = tri.tri_indices();
        let m = M3x3::from_cols(verts[a], verts[b], verts[c]);
        let vol = m.determinant();
        (acom + vol * (m.x + m.y + m.z),
         avol + vol)
    });
    com / (vol * 4.0)
}

pub fn inertia<Tri: TriIndices>(verts: &[V3], tris: &[Tri], com: V3) -> M3x3 {
    let mut volume = 0.0f32;
    let mut diag = V3::zero();
    let mut offd = V3::zero();

    for tri in tris.iter() {
        let (a, b, c) = tri.tri_indices();
        let m = M3x3::from_cols(verts[a]-com, verts[b]-com, verts[c]-com);
        let d = m.determinant();
        volume += d;
        for j in 0..3 {
            let j1 = (j + 1) % 3;
            let j2 = (j + 2) % 3;
            diag[j] += (m.x[j]*m.y[j] + m.y[j]*m.z[j] + m.z[j]*m.x[j] +
                        m.x[j]*m.x[j] + m.y[j]*m.y[j] + m.z[j]*m.z[j]) * d;
            offd[j] += ((m.x[j1]*m.y[j2] + m.y[j1]*m.z[j2] + m.z[j1]*m.x[j2]) +
                        (m.x[j1]*m.z[j2] + m.y[j1]*m.x[j2] + m.z[j1]*m.y[j2]) +
                        (m.x[j1]*m.x[j2] + m.y[j1]*m.y[j2] + m.z[j1]*m.z[j2]) * 2.0) * d;
        }
    }

    diag /= volume * 10.0;
    offd /= volume * 20.0;

    mat3(diag.y + diag.z, -offd.z,         -offd.y,
         -offd.z,         diag.x + diag.z, -offd.x,
         -offd.y,         -offd.x,          diag.x + diag.y)
}

#[derive(Copy, Clone, Default, Debug)]
pub struct HitInfo {
    pub did_hit: bool,
    pub impact: V3,
    pub normal: V3,
}

impl HitInfo {
    pub fn new(did_hit: bool, impact: V3, normal: V3) -> HitInfo {
        HitInfo {
            did_hit: did_hit,
            impact: impact,
            normal: normal
        }
    }
}


pub fn poly_hit_check_p(verts: &[V3], plane: Plane, v0: V3, v1: V3) -> HitInfo {
    let d0 = dot(Plane::new(v0, 1.0), plane);
    let d1 = dot(Plane::new(v1, 1.0), plane);
    let mut hit_info = HitInfo {
        did_hit: d0 > 0.0 && d1 < 0.0,
        normal: plane.normal,
        impact: v0 + (v1 - v0) * safe_div0(d0,  d0 - d1)
    };
    for (i, &v) in verts.iter().enumerate() {
        if !hit_info.did_hit {
            break;
        }
        hit_info.did_hit = hit_info.did_hit &&
            M3x3::from_cols(verts[(i+1)%verts.len()]-v0, v-v0, v1-v0).determinant() >= 0.0;
    }
    hit_info
}

pub fn poly_hit_check(verts: &[V3], v0: V3, v1: V3) -> HitInfo {
    poly_hit_check_p(verts, Plane::from_points(verts), v0, v1)
}

pub fn convex_hit_check(planes: &[Plane], p0: V3, p1: V3) -> HitInfo {
    let mut n = V3::zero();
    let mut v0 = p0;
    let mut v1 = p1;
    for &plane in planes.iter() {
        let d0 = dot(Plane::new(v0, 1.0), plane);
        let d1 = dot(Plane::new(v1, 1.0), plane);
        if d0 >= 0.0 && d1 >= 0.0 {
            return HitInfo::new(false, p1, V3::zero());
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
    return HitInfo::new(true, v0, n);
}

pub fn convex_hit_check_posed(planes: &[Plane], pose: Pose, p0: V3, p1: V3) -> HitInfo {
    let inv_pose = pose.inverse();
    let hit = convex_hit_check(planes, inv_pose*p0, inv_pose*p1);
    HitInfo::new(hit.did_hit, pose*hit.impact, pose.orientation*hit.normal)
}











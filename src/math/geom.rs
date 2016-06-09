use math::vec::*;
use math::mat::*;
use math::scalar::*;
use math::traits::*;
use math::quat::*;

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
    // debug_assert!(dot(d, d) != 0.0);
    safe_div0(dot(d, a - p0), dot(d, d))
}

#[inline]
pub fn line_project(p0: V3, p1: V3, a: V3) -> V3 {
    p0 + (p1 - p0) * line_project_time(p0, p1, a)//p0.lerp(p1, line_project_time(p0, p1, a))
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
    /*
    let m = M3x3::from_cols(v0, v1, v2);
    if let Some(inv) = m.inverse() {
        inv * s
    } else {
        let k: usize = if v2.dist_sq(v1) > v2.dist_sq(v0) { 1 } else { 0 };
        let t = line_project_time(v2, m.col(k), s);
        let kf = k as f32;
        vec3((1.0 - kf) * t, kf * t, 1.0 - t)
    }*/
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

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Plane {
    pub normal: V3,
    pub offset: f32,
}

impl Plane {
    #[inline]
    pub fn to_v4(&self) -> V4 {
        V4::expand(self.normal, self.offset)
    }

    #[inline]
    pub fn from_v4(v: V4) -> Plane {
        Plane::new(v.xyz(), v.w)
    }

    #[inline]
    pub fn new(normal: V3, offset: f32) -> Plane {
        Plane{ normal: normal, offset: offset }
    }

    #[inline]
    pub fn from_tri(v0: V3, v1: V3, v2: V3) -> Plane {
        Plane::from_norm_and_point(tri_normal(v0, v1, v2), v0)
    }

    #[inline]
    pub fn from_norm_and_point(n: V3, pt: V3) -> Plane {
        Plane::new(n, -dot(n, pt))
    }

    #[inline]
    pub fn intersect_with_line(&self, line_p0: V3, line_p1: V3) -> V3 {
        let dif = line_p1 - line_p0;
        let dn = self.normal.dot(dif);
        let t = safe_div0(-(self.offset + dot(self.normal, line_p0)), dn);
        line_p0 + dif*t
    }

    #[inline]
    pub fn project(&self, pt: V3) -> V3 {
        pt - self.normal * dot(self.to_v4(), V4::expand(pt, 1.0))
    }

    #[inline]
    pub fn translate(&self, v: V3) -> Plane {
        Plane::new(self.normal, self.offset - dot(self.normal, v))
    }

    #[inline]
    pub fn rotate(&self, r: Quat) -> Plane {
        Plane::new(r * self.normal, self.offset)
    }

    #[inline]
    pub fn scale3(&self, s: V3) -> Plane {
        let new_normal = self.normal / s;
        let len = new_normal.length();
        debug_assert!(!len.approx_zero());
        Plane::new(new_normal/len, self.offset/len)
    }

    #[inline]
    pub fn scale(&self, s: f32) -> Plane {
        Plane::new(self.normal, self.offset*s)
    }
}


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





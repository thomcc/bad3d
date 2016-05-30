use math::*;

#[inline]
pub fn tri_normal(v0: V3, v1: V3, v2: V3) -> V3 {
    cross(v1 - v0, v2 - v1).norm_or(0.0, 0.0, 1.0)
}

#[inline]
pub fn line_project_time(p0: V3, p1: V3, a: V3) -> f32 {
    let d = p1 - p0;
    safe_div0(dot(d, a - p0), dot(d, d))
}

#[inline]
pub fn line_project(p0: V3, p1: V3, a: V3) -> V3 {
    p0.lerp(p1, line_project_time(p0, p1, a))
}

pub fn barycentric(v0: V3, v1: V3, v2: V3, s: V3) -> V3 {
    let m = M3x3::from_cols(v0, v1, v2);
    if let Some(inv) = m.inverse() {
        inv * s
    } else {
        let k: usize = if v2.distance_sq(v1) > v2.distance_sq(v0) { 1 } else { 0 };
        let t = line_project_time(v2, m.col(k), s);
        let kf = k as f32;
        vec3((1.0 - kf) * t, kf * t, 1.0 - t)
    }
}

pub fn tri_project(v0: V3, v1: V3, v2: V3, pt: V3) -> V3 {
    let cp = cross(v2 - v0, v2 - v1);
    let cp_pd = -dot(cp, v0);
    let cp_l2 = cp.length_sq();
    if cp_l2.approx_zero() {
        let line_end = if v0.distance_sq(v1) > v0.distance_sq(v2) { v1 } else { v2 };
        line_project(v0, line_end, pt )
    } else {
        pt - cp * ((dot(cp, pt) + cp_pd) / cp_l2)
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
        let n = tri_normal(v0, v1, v2);
        Plane::new(n, -dot(n, v0))
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





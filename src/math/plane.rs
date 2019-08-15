use crate::math::geom::*;
use crate::math::quat::*;
use crate::math::scalar::*;
use crate::math::traits::*;
use crate::math::vec::*;

use std::{fmt, ops};

pub const DEFAULT_PLANE_WIDTH: f32 = 0.00008_f32;

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Plane {
    pub normal: V3,
    pub offset: f32,
}

impl Default for Plane {
    #[inline]
    fn default() -> Plane {
        plane(0.0, 0.0, 1.0, 0.0)
    }
}

impl fmt::Display for Plane {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "plane(({}, {}, {}), {})",
            self.normal.x(),
            self.normal.y(),
            self.normal.z(),
            self.offset
        )
    }
}

impl From<Plane> for V4 {
    #[inline]
    fn from(p: Plane) -> Self {
        p.to_v4()
    }
}

impl From<V4> for Plane {
    #[inline]
    fn from(v: V4) -> Self {
        Plane::from_v4(v)
    }
}

#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub enum PlaneTestResult {
    Coplanar = 0b00,
    Under = 0b01,
    Over = 0b10,
    Split = 0b11, // Under | Over, not possible for points
}

impl ops::Neg for Plane {
    type Output = Plane;
    #[inline]
    fn neg(self) -> Plane {
        Plane::new(-self.normal, -self.offset)
    }
}

impl Plane {
    #[inline]
    pub const fn zero() -> Self {
        Self::ZERO
    }

    #[inline]
    pub fn to_v4(&self) -> V4 {
        V4 {
            x: self.normal.x(),
            y: self.normal.y(),
            z: self.normal.z(),
            w: self.offset,
        }
    }

    #[inline]
    pub fn tup(&self) -> (f32, f32, f32, f32) {
        (self.normal.x(), self.normal.y(), self.normal.z(), self.offset)
    }

    #[inline]
    pub fn from_v4(v: V4) -> Plane {
        Plane::new(v.xyz(), v.w)
    }

    #[inline]
    pub fn from_points(points: &[V3]) -> Plane {
        chek::ge!(points.len(), 3);
        let c = points.iter().fold(V3::zero(), |a, &b| a + b) / (points.len() as f32);
        let mut n = V3::zero();
        for i in 0..points.len() {
            let i1 = (i + 1) % points.len();
            n += cross(points[i] - c, points[i1] - c);
        }
        Plane::from_norm_and_point(n.norm_or(0.0, 0.0, 1.0), c)
    }

    #[inline]
    pub const fn new(normal: V3, offset: f32) -> Plane {
        Plane { normal, offset }
    }

    #[inline]
    pub fn from_tri(v0: V3, v1: V3, v2: V3) -> Plane {
        Plane::from_norm_and_point(tri_normal(v0, v1, v2), v0)
    }

    #[inline]
    pub fn from_norm_and_point(n: V3, pt: V3) -> Plane {
        Plane::new(n, -n.dot(pt))
    }

    #[inline]
    pub fn intersect_with_line(&self, line_p0: V3, line_p1: V3) -> V3 {
        let dif = line_p1 - line_p0;
        let dn = self.normal.dot(dif);
        let t = safe_div0(-(self.offset + dot(self.normal, line_p0)), dn);
        line_p0 + dif * t
    }

    #[inline]
    pub fn try_intersect_with_line(&self, line_p0: V3, line_p1: V3) -> Option<(V3, f32)> {
        let dif = line_p1 - line_p0;
        let dn = self.normal.dot(dif);
        if dn != 0.0 {
            let t = -(self.offset + dot(self.normal, line_p0)) / dn;
            Some((line_p0 + dif * t, t))
        } else {
            None
        }
    }

    #[inline]
    pub fn project(&self, pt: V3) -> V3 {
        pt - self.normal * self.to_v4().dot(V4::expand(pt, 1.0))
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
        let inv = safe_div0(1.0, len);
        Plane::new(new_normal * inv, self.offset * inv)
    }

    #[inline]
    pub fn scale(&self, s: f32) -> Plane {
        Plane::new(self.normal, self.offset * s)
    }

    #[inline]
    pub fn test_e(&self, pos: V3, e: f32) -> PlaneTestResult {
        chek::debug_ge!(e, 0.0);
        let a = dot(pos, self.normal) + self.offset;
        if a > e {
            PlaneTestResult::Over
        } else if a < -e {
            PlaneTestResult::Under
        } else {
            PlaneTestResult::Coplanar
        }
    }

    #[inline]
    pub fn split_test_e(&self, verts: &[V3], e: f32) -> PlaneTestResult {
        let u = self.split_test_val_e(verts, e);
        if u == PlaneTestResult::Coplanar as usize {
            PlaneTestResult::Coplanar
        } else if u == PlaneTestResult::Under as usize {
            PlaneTestResult::Under
        } else if u == PlaneTestResult::Over as usize {
            PlaneTestResult::Over
        } else if u == PlaneTestResult::Split as usize {
            PlaneTestResult::Split
        } else {
            unreachable!("bad plane test result: {}", u)
        }
    }

    #[inline]
    pub fn split_test_val_e(&self, verts: &[V3], e: f32) -> usize {
        let mut u = 0usize;
        for &v in verts {
            u |= self.test_e(v, e) as usize;
            if u == PlaneTestResult::Split as usize {
                break;
            }
        }
        u
    }

    #[inline]
    pub fn split_line_e(&self, v0: V3, v1: V3, e: f32) -> Option<V3> {
        let t0 = self.test_e(v0, e) as usize;
        let t1 = self.test_e(v1, e) as usize;

        if (t0 | t1) == PlaneTestResult::Split as usize {
            Some(self.intersect_with_line(v0, v1))
        } else {
            None
        }
    }

    #[inline]
    pub fn split_line(&self, v0: V3, v1: V3) -> Option<V3> {
        self.split_line_e(v0, v1, DEFAULT_PLANE_WIDTH)
    }

    #[inline]
    pub fn offset_by(&self, o: f32) -> Plane {
        Plane::new(self.normal, self.offset + o)
    }

    #[inline]
    pub fn split_test(&self, verts: &[V3]) -> PlaneTestResult {
        self.split_test_e(verts, DEFAULT_PLANE_WIDTH)
    }

    #[inline]
    pub fn split_test_val(&self, verts: &[V3]) -> usize {
        self.split_test_val_e(verts, DEFAULT_PLANE_WIDTH)
    }

    #[inline]
    pub fn test(&self, pos: V3) -> PlaneTestResult {
        self.test_e(pos, DEFAULT_PLANE_WIDTH)
    }

    #[inline]
    pub fn dot(self, o: Plane) -> f32 {
        self.to_v4().dot(o.to_v4())
    }

    pub const ZERO: Plane = Plane {
        normal: V3::ZERO,
        offset: 0.0,
    };
}

#[inline]
pub fn plane(nx: f32, ny: f32, nz: f32, o: f32) -> Plane {
    Plane::new(vec3(nx, ny, nz), o)
}

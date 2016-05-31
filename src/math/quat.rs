use math::vec::*;
use math::traits::*;
use math::mat::*;

use std::ops::*;
use std::mem;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Quat(pub V4);

#[inline(always)]
pub fn quat(x: f32, y: f32, z: f32, w: f32) -> Quat {
    Quat(vec4(x, y, z, w))
}

impl From<V4> for Quat { #[inline] fn from(v: V4) -> Quat { Quat(v) } }

impl AsRef<V4> for Quat { #[inline] fn as_ref(&    self) -> &    V4 { unsafe { mem::transmute(self) } } }
impl AsMut<V4> for Quat { #[inline] fn as_mut(&mut self) -> &mut V4 { unsafe { mem::transmute(self) } } }

impl Identity for Quat {
    #[inline(always)] fn identity() -> Quat { quat(0.0, 0.0, 0.0, 1.0) }
}

impl Default for Quat {
    #[inline(always)] fn default() -> Quat { quat(0.0, 0.0, 0.0, 1.0) }
}

impl Mul<f32> for Quat {
    type Output = Quat;
    #[inline] fn mul(self, o: f32) -> Quat { Quat(self.0 * o) }
}

impl Div<f32> for Quat {
    type Output = Quat;
    #[inline] fn div(self, o: f32) -> Quat { Quat(self.0 * (1.0 / o)) }
}

impl Add for Quat {
    type Output = Quat;
    #[inline] fn add(self, o: Quat) -> Quat { Quat(self.0 + o.0) }
}

impl Sub for Quat {
    type Output = Quat;
    #[inline] fn sub(self, o: Quat) -> Quat { Quat(self.0 - o.0) }
}

impl Neg for Quat {
    type Output = Quat;
    #[inline] fn neg(self) -> Quat { Quat(-self.0) }
}

impl Mul<Quat> for Quat {
    type Output = Quat;
    #[inline]
    fn mul(self, other: Quat) -> Quat {
        let Quat(V4{x: sx, y: sy, z: sz, w: sw}) = self;
        let Quat(V4{x: ox, y: oy, z: oz, w: ow}) = other;
        Quat::new(sx*ow + sw*ox + sy*oz - sz*oy,
                  sy*ow + sw*oy + sz*ox - sx*oz,
                  sz*ow + sw*oz + sx*oy - sy*ox,
                  sw*ow - sx*ox - sy*oy - sz*oz)
    }
}

impl MulAssign for Quat {
    #[inline]
    fn mul_assign(&mut self, other: Quat) {
        let res = *self * other;
        *self = res;
    }
}

impl AddAssign for Quat {
    #[inline]
    fn add_assign(&mut self, other: Quat) {
        let res = *self + other;
        *self = res;
    }
}

impl SubAssign for Quat {
    #[inline]
    fn sub_assign(&mut self, other: Quat) {
        let res = *self - other;
        *self = res;
    }
}


impl MulAssign<f32> for Quat {
    #[inline]
    fn mul_assign(&mut self, rhs: f32) {
        let res = *self * rhs;
        *self = res;
    }
}

impl DivAssign<f32> for Quat {
    #[inline]
    fn div_assign(&mut self, rhs: f32) {
        let res = *self * (1.0 / rhs);
        *self = res;
    }
}

impl Lerp for Quat {
    #[inline]
    fn lerp(self, o: Quat, t: f32) -> Quat {
        Quat(self.0.lerp(o.0, t))
    }
}

impl Quat {
    #[inline] pub fn new(x: f32, y: f32, z: f32, w: f32) -> Quat { quat(x, y, z, w) }
    #[inline] pub fn axis(self) -> V3 { V3::from(self.0).norm_or(1.0, 0.0, 0.0) }
    #[inline] pub fn angle(self) -> f32 { self.0.w.acos() * 2.0 }

    #[inline]
    pub fn conj(self) -> Quat {
        quat(-self.0.x, -self.0.y, -self.0.z, self.0.w)
    }

    #[inline] pub fn length_sq(self) -> f32 { self.0.length_sq() }
    #[inline] pub fn length(self) -> f32 { self.0.length() }
    #[inline] pub fn dot(self, o: Quat) -> f32 { self.0.dot(o.0) }

    #[inline]
    pub fn x_dir(self) -> V3 {
        let Quat(V4{x, y, z, w}) = self;
        vec3(w*w + x*x - y*y - z*z,
             x*y + z*w + x*y + z*w,
             z*x - y*w + z*x - y*w)
    }

    #[inline]
    pub fn y_dir(self) -> V3 {
        let Quat(V4{x, y, z, w}) = self;
        vec3(x*y - z*w + x*y - z*w,
             w*w - x*x + y*y - z*z,
             y*z + x*w + y*z + x*w)
    }

    #[inline]
    pub fn z_dir(self) -> V3 {
        let Quat(V4{x, y, z, w}) = self;
        vec3(z*x + y*w + z*x + y*w,
             y*z - x*w + y*z - x*w,
             w*w - x*x - y*y + z*z)
    }

    #[inline]
    pub fn to_mat3(self) -> M3x3 {
        M3x3::from_cols(self.x_dir(), self.y_dir(), self.z_dir())
    }

    #[inline]
    pub fn nlerp(self, o: Quat, t: f32) -> Quat {
        Quat(self.0.nlerp(o.0, t))
    }

    #[inline]
    pub fn slerp(self, o: Quat, t: f32) -> Quat {
        Quat(self.0.slerp(o.0, t))
    }

    #[inline]
    pub fn nlerp_closer(self, o: Quat, t: f32) -> Quat {
        Quat(self.0.nlerp(if self.dot(o) < 0.0 { -o.0 } else { o.0 }, t))
    }

    #[inline]
    pub fn slerp_closer(self, o: Quat, t: f32) -> Quat {
        Quat(self.0.slerp(if self.0.dot(o.0) < 0.0 { -o.0 } else { o.0 }, t))
    }

    #[inline]
    pub fn axis_angle(axis: V3, angle: f32) -> Quat {
        let ha = angle * 0.5;
        Quat(V4::expand(axis * ha.sin(), ha.cos()))
    }

    #[inline]
    pub fn inverse(self) -> Quat {
        self.conj() / self.length_sq()
    }

    #[inline]
    pub fn shortest_arc(v0: V3, v1: V3) -> Quat {
        let v0 = v0.norm_or_zero();
        let v1 = v1.norm_or_zero();

        if v0.approx_eq(&v1) {
            return Quat::identity();
        }

        let c = v0.cross(v1);
        let d = v0.dot(v1);
        if d <= -1.0 {
            let a = v0.orth();
            Quat(V4::expand(a, 0.0))
        } else {
            let s = ((1.0 + d) * 2.0).sqrt();
            Quat(V4::expand(c / s, s / 2.0))
        }
    }
}


impl Mul<V3> for Quat {
    type Output = V3;
    #[inline]
    fn mul(self, o: V3) -> V3 {
        // this is slow and bad...
        self.to_mat3() * o
    }
}



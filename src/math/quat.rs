use math::vec::*;
use math::traits::*;
use math::mat::*;
use math::scalar::*;
use math::geom;
use std::ops::*;
use std::{mem, fmt, f32};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Quat(pub V4);

#[inline]
pub fn quat(x: f32, y: f32, z: f32, w: f32) -> Quat {
    Quat(vec4(x, y, z, w))
}

impl fmt::Display for Quat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "quat({}, {}, {}, {})", self.0.x, self.0.y, self.0.z, self.0.w)
    }
}

impl From<V4> for Quat { #[inline] fn from(v: V4) -> Quat { Quat(v) } }

impl AsRef<V4> for Quat { #[inline] fn as_ref(&    self) -> &    V4 { unsafe { mem::transmute(self) } } }
impl AsMut<V4> for Quat { #[inline] fn as_mut(&mut self) -> &mut V4 { unsafe { mem::transmute(self) } } }

impl Identity for Quat {
    #[inline] fn identity() -> Quat { quat(0.0, 0.0, 0.0, 1.0) }
}

impl Default for Quat {
    #[inline] fn default() -> Quat { quat(0.0, 0.0, 0.0, 1.0) }
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
    pub fn from_axis_angle(axis: V3, angle: f32) -> Quat {
        Quat(V4::expand(axis*(angle*0.5).sin(), (angle*0.5).cos()))
    }

    #[inline]
    pub fn conj(self) -> Quat {
        quat(-self.0.x, -self.0.y, -self.0.z, self.0.w)
    }

    #[inline]
    pub fn normalize(self) -> Option<Quat> {
        match self.0.normalize() {
            Some(n) => Some(Quat(n)),
            None => None
        }
    }

    #[inline]
    pub fn must_norm(self) -> Quat {
        Quat(self.0.normalize().unwrap())
    }

    #[inline]
    pub fn norm_or_zero(self) -> Quat {
        Quat(self.0.norm_or_zero())
    }

    #[inline]
    pub fn norm_or_identity(self) -> Quat {
        Quat(self.0.norm_or(0.0, 0.0, 0.0, 1.0))
    }

    #[inline]
    pub fn norm_or_q(self, q: Quat) -> Quat {
        Quat(self.0.norm_or_v(q.0))
    }

    #[inline]
    pub fn norm_or_v(self, v: V4) -> Quat {
        Quat(self.0.norm_or_v(v))
    }

    #[inline]
    pub fn norm_or(self, x: f32, y: f32, z: f32, w: f32) -> Quat {
        Quat(self.0.norm_or(x, y, z, w))
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
        let x2 = self.0.x + self.0.x;
        let y2 = self.0.y + self.0.y;
        let z2 = self.0.z + self.0.z;
        let xx = self.0.x * x2;
        let yx = self.0.y * x2;
        let yy = self.0.y * y2;
        let zx = self.0.z * x2;
        let zy = self.0.z * y2;
        let zz = self.0.z * z2;
        let wx = self.0.w * x2;
        let wy = self.0.w * y2;
        let wz = self.0.w * z2;

        mat3(1.0 - yy - zz, yx + wz, zx - wy,
             yx - wz, 1.0 - xx - zz, zy + wx,
             zx + wy, zy - wx, 1.0 - xx - yy)
        // M3x3::from_cols(self.x_dir(), self.y_dir(), self.z_dir())
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
        Quat(self.0.must_norm()).conj()
        // self.conj() / self.length_sq()
    }

    #[inline]
    pub fn shortest_arc(v0: V3, v1: V3) -> Quat {
        debug_assert!(dot(v0, v0) != 0.0);
        debug_assert!(dot(v1, v1) != 0.0);
        let v0 = v0.norm_or_zero();
        let v1 = v1.norm_or_zero();

        // if v0.approx_eq(&v1) {
            // return Quat::identity();
        // }

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

    pub fn virtual_track_ball(cop: V3, cor: V3, dir1: V3, dir2: V3) -> Quat {
        let normal = cor - cop;
        let fudge = 1.0 - normal.length()*0.25;
        let normal = normal.norm_or(0.0, 0.0, 1.0);
        let plane = geom::Plane::from_norm_and_point(normal, cor);

        let u = (plane.intersect_with_line(cop, cop+dir1) - cor) * fudge;
        let v = (plane.intersect_with_line(cop, cop+dir2) - cor) * fudge;

        let mu = u.length();
        let mv = v.length();
        Quat::shortest_arc(if mu > 1.0 { u / mu } else { u - normal * (1.0 - mu*mu).sqrt() },
                           if mv > 1.0 { v / mv } else { v - normal * (1.0 - mv*mv).sqrt() })
    }

    #[inline]
    pub fn from_yaw_pitch_roll(yaw: f32, pitch: f32, roll: f32) -> Quat {
        Quat::from_axis_angle(vec3(0.0, 0.0, 1.0), yaw) *
        Quat::from_axis_angle(vec3(1.0, 0.0, 0.0), pitch) *
        Quat::from_axis_angle(vec3(0.0, 1.0, 0.0), roll)
    }

    #[inline]
    pub fn yaw_pitch_roll(&self) -> (f32, f32, f32) {
        // could optimize
        (self.yaw(), self.pitch(), self.roll())
    }

    #[inline]
    pub fn yaw(self) -> f32 {
        let v = self.y_dir();
        if v.x == 0.0 && v.y == 0.0 { 0.0 }
        else { (-v.x).atan2(v.y) }
    }

    #[inline]
    pub fn pitch(self) -> f32 {
        let v = self.y_dir();
        v.z.atan2((v.x*v.x + v.y*v.y).sqrt())
    }

    #[inline]
    pub fn roll(self) -> f32 {
        let q = self;
        let q = Quat::from_axis_angle(vec3(0.0, 0.0, 1.0), -q.yaw()) * q;
        let q = Quat::from_axis_angle(vec3(1.0, 0.0, 0.0), -q.pitch()) * q;
        let v = q.x_dir();
        (-v.z).atan2(v.x)
    }
}


impl Mul<V3> for Quat {
    type Output = V3;
    #[inline]
    fn mul(self, o: V3) -> V3 {
        let V3 { x: vx, y: vy, z: vz } = o;
        let Quat(V4 { x: qx, y: qy, z: qz, w: qw }) = self;
        let ix =  qw*vx + qy*vz - qz*vy;
        let iy =  qw*vy + qz*vx - qx*vz;
        let iz =  qw*vz + qx*vy - qy*vx;
        let iw = -qx*vx - qy*vy - qz*vz;
        vec3(ix*qw + iw*-qx + iy*-qz - iz*-qy,
             iy*qw + iw*-qy + iz*-qx - ix*-qz,
             iz*qw + iw*-qz + ix*-qy - iy*-qx)
        // this is slow and bad...
        // self.to_mat3() * o
    }
}




use crate::mat::*;
use crate::plane::*;
use crate::traits::*;
use crate::vec::*;

use std::ops::*;

#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Quat(pub V4);

#[inline]
pub fn quat(x: f32, y: f32, z: f32, w: f32) -> Quat {
    Quat(vec4(x, y, z, w))
}

impl std::fmt::Display for Quat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "quat({}, {}, {}, {})",
            self.0.x(),
            self.0.y(),
            self.0.z(),
            self.0.w()
        )
    }
}

impl From<V4> for Quat {
    #[inline]
    fn from(v: V4) -> Quat {
        Quat(v)
    }
}
impl From<Quat> for V4 {
    #[inline]
    fn from(q: Quat) -> V4 {
        q.0
    }
}

impl AsRef<V4> for Quat {
    #[inline]
    fn as_ref(&self) -> &V4 {
        &self.0
    }
}
impl AsMut<V4> for Quat {
    #[inline]
    fn as_mut(&mut self) -> &mut V4 {
        &mut self.0
    }
}

impl AsRef<Quat> for V4 {
    #[inline]
    fn as_ref(&self) -> &Quat {
        unsafe { &*(self as *const V4 as *const Quat) }
    }
}

impl AsMut<Quat> for V4 {
    #[inline]
    fn as_mut(&mut self) -> &mut Quat {
        unsafe { &mut *(self as *mut V4 as *mut Quat) }
    }
}

impl From<M3x3> for Quat {
    #[inline]
    fn from(m: M3x3) -> Self {
        m.to_quat()
    }
}

impl From<Quat> for M3x3 {
    #[inline]
    fn from(q: Quat) -> Self {
        q.to_mat3()
    }
}

impl From<Quat> for (f32, f32, f32, f32) {
    #[inline]
    fn from(q: Quat) -> Self {
        q.0.into()
    }
}

impl From<(f32, f32, f32, f32)> for Quat {
    #[inline]
    fn from(q: (f32, f32, f32, f32)) -> Self {
        Quat(q.into())
    }
}
impl From<[f32; 4]> for Quat {
    #[inline]
    fn from(q: [f32; 4]) -> Self {
        Quat(q.into())
    }
}

impl From<Quat> for [f32; 4] {
    #[inline]
    fn from(q: Quat) -> Self {
        q.0.into()
    }
}

impl Default for Quat {
    #[inline]
    fn default() -> Quat {
        Quat::IDENTITY
    }
}

impl Mul<f32> for Quat {
    type Output = Quat;
    #[inline]
    fn mul(self, o: f32) -> Quat {
        Quat(self.0 * o)
    }
}

impl Div<f32> for Quat {
    type Output = Quat;
    #[inline]
    fn div(self, o: f32) -> Quat {
        Quat(self.0 * (1.0 / o))
    }
}

impl Add for Quat {
    type Output = Quat;
    #[inline]
    fn add(self, o: Quat) -> Quat {
        Quat(self.0 + o.0)
    }
}

impl Sub for Quat {
    type Output = Quat;
    #[inline]
    fn sub(self, o: Quat) -> Quat {
        Quat(self.0 - o.0)
    }
}

impl Neg for Quat {
    type Output = Quat;
    #[inline]
    fn neg(self) -> Quat {
        Quat(-self.0)
    }
}

impl Index<usize> for Quat {
    type Output = f32;
    #[inline]
    fn index(&self, index: usize) -> &f32 {
        self.0.index(index)
    }
}

impl IndexMut<usize> for Quat {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut f32 {
        self.0.index_mut(index)
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

impl Quat {
    #[inline]
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Quat {
        quat(x, y, z, w)
    }

    #[inline]
    pub fn angle(self) -> f32 {
        self.0.w().acos() * 2.0
    }

    #[inline]
    pub fn axis(self) -> V3 {
        self.axis_angle().0
    }

    #[inline]
    pub fn axis_angle(self) -> (V3, f32) {
        let angle = self.angle();
        if approx_zero(angle) {
            (vec3(1.0, 0.0, 0.0), angle)
        } else {
            (V3::from(self.0) / (angle * 0.5).sin(), angle)
        }
    }

    #[inline]
    pub fn from_basis(tangent: V3, bitangent: V3, normal: V3) -> Self {
        M3x3::from_cols(tangent, bitangent, normal).to_quat()
    }

    #[inline]
    pub fn from_axis_angle(axis: V3, angle: f32) -> Quat {
        Quat(V4::expand(axis * (angle * 0.5).sin(), (angle * 0.5).cos()))
    }

    #[inline]
    pub fn conj(self) -> Quat {
        simd_match! {
            "sse2" => unsafe {
                use std::arch::{x86_64 as sse};
                const SIGNBITS: sse::__m128 =
                    const_simd_mask![0x8000_0000u32, 0x8000_0000u32, 0x8000_0000u32, 0];
                Quat(V4(sse::_mm_xor_ps(SIGNBITS, (self.0).0)))
            },
            _ => {
                quat(-self.0.x, -self.0.y, -self.0.z, self.0.w)
            }
        }
    }

    #[inline]
    pub fn normalize(self) -> Option<Quat> {
        match self.0.normalize() {
            Some(n) => Some(Quat(n)),
            None => None,
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

    #[inline]
    pub fn length_sq(self) -> f32 {
        self.0.length_sq()
    }
    #[inline]
    pub fn length(self) -> f32 {
        self.0.length()
    }
    #[inline]
    pub fn dot(self, o: Quat) -> f32 {
        self.0.dot(o.0)
    }

    #[inline]
    pub fn tup(self) -> (f32, f32, f32, f32) {
        self.0.tup()
    }

    #[inline]
    pub fn arr(self) -> [f32; 4] {
        self.0.arr()
    }

    #[inline]
    pub fn x_dir(self) -> V3 {
        let [x, y, z, w] = self.arr();
        vec3(
            w * w + x * x - y * y - z * z,
            (x * y + z * w) * 2.0,
            (z * x - y * w) * 2.0,
        )
    }

    #[inline]
    pub fn y_dir(self) -> V3 {
        let [x, y, z, w] = self.arr();
        vec3(
            (x * y - z * w) * 2.0,
            w * w - x * x + y * y - z * z,
            (y * z + x * w) * 2.0,
        )
    }

    #[inline]
    pub fn z_dir(self) -> V3 {
        let [x, y, z, w] = self.arr();
        vec3(
            (z * x + y * w) * 2.0,
            (y * z - x * w) * 2.0,
            w * w - x * x - y * y + z * z,
        )
    }

    #[inline]
    pub fn to_mat3(self) -> M3x3 {
        let [x, y, z, w] = self.0.arr();
        let x2 = x + x;
        let y2 = y + y;
        let z2 = z + z;

        let xx = x * x2;
        let yx = y * x2;
        let yy = y * y2;

        let zx = z * x2;
        let zy = z * y2;
        let zz = z * z2;

        let wx = w * x2;
        let wy = w * y2;
        let wz = w * z2;

        mat3(
            1.0 - yy - zz,
            yx + wz,
            zx - wy,
            yx - wz,
            1.0 - xx - zz,
            zy + wx,
            zx + wy,
            zy - wx,
            1.0 - xx - yy,
        )
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
    pub fn inverse(self) -> Quat {
        Quat(self.0.must_norm()).conj()
    }

    #[inline]
    pub fn shortest_arc(v0: V3, v1: V3) -> Quat {
        debug_assert!(dot(v0, v0) != 0.0);
        debug_assert!(dot(v1, v1) != 0.0);
        let v0 = v0.norm_or_zero();
        let v1 = v1.norm_or_zero();

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
        let fudge = 1.0 - normal.length() * 0.25;
        let normal = normal.norm_or(0.0, 0.0, 1.0);
        let plane = Plane::from_norm_and_point(normal, cor);

        let u = (plane.intersect_with_line(cop, cop + dir1) - cor) * fudge;
        let v = (plane.intersect_with_line(cop, cop + dir2) - cor) * fudge;

        let mu = u.length();
        let mv = v.length();
        Quat::shortest_arc(
            if mu > 1.0 {
                u / mu
            } else {
                u - normal * (1.0 - mu * mu).sqrt()
            },
            if mv > 1.0 {
                v / mv
            } else {
                v - normal * (1.0 - mv * mv).sqrt()
            },
        )
    }

    #[inline]
    pub fn from_yaw_pitch_roll(yaw: f32, pitch: f32, roll: f32) -> Quat {
        (Quat::from_axis_angle(vec3(0.0, 0.0, 1.0), yaw)
            * Quat::from_axis_angle(vec3(1.0, 0.0, 0.0), pitch)
            * Quat::from_axis_angle(vec3(0.0, 1.0, 0.0), roll))
        .must_norm()
    }

    #[inline]
    pub fn yaw_pitch_roll(&self) -> (f32, f32, f32) {
        // could optimize
        (self.yaw(), self.pitch(), self.roll())
    }

    #[inline]
    pub fn yaw(self) -> f32 {
        let v = self.y_dir();
        if v.x() == 0.0 && v.y() == 0.0 {
            0.0
        } else {
            (-v.x()).atan2(v.y())
        }
    }

    #[inline]
    pub fn pitch(self) -> f32 {
        let v = self.y_dir();
        v.z().atan2((v.x() * v.x() + v.y() * v.y()).sqrt())
    }

    #[inline]
    pub fn roll(self) -> f32 {
        let q = self;
        let q = Quat::from_axis_angle(vec3(0.0, 0.0, 1.0), -q.yaw()) * q;
        let q = Quat::from_axis_angle(vec3(1.0, 0.0, 0.0), -q.pitch()) * q;
        let v = q.x_dir();
        (-v.z()).atan2(v.x())
    }

    #[inline]
    pub fn to_arr(self) -> [f32; 4] {
        self.0.into()
    }

    #[inline]
    pub const fn zero() -> Self {
        Self::ZERO
    }

    #[inline]
    pub const fn identity() -> Self {
        Self::IDENTITY
    }

    pub const ZERO: Quat = Quat(V4::ZERO);
    pub const IDENTITY: Quat = Quat(V4::POS_W);
}

// impl Lerp for Quat {
//     #[inline]
//     fn lerp(self, o: Self, t: f32) -> Self {
//         Quat(self.0.slerp(o.0, t))
//     }
// }

impl Mul<V3> for Quat {
    type Output = V3;
    // #[inline]
    fn mul(self, o: V3) -> V3 {
        simd_match! {
            "sse2" => {
                super::simd::quat_rot3(self, o)
            },
            _ => {
                naive_quat_rot3(self, o)
            }
        }
    }
}

#[inline]
#[allow(dead_code)]
pub(crate) fn naive_quat_rot3(q: Quat, o: V3) -> V3 {
    let (vx, vy, vz) = o.tup();
    let (qx, qy, qz, qw) = q.tup();
    let ix = qw * vx + qy * vz - qz * vy;
    let iy = qw * vy + qz * vx - qx * vz;
    let iz = qw * vz + qx * vy - qy * vx;
    let iw = -qx * vx - qy * vy - qz * vz;
    vec3(
        ix * qw - iw * qx - iy * qz + iz * qy,
        iy * qw - iw * qy - iz * qx + ix * qz,
        iz * qw - iw * qz - ix * qy + iy * qx,
    )
}
#[inline]
#[allow(dead_code)]
pub(crate) fn naive_quat_mul_quat(q: Quat, o: Quat) -> Quat {
    let (sx, sy, sz, sw) = q.tup();
    let (ox, oy, oz, ow) = o.tup();
    Quat::new(
        sx * ow + sw * ox + sy * oz - sz * oy,
        sy * ow + sw * oy + sz * ox - sx * oz,
        sz * ow + sw * oz + sx * oy - sy * ox,
        sw * ow - sx * ox - sy * oy - sz * oz,
    )
}

impl Mul<Quat> for Quat {
    type Output = Quat;
    #[inline]
    fn mul(self, other: Quat) -> Quat {
        simd_match! {
            "sse2" => { super::simd::quat_mul_quat(self, other) },
            _ => { naive_quat_mul_quat(self, o) }
        }
    }
}

impl ApproxEq for Quat {
    #[inline]
    fn approx_zero_e(&self, e: f32) -> bool {
        self.0.approx_zero_e(e)
    }

    #[inline]
    fn approx_eq_e(&self, o: &Self, e: f32) -> bool {
        self.0.approx_eq_e(&o.0, e)
    }
}

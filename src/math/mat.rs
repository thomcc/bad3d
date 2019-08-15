#![allow(clippy::too_many_arguments, clippy::op_ref)]
use crate::math::quat::*;
use crate::math::traits::*;
use crate::math::vec::*;
use std::{f32, fmt, ops::*};

#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct M2x2 {
    pub x: V2,
    pub y: V2,
}

#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct M3x3 {
    pub x: V3,
    pub y: V3,
    pub z: V3,
}

#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct M4x4 {
    pub x: V4,
    pub y: V4,
    pub z: V4,
    pub w: V4,
}

impl M2x2 {
    pub const IDENTITY: M2x2 = M2x2 {
        x: V2 { x: 1.0, y: 0.0 },
        y: V2 { x: 0.0, y: 1.0 },
    };
}

impl M3x3 {
    pub const IDENTITY: M3x3 = M3x3 {
        x: V3::POS_X,
        y: V3::POS_Y,
        z: V3::POS_Z,
    };
}

impl M4x4 {
    pub const IDENTITY: M4x4 = M4x4 {
        x: V4::POS_X,
        y: V4::POS_Y,
        z: V4::POS_Z,
        w: V4::POS_W,
    };
}

impl M2x2 {
    pub const ZERO: M2x2 = M2x2 {
        x: V2 { x: 0.0, y: 0.0 },
        y: V2 { x: 0.0, y: 0.0 },
    };
}

impl M3x3 {
    #[rustfmt::skip]
    pub const ZERO: M3x3 = M3x3 {
        x: V3::ZERO,
        y: V3::ZERO,
        z: V3::ZERO,
    };
}

impl M4x4 {
    pub const ZERO: M4x4 = M4x4 {
        x: V4::ZERO,
        y: V4::ZERO,
        z: V4::ZERO,
        w: V4::ZERO,
    };
}

impl fmt::Display for M2x2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "mat2({}, {})", self.x, self.y)
    }
}

impl fmt::Display for M3x3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "mat3({}, {}, {})", self.x, self.y, self.z)
    }
}

impl fmt::Display for M4x4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "mat4({}, {}, {}, {})", self.x, self.y, self.z, self.w)
    }
}

impl Default for M2x2 {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}
impl Default for M3x3 {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}
impl Default for M4x4 {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

#[inline]
pub const fn mat2(m00: f32, m01: f32, m10: f32, m11: f32) -> M2x2 {
    M2x2::new(m00, m01, m10, m11)
}

#[inline]
#[rustfmt::skip]
pub fn mat3(
    m00: f32, m01: f32, m02: f32,
    m10: f32, m11: f32, m12: f32,
    m20: f32, m21: f32, m22: f32,
) -> M3x3 {
    M3x3::new(m00, m01, m02, m10, m11, m12, m20, m21, m22)
}

#[inline]
#[rustfmt::skip]
pub fn mat4(
    m00: f32, m01: f32, m02: f32, m03: f32,
    m10: f32, m11: f32, m12: f32, m13: f32,
    m20: f32, m21: f32, m22: f32, m23: f32,
    m30: f32, m31: f32, m32: f32, m33: f32,
) -> M4x4 {
    M4x4::new(
        m00, m01, m02, m03,
        m10, m11, m12, m13,
        m20, m21, m22, m23,
        m30, m31, m32, m33,
    )
}

impl<'a> Mul<V2> for &'a M2x2 {
    type Output = V2;
    #[inline]
    fn mul(self, v: V2) -> V2 {
        self.x * v.x + self.y * v.y
    }
}

impl<'a> Mul<V3> for &'a M3x3 {
    type Output = V3;
    #[inline]
    fn mul(self, v: V3) -> V3 {
        self.x * v.xxx() + self.y * v.yyy() + self.z * v.zzz()
    }
}

impl<'a> Mul<V4> for &'a M4x4 {
    type Output = V4;
    #[inline]
    fn mul(self, v: V4) -> V4 {
        self.x * v.x() + self.y * v.y() + self.z * v.z() + self.w * v.w()
    }
}

impl Mul<V2> for M2x2 {
    type Output = V2;
    #[inline]
    fn mul(self, v: V2) -> V2 {
        (&self) * v
    }
}
impl Mul<V3> for M3x3 {
    type Output = V3;
    #[inline]
    fn mul(self, v: V3) -> V3 {
        (&self) * v
    }
}
impl Mul<V4> for M4x4 {
    type Output = V4;
    #[inline]
    fn mul(self, v: V4) -> V4 {
        (&self) * v
    }
}

macro_rules! impl_ref_operators {
    ($OperTrait:ident :: $func:ident, $lhs:ty, $rhs:ty) => {
        impl<'a> $OperTrait<$rhs> for &'a $lhs {
            type Output = <$lhs as $OperTrait<$rhs>>::Output;
            #[inline]
            fn $func(self, other: $rhs) -> <$lhs as $OperTrait<$rhs>>::Output {
                $OperTrait::$func(*self, other)
            }
        }

        impl<'a> $OperTrait<&'a $rhs> for $lhs {
            type Output = <$lhs as $OperTrait<$rhs>>::Output;

            #[inline]
            fn $func(self, other: &'a $rhs) -> <$lhs as $OperTrait<$rhs>>::Output {
                $OperTrait::$func(self, *other)
            }
        }

        impl<'a, 'b> $OperTrait<&'a $rhs> for &'b $lhs {
            type Output = <$lhs as $OperTrait<$rhs>>::Output;

            #[inline]
            fn $func(self, other: &'a $rhs) -> <$lhs as $OperTrait<$rhs>>::Output {
                $OperTrait::$func(*self, *other)
            }
        }
    };
}

macro_rules! do_mat_boilerplate {
    ($Mn: ident { $($field: ident : $index: expr),+ },
     $Vn: ident, $size: expr,
     $elems: expr) =>
    {
        impl AsRef<[$Vn; $size]> for $Mn {
            #[inline]
            fn as_ref(&self) -> &[$Vn; $size] {
                unsafe { &*(self as *const $Mn as *const [$Vn; $size]) }
            }
        }

        impl AsMut<[$Vn; $size]> for $Mn {
            #[inline]
            fn as_mut(&mut self) -> &mut [$Vn; $size] {
                unsafe { &mut *(self as *mut $Mn as *mut [$Vn; $size]) }
            }
        }

        impl From<$Mn> for [$Vn; $size] {
            #[inline]
            fn from(m: $Mn) -> [$Vn; $size] {
                *m.as_ref()
            }
        }

        impl From<[$Vn; $size]> for $Mn {
            #[inline]
            fn from(m: [$Vn; $size]) -> $Mn {
                $Mn { $($field: m[$index]),+ }
            }
        }

        impl AsRef<[$Vn]> for $Mn {
            #[inline]
            fn as_ref(&self) -> &[$Vn] {
                let m: &[$Vn; $size] = self.as_ref();
                &m[..]
            }
        }

        impl AsMut<[$Vn]> for $Mn {
            #[inline]
            fn as_mut(&mut self) -> &mut [$Vn] {
                let m: &mut[$Vn; $size] = self.as_mut();
                &mut m[..]
            }
        }

        impl $Mn {
            #[inline] pub fn as_slice(&self) -> &[$Vn] { self.as_ref() }
            #[inline] pub fn as_mut_slice(&mut self) -> &mut [$Vn] { self.as_mut() }
        }

        impl Index<usize> for $Mn {
            type Output = $Vn;
            #[inline] fn index(&self, i: usize) -> &$Vn { &self.as_slice()[i] }
        }

        impl IndexMut<usize> for $Mn {
            #[inline] fn index_mut(&mut self, i: usize) -> &mut $Vn { &mut self.as_mut_slice()[i] }
        }

        impl Add for $Mn {
            type Output = $Mn;
            #[inline] fn add(self, o: $Mn) -> $Mn { $Mn { $($field: self.$field + o.$field),+ } }
        }

        impl Sub for $Mn {
            type Output = $Mn;
            #[inline] fn sub(self, o: $Mn) -> $Mn { $Mn { $($field: self.$field - o.$field),+ } }
        }
        impl Mul<f32> for $Mn {
            type Output = $Mn;
            #[inline] fn mul(self, rhs: f32) -> $Mn { $Mn { $($field: self.$field * rhs),+ } }
        }

        impl Div<f32> for $Mn {
            type Output = $Mn;
            #[inline]
            #[allow(clippy::suspicious_arithmetic_impl)]
            fn div(self, rhs: f32) -> $Mn {
                debug_assert!(rhs != 0.0);
                self * (1.0 / rhs)
            }
        }

        impl_ref_operators!(Mul::mul, $Mn, $Mn);
        impl_ref_operators!(Add::add, $Mn, $Mn);
        impl_ref_operators!(Sub::sub, $Mn, $Mn);
        impl_ref_operators!(Mul::mul, $Mn, f32);
        impl_ref_operators!(Div::div, $Mn, f32);

        // TODO: Does the optimizer do an ok job here?
        impl<'a> MulAssign<&'a $Mn> for $Mn {
            #[inline] fn mul_assign(&mut self, rhs: &'a $Mn) { let res = self.mul(rhs); *self = res; }
        }

        impl<'a> AddAssign<&'a $Mn> for $Mn {
            #[inline] fn add_assign(&mut self, rhs: &'a $Mn) { let res = self.add(rhs); *self = res; }
        }

        impl<'a> SubAssign<&'a $Mn> for $Mn {
            #[inline] fn sub_assign(&mut self, rhs: &'a $Mn) { let res = self.sub(rhs); *self = res; }
        }

        impl AddAssign<$Mn> for $Mn {
            #[inline] fn add_assign(&mut self, rhs: $Mn) { self.add_assign(&rhs); }
        }

        impl SubAssign<$Mn> for $Mn {
            #[inline] fn sub_assign(&mut self, rhs: $Mn) { self.sub_assign(&rhs) }
        }

        impl MulAssign<$Mn> for $Mn {
            #[inline] fn mul_assign(&mut self, rhs: $Mn) { self.mul_assign(&rhs); }
        }

        impl MulAssign<f32> for $Mn {
            #[inline] fn mul_assign(&mut self, rhs: f32) { let res = self.mul(rhs); *self = res; }
        }

        impl DivAssign<f32> for $Mn {
            #[inline] fn div_assign(&mut self, rhs: f32) { let res = self.div(rhs); *self = res; }
        }
    }
}

do_mat_boilerplate!(M2x2 { x: 0, y: 1 }, V2, 2, 4);
do_mat_boilerplate!(M3x3 { x: 0, y: 1, z: 2 }, V3, 3, 9);
do_mat_boilerplate!(M4x4 { x: 0, y: 1, z: 2, w: 3 }, V4, 4, 16);

impl Mul<M2x2> for M2x2 {
    type Output = M2x2;
    #[inline]
    fn mul(self, rhs: M2x2) -> M2x2 {
        M2x2 {
            x: self * rhs.x,
            y: self * rhs.y,
        }
    }
}

impl Mul<M4x4> for M4x4 {
    type Output = M4x4;
    #[inline]
    fn mul(self, rhs: M4x4) -> M4x4 {
        M4x4 {
            x: self * rhs.x,
            y: self * rhs.y,
            z: self * rhs.z,
            w: self * rhs.w,
        }
    }
}
impl Mul<M3x3> for M3x3 {
    type Output = M3x3;
    #[inline]
    fn mul(self, rhs: M3x3) -> M3x3 {
        Self {
            x: self * rhs.x,
            y: self * rhs.y,
            z: self * rhs.z,
        }
    }
}

impl M2x2 {
    #[inline]
    pub const fn new(xx: f32, xy: f32, yx: f32, yy: f32) -> M2x2 {
        M2x2 {
            x: V2 { x: xx, y: xy },
            y: V2 { x: yx, y: yy },
        }
    }

    #[inline]
    pub const fn from_cols(x: V2, y: V2) -> M2x2 {
        M2x2 { x, y }
    }

    #[inline]
    pub const fn from_rows(x: V2, y: V2) -> M2x2 {
        M2x2::new(x.x, y.x, x.y, y.y)
    }

    #[inline]
    pub fn to_arr(self) -> [[f32; 2]; 2] {
        self.into()
    }

    #[inline]
    pub fn diagonal(&self) -> V2 {
        vec2(self.x.x, self.y.y)
    }
    #[inline]
    pub fn determinant(&self) -> f32 {
        self.x.x * self.y.y - self.x.y * self.y.x
    }
    #[inline]
    pub fn adjugate(&self) -> M2x2 {
        M2x2::new(self.y.y, -self.x.y, -self.y.x, self.x.x)
    }

    #[inline]
    pub fn transpose(&self) -> M2x2 {
        M2x2::from_rows(self.x, self.y)
    }

    #[inline]
    pub fn row(&self, i: usize) -> V2 {
        V2::new(self.x[i], self.y[i])
    }

    #[inline]
    pub fn col(&self, i: usize) -> V2 {
        self[i]
    }

    #[inline]
    pub fn inverse(&self) -> Option<Self> {
        let d = self.determinant();
        if d == 0.0 {
            None
        } else {
            Some(self.adjugate() * (1.0 / d))
        }
    }

    #[inline]
    pub const fn identity() -> Self {
        Self::IDENTITY
    }

    #[inline]
    pub const fn zero() -> Self {
        Self::ZERO
    }
}

impl M3x3 {
    #[inline]
    #[rustfmt::skip]
    pub fn new(
        xx: f32, xy: f32, xz: f32,
        yx: f32, yy: f32, yz: f32,
        zx: f32, zy: f32, zz: f32,
    ) -> Self {
        Self {
            x: vec3(xx, xy, xz),
            y: vec3(yx, yy, yz),
            z: vec3(zx, zy, zz),
        }
    }

    #[inline]
    pub const fn from_cols(x: V3, y: V3, z: V3) -> M3x3 {
        M3x3 { x, y, z }
    }

    #[inline]
    pub fn from_rows(x: V3, y: V3, z: V3) -> M3x3 {
        M3x3::new(x.x(), y.x(), z.x(), x.y(), y.y(), z.y(), x.z(), y.z(), z.z())
    }

    #[inline]
    pub fn to_quat(&self) -> Quat {
        let [xx, xy, xz] = self.x.arr();
        let [yx, yy, yz] = self.y.arr();
        let [zx, zy, zz] = self.z.arr();
        let mag_w = xx + yy + zz;

        let (mag_zw, pre_zw, post_zw) = if mag_w > zz {
            (mag_w, vec3(1.0, 1.0, 1.0), quat(0.0, 0.0, 0.0, 1.0))
        } else {
            (zz, vec3(-1.0, -1.0, 1.0), quat(0.0, 0.0, 1.0, 0.0))
        };

        let (mag_xy, pre_xy, post_xy) = if xx > yy {
            (xx, vec3(1.0, -1.0, -1.0), quat(1.0, 0.0, 0.0, 0.0))
        } else {
            (yy, vec3(-1.0, 1.0, -1.0), quat(0.0, 1.0, 0.0, 0.0))
        };

        let (pre, post) = if mag_zw > mag_xy {
            (pre_zw, post_zw)
        } else {
            (pre_xy, post_xy)
        };

        let t = pre.x() * xx + pre.y() * yy + pre.z() * zz + 1.0;
        let s = 0.5 / t.sqrt();
        let qp = quat(
            (pre.y() * yz - pre.z() * zy) * s,
            (pre.z() * zx - pre.x() * xz) * s,
            (pre.x() * xy - pre.y() * yx) * s,
            t * s,
        );
        debug_assert!(approx_eq(qp.length(), 1.0));
        qp * post
    }

    #[inline]
    pub fn to_mat4(&self) -> M4x4 {
        M4x4 {
            x: V4::expand(self.x, 0.0),
            y: V4::expand(self.y, 0.0),
            z: V4::expand(self.z, 0.0),
            w: vec4(0.0, 0.0, 0.0, 1.0),
        }
    }

    #[inline]
    pub fn to_arr(self) -> [[f32; 3]; 3] {
        self.into()
    }

    #[inline]
    pub fn diagonal(&self) -> V3 {
        vec3(self.x.x(), self.y.y(), self.z.z())
    }

    #[inline]
    pub fn determinant(&self) -> f32 {
        self.x.x() * (self.y.y() * self.z.z() - self.z.y() * self.y.z())
            + self.x.y() * (self.y.z() * self.z.x() - self.z.z() * self.y.x())
            + self.x.z() * (self.y.x() * self.z.y() - self.z.x() * self.y.y())
    }

    #[inline]
    pub fn adjugate(&self) -> M3x3 {
        M3x3 {
            x: vec3(
                self.y.y() * self.z.z() - self.z.y() * self.y.z(),
                self.z.y() * self.x.z() - self.x.y() * self.z.z(),
                self.x.y() * self.y.z() - self.y.y() * self.x.z(),
            ),
            y: vec3(
                self.y.z() * self.z.x() - self.z.z() * self.y.x(),
                self.z.z() * self.x.x() - self.x.z() * self.z.x(),
                self.x.z() * self.y.x() - self.y.z() * self.x.x(),
            ),
            z: vec3(
                self.y.x() * self.z.y() - self.z.x() * self.y.y(),
                self.z.x() * self.x.y() - self.x.x() * self.z.y(),
                self.x.x() * self.y.y() - self.y.x() * self.x.y(),
            ),
        }
    }

    #[inline]
    pub fn transpose(&self) -> M3x3 {
        M3x3::from_rows(self.x, self.y, self.z)
    }

    #[inline]
    pub fn row(&self, i: usize) -> V3 {
        vec3(self.x[i], self.y[i], self.z[i])
    }

    #[inline]
    pub fn col(&self, i: usize) -> V3 {
        self[i]
    }

    #[inline]
    pub fn inverse(&self) -> Option<Self> {
        let d = self.determinant();
        if d == 0.0 {
            None
        } else {
            Some(self.adjugate() * (1.0 / d))
        }
    }

    #[inline]
    pub const fn identity() -> Self {
        Self::IDENTITY
    }

    #[inline]
    pub const fn zero() -> Self {
        Self::ZERO
    }

    #[inline]
    pub fn is_symmetric(&self) -> bool {
        // Should this be strict comparison?
        self.x.y() == self.y.x() && self.x.z() == self.z.x() && self.y.z() == self.z.y()
    }

    #[inline]
    pub fn is_approx_symmetric(&self) -> bool {
        // Should this be strict comparison?
        approx_eq(self.x.y(), self.y.x()) && approx_eq(self.x.z(), self.z.x()) && approx_eq(self.y.z(), self.z.y())
    }

    pub fn diagonalizer(&self) -> Quat {
        debug_assert!(
            self.is_approx_symmetric(),
            "no diagonalizer for asymmetric matrix {:?}",
            self
        );
        let max_steps = 24; // Defo won't need this many.
        let mut q = Quat::IDENTITY;
        for _ in 0..max_steps {
            let qm = q.to_mat3();
            let diag = qm.transpose() * self * qm;
            let off_diag = vec3(diag.y.z(), diag.x.z(), diag.x.y());
            let off_mag = off_diag.abs();
            let k = off_mag.max_index();
            // We shouldn't need epsilon here, we'll just do
            // another iteration or so, and the precision is probably worth it.
            // (TODO: is it?)
            if off_diag[k] == 0.0 {
                break;
            }
            let k1 = (k + 1) % 3;
            let k2 = (k + 2) % 3;

            let thet = (diag[k2][k2] - diag[k1][k1]) / (2.0 * off_diag[k]);

            let sgn = thet.signum(); // if thet_val > 0.0 { 1.0 } else { -1.0 };
            let thet = thet * sgn;
            // Use the more accurate formula if we're close.
            let t2p1 = if thet < crate::math::scalar::DEFAULT_EPSILON {
                (thet * thet + 1.0).sqrt()
            } else {
                thet
            };
            // sign(t) / (abs(t) * sqrt(t^2 + 1))
            let t = sgn / (thet + t2p1);

            let cosine = 1.0 / (t * t + 1.0);
            if cosine == 1.0 {
                // Hit numeric precision limit.
                break;
            }
            let mut jacobi_rot = Quat::zero();
            // Use half angle identity: sin(a / 2) == sqrt((1 - cos(a))/2)
            let axis_val = sgn * ((1.0 - cosine) * 0.5).sqrt();
            // Negated to go from the matrix to the diagonal and not vice versa.
            jacobi_rot[k] = -axis_val;
            *jacobi_rot.0.mw() = (1.0 - axis_val * axis_val).sqrt();
            if jacobi_rot.0.w() == 1.0 {
                // Hit numeric precision limit.
                break;
            }
            q = (q * jacobi_rot).normalize().unwrap();
        }

        // Not sure if fixing the eigenval order here is
        // worth the trouble...
        let h = f32::consts::FRAC_1_SQRT_2;
        // should optimize...
        let eigen_cmp = |q: Quat, a: usize, b: usize| -> bool {
            let qm = q.to_mat3();
            let es = (qm.transpose() * self * qm).diagonal();
            es[a] < es[b]
        };

        if eigen_cmp(q, 0, 2) {
            q *= quat(0.0, h, 0.0, h);
        }
        if eigen_cmp(q, 1, 2) {
            q *= quat(h, 0.0, 0.0, h);
        }
        if eigen_cmp(q, 0, 1) {
            q *= quat(0.0, 0.0, h, h);
        }

        if q.x_dir().z() < 0.0 {
            q *= quat(1.0, 0.0, 0.0, 0.0);
        }
        if q.y_dir().y() < 0.0 {
            q *= quat(0.0, 0.0, 1.0, 0.0);
        }
        if q.0.w() < 0.0 {
            q = -q;
        }
        q
    }
}

impl M4x4 {
    #[inline]
    #[rustfmt::skip]
    pub fn new(
        xx: f32, xy: f32, xz: f32, xw: f32,
        yx: f32, yy: f32, yz: f32, yw: f32,
        zx: f32, zy: f32, zz: f32, zw: f32,
        wx: f32, wy: f32, wz: f32, ww: f32,
    ) -> M4x4 {
        M4x4 {
            x: vec4(xx, xy, xz, xw),
            y: vec4(yx, yy, yz, yw),
            z: vec4(zx, zy, zz, zw),
            w: vec4(wx, wy, wz, ww),
        }
    }

    #[inline]
    pub const fn from_cols(x: V4, y: V4, z: V4, w: V4) -> M4x4 {
        M4x4 { x, y, z, w }
    }

    #[inline]
    #[rustfmt::skip]
    pub fn from_rows(x: V4, y: V4, z: V4, w: V4) -> M4x4 {
        mat4(
            x.x(), y.x(), z.x(), w.x(),
            x.y(), y.y(), z.y(), w.y(),
            x.z(), y.z(), z.z(), w.z(),
            x.w(), y.w(), z.w(), w.w(),
        )
    }

    #[inline]
    pub fn to_arr(self) -> [[f32; 4]; 4] {
        self.into()
    }

    #[inline]
    pub fn diagonal(&self) -> V4 {
        vec4(self.x.x(), self.y.y(), self.z.z(), self.w.w())
    }

    #[inline]
    pub fn determinant(&self) -> f32 {
        let [xx, xy, xz, xw] = self.x.arr();
        let [yx, yy, yz, yw] = self.y.arr();
        let [zx, zy, zz, zw] = self.z.arr();
        let [wx, wy, wz, ww] = self.w.arr();
        xx * (yy * zz * ww + wy * yz * zw + zy * wz * yw - yy * wz * zw - zy * yz * ww - wy * zz * yw)
            + xy * (yz * ww * zx + zz * yw * wx + wz * zw * yx - yz * zw * wx - wz * yw * zx - zz * ww * yx)
            + xz * (yw * zx * wy + ww * yx * zy + zw * wx * yy - yw * wx * zy - zw * yx * wy - ww * zx * yy)
            + xw * (yx * wy * zz + zx * yy * wz + wx * zy * yz - yx * zy * wz - wx * yy * zz - zx * wy * yz)
    }

    #[inline]
    pub fn adjugate(&self) -> M4x4 {
        let [xx, xy, xz, xw] = self.x.arr();
        let [yx, yy, yz, yw] = self.y.arr();
        let [zx, zy, zz, zw] = self.z.arr();
        let [wx, wy, wz, ww] = self.w.arr();
        M4x4 {
            x: vec4(
                yy * zz * ww + wy * yz * zw + zy * wz * yw - yy * wz * zw - zy * yz * ww - wy * zz * yw,
                xy * wz * zw + zy * xz * ww + wy * zz * xw - wy * xz * zw - zy * wz * xw - xy * zz * ww,
                xy * yz * ww + wy * xz * yw + yy * wz * xw - xy * wz * yw - yy * xz * ww - wy * yz * xw,
                xy * zz * yw + yy * xz * zw + zy * yz * xw - xy * yz * zw - zy * xz * yw - yy * zz * xw,
            ),
            y: vec4(
                yz * ww * zx + zz * yw * wx + wz * zw * yx - yz * zw * wx - wz * yw * zx - zz * ww * yx,
                xz * zw * wx + wz * xw * zx + zz * ww * xx - xz * ww * zx - zz * xw * wx - wz * zw * xx,
                xz * ww * yx + yz * xw * wx + wz * yw * xx - xz * yw * wx - wz * xw * yx - yz * ww * xx,
                xz * yw * zx + zz * xw * yx + yz * zw * xx - xz * zw * yx - yz * xw * zx - zz * yw * xx,
            ),
            z: vec4(
                yw * zx * wy + ww * yx * zy + zw * wx * yy - yw * wx * zy - zw * yx * wy - ww * zx * yy,
                xw * wx * zy + zw * xx * wy + ww * zx * xy - xw * zx * wy - ww * xx * zy - zw * wx * xy,
                xw * yx * wy + ww * xx * yy + yw * wx * xy - xw * wx * yy - yw * xx * wy - ww * yx * xy,
                xw * zx * yy + yw * xx * zy + zw * yx * xy - xw * yx * zy - zw * xx * yy - yw * zx * xy,
            ),
            w: vec4(
                yx * wy * zz + zx * yy * wz + wx * zy * yz - yx * zy * wz - wx * yy * zz - zx * wy * yz,
                xx * zy * wz + wx * xy * zz + zx * wy * xz - xx * wy * zz - zx * xy * wz - wx * zy * xz,
                xx * wy * yz + yx * xy * wz + wx * yy * xz - xx * yy * wz - wx * xy * yz - yx * wy * xz,
                xx * yy * zz + zx * xy * yz + yx * zy * xz - xx * zy * yz - yx * xy * zz - zx * yy * xz,
            ),
        }
    }

    #[inline]
    pub fn transpose(&self) -> M4x4 {
        M4x4::from_rows(self.x, self.y, self.z, self.w)
    }

    #[inline]
    pub fn row(&self, i: usize) -> V4 {
        vec4(self.x[i], self.y[i], self.z[i], self.w[i])
    }

    #[inline]
    pub fn col(&self, i: usize) -> V4 {
        self[i]
    }

    #[inline]
    #[rustfmt::skip]
    pub fn from_translation(v: V3) -> M4x4 {
        M4x4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            v.x(), v.y(), v.z(), 1.0,
        )
    }

    #[inline]
    pub fn from_rotation(q: Quat) -> M4x4 {
        M4x4::from_cols(
            V4::expand(q.x_dir(), 0.0),
            V4::expand(q.y_dir(), 0.0),
            V4::expand(q.z_dir(), 0.0),
            V4::new(0.0, 0.0, 0.0, 1.0),
        )
    }

    #[inline]
    #[rustfmt::skip]
    pub fn from_scale(v: V3) -> M4x4 {
        M4x4::new(
            v.x(), 0.0, 0.0, 0.0,
            0.0, v.y(), 0.0, 0.0,
            0.0, 0.0, v.z(), 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
    }

    #[inline]
    pub fn from_pose(p: V3, q: Quat) -> M4x4 {
        M4x4::from_cols(
            V4::expand(q.x_dir(), 0.0),
            V4::expand(q.y_dir(), 0.0),
            V4::expand(q.z_dir(), 0.0),
            V4::expand(p, 1.0),
        )
    }

    #[inline]
    pub fn from_pose_2(p: V3, q: Quat) -> M4x4 {
        M4x4::from_rotation(q) * M4x4::from_translation(p)
    }

    #[inline]
    #[allow(clippy::many_single_char_names)]
    pub fn new_frustum(l: f32, r: f32, b: f32, t: f32, n: f32, f: f32) -> M4x4 {
        M4x4::new(
            2.0 * n / (r - l),
            0.0,
            0.0,
            0.0,
            0.0,
            2.0 * n / (t - b),
            0.0,
            0.0,
            (r + l) / (r - l),
            (t + b) / (t - b),
            -(f + n) / (f - n),
            -1.0,
            0.0,
            0.0,
            -2.0 * f * n / (f - n),
            0.0,
        )
    }

    #[inline]
    pub fn perspective(fovy: f32, aspect: f32, n: f32, f: f32) -> M4x4 {
        let y = n * (fovy * 0.5).tan();
        let x = y * aspect;
        M4x4::new_frustum(-x, x, -y, y, n, f)
    }

    #[inline]
    #[rustfmt::skip]
    pub fn look_towards(fwd: V3, up: V3) -> M4x4 {
        let f = fwd.norm_or(1.0, 0.0, 0.0);
        let s = f.cross(up).norm_or(0.0, 1.0, 0.0);
        let u = s.cross(f);
        M4x4::new(
            s.x(), u.x(), -f.x(), 0.0,
            s.y(), u.y(), -f.y(), 0.0,
            s.z(), u.z(), -f.z(), 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
    }

    #[inline]
    pub fn look_at(eye: V3, target: V3, up: V3) -> M4x4 {
        M4x4::look_towards(target - eye, up) * M4x4::from_translation(-eye)
    }

    #[inline]
    pub const fn identity() -> Self {
        Self::IDENTITY
    }

    #[inline]
    pub const fn zero() -> Self {
        Self::ZERO
    }

    pub fn inverse(&self) -> Option<Self> {
        let d = self.determinant();
        if d == 0.0 {
            None
        } else {
            Some(self.adjugate() * (1.0 / d))
        }
    }
}

impl From<M2x2> for [f32; 4] {
    #[inline]
    fn from(m: M2x2) -> [f32; 4] {
        [m.x.x, m.x.y, m.y.x, m.y.y]
    }
}

impl From<[f32; 4]> for M2x2 {
    #[inline]
    fn from(m: [f32; 4]) -> M2x2 {
        mat2(m[0], m[1], m[2], m[3])
    }
}

impl From<M2x2> for [[f32; 2]; 2] {
    #[inline]
    fn from(m: M2x2) -> [[f32; 2]; 2] {
        [[m.x.x, m.x.y], [m.y.x, m.y.y]]
    }
}

impl From<[[f32; 2]; 2]> for M2x2 {
    #[inline]
    fn from(m: [[f32; 2]; 2]) -> M2x2 {
        mat2(m[0][0], m[0][1], m[1][0], m[1][1])
    }
}

impl From<M3x3> for [f32; 9] {
    #[inline]
    fn from(m: M3x3) -> [f32; 9] {
        let [xx, xy, xz] = m.x.arr();
        let [yx, yy, yz] = m.y.arr();
        let [zx, zy, zz] = m.z.arr();
        [xx, xy, xz, yx, yy, yz, zx, zy, zz]
    }
}

impl From<[f32; 9]> for M3x3 {
    #[inline]
    fn from(m: [f32; 9]) -> M3x3 {
        mat3(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8])
    }
}

impl From<M3x3> for [[f32; 3]; 3] {
    #[inline]
    fn from(m: M3x3) -> [[f32; 3]; 3] {
        [m[0].into(), m[1].into(), m[2].into()]
    }
}

impl From<[[f32; 3]; 3]> for M3x3 {
    #[inline]
    fn from(m: [[f32; 3]; 3]) -> M3x3 {
        M3x3::from_cols(m[0].into(), m[1].into(), m[2].into())
    }
}

impl From<M4x4> for [f32; 16] {
    #[inline]
    #[rustfmt::skip]
    fn from(m: M4x4) -> [f32; 16] {
        let [xx, xy, xz, xw] = m.x.arr();
        let [yx, yy, yz, yw] = m.y.arr();
        let [zx, zy, zz, zw] = m.z.arr();
        let [wx, wy, wz, ww] = m.w.arr();
        [
            xx, xy, xz, xw,
            yx, yy, yz, yw,
            zx, zy, zz, zw,
            wx, wy, wz, ww,
        ]
    }
}

impl From<[f32; 16]> for M4x4 {
    #[inline]
    #[rustfmt::skip]
    fn from(m: [f32; 16]) -> M4x4 {
        mat4(
            m[0],  m[1],  m[2],  m[3],
            m[4],  m[5],  m[6],  m[7],
            m[8],  m[9],  m[10], m[11],
            m[12], m[13], m[14], m[15],
        )
    }
}

impl From<M4x4> for [[f32; 4]; 4] {
    #[inline]
    fn from(m: M4x4) -> [[f32; 4]; 4] {
        [m[0].into(), m[1].into(), m[2].into(), m[3].into()]
    }
}

impl From<[[f32; 4]; 4]> for M4x4 {
    #[inline]
    fn from(m: [[f32; 4]; 4]) -> M4x4 {
        M4x4::from_cols(m[0].into(), m[1].into(), m[2].into(), m[3].into())
    }
}
pub trait MatType:
    Copy
    + Clone
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Self, Output = Self>
    + Mul<f32, Output = Self>
    + Div<f32, Output = Self>
{
    type Vec: VecType;

    const ROWS: usize;
    const COLS: usize;

    const ELEMS: usize;
}

impl MatType for M2x2 {
    type Vec = V2;
    const ROWS: usize = 2;
    const COLS: usize = 2;
    const ELEMS: usize = 4;
}

impl MatType for M3x3 {
    type Vec = V3;
    const ROWS: usize = 3;
    const COLS: usize = 3;
    const ELEMS: usize = 9;
}

impl MatType for M4x4 {
    type Vec = V4;
    const ROWS: usize = 4;
    const COLS: usize = 4;
    const ELEMS: usize = 16;
}

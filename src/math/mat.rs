use crate::math::quat::*;
use crate::math::traits::*;
use crate::math::vec::*;
use std::{f32, fmt, mem, ops::*};

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

impl Identity for M2x2 {
    const IDENTITY: M2x2 = M2x2 {
        x: V2 { x: 1.0, y: 0.0 },
        y: V2 { x: 0.0, y: 1.0 },
    };
}

impl Identity for M3x3 {
    const IDENTITY: M3x3 = M3x3 {
        x: V3 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        },
        y: V3 {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        },
        z: V3 {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        },
    };
}

impl Identity for M4x4 {
    const IDENTITY: M4x4 = M4x4 {
        x: V4 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
            w: 0.0,
        },
        y: V4 {
            x: 0.0,
            y: 1.0,
            z: 0.0,
            w: 0.0,
        },
        z: V4 {
            x: 0.0,
            y: 0.0,
            z: 1.0,
            w: 0.0,
        },
        w: V4 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        },
    };
}

impl Zero for M2x2 {
    const ZERO: M2x2 = M2x2 {
        x: V2 { x: 0.0, y: 0.0 },
        y: V2 { x: 0.0, y: 0.0 },
    };
}

impl Zero for M3x3 {
    const ZERO: M3x3 = M3x3 {
        x: V3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
        y: V3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
        z: V3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
    };
}

impl Zero for M4x4 {
    const ZERO: M4x4 = M4x4 {
        x: V4 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 0.0,
        },
        y: V4 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 0.0,
        },
        z: V4 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 0.0,
        },
        w: V4 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 0.0,
        },
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
        <Self as Identity>::IDENTITY
    }
}
impl Default for M3x3 {
    #[inline]
    fn default() -> Self {
        <Self as Identity>::IDENTITY
    }
}
impl Default for M4x4 {
    #[inline]
    fn default() -> Self {
        <Self as Identity>::IDENTITY
    }
}

#[inline]
pub fn mat2(m00: f32, m01: f32, m10: f32, m11: f32) -> M2x2 {
    M2x2::new(m00, m01, m10, m11)
}

#[inline]
pub fn mat3(
    m00: f32,
    m01: f32,
    m02: f32,
    m10: f32,
    m11: f32,
    m12: f32,
    m20: f32,
    m21: f32,
    m22: f32,
) -> M3x3 {
    M3x3::new(m00, m01, m02, m10, m11, m12, m20, m21, m22)
}

#[inline]
pub fn mat4(
    m00: f32,
    m01: f32,
    m02: f32,
    m03: f32,
    m10: f32,
    m11: f32,
    m12: f32,
    m13: f32,
    m20: f32,
    m21: f32,
    m22: f32,
    m23: f32,
    m30: f32,
    m31: f32,
    m32: f32,
    m33: f32,
) -> M4x4 {
    M4x4::new(
        m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33,
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
        self.x * v.x + self.y * v.y + self.z * v.z
    }
}

impl<'a> Mul<V4> for &'a M4x4 {
    type Output = V4;
    #[inline]
    fn mul(self, v: V4) -> V4 {
        self.x * v.x + self.y * v.y + self.z * v.z + self.w * v.w
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

macro_rules! define_conversions {
    ($src_type: ty, $dst_type: ty) => {
        impl AsRef<$dst_type> for $src_type {
            #[inline]
            fn as_ref(&self) -> &$dst_type {
                unsafe { mem::transmute(self) }
            }
        }

        impl AsMut<$dst_type> for $src_type {
            #[inline]
            fn as_mut(&mut self) -> &mut $dst_type {
                unsafe { mem::transmute(self) }
            }
        }

        impl From<$src_type> for $dst_type {
            #[inline]
            fn from(m: $src_type) -> $dst_type {
                unsafe { mem::transmute(m) }
            }
        }

        impl From<$dst_type> for $src_type {
            #[inline]
            fn from(m: $dst_type) -> $src_type {
                unsafe { mem::transmute(m) }
            }
        }
    };
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
        // would be nice if we did this for tuples too...
        define_conversions!($Mn, [f32; $elems]);
        define_conversions!($Mn, [[f32; $size]; $size]);
        define_conversions!($Mn, [$Vn; $size]);

        impl AsRef<[f32]> for $Mn {
            #[inline]
            fn as_ref(&self) -> &[f32] {
                let m: &[f32; $elems] = self.as_ref();
                &m[..]
            }
        }

        impl AsMut<[f32]> for $Mn {
            #[inline]
            fn as_mut(&mut self) -> &mut [f32] {
                let m: &mut[f32; $elems] = self.as_mut();
                &mut m[..]
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
            #[inline] pub fn as_ptr(&self) -> *const f32 { (&self.x.x) as *const f32 }
            #[inline] pub fn as_mut_ptr(&mut self) -> *mut f32 { (&mut self.x.x) as *mut f32 }

            #[inline] pub fn as_f32_slice(&self) -> &[f32] { self.as_ref() }
            #[inline] pub fn as_mut_f32_slice(&mut self) -> &mut [f32] { self.as_mut() }

            #[inline] pub fn as_slice(&self) -> &[$Vn] { self.as_ref() }
            #[inline] pub fn as_mut_slice(&mut self) -> &mut [$Vn] { self.as_mut() }

            #[inline]
            pub fn map<F: Fn($Vn) -> $Vn>(self, f: F) -> $Mn {
                $Mn { $($field: f(self.$field)),+ }
            }

            #[inline]
            pub fn map2<F: Fn($Vn, $Vn) -> $Vn>(self, o: $Mn, f: F) -> $Mn {
                $Mn { $($field: f(self.$field, o.$field)),+ }
            }
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
            #[inline] fn add(self, o: $Mn) -> $Mn { self.map2(o, |a, b| a + b) }
        }

        impl Sub for $Mn {
            type Output = $Mn;
            #[inline] fn sub(self, o: $Mn) -> $Mn { self.map2(o, |a, b| a - b) }
        }

        impl Mul<$Mn> for $Mn {
            type Output = $Mn;
            #[inline] fn mul(self, rhs: $Mn) -> $Mn { $Mn { $($field: self * rhs.$field),+ } }
        }

        impl Mul<f32> for $Mn {
            type Output = $Mn;
            #[inline] fn mul(self, rhs: f32) -> $Mn { self.map(|a| a * rhs) }
        }

        impl Div<f32> for $Mn {
            type Output = $Mn;
            #[inline] fn div(self, rhs: f32) -> $Mn { self.map(|a| a / rhs) }
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
do_mat_boilerplate!(
    M4x4 {
        x: 0,
        y: 1,
        z: 2,
        w: 3
    },
    V4,
    4,
    16
);

impl M2x2 {
    #[inline]
    pub fn new(xx: f32, xy: f32, yx: f32, yy: f32) -> M2x2 {
        M2x2 {
            x: V2 { x: xx, y: xy },
            y: V2 { x: yx, y: yy },
        }
    }

    #[inline]
    pub fn from_cols(x: V2, y: V2) -> M2x2 {
        M2x2 { x, y }
    }

    #[inline]
    pub fn from_rows(x: V2, y: V2) -> M2x2 {
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
    pub fn identity() -> Self {
        mat2(1.0, 0.0, 0.0, 1.0)
    }

    #[inline]
    pub fn zero() -> Self {
        mat2(0.0, 0.0, 0.0, 0.0)
    }
}

impl M3x3 {
    #[inline]
    pub fn new(
        xx: f32,
        xy: f32,
        xz: f32,
        yx: f32,
        yy: f32,
        yz: f32,
        zx: f32,
        zy: f32,
        zz: f32,
    ) -> Self {
        Self {
            x: V3 {
                x: xx,
                y: xy,
                z: xz,
            },
            y: V3 {
                x: yx,
                y: yy,
                z: yz,
            },
            z: V3 {
                x: zx,
                y: zy,
                z: zz,
            },
        }
    }

    #[inline]
    pub fn from_cols(x: V3, y: V3, z: V3) -> M3x3 {
        M3x3 { x, y, z }
    }

    #[inline]
    pub fn from_rows(x: V3, y: V3, z: V3) -> M3x3 {
        M3x3::new(x.x, y.x, z.x, x.y, y.y, z.y, x.z, y.z, z.z)
    }

    #[inline]
    pub fn to_quat(&self) -> Quat {
        let mag_w = self.x.x + self.y.y + self.z.z;

        let (mag_zw, pre_zw, post_zw) = if mag_w > self.z.z {
            (mag_w, vec3(1.0, 1.0, 1.0), quat(0.0, 0.0, 0.0, 1.0))
        } else {
            (self.z.z, vec3(-1.0, -1.0, 1.0), quat(0.0, 0.0, 1.0, 0.0))
        };

        let (mag_xy, pre_xy, post_xy) = if self.x.x > self.y.y {
            (self.x.x, vec3(1.0, -1.0, -1.0), quat(1.0, 0.0, 0.0, 0.0))
        } else {
            (self.y.y, vec3(-1.0, 1.0, -1.0), quat(0.0, 1.0, 0.0, 0.0))
        };

        let (pre, post) = if mag_zw > mag_xy {
            (pre_zw, post_zw)
        } else {
            (pre_xy, post_xy)
        };

        let t = pre.x * self.x.x + pre.y * self.y.y + pre.z * self.z.z + 1.0;
        let s = 0.5 / t.sqrt();
        let qp = quat(
            (pre.y * self.y.z - pre.z * self.z.y) * s,
            (pre.z * self.z.x - pre.x * self.x.z) * s,
            (pre.x * self.x.y - pre.y * self.y.x) * s,
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
        vec3(self.x.x, self.y.y, self.z.z)
    }

    #[inline]
    pub fn determinant(&self) -> f32 {
        self.x.x * (self.y.y * self.z.z - self.z.y * self.y.z)
            + self.x.y * (self.y.z * self.z.x - self.z.z * self.y.x)
            + self.x.z * (self.y.x * self.z.y - self.z.x * self.y.y)
    }

    #[inline]
    pub fn adjugate(&self) -> M3x3 {
        M3x3 {
            x: vec3(
                self.y.y * self.z.z - self.z.y * self.y.z,
                self.z.y * self.x.z - self.x.y * self.z.z,
                self.x.y * self.y.z - self.y.y * self.x.z,
            ),
            y: vec3(
                self.y.z * self.z.x - self.z.z * self.y.x,
                self.z.z * self.x.x - self.x.z * self.z.x,
                self.x.z * self.y.x - self.y.z * self.x.x,
            ),
            z: vec3(
                self.y.x * self.z.y - self.z.x * self.y.y,
                self.z.x * self.x.y - self.x.x * self.z.y,
                self.x.x * self.y.y - self.y.x * self.x.y,
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
    pub fn identity() -> Self {
        mat3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    }

    #[inline]
    pub fn zero() -> Self {
        mat3(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    }

    #[inline]
    pub fn is_symmetric(&self) -> bool {
        // Should this be strict comparison?
        self.x.y == self.y.x && self.x.z == self.z.x && self.y.z == self.z.y
    }

    #[inline]
    pub fn is_approx_symmetric(&self) -> bool {
        // Should this be strict comparison?
        approx_eq(self.x.y, self.y.x)
            && approx_eq(self.x.z, self.z.x)
            && approx_eq(self.y.z, self.z.y)
    }

    /// Returns quat s.t. q.to_mat3() diagonalizes this matrix. Requires `self`
    /// be symmetric.
    ///
    /// If you have
    /// ```rust,no_run
    /// q = some_mat3.diagonalizer().to_mat3();
    /// d = q * some_mat3 * q.transpose();
    /// ```
    /// Then the rows of `q` are the eigenvectors, and `d`'s diagonal are the
    /// eigenvalues.
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
            let off_diag = vec3(diag.y.z, diag.x.z, diag.x.y);
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
            jacobi_rot.0.w = (1.0 - axis_val * axis_val).sqrt();
            if jacobi_rot.0.w == 1.0 {
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

        if q.x_dir().z < 0.0 {
            q *= quat(1.0, 0.0, 0.0, 0.0);
        }
        if q.y_dir().y < 0.0 {
            q *= quat(0.0, 0.0, 1.0, 0.0);
        }
        if q.0.w < 0.0 {
            q = -q;
        }
        q
    }
}

impl M4x4 {
    #[inline]
    pub fn new(
        xx: f32,
        xy: f32,
        xz: f32,
        xw: f32,
        yx: f32,
        yy: f32,
        yz: f32,
        yw: f32,
        zx: f32,
        zy: f32,
        zz: f32,
        zw: f32,
        wx: f32,
        wy: f32,
        wz: f32,
        ww: f32,
    ) -> M4x4 {
        M4x4 {
            x: V4 {
                x: xx,
                y: xy,
                z: xz,
                w: xw,
            },
            y: V4 {
                x: yx,
                y: yy,
                z: yz,
                w: yw,
            },
            z: V4 {
                x: zx,
                y: zy,
                z: zz,
                w: zw,
            },
            w: V4 {
                x: wx,
                y: wy,
                z: wz,
                w: ww,
            },
        }
    }

    #[inline]
    pub fn from_cols(x: V4, y: V4, z: V4, w: V4) -> M4x4 {
        M4x4 { x, y, z, w }
    }

    #[inline]
    pub fn from_rows(x: V4, y: V4, z: V4, w: V4) -> M4x4 {
        mat4(
            x.x, y.x, z.x, w.x, x.y, y.y, z.y, w.y, x.z, y.z, z.z, w.z, x.w, y.w, z.w, w.w,
        )
    }

    #[inline]
    pub fn to_arr(self) -> [[f32; 4]; 4] {
        self.into()
    }

    #[inline]
    pub fn diagonal(&self) -> V4 {
        vec4(self.x.x, self.y.y, self.z.z, self.w.w)
    }

    #[inline]
    pub fn determinant(&self) -> f32 {
        self.x.x
            * (self.y.y * self.z.z * self.w.w
                + self.w.y * self.y.z * self.z.w
                + self.z.y * self.w.z * self.y.w
                - self.y.y * self.w.z * self.z.w
                - self.z.y * self.y.z * self.w.w
                - self.w.y * self.z.z * self.y.w)
            + self.x.y
                * (self.y.z * self.w.w * self.z.x
                    + self.z.z * self.y.w * self.w.x
                    + self.w.z * self.z.w * self.y.x
                    - self.y.z * self.z.w * self.w.x
                    - self.w.z * self.y.w * self.z.x
                    - self.z.z * self.w.w * self.y.x)
            + self.x.z
                * (self.y.w * self.z.x * self.w.y
                    + self.w.w * self.y.x * self.z.y
                    + self.z.w * self.w.x * self.y.y
                    - self.y.w * self.w.x * self.z.y
                    - self.z.w * self.y.x * self.w.y
                    - self.w.w * self.z.x * self.y.y)
            + self.x.w
                * (self.y.x * self.w.y * self.z.z
                    + self.z.x * self.y.y * self.w.z
                    + self.w.x * self.z.y * self.y.z
                    - self.y.x * self.z.y * self.w.z
                    - self.w.x * self.y.y * self.z.z
                    - self.z.x * self.w.y * self.y.z)
    }

    #[inline]
    pub fn adjugate(&self) -> M4x4 {
        let M4x4 { x, y, z, w } = *self;
        return M4x4 {
            x: vec4(
                y.y * z.z * w.w + w.y * y.z * z.w + z.y * w.z * y.w
                    - y.y * w.z * z.w
                    - z.y * y.z * w.w
                    - w.y * z.z * y.w,
                x.y * w.z * z.w + z.y * x.z * w.w + w.y * z.z * x.w
                    - w.y * x.z * z.w
                    - z.y * w.z * x.w
                    - x.y * z.z * w.w,
                x.y * y.z * w.w + w.y * x.z * y.w + y.y * w.z * x.w
                    - x.y * w.z * y.w
                    - y.y * x.z * w.w
                    - w.y * y.z * x.w,
                x.y * z.z * y.w + y.y * x.z * z.w + z.y * y.z * x.w
                    - x.y * y.z * z.w
                    - z.y * x.z * y.w
                    - y.y * z.z * x.w,
            ),
            y: vec4(
                y.z * w.w * z.x + z.z * y.w * w.x + w.z * z.w * y.x
                    - y.z * z.w * w.x
                    - w.z * y.w * z.x
                    - z.z * w.w * y.x,
                x.z * z.w * w.x + w.z * x.w * z.x + z.z * w.w * x.x
                    - x.z * w.w * z.x
                    - z.z * x.w * w.x
                    - w.z * z.w * x.x,
                x.z * w.w * y.x + y.z * x.w * w.x + w.z * y.w * x.x
                    - x.z * y.w * w.x
                    - w.z * x.w * y.x
                    - y.z * w.w * x.x,
                x.z * y.w * z.x + z.z * x.w * y.x + y.z * z.w * x.x
                    - x.z * z.w * y.x
                    - y.z * x.w * z.x
                    - z.z * y.w * x.x,
            ),
            z: vec4(
                y.w * z.x * w.y + w.w * y.x * z.y + z.w * w.x * y.y
                    - y.w * w.x * z.y
                    - z.w * y.x * w.y
                    - w.w * z.x * y.y,
                x.w * w.x * z.y + z.w * x.x * w.y + w.w * z.x * x.y
                    - x.w * z.x * w.y
                    - w.w * x.x * z.y
                    - z.w * w.x * x.y,
                x.w * y.x * w.y + w.w * x.x * y.y + y.w * w.x * x.y
                    - x.w * w.x * y.y
                    - y.w * x.x * w.y
                    - w.w * y.x * x.y,
                x.w * z.x * y.y + y.w * x.x * z.y + z.w * y.x * x.y
                    - x.w * y.x * z.y
                    - z.w * x.x * y.y
                    - y.w * z.x * x.y,
            ),
            w: vec4(
                y.x * w.y * z.z + z.x * y.y * w.z + w.x * z.y * y.z
                    - y.x * z.y * w.z
                    - w.x * y.y * z.z
                    - z.x * w.y * y.z,
                x.x * z.y * w.z + w.x * x.y * z.z + z.x * w.y * x.z
                    - x.x * w.y * z.z
                    - z.x * x.y * w.z
                    - w.x * z.y * x.z,
                x.x * w.y * y.z + y.x * x.y * w.z + w.x * y.y * x.z
                    - x.x * y.y * w.z
                    - w.x * x.y * y.z
                    - y.x * w.y * x.z,
                x.x * y.y * z.z + z.x * x.y * y.z + y.x * z.y * x.z
                    - x.x * z.y * y.z
                    - y.x * x.y * z.z
                    - z.x * y.y * x.z,
            ),
        };
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
    pub fn from_translation(v: V3) -> M4x4 {
        M4x4::new(
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, v.x, v.y, v.z, 1.0,
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
    pub fn from_scale(v: V3) -> M4x4 {
        M4x4::new(
            v.x, 0.0, 0.0, 0.0, 0.0, v.y, 0.0, 0.0, 0.0, 0.0, v.z, 0.0, 0.0, 0.0, 0.0, 1.0,
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
    pub fn look_towards(fwd: V3, up: V3) -> M4x4 {
        let f = fwd.norm_or(1.0, 0.0, 0.0);
        let s = f.cross(up).norm_or(0.0, 1.0, 0.0);
        let u = s.cross(f);
        M4x4::new(
            s.x, u.x, -f.x, 0.0, s.y, u.y, -f.y, 0.0, s.z, u.z, -f.z, 0.0, 0.0, 0.0, 0.0, 1.0,
        )
    }

    #[inline]
    pub fn look_at(eye: V3, target: V3, up: V3) -> M4x4 {
        M4x4::look_towards(target - eye, up) * M4x4::from_translation(-eye)
    }

    #[inline]
    pub fn identity() -> Self {
        mat4(
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        )
    }

    #[inline]
    pub fn zero() -> Self {
        mat4(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        )
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

pub trait MatType:
    Copy
    + Clone
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Self, Output = Self>
    + Mul<f32, Output = Self>
    + Div<f32, Output = Self>
    + Identity
    + Zero
// + Mul<Self::Vec, Output = Self::Vec>
// + Index<usize, Output = Self::Vec>
// + IndexMut<usize>
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

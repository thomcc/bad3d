use math::vec::*;
use math::quat::*;
use math::traits::*;

use std::ops::*;
use std::mem;

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


#[inline]
pub fn mat2(m00: f32, m01: f32, m10: f32, m11: f32) -> M2x2 {
    M2x2::new(m00, m01,
              m10, m11)
}


#[inline]
pub fn mat3(m00: f32, m01: f32, m02: f32,
            m10: f32, m11: f32, m12: f32,
            m20: f32, m21: f32, m22: f32) -> M3x3 {
    M3x3::new(m00, m01, m02,
              m10, m11, m12,
              m20, m21, m22)
}

#[inline]
pub fn mat4(m00: f32, m01: f32, m02: f32, m03: f32,
            m10: f32, m11: f32, m12: f32, m13: f32,
            m20: f32, m21: f32, m22: f32, m23: f32,
            m30: f32, m31: f32, m32: f32, m33: f32) -> M4x4 {
    M4x4::new(m00, m01, m02, m03,
              m10, m11, m12, m13,
              m20, m21, m22, m23,
              m30, m31, m32, m33)
}


impl Mul<V2> for M2x2 {
    type Output = V2;
    #[inline]
    fn mul(self, v: V2) -> V2 {
        self.x*v.x + self.y*v.y
    }
}

impl Mul<V3> for M3x3 {
    type Output = V3;
    #[inline]
    fn mul(self, v: V3) -> V3 {
        self.x*v.x + self.y*v.y + self.z*v.z
    }
}

impl Mul<V4> for M4x4 {
    type Output = V4;
    #[inline]
    fn mul(self, v: V4) -> V4 {
        self.x*v.x + self.y*v.y + self.z*v.z + self.w*v.w
    }
}

// @@ partially duplicated in vec.rs...
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

        impl<'a> From<&'a $dst_type> for &'a $src_type {
            #[inline]
            fn from(v: &'a $dst_type) -> &'a $src_type {
                unsafe { mem::transmute(v) }
            }
        }

        impl<'a> From<&'a mut $dst_type> for &'a mut $src_type {
            #[inline]
            fn from(v: &'a mut $dst_type) -> &'a mut $src_type {
                unsafe { mem::transmute(v) }
            }
        }

        impl From<$src_type> for $dst_type {
            #[inline]
            fn from(m: $src_type) -> $dst_type {
                let r: &$dst_type = m.as_ref();
                *r
            }
        }

        impl From<$dst_type> for $src_type {
            #[inline]
            fn from(m: $dst_type) -> $src_type {
                let r: &$src_type = (&m).into();
                *r
            }
        }
    }
}


macro_rules! do_mat_boilerplate {
    ($Mn: ident { $($field: ident : $index: expr),+ },
     $Vn: ident, $size: expr,
     $elems: expr) =>
    {

        impl $Mn {
            #[inline]
            fn map<F: Fn($Vn) -> $Vn>(self, f: F) -> $Mn {
                $Mn{$($field: f(self.$field)),+}
            }

            #[inline]
            fn map2<F: Fn($Vn, $Vn) -> $Vn>(self, o: $Mn, f: F) -> $Mn {
                $Mn{$($field: f(self.$field, o.$field)),+}
            }
        }

        // would be nice if we did this for tuples too...
        define_conversions!($Mn, [f32; $elems]);
        define_conversions!($Mn, [[f32; $size]; $size]);
        define_conversions!($Mn, [$Vn; $size]);


        impl Index<usize> for $Mn {
            type Output = $Vn;
            #[inline]
            fn index(&self, i: usize) -> &$Vn {
                let v: &[$Vn; $size] = self.as_ref();
                &v[i]
            }
        }

        impl IndexMut<usize> for $Mn {
            #[inline]
            fn index_mut(&mut self, i: usize) -> &mut $Vn {
                let v: &mut [$Vn; $size] = self.as_mut();
                &mut v[i]
            }
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
            #[inline] fn mul(self, rhs: $Mn) -> $Mn { $Mn{$($field: self*rhs.$field),+} }
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

        impl AddAssign<$Mn> for $Mn {
            #[inline] fn add_assign(&mut self, rhs: $Mn) { let res = self.add(rhs); *self = res; }
        }

        impl SubAssign<$Mn> for $Mn {
            #[inline] fn sub_assign(&mut self, rhs: $Mn) { let res = self.sub(rhs); *self = res; }
        }

        impl MulAssign<$Mn> for $Mn {
            #[inline] fn mul_assign(&mut self, rhs: $Mn) { let res = self.mul(rhs); *self = res; }
        }

        impl MulAssign<f32> for $Mn {
            #[inline] fn mul_assign(&mut self, rhs: f32) { let res = self.mul(rhs); *self = res; }
        }

        impl DivAssign<f32> for $Mn {
            #[inline] fn div_assign(&mut self, rhs: f32) { let res = self.div(rhs); *self = res; }
        }
    }
}

do_mat_boilerplate!(M2x2{x: 0, y: 1            }, V2, 2, 4);
do_mat_boilerplate!(M3x3{x: 0, y: 1, z: 2      }, V3, 3, 9);
do_mat_boilerplate!(M4x4{x: 0, y: 1, z: 2, w: 3}, V4, 4, 16);

impl M2x2 {
    #[inline]
    pub fn new(xx: f32, xy: f32, yx: f32, yy: f32) -> M2x2 {
        M2x2{x: V2::new(xx, xy), y: V2::new(yx, yy)}
    }

    #[inline]
    pub fn from_cols(x: V2, y: V2) -> M2x2 {
        M2x2{x: x, y: y}
    }

    #[inline]
    pub fn from_rows(x: V2, y: V2) -> M2x2 {
        M2x2::new(x.x, y.x, x.y, y.y)
    }
}

impl M3x3 {

    #[inline]
    pub fn new(xx: f32, xy: f32, xz: f32,
               yx: f32, yy: f32, yz: f32,
               zx: f32, zy: f32, zz: f32) -> M3x3 {
        M3x3{
            x: vec3(xx, xy, xz),
            y: vec3(yx, yy, yz),
            z: vec3(zx, zy, zz)
        }
    }

    #[inline]
    pub fn from_cols(x: V3, y: V3, z: V3) -> M3x3 {
        M3x3{x: x, y: y, z: z}
    }

    #[inline]
    pub fn from_rows(x: V3, y: V3, z: V3) -> M3x3 {
        M3x3::new(x.x, y.x, z.x,
                  x.y, y.y, z.y,
                  x.z, y.z, z.z)
    }

    #[inline]
    pub fn to_quat(&self) -> Quat {
        let mag_w = self.x.x + self.y.y + self.z.z;

        let (mag_zw, pre_zw, post_zw) =
            if mag_w > self.z.z {
                (mag_w,    vec3( 1.0,  1.0, 1.0), quat(0.0, 0.0, 0.0, 1.0))
            } else {
                (self.z.z, vec3(-1.0, -1.0, 1.0), quat(0.0, 0.0, 1.0, 0.0))
            };

        let (mag_xy, pre_xy, post_xy) =
            if self.x.x > self.y.y {
                (self.x.x, vec3( 1.0, -1.0, -1.0), quat(1.0, 0.0, 0.0, 0.0))
            } else {
                (self.y.y, vec3(-1.0,  1.0, -1.0), quat(0.0, 1.0, 0.0, 0.0))
            };

        let (pre, post) =
            if mag_zw > mag_xy {
                (pre_zw, post_zw)
            } else {
                (pre_xy, post_xy)
            };

        let t = pre.x*self.x.x + pre.y*self.y.y + pre.z*self.z.z + 1.0;
        let s = 0.5 / t.sqrt();
        let qp = quat((pre.y * self.y.z - pre.z * self.z.y) * s,
                      (pre.z * self.z.x - pre.x * self.x.z) * s,
                      (pre.x * self.x.y - pre.y * self.y.x) * s,
                      t * s);
        debug_assert!(approx_eq(qp.length(), 1.0));
        qp * post
    }

    #[inline]
    pub fn to_mat4(&self) -> M4x4 {
        M4x4{
            x: V4::expand(self.x, 0.0),
            y: V4::expand(self.y, 0.0),
            z: V4::expand(self.z, 0.0),
            w: vec4(0.0, 0.0, 0.0, 1.0)
        }
    }
}

impl M4x4 {
    #[inline]
    pub fn new(xx: f32, xy: f32, xz: f32, xw: f32,
               yx: f32, yy: f32, yz: f32, yw: f32,
               zx: f32, zy: f32, zz: f32, zw: f32,
               wx: f32, wy: f32, wz: f32, ww: f32) -> M4x4 {
        M4x4 {
            x: vec4(xx, xy, xz, xw),
            y: vec4(yx, yy, yz, yw),
            z: vec4(zx, zy, zz, zw),
            w: vec4(wx, wy, wz, ww)
        }
    }

    #[inline]
    pub fn from_cols(x: V4, y: V4, z: V4, w: V4) -> M4x4 {
        M4x4{x: x, y: y, z: z, w: w}
    }

    #[inline]
    pub fn from_rows(x: V4, y: V4, z: V4, w: V4) -> M4x4 {
        mat4(x.x, y.x, z.x, w.x,
             x.y, y.y, z.y, w.y,
             x.z, y.z, z.z, w.z,
             x.w, y.w, z.w, w.w)
    }
}

impl Identity for M2x2 {
    #[inline] fn identity() -> M2x2 { mat2(1.0, 0.0, 0.0, 1.0) }
}

impl Identity for M3x3 {
    #[inline] fn identity() -> M3x3 { mat3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) }
}

impl Identity for M4x4 {
    #[inline]
    fn identity() -> M4x4 {
        mat4(1.0, 0.0, 0.0, 0.0,
             0.0, 1.0, 0.0, 0.0,
             0.0, 0.0, 1.0, 0.0,
             0.0, 0.0, 0.0, 1.0)
    }
}

pub trait MatType
    : Copy
    + Clone
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Self, Output = Self>
    + Mul<f32, Output = Self>
    + Div<f32, Output = Self>
    + Identity
    // + Mul<Self::Vec, Output = Self::Vec>
    // + Index<usize, Output = Self::Vec>
    // + IndexMut<usize>
{
    type Vec: VecType;

    fn determinant(&self) -> f32;
    fn adjugate(&self) -> Self;
    fn transpose(&self) -> Self;
    fn row(&self, usize) -> Self::Vec;
    fn col(&self, usize) -> Self::Vec;

    #[inline]
    fn inverse(&self) -> Option<Self> {
        let d = self.determinant();
        if d == 0.0 {
            None
        } else {
            Some(self.adjugate() * (1.0 / d))
        }
    }

    #[inline]
    fn inverse_or_id(&self) -> Self {
        self.inverse().unwrap_or(Identity::identity())
    }
}

impl MatType for M2x2 {
    type Vec = V2;
    #[inline] fn determinant(&self) -> f32 { self.x.x*self.y.y - self.x.y*self.y.x }
    #[inline] fn adjugate(&self) -> M2x2 { M2x2::new(self.y.y, -self.x.y, -self.y.x, self.x.x) }
    #[inline] fn transpose(&self) -> M2x2 { M2x2::from_rows(self.x, self.y) }
    #[inline] fn row(&self, i: usize) -> V2 { V2::new(self.x[i], self.y[i]) }
    #[inline] fn col(&self, i: usize) -> V2 { self[i] }
}

impl MatType for M3x3 {
    type Vec = V3;

    #[inline]
    fn determinant(&self) -> f32 {
        self.x.x*(self.y.y*self.z.z - self.z.y*self.y.z) +
        self.x.y*(self.y.z*self.z.x - self.z.z*self.y.x) +
        self.x.z*(self.y.x*self.z.y - self.z.x*self.y.y)
    }

    #[inline]
    fn adjugate(&self) -> M3x3 {
        return M3x3 {
            x: vec3(self.y.y*self.z.z - self.z.y*self.y.z,
                    self.z.y*self.x.z - self.x.y*self.z.z,
                    self.x.y*self.y.z - self.y.y*self.x.z),
            y: vec3(self.y.z*self.z.x - self.z.z*self.y.x,
                    self.z.z*self.x.x - self.x.z*self.z.x,
                    self.x.z*self.y.x - self.y.z*self.x.x),
            z: vec3(self.y.x*self.z.y - self.z.x*self.y.y,
                    self.z.x*self.x.y - self.x.x*self.z.y,
                    self.x.x*self.y.y - self.y.x*self.x.y),
        }
    }

    #[inline]
    fn transpose(&self) -> M3x3 {
        M3x3::from_rows(self.x, self.y, self.z)
    }

    #[inline]
    fn row(&self, i: usize) -> V3 {
        vec3(self.x[i], self.y[i], self.z[i])
    }

    #[inline]
    fn col(&self, i: usize) -> V3 {
        self[i]
    }
}

impl MatType for M4x4 {
    type Vec = V4;
    #[inline]
    fn determinant(&self) -> f32 {
        self.x.x*(self.y.y*self.z.z*self.w.w + self.w.y*self.y.z*self.z.w +
                  self.z.y*self.w.z*self.y.w - self.y.y*self.w.z*self.z.w -
                  self.z.y*self.y.z*self.w.w - self.w.y*self.z.z*self.y.w) +

        self.x.y*(self.y.z*self.w.w*self.z.x + self.z.z*self.y.w*self.w.x +
                  self.w.z*self.z.w*self.y.x - self.y.z*self.z.w*self.w.x -
                  self.w.z*self.y.w*self.z.x - self.z.z*self.w.w*self.y.x) +

        self.x.z*(self.y.w*self.z.x*self.w.y + self.w.w*self.y.x*self.z.y +
                  self.z.w*self.w.x*self.y.y - self.y.w*self.w.x*self.z.y -
                  self.z.w*self.y.x*self.w.y - self.w.w*self.z.x*self.y.y) +

        self.x.w*(self.y.x*self.w.y*self.z.z + self.z.x*self.y.y*self.w.z +
                  self.w.x*self.z.y*self.y.z - self.y.x*self.z.y*self.w.z -
                  self.w.x*self.y.y*self.z.z - self.z.x*self.w.y*self.y.z)
    }

    #[inline]
    fn adjugate(&self) -> M4x4 {
        let M4x4{x, y, z, w} = *self;
        return M4x4 {
            x: vec4(y.y*z.z*w.w + w.y*y.z*z.w + z.y*w.z*y.w - y.y*w.z*z.w - z.y*y.z*w.w - w.y*z.z*y.w,
                    x.y*w.z*z.w + z.y*x.z*w.w + w.y*z.z*x.w - w.y*x.z*z.w - z.y*w.z*x.w - x.y*z.z*w.w,
                    x.y*y.z*w.w + w.y*x.z*y.w + y.y*w.z*x.w - x.y*w.z*y.w - y.y*x.z*w.w - w.y*y.z*x.w,
                    x.y*z.z*y.w + y.y*x.z*z.w + z.y*y.z*x.w - x.y*y.z*z.w - z.y*x.z*y.w - y.y*z.z*x.w),
            y: vec4(y.z*w.w*z.x + z.z*y.w*w.x + w.z*z.w*y.x - y.z*z.w*w.x - w.z*y.w*z.x - z.z*w.w*y.x,
                    x.z*z.w*w.x + w.z*x.w*z.x + z.z*w.w*x.x - x.z*w.w*z.x - z.z*x.w*w.x - w.z*z.w*x.x,
                    x.z*w.w*y.x + y.z*x.w*w.x + w.z*y.w*x.x - x.z*y.w*w.x - w.z*x.w*y.x - y.z*w.w*x.x,
                    x.z*y.w*z.x + z.z*x.w*y.x + y.z*z.w*x.x - x.z*z.w*y.x - y.z*x.w*z.x - z.z*y.w*x.x),
            z: vec4(y.w*z.x*w.y + w.w*y.x*z.y + z.w*w.x*y.y - y.w*w.x*z.y - z.w*y.x*w.y - w.w*z.x*y.y,
                    x.w*w.x*z.y + z.w*x.x*w.y + w.w*z.x*x.y - x.w*z.x*w.y - w.w*x.x*z.y - z.w*w.x*x.y,
                    x.w*y.x*w.y + w.w*x.x*y.y + y.w*w.x*x.y - x.w*w.x*y.y - y.w*x.x*w.y - w.w*y.x*x.y,
                    x.w*z.x*y.y + y.w*x.x*z.y + z.w*y.x*x.y - x.w*y.x*z.y - z.w*x.x*y.y - y.w*z.x*x.y),
            w: vec4(y.x*w.y*z.z + z.x*y.y*w.z + w.x*z.y*y.z - y.x*z.y*w.z - w.x*y.y*z.z - z.x*w.y*y.z,
                    x.x*z.y*w.z + w.x*x.y*z.z + z.x*w.y*x.z - x.x*w.y*z.z - z.x*x.y*w.z - w.x*z.y*x.z,
                    x.x*w.y*y.z + y.x*x.y*w.z + w.x*y.y*x.z - x.x*y.y*w.z - w.x*x.y*y.z - y.x*w.y*x.z,
                    x.x*y.y*z.z + z.x*x.y*y.z + y.x*z.y*x.z - x.x*z.y*y.z - y.x*x.y*z.z - z.x*y.y*x.z)
        }
    }

    #[inline]
    fn transpose(&self) -> M4x4 {
        M4x4::from_rows(self.x, self.y, self.z, self.w)
    }

    #[inline]
    fn row(&self, i: usize) -> V4 {
        vec4(self.x[i], self.y[i], self.z[i], self.w[i])
    }

    #[inline]
    fn col(&self, i: usize) -> V4 {
        self[i]
    }
}

impl M4x4 {
    #[inline]
    pub fn from_translation(v: V3) -> M4x4 {
        M4x4::new(1.0, 0.0, 0.0, 0.0,
                  0.0, 1.0, 0.0, 0.0,
                  0.0, 0.0, 1.0, 0.0,
                  v.x, v.y, v.z, 1.0)
    }

    #[inline]
    pub fn from_rotation(q: Quat) -> M4x4 {
        M4x4::from_cols(V4::expand(q.x_dir(), 0.0),
                        V4::expand(q.y_dir(), 0.0),
                        V4::expand(q.z_dir(), 0.0),
                        V4::new(0.0, 0.0, 0.0, 1.0))
    }

     #[inline]
    pub fn from_scale(v: V3) -> M4x4 {
        M4x4::new(v.x, 0.0, 0.0, 0.0,
                  0.0, v.y, 0.0, 0.0,
                  0.0, 0.0, v.z, 0.0,
                  0.0, 0.0, 0.0, 1.0)
    }

    #[inline]
    pub fn from_pose(p: V3, q: Quat) -> M4x4 {
        M4x4::from_cols(V4::expand(q.x_dir(), 0.0),
                        V4::expand(q.y_dir(), 0.0),
                        V4::expand(q.z_dir(), 0.0),
                        V4::expand(p, 1.0))
    }

    #[inline]
    pub fn from_pose_2(p: V3, q: Quat) -> M4x4 {
        M4x4::from_rotation(q)*M4x4::from_translation(p)
    }

    #[inline]
    pub fn new_frustum(l: f32, r: f32, b: f32, t: f32, n: f32, f: f32) -> M4x4 {
        M4x4::new(2.0 * n / (r-l),   0.0,               0.0,                    0.0,
                  0.0,               2.0 * n / (t - b), 0.0,                    0.0,
                  (r + l) / (r - l), (t + b) / (t - b), -(f + n) / (f - n),    -1.0,
                  0.0,               0.0,               -2.0 * f * n / (f - n), 0.0)
    }

    #[inline]
    pub fn perspective(fovy: f32, aspect: f32, n: f32, f: f32) -> M4x4 {
        let y = n * (fovy*0.5).tan();
        let x = y * aspect;
        M4x4::new_frustum(-x, x, -y, y, n, f)
    }

    #[inline]
    pub fn look_towards(fwd: V3, up: V3) -> M4x4 {
        let f = fwd.norm_or(1.0, 0.0, 0.0);
        let s = f.cross(up).norm_or(0.0, 1.0, 0.0);
        let u = s.cross(f);
        M4x4::new(s.x, u.x, -f.x, 0.0,
                  s.y, u.y, -f.y, 0.0,
                  s.z, u.z, -f.z, 0.0,
                  0.0, 0.0,  0.0, 1.0)
    }

    #[inline]
    pub fn look_at(eye: V3, center: V3, up: V3) -> M4x4 {
        M4x4::look_towards(center-eye, up) * M4x4::from_translation(-eye)
    }

}

// #[inline]
// pub fn determinant<M: MatType>(m: &M) -> f32 {
//     m.determinant()
// }

// #[inline]
// pub fn adjugate<M: MatType>(m: &M) -> M {
//     m.adjugate()
// }

// #[inline]
// pub fn transpose<M: MatType>(m: &M) -> M {
//     m.transpose()
// }

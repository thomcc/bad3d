// XXX this file needs cleanup!
use crate::mat::*;
use crate::scalar::*;
use crate::traits::*;
use std::ops::*;
use std::{self, fmt, slice};
mod iter;

mod v3;
mod v4;
pub use iter::VecIter;
#[cfg(not(target_feature = "sse2"))]
#[doc(hidden)]
pub use v3::__v3_const;
pub use v3::{vec3, V3};
#[cfg(not(target_feature = "sse2"))]
#[doc(hidden)]
pub use v4::__v4_const;
pub use v4::{vec4, V4};
pub(crate) mod idx3;
pub mod ivec;

#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[repr(C)]
pub struct V2 {
    pub x: f32,
    pub y: f32,
}

#[inline]
pub const fn vec2(x: f32, y: f32) -> V2 {
    V2 { x, y }
}

impl fmt::Display for V2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "vec2({}, {})", self.x, self.y)
    }
}
// pub trait VecType:
//     Copy
//     + Clone
//     + Fold
//     + Map
//     + ApproxEq
//     + Add<Output = Self>
//     + Sub<Output = Self>
//     + AddAssign
//     + SubAssign
//     + Mul<f32, Output = Self>
//     + Div<f32, Output = Self>
//     + MulAssign<f32>
//     + DivAssign<f32>
//     + Neg<Output = Self>
//     + Mul<Output = Self>
//     + Div<Output = Self>
//     + MulAssign
//     + DivAssign
//     + AsRef<[f32]>
//     + AsMut<[f32]>
//     + Index<usize, Output = f32>
//     + Index<Range<usize>, Output = [f32]>
//     + Index<RangeFrom<usize>, Output = [f32]>
//     + Index<RangeTo<usize>, Output = [f32]>
//     + Index<RangeFull, Output = [f32]>
//     + IndexMut<usize, Output = f32>
//     + IndexMut<Range<usize>, Output = [f32]>
//     + IndexMut<RangeFrom<usize>, Output = [f32]>
//     + IndexMut<RangeTo<usize>, Output = [f32]>
//     + IndexMut<RangeFull, Output = [f32]>
// {
//     const SIZE: usize;

//     fn splat(v: f32) -> Self;
//     #[inline]
//     fn min(self, o: Self) -> Self {
//         self.map2(o, |a, b| a.min(b))
//     }
//     #[inline]
//     fn max(self, o: Self) -> Self {
//         self.map2(o, |a, b| a.max(b))
//     }
//     fn dot(self, o: Self) -> f32;
// }

impl V2 {
    #[inline]
    fn fold(self, f: impl Fn(f32, f32) -> f32) -> f32 {
        f(self.x, self.y)
    }

    #[inline]
    fn fold2_init<T>(self, o: Self, init: T, f: impl Fn(T, f32, f32) -> T) -> T {
        f(f(init, o.x, self.x), o.y, self.y)
    }

    #[inline]
    pub fn map2<F>(self, o: Self, f: F) -> Self
    where
        F: Fn(f32, f32) -> f32,
    {
        self.map3(o, self, |a, b, _| f(a, b))
    }

    #[inline]
    pub fn map<F>(self, f: F) -> Self
    where
        F: Fn(f32) -> f32,
    {
        self.map3(self, self, |a, _, _| f(a))
    }
}

macro_rules! vec_from {
    ($dst:ty [$(($id:ident : $src:ty) -> $ex:expr);+ $(;)* ]) => {$(
        impl From<$src> for $dst {
            #[inline] fn from($id : $src) -> Self { $ex }
        }
    )+};
}

vec_from! { V3 [
    (t: (V2, f32)) -> vec3(t.0.x, t.0.y, t.1);
    (t: (f32, V2)) -> vec3(t.0, t.1.x, t.1.y);
]}

vec_from! { V4 [
    (t: (V2, f32, f32)) -> vec4(t.0.x, t.0.y, t.1, t.2);
    (t: (f32, V2, f32)) -> vec4(t.0, t.1.x, t.1.y, t.2);
    (t: (f32, f32, V2)) -> vec4(t.0, t.1, t.2.x, t.2.y);
    (t: (V2, V2)) -> vec4(t.0.x, t.0.y, t.1.x, t.1.y);
    (t: (V3, f32)) -> vec4(t.0.x(), t.0.y(), t.0.z(), t.1);
    (t: (f32, V3)) -> vec4(t.0, t.1.x(), t.1.y(), t.1.z());
]}

macro_rules! impl_index_op {
    ($Vn:ident, $Indexer:ty, $out_type:ty) => {
        impl Index<$Indexer> for $Vn {
            type Output = $out_type;
            #[inline]
            fn index(&self, index: $Indexer) -> &$out_type {
                &self.as_array()[index]
            }
        }

        impl IndexMut<$Indexer> for $Vn {
            #[inline]
            fn index_mut(&mut self, index: $Indexer) -> &mut $out_type {
                &mut self.as_mut_array()[index]
            }
        }
    };
}

macro_rules! do_vec_boilerplate {
    ($Vn: ident { $($field: ident : $index: expr),+ }, $length: expr, $tuple_ty: ty) => {
        impl AsRef<[f32; $length]> for $Vn {
            #[inline]
            fn as_ref(&self) -> &[f32; $length] {
                self.as_array()
            }
        }

        impl AsMut<[f32; $length]> for $Vn {
            #[inline]
            fn as_mut(&mut self) -> &mut [f32; $length] {
                self.as_mut_array()
            }
        }

        impl AsRef<[f32]> for $Vn {
            #[inline]
            fn as_ref(&self) -> &[f32] {
                self.as_slice()
            }
        }

        impl AsMut<[f32]> for $Vn {
            #[inline]
            fn as_mut(&mut self) -> &mut [f32] {
                self.as_mut_slice()
            }
        }

        impl From<$Vn> for [f32; $length] {
            #[inline] fn from(v: $Vn) -> [f32; $length] { [$(v.$field),+] }
        }

        impl From<$Vn> for $tuple_ty {
            #[inline] fn from(v: $Vn) -> $tuple_ty { ($(v.$field),+) }
        }

        impl From<[f32; $length]> for $Vn {
            #[inline] fn from(v: [f32; $length]) -> $Vn { $Vn { $($field: v[$index]),+ } }
        }

        impl From<$tuple_ty> for $Vn {
            #[inline]
            fn from(v: $tuple_ty) -> Self {
                let ($($field),+) = v;
                $Vn { $($field: $field),+ }
            }
        }

        impl From<f32> for $Vn {
            #[inline] fn from(v: f32) -> Self { Self { $($field: v),+ } }
        }

        impl_index_op!($Vn, usize, f32);
        impl_index_op!($Vn, Range<usize>, [f32]);
        impl_index_op!($Vn, RangeFrom<usize>, [f32]);
        impl_index_op!($Vn, RangeTo<usize>, [f32]);
        impl_index_op!($Vn, RangeFull, [f32]);

        impl Neg for $Vn {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self {
                Self { $($field: -self.$field),+ }
            }
        }

        impl Add for $Vn {
            type Output = Self;
            #[inline]
            fn add(self, o: Self) -> Self {
                Self { $($field: (self.$field + o.$field)),+ }
            }
        }

        impl Sub for $Vn {
            type Output = Self;
            #[inline]
            fn sub(self, o: Self) -> Self {
                Self { $($field: (self.$field - o.$field)),+ }
            }
        }

        impl Mul for $Vn {
            type Output = Self;
            #[inline]
            fn mul(self, o: Self) -> Self {
                Self { $($field: (self.$field * o.$field)),+ }
            }
        }

        impl Div for $Vn {
            type Output = Self;
            #[inline]
            fn div(self, o: Self) -> Self {
                $(debug_assert_ne!(o.$field, 0.0));+;
                Self { $($field: (self.$field / o.$field)),+ }
            }
        }

        impl Mul<f32> for $Vn {
            type Output = Self;
            #[inline]
            fn mul(self, o: f32) -> Self {
                Self { $($field: (self.$field * o)),+ }
            }
        }

        impl Mul<$Vn> for f32 {
            type Output = $Vn;
            #[inline]
            fn mul(self, v: $Vn) -> $Vn {
                v * self
            }
        }

        impl Div<f32> for $Vn {
            type Output = Self;
            #[inline]
            #[allow(clippy::suspicious_arithmetic_impl)]
            fn div(self, o: f32) -> Self {
                debug_assert_ne!(o, 0.0);
                let inv = 1.0 / o;
                Self { $($field: (self.$field * inv)),+ }
            }
        }

        impl Div<$Vn> for f32 {
            type Output = $Vn;
            #[inline]
            fn div(self, v: $Vn) -> $Vn {
                $(debug_assert_ne!(v.$field, 0.0);)+;
                $Vn { $($field: (self / v.$field)),+ }
            }
        }

        impl MulAssign for $Vn {
            #[inline]
            fn mul_assign(&mut self, rhs: Self) {
                $(self.$field *= rhs.$field;)+
            }
        }

        impl DivAssign for $Vn {
            #[inline]
            fn div_assign(&mut self, rhs: Self) {
                $(self.$field /= rhs.$field;)+
            }
        }

        impl MulAssign<f32> for $Vn {
            #[inline]
            fn mul_assign(&mut self, rhs: f32) {
                $(self.$field *= rhs;)+
            }
        }

        impl DivAssign<f32> for $Vn {
            #[inline]
            fn div_assign(&mut self, rhs: f32) {
                let inv = 1.0 / rhs;
                $(self.$field *= inv;)+
            }
        }

        impl AddAssign for $Vn {
            #[inline]
            fn add_assign(&mut self, rhs: Self) {
                $(self.$field += rhs.$field;)+
            }
        }

        impl SubAssign for $Vn {
            #[inline]
            fn sub_assign(&mut self, rhs: Self) {
                $(self.$field -= rhs.$field;)+
            }
        }


        impl $Vn {
            pub const ZERO: $Vn = $Vn { $($field: 0.0),+ };

            #[inline]
            pub const fn zero() -> Self {
                Self::ZERO
            }

            #[inline]
            pub fn new($($field: f32),+) -> Self {
                Self { $($field: $field),+ }
            }

            #[inline]
            pub fn splat(v: f32) -> Self {
                Self { $($field: v),+ }
            }

            #[inline]
            pub fn as_slice(&self) -> &[f32] {
                self.as_array()
            }
            #[inline]
            pub fn as_mut_slice(&mut self) -> &mut [f32] {
                self.as_mut_array()
            }

            #[inline]
            pub fn as_ptr(&self) -> *const f32 {
                &self.x as *const f32
            }

            #[inline]
            pub fn as_mut_ptr(&mut self) -> *mut f32 {
                &mut self.x as *mut f32
            }

            #[inline]
            pub fn as_array(&self) -> &[f32; $length] {
                unsafe { &*(self as *const $Vn as *const [f32; $length]) }
            }

            #[inline]
            pub fn as_mut_array(&mut self) -> &mut [f32; $length] {
                unsafe { &mut *(self as *mut $Vn as *mut [f32; $length]) }
            }

            #[inline] pub fn tup(self) -> $tuple_ty { self.into() }

            #[inline] pub fn len(&self) -> usize { $length }

            #[inline]
            pub fn iter(&self) -> slice::Iter<'_, f32> {
                self.as_slice().iter()
            }

            #[inline]
            pub fn iter_mut(&mut self) -> slice::IterMut<'_, f32> {
                self.as_mut_slice().iter_mut()
            }

            #[inline] pub fn count() -> usize { $length }

            #[inline] pub fn max_elem(self) -> f32 { self.fold(|a, b| a.max(b)) }
            #[inline] pub fn min_elem(self) -> f32 { self.fold(|a, b| a.min(b)) }

            #[inline] pub fn abs(self) -> Self { self.map(|x| x.abs()) }

            #[inline] pub fn floor(self) -> Self { self.map(|x| x.floor()) }
            #[inline] pub fn ceil(self) -> Self { self.map(|x| x.ceil()) }
            #[inline] pub fn round(self) -> Self { self.map(|x| x.round()) }

            #[inline] pub fn min(self, o: Self) -> Self { self.map2(o, |a, b| a.min(b)) }
            #[inline] pub fn max(self, o: Self) -> Self { self.map2(o, |a, b| a.max(b)) }

            #[inline] pub fn length_sq(self) -> f32 { self.dot(self) }
            #[inline] pub fn length(self) -> f32 { self.length_sq().sqrt() }

            #[inline] pub fn towards(self, o: Self) -> Self { o - self }
            #[inline] pub fn dir_towards(self, o: Self) -> Self { self.towards(o).norm_or_zero() }

            #[inline] pub fn dist_sq(self, o: Self) -> f32 { (o - self).length_sq() }
            #[inline] pub fn dist(self, o: Self) -> f32 { (o - self).length() }

            #[inline] pub fn same_dir(self, o: Self) -> bool { self.dot(o) > 0.0 }

            #[inline]
            pub fn clamp(self, min: Self, max: Self) -> Self {
                self.map3(min, max, clamp)
            }

            #[inline]
            pub fn lerp(self, b: Self, t: f32) -> Self {
                self.map2(b, |x, y| lerp(x, y, t))
            }

            #[inline]
            pub fn safe_div(self, v: f32) -> Option<Self> {
                safe_div(1.0, v).map(|inv| self * inv)
            }

            #[inline]
            pub fn safe_div0(self, v: f32) -> Self {
                self * safe_div0(1.0, v)
            }

            #[inline]
            pub fn div_unless_zero(self, v: f32) -> Self {
                self * safe_div1(1.0, v)
            }

            #[inline]
            pub fn clamp_length(self, max_len: f32) -> Self {
                let len = self.length();
                if len < max_len { self }
                else { self * (max_len / len) }
            }

            #[inline]
            pub fn norm_len(self) -> (Option<Self>, f32) {
                let l = self.length();
                if l == 0.0 {
                    (None, l)
                } else {
                    let il = 1.0 / l;
                    (Some(self * il), l)
                }
            }

            #[inline]
            pub fn normalize(self) -> Option<Self> {
                self.norm_len().0
            }

            #[inline]
            pub fn norm_or_zero(self) -> Self {
                self.normalize().unwrap_or(Self::zero())
            }

            #[inline]
            pub fn norm_or_v(self, v: Self) -> Self {
                self.normalize().unwrap_or(v)
            }

            #[inline]
            pub fn must_norm(self) -> Self {
                self.normalize().unwrap()
            }

            #[inline]
            pub fn fast_norm(self) -> Self {
                let mut len_sq = 0.0;
                $(len_sq += self.$field * self.$field;)+
                debug_assert_ne!(len_sq, 0.0);
                // SOMEDAY: rsqrt
                let ilen = 1.0 / len_sq.sqrt();
                self * ilen
            }

            #[inline]
            pub fn is_normalized(self) -> bool {
                self.length_sq().approx_eq(&1.0)
            }

            #[inline]
            pub fn is_normalized_e(self, epsilon: f32) -> bool {
                self.length_sq().approx_eq_e(&1.0, epsilon)
            }

            #[inline]
            pub fn is_zero(self) -> bool {
                self.dot(self) == 0.0_f32
            }

            #[inline]
            pub fn hadamard(self, o: Self) -> Self {
                self.map2(o, |a, b| a * b)
            }

            #[inline]
            pub fn round_to(self, p: f32) -> Self {
                self.map(|v| round_to(v, p))
            }

            #[inline]
            pub fn unit_axis(axis: usize) -> Self {
                chek::debug_lt!(axis, $length, "Invalid axis");
                let mut v = Self::zero();
                v[axis] = 1.0;
                v
            }

            #[inline]
            pub fn angle(self, o: Self) -> f32 {
                let d = self.dot(o);
                if d > 1.0 { 0.0 }
                else { d.max(-1.0).acos() }
            }

            #[inline]
            pub fn slerp(self, o: Self, t: f32) -> Self {
                let th: f32 = self.angle(o);
                if th.approx_zero() {
                    self
                } else {
                    let isth = 1.0 / th.sin();
                    let s0 = isth * (th * (1.0 - t)).sin();
                    let s1 = isth * (th * t).sin();
                    self * s0 + o * s1
                }
            }

            #[inline]
            pub fn nlerp(self, o: Self, t: f32) -> Self {
                self.lerp(o, t).norm_or_v(o)
            }

            #[inline]
            pub fn norm_or(self, $($field: f32),+) -> Self {
                self.norm_or_v(Self { $($field),+ })
            }

            // #[inline]
            // pub fn identity() -> Self {
            //     $Vn { $($field: 0.0),+ }
            // }
        }


        impl ApproxEq for $Vn {

            #[inline]
            fn approx_zero_e(&self, e: f32) -> bool {
                self.fold2_init(*self, true, |cnd, val, _| cnd && val.approx_zero_e(e))
            }

            #[inline]
            fn approx_eq_e(&self, o: &Self, e: f32) -> bool {
                self.fold2_init(*o, true, |cnd, l, r| cnd && l.approx_eq_e(&r, e))
            }
        }

        impl $Vn {
            #[inline]
            fn map3<F: Fn(f32, f32, f32) -> f32>(self, a: Self, b: Self, f: F) -> Self {
                Self { $($field: f(self.$field, a.$field, b.$field)),+ }
            }
        }

        // impl VecType for $Vn {
        //     const SIZE: usize = $length;

        //     #[inline]
        //     fn splat(v: f32) -> Self {
        //         Self { $($field: v),+ }
        //     }
        //     #[inline]
        //     fn dot(self, o: Self) -> f32 {
        //         $Vn::dot(self, o)
        //     }
        // }
    }
}

do_vec_boilerplate!(V2 { x: 0, y: 1 }, 2, (f32, f32));

// do_vec_boilerplate!(V3 { x: 0, y: 1, z: 2 }, 3, (f32, f32, f32));

// do_vec_boilerplate!(V4 { x: 0, y: 1, z: 2, w: 3 }, 4, (f32, f32, f32, f32));

impl_index_op!(V4, usize, f32);
impl_index_op!(V4, Range<usize>, [f32]);
impl_index_op!(V4, RangeFrom<usize>, [f32]);
impl_index_op!(V4, RangeTo<usize>, [f32]);
impl_index_op!(V4, RangeFull, [f32]);

impl_index_op!(V3, usize, f32);
impl_index_op!(V3, Range<usize>, [f32]);
impl_index_op!(V3, RangeFrom<usize>, [f32]);
impl_index_op!(V3, RangeTo<usize>, [f32]);
impl_index_op!(V3, RangeFull, [f32]);

impl V2 {
    #[inline]
    pub fn unit() -> V2 {
        V2::new(1.0, 0.0)
    }
    #[inline]
    pub fn outer_prod(self, o: V2) -> M2x2 {
        M2x2 {
            x: self * o.x,
            y: self * o.y,
        }
    }
    #[inline]
    pub fn cross(self, o: V2) -> f32 {
        self.x * o.y - self.y * o.x
    }
    #[inline]
    pub fn dot(self, o: V2) -> f32 {
        self.x * o.x + self.y * o.y
    }
    #[inline]
    pub fn norm_or_unit(self) -> V2 {
        self.norm_or(0.0, 1.0)
    }
    #[inline]
    pub fn to_arr(self) -> [f32; 2] {
        [self.x, self.y]
    }

    #[inline]
    pub fn to_arr16(self) -> [u16; 2] {
        debug_assert!(self.x >= 0.0 && self.x <= 1.0, "x out of range {}", self.x);
        debug_assert!(self.y >= 0.0 && self.y <= 1.0, "y out of range {}", self.y);
        [
            (self.x * (std::u16::MAX as f32)).trunc() as u16,
            (self.y * (std::u16::MAX as f32)).trunc() as u16,
        ]
    }

    #[inline]
    pub fn max_index(self) -> usize {
        if self.x > self.y {
            0
        } else {
            1
        }
    }

    #[inline]
    pub fn min_index(self) -> usize {
        if self.x > self.y {
            1
        } else {
            0
        }
    }
}

impl From<V3> for V2 {
    #[inline]
    fn from(v: V3) -> V2 {
        V2 { x: v.x(), y: v.y() }
    }
}

impl From<V3> for V4 {
    #[inline]
    fn from(v: V3) -> V4 {
        V4::expand(v, 0.0)
    }
}
impl From<V4> for V2 {
    #[inline]
    fn from(v: V4) -> V2 {
        V2 { x: v.x(), y: v.y() }
    }
}

impl From<V2> for V4 {
    #[inline]
    fn from(v: V2) -> V4 {
        vec4(v.x, v.y, 0.0, 0.0)
    }
}

impl IntoIterator for V2 {
    type Item = f32;
    type IntoIter = VecIter<f32>;
    #[inline]
    fn into_iter(self) -> VecIter<f32> {
        VecIter::new([self.x, self.y, 0.0, 0.0], 2)
    }
}

impl IntoIterator for V4 {
    type Item = f32;
    type IntoIter = VecIter<f32>;
    #[inline]
    fn into_iter(self) -> VecIter<f32> {
        VecIter::new(self.arr(), 4)
    }
}

use crate::math::mat::*;
use crate::math::scalar::*;
use crate::math::traits::*;
use std::ops::*;
use std::{self, fmt, iter, mem, slice};

#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[repr(C)]
pub struct V2 {
    pub x: f32,
    pub y: f32,
}

#[inline]
pub fn vec2(x: f32, y: f32) -> V2 {
    V2 { x, y }
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[repr(C)]
pub struct V3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[inline]
pub fn vec3(x: f32, y: f32, z: f32) -> V3 {
    V3 { x, y, z }
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[repr(C)]
pub struct V4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

#[inline]
pub fn vec4(x: f32, y: f32, z: f32, w: f32) -> V4 {
    V4 { x, y, z, w }
}

impl fmt::Display for V2 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "vec2({}, {})", self.x, self.y)
    }
}

impl fmt::Display for V3 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "vec3({}, {}, {})", self.x, self.y, self.z)
    }
}

impl fmt::Display for V4 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "vec4({}, {}, {}, {})", self.x, self.y, self.z, self.w)
    }
}

pub trait VecType:
    Copy
    + Clone
    + Fold
    + Map
    + ApproxEq
    + Identity
    + Zero
    + Add<Output = Self>
    + Sub<Output = Self>
    + AddAssign
    + SubAssign
    + Mul<f32, Output = Self>
    + Div<f32, Output = Self>
    + MulAssign<f32>
    + DivAssign<f32>
    + Neg<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + MulAssign
    + DivAssign
    + AsRef<[f32]>
    + AsMut<[f32]>
    + Index<usize, Output = f32>
    + Index<Range<usize>, Output = [f32]>
    + Index<RangeFrom<usize>, Output = [f32]>
    + Index<RangeTo<usize>, Output = [f32]>
    + Index<RangeFull, Output = [f32]>
    + IndexMut<usize, Output = f32>
    + IndexMut<Range<usize>, Output = [f32]>
    + IndexMut<RangeFrom<usize>, Output = [f32]>
    + IndexMut<RangeTo<usize>, Output = [f32]>
    + IndexMut<RangeFull, Output = [f32]>
{
    const SIZE: usize;

    fn splat(v: f32) -> Self;
    #[inline]
    fn min(self, o: Self) -> Self {
        self.map2(o, |a, b| a.min(b))
    }
    #[inline]
    fn max(self, o: Self) -> Self {
        self.map2(o, |a, b| a.max(b))
    }

}

impl Fold for V2 {
    #[inline]
    fn fold(self, f: impl Fn(f32, f32) -> f32) -> f32 {
        f(self.x, self.y)
    }

    #[inline]
    fn fold2_init<T>(self, o: Self, init: T, f: impl Fn(T, f32, f32) -> T) -> T {
        f(f(init, o.x, self.x), o.y, self.y)
    }
}

impl Fold for V3 {
    #[inline]
    fn fold(self, f: impl Fn(f32, f32) -> f32) -> f32 {
        f(f(self.x, self.y), self.z)
    }

    #[inline]
    fn fold2_init<T>(self, o: Self, init: T, f: impl Fn(T, f32, f32) -> T) -> T {
        f(f(f(init, o.x, self.x), o.y, self.y), o.z, self.z)
    }
}

impl Fold for V4 {
    #[inline]
    fn fold(self, f: impl Fn(f32, f32) -> f32) -> f32 {
        f(f(f(self.x, self.y), self.z), self.w)
    }

    #[inline]
    fn fold2_init<T>(self, o: Self, init: T, f: impl Fn(T, f32, f32) -> T) -> T {
        f(
            f(f(f(init, o.x, self.x), o.y, self.y), o.z, self.z),
            o.w,
            self.w,
        )
    }
}

macro_rules! vec_from {
    ($dst:ty [$(($id:ident : $src:ty) -> $ex:expr);+ $(;)* ]) => {$(
        impl From<$src> for $dst {
            #[inline] fn from($id : $src) -> Self { $ex }
        }
    )+};
}

vec_from! { V2 [
    (t: (i32, i32)) -> vec2(t.0 as f32, t.1 as f32);
    (t: (u32, u32)) -> vec2(t.0 as f32, t.1 as f32);
    (t: (usize, usize)) -> vec2(t.0 as f32, t.1 as f32);
    (t: (isize, isize)) -> vec2(t.0 as f32, t.1 as f32);
    (t: (f64, f64)) -> vec2(t.0 as f32, t.1 as f32);

    (t: [i32; 2]) -> vec2(t[0] as f32, t[1] as f32);
    (t: [u32; 2]) -> vec2(t[0] as f32, t[1] as f32);
    (t: [isize; 2]) -> vec2(t[0] as f32, t[1] as f32);
    (t: [usize; 2]) -> vec2(t[0] as f32, t[1] as f32);
    (t: [f64; 2]) -> vec2(t[0] as f32, t[1] as f32);
]}

vec_from! { V3 [
    (t: (V2, f32)) -> vec3(t.0.x, t.0.y, t.1);
    (t: (f32, V2)) -> vec3(t.0, t.1.x, t.1.y);

    (t: (V2, i32)) -> vec3(t.0.x, t.0.y, t.1 as f32);
    (t: (i32, V2)) -> vec3(t.0 as f32, t.1.x, t.1.y);

    (t: (i32, i32, i32)) -> vec3(t.0 as f32, t.1 as f32, t.2 as f32);
    (t: (u32, u32, u32)) -> vec3(t.0 as f32, t.1 as f32, t.2 as f32);
    (t: (isize, isize, isize)) -> vec3(t.0 as f32, t.1 as f32, t.2 as f32);
    (t: (usize, usize, usize)) -> vec3(t.0 as f32, t.1 as f32, t.2 as f32);
    (t: (f64, f64, f64)) -> vec3(t.0 as f32, t.1 as f32, t.2 as f32);

    (t: [i32; 3]) -> vec3(t[0] as f32, t[1] as f32, t[2] as f32);
    (t: [u32; 3]) -> vec3(t[0] as f32, t[1] as f32, t[2] as f32);
    (t: [isize; 3]) -> vec3(t[0] as f32, t[1] as f32, t[2] as f32);
    (t: [usize; 3]) -> vec3(t[0] as f32, t[1] as f32, t[2] as f32);
    (t: [f64; 3]) -> vec3(t[0] as f32, t[1] as f32, t[2] as f32);
]}

vec_from! { V4 [
    (t: (V2, f32, f32)) -> vec4(t.0.x, t.0.y, t.1, t.2);
    (t: (f32, V2, f32)) -> vec4(t.0, t.1.x, t.1.y, t.2);
    (t: (f32, f32, V2)) -> vec4(t.0, t.1, t.2.x, t.2.y);
    (t: (V2, V2)) -> vec4(t.0.x, t.0.y, t.1.x, t.1.y);

    (t: (V3, f32)) -> vec4(t.0.x, t.0.y, t.0.z, t.1);
    (t: (f32, V3)) -> vec4(t.0, t.1.x, t.1.y, t.1.z);

    (t: (V2, i32, i32)) -> vec4(t.0.x, t.0.y, t.1 as f32, t.2 as f32);
    (t: (i32, V2, i32)) -> vec4(t.0 as f32, t.1.x, t.1.y, t.2 as f32);
    (t: (i32, i32, V2)) -> vec4(t.0 as f32, t.1 as f32, t.2.x, t.2.y);

    (t: (V3, i32)) -> vec4(t.0.x, t.0.y, t.0.z, t.1 as f32);
    (t: (i32, V3)) -> vec4(t.0 as f32, t.1.x, t.1.y, t.1.z);

    (t: (i32, i32, i32, i32)) -> vec4(t.0 as f32, t.1 as f32, t.2 as f32, t.3 as f32);
    (t: (u32, u32, u32, u32)) -> vec4(t.0 as f32, t.1 as f32, t.2 as f32, t.3 as f32);
    (t: (f64, f64, f64, f64)) -> vec4(t.0 as f32, t.1 as f32, t.2 as f32, t.3 as f32);

    (t: [i32;   4]) -> vec4(t[0] as f32, t[1] as f32, t[2] as f32, t[3] as f32);
    (t: [u32;   4]) -> vec4(t[0] as f32, t[1] as f32, t[2] as f32, t[3] as f32);
    (t: [isize; 4]) -> vec4(t[0] as f32, t[1] as f32, t[2] as f32, t[3] as f32);
    (t: [usize; 4]) -> vec4(t[0] as f32, t[1] as f32, t[2] as f32, t[3] as f32);
    (t: [f64;   4]) -> vec4(t[0] as f32, t[1] as f32, t[2] as f32, t[3] as f32);
]}

// @@ partially duplicated in mat.rs...
macro_rules! define_transmute_conversions {
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
    };
}

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

#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct VecIter<T: VecType> {
    pub v: T,
    p: usize,
    e: usize,
}

impl<T: VecType> VecIter<T> {
    #[inline]
    pub fn new(v: T) -> Self {
        Self {
            v,
            p: 0,
            e: <T as VecType>::SIZE,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        debug_assert_le!(self.e, <T as VecType>::SIZE);
        self.e.saturating_sub(self.p)
    }

    // #[inline]
    // fn remaining(&self, offset: isize) -> usize {
    //     debug_assert_le!(self.e, <T as VecType>::SIZE);
    //     let (p, e) = (self.p as isize, self.e as isize);
    //     let l = e - (p + offset);
    //     if l < 0 { 0 } else { l as usize }
    // }

    #[inline]
    unsafe fn raw_get(&self, p: usize) -> f32 {
        debug_assert_le!(self.e, <T as VecType>::SIZE);
        debug_assert_lt!(p, <T as VecType>::SIZE);
        if cfg!(debug_assertions) {
            *self.v.as_ref().get_unchecked(p)
        } else {
            self.v[p]
        }
    }

    #[inline]
    fn do_iter_fwd(&mut self, pre: usize, post: usize) -> Option<usize> {
        debug_assert_le!(self.e, <T as VecType>::SIZE);
        let (p, e) = (self.p + pre, self.e);
        if p >= e {
            self.p = self.e;
            None
        } else {
            self.p = p + post;
            Some(p)
        }
    }

    // #[inline]
    // fn do_iter_back(&mut self, pre: usize, post: usize) -> Option<usize> {
    //     debug_assert_le!(self.e, <T as VecType>::SIZE);
    //     let (p, e) = (pre, self.e);
    //     if p >= e { self.p = self.e; None }
    //     else { self.p = p + post; Some(p) }
    // }
}

impl<T: VecType> Iterator for VecIter<T> {
    type Item = f32;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.do_iter_fwd(0, 1).map(|i| unsafe { self.raw_get(i) })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        if self.p >= self.e {
            None
        } else {
            Some(unsafe { self.raw_get(<T as VecType>::SIZE - 1) })
        }
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.do_iter_fwd(n, 1).map(|i| unsafe { self.raw_get(i) })
    }
}

impl<T: VecType> iter::ExactSizeIterator for VecIter<T> {}

impl<T: VecType> iter::DoubleEndedIterator for VecIter<T> {
    #[inline]
    fn next_back(&mut self) -> Option<f32> {
        if self.p >= self.e {
            return None;
        }
        debug_assert_gt!(self.e, 1);
        self.e -= 1;
        Some(unsafe { self.raw_get(self.e) })
    }
}

macro_rules! first_expr {
    ($fst:expr, $($_rest:expr),+) => {
        $fst
    };
}

macro_rules! do_vec_boilerplate {
    ($Vn: ident { $($field: ident : $index: expr),+ }, $length: expr, $tuple_ty: ty) => {
        define_transmute_conversions!($Vn, [f32; $length]);
        define_transmute_conversions!($Vn, $tuple_ty);

        impl AsRef<[f32]> for $Vn {
            #[inline]
            fn as_ref(&self) -> &[f32] {
                unsafe { slice::from_raw_parts(self.as_ptr(), $length) }
            }
        }

        impl AsMut<[f32]> for $Vn {
            #[inline]
            fn as_mut(&mut self) -> &mut [f32] {
                unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), $length) }
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

        impl Zero for $Vn {
            const ZERO: $Vn = $Vn { $($field: 0.0),+ };
        }

        impl Identity for $Vn {
            const IDENTITY: $Vn = $Vn { $($field: 0.0),+ };
        }

        impl $Vn {
            #[inline]
            pub fn new($($field: f32),+) -> Self {
                Self { $($field: $field),+ }
            }

            #[inline]
            pub fn splat(v: f32) -> Self {
                Self { $($field: v),+ }
            }

            #[inline] pub fn as_slice(&self) -> &[f32] { self.as_ref() }
            #[inline] pub fn as_mut_slice(&mut self) -> &mut [f32] { self.as_mut() }

            #[inline]
            pub fn as_ptr(&self) -> *const f32 {
                (& first_expr!($(self.$field),+)) as *const f32
            }

            #[inline]
            pub fn as_mut_ptr(&mut self) -> *mut f32 {
                (&mut first_expr!($(self.$field),+)) as *mut f32
            }

            #[inline] pub fn as_array(&self) -> &[f32; $length] { self.as_ref() }
            #[inline] pub fn as_mut_array(&mut self) -> &mut [f32; $length] { self.as_mut() }

            #[inline] pub fn as_mut_tuple(&mut self) -> &mut $tuple_ty { self.as_mut() }
            #[inline] pub fn as_tuple(&self) -> &$tuple_ty { self.as_ref() }

            #[inline] pub fn tup(self) -> $tuple_ty { self.into() }

            #[inline] pub fn len(&self) -> usize { $length }

            #[inline]
            pub fn iter<'a>(&'a self) -> slice::Iter<'a, f32> {
                self.as_slice().iter()
            }

            #[inline]
            pub fn iter_mut<'a>(&'a mut self) -> slice::IterMut<'a, f32> {
                self.as_mut_slice().iter_mut()
            }

            #[inline] pub fn into_iter(self) -> VecIter<Self> { VecIter::new(self) }
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
                self.map3(min, max, |v, l, h| clamp(v, l, h))
            }

            #[inline]
            pub fn lerp(self, b: Self, t: f32) -> Self {
                self.map2(b, |x, y| x.lerp(y, t))
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
            pub fn dot(self, o: Self) -> f32 {
                self.map2(o, |x, y| x * y).fold(|a, b| a + b)
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
                assert_lt!(axis, $length, "Invalid axis");
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

            #[inline]
            pub fn zero() -> Self {
                $Vn { $($field: 0.0),+ }
            }

            #[inline]
            pub fn identity() -> Self {
                $Vn { $($field: 0.0),+ }
            }
        }

        impl iter::IntoIterator for $Vn {
            type Item = f32;
            type IntoIter = VecIter<$Vn>;
            #[inline]
            fn into_iter(self) -> VecIter<Self> {
                VecIter::new(self)
            }
        }

        impl ApproxEq for $Vn {

            #[inline]
            fn approx_zero_e(&self, e: f32) -> bool {
                self.fold_init(true, |cnd, val| cnd && val.approx_zero_e(e))
            }

            #[inline]
            fn approx_eq_e(&self, o: &Self, e: f32) -> bool {
                self.fold2_init(*o, true, |cnd, l, r| cnd && l.approx_eq_e(&r, e))
            }
        }

        impl Map for $Vn {
            #[inline]
            fn map3<F: Fn(f32, f32, f32) -> f32>(self, a: Self, b: Self, f: F) -> Self {
                Self { $($field: f(self.$field, a.$field, b.$field)),+ }
            }
        }

        impl VecType for $Vn {
            const SIZE: usize = $length;

            #[inline]
            fn splat(v: f32) -> Self {
                Self { $($field: v),+ }
            }
        }
    }
}

do_vec_boilerplate!(V2 { x: 0, y: 1 }, 2, (f32, f32));

do_vec_boilerplate!(V3 { x: 0, y: 1, z: 2 }, 3, (f32, f32, f32));

do_vec_boilerplate!(
    V4 {
        x: 0,
        y: 1,
        z: 2,
        w: 3
    },
    4,
    (f32, f32, f32, f32)
);

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
    pub fn max_index(&self) -> usize {
        if self.x > self.y {
            0
        } else {
            1
        }
    }

    #[inline]
    pub fn min_index(&self) -> usize {
        if self.x > self.y {
            1
        } else {
            0
        }
    }
}

impl V3 {
    #[inline]
    pub fn expand(v: V2, z: f32) -> V3 {
        V3 { x: v.x, y: v.y, z }
    }

    #[inline]
    pub fn outer_prod(self, o: V3) -> M3x3 {
        M3x3 {
            x: self * o.x,
            y: self * o.y,
            z: self * o.z,
        }
    }

    #[inline]
    pub fn cross(&self, b: V3) -> V3 {
        vec3(
            self.y * b.z - self.z * b.y,
            self.z * b.x - self.x * b.z,
            self.x * b.y - self.y * b.x,
        )
    }

    #[inline]
    pub fn orth(self) -> V3 {
        let abs_v = self.abs();
        let mut u = V3::splat(1.0);
        u[abs_v.max_index()] = 0.0;
        // let u = u;
        u.cross(self).norm_or(1.0, 0.0, 0.0)
    }

    #[inline]
    pub fn basis(self) -> (V3, V3, V3) {
        let a = self.norm_or(1.0, 0.0, 0.0);
        let bu = if self.x.abs() > 0.57735 {
            // sqrt(1/3)
            vec3(self.y, -self.x, 0.0)
        } else {
            vec3(0.0, self.z, -self.y)
        };
        // should never need normalizing, but there may be degenerate cases...
        let b = bu.norm_or(0.0, -1.0, 0.0);
        let c = a.cross(b);
        (a, b, c)
    }

    #[inline]
    pub fn norm_or_unit(self) -> V3 {
        self.norm_or(0.0, 0.0, 1.0)
    }
    #[inline]
    pub fn to_arr(self) -> [f32; 3] {
        self.into()
    }

    #[inline]
    pub fn max_index(&self) -> usize {
        if self.x > self.y {
            if self.x > self.z {
                0
            } else {
                2
            }
        } else {
            if self.y > self.z {
                1
            } else {
                2
            }
        }
    }

    #[inline]
    pub fn min_index(&self) -> usize {
        if self.x < self.y {
            if self.x < self.z {
                0
            } else {
                2
            }
        } else {
            if self.y < self.z {
                1
            } else {
                2
            }
        }
    }
}

impl V4 {
    #[inline]
    pub fn expand(v: V3, w: f32) -> V4 {
        V4 {
            x: v.x,
            y: v.y,
            z: v.z,
            w: w,
        }
    }
    #[inline]
    pub fn xyz(self) -> V3 {
        V3 {
            x: self.x,
            y: self.y,
            z: self.z,
        }
    }
    #[inline]
    pub fn norm_or_unit(self) -> V4 {
        self.norm_or(0.0, 0.0, 0.0, 1.0)
    }
    #[inline]
    pub fn to_arr(self) -> [f32; 4] {
        self.into()
    }
    #[inline]
    pub fn outer_prod(self, o: V4) -> M4x4 {
        M4x4 {
            x: self * o.x,
            y: self * o.y,
            z: self * o.z,
            w: self * o.w,
        }
    }
    #[inline]
    pub fn to_arr8(self) -> [u8; 4] {
        [
            (self.x * 255.0).trunc() as u8,
            (self.y * 255.0).trunc() as u8,
            (self.z * 255.0).trunc() as u8,
            (self.w * 255.0).trunc() as u8,
        ]
    }

    #[inline]
    pub fn max_index(&self) -> usize {
        if self.x > self.y {
            // y out
            if self.x > self.z {
                if self.x > self.w {
                    0
                } else {
                    3
                }
            }
            // z out
            else {
                if self.z > self.w {
                    2
                } else {
                    3
                }
            } // x out
        } else {
            // x out
            if self.y > self.z {
                if self.y > self.w {
                    1
                } else {
                    3
                }
            }
            // z out
            else {
                if self.z > self.w {
                    2
                } else {
                    3
                }
            } // y out
        }
    }

    #[inline]
    pub fn min_index(&self) -> usize {
        if self.x < self.y {
            // y out
            if self.x < self.z {
                if self.x < self.w {
                    0
                } else {
                    3
                }
            }
            // z out
            else {
                if self.z < self.w {
                    2
                } else {
                    3
                }
            } // x out
        } else {
            // x out
            if self.y < self.z {
                if self.y < self.w {
                    1
                } else {
                    3
                }
            }
            // z out
            else {
                if self.z < self.w {
                    2
                } else {
                    3
                }
            } // y out
        }
    }
}

impl From<V3> for V2 {
    #[inline]
    fn from(v: V3) -> V2 {
        V2 { x: v.x, y: v.y }
    }
}

impl From<V2> for V3 {
    #[inline]
    fn from(v: V2) -> V3 {
        V3 {
            x: v.x,
            y: v.y,
            z: 0.0,
        }
    }
}

impl From<V4> for V3 {
    #[inline]
    fn from(v: V4) -> V3 {
        V3 {
            x: v.x,
            y: v.y,
            z: v.z,
        }
    }
}

impl From<V3> for V4 {
    #[inline]
    fn from(v: V3) -> V4 {
        V4 {
            x: v.x,
            y: v.y,
            z: v.z,
            w: 0.0,
        }
    }
}

impl From<V4> for V2 {
    #[inline]
    fn from(v: V4) -> V2 {
        V2 { x: v.x, y: v.y }
    }
}

impl From<V2> for V4 {
    #[inline]
    fn from(v: V2) -> V4 {
        V4 {
            x: v.x,
            y: v.y,
            z: 0.0,
            w: 0.0,
        }
    }
}

impl AsRef<V2> for V3 {
    #[inline]
    fn as_ref(&self) -> &V2 {
        unsafe { mem::transmute(self) }
    }
}

impl AsRef<V2> for V4 {
    #[inline]
    fn as_ref(&self) -> &V2 {
        unsafe { mem::transmute(self) }
    }
}

impl AsRef<V3> for V4 {
    #[inline]
    fn as_ref(&self) -> &V3 {
        unsafe { mem::transmute(self) }
    }
}

impl AsMut<V2> for V3 {
    #[inline]
    fn as_mut(&mut self) -> &mut V2 {
        unsafe { mem::transmute(self) }
    }
}

impl AsMut<V2> for V4 {
    #[inline]
    fn as_mut(&mut self) -> &mut V2 {
        unsafe { mem::transmute(self) }
    }
}

impl AsMut<V3> for V4 {
    #[inline]
    fn as_mut(&mut self) -> &mut V3 {
        unsafe { mem::transmute(self) }
    }
}

#[inline]
pub fn cross(a: V3, b: V3) -> V3 {
    a.cross(b)
}

#[inline]
pub fn scalar_triple(a: V3, b: V3, c: V3) -> f32 {
    a.cross(b).dot(c)
}

#[inline]
pub fn max_dir(arr: &[V3], dir: V3) -> Option<V3> {
    match max_dir_index(arr, dir) {
        Some(index) => Some(arr[index]),
        None => None,
    }
}

#[inline]
pub fn max_dir_index(arr: &[V3], dir: V3) -> Option<usize> {
    if arr.len() == 0 {
        return None;
    }
    let mut best_idx = 0;
    for (idx, item) in arr.iter().enumerate() {
        if dir.dot(*item) > dir.dot(arr[best_idx]) {
            best_idx = idx;
        }
    }
    Some(best_idx)
}

#[inline]
pub fn max_dir_i<I: Iterator<Item = V3>>(dir: V3, iter: &mut I) -> Option<V3> {
    let initial = iter.next()?;

    let mut best = initial;
    for item in iter {
        if dir.dot(item) > dir.dot(best) {
            best = item;
        }
    }
    Some(best)
}

#[inline]
pub fn compute_bounds_i<I, Vt>(iter: &mut I) -> Option<(Vt, Vt)>
where
    I: Iterator<Item = Vt>,
    Vt: VecType,
{
    let initial = iter.next()?;

    let mut min_bound = initial;
    let mut max_bound = initial;
    for item in iter {
        min_bound = min_bound.min(item);
        max_bound = max_bound.max(item);
    }
    Some((min_bound, max_bound))
}

#[inline]
pub fn compute_bounds<Vt: VecType>(arr: &[Vt]) -> Option<(Vt, Vt)> {
    compute_bounds_i(&mut arr.iter().map(|v| *v))
}

#[inline]
pub fn same_dir<T: VecType>(a: T, b: T) -> bool {
    a.dot(b) > 0.0
}

#[inline]
pub fn same_dir_e<T: VecType>(a: T, b: T, epsilon: f32) -> bool {
    a.dot(b) > epsilon
}

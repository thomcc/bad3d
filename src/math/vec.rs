use std::ops::*;
use std::{mem, fmt};

use util::{min_index, max_index};
use math::scalar::round_to;
use math::mat::*;
use math::quat::*;
use math::traits::*;

#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[repr(C)]
pub struct V2 {
    pub x: f32,
    pub y: f32,
}

#[inline]
pub fn vec2(x: f32, y: f32) -> V2 {
    V2 { x: x, y: y }
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
    V3 { x: x, y: y, z: z }
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
    V4 { x: x, y: y, z: z, w: w }
}

impl fmt::Display for V2 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "V2({}, {})", self.x, self.y) }
}

impl fmt::Display for V3 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "V3({}, {}, {})", self.x, self.y, self.z) }
}

impl fmt::Display for V4 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "V4({}, {}, {}, {})", self.x, self.y, self.z, self.w) }
}

pub trait VecType
    : Copy
    + Clone

    + Fold
    + Map
    + ApproxEq

    + Identity

    + Add<Output = Self>
    + Sub<Output = Self>

    + AddAssign
    + SubAssign

    + Mul<f32, Output = Self>
    + Div<f32, Output = Self>

    + MulAssign<f32>
    + DivAssign<f32>

    + Neg<Output = Self>

    + Index<usize, Output = f32>
    + IndexMut<usize>
    + Mul<Output = Self>
    + Div<Output = Self>
    + MulAssign
    + DivAssign
{
    fn count() -> usize;
    fn splat(v: f32) -> Self;

    #[inline] fn zero() -> Self { Self::splat(0.0) }

    #[inline] fn max_elem(self) -> f32 { self.fold(|a, b| a.max(b)) }
    #[inline] fn min_elem(self) -> f32 { self.fold(|a, b| a.min(b)) }

    #[inline] fn abs(self) -> Self { self.map(|x| x.abs()) }

    #[inline] fn floor(self) -> Self { self.map(|x| x.floor()) }
    #[inline] fn ceil(self) -> Self { self.map(|x| x.ceil()) }
    #[inline] fn round(self) -> Self { self.map(|x| x.round()) }

    #[inline] fn min(self, o: Self) -> Self { self.map2(o, |a, b| a.min(b)) }
    #[inline] fn max(self, o: Self) -> Self { self.map2(o, |a, b| a.max(b)) }

    #[inline] fn length_sq(self) -> f32 { self.dot(self) }
    #[inline] fn length(self) -> f32 { self.length_sq().sqrt() }

    #[inline] fn towards(self, o: Self) -> Self { o - self }
    #[inline] fn dir_towards(self, o: Self) -> Self { self.towards(o).norm_or_zero() }

    #[inline] fn dist_sq(self, o: Self) -> f32 { (o - self).length_sq() }
    #[inline] fn dist(self, o: Self) -> f32 { (o - self).length() }

    #[inline] fn same_dir(self, o: Self) -> bool { self.dot(o) > 0.0 }

    #[inline]
    fn clamp_length(self, max_len: f32) -> Self {
        let len = self.length();
        if len < max_len { self }
        else { self * (max_len / len) }
    }

    #[inline]
    fn norm_len(self) -> (Option<Self>, f32) {
        let l = self.length();
        if l == 0.0 {
            (None, l)
        } else {
            let il = 1.0 / l;
            (Some(self * il), l)
        }
    }

    #[inline] fn normalize(self) -> Option<Self> { self.norm_len().0 }
    #[inline] fn norm_or_zero(self) -> Self { self.normalize().unwrap_or(Self::zero()) }
    #[inline] fn norm_or_v(self, v: Self) -> Self { self.normalize().unwrap_or(v) }
    #[inline] fn must_norm(self) -> Self { self.normalize().unwrap() }

    #[inline] fn is_normalized(self) -> bool { self.length().approx_eq(&1.0) }
    #[inline] fn is_normalized_e(self, epsilon: f32) -> bool { self.length().approx_eq_e(&1.0, epsilon) }

    #[inline] fn is_zero(self) -> bool { self.dot(self) == 0.0_f32 }
    #[inline] fn hadamard(self, o: Self) -> Self { self.map2(o, |a, b| a * b) }

    #[inline] fn round_to(self, p: f32) -> Self { self.map(|v| round_to(v, p)) }

}


impl<T: Fold> ApproxEq for T {
    #[inline]
    fn approx_zero_e(&self, e: f32) -> bool {
        self.fold_init(true, |cnd, val| cnd && val.approx_zero_e(e))
    }

    #[inline]
    fn approx_eq_e(&self, o: &Self, e: f32) -> bool {
        self.fold2_init(*o, true, |cnd, l, r| cnd && l.approx_eq_e(&r, e))
    }
}


impl<T: Map> Clamp for T {
    #[inline]
    fn clamp(self, min: Self, max: Self) -> Self {
        self.map3(min, max, |v, l, h| v.clamp(l, h))
    }
}

impl<T: Map> Lerp for T {
    #[inline]
    fn lerp(self, b: T, t: f32) -> T {
        self.map2(b, |x, y| x.lerp(y, t))
    }
}

impl Fold for V2 {
    #[inline]
    fn fold<F: Fn(f32, f32) -> f32>(self, f: F) -> f32 {
        f(self.x, self.y)
    }

    #[inline]
    fn fold2_init<T, F>(self, o: Self, init: T, f: F) -> T
            where F: Fn(T, f32, f32) -> T {
        f(f(init, o.x, self.x), o.y, self.y)
    }
}

impl Fold for V3 {
    #[inline]
    fn fold<F: Fn(f32, f32) -> f32>(self, f: F) -> f32 {
        f(f(self.x, self.y), self.z)
    }

    #[inline]
    fn fold2_init<T, F>(self, o: Self, init: T, f: F) -> T
            where F: Fn(T, f32, f32) -> T {
        f(f(f(init, o.x, self.x), o.y, self.y), o.z, self.z)
    }
}

impl Fold for V4 {
    #[inline]
    fn fold<F: Fn(f32, f32) -> f32>(self, f: F) -> f32 {
        f(f(f(self.x, self.y), self.z), self.w)
    }

    #[inline]
    fn fold2_init<T, F>(self, o: Self, init: T, f: F) -> T
            where F: Fn(T, f32, f32) -> T {
        f(f(f(f(init, o.x, self.x), o.y, self.y), o.z, self.z), o.w, self.w)
    }
}

// @@ partially duplicated in mat.rs...
macro_rules! define_unsafe_conversions {
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
    }
}

macro_rules! do_vec_boilerplate {
    ($Vn: ident { $($field: ident : $index: expr),+ }, $length: expr, $tuple_ty: ty) => {
        define_unsafe_conversions!($Vn, [f32; $length]);
        define_unsafe_conversions!($Vn, $tuple_ty);

        impl From<$Vn> for [f32; $length] {
            #[inline]
            fn from(v: $Vn) -> [f32; $length] {
                [$(v.$field),+]
            }
        }

        impl From<$Vn> for $tuple_ty {
            #[inline]
            fn from(v: $Vn) -> $tuple_ty {
                ($(v.$field),+)
            }
        }

        impl From<[f32; $length]> for $Vn {
            #[inline]
            fn from(v: [f32; $length]) -> $Vn {
                $Vn { $($field: v[$index]),+ }
            }
        }

        impl From<$tuple_ty> for $Vn {
            #[inline]
            fn from(v: $tuple_ty) -> $Vn {
                let ($($field),+) = v;
                $Vn { $($field: $field),+ }
            }
        }

        impl Index<usize> for $Vn {
            type Output = f32;
            #[inline]
            fn index(&self, i: usize) -> &f32 {
                let v: &[f32; $length] = self.as_ref();
                &v[i]
            }
        }

        impl IndexMut<usize> for $Vn {
            #[inline]
            fn index_mut(&mut self, i: usize) -> &mut f32 {
                let v: &mut [f32; $length] = self.as_mut();
                &mut v[i]
            }
        }

        impl Neg for $Vn {
            type Output = $Vn;
            #[inline]
            fn neg(self) -> $Vn {
                $Vn{ $($field: -self.$field),+ }
            }
        }

        impl Add for $Vn {
            type Output = $Vn;
            #[inline]
            fn add(self, o: $Vn) -> $Vn {
                $Vn{ $($field: (self.$field + o.$field)),+ }
            }
        }

        impl Sub for $Vn {
            type Output = $Vn;
            #[inline]
            fn sub(self, o: $Vn) -> $Vn {
                $Vn{ $($field: (self.$field - o.$field)),+ }
            }
        }

        impl Mul for $Vn {
            type Output = $Vn;
            #[inline]
            fn mul(self, o: $Vn) -> $Vn {
                $Vn{ $($field: (self.$field * o.$field)),+ }
            }
        }

        impl Div for $Vn {
            type Output = $Vn;
            #[inline]
            fn div(self, o: $Vn) -> $Vn {
                debug_assert!($(o.$field != 0.0) && +);
                $Vn{ $($field: (self.$field / o.$field)),+ }
            }
        }

        impl Mul<f32> for $Vn {
            type Output = $Vn;
            #[inline]
            fn mul(self, o: f32) -> $Vn {
                $Vn{ $($field: (self.$field * o)),+ }
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
            type Output = $Vn;
            #[inline]
            fn div(self, o: f32) -> $Vn {
                debug_assert_ne!(o, 0.0);
                let inv = 1.0 / o;
                $Vn{ $($field: (self.$field * inv)),+ }
            }
        }

        impl MulAssign for $Vn {
            #[inline]
            fn mul_assign(&mut self, rhs: $Vn) {
                $(self.$field *= rhs.$field;)+
            }
        }

        impl DivAssign for $Vn {
            #[inline]
            fn div_assign(&mut self, rhs: $Vn) {
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
            fn add_assign(&mut self, rhs: $Vn) {
                $(self.$field += rhs.$field;)+
            }
        }

        impl SubAssign for $Vn {
            #[inline]
            fn sub_assign(&mut self, rhs: $Vn) {
                $(self.$field -= rhs.$field;)+
            }
        }

        impl $Vn {
            #[inline] pub fn new($($field: f32),+) -> $Vn { $Vn{ $($field: $field),+ } }

            #[inline]
            pub fn angle(self, o: $Vn) -> f32 {
                let d = self.dot(o);
                if d > 1.0 { 0.0 }
                else { d.max(-1.0).acos() }
            }

            #[inline]
            pub fn slerp(self, o: $Vn, t: f32) -> $Vn {
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

            #[inline] pub fn nlerp(self, o: $Vn, t: f32) -> $Vn { self.lerp(o, t).norm_or_v(o) }

            #[inline]
            pub fn norm_or(self, $($field: f32),+) -> $Vn {
                self.norm_or_v($Vn { $($field: $field),+ })
            }
        }

        impl Identity for $Vn {
            #[inline]
            fn identity() -> Self { $Vn::zero() }
        }

        impl Map for $Vn {
            #[inline]
            fn map3<F: Fn(f32, f32, f32) -> f32>(self, a: Self, b: Self, f: F) -> Self {
                $Vn{ $($field: f(self.$field, a.$field, b.$field)),+ }
            }
        }

        impl VecType for $Vn {
            #[inline] fn count() -> usize { $length }
            #[inline] fn splat(v: f32) -> $Vn { $Vn{ $($field: v),+ } }
        }
    }
}

do_vec_boilerplate!(V2 { x: 0, y: 1             }, 2, (f32, f32));
do_vec_boilerplate!(V3 { x: 0, y: 1, z: 2       }, 3, (f32, f32, f32));
do_vec_boilerplate!(V4 { x: 0, y: 1, z: 2, w: 3 }, 4, (f32, f32, f32, f32));

impl V2 {
    #[inline] pub fn unit() -> V2 { V2::new(1.0, 0.0) }
    #[inline] pub fn min_index(&self) -> usize { if self.x < self.y { 0 } else { 1 } }
    #[inline] pub fn max_index(&self) -> usize { if self.x > self.y { 0 } else { 1 } }
    #[inline] pub fn outer_prod(self, o: V2) -> M2x2 { M2x2{x: self*o.x, y: self*o.y } }
    #[inline] pub fn cross(self, o: V2) -> f32 { self.x*o.y - self.y*o.x }
    #[inline] pub fn norm_or_unit(self) -> V2 { self.norm_or(0.0, 1.0) }
}

impl V3 {
    #[inline]
    pub fn expand(v: V2, z: f32) -> V3 {
        V3 { x: v.x, y: v.y, z: z }
    }
    #[inline]
    pub fn outer_prod(self, o: V3) -> M3x3 {
        M3x3 { x: self*o.x, y: self*o.y, z: self*o.z }
    }

    #[inline]
    pub fn max_index(&self) -> usize {
        if self.x > self.y { if self.x > self.z { 0 } else { 2 } }
        else               { if self.y > self.z { 1 } else { 2 } }
    }

    #[inline]
    pub fn min_index(&self) -> usize {
        if self.x < self.y { if self.x < self.z { 0 } else { 2 } }
        else               { if self.y < self.z { 1 } else { 2 } }
    }

    #[inline]
    pub fn cross(&self, b: V3) -> V3 {
        vec3(self.y*b.z - self.z*b.y,
             self.z*b.x - self.x*b.z,
             self.x*b.y - self.y*b.x)
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
        let bu =
            if self.x.abs() > 0.57735 { // sqrt(1/3)
                vec3(self.y, -self.x, 0.0)
            } else {
                vec3(0.0, self.z, -self.y)
            };
        // should never need normalizing, but there may be degenerate cases...
        let b = bu.norm_or(0.0, -1.0, 0.0);
        let c = a.cross(b);
        (a, b, c)
    }
    #[inline] pub fn norm_or_unit(self) -> V3 { self.norm_or(0.0, 0.0, 1.0) }
}

impl V4 {
    #[inline] pub fn expand(v: V3, w: f32) -> V4 { V4{ x: v.x, y: v.y, z: v.z, w: w } }
    #[inline] pub fn min_index(&self) -> usize { min_index(&<[f32; 4]>::from(*self)) }
    #[inline] pub fn max_index(&self) -> usize { max_index(&<[f32; 4]>::from(*self)) }
    #[inline] pub fn outer_prod(self, o: V4) -> M4x4 { M4x4{ x: self*o.x, y: self*o.y, z: self*o.z, w: self*o.w } }
    #[inline] pub fn xyz(self) -> V3 { V3{ x: self.x, y: self.y, z: self.z } }
    #[inline] pub fn norm_or_unit(self) -> V4 { self.norm_or(0.0, 0.0, 0.0, 1.0) }
}

impl From<V3> for V2 { #[inline] fn from(v: V3) -> V2 { V2 { x: v.x, y: v.y                 } } }
impl From<V2> for V3 { #[inline] fn from(v: V2) -> V3 { V3 { x: v.x, y: v.y, z: 0.0         } } }
impl From<V4> for V3 { #[inline] fn from(v: V4) -> V3 { V3 { x: v.x, y: v.y, z: v.z         } } }
impl From<V3> for V4 { #[inline] fn from(v: V3) -> V4 { V4 { x: v.x, y: v.y, z: v.z, w: 0.0 } } }
impl From<V4> for V2 { #[inline] fn from(v: V4) -> V2 { V2 { x: v.x, y: v.y                 } } }
impl From<V2> for V4 { #[inline] fn from(v: V2) -> V4 { V4 { x: v.x, y: v.y, z: 0.0, w: 0.0 } } }

impl AsRef<V2> for V3 { #[inline] fn as_ref(&self) -> &V2 { unsafe { mem::transmute(self) } } }
impl AsRef<V2> for V4 { #[inline] fn as_ref(&self) -> &V2 { unsafe { mem::transmute(self) } } }
impl AsRef<V3> for V4 { #[inline] fn as_ref(&self) -> &V3 { unsafe { mem::transmute(self) } } }

impl AsMut<V2> for V3 { #[inline] fn as_mut(&mut self) -> &mut V2 { unsafe { mem::transmute(self) } } }
impl AsMut<V2> for V4 { #[inline] fn as_mut(&mut self) -> &mut V2 { unsafe { mem::transmute(self) } } }
impl AsMut<V3> for V4 { #[inline] fn as_mut(&mut self) -> &mut V3 { unsafe { mem::transmute(self) } } }


impl From<Quat> for V4 { #[inline] fn from(q: Quat) -> V4 { q.0 } }

impl AsRef<Quat> for V4 { #[inline] fn as_ref(&    self) -> &    Quat { unsafe { mem::transmute(self) } } }
impl AsMut<Quat> for V4 { #[inline] fn as_mut(&mut self) -> &mut Quat { unsafe { mem::transmute(self) } } }

#[inline]
pub fn cross(a: V3, b: V3) -> V3 {
    a.cross(b)
}

#[inline]
pub fn clamp_s<T: VecType>(a: T, min: f32, max: f32) -> T {
    a.clamp(T::splat(min), T::splat(max))
}

pub fn max_dir(arr: &[V3], dir: V3) -> Option<V3> {
    match max_dir_index(arr, dir) {
        Some(index) => Some(arr[index]),
        None => None
    }
}

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

pub fn max_dir_i<I: Iterator<Item = V3>>(dir: V3, iter: &mut I) -> Option<V3> {
    let initial = try_opt!(iter.next());

    let mut best = initial;
    for item in iter {
        if dir.dot(item) > dir.dot(best) {
            best = item;
        }
    }
    Some(best)
}


pub fn compute_bounds_i<I, Vt>(iter: &mut I) -> Option<(Vt, Vt)>
        where I: Iterator<Item = Vt>, Vt: VecType {
    let initial = try_opt!(iter.next());

    let mut min_bound = initial;
    let mut max_bound = initial;
    for item in iter {
        min_bound = min_bound.min(item);
        max_bound = max_bound.max(item);
    }
    Some((min_bound, max_bound))
}

pub fn compute_bounds<Vt: VecType>(arr: &[Vt]) -> Option<(Vt, Vt)> {
    compute_bounds_i(&mut arr.iter().map(|v| *v))
}

#[inline]
pub fn same_dir<T: VecType>(a: T, b: T) -> bool {
    a.same_dir(b)
}

#[inline]
pub fn same_dir_e<T: VecType>(a: T, b: T, epsilon: f32) -> bool {
    a.dot(b) > epsilon
}

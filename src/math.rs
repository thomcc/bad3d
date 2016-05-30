use std::mem;
use std::ops::*;
use std::fmt::Debug;
use std::default::Default;

// should split this up...

pub const EPSILON: f32 = 1e-6f32;

#[inline]
pub fn near_zero(n: f32) -> bool {
    n.abs() < EPSILON
}

#[inline]
pub fn near_eq(a: f32, b: f32) -> bool {
    let scaled_eps = a.max(b).max(1.0) * EPSILON;
    (a - b).abs() < scaled_eps
}

#[inline]
pub fn safe_div(a: f32, b: f32, fallback: f32) -> f32 {
    if near_zero(b) { fallback }
    else { a / b }
}

pub trait Lerp {
    fn lerp(self, Self, f32) -> Self;
}

pub trait Clamp {
    fn clamp(self, min: Self, max: Self) -> Self;
}

impl Lerp for f32 {
    #[inline]
    fn lerp(self, o: f32, t: f32) -> f32 {
        self * (1.0-t) + o * t
    }
}

impl Clamp for f32 {
    #[inline]
    fn clamp(self, min: f32, max: f32) -> f32 {
        self.min(max).max(min)
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct V2 {
    pub x: f32,
    pub y: f32,
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct V3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct V4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct M2x2 {
    pub x: V2,
    pub y: V2,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct M3x3 {
    pub x: V3,
    pub y: V3,
    pub z: V3,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct M4x4 {
    pub x: V4,
    pub y: V4,
    pub z: V4,
    pub w: V4,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Quat(pub V4);

impl Default for M2x2 {
    #[inline]
    fn default() -> M2x2 {
        M2x2{ x: V2 { x: 1.0, y: 0.0 },
              y: V2 { x: 0.0, y: 1.0 } }
    }
}

impl Default for M3x3 {
    #[inline]
    fn default() -> M3x3 {
        M3x3{ x: V3 { x: 1.0, y: 0.0, z: 0.0 },
              y: V3 { x: 0.0, y: 1.0, z: 0.0 },
              z: V3 { x: 0.0, y: 0.0, z: 1.0 } }
    }
}

impl Default for M4x4 {
    #[inline]
    fn default() -> M4x4 {
        M4x4{ x: V4 { x: 1.0, y: 0.0, z: 0.0, w: 0.0 },
              y: V4 { x: 0.0, y: 1.0, z: 0.0, w: 0.0 },
              z: V4 { x: 0.0, y: 0.0, z: 1.0, w: 0.0 },
              w: V4 { x: 0.0, y: 0.0, z: 0.0, w: 1.0 } }
    }
}


pub trait Foldable<T> {
    fn fold<F: Fn(T, T) -> T>(self, f: F) -> T;
}

pub trait VecType
    : Copy
    + Clone
    + Debug

    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>

    + Mul<f32, Output = Self>
    + Div<f32, Output = Self>

    + Neg<Output = Self>
    + Index<usize, Output = f32>
    + IndexMut<usize>
    + Foldable<f32>
{
    fn size() -> usize;
    fn map<F: Fn(f32) -> f32>(self, f: F) -> Self;
    fn zip<F: Fn(f32, f32) -> f32>(self, o: Self, f: F) -> Self;

    fn zero() -> Self;
    fn splat(v: f32) -> Self;

    #[inline] fn max_elem(self) -> f32 { self.fold(|a, b| a.max(b)) }
    #[inline] fn min_elem(self) -> f32 { self.fold(|a, b| a.min(b)) }

    #[inline] fn abs(self) -> Self { self.map(|x| x.abs()) }

    #[inline] fn floor(self) -> Self { self.map(|x| x.floor()) }
    #[inline] fn ceil(self) -> Self { self.map(|x| x.ceil()) }
    #[inline] fn round(self) -> Self { self.map(|x| x.round()) }

    #[inline] fn min(self, o: Self) -> Self { self.zip(o, |a, b| a.min(b)) }
    #[inline] fn max(self, o: Self) -> Self { self.zip(o, |a, b| a.max(b)) }
    #[inline] fn dot(self, o: Self) -> f32 { (self * o).fold(|a, b| a+b) }
    #[inline] fn length_sq(self) -> f32 { self.dot(self) }
    #[inline] fn length(self) -> f32 { self.length_sq().sqrt() }

    #[inline]
    fn normalize_len(self) -> (Option<Self>, f32) {
        let l = self.length();
        if near_zero(l) {
            (None, l)
        } else {
            (Some(self / l), l)
        }
    }

    #[inline]
    fn normalize(self) -> Option<Self> {
        self.normalize_len().0
    }

    #[inline]
    fn normalize_or_zero(self) -> Self {
        self.normalize().unwrap_or(Self::zero())
    }
}

impl Foldable<f32> for V2 {
    #[inline] fn fold<F: Fn(f32, f32) -> f32>(self, f: F) -> f32 { f(self.x, self.y) }
}

impl Foldable<f32> for V3 {
    #[inline] fn fold<F: Fn(f32, f32) -> f32>(self, f: F) -> f32 { f(f(self.x, self.y), self.z) }
}

impl Foldable<f32> for V4 {
    #[inline] fn fold<F: Fn(f32, f32) -> f32>(self, f: F) -> f32 { f(f(f(self.x, self.y), self.z), self.w) }
}

macro_rules! do_vec_boilerplate {
    ($Vn: ident { $($field: ident : $index: expr),+ }, $length: expr, $tupleTy: ty) => {
        impl From<$Vn> for [f32; $length] {
            #[inline]
            fn from(v: $Vn) -> [f32; $length] {
                [$(v.$field),+]
            }
        }

        impl AsRef<[f32; $length]> for $Vn {
            #[inline]
            fn as_ref(&self) -> &[f32; $length] {
                unsafe { mem::transmute(self) }
            }
        }

        impl AsMut<[f32; $length]> for $Vn {
            #[inline]
            fn as_mut(&mut self) -> &mut [f32; $length] {
                unsafe { mem::transmute(self) }
            }
        }

        impl From<[f32; $length]> for $Vn {
            #[inline]
            fn from(v: [f32; $length]) -> $Vn {
                $Vn { $($field: v[$index].clone()),+ }
            }
        }

        impl<'a> From<&'a [f32; $length]> for &'a $Vn {
            #[inline]
            fn from(v: &'a [f32; $length]) -> &'a $Vn {
                unsafe { mem::transmute(v) }
            }
        }

        impl<'a> From<&'a mut [f32; $length]> for &'a mut $Vn {
            #[inline]
            fn from(v: &'a mut [f32; $length]) -> &'a mut $Vn {
                unsafe { mem::transmute(v) }
            }
        }

        impl Index<usize> for $Vn {
            type Output = f32;
            #[inline]
            fn index<'a>(&'a self, i: usize) -> &'a f32 {
                let v: &[f32; $length] = self.as_ref();
                &v[i]
            }
        }

        impl IndexMut<usize> for $Vn {
            #[inline]
            fn index_mut<'a>(&'a mut self, i: usize) -> &'a mut f32 {
                let v: &mut [f32; $length] = self.as_mut();
                &mut v[i]
            }
        }

        impl From<$tupleTy> for $Vn {
            #[inline]
            fn from(v: $tupleTy) -> $Vn {
                let ($($field),+) = v;
                $Vn{$($field: $field),+}
            }
        }

        impl From<$Vn> for $tupleTy {
            #[inline]
            fn from(v: $Vn) -> $tupleTy {
                ($(v.$field),+)
            }
        }

        impl Lerp for $Vn {
            #[inline]
            fn lerp(self, o: $Vn, t: f32) -> $Vn {
                self.zip(o, |a, b| a.lerp(b, t))
            }
        }

        impl Clamp for $Vn {
            #[inline]
            fn clamp(self, min: $Vn, max: $Vn) -> $Vn {
                $Vn{ $($field: self.$field.clamp(min.$field, max.$field)),+ }
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
                if near_zero(th) {
                    self
                } else {
                    let isth = 1.0 / th.sin();
                    let s0 = isth * (th * (1.0 - t)).sin();
                    let s1 = isth * (th * t).sin();
                    self * s0 + o * s1
                }
            }

            #[inline] pub fn nlerp(self, o: $Vn, t: f32) -> $Vn { self.lerp(o, t).normalize_or_zero() }
            #[inline] pub fn distance_sq(self, o: $Vn) -> f32 { (o - self).length_sq() }
            #[inline] pub fn distance(self, o: $Vn) -> f32 { (o - self).length() }
            #[inline] pub fn toward(self, o: $Vn) -> $Vn { o - self }

            #[inline] pub fn near_eq(self, o: $Vn) -> bool { $(near_eq(self.$field, o.$field)) && + }

        }

        impl VecType for $Vn {
            #[inline]
            fn size() -> usize {
                $length
            }

            #[inline]
            fn map<F: Fn(f32) -> f32>(self, f: F) -> $Vn {
                $Vn{ $($field: f(self.$field)),+ }
            }

            #[inline]
            fn zip<F: Fn(f32, f32) -> f32>(self, o: $Vn, f: F) -> $Vn {
                $Vn{ $($field: f(o.$field, self.$field)),+ }
            }

            #[inline] fn zero() -> $Vn { $Vn{ $($field: 0.0),+ } }
            #[inline] fn splat(v: f32) -> $Vn { $Vn{ $($field: v),+ } }

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
                let inv = 1.0 / o;
                $Vn{ $($field: (self.$field * inv)),+ }
            }
        }

        impl Neg for $Vn {
            type Output = $Vn;
            #[inline]
            fn neg(self) -> $Vn {
                $Vn{ $($field: -self.$field),+ }
            }
        }
    }
}

do_vec_boilerplate!(V2 {x: 0, y: 1            }, 2, (f32, f32));
do_vec_boilerplate!(V3 {x: 0, y: 1, z: 2      }, 3, (f32, f32, f32));
do_vec_boilerplate!(V4 {x: 0, y: 1, z: 2, w: 3}, 4, (f32, f32, f32, f32));

pub fn min_index<T: PartialOrd>(arr: &[T]) -> usize {
    let mut min_idx = 0;
    for i in 1..4 {
        if arr[i] < arr[min_idx] {
            min_idx = i;
        }
    }
    min_idx
}

pub fn max_index<T: PartialOrd>(arr: &[T]) -> usize {
    let mut max_idx = 0;
    for i in 1..arr.len() {
        if arr[i] > arr[max_idx] {
            max_idx = i;
        }
    }
    max_idx
}

impl V2 {
    #[inline] pub fn unit() -> V2 { V2::new(1.0, 0.0) }
    #[inline] pub fn normalize_or_unit(self) -> V2 { self.normalize().unwrap_or(V2::unit()) }
    #[inline] pub fn normalize_or(self, x: f32, y: f32) -> V2 { self.normalize().unwrap_or(V2::new(x, y)) }
    #[inline] pub fn min_index(&self) -> usize { if self.x < self.y { 0 } else { 1 } }
    #[inline] pub fn max_index(&self) -> usize { if self.x > self.y { 0 } else { 1 } }
    #[inline] pub fn outer_prod(self, o: V2) -> M2x2 { M2x2{x: self*o.x, y: self*o.y } }
    #[inline] pub fn cross(self, o: V2) -> f32 { self.x*o.y - self.y*o.x }
}

impl V3 {
    #[inline] pub fn unit() -> V3 { V3::new(0.0, 0.0, 1.0) }
    #[inline] pub fn expand(v: V2, z: f32) -> V3 { V3{x: v.x, y: v.y, z: z} }
    #[inline] pub fn normalize_or_unit(self) -> V3 { self.normalize().unwrap_or(V3::unit()) }
    #[inline] pub fn normalize_or(self, x: f32, y: f32, z: f32) -> V3 { self.normalize().unwrap_or(V3::new(x, y, z)) }
    #[inline] pub fn outer_prod(self, o: V3) -> M3x3 { M3x3{x: self*o.x, y: self*o.y, z: self*o.z } }

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
        V3::new(self.y*b.z - self.z*b.y,
                self.z*b.x - self.x*b.z,
                self.x*b.y - self.y*b.x)
    }

    #[inline]
    pub fn orth(self) -> V3 {
        let abs_v = self.abs();
        let mut u = V3::splat(1.0);
        u[abs_v.max_index()] = 0.0;
        // let u = u;
        u.cross(self).normalize_or(1.0, 0.0, 0.0)
    }

    #[inline]
    pub fn basis(self) -> (V3, V3, V3) {
        let a = self.normalize_or(1.0, 0.0, 0.0);
        let bu =
            if self.x.abs() > 0.57735 { // sqrt(1/3)
                V3::new(self.y, -self.x, 0.0)
            } else {
                V3::new(0.0, self.z, -self.y)
            };
        // should never need normalizing, but there may be degenerate cases...
        let b = bu.normalize_or(0.0, -1.0, 0.0);
        let c = a.cross(b);
        (a, b, c)
    }

}

impl V4 {
    #[inline] pub fn unit() -> V4 { V4::new(0.0, 0.0, 0.0, 1.0) }
    #[inline] pub fn normalize_or_unit(self) -> V4 { self.normalize().unwrap_or(V4::unit()) }
    #[inline] pub fn normalize_or(self, x: f32, y: f32, z: f32, w: f32) -> V4 { self.normalize().unwrap_or(V4::new(x, y, z, w)) }
    #[inline] pub fn expand(v: V3, w: f32) -> V4 { V4{ x: v.x, y: v.y, z: v.z, w: w } }
    #[inline] pub fn min_index(&self) -> usize { min_index(&<[f32; 4]>::from(*self)) }
    #[inline] pub fn max_index(&self) -> usize { max_index(&<[f32; 4]>::from(*self)) }
    #[inline] pub fn outer_prod(self, o: V4) -> M4x4 { M4x4{ x: self*o.x, y: self*o.y, z: self*o.z, w: self*o.w } }
    #[inline] pub fn xyz(self) -> V3 { V3{ x: self.x, y: self.y, z: self.z } }
}

impl From<V3> for V2 { #[inline] fn from(v: V3) -> V2 { V2{x: v.x, y: v.y} } }
impl From<V2> for V3 { #[inline] fn from(v: V2) -> V3 { V3{x: v.x, y: v.y, z: 0.0} } }
impl From<V4> for V3 { #[inline] fn from(v: V4) -> V3 { V3{x: v.x, y: v.y, z: v.z} } }
impl From<V3> for V4 { #[inline] fn from(v: V3) -> V4 { V4{x: v.x, y: v.y, z: v.z, w: 0.0} } }

impl From<V4> for V2 { #[inline] fn from(v: V4) -> V2 { V2{x: v.x, y: v.y} } }
impl From<V2> for V4 { #[inline] fn from(v: V2) -> V4 { V4{x: v.x, y: v.y, z: 0.0, w: 0.0} } }

impl AsRef<V2> for V3 { #[inline] fn as_ref(&self) -> &V2 { unsafe { mem::transmute(self) } } }
impl AsRef<V2> for V4 { #[inline] fn as_ref(&self) -> &V2 { unsafe { mem::transmute(self) } } }
impl AsRef<V3> for V4 { #[inline] fn as_ref(&self) -> &V3 { unsafe { mem::transmute(self) } } }

impl AsMut<V2> for V3 { #[inline] fn as_mut(&mut self) -> &mut V2 { unsafe { mem::transmute(self) } } }
impl AsMut<V2> for V4 { #[inline] fn as_mut(&mut self) -> &mut V2 { unsafe { mem::transmute(self) } } }
impl AsMut<V3> for V4 { #[inline] fn as_mut(&mut self) -> &mut V3 { unsafe { mem::transmute(self) } } }

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

macro_rules! do_mat_boilerplate {
    ($Mn: ident { $($field: ident : $index: expr),+ }, $Vn: ident,
     $size: expr, $elems: expr) =>
    {
        impl $Mn {
            #[inline]
            fn map<F: Fn($Vn) -> $Vn>(self, f: F) -> $Mn {
                $Mn{$($field: f(self.$field)),+}
            }

            #[inline]
            fn zip<F: Fn($Vn, $Vn) -> $Vn>(self, o: $Mn, f: F) -> $Mn {
                $Mn{$($field: f(self.$field, o.$field)),+}
            }
        }

        impl AsRef<[f32; $elems]> for $Mn {
            #[inline]
            fn as_ref(&self) -> &[f32; $elems] {
                unsafe { mem::transmute(self) }
            }
        }

         impl AsMut<[f32; $elems]> for $Mn {
            #[inline]
            fn as_mut(&mut self) -> &mut [f32; $elems] {
                unsafe { mem::transmute(self) }
            }
        }

        impl From<$Mn> for [f32; $elems] {
            #[inline]
            fn from(m: $Mn) -> [f32; $elems] {
                let r: &[f32; $elems] = m.as_ref();
                *r
            }
        }

        impl AsRef<[$Vn; $size]> for $Mn {
            #[inline]
            fn as_ref(&self) -> &[$Vn; $size] {
                unsafe { mem::transmute(self) }
            }
        }

        impl AsMut<[$Vn; $size]> for $Mn {
            #[inline]
            fn as_mut(&mut self) -> &mut [$Vn; $size] {
                unsafe { mem::transmute(self) }
            }
        }

        impl From<$Mn> for [$Vn; $size] {
            #[inline]
            fn from(m: $Mn) -> [$Vn; $size] {
                let r: &[$Vn; $size] = m.as_ref();
                *r
            }
        }

        impl Index<usize> for $Mn {
            type Output = $Vn;
            #[inline]
            fn index<'a>(&'a self, i: usize) -> &'a $Vn {
                let v: &[$Vn; $size] = self.as_ref();
                &v[i]
            }
        }

        impl IndexMut<usize> for $Mn {
            #[inline]
            fn index_mut<'a>(&'a mut self, i: usize) -> &'a mut $Vn {
                let v: &mut [$Vn; $size] = self.as_mut();
                &mut v[i]
            }
        }

        impl Add for $Mn {
            type Output = $Mn;
            #[inline]
            fn add(self, o: $Mn) -> $Mn {
                self.zip(o, |a, b| a + b)
            }
        }

        impl Sub for $Mn {
            type Output = $Mn;
            #[inline]
            fn sub(self, o: $Mn) -> $Mn {
                self.zip(o, |a, b| a - b)
            }
        }

        impl Mul<$Mn> for $Mn {
            type Output = $Mn;
            #[inline]
            fn mul(self, rhs: $Mn) -> $Mn {
                $Mn{$($field: self*rhs.$field),+}
            }
        }

        impl Mul<f32> for $Mn {
            type Output = $Mn;
            #[inline]
            fn mul(self, rhs: f32) -> $Mn {
                self.map(|a| a * rhs)
            }
        }

        impl Div<f32> for $Mn {
            type Output = $Mn;
            #[inline]
            fn div(self, rhs: f32) -> $Mn {
                self.map(|a| a / rhs)
            }
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
            x: V3::new(xx, xy, xz),
            y: V3::new(yx, yy, yz),
            z: V3::new(zx, zy, zz)
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
                (mag_w, V3::new( 1.0,  1.0, 1.0), Quat::new(0.0, 0.0, 0.0, 1.0))
            } else {
                (self.z.z, V3::new(-1.0, -1.0, 1.0), Quat::new(0.0, 0.0, 1.0, 0.0))
            };

        let (mag_xy, pre_xy, post_xy) =
            if self.x.x > self.y.y {
                (self.x.x, V3::new( 1.0, -1.0, -1.0), Quat::new(1.0, 0.0, 0.0, 0.0))
            } else {
                (self.y.y, V3::new(-1.0,  1.0, -1.0), Quat::new(0.0, 1.0, 0.0, 0.0))
            };

        let (pre, post) =
            if mag_zw > mag_xy {
                (pre_zw, post_zw)
            } else {
                (pre_xy, post_xy)
            };

        let t = pre.x*self.x.x + pre.y*self.y.y + pre.z*self.z.z + 1.0;
        let s = 0.5 / t.sqrt();
        let qp = Quat::new((pre.y * self.y.z - pre.z * self.z.y) * s,
                           (pre.z * self.z.x - pre.x * self.x.z) * s,
                           (pre.x * self.x.y - pre.y * self.y.x) * s,
                            t * s);
        debug_assert!(near_eq(qp.length(), 1.0));
        qp * post
    }

    #[inline]
    pub fn to_mat4(&self) -> M4x4 {
        M4x4{
            x: V4::expand(self.x, 0.0),
            y: V4::expand(self.y, 0.0),
            z: V4::expand(self.z, 0.0),
            w: V4::new(0.0, 0.0, 0.0, 1.0)
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
            x: V4::new(xx, xy, xz, xw),
            y: V4::new(yx, yy, yz, yw),
            z: V4::new(zx, zy, zz, zw),
            w: V4::new(wx, wy, wz, ww)
        }
    }

    #[inline]
    pub fn from_cols(x: V4, y: V4, z: V4, w: V4) -> M4x4 {
        M4x4{x: x, y: y, z: z, w: w}
    }

    #[inline]
    pub fn from_rows(x: V4, y: V4, z: V4, w: V4) -> M4x4 {
        M4x4::new(x.x, y.x, z.x, w.x,
                  x.y, y.y, z.y, w.y,
                  x.z, y.z, z.z, w.z,
                  x.w, y.w, z.w, w.w)
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
            x: V3::new(self.y.y*self.z.z - self.z.y*self.y.z,
                       self.z.y*self.x.z - self.x.y*self.z.z,
                       self.x.y*self.y.z - self.y.y*self.x.z),
            y: V3::new(self.y.z*self.z.x - self.z.z*self.y.x,
                       self.z.z*self.x.x - self.x.z*self.z.x,
                       self.x.z*self.y.x - self.y.z*self.x.x),
            z: V3::new(self.y.x*self.z.y - self.z.x*self.y.y,
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
        V3::new(self.x[i], self.y[i], self.z[i])
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
            x: V4::new(y.y*z.z*w.w + w.y*y.z*z.w + z.y*w.z*y.w - y.y*w.z*z.w - z.y*y.z*w.w - w.y*z.z*y.w,
                       x.y*w.z*z.w + z.y*x.z*w.w + w.y*z.z*x.w - w.y*x.z*z.w - z.y*w.z*x.w - x.y*z.z*w.w,
                       x.y*y.z*w.w + w.y*x.z*y.w + y.y*w.z*x.w - x.y*w.z*y.w - y.y*x.z*w.w - w.y*y.z*x.w,
                       x.y*z.z*y.w + y.y*x.z*z.w + z.y*y.z*x.w - x.y*y.z*z.w - z.y*x.z*y.w - y.y*z.z*x.w),
            y: V4::new(y.z*w.w*z.x + z.z*y.w*w.x + w.z*z.w*y.x - y.z*z.w*w.x - w.z*y.w*z.x - z.z*w.w*y.x,
                       x.z*z.w*w.x + w.z*x.w*z.x + z.z*w.w*x.x - x.z*w.w*z.x - z.z*x.w*w.x - w.z*z.w*x.x,
                       x.z*w.w*y.x + y.z*x.w*w.x + w.z*y.w*x.x - x.z*y.w*w.x - w.z*x.w*y.x - y.z*w.w*x.x,
                       x.z*y.w*z.x + z.z*x.w*y.x + y.z*z.w*x.x - x.z*z.w*y.x - y.z*x.w*z.x - z.z*y.w*x.x),
            z: V4::new(y.w*z.x*w.y + w.w*y.x*z.y + z.w*w.x*y.y - y.w*w.x*z.y - z.w*y.x*w.y - w.w*z.x*y.y,
                       x.w*w.x*z.y + z.w*x.x*w.y + w.w*z.x*x.y - x.w*z.x*w.y - w.w*x.x*z.y - z.w*w.x*x.y,
                       x.w*y.x*w.y + w.w*x.x*y.y + y.w*w.x*x.y - x.w*w.x*y.y - y.w*x.x*w.y - w.w*y.x*x.y,
                       x.w*z.x*y.y + y.w*x.x*z.y + z.w*y.x*x.y - x.w*y.x*z.y - z.w*x.x*y.y - y.w*z.x*x.y),
            w: V4::new(y.x*w.y*z.z + z.x*y.y*w.z + w.x*z.y*y.z - y.x*z.y*w.z - w.x*y.y*z.z - z.x*w.y*y.z,
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
        V4::new(self.x[i], self.y[i], self.z[i], self.w[i])
    }

    #[inline]
    fn col(&self, i: usize) -> V4 {
        self[i]
    }
}

#[inline]
pub fn determinant<M: MatType>(m: &M) -> f32 {
    m.determinant()
}

#[inline]
pub fn adjugate<M: MatType>(m: &M) -> M {
    m.adjugate()
}

#[inline]
pub fn transpose<M: MatType>(m: &M) -> M {
    m.transpose()
}

#[inline]
pub fn inverse<M: MatType>(m: &M) -> Option<M> {
    let d = m.determinant();
    if near_zero(d) {
        None
    } else {
        Some(m.adjugate() * (1.0 / d))
    }
}


impl Default for Quat {
    #[inline] fn default() -> Quat { Quat(V4::new(0.0, 0.0, 0.0, 1.0)) }
}

impl Mul<f32> for Quat {
    type Output = Quat;
    #[inline] fn mul(self, o: f32) -> Quat { Quat(self.0 * o) }
}

impl Div<f32> for Quat {
    type Output = Quat;
    #[inline] fn div(self, o: f32) -> Quat { Quat(self.0 * (1.0 / o)) }
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

impl Lerp for Quat {
    #[inline]
    fn lerp(self, o: Quat, t: f32) -> Quat {
        Quat(self.0.lerp(o.0, t))
    }
}

impl Quat {
    #[inline] pub fn new(x: f32, y: f32, z: f32, w: f32) -> Quat { Quat(V4::new(x, y, z, w)) }
    #[inline] pub fn identity() -> Quat { Quat::new(0.0, 0.0, 0.0, 1.0) }
    #[inline] pub fn axis(self) -> V3 { V3::from(self.0).normalize_or_unit() }
    #[inline] pub fn angle(self) -> f32 { self.0.w.acos() * 2.0 }

    #[inline]
    pub fn conj(self) -> Quat {
        Quat::new(-self.0.x, -self.0.y, -self.0.z, self.0.w)
    }

    #[inline] pub fn length_sq(self) -> f32 { self.0.length_sq() }
    #[inline] pub fn length(self) -> f32 { self.0.length() }
    #[inline] pub fn dot(self, o: Quat) -> f32 { self.0.dot(o.0) }

    #[inline]
    pub fn x_dir(self) -> V3 {
        let Quat(V4{x, y, z, w}) = self;
        V3::new(w*w + x*x - y*y - z*z,
                x*y + z*w + x*y + z*w,
                z*x - y*w + z*x - y*w)
    }

    #[inline]
    pub fn y_dir(self) -> V3 {
        let Quat(V4{x, y, z, w}) = self;
        V3::new(x*y - z*w + x*y - z*w,
                w*w - x*x + y*y - z*z,
                y*z + x*w + y*z + x*w)
    }

    #[inline]
    pub fn z_dir(self) -> V3 {
        let Quat(V4{x, y, z, w}) = self;
        V3::new(z*x + y*w + z*x + y*w,
                y*z - x*w + y*z - x*w,
                w*w - x*x - y*y + z*z)
    }

    #[inline]
    pub fn to_matrix(self) -> M3x3 {
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
        let v0 = v0.normalize_or_zero();
        let v1 = v1.normalize_or_zero();

        if v0.near_eq(v1) {
            return Quat::identity();
        }

        let c = v0.cross(v1);
        let d = v0.dot(v1);
        if d <= -1.0 {
            let a = v0.orth();
            Quat(V4::expand(a, 0.0))
        } else {
            let s = ((1.0 + d) * 2.0).sqrt();
            return Quat(V4::expand(c / s, s / 2.0))
        }
    }

}


impl Mul<V3> for Quat {
    type Output = V3;
    #[inline]
    fn mul(self, o: V3) -> V3 {
        // this is slow and bad...
        self.to_matrix() * o
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
    pub fn from_pose(p: V3, q: Quat) -> M4x4 {
        M4x4::from_cols(V4::expand(q.x_dir(), 0.0),
                        V4::expand(q.y_dir(), 0.0),
                        V4::expand(q.z_dir(), 0.0),
                        V4::expand(p, 1.0))
    }

    #[inline]
    pub fn new_frustum(l: f32, r: f32, b: f32, t: f32, n: f32, f: f32) -> M4x4 {
        M4x4::new(2.0 * n / (r-l),   0.0,               0.0,                    0.0,
                  0.0,               2.0 * n / (t - b), 0.0,                    0.0,
                  (r + l) / (r - l), (t + b) / (t - b), -(f + n) / (f - n),    -1.0,
                  0.0,               0.0,               -2.0 * f * n / (f - n), 0.0)
    }

    #[inline]
    pub fn from_perspective(fovy: f32, aspect: f32, n: f32, f: f32) -> M4x4 {
        let y = n * (fovy*0.5).tan();
        let x = y * aspect;
        M4x4::new_frustum(-x, x, -y, y, n, f)
    }

    #[inline]
    pub fn look_towards(fwd: V3, up: V3) -> M4x4 {
        let f = fwd.normalize_or_unit();
        let s = f.cross(up).normalize_or_unit(); //
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

#[inline]
pub fn dot<Vt: VecType>(a: Vt, b: Vt) -> f32 {
    a.dot(b)
}

#[inline]
pub fn cross(a: V3, b: V3) -> V3 {
    a.cross(b)
}

pub fn max_dir(arr: &[V3], dir: V3) -> Option<usize> {
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

pub fn compute_bounds<Vt: VecType>(arr: &[Vt]) -> Option<(Vt, Vt)> {
    if arr.len() == 0 {
        return None;
    }

    let mut min_bound = arr[0];
    let mut max_bound = arr[0];

    for item in arr.iter() {
        min_bound = min_bound.min(*item);
        max_bound = max_bound.max(*item);
    }

    Some((min_bound, max_bound))
}

#[inline]
pub fn normalize<Vt: VecType>(a: Vt) -> Option<Vt> {
    a.normalize()
}

#[inline(always)]
pub fn vec2(x: f32, y: f32) -> V2 {
    V2::new(x, y)
}

#[inline(always)]
pub fn vec3(x: f32, y: f32, z: f32) -> V3 {
    V3::new(x, y, z)
}

#[inline(always)]
pub fn vec4(x: f32, y: f32, z: f32, w: f32) -> V4 {
    V4::new(x, y, z, w)
}

#[inline(always)]
pub fn quat(x: f32, y: f32, z: f32, w: f32) -> Quat {
    Quat::new(x, y, z, w)
}



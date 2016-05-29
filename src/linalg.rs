use std::mem;
use std::cmp;
use std::ops::*;
use std::fmt::Debug;

pub trait Number
: Copy + Clone + Default + PartialEq + PartialOrd + Debug + Add<Output = Self>
+ Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + Neg<Output = Self> {
    fn zero() -> Self;
    fn max(self, Self) -> Self;
    fn min(self, Self) -> Self;
}

impl Number for f64 {
    #[inline(always)] fn zero() -> f64 { 0.0f64 }
    #[inline] fn max(self, o: Self) -> Self { f64::max(self, o) }
    #[inline] fn min(self, o: Self) -> Self { f64::min(self, o) }
}

impl Number for f32 {
    #[inline(always)] fn zero() -> f32 { 0.0f32 }
    #[inline] fn max(self, o: Self) -> Self { f32::max(self, o) }
    #[inline] fn min(self, o: Self) -> Self { f32::min(self, o) }
}

impl Number for i64 {
    #[inline(always)] fn zero() -> i64 { 0i64 }
    #[inline] fn max(self, o: Self) -> Self { cmp::max(self, o) }
    #[inline] fn min(self, o: Self) -> Self { cmp::min(self, o) }
}

impl Number for i32 {
    #[inline(always)] fn zero() -> i32 { 0i32 }
    #[inline] fn max(self, o: Self) -> Self { cmp::max(self, o) }
    #[inline] fn min(self, o: Self) -> Self { cmp::min(self, o) }
}

impl Number for i16 {
    #[inline(always)] fn zero() -> i16 { 0i16 }
    #[inline] fn max(self, o: Self) -> Self { cmp::max(self, o) }
    #[inline] fn min(self, o: Self) -> Self { cmp::min(self, o) }
}

impl Number for i8 {
    #[inline(always)] fn zero() -> i8 { 0i8 }
    #[inline] fn max(self, o: Self) -> Self { cmp::max(self, o) }
    #[inline] fn min(self, o: Self) -> Self { cmp::min(self, o) }
}


#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct V2<T: Number> {
    pub x: T,
    pub y: T,
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct V3<T: Number> {
    pub x: T,
    pub y: T,
    pub z: T,
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct V4<T: Number> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub w: T,
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Mat2<T: Number> {
    pub x: V2<T>,
    pub y: V2<T>,
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Mat3<T: Number> {
    pub x: V3<T>,
    pub y: V3<T>,
    pub z: V3<T>,
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Mat4<T: Number> {
    pub x: V4<T>,
    pub y: V4<T>,
    pub z: V4<T>,
    pub w: V4<T>,
}

pub trait VecTypeBase: Copy + Clone
        + Add<Output = Self> + Sub<Output = Self>
        + Mul<Output = Self> + Div<Output = Self> {
    type Elem : Number;
    fn map<F: Fn(Self::Elem) -> Self::Elem>(self, f: F) -> Self;
    fn zip<F: Fn(Self::Elem, Self::Elem) -> Self::Elem>(self, o: Self, f: F) -> Self;
    fn fold<F: Fn(Self::Elem, Self::Elem) -> Self::Elem>(self, f: F) -> Self::Elem;
}

pub trait VecType : VecTypeBase
    + Mul< <Self as VecTypeBase>::Elem, Output = Self>
    + Div< <Self as VecTypeBase>::Elem, Output = Self>
    + Index<usize, Output = <Self as VecTypeBase>::Elem>
    + IndexMut<usize>
{}


// not sure how to do it in the macro for fold for these so just do these here i guess...
impl<T: Number> VecTypeBase for V2<T> {
    type Elem = T;
    #[inline] fn map<F: Fn(T) -> T>(self, f: F) -> Self { V2{x: f(self.x), y: f(self.y)} }
    #[inline] fn zip<F: Fn(T, T) -> T>(self, o: Self, f: F) -> Self { V2{x: f(self.x, o.x), y: f(self.y, o.y)} }
    #[inline] fn fold<F: Fn(T, T) -> T>(self, f: F) -> T { f(self.x, self.y) }
}

impl<T: Number> VecTypeBase for V3<T> {
    type Elem = T;
    #[inline] fn map<F: Fn(T) -> T>(self, f: F) -> Self { V3{x: f(self.x), y: f(self.y), z: f(self.z)} }
    #[inline] fn zip<F: Fn(T, T) -> T>(self, o: Self, f: F) -> Self { V3{x: f(self.x, o.x), y: f(self.y, o.y), z: f(self.z, o.z)} }
    #[inline] fn fold<F: Fn(T, T) -> T>(self, f: F) -> T { f(f(self.x, self.y), self.z) }
}

impl<T: Number> VecTypeBase for V4<T> {
    type Elem = T;
    #[inline] fn map<F: Fn(T) -> T>(self, f: F) -> Self { V4{x: f(self.x), y: f(self.y), z: f(self.z), w: f(self.w)} }
    #[inline] fn zip<F: Fn(T, T) -> T>(self, o: Self, f: F) -> Self { V4{x: f(self.x, o.x), y: f(self.y, o.y), z: f(self.z, o.z), w: f(self.w, o.w)} }
    #[inline] fn fold<F: Fn(T, T) -> T>(self, f: F) -> T { f(f(f(self.x, self.y), self.z), self.w) }
}

impl<T: Number> VecType for V2<T> {}
impl<T: Number> VecType for V3<T> {}
impl<T: Number> VecType for V4<T> {}


#[inline]
pub fn dot<Vn: VecType>(a: Vn, b: Vn) -> <Vn as VecTypeBase>::Elem {
    a.zip(b, |x, y| x*y).fold(|a, b| a + b)
}

#[inline]
pub fn length2<Vn: VecType>(a: Vn) -> <Vn as VecTypeBase>::Elem {
    dot(a, a)
}

macro_rules! do_vec_boilerplate {
    ($Vn: ident <$T: ident> { $($field: ident : $index: expr),+ }, $length: expr, $tupleTy: ty) => {
        impl<$T: Number> Into<[$T; $length]> for $Vn<$T> {
            #[inline]
            fn into(self) -> [$T; $length] {
                match self {
                    $Vn{ $($field),+ } => [$($field),+]
                }
            }
        }

        impl<$T: Number> AsRef<[$T; $length]> for $Vn<$T> {
            #[inline]
            fn as_ref(&self) -> &[$T; $length] {
                unsafe { mem::transmute(self) }
            }
        }

        impl<$T: Number> AsMut<[$T; $length]> for $Vn<$T> {
            #[inline]
            fn as_mut(&mut self) -> &mut [$T; $length] {
                unsafe { mem::transmute(self) }
            }
        }

        impl<$T: Number> From<[$T; $length]> for $Vn<$T> {
            #[inline]
            fn from(v: [$T; $length]) -> $Vn<$T> {
                $Vn { $($field: v[$index].clone()),+ }
            }
        }

        impl<'a, $T: Number> From<&'a [$T; $length]> for &'a $Vn<$T> {
            #[inline]
            fn from(v: &'a [$T; $length]) -> &'a $Vn<$T> {
                unsafe { mem::transmute(v) }
            }
        }

        impl<'a, $T: Number> From<&'a mut [$T; $length]> for &'a mut $Vn<$T> {
            #[inline]
            fn from(v: &'a mut [$T; $length]) -> &'a mut $Vn<$T> {
                unsafe { mem::transmute(v) }
            }
        }

        impl<$T: Number> Index<usize> for $Vn<$T> {
            type Output = $T;
            #[inline]
            fn index<'a>(&'a self, i: usize) -> &'a $T {
                let v: &[$T; $length] = self.as_ref();
                &v[i]
            }
        }

        impl<$T: Number> IndexMut<usize> for $Vn<$T> {
            #[inline]
            fn index_mut<'a>(&'a mut self, i: usize) -> &'a mut $T {
                let v: &mut [$T; $length] = self.as_mut();
                &mut v[i]
            }
        }

        impl<$T: Number> $Vn<$T> {
            #[inline]
            pub fn new($($field: $T),+) -> $Vn<$T> {
                 $Vn{ $($field: $field),+ }
            }

            #[inline]
            pub fn splat(v: $T) -> $Vn<$T> {
                $Vn{ $($field: v),+ }
            }

            #[inline]
            pub fn max_elem(self) -> $T {
                self.fold(|a, b| a.max(b))
            }

            #[inline]
            pub fn min_elem(self) -> $T {
                self.fold(|a, b| a.min(b))
            }
        }

        impl<$T: Number> Add for $Vn<$T> {
            type Output = $Vn<$T>;
            #[inline]
            fn add(self, o: $Vn<$T>) -> $Vn<$T> {
                self.zip(o, |a, b| a + b)
            }
        }

        impl<$T: Number> Sub for $Vn<$T> {
            type Output = $Vn<$T>;
            #[inline]
            fn sub(self, o: $Vn<$T>) -> $Vn<$T> {
                self.zip(o, |a, b| a - b)
            }
        }

        impl<$T: Number> Mul for $Vn<$T> {
            type Output = $Vn<$T>;
            #[inline]
            fn mul(self, o: $Vn<$T>) -> $Vn<$T> {
                self.zip(o, |a, b| a * b)
            }
        }

        impl<$T: Number> Div for $Vn<$T> {
            type Output = $Vn<$T>;
            #[inline]
            fn div(self, o: $Vn<$T>) -> $Vn<$T> {
                self.zip(o, |a, b| a / b)
            }
        }

        impl<$T: Number> Mul<$T> for $Vn<$T> {
            type Output = $Vn<$T>;
            #[inline]
            fn mul(self, o: $T) -> $Vn<$T> {
                self.map(|a| a * o)
            }
        }

        impl<$T: Number> Div<$T> for $Vn<$T> {
            type Output = $Vn<$T>;
            #[inline]
            fn div(self, o: $T) -> $Vn<$T> {
                self.map(|a| a / o)
            }
        }

        impl<$T: Number + Neg<Output = $T>> Neg for $Vn<$T> {
            type Output = $Vn<$T>;
            #[inline]
            fn neg(self) -> $Vn<$T> {
                self.map(|a| -a)
            }
        }

        impl<$T: Number> From<$tupleTy> for $Vn<$T> {
            #[inline]
            fn from(v: $tupleTy) -> $Vn<$T> {
                match v {
                    ($($field),+) => $Vn{$($field: $field),+}
                }
            }
        }

        impl<$T: Number> Into<$tupleTy> for $Vn<$T> {
            #[inline]
            fn into(self) -> $tupleTy {
                ($(self.$field),+)
            }
        }
    }
}

do_vec_boilerplate!(V2<T> {x: 0, y: 1}, 2, (T, T));
do_vec_boilerplate!(V3<T> {x: 0, y: 1, z: 2}, 3, (T, T, T));
do_vec_boilerplate!(V4<T> {x: 0, y: 1, z: 2, w: 3}, 4, (T, T, T, T));

impl<T: Number> V3<T> {
    #[inline]
    pub fn expand(v: V2<T>, z: T) -> V3<T> {
        V3{x: v.x, y: v.y, z: z}
    }
}

impl<T: Number> V4<T> {
    #[inline]
    pub fn expand(v: V3<T>, w: T) -> V4<T> {
        V4{x: v.x, y: v.y, z: v.z, w: w}
    }
}

#[inline]
pub fn cross<T: Number>(a: V3<T>, b: V3<T>) -> V3<T> {
    V3::new(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x)
}


impl<T: Number> From<V3<T>> for V2<T> { #[inline] fn from(v: V3<T>) -> V2<T> { V2{x: v.x, y: v.y} } }
impl<T: Number> From<V2<T>> for V3<T> { #[inline] fn from(v: V2<T>) -> V3<T> { V3{x: v.x, y: v.y, z: Number::zero()} } }

impl<T: Number> From<V4<T>> for V3<T> { #[inline] fn from(v: V4<T>) -> V3<T> { V3{x: v.x, y: v.y, z: v.z} } }
impl<T: Number> From<V3<T>> for V4<T> { #[inline] fn from(v: V3<T>) -> V4<T> { V4{x: v.x, y: v.y, z: v.z, w: Number::zero()} } }

impl<T: Number> From<V4<T>> for V2<T> { #[inline] fn from(v: V4<T>) -> V2<T> { V2{x: v.x, y: v.y} } }
impl<T: Number> From<V2<T>> for V4<T> { #[inline] fn from(v: V2<T>) -> V4<T> { V4{x: v.x, y: v.y, z: Number::zero(), w: Number::zero()} } }

impl<T: Number> AsRef<V2<T>> for V3<T> { #[inline] fn as_ref(&self) -> &V2<T> { unsafe { mem::transmute(self) } } }
impl<T: Number> AsRef<V2<T>> for V4<T> { #[inline] fn as_ref(&self) -> &V2<T> { unsafe { mem::transmute(self) } } }
impl<T: Number> AsRef<V3<T>> for V4<T> { #[inline] fn as_ref(&self) -> &V3<T> { unsafe { mem::transmute(self) } } }

impl<T: Number> AsMut<V2<T>> for V3<T> { #[inline] fn as_mut(&mut self) -> &mut V2<T> { unsafe { mem::transmute(self) } } }
impl<T: Number> AsMut<V2<T>> for V4<T> { #[inline] fn as_mut(&mut self) -> &mut V2<T> { unsafe { mem::transmute(self) } } }
impl<T: Number> AsMut<V3<T>> for V4<T> { #[inline] fn as_mut(&mut self) -> &mut V3<T> { unsafe { mem::transmute(self) } } }

impl<T: Number> Mul<V2<T>> for Mat2<T> where V2<T>: VecType {
    type Output = V2<T>;
    #[inline]
    fn mul(self, v: V2<T>) -> V2<T> {
        self.x*v.x + self.y*v.y
    }
}

impl<T: Number> Mul<V3<T>> for Mat3<T> where V3<T>: VecType {
    type Output = V3<T>;
    #[inline]
    fn mul(self, v: V3<T>) -> V3<T> {
        self.x*v.x + self.y*v.y + self.z*v.z
    }
}

impl<T: Number> Mul<V4<T>> for Mat4<T> where V4<T>: VecType {
    type Output = V4<T>;
    #[inline]
    fn mul(self, v: V4<T>) -> V4<T> {
        self.x*v.x + self.y*v.y + self.z*v.z + self.w*v.w
    }
}

macro_rules! do_mat_boilerplate {
    ($Mn: ident <$T: ident> { $($field: ident : $index: expr),+ }, $Vn: ident,
     $size: expr, $elems: expr) =>
    {
        impl<$T: Number> $Mn<$T> {

            #[inline]
            fn map<F: Fn($Vn<$T>) -> $Vn<$T>>(self, f: F) -> Self {
                $Mn{$($field: f(self.$field)),+}
            }

            #[inline]
            fn zip<F: Fn($Vn<$T>, $Vn<$T>) -> $Vn<$T>>(self, o: $Mn<$T>, f: F) -> Self {
                $Mn{$($field: f(self.$field, o.$field)),+}
            }
        }

        impl<$T: Number> AsRef<[$T; $elems]> for $Mn<$T> {
            #[inline]
            fn as_ref(&self) -> &[$T; $elems] {
                unsafe { mem::transmute(self) }
            }
        }

         impl<$T: Number> AsMut<[$T; $elems]> for $Mn<$T> {
            #[inline]
            fn as_mut(&mut self) -> &mut [$T; $elems] {
                unsafe { mem::transmute(self) }
            }
        }

        impl<$T: Number> Into<[$T; $elems]> for $Mn<$T> {
            #[inline]
            fn into(self) -> [$T; $elems] {
                let r: &[$T; $elems] = self.as_ref();
                *r
            }
        }

        impl<$T: Number> AsRef<[$Vn<$T>; $size]> for $Mn<$T> {
            #[inline]
            fn as_ref(&self) -> &[$Vn<$T>; $size] {
                unsafe { mem::transmute(self) }
            }
        }

        impl<$T: Number> AsMut<[$Vn<$T>; $size]> for $Mn<$T> {
            #[inline]
            fn as_mut(&mut self) -> &mut [$Vn<$T>; $size] {
                unsafe { mem::transmute(self) }
            }
        }

        impl<$T: Number> Into<[$Vn<$T>; $size]> for $Mn<$T> {
            #[inline]
            fn into(self) -> [$Vn<$T>; $size] {
                let r: &[$Vn<$T>; $size] = self.as_ref();
                *r
            }
        }


        impl<$T: Number> Index<usize> for $Mn<$T> {
            type Output = $Vn<$T>;
            #[inline]
            fn index<'a>(&'a self, i: usize) -> &'a $Vn<$T> {
                let v: &[$Vn<$T>; $size] = self.as_ref();
                &v[i]
            }
        }

        impl<$T: Number> IndexMut<usize> for $Mn<$T> {
            #[inline]
            fn index_mut<'a>(&'a mut self, i: usize) -> &'a mut $Vn<$T> {
                let v: &mut [$Vn<$T>; $size] = self.as_mut();
                &mut v[i]
            }
        }

        impl<$T: Number> Add for $Mn<$T> where $Vn<$T>: VecTypeBase {
            type Output = $Mn<$T>;
            #[inline]
            fn add(self, o: $Mn<$T>) -> $Mn<$T> {
                self.zip(o, |a, b| a + b)
            }
        }

        impl<$T: Number> Sub for $Mn<$T> where $Vn<$T>: VecType {
            type Output = $Mn<$T>;
            #[inline]
            fn sub(self, o: $Mn<$T>) -> $Mn<$T> {
                self.zip(o, |a, b| a - b)
            }
        }

        impl<$T: Number> Mul<$Mn<$T>> for $Mn<$T> where $Vn<$T>: VecType {
            type Output = $Mn<$T>;
            #[inline]
            fn mul(self, rhs: $Mn<$T>) -> $Mn<$T> {
                $Mn{$($field: self*rhs.$field),+}
            }
        }

        impl<$T: Number> Mul<$T> for $Mn<$T> where $Vn<$T>: VecType {
            type Output = $Mn<$T>;
            #[inline]
            fn mul(self, rhs: $T) -> $Mn<$T> {
                self.map(|a| a * rhs)
            }
        }

        impl<$T: Number> Div<$T> for $Mn<$T> where $Vn<$T>: VecType {
            type Output = $Mn<$T>;
            #[inline]
            fn div(self, rhs: $T) -> $Mn<$T> {
                self.map(|a| a / rhs)
            }
        }
    }

}

do_mat_boilerplate!(Mat2<T>{x: 0, y: 1}, V2, 2, 4);
do_mat_boilerplate!(Mat3<T>{x: 0, y: 1, z: 2}, V3, 3, 9);
do_mat_boilerplate!(Mat4<T>{x: 0, y: 1, z: 2, w: 3}, V4, 4, 16);

impl<T: Number> Mat2<T> {
    #[inline]
    pub fn new(xx: T, xy: T, yx: T, yy: T) -> Self {
        Mat2{x: V2::new(xx, xy), y: V2::new(yx, yy)}
    }

    #[inline]
    pub fn from_cols(x: V2<T>, y: V2<T>) -> Self {
        Mat2{x: x, y: y}
    }

    #[inline]
    pub fn from_rows(x: V2<T>, y: V2<T>) -> Self {
        Self::new(x.x, y.x, x.y, y.y)
    }
}

impl<T: Number> Mat3<T> {

    #[inline]
    pub fn new(xx: T, xy: T, xz: T,
               yx: T, yy: T, yz: T,
               zx: T, zy: T, zz: T) -> Self {
        Mat3{
            x: V3::new(xx, xy, xz),
            y: V3::new(yx, yy, yz),
            z: V3::new(zx, zy, zz)
        }
    }

    #[inline]
    pub fn from_cols(x: V3<T>, y: V3<T>, z: V3<T>) -> Self {
        Mat3{x: x, y: y, z: z}
    }

    #[inline]
    pub fn from_rows(x: V3<T>, y: V3<T>, z: V3<T>) -> Self {
        Self::new(x.x, y.x, z.x,
                  x.y, y.y, z.y,
                  x.z, y.z, z.z)
    }
}

impl<T: Number> Mat4<T> {

    #[inline]
    pub fn new(xx: T, xy: T, xz: T, xw: T,
               yx: T, yy: T, yz: T, yw: T,
               zx: T, zy: T, zz: T, zw: T,
               wx: T, wy: T, wz: T, ww: T) -> Self {
        Mat4 {
            x: V4::new(xx, xy, xz, xw),
            y: V4::new(yx, yy, yz, yw),
            z: V4::new(zx, zy, zz, zw),
            w: V4::new(wx, wy, wz, ww)
        }
    }

    #[inline]
    pub fn from_cols(x: V4<T>, y: V4<T>, z: V4<T>, w: V4<T>) -> Self {
        Mat4{x: x, y: y, z: z, w: w}
    }

    #[inline]
    pub fn from_rows(x: V4<T>, y: V4<T>, z: V4<T>, w: V4<T>) -> Self {
        Self::new(x.x, y.x, z.x, w.x,
                  x.y, y.y, z.y, w.y,
                  x.z, y.z, z.z, w.z,
                  x.w, y.w, z.w, w.w)
    }
}

pub trait MatType: Copy + Clone
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Self, Output = Self>
{
    type Col: VecType;
    type Elem: Number;

    fn determinant(&self) -> Self::Elem;
    fn adjugate(&self) -> Self;
    fn transpose(&self) -> Self;
    fn inverse(&self) -> Option<Self>;
    fn row(&self, usize) -> Self::Col;
}

impl<T: Number> MatType for Mat2<T> {
    type Col = V2<T>;
    type Elem = T;
    #[inline] fn determinant(&self) -> T { self.x.x*self.y.y - self.x.y*self.y.x }
    #[inline] fn adjugate(&self) -> Self { Mat2::new(self.y.y, -self.x.y, -self.y.x, self.x.x) }
    #[inline] fn transpose(&self) -> Self { Mat2::from_rows(self.x, self.y) }
    #[inline] fn row(&self, i: usize) -> V2<T> { V2::new(self.x[i], self.y[i]) }

    #[inline]
    fn inverse(&self) -> Option<Self> {
        let d = self.determinant();
        if d == Number::zero() {
            None
        } else {
            Some(self.adjugate() / d)
        }
    }
}

impl<T: Number> MatType for Mat3<T> {
    type Col = V3<T>;
    type Elem = T;
    #[inline]
    fn determinant(&self) -> T {
        self.x.x*(self.y.y*self.z.z - self.z.y*self.y.z) +
        self.x.y*(self.y.z*self.z.x - self.z.z*self.y.x) +
        self.x.z*(self.y.x*self.z.y - self.z.x*self.y.y)
    }

    #[inline]
    fn adjugate(&self) -> Self {
        return Mat3 {
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
    fn transpose(&self) -> Self {
        Mat3::from_rows(self.x, self.y, self.z)
    }

    #[inline]
    fn row(&self, i: usize) -> V3<T> {
        V3::new(self.x[i], self.y[i], self.z[i])
    }

    #[inline]
    fn inverse(&self) -> Option<Self> {
        let d = self.determinant();
        if d == Number::zero() {
            None
        } else {
            Some(self.adjugate() / d)
        }
    }
}

impl<T: Number> MatType for Mat4<T> {
    type Col = V4<T>;
    type Elem = T;
    #[inline]
    fn determinant(&self) -> T {
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
    fn adjugate(&self) -> Self {
        let Mat4{x, y, z, w} = *self;
        return Mat4 {
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
    fn transpose(&self) -> Self {
        Mat4::from_rows(self.x, self.y, self.z, self.w)
    }

    #[inline]
    fn row(&self, i: usize) -> V4<T> {
        V4::new(self.x[i], self.y[i], self.z[i], self.w[i])
    }

    #[inline]
    fn inverse(&self) -> Option<Self> {
        let d = self.determinant();
        if d == Number::zero() {
            None
        } else {
            Some(self.adjugate() / d)
        }
    }
}

#[inline]
pub fn determinant<M: MatType>(m: &M) -> M::Elem {
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
    m.inverse()
}

pub type F2 = V2<f32>;
pub type F3 = V3<f32>;
pub type F4 = V4<f32>;

pub type I2 = V2<i32>;
pub type I3 = V3<i32>;
pub type I4 = V4<i32>;

pub type F2x2 = Mat2<f32>;
pub type F3x3 = Mat3<f32>;
pub type F4x4 = Mat4<f32>;

#![allow(nonstandard_style)]

#[derive(Clone, Copy, Debug, Hash, PartialEq, PartialOrd, Eq, Ord, Default)]
#[repr(C)]
pub struct IVec2<T: Copy> {
    pub x: T,
    pub y: T,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, PartialOrd, Eq, Ord, Default)]
#[repr(C)]
pub struct IVec3<T: Copy> {
    pub x: T,
    pub y: T,
    pub z: T,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, PartialOrd, Eq, Ord, Default)]
#[repr(C)]
pub struct IVec4<T: Copy> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub w: T,
}

// pub type Int2 = IVec2<i32>;
// pub type Int3 = IVec3<i32>;
// pub type Int4 = IVec4<i32>;

// pub type UInt2 = IVec2<u32>;
// pub type UInt3 = IVec3<u32>;
// pub type UInt4 = IVec4<u32>;
pub mod aliases {
    use super::*;
    pub type I32x2 = IVec2<i32>;
    pub type I32x3 = IVec3<i32>;
    pub type I32x4 = IVec4<i32>;

    pub type U32x2 = IVec2<u32>;
    pub type U32x3 = IVec3<u32>;
    pub type U32x4 = IVec4<u32>;

    pub type U16x2 = IVec2<u16>;
    pub type U16x3 = IVec3<u16>;
    pub type U16x4 = IVec4<u16>;

    pub type I16x2 = IVec2<i16>;
    pub type I16x3 = IVec3<i16>;
    pub type I16x4 = IVec4<i16>;

    pub type U8x2 = IVec2<u8>;
    pub type U8x3 = IVec3<u8>;
    pub type U8x4 = IVec4<u8>;

    pub type Size2 = IVec2<usize>;
    pub type Size3 = IVec3<usize>;
    pub type Size4 = IVec4<usize>;

    pub type Byte2 = IVec2<u8>;
    pub type Byte3 = IVec3<u8>;
    pub type Byte4 = IVec4<u8>;

    #[inline]
    pub fn int2(x: i32, y: i32) -> I32x2 {
        IVec2::new(x, y)
    }
    #[inline]
    pub fn int3(x: i32, y: i32, z: i32) -> I32x3 {
        IVec3::new(x, y, z)
    }
    #[inline]
    pub fn int4(x: i32, y: i32, z: i32, w: i32) -> I32x4 {
        IVec4::new(x, y, z, w)
    }

    #[inline]
    pub fn uint2(x: u32, y: u32) -> U32x2 {
        IVec2::new(x, y)
    }
    #[inline]
    pub fn uint3(x: u32, y: u32, z: u32) -> U32x3 {
        IVec3::new(x, y, z)
    }
    #[inline]
    pub fn uint4(x: u32, y: u32, z: u32, w: u32) -> U32x4 {
        IVec4::new(x, y, z, w)
    }

    #[inline]
    pub fn ushort2(x: u16, y: u16) -> U16x2 {
        IVec2::new(x, y)
    }

    #[inline]
    pub fn ushort3(x: u16, y: u16, z: u16) -> U16x3 {
        IVec3::new(x, y, z)
    }

    #[inline]
    pub fn ushort4(x: u16, y: u16, z: u16, w: u16) -> U16x4 {
        IVec4::new(x, y, z, w)
    }

    #[inline]
    pub fn byte4(x: u8, y: u8, z: u8, w: u8) -> Byte4 {
        IVec4::new(x, y, z, w)
    }

    #[inline]
    pub fn byte3(x: u8, y: u8, z: u8) -> Byte3 {
        IVec3::new(x, y, z)
    }
}

impl<T: Copy> IVec2<T> {
    #[inline]
    pub fn fold<F: FnMut(T, T) -> T>(self, mut f: F) -> T {
        f(self.x, self.y)
    }

    #[inline]
    pub fn fold2_init<Acc, F: FnMut(Acc, T, T) -> Acc>(self, o: Self, init: Acc, mut f: F) -> Acc {
        let acc = f(init, o.x, self.x);
        f(acc, o.y, self.y)
    }

    #[inline]
    pub fn fold_init<Acc, F: FnMut(Acc, T) -> Acc>(self, init: Acc, mut f: F) -> Acc {
        let acc = f(init, self.x);
        f(acc, self.y)
    }
    #[inline]
    pub fn tup(self) -> (T, T) {
        self.into()
    }
}

impl<T: Copy> IVec3<T> {
    #[inline]
    pub fn fold<F: FnMut(T, T) -> T>(self, mut f: F) -> T {
        let acc = f(self.x, self.y);
        f(acc, self.z)
    }

    #[inline]
    pub fn fold2_init<Acc, F: FnMut(Acc, T, T) -> Acc>(self, o: Self, init: Acc, mut f: F) -> Acc {
        let acc = f(init, o.x, self.x);
        let acc = f(acc, o.y, self.y);
        f(acc, o.z, self.z)
    }

    #[inline]
    pub fn fold_init<Acc, F: FnMut(Acc, T) -> Acc>(self, init: Acc, mut f: F) -> Acc {
        let acc = f(init, self.x);
        let acc = f(acc, self.y);
        f(acc, self.z)
    }

    /// Convert this vector into a tuple of its items.
    #[inline]
    pub fn tup(self) -> (T, T, T) {
        self.into()
    }
}

impl<T: Copy> IVec4<T> {
    #[inline]
    pub fn fold<F: FnMut(T, T) -> T>(self, mut f: F) -> T {
        let acc = f(self.x, self.y);
        let acc = f(acc, self.z);
        f(acc, self.w)
    }

    #[inline]
    pub fn fold2_init<Acc, F: FnMut(Acc, T, T) -> Acc>(self, o: Self, init: Acc, mut f: F) -> Acc {
        let acc = f(init, o.x, self.x);
        let acc = f(acc, o.y, self.y);
        let acc = f(acc, o.z, self.z);
        f(acc, o.w, self.w)
    }

    #[inline]
    pub fn fold_init<Acc, F: FnMut(Acc, T) -> Acc>(self, init: Acc, mut f: F) -> Acc {
        let acc = f(init, self.x);
        let acc = f(acc, self.y);
        let acc = f(acc, self.z);
        f(acc, self.w)
    }

    /// Convert this vector into a tuple of its items.
    #[inline]
    pub fn tup(self) -> (T, T, T, T) {
        self.into()
    }

    #[inline]
    pub fn dot(self, o: Self) -> T
    where
        T: std::ops::Mul<T, Output = T> + std::ops::Add<T, Output = T>,
    {
        self.x * o.x + self.y * o.y
    }
}

impl<T: Copy> IVec3<T> {
    #[inline]
    pub fn dot(self, o: Self) -> T
    where
        T: std::ops::Mul<T, Output = T> + std::ops::Add<T, Output = T>,
    {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
}

macro_rules! vec_from {
    () => {};
    ($dst:ident<$T:ident> [$(($id:ident : $src:ty) -> $ex:expr);+ $(;)? ] $($rest:tt)*) => {
        $(impl<'a, $T: Copy> From<$src> for $dst<T> {
            #[inline]
            fn from($id : $src) -> Self {
                $ex
            }
        })+
        vec_from!($($rest)*);
    };
}

vec_from! {
    IVec2<T>[
        (t: (T, T)) -> IVec2::new(t.0, t.1);
        (t: [T; 2]) -> IVec2::new(t[0], t[1]);
        (t: &'a [T; 2]) -> IVec2::new(t[0], t[1]);

        (v: IVec4<T>) -> IVec2::new(v.x, v.y);
        (v: IVec3<T>) -> IVec2::new(v.x, v.y);
    ]
    IVec3<T>[
        (t: (T, T, T)) -> IVec3::new(t.0, t.1, t.2);
        (t: [T; 3]) -> IVec3::new(t[0], t[1], t[2]);
        (t: &'a [T; 3]) -> IVec3::new(t[0], t[1], t[2]);

        // (v: IVec2<T>) -> IVec3::new(v.x, v.y, T::ZERO);

        (t: (IVec2<T>, T)) -> IVec3::new(t.0.x, t.0.y, t.1);
        (t: (T, IVec2<T>)) -> IVec3::new(t.0, t.1.x, t.1.y);

        (v: IVec4<T>) -> IVec3::new(v.x, v.y, v.z);
    ]
    IVec4<T>[
        (t: (T, T, T, T)) -> IVec4::new(t.0, t.1, t.2, t.3);
        (t: [T; 4]) -> IVec4::new(t[0], t[1], t[2], t[3]);
        (t: &'a [T; 4]) -> IVec4::new(t[0], t[1], t[2], t[3]);

        // (v: IVec2<T>) -> IVec4::new(v.x, v.y, T::ZERO, T::ZERO);
        // (v: IVec3<T>) -> IVec4::new(v.x, v.y, v.z, T::ZERO);

        (t: (IVec2<T>, T, T)) -> IVec4::new(t.0.x, t.0.y, t.1, t.2);
        (t: (T, IVec2<T>, T)) -> IVec4::new(t.0, t.1.x, t.1.y, t.2);
        (t: (T, T, IVec2<T>)) -> IVec4::new(t.0, t.1, t.2.x, t.2.y);
        (t: (IVec2<T>, IVec2<T>)) -> IVec4::new(t.0.x, t.0.y, t.1.x, t.1.y);

        (t: (IVec3<T>, T)) -> IVec4::new(t.0.x, t.0.y, t.0.z, t.1);
        (t: (T, IVec3<T>)) -> IVec4::new(t.0, t.1.x, t.1.y, t.1.z);
    ]
}

impl<T: Copy> Into<(T, T)> for IVec2<T> {
    #[inline]
    fn into(self) -> (T, T) {
        (self.x, self.y)
    }
}

impl<T: Copy> Into<(T, T, T)> for IVec3<T> {
    #[inline]
    fn into(self) -> (T, T, T) {
        (self.x, self.y, self.z)
    }
}

impl<T: Copy> Into<(T, T, T, T)> for IVec4<T> {
    #[inline]
    fn into(self) -> (T, T, T, T) {
        (self.x, self.y, self.z, self.w)
    }
}

macro_rules! ivec_idx_ops {
    ($Vn:ident <$T:ident>, $Indexer:ty, $out_type:ty) => {
        impl<$T: Copy> std::ops::Index<$Indexer> for $Vn<$T> {
            type Output = $out_type;
            #[inline]
            fn index(&self, index: $Indexer) -> &$out_type {
                &self.as_array()[index]
            }
        }

        impl<$T: Copy> std::ops::IndexMut<$Indexer> for $Vn<$T> {
            #[inline]
            fn index_mut(&mut self, index: $Indexer) -> &mut $out_type {
                &mut self.as_mut_array()[index]
            }
        }
    };
}

macro_rules! ivec_boilerplate {
    ($Vn: ident { $($field: ident : $index: expr),+ }, $length: expr) => {
        impl<T: Copy> $Vn<T> {

            #[inline(always)]
            pub fn as_array(&self) -> &[T; $length] {
                unsafe { &*(self as *const $Vn<T> as *const [T; $length]) }
            }

            #[inline(always)]
            pub fn as_mut_array(&mut self) -> &mut [T; $length] {
                unsafe { &mut *(self as *mut $Vn<T> as *mut [T; $length]) }
            }
            #[inline]
            pub fn as_slice(&self) -> &[T] {
                self.as_array()
            }
            #[inline]
            pub fn as_mut_slice(&mut self) -> &mut [T] {
                self.as_mut_array()
            }

            #[inline]
            pub fn to_array(self) -> [T; $length] {
                *self.as_array()
            }

            #[inline(always)]
            pub fn arr(self) -> [T; $length] {
                *self.as_array()
            }

            #[inline]
            pub fn as_ptr(&self) -> *const T {
                self as *const $Vn<T> as *const T
            }

            #[inline]
            pub fn as_mut_ptr(&mut self) -> *mut T {
                self as *mut $Vn<T> as *mut T
            }
        }

        impl<T: Copy> AsRef<[T]> for $Vn<T> {
            #[inline]
            fn as_ref(&self) -> &[T] {
                self.as_slice()
            }
        }

        impl<T: Copy> AsMut<[T]> for $Vn<T> {
            #[inline]
            fn as_mut(&mut self) -> &mut [T] {
                self.as_mut_slice()
            }
        }

        impl<T: Copy> Into<[T; $length]> for $Vn<T> {
            #[inline]
            fn into(self) -> [T; $length] {
                self.arr()
            }
        }

        ivec_idx_ops!($Vn<T>, usize, T);
        ivec_idx_ops!($Vn<T>, std::ops::Range<usize>, [T]);
        ivec_idx_ops!($Vn<T>, std::ops::RangeFrom<usize>, [T]);
        ivec_idx_ops!($Vn<T>, std::ops::RangeTo<usize>, [T]);
        ivec_idx_ops!($Vn<T>, std::ops::RangeInclusive<usize>, [T]);
        ivec_idx_ops!($Vn<T>, std::ops::RangeToInclusive<usize>, [T]);
        ivec_idx_ops!($Vn<T>, std::ops::RangeFull, [T]);

        impl<T: Copy + std::ops::Neg<Output = T>> std::ops::Neg for $Vn<T> {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self {
                Self { $($field: -self.$field),+ }
            }
        }

        impl<T: Copy + std::ops::Add<T, Output = T>> std::ops::Add for $Vn<T> {
            type Output = Self;
            #[inline]
            fn add(self, o: Self) -> Self {
                Self { $($field: (self.$field + o.$field)),+ }
            }
        }

        impl<T: Copy + std::ops::Sub<T, Output = T>> std::ops::Sub for $Vn<T> {
            type Output = Self;
            #[inline]
            fn sub(self, o: Self) -> Self {
                Self { $($field: (self.$field - o.$field)),+ }
            }
        }

        impl<T: Copy + std::ops::Mul<T, Output = T>> std::ops::Mul for $Vn<T> {
            type Output = Self;
            #[inline]
            fn mul(self, o: Self) -> Self {
                Self { $($field: (self.$field * o.$field)),+ }
            }
        }

        // impl<T: Copy + std::ops::Div<T, Output = T>> std::ops::Div for $Vn<T> {
        //     type Output = Self;
        //     #[inline]
        //     fn div(self, o: Self) -> Self {
        //         $(debug_assert!(!o.$field.is_zero()));+;
        //         Self { $($field: (self.$field / o.$field)),+ }
        //     }
        // }

        impl<T: Copy + std::ops::Mul<T, Output = T>> std::ops::Mul<T> for $Vn<T> {
            type Output = Self;
            #[inline]
            fn mul(self, o: T) -> Self {
                Self { $($field: (self.$field * o)),+ }
            }
        }

        // impl std::ops::Mul<$Vn<f32>> for f32 {
        //     type Output = $Vn<f32>;
        //     #[inline]
        //     fn mul(self, v: Self::Output) -> Self::Output {
        //         Self::Output { $($field: (v.$field * self)),+ }
        //     }
        // }

        // impl std::ops::Mul<$Vn<i32>> for i32 {
        //     type Output = $Vn<i32>;
        //     #[inline]
        //     fn mul(self, v: Self::Output) -> Self::Output {
        //         Self::Output { $($field: (v.$field * self)),+ }
        //     }
        // }

        // impl std::ops::Mul<$Vn<u32>> for u32 {
        //     type Output = $Vn<u32>;
        //     #[inline]
        //     fn mul(self, v: Self::Output) -> Self::Output {
        //         Self::Output { $($field: (v.$field * self)),+ }
        //     }
        // }

        impl<T: Copy + std::ops::Div<T, Output = T>> std::ops::Div<T> for $Vn<T> {
            type Output = Self;
            #[inline]
            fn div(self, o: T) -> Self {
                Self { $($field: (self.$field / o)),+ }
            }
        }

        impl<T: Copy + std::ops::MulAssign> std::ops::MulAssign for $Vn<T> {
            #[inline]
            fn mul_assign(&mut self, rhs: Self) {
                $(self.$field *= rhs.$field;)+
            }
        }

        impl<T: Copy + std::ops::DivAssign> std::ops::DivAssign for $Vn<T> {
            #[inline]
            fn div_assign(&mut self, rhs: Self) {
                $(self.$field /= rhs.$field;)+
            }
        }

        impl<T: Copy + std::ops::MulAssign> std::ops::MulAssign<T> for $Vn<T> {
            #[inline]
            fn mul_assign(&mut self, rhs: T) {
                $(self.$field *= rhs;)+
            }
        }

        impl<T: Copy + std::ops::DivAssign> std::ops::DivAssign<T> for $Vn<T> {
            #[inline]
            fn div_assign(&mut self, rhs: T) {
                $(self.$field /= rhs;)+
            }
        }

        impl<T: Copy + std::ops::AddAssign> std::ops::AddAssign for $Vn<T> {
            #[inline]
            fn add_assign(&mut self, rhs: Self) {
                $(self.$field += rhs.$field;)+
            }
        }

        impl<T: Copy + std::ops::SubAssign> std::ops::SubAssign for $Vn<T> {
            #[inline]
            fn sub_assign(&mut self, rhs: Self) {
                $(self.$field -= rhs.$field;)+
            }
        }

        impl<T: Copy> $Vn<T> {
            pub const SIZE: usize = $length;
            #[inline]
            pub fn new($($field: T),+) -> Self {
                Self { $($field: $field),+ }
            }

            #[inline]
            pub fn splat(v: T) -> Self {
                Self { $($field: v),+ }
            }

            #[inline]
            pub fn map<F: FnMut(T) -> T>(self, mut f: F) -> Self {
                Self { $($field: f(self.$field)),+ }
            }

            // #[inline]
            // fn zip<F: FnMut(T, T) -> T>(self, a: Self, mut f: F) -> Self {
            //     Self { $($field: f(self.$field, a.$field)),+ }
            // }

            #[inline]
            pub fn iter<'a>(&'a self) -> impl Iterator<Item = T> + 'a {
                self.as_slice().iter().copied()
            }

            #[inline]
            pub fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut T> + 'a {
                self.as_mut_slice().iter_mut()
            }

            #[inline]
            pub fn count() -> usize {
                $length
            }

            #[inline]
            pub fn max_elem(self) -> T where T: PartialOrd {
                self.fold(|a, b| if a < b { b } else { a })
            }

            #[inline]
            pub fn min_elem(self) -> T where T: PartialOrd {
                self.fold(|a, b| if a < b { a } else { b })
            }

            // #[inline]
            // pub fn clamp(self, min: Self, max: Self) -> Self {
            //     Self { $($field: crate::clamp(self.$field, min.$field, max.$field)),+ }
            // }

            #[inline]
            pub fn safe_div(self, denom: T) -> Option<Self>
            where T: PartialEq + Default + std::ops::Div<T, Output = T> {
                if denom == T::default() {
                    None
                } else {
                    Some(self / denom)
                }
            }

            #[inline]
            pub fn div_or_zero(self, denom: T) -> Self
            where T: PartialEq + Default + std::ops::Div<T, Output = T>
            {
                self.safe_div(denom).unwrap_or_default()
            }

            // /// Shorthand for `self.safe_div(denom).unwrap_or(Self::ZERO)`.
            // #[inline]
            // pub fn lerp(self, b: Self, t: f32) -> Self
            // where T: Lerp
            // {
            //     Self { $($field: self.$field.lerp(b.$field, t)),+ }
            // }
        }

    }
}

ivec_boilerplate!(IVec2 { x: 0, y: 1 }, 2);
ivec_boilerplate!(IVec3 { x: 0, y: 1, z: 2 }, 3);
ivec_boilerplate!(IVec4 { x: 0, y: 1, z: 2, w: 3 }, 4);

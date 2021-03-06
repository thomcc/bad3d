mod seal {
    pub trait Sealed {}
}

// pub trait NumCast<T: Copy + seal::Sealed>: Copy + seal::Sealed {
//     fn numcast(self) -> T;
// }

// pub trait NumCastFrom<T: Copy + seal::Sealed>: Copy + seal::Sealed {
//     fn numcast_from(v: T) -> Self;
// }

// impl<T, U> NumCastFrom<T> for U
// where
//     T: NumCast<U> + Copy + seal::Sealed,
//     U: Copy + seal::Sealed,
// {
//     #[inline]
//     fn numcast_from(v: T) -> U {
//         v.numcast()
//     }
// }

// macro_rules! impl_number {
//     ($Ty:ident [$($OtherT:ty),+ $(,)?]) => {
//         impl seal::Sealed for $Ty {}
//         $(impl NumCast<$OtherT> for $Ty {
//             #[inline(always)]
//             fn numcast(self) -> $OtherT {
//                 self as $OtherT
//             }
//         })+
//     }
// }

// impl_number!(i8 [i8, i16, i32, i64, u8, u16, u32, u64, isize, usize, f32, f64]);
// impl_number!(i16 [i8, i16, i32, i64, u8, u16, u32, u64, isize, usize, f32, f64]);
// impl_number!(i32 [i8, i16, i32, i64, u8, u16, u32, u64, isize, usize, f32, f64]);
// impl_number!(i64 [i8, i16, i32, i64, u8, u16, u32, u64, isize, usize, f32, f64]);
// impl_number!(u8 [i8, i16, i32, i64, u8, u16, u32, u64, isize, usize, f32, f64]);
// impl_number!(u16 [i8, i16, i32, i64, u8, u16, u32, u64, isize, usize, f32, f64]);
// impl_number!(u32 [i8, i16, i32, i64, u8, u16, u32, u64, isize, usize, f32, f64]);
// impl_number!(u64 [i8, i16, i32, i64, u8, u16, u32, u64, isize, usize, f32, f64]);
// impl_number!(usize [i8, i16, i32, i64, u8, u16, u32, u64, isize, usize, f32, f64]);
// impl_number!(isize [i8, i16, i32, i64, u8, u16, u32, u64, isize, usize, f32, f64]);
// impl_number!(f32 [i8, i16, i32, i64, u8, u16, u32, u64, isize, usize, f32, f64]);
// impl_number!(f64 [i8, i16, i32, i64, u8, u16, u32, u64, isize, usize, f32, f64]);

pub trait ApproxEq {
    fn approx_zero_e(&self, e: f32) -> bool;
    fn approx_eq_e(&self, o: &Self, e: f32) -> bool;

    #[inline]
    fn approx_zero(&self) -> bool {
        self.approx_zero_e(crate::scalar::DEFAULT_EPSILON)
    }

    #[inline]
    fn approx_eq(&self, o: &Self) -> bool {
        self.approx_eq_e(o, crate::scalar::DEFAULT_EPSILON)
    }
}

#[inline]
pub fn approx_zero<T: ApproxEq>(n: T) -> bool {
    n.approx_zero()
}

#[inline]
pub fn approx_eq<T: ApproxEq>(a: T, b: T) -> bool {
    a.approx_eq(&b)
}

#[inline]
pub fn approx_zero_e<T: ApproxEq>(n: T, e: f32) -> bool {
    n.approx_zero_e(e)
}

#[inline]
pub fn approx_eq_e<T: ApproxEq>(a: T, b: T, e: f32) -> bool {
    a.approx_eq_e(&b, e)
}

pub trait Fold: Copy + Clone {
    fn fold(self, f: impl Fn(f32, f32) -> f32) -> f32;
    fn fold2_init<T>(self, _: Self, init: T, f: impl Fn(T, f32, f32) -> T) -> T;

    #[inline]
    fn fold_init<T>(self, init: T, f: impl Fn(T, f32) -> T) -> T {
        self.fold2_init(self, init, |acc, v, _| f(acc, v))
    }
}

// pub trait TriIndices: Copy {
//     type IndexT: Copy;

//     fn tri_indices(self) -> (usize, usize, usize);

//     #[inline]
//     fn tri_verts<V: Clone>(self, vs: &[V]) -> (V, V, V) {
//         let (a, b, c) = self.tri_indices();
//         (vs[a].clone(), vs[b].clone(), vs[c].clone())
//     }

//     #[inline]
//     fn tri_verts_opt<V: Clone>(self, vs: &[V]) -> Option<(V, V, V)> {
//         let (a, b, c) = self.tri_indices();
//         // There's a clever way to optimize this: ((a-len)|(b-len)|(c-len)) >= 0
//         if a < vs.len() && b < vs.len() && c < vs.len() {
//             Some((vs[a].clone(), vs[b].clone(), vs[c].clone()))
//         } else {
//             None
//         }
//     }

//     #[inline]
//     fn tri_vert_ref<V>(self, vs: &[V]) -> (&V, &V, &V) {
//         let (a, b, c) = self.tri_indices();
//         (&vs[a], &vs[b], &vs[c])
//     }
// }

// pub trait Map: Copy + Clone {
//     fn map3<F>(self, a: Self, b: Self, f: F) -> Self
//     where
//         F: Fn(f32, f32, f32) -> f32;

//     #[inline]
//     fn map2<F>(self, o: Self, f: F) -> Self
//     where
//         F: Fn(f32, f32) -> f32,
//     {
//         self.map3(o, self, |a, b, _| f(a, b))
//     }

//     #[inline]
//     fn map<F>(self, f: F) -> Self
//     where
//         F: Fn(f32) -> f32,
//     {
//         self.map3(self, self, |a, _, _| f(a))
//     }
// }

// impl TriIndices for [u16; 3] {
//     type IndexT = u16;
//     #[inline]
//     fn tri_indices(self) -> (usize, usize, usize) {
//         (self[0] as usize, self[1] as usize, self[2] as usize)
//     }
// }

// impl TriIndices for (u16, u16, u16) {
//     type IndexT = u16;
//     #[inline]
//     fn tri_indices(self) -> (usize, usize, usize) {
//         (self.0 as usize, self.1 as usize, self.2 as usize)
//     }
// }

// impl TriIndices for [u32; 3] {
//     type IndexT = u32;
//     #[inline]
//     fn tri_indices(self) -> (usize, usize, usize) {
//         (self[0] as usize, self[1] as usize, self[2] as usize)
//     }
// }

// impl TriIndices for (u32, u32, u32) {
//     type IndexT = u32;
//     #[inline]
//     fn tri_indices(self) -> (usize, usize, usize) {
//         (self.0 as usize, self.1 as usize, self.2 as usize)
//     }
// }

#[inline(always)]
pub fn dot(a: super::vec::V3, b: super::vec::V3) -> f32 {
    a.dot(b)
}

#[inline]
pub fn clamp(a: f32, min: f32, max: f32) -> f32 {
    if a < min {
        min
    } else if a > max {
        max
    } else {
        a
    }
}

// #[inline]
// pub fn lerp<T: Lerp>(a: T, b: T, t: f32) -> T {
//     a.lerp(b, t)
// }

use std::{cmp,fmt};

#[macro_export]
macro_rules! try_opt {
    ($e: expr) => (
        match $e {
            Some(e) => e,
            None => return None
        }
    );
    ($e: expr, $($msg: expr),+) => (
        match $e {
            Some(e) => e,
            None => {
                println!($($msg),+);
                return None
            }
        }
    )
}

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

#[macro_export]
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

            #[inline] fn $func(self, other: &'a $rhs) -> <$lhs as $OperTrait<$rhs>>::Output {
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
    }
}


#[inline]
pub fn maximum<T: Copy + PartialOrd>(a: T, b: T) -> T {
    if a < b { b } else { a }
}

#[inline]
pub fn minimum<T: Copy + PartialOrd>(a: T, b: T) -> T {
    if a < b { a } else { b }
}

#[macro_export]
macro_rules! max {
    ($e: expr) => ($e);
    ($a: expr, $b: expr) => ({let _a = $a; let _b = $b; if _a < _b { _b } else { _a }});
    ($a: expr, $b: expr, $($rest: expr),+) => (max!($a, max!($b, $($rest),+)))
}

#[macro_export]
macro_rules! min {
    ($e: expr) => ($e);
    ($a: expr, $b: expr) => ({let _a = $a; let _b = $b; if _a < _b { _a } else { _b }});
    ($a: expr, $b: expr, $($rest: expr),+) => (min!($a, min!($b, $($rest),+)))
}

#[macro_export]
macro_rules! default_for_enum {
    ($ty: ident :: $which: ident) => {
        impl ::std::default::Default for $ty {
            #[inline] fn default() -> $ty { $ty :: $which }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct OrdFloat(pub f32);
// TODO Hash?
impl cmp::Eq for OrdFloat {}
impl cmp::Ord for OrdFloat {
    fn cmp(&self, o: &Self) -> cmp::Ordering { self.partial_cmp(o).unwrap_or(cmp::Ordering::Equal) }
}

impl fmt::Display for OrdFloat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{}", self.0) }
}

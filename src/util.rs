
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

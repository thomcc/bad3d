use std::{cmp, fmt, slice};

pub fn min_index<T: PartialOrd>(arr: &[T]) -> usize {
    assert!(arr.len() > 0);
    let mut min_idx = 0;
    for i in 1..arr.len() {
        if arr[i] < arr[min_idx] {
            min_idx = i;
        }
    }
    min_idx
}

pub fn max_index<T: PartialOrd>(arr: &[T]) -> usize {
    assert!(arr.len() > 0);
    let mut max_idx = 0;
    for i in 1..arr.len() {
        if arr[i] > arr[max_idx] {
            max_idx = i;
        }
    }
    max_idx
}

#[inline]
pub fn unpack_arr3<'a, T>(arrays: &'a [[T; 3]]) -> &'a [T] {
    unsafe { slice::from_raw_parts(arrays.as_ptr() as *const T, arrays.len() * 3) }
}

#[inline]
pub fn unpack_arr3_mut<'a, T>(arrays: &'a mut [[T; 3]]) -> &'a mut [T] {
    unsafe { slice::from_raw_parts_mut(arrays.as_ptr() as *mut T, arrays.len() * 3) }
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
    ($e:expr) => ($e);
    ($a:expr, $b:expr) => ({let _a = $a; let _b = $b; if _a < _b { _b } else { _a }});
    ($a:expr, $b:expr, $($rest:expr),+) => (max!($a, max!($b, $($rest),+)))
}

#[macro_export]
macro_rules! min {
    ($e:expr) => ($e);
    ($a:expr, $b:expr) => ({let _a = $a; let _b = $b; if _a < _b { _a } else { _b }});
    ($a:expr, $b:expr, $($rest:expr),+) => (min!($a, min!($b, $($rest),+)))
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct OrdFloat(pub f32);
// TODO Hash?
impl cmp::Eq for OrdFloat {}
impl cmp::Ord for OrdFloat {
    #[inline]
    fn cmp(&self, o: &Self) -> cmp::Ordering {
        self.partial_cmp(o).unwrap_or(cmp::Ordering::Equal)
    }
}

impl fmt::Display for OrdFloat {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

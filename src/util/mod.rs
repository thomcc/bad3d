use std::{cmp, fmt};

#[macro_use]
mod macros;

mod wrap_iter;

pub use crate::util::wrap_iter::*;

#[inline]
pub fn some_if<T>(cond: bool, val: T) -> Option<T> {
    if cond {
        Some(val)
    } else {
        None
    }
}

#[inline]
pub fn some_when<T>(cond: bool, func: impl FnOnce() -> T) -> Option<T> {
    if cond {
        Some(func())
    } else {
        None
    }
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

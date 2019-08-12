use std::{cmp, fmt};

#[macro_use]
mod macros;

mod wrap_iter;

pub use self::wrap_iter::*;

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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

pub struct PerfLog {
    pub sections: std::sync::Mutex<Vec<(&'static str, std::time::Duration, usize)>>,
    pub ctr: std::sync::atomic::AtomicUsize,
}

impl PerfLog {
    pub fn new() -> Self {
        Self {
            sections: std::sync::Mutex::new(vec![]),
            ctr: std::sync::atomic::AtomicUsize::new(0),
        }
    }
    #[must_use]
    pub fn begin(&self, t: &'static str) -> PerfEntry<'_> {
        PerfEntry {
            log: self,
            name: t,
            start: std::time::Instant::now(),
            i: self.ctr.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        }
    }
}

pub struct PerfEntry<'a> {
    log: &'a PerfLog,
    name: &'static str,
    start: std::time::Instant,
    i: usize,
}

impl<'a> Drop for PerfEntry<'a> {
    fn drop(&mut self) {
        let end = self.start.elapsed();
        let mut l = self.log.sections.lock().unwrap();
        l.push((self.name, end, self.i));
    }
}

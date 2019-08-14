pub mod geom;
pub mod mat;
pub mod plane;
pub mod pose;
pub mod quat;
pub mod scalar;
pub mod traits;
pub mod tri;
pub mod vec;

pub mod prelude;
pub use self::prelude::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) mod simd;

#[cfg(test)]
mod test_traits;

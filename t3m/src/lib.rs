#[macro_use]
mod macros;
pub use macros::*;

pub mod vec;

pub mod util;

pub mod geom;
pub mod mat;
pub mod plane;
pub mod pose;
pub mod quat;
pub mod scalar;
pub mod traits;
// pub mod tri;

pub mod prelude;
pub use prelude::*;

#[cfg(test)]
mod test_traits;

#[cfg(target_feature = "sse2")]
pub(crate) mod simd;

pub use crate::vec::idx3::Idx3;
pub use crate::vec::ivec::aliases::{self, *};

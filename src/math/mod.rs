

mod traits;
mod scalar;
mod vec;
mod quat;
mod mat;
mod plane;

pub mod pose;
pub mod geom;

pub use math::traits::*;
pub use math::scalar::*;
pub use math::vec::*;
pub use math::quat::*;
pub use math::mat::*;
pub use math::pose::*;
pub use math::geom::*;

pub mod v2 {
    use super::*;

    pub const ZERO: V2 = V2 { x: 0.0, y: 0.0 };
    pub const ONE:  V2 = V2 { x: 1.0, y: 1.0 };

    pub const POS_X: V2 = V2 { x: 1.0, y: 0.0 };
    pub const POS_Y: V2 = V2 { x: 0.0, y: 1.0 };

    pub const NEG_X: V2 = V2 { x: -1.0, y:  0.0 };
    pub const NEG_Y: V2 = V2 { x:  0.0, y: -1.0 };
}

pub mod v3 {
    use super::*;

    pub const ZERO: V3 = V3 { x: 0.0, y: 0.0, z: 0.0 };
    pub const ONE:  V3 = V3 { x: 1.0, y: 1.0, z: 1.0 };

    pub const POS_X: V3 = V3 { x: 1.0, y: 0.0, z: 0.0 };
    pub const POS_Y: V3 = V3 { x: 0.0, y: 1.0, z: 0.0 };
    pub const POS_Z: V3 = V3 { x: 0.0, y: 0.0, z: 1.0 };

    pub const NEG_X: V3 = V3 { x: -1.0, y:  0.0, z:  0.0 };
    pub const NEG_Y: V3 = V3 { x:  0.0, y: -1.0, z:  0.0 };
    pub const NEG_Z: V3 = V3 { x:  0.0, y:  0.0, z: -1.0 };
}

pub mod v4 {
    use super::*;

    pub const ZERO: V4 = V4 { x: 0.0, y: 0.0, z: 0.0, w: 0.0 };
    pub const ONE:  V4 = V4 { x: 1.0, y: 1.0, z: 1.0, w: 1.0 };

    pub const POS_X: V4 = V4 { x: 1.0, y: 0.0, z: 0.0, w: 0.0 };
    pub const POS_Y: V4 = V4 { x: 0.0, y: 1.0, z: 0.0, w: 0.0 };
    pub const POS_Z: V4 = V4 { x: 0.0, y: 0.0, z: 1.0, w: 0.0 };
    pub const POS_W: V4 = V4 { x: 0.0, y: 0.0, z: 0.0, w: 1.0 };

    pub const NEG_X: V4 = V4 { x: -1.0, y:  0.0, z:  0.0, w:  0.0 };
    pub const NEG_Y: V4 = V4 { x:  0.0, y: -1.0, z:  0.0, w:  0.0 };
    pub const NEG_Z: V4 = V4 { x:  0.0, y:  0.0, z: -1.0, w:  0.0 };
    pub const NEG_W: V4 = V4 { x:  0.0, y:  0.0, z: -1.0, w: -1.0 };
}

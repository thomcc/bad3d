use std::ops::*;
use std::default::Default;
use math::*;


#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Pose {
    pub position: V3,
    pub orientation: Quat,
}

// rigid transformation
impl Pose {
    #[inline]
    pub fn new(position: V3, orientation: Quat) -> Pose {
        Pose{position: position, orientation: orientation}
    }

    #[inline]
    pub fn from_translation(xyz: V3) -> Pose {
        Pose::new(xyz, Quat::identity())
    }

    #[inline]
    pub fn from_rotation(q: Quat) -> Pose {
        Pose::new(V3::zero(), q)
    }

    #[inline]
    pub fn inverse(&self) -> Pose {
        let q = self.orientation.conj();
        Pose::new(q * -self.position, q)
    }

    #[inline]
    pub fn to_mat4(&self) -> M4x4 {
        M4x4::from_pose(self.position, self.orientation)
    }
    #[inline]
    pub fn from_mat4(m: &M4x4) -> Pose {
        Pose::new(m.w.xyz(), M3x3::from_cols(m.x.xyz(), m.y.xyz(), m.z.xyz()).to_quat())
    }
}

impl Identity for Pose {
    #[inline] fn identity() -> Pose { Pose::new(V3::zero(), Quat::identity()) }
}

impl Mul<V3> for Pose {
    type Output = V3;
    #[inline]
    fn mul(self, p: V3) -> V3 {
        self.position + (self.orientation * p)
    }
}

impl Mul<Pose> for Pose {
    type Output = Pose;
    #[inline]
    fn mul(self, pose: Pose) -> Pose {
        Pose::new(self * pose.position, self.orientation * pose.orientation)
    }
}

impl Default for Pose {
    #[inline] fn default() -> Pose { Pose::identity() }
}


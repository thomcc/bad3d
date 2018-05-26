use std::ops::*;
use std::default::Default;
use math::*;


#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Pose {
    pub position: V3,
    pub orientation: Quat,
}

/// rigid transformation
impl Pose {
    #[inline]
    pub fn new(position: V3, orientation: Quat) -> Pose {
        Pose { position, orientation }
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
    pub fn to_mat4(self) -> M4x4 {
        M4x4::from_pose(self.position, self.orientation)
    }

    #[inline]
    pub fn from_mat4(m: M4x4) -> Pose {
        Pose::new(m.w.xyz(), M3x3::from_cols(m.x.xyz(), m.y.xyz(), m.z.xyz()).to_quat())
    }

    #[inline]
    pub fn new_look_at(eye: V3, target: V3, up: V3) -> Pose {
        Pose::from_mat4(M4x4::look_at(eye, target, up))
    }
}

// Hrm... This won't always result in a sane pose...
impl From<M4x4> for Pose {
    #[inline] fn from(m: M4x4) -> Pose { Pose::from_mat4(m) }
}

impl From<Pose> for M4x4 {
    #[inline] fn from(p: Pose) -> M4x4 { p.to_mat4() }
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


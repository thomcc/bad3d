use math::*;
// use math::geom::{Plane, tri_normal, tri_area};

pub trait Support {
    fn support(&self, V3) -> V3;
}

pub struct TransformedSupport<'a, T: 'a + Support + ?Sized> {
    pub pose: pose::Pose,
    pub object: &'a T
}

impl<'a, T: Support> TransformedSupport<'a, T> {
    #[inline]
    pub fn new(pose: pose::Pose, object: &'a T) -> TransformedSupport<'a, T> {
        TransformedSupport { pose: pose, object: object }
    }
}

impl<'a, T: Support + ?Sized> Support for TransformedSupport<'a, T> {
    #[inline]
    fn support(&self, dir: V3) -> V3 {
        let tranformed_dir = self.pose.orientation.conj() * dir;
        self.pose.position + self.pose.orientation * self.object.support(tranformed_dir)
    }
}

impl<'a> Support for &'a [V3] {
    #[inline]
    fn support(&self, dir: V3) -> V3 {
        max_dir(self, dir).unwrap_or(V3::zero())
    }
}


use crate::math::prelude::*;

pub trait Support {
    fn support(&self, _: V3) -> V3;
}

impl<'a, T> Support for &'a T where T: Support {
    #[inline]
    fn support(&self, v: V3) -> V3 {
        <T as Support>::support(*self, v)
    }
}

pub struct TransformedSupport<'a, T: Support + ?Sized> {
    pub pose: Pose,
    pub object: &'a T,
}

impl<'a, T: Support> TransformedSupport<'a, T> {
    #[inline]
    pub fn new(pose: Pose, object: &'a T) -> TransformedSupport<'a, T> {
        TransformedSupport { pose, object }
    }
}

impl<'a, T: Support + ?Sized> Support for TransformedSupport<'a, T> {
    #[inline]
    fn support(&self, dir: V3) -> V3 {
        self.pose * self.object.support(self.pose.orientation.conj() * dir)
    }
}

impl<'a> Support for &'a [V3] {
    #[inline]
    fn support(&self, dir: V3) -> V3 {
        assert!(self.len() != 0);
        let mut m = 0;
        for i in 1..self.len() {
            if dot(self[i], dir) > dot(self[m], dir) {
                m = i;
            }
        }
        self[m]
    }
}

use crate::math::prelude::*;

pub trait Support {
    fn support(&self, _: V3) -> V3;
}

impl<'a, T> Support for &'a T
where
    T: Support,
{
    #[inline]
    fn support(&self, v: V3) -> V3 {
        <T as Support>::support(*self, v)
    }
}

pub struct TransformedSupport<'a, T: Support + ?Sized> {
    pub pose: Pose,
    pub object: &'a T,
}

impl<'a, T: Support + ?Sized> TransformedSupport<'a, T> {
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

impl Support for [V3] {
    // #[inline]
    fn support(&self, dir: V3) -> V3 {
        assert!(!self.is_empty());
        let mut best = self[0];
        let mut best_dot = best.dot(dir);
        for &v in self.iter() {
            let new_dot = dot(v, dir);
            if new_dot > best_dot {
                best = v;
                best_dot = new_dot;
            }
        }
        best
    }
}

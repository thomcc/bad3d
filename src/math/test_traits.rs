use super::prelude::*;
use quickcheck::{Arbitrary, Gen};

impl Arbitrary for M3x3 {
    fn arbitrary<G: Gen>(g: &mut G) -> M3x3 {
        M3x3::from_cols(V3::arbitrary(g), V3::arbitrary(g), V3::arbitrary(g))
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = M3x3>> {
        let iter = std::iter::empty();
        let copy = *self;
        let iter = iter.chain(self.x.shrink().map(move |shr_value| {
            let mut result = copy;
            result.x = shr_value;
            result
        }));
        let iter = iter.chain(self.y.shrink().map(move |shr_value| {
            let mut result = copy;
            result.y = shr_value;
            result
        }));
        let iter = iter.chain(self.z.shrink().map(move |shr_value| {
            let mut result = copy;
            result.z = shr_value;
            result
        }));
        Box::new(iter)
    }
}

impl Arbitrary for M4x4 {
    fn arbitrary<G: Gen>(g: &mut G) -> M4x4 {
        M4x4::from_cols(V4::arbitrary(g), V4::arbitrary(g), V4::arbitrary(g), V4::arbitrary(g))
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = M4x4>> {
        let iter = std::iter::empty();
        let copy = *self;
        let iter = iter.chain(self.x.shrink().map(move |shr_value| {
            let mut result = copy;
            result.x = shr_value;
            result
        }));
        let iter = iter.chain(self.y.shrink().map(move |shr_value| {
            let mut result = copy;
            result.y = shr_value;
            result
        }));
        let iter = iter.chain(self.z.shrink().map(move |shr_value| {
            let mut result = copy;
            result.z = shr_value;
            result
        }));
        let iter = iter.chain(self.w.shrink().map(move |shr_value| {
            let mut result = copy;
            result.w = shr_value;
            result
        }));
        Box::new(iter)
    }
}

impl Arbitrary for V2 {
    fn arbitrary<G: Gen>(g: &mut G) -> V2 {
        V2::new(f32::arbitrary(g), f32::arbitrary(g))
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = V2>> {
        let iter = std::iter::empty();
        let copy = *self;
        let iter = iter.chain(self.x.shrink().map(move |shr_value| {
            let mut result = copy;
            result.x = shr_value;
            result
        }));
        let iter = iter.chain(self.y.shrink().map(move |shr_value| {
            let mut result = copy;
            result.y = shr_value;
            result
        }));
        Box::new(iter)
    }
}

impl Arbitrary for V3 {
    fn arbitrary<G: Gen>(g: &mut G) -> V3 {
        V3::new(f32::arbitrary(g), f32::arbitrary(g), f32::arbitrary(g))
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = V3>> {
        let iter = std::iter::empty();
        let copy = *self;
        let iter = iter.chain(self.x.shrink().map(move |shr_value| {
            let mut result = copy;
            result.x = shr_value;
            result
        }));
        let iter = iter.chain(self.y.shrink().map(move |shr_value| {
            let mut result = copy;
            result.y = shr_value;
            result
        }));
        let iter = iter.chain(self.z.shrink().map(move |shr_value| {
            let mut result = copy;
            result.z = shr_value;
            result
        }));
        Box::new(iter)
    }
}

impl Arbitrary for V4 {
    fn arbitrary<G: Gen>(g: &mut G) -> V4 {
        V4::new(
            f32::arbitrary(g),
            f32::arbitrary(g),
            f32::arbitrary(g),
            f32::arbitrary(g),
        )
    }
    fn shrink(&self) -> Box<dyn Iterator<Item = V4>> {
        let iter = std::iter::empty();
        let copy = *self;
        let iter = iter.chain(self.x.shrink().map(move |shr_value| {
            let mut result = copy;
            result.x = shr_value;
            result
        }));
        let iter = iter.chain(self.y.shrink().map(move |shr_value| {
            let mut result = copy;
            result.y = shr_value;
            result
        }));
        let iter = iter.chain(self.z.shrink().map(move |shr_value| {
            let mut result = copy;
            result.z = shr_value;
            result
        }));
        let iter = iter.chain(self.w.shrink().map(move |shr_value| {
            let mut result = copy;
            result.w = shr_value;
            result
        }));
        Box::new(iter)
    }
}

impl Arbitrary for Quat {
    fn arbitrary<G: Gen>(g: &mut G) -> Quat {
        let ax = V3::arbitrary(g);
        if ax.is_zero() {
            Quat::IDENTITY
        } else {
            Quat::new(ax.x, ax.y, ax.z, 1.0).normalize().unwrap()
        }
    }
}

impl Arbitrary for Plane {
    fn arbitrary<G: Gen>(g: &mut G) -> Plane {
        Plane::new(V3::arbitrary(g).norm_or_unit(), f32::arbitrary(g))
    }
}

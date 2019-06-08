use crate::math::traits::*;

/// `std::f32::consts::EPSILON.sqrt()`, under the logic that
/// we should assume that any arbitary calculation has probably
/// lost half of it's bits of precision
pub const DEFAULT_EPSILON: f32 = 1.0e-6; //0.00034526698_f32;

impl ApproxEq for f32 {
    #[inline]
    fn approx_zero_e(&self, e: f32) -> bool {
        self.abs() < e
    }

    #[inline]
    fn approx_eq_e(&self, o: &Self, e: f32) -> bool {
        let a = *self;
        let b = *o;
        debug_assert_le!({ e }, 1.0);
        debug_assert_ge!({ e }, 0.0);
        debug_assert!(a.is_finite(), "non-finite number: {}", { a });
        debug_assert!(b.is_finite(), "non-finite number: {}", { b });
        let sc = max!(a.abs(), b.abs(), std::f32::MIN_POSITIVE);
        (a - b).abs() < sc * e
    }
}

impl Lerp for f32 {
    #[inline]
    fn lerp(self, b: f32, t: f32) -> f32 {
        self * (1.0 - t) + b * t
    }
}

#[inline]
pub fn safe_div(a: f32, b: f32) -> Option<f32> {
    if b == 0.0 {
        None
    } else {
        Some(a / b)
    }
}

#[inline]
pub fn safe_div0(a: f32, b: f32) -> f32 {
    safe_div(a, b).unwrap_or(0.0)
}

#[inline]
pub fn safe_div1(a: f32, b: f32) -> f32 {
    safe_div(a, b).unwrap_or(1.0)
}

#[inline]
pub fn safe_div_e(a: f32, b: f32, e: f32) -> Option<f32> {
    if b.approx_zero_e(e) {
        None
    } else {
        Some(a / b)
    }
}

#[inline]
pub fn repeat(v: f32, rep: f32) -> f32 {
    ((v % rep) + rep) % rep
}

#[inline]
pub fn wrap_between(v: f32, lo: f32, hi: f32) -> f32 {
    debug_assert_lt!(lo, hi);
    repeat(v - lo, hi - lo) + lo
}

#[inline]
pub fn wrap_degrees(a: f32) -> f32 {
    repeat(a, 360.0)
}

#[inline]
pub fn round_to(a: f32, p: f32) -> f32 {
    (a / p).round() * p
}

#[inline]
pub fn clamp01(a: f32) -> f32 {
    clamp(a, 0.0, 1.0)
}

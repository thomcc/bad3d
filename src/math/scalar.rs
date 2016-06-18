use math::traits::*;

impl ApproxEq for f32 {
    #[inline]
    fn approx_zero_e(&self, e: f32) -> bool {
        self.abs() < e
    }

    #[inline]
    fn approx_eq_e(&self, o: &Self, e: f32) -> bool {
        (self - o).abs() <= e * self.abs().max(o.abs()).max(1.0)
    }
}

impl Lerp for f32 {
    #[inline]
    fn lerp(self, b: f32, t: f32) -> f32 {
        self * (1.0 - t) + b * t
    }
}

impl Clamp for f32 {
    #[inline]
    fn clamp(self, min: f32, max: f32) -> f32 {
        self.max(min).min(max)
    }
}

#[inline]
pub fn safe_div(a: f32, b: f32, default: f32) -> f32 {
    if b == 0.0 { default } else { a / b }
}

#[inline] pub fn safe_div0(a: f32, b: f32) -> f32 { safe_div(a, b, 0.0) }
#[inline] pub fn safe_div1(a: f32, b: f32) -> f32 { safe_div(a, b, 1.0) }

#[inline]
pub fn repeat(v: f32, rep: f32) -> f32 {
    ((v % rep) + rep) % rep
}

#[inline]
pub fn wrap_between(v: f32, lo: f32, hi: f32) -> f32 {
    assert!(lo < hi);
    repeat(v-lo, hi-lo) + lo
}

#[inline]
pub fn wrap_degrees(a: f32) -> f32 {
    repeat(a, 360.0)
}


pub fn round_to(a: f32, p: f32) -> f32 {
    (a / p).round() * p
}


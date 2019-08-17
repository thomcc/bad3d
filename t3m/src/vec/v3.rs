use super::*;

cfg_if::cfg_if! {
    if #[cfg(target_feature = "sse2")] {
        use std::arch::x86_64::{self as sse, __m128};

        #[derive(Copy, Clone)]
        #[repr(transparent)]
        pub struct V3(pub(crate) __m128);


        impl Default for V3 {
            #[inline(always)]
            fn default()-> Self {
                unsafe { V3(sse::_mm_setzero_ps()) }
            }
        }

        const XYZ_MASK: __m128 = const_simd_mask![!0u32; 3];
        const ABS_MASK: __m128 = const_simd_mask![0x7fff_ffffu32; 3];
        const SIGN_MASK: __m128 = const_simd_mask![0x8000_0000u32; 4];

    } else {

        #[derive(Copy, Clone, Default)]
        #[repr(C, align(16))]
        pub struct V3 {
            x: f32,
            y: f32,
            z: f32,
            // only exists for simd...
            w: f32,
        }
        #[doc(hidden)]
        pub const fn __v3_const(x: f32, y: f32, z: f32) -> V3 {
            V3 { x, y, z, w: 0.0 }
        }
    }
}

#[inline(always)]
pub fn vec3(x: f32, y: f32, z: f32) -> V3 {
    simd_match! {
        "sse2" => unsafe {
            V3(sse::_mm_set_ps(0.0, z, y, x))
        },
        _ => V3 { x, y, z, w: 0.0 }
    }
}

impl AsRef<[f32; 3]> for V3 {
    #[inline(always)]
    fn as_ref(&self) -> &[f32; 3] {
        self.as_array()
    }
}

impl AsMut<[f32; 3]> for V3 {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut [f32; 3] {
        self.as_mut_array()
    }
}

impl AsRef<[f32]> for V3 {
    #[inline(always)]
    fn as_ref(&self) -> &[f32] {
        self.as_slice()
    }
}

impl AsMut<[f32]> for V3 {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut [f32] {
        self.as_mut_slice()
    }
}

impl From<V3> for [f32; 3] {
    #[inline]
    fn from(v: V3) -> [f32; 3] {
        *v.as_array()
    }
}

impl From<V3> for (f32, f32, f32) {
    #[inline]
    fn from(v: V3) -> (f32, f32, f32) {
        (v.x(), v.y(), v.z())
    }
}

impl From<[f32; 3]> for V3 {
    #[inline]
    fn from(v: [f32; 3]) -> V3 {
        vec3(v[0], v[1], v[2])
    }
}

impl From<(f32, f32, f32)> for V3 {
    #[inline]
    fn from(v: (f32, f32, f32)) -> Self {
        vec3(v.0, v.1, v.2)
    }
}

// impl From<f32> for V3 {
//     #[inline] fn from(v: f32) -> Self { Self { $($field: v),+ } }
// }
impl Neg for V3 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        simd_match! {
            "sse2" => unsafe {
                let v = sse::_mm_xor_ps(self.0, SIGN_MASK);
                V3(sse::_mm_and_ps(v, XYZ_MASK))
            },
            _ => Self {
                x: -self.x,
                y: -self.y,
                z: -self.z,
                w: 0.0
            }
        }
    }
}

impl Add for V3 {
    type Output = Self;
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        simd_match! {
            "sse2" => unsafe {
                let v = sse::_mm_add_ps(self.0, o.0);
                // let r = sse::_mm_and_ps(v, crate::simd::CLEARW_MASK);
                V3(v)
            },
            _ => Self {
                x: self.x + o.x,
                y: self.y + o.y,
                z: self.z + o.z,
                w: 0.0,
            }
        }
    }
}

impl Sub for V3 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        simd_match! {
            "sse2" => unsafe {
                let v = sse::_mm_sub_ps(self.0, o.0);
                let v = sse::_mm_and_ps(v, XYZ_MASK);
                V3(v)
            },
            _ => Self {
                x: self.x - o.x,
                y: self.y - o.y,
                z: self.z - o.z,
                w: 0.0,
            }
        }
    }
}

impl Mul for V3 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, o: Self) -> Self {
        simd_match! {
            "sse2" => unsafe {
                let v = sse::_mm_mul_ps(self.0, o.0);
                V3(v)
            },
            _ => Self {
                x: self.x * o.x,
                y: self.y * o.y,
                z: self.z * o.z,
                w: 0.0,
            }
        }
        // Self { $($field: (self.$field * o.$field)),+ }
    }
}

impl Div for V3 {
    type Output = Self;
    #[inline(always)]
    fn div(self, o: Self) -> Self {
        debug_assert!(!o.any_zero());
        simd_match! {
            "sse2" => unsafe {
                let v = sse::_mm_div_ps(self.0, o.0);
                let v = sse::_mm_and_ps(v, XYZ_MASK);
                V3(v)
            },
            _ => Self {
                x: self.x / o.x,
                y: self.y / o.y,
                z: self.z / o.z,
                w: 0.0,
            }
        }
        // debug_assert!(o.x != 0.0);
        // debug_assert!(o.y != 0.0);
        // debug_assert!(o.z != 0.0);
        // debug_assert!(o.w != 0.0);
        // Self { $($field: (self.$field / o.$field)),+ }
    }
}

impl Mul<f32> for V3 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, o: f32) -> Self {
        self * V3::splat(o)
    }
}

impl Mul<V3> for f32 {
    type Output = V3;
    #[inline(always)]
    fn mul(self, v: V3) -> V3 {
        v * self
    }
}

impl Div<f32> for V3 {
    type Output = Self;
    #[inline(always)]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, o: f32) -> Self {
        debug_assert!(o != 0.0);
        // Note: AFAICT no benefit for simd here.
        let inv = 1.0 / o;
        self * inv
    }
}

impl Div<V3> for f32 {
    type Output = V3;
    #[inline(always)]
    fn div(self, v: V3) -> V3 {
        // TODO: use _mm_rcp_ps?
        V3::splat(self) / v
    }
}

impl MulAssign for V3 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl DivAssign for V3 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl MulAssign<f32> for V3 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: f32) {
        *self = *self * rhs;
    }
}

impl DivAssign<f32> for V3 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: f32) {
        *self = *self / rhs;
    }
}

impl AddAssign for V3 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for V3 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl V3 {
    #[inline(always)]
    pub fn x(&self) -> f32 {
        self[0]
    }
    #[inline(always)]
    pub fn y(&self) -> f32 {
        self[1]
    }
    #[inline(always)]
    pub fn z(&self) -> f32 {
        self[2]
    }

    #[inline(always)]
    pub fn mx(&mut self) -> &mut f32 {
        &mut self[0]
    }
    #[inline(always)]
    pub fn my(&mut self) -> &mut f32 {
        &mut self[1]
    }
    #[inline(always)]
    pub fn mz(&mut self) -> &mut f32 {
        &mut self[2]
    }

    #[inline(always)]
    pub fn set_x(&mut self, v: f32) {
        self[0] = v;
    }

    #[inline(always)]
    pub fn set_y(&mut self, v: f32) {
        self[1] = v;
    }

    #[inline(always)]
    pub fn set_z(&mut self, v: f32) {
        self[2] = v;
    }

    #[inline(always)]
    pub fn with_x(mut self, v: f32) -> V3 {
        self[0] = v;
        self
    }

    #[inline(always)]
    pub fn with_y(mut self, v: f32) -> V3 {
        self[1] = v;
        self
    }

    #[inline(always)]
    pub fn with_z(mut self, v: f32) -> V3 {
        self[2] = v;
        self
    }

    pub const ZERO: V3 = vec3_const![0.0, 0.0, 0.0];
    pub const ONES: V3 = vec3_const![1.0; 3];
    pub const NEG_ONES: V3 = vec3_const![-1.0; 3];
    pub const POS_X: V3 = vec3_const![1.0, 0.0, 0.0];
    pub const POS_Y: V3 = vec3_const![0.0, 1.0, 0.0];
    pub const POS_Z: V3 = vec3_const![0.0, 0.0, 1.0];
    pub const NEG_X: V3 = vec3_const![-1.0, 0.0, 0.0];
    pub const NEG_Y: V3 = vec3_const![0.0, -1.0, 0.0];
    pub const NEG_Z: V3 = vec3_const![0.0, 0.0, -1.0];

    #[inline(always)]
    pub const fn zero() -> Self {
        Self::ZERO
    }

    #[inline(always)]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        simd_match! {
            "sse2" => unsafe {
                V3(sse::_mm_set_ps(0.0, z, y, x))
            },
            _ => Self { x, y, z, w: 0.0 },
        }
    }

    #[inline(always)]
    pub fn splat(v: f32) -> Self {
        // TODO: simd by load_ss and shuffle?
        vec3(v, v, v)
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[f32] {
        self.as_array()
    }
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        self.as_mut_array()
    }

    #[inline(always)]
    pub fn as_ptr(&self) -> *const f32 {
        self as *const V3 as *const f32
    }

    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self as *mut V3 as *mut f32
        // self.as_mut_array().as_mut_ptr()
    }

    #[inline(always)]
    pub fn as_array(&self) -> &[f32; 3] {
        unsafe { &*(self as *const V3 as *const [f32; 3]) }
    }

    #[inline(always)]
    pub fn as_mut_array(&mut self) -> &mut [f32; 3] {
        unsafe { &mut *(self as *mut V3 as *mut [f32; 3]) }
    }

    #[inline(always)]
    pub fn as_array4(&self) -> &[f32; 4] {
        unsafe { &*(self as *const V3 as *const [f32; 4]) }
    }

    #[inline(always)]
    pub fn as_mut_array4(&mut self) -> &mut [f32; 4] {
        unsafe { &mut *(self as *mut V3 as *mut [f32; 4]) }
    }

    #[inline(always)]
    pub fn tup(self) -> (f32, f32, f32) {
        self.into()
    }
    #[inline(always)]
    pub fn arr(self) -> [f32; 3] {
        self.into()
    }

    #[inline]
    pub fn any_zero(self) -> bool {
        simd_match! {
            "sse2" => unsafe {
                let m = sse::_mm_movemask_ps(
                    sse::_mm_cmpeq_ps(self.0, sse::_mm_setzero_ps()));
                (m & 0b111) != 0
            },
            _ => {
                (self.x != 0.0) | (self.y != 0.0) | (self.z != 0.0)
            }
        }
    }

    // #[inline] pub fn len(&self) -> usize { 3 }

    #[inline]
    pub fn iter(&self) -> slice::Iter<'_, f32> {
        self.as_slice().iter()
    }

    #[inline]
    pub fn iter_mut(&mut self) -> slice::IterMut<'_, f32> {
        self.as_mut_slice().iter_mut()
    }

    #[inline]
    pub fn count() -> usize {
        3
    }

    #[inline]
    pub fn max_elem(self) -> f32 {
        self.x().max(self.y()).max(self.z())
    }
    #[inline]
    pub fn min_elem(self) -> f32 {
        self.x().min(self.y()).min(self.z())
    }

    #[inline]
    pub fn abs(self) -> Self {
        simd_match! {
            "sse2" => unsafe { V3(sse::_mm_and_ps(self.0, ABS_MASK)) },
            _ => self.map(|x| x.abs()),
        }
    }

    #[inline]
    pub fn floor(self) -> Self {
        simd_match! {
            "sse4.1" => unsafe { V3(sse::_mm_floor_ps(self.0)) },
            _ => self.map(|x| x.floor()),
        }
    }
    #[inline]
    pub fn ceil(self) -> Self {
        simd_match! {
            "sse4.1" => unsafe { V3(sse::_mm_ceil_ps(self.0)) },
            _ => self.map(|x| x.ceil()),
        }
    }
    #[inline]
    pub fn round(self) -> Self {
        simd_match! {
            "sse4.1" => unsafe { V3(sse::_mm_round_ps(self.0, sse::_MM_FROUND_TO_NEAREST_INT)) },
            _ => self.map(|x| x.round()),
        }
    }

    #[inline]
    pub fn min(self, o: Self) -> Self {
        simd_match! {
            "sse2" => unsafe { V3(sse::_mm_min_ps(self.0, o.0)) },
            _ => self.map2(o, |a, b| a.min(b)),
        }
    }
    #[inline]
    pub fn max(self, o: Self) -> Self {
        simd_match! {
            "sse2" => unsafe { V3(sse::_mm_max_ps(self.0, o.0)) },
            _ => self.map2(o, |a, b| a.max(b)),
        }
    }

    #[inline]
    pub fn is_zeroish(self) -> bool {
        self.length_sq() < std::f32::MIN_POSITIVE * std::f32::MIN_POSITIVE
    }

    #[inline(always)]
    pub fn length_sq(self) -> f32 {
        self.dot(self)
    }

    #[inline(always)]
    pub fn length(self) -> f32 {
        self.length_sq().sqrt()
    }

    #[inline]
    pub fn towards(self, o: Self) -> Self {
        o - self
    }

    #[inline]
    pub fn dir_towards(self, o: Self) -> Self {
        self.towards(o).norm_or_zero()
    }

    #[inline(always)]
    pub fn dist_sq(self, o: Self) -> f32 {
        (o - self).length_sq()
    }
    #[inline(always)]
    pub fn dist(self, o: Self) -> f32 {
        (o - self).length()
    }

    #[inline(always)]
    pub fn same_dir(self, o: Self) -> bool {
        self.dot(o) > 0.0
    }

    #[inline]
    pub fn clamp(self, min: Self, max: Self) -> Self {
        self.max(min).min(max)
    }

    #[inline]
    pub fn lerp(self, b: Self, t: f32) -> Self {
        self * (1.0 - t) + b * t
        // self.map2(b, |x, y| x.lerp(y, t))
    }

    #[inline]
    pub fn safe_div(self, v: f32) -> Option<Self> {
        safe_div(1.0, v).map(|inv| self * inv)
    }

    #[inline]
    pub fn safe_div0(self, v: f32) -> Self {
        self * safe_div0(1.0, v)
    }

    #[inline]
    pub fn div_unless_zero(self, v: f32) -> Self {
        self * safe_div1(1.0, v)
    }

    #[inline]
    pub fn clamp_length(self, max_len: f32) -> Self {
        let len = self.length();
        if len < max_len {
            self
        } else {
            self * (max_len / len)
        }
    }

    #[inline]
    pub fn norm_len(self) -> (Option<Self>, f32) {
        let l = self.length();
        if l == 0.0 {
            (None, l)
        } else {
            let il = 1.0 / l;
            (Some(self * il), l)
        }
    }

    #[inline]
    pub fn normalize(self) -> Option<Self> {
        self.norm_len().0
    }

    #[inline]
    pub fn norm_or_zero(self) -> Self {
        self.normalize().unwrap_or(Self::zero())
    }

    #[inline]
    pub fn norm_or_v(self, v: Self) -> Self {
        self.normalize().unwrap_or(v)
    }

    #[inline]
    pub fn must_norm(self) -> Self {
        self.normalize().unwrap()
    }

    #[inline]
    pub fn fast_norm(self) -> Self {
        let len_sq = self.dot(self);
        debug_assert_ne!(len_sq, 0.0);
        // SOMEDAY: rsqrt
        let ilen = 1.0 / len_sq.sqrt();
        self * ilen
    }

    #[inline]
    pub fn is_normalized(self) -> bool {
        self.length_sq().approx_eq(&1.0)
    }

    #[inline]
    pub fn is_normalized_e(self, epsilon: f32) -> bool {
        self.length_sq().approx_eq_e(&1.0, epsilon)
    }

    #[inline]
    pub fn is_zero(self) -> bool {
        self == V3::ZERO
    }

    #[inline]
    pub fn round_to(self, p: f32) -> Self {
        self.map(|v| round_to(v, p))
    }

    #[inline]
    pub fn unit_axis(axis: usize) -> Self {
        chek::debug_lt!(axis, 3, "Invalid axis");
        let mut v = Self::zero();
        v[axis] = 1.0;
        v
    }

    #[inline]
    pub fn angle(self, o: Self) -> f32 {
        let d = self.dot(o);
        if d > 1.0 {
            0.0
        } else {
            d.max(-1.0).acos()
        }
    }

    #[inline]
    pub fn slerp(self, o: Self, t: f32) -> Self {
        let th: f32 = self.angle(o);
        if th.approx_zero() {
            self
        } else {
            let isth = 1.0 / th.sin();
            let s0 = isth * (th * (1.0 - t)).sin();
            let s1 = isth * (th * t).sin();
            self * s0 + o * s1
        }
    }

    #[inline]
    pub fn nlerp(self, o: Self, t: f32) -> Self {
        self.lerp(o, t).norm_or_v(o)
    }

    #[inline]
    pub fn no_zeroes(self) -> bool {
        !self.any_zero()
    }

    #[inline]
    pub fn inverse(self) -> Self {
        debug_assert!(self.no_zeroes());
        // TODO: _mm_rcp_ps?
        V3::splat(1.0) / self
    }

    #[inline]
    pub fn norm_or(self, x: f32, y: f32, z: f32) -> Self {
        self.norm_or_v(vec3(x, y, z))
    }
    #[inline(always)]
    pub fn dot3(self, a: V3, b: V3, c: V3) -> V3 {
        simd_match! {
            "sse2" => {
                crate::simd::dot3(self, a, b, c)
            },
            _ => {
                self.naive_dot3(self, a, b, c)
            }
        }
    }

    #[inline(always)]
    pub fn dot4(self, a: V3, b: V3, c: V3, d: V3) -> V4 {
        simd_match! {
            "sse2" => {
                crate::simd::dot4(self, a, b, c, d)
            },
            _ => {
                self.naive_dot4(self, a, b, c, d)
            }
        }
    }

    #[inline]
    #[allow(dead_code)]
    pub(crate) fn naive_dot3(self, a: V3, b: V3, c: V3) -> V3 {
        let va = self.x() * a.x() + self.y() * a.y() + self.z() * a.z();
        let vb = self.x() * b.x() + self.y() * b.y() + self.z() * b.z();
        let vc = self.x() * c.x() + self.y() * c.y() + self.z() * c.z();
        vec3(va, vb, vc)
    }

    #[inline]
    #[allow(dead_code)]
    pub(crate) fn naive_dot4(self, a: V3, b: V3, c: V3, d: V3) -> V4 {
        let va = self.x() * a.x() + self.y() * a.y() + self.z() * a.z();
        let vb = self.x() * b.x() + self.y() * b.y() + self.z() * b.z();
        let vc = self.x() * c.x() + self.y() * c.y() + self.z() * c.z();
        let vd = self.x() * d.x() + self.y() * d.y() + self.z() * d.z();
        vec4(va, vb, vc, vd)
    }

    #[inline(always)]
    pub fn dot(self, o: V3) -> f32 {
        simd_match! {
            "sse2" => unsafe {
                let so = sse::_mm_mul_ps(self.0, o.0);
                let z = sse::_mm_movehl_ps(so, so);
                let y = sse::_mm_shuffle_ps(so, so, shuf![1, 1, 1, 1]);
                let hadd = sse::_mm_add_ss(so, y);
                let hadd = sse::_mm_add_ss(hadd, z);
                sse::_mm_cvtss_f32(hadd)
            },
            _ => {
                self.x * o.x + self.y * o.y + self.z * o.z
            }
        }
    }

    #[inline]
    pub fn outer_prod(self, o: V3) -> M3x3 {
        M3x3 {
            x: self * o.x(),
            y: self * o.y(),
            z: self * o.z(),
        }
    }

    #[allow(dead_code)]
    #[inline]
    pub(crate) fn naive_cross(self: V3, b: V3) -> V3 {
        vec3(
            self.y() * b.z() - self.z() * b.y(),
            self.z() * b.x() - self.x() * b.z(),
            self.x() * b.y() - self.y() * b.x(),
        )
    }

    #[inline(always)]
    pub fn cross(self, b: V3) -> V3 {
        simd_match! {
            "sse2" => {
                crate::simd::v3_cross(self, b)
            },
            _ => {
                self.naive_cross(b)
            }
        }
    }

    #[inline]
    pub fn orth(self) -> V3 {
        let abs_v = self.abs();
        let mut u = V3::ONES;
        u[abs_v.max_index()] = 0.0;
        // let u = u;
        u.cross(self).norm_or(1.0, 0.0, 0.0)
    }

    #[inline]
    pub fn basis(self) -> (V3, V3, V3) {
        let a = self.norm_or_v(V3::POS_X);
        let bu = if self.x().abs() > 0.57735 {
            // sqrt(1/3)
            vec3(self.y(), -self.x(), 0.0)
        } else {
            vec3(0.0, self.z(), -self.y())
        };
        // should never need normalizing, but there may be degenerate cases...
        let b = bu.norm_or_v(V3::NEG_Y);
        let c = a.cross(b);
        (a, b, c)
    }

    #[inline]
    pub fn norm_or_unit(self) -> V3 {
        self.norm_or_v(V3::POS_Z)
    }

    #[inline]
    pub fn to_arr(self) -> [f32; 3] {
        self.into()
    }

    #[inline]
    #[allow(clippy::collapsible_if)]
    #[rustfmt::skip]
    pub fn max_index(&self) -> usize {
        if self.x() > self.y() {
            if self.x() > self.z() { 0 } else { 2 }
        } else {
            if self.y() > self.z() { 1 } else { 2 }
        }
    }

    #[inline]
    #[allow(clippy::collapsible_if)]
    #[rustfmt::skip]
    pub fn min_index(&self) -> usize {
        if self.x() < self.y() {
            if self.x() < self.z() { 0 } else { 2 }
        } else {
            if self.y() < self.z() { 1 } else { 2 }
        }
    }

    #[inline]
    pub fn expand(self, w: f32) -> V4 {
        V4::expand(self, w)
    }
}

impl ApproxEq for V3 {
    #[inline]
    fn approx_zero_e(&self, e: f32) -> bool {
        self.fold_init(true, |cnd, val| cnd && val.approx_zero_e(e))
    }

    #[inline]
    fn approx_eq_e(&self, o: &Self, e: f32) -> bool {
        self.fold2_init(*o, true, |cnd, l, r| cnd && l.approx_eq_e(&r, e))
    }
}

impl V3 {
    #[inline]
    fn map3(self, a: Self, b: Self, f: impl Fn(f32, f32, f32) -> f32) -> Self {
        vec3(
            f(self.x(), a.x(), b.x()),
            f(self.y(), a.y(), b.y()),
            f(self.z(), a.z(), b.z()),
        )
    }

    // #[inline]
    // fn map2(self, o: Self, f: impl Fn(f32, f32) -> f32) -> Self {
    //     self.map3(o, self, |a, b, _| f(a, b))
    // }
    #[inline]
    pub fn map(self, f: impl Fn(f32) -> f32) -> Self {
        self.map3(self, self, |a, _, _| f(a))
    }
}

// impl VecType for V3 {
//     const SIZE: usize = 3;

//     #[inline]
//     fn splat(v: f32) -> Self {
//         V3::splat(v)
//     }
//     #[inline]
//     fn dot(self, o: Self) -> f32 {
//         V3::dot(self, o)
//     }
// }

impl std::fmt::Debug for V3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("vec3")
            .field(&self[0])
            .field(&self[1])
            .field(&self[2])
            .finish()
    }
}

impl PartialEq for V3 {
    #[inline(always)]
    fn eq(&self, o: &V3) -> bool {
        simd_match! {
            "sse2" => unsafe {
                let v = sse::_mm_cmpeq_ps(self.0, o.0);
                let m = sse::_mm_movemask_ps(v);
                (m & 0b111) == 0b111
            },
            _ => {
                self.x == o.x && self.y == o.y && self.z == o.z
            }
        }
    }
}
impl From<V2> for V3 {
    #[inline]
    fn from(v: V2) -> V3 {
        vec3(v.x, v.y, 0.0)
    }
}

impl From<V4> for V3 {
    #[inline]
    fn from(v: V4) -> V3 {
        v.xyz()
    }
}

impl IntoIterator for V3 {
    type Item = f32;
    type IntoIter = VecIter<f32>;
    #[inline]
    fn into_iter(self) -> VecIter<f32> {
        VecIter::new(*self.as_array4(), 3)
    }
}
impl Fold for V3 {
    #[inline]
    fn fold(self, f: impl Fn(f32, f32) -> f32) -> f32 {
        f(f(self.x(), self.y()), self.z())
    }

    #[inline]
    fn fold2_init<T>(self, o: Self, init: T, f: impl Fn(T, f32, f32) -> T) -> T {
        f(f(f(init, o.x(), self.x()), o.y(), self.y()), o.z(), self.z())
    }
}

impl fmt::Display for V3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "vec3({}, {}, {})", self.x(), self.y(), self.z())
    }
}
#[cfg(target_feature = "sse2")]
#[rustfmt::skip]
impl V3 {
    #[inline(always)] pub fn xxx(self) -> Self { unsafe { V3(sse::_mm_shuffle_ps(self.0, self.0, shuf![0, 0, 0, 3])) } }
    #[inline(always)] pub fn xxy(self) -> Self { unsafe { V3(sse::_mm_shuffle_ps(self.0, self.0, shuf![0, 0, 1, 3])) } }
    #[inline(always)] pub fn xxz(self) -> Self { unsafe { V3(sse::_mm_shuffle_ps(self.0, self.0, shuf![0, 0, 2, 3])) } }
    #[inline(always)] pub fn xyx(self) -> Self { unsafe { V3(sse::_mm_shuffle_ps(self.0, self.0, shuf![0, 1, 0, 3])) } }
    #[inline(always)] pub fn xyy(self) -> Self { unsafe { V3(sse::_mm_shuffle_ps(self.0, self.0, shuf![0, 1, 1, 3])) } }
    #[inline(always)] pub fn xyz(self) -> Self { unsafe { V3(sse::_mm_shuffle_ps(self.0, self.0, shuf![0, 1, 2, 3])) } }
    #[inline(always)] pub fn xzx(self) -> Self { unsafe { V3(sse::_mm_shuffle_ps(self.0, self.0, shuf![0, 2, 0, 3])) } }
    #[inline(always)] pub fn xzy(self) -> Self { unsafe { V3(sse::_mm_shuffle_ps(self.0, self.0, shuf![0, 2, 1, 3])) } }
    #[inline(always)] pub fn xzz(self) -> Self { unsafe { V3(sse::_mm_shuffle_ps(self.0, self.0, shuf![0, 2, 2, 3])) } }
    #[inline(always)] pub fn yxx(self) -> Self { unsafe { V3(sse::_mm_shuffle_ps(self.0, self.0, shuf![1, 0, 0, 3])) } }
    #[inline(always)] pub fn yxy(self) -> Self { unsafe { V3(sse::_mm_shuffle_ps(self.0, self.0, shuf![1, 0, 1, 3])) } }
    #[inline(always)] pub fn yxz(self) -> Self { unsafe { V3(sse::_mm_shuffle_ps(self.0, self.0, shuf![1, 0, 2, 3])) } }
    #[inline(always)] pub fn yyx(self) -> Self { unsafe { V3(sse::_mm_shuffle_ps(self.0, self.0, shuf![1, 1, 0, 3])) } }
    #[inline(always)] pub fn yyy(self) -> Self { unsafe { V3(sse::_mm_shuffle_ps(self.0, self.0, shuf![1, 1, 1, 3])) } }
    #[inline(always)] pub fn yyz(self) -> Self { unsafe { V3(sse::_mm_shuffle_ps(self.0, self.0, shuf![1, 1, 2, 3])) } }
    #[inline(always)] pub fn yzx(self) -> Self { unsafe { V3(sse::_mm_shuffle_ps(self.0, self.0, shuf![1, 2, 0, 3])) } }
    #[inline(always)] pub fn yzy(self) -> Self { unsafe { V3(sse::_mm_shuffle_ps(self.0, self.0, shuf![1, 2, 1, 3])) } }
    #[inline(always)] pub fn yzz(self) -> Self { unsafe { V3(sse::_mm_shuffle_ps(self.0, self.0, shuf![1, 2, 2, 3])) } }
    #[inline(always)] pub fn zxx(self) -> Self { unsafe { V3(sse::_mm_shuffle_ps(self.0, self.0, shuf![2, 0, 0, 3])) } }
    #[inline(always)] pub fn zxy(self) -> Self { unsafe { V3(sse::_mm_shuffle_ps(self.0, self.0, shuf![2, 0, 1, 3])) } }
    #[inline(always)] pub fn zxz(self) -> Self { unsafe { V3(sse::_mm_shuffle_ps(self.0, self.0, shuf![2, 0, 2, 3])) } }
    #[inline(always)] pub fn zyx(self) -> Self { unsafe { V3(sse::_mm_shuffle_ps(self.0, self.0, shuf![2, 1, 0, 3])) } }
    #[inline(always)] pub fn zyy(self) -> Self { unsafe { V3(sse::_mm_shuffle_ps(self.0, self.0, shuf![2, 1, 1, 3])) } }
    #[inline(always)] pub fn zyz(self) -> Self { unsafe { V3(sse::_mm_shuffle_ps(self.0, self.0, shuf![2, 1, 2, 3])) } }
    #[inline(always)] pub fn zzx(self) -> Self { unsafe { V3(sse::_mm_shuffle_ps(self.0, self.0, shuf![2, 2, 0, 3])) } }
    #[inline(always)] pub fn zzy(self) -> Self { unsafe { V3(sse::_mm_shuffle_ps(self.0, self.0, shuf![2, 2, 1, 3])) } }
    #[inline(always)] pub fn zzz(self) -> Self { unsafe { V3(sse::_mm_shuffle_ps(self.0, self.0, shuf![2, 2, 2, 3])) } }

    #[inline(always)] pub fn sse_xxxx(self) -> __m128 { unsafe { sse::_mm_shuffle_ps(self.0, self.0, shuf![0, 0, 0, 0]) } }
    #[inline(always)] pub fn sse_yyyy(self) -> __m128 { unsafe { sse::_mm_shuffle_ps(self.0, self.0, shuf![1, 1, 1, 1]) } }
    #[inline(always)] pub fn sse_zzzz(self) -> __m128 { unsafe { sse::_mm_shuffle_ps(self.0, self.0, shuf![2, 2, 2, 2]) } }

    #[inline(always)] pub const fn to_simd(self) -> __m128 { self.0 }
    #[inline(always)] pub const fn from_simd(v: __m128) -> Self { Self(v) }
}
#[cfg(not(target_feature = "sse2"))]
#[rustfmt::skip]
impl V3 {
    #[inline(always)] pub fn xxx(self) -> Self { vec3(self.x, self.x, self.x) }
    #[inline(always)] pub fn xxy(self) -> Self { vec3(self.x, self.x, self.y) }
    #[inline(always)] pub fn xxz(self) -> Self { vec3(self.x, self.x, self.z) }
    #[inline(always)] pub fn xyx(self) -> Self { vec3(self.x, self.y, self.x) }
    #[inline(always)] pub fn xyy(self) -> Self { vec3(self.x, self.y, self.y) }
    #[inline(always)] pub fn xyz(self) -> Self { vec3(self.x, self.y, self.z) }
    #[inline(always)] pub fn xzx(self) -> Self { vec3(self.x, self.z, self.x) }
    #[inline(always)] pub fn xzy(self) -> Self { vec3(self.x, self.z, self.y) }
    #[inline(always)] pub fn xzz(self) -> Self { vec3(self.x, self.z, self.z) }
    #[inline(always)] pub fn yxx(self) -> Self { vec3(self.y, self.x, self.x) }
    #[inline(always)] pub fn yxy(self) -> Self { vec3(self.y, self.x, self.y) }
    #[inline(always)] pub fn yxz(self) -> Self { vec3(self.y, self.x, self.z) }
    #[inline(always)] pub fn yyx(self) -> Self { vec3(self.y, self.y, self.x) }
    #[inline(always)] pub fn yyy(self) -> Self { vec3(self.y, self.y, self.y) }
    #[inline(always)] pub fn yyz(self) -> Self { vec3(self.y, self.y, self.z) }
    #[inline(always)] pub fn yzx(self) -> Self { vec3(self.y, self.z, self.x) }
    #[inline(always)] pub fn yzy(self) -> Self { vec3(self.y, self.z, self.y) }
    #[inline(always)] pub fn yzz(self) -> Self { vec3(self.y, self.z, self.z) }
    #[inline(always)] pub fn zxx(self) -> Self { vec3(self.z, self.x, self.x) }
    #[inline(always)] pub fn zxy(self) -> Self { vec3(self.z, self.x, self.y) }
    #[inline(always)] pub fn zxz(self) -> Self { vec3(self.z, self.x, self.z) }
    #[inline(always)] pub fn zyx(self) -> Self { vec3(self.z, self.y, self.x) }
    #[inline(always)] pub fn zyy(self) -> Self { vec3(self.z, self.y, self.y) }
    #[inline(always)] pub fn zyz(self) -> Self { vec3(self.z, self.y, self.z) }
    #[inline(always)] pub fn zzx(self) -> Self { vec3(self.z, self.z, self.x) }
    #[inline(always)] pub fn zzy(self) -> Self { vec3(self.z, self.z, self.y) }
    #[inline(always)] pub fn zzz(self) -> Self { vec3(self.z, self.z, self.z) }
}

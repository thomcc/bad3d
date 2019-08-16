use super::*;

cfg_if::cfg_if! {
    if #[cfg(target_feature = "sse2")] {
        use std::arch::x86_64::{self as sse, __m128};

        #[derive(Copy, Clone)]
        #[repr(transparent)]
        pub struct V4(pub(crate) __m128);

        impl Default for V4 {
            #[inline(always)]
            fn default()-> Self {
                unsafe { V4(sse::_mm_setzero_ps()) }
            }
        }

        const ABS_MASK: __m128 = const_simd_mask![0x7fff_ffffu32; 4];
        const SIGN_MASK: __m128 = const_simd_mask![0x8000_0000u32; 4];
    } else {

        #[derive(Copy, Clone, Default)]
        #[repr(C, align(16))]
        pub struct V4 {
            x: f32,
            y: f32,
            z: f32,
            w: f32,
        }

        #[doc(hidden)]
        pub const fn __v4_const(x: f32, y: f32, z: f32, w: f32) -> V4 {
            V4 { x, y, z, w }
        }
        #[macro_export]
        macro_rules! vec4_const {
            ($x:expr, $y:expr, $z:expr, $w:expr) => {{
                const VV: V4 = $crate::vec::__v4_const($x, $y, $z, $w);
                VV
            }};
            ($x:expr; 4) => {
                const VV: V4 = $crate::vec::__v4_const($x, $x, $x, $x);
                VV
            };
        }
    }
}

impl std::fmt::Debug for V4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("vec4")
            .field(&self[0])
            .field(&self[1])
            .field(&self[2])
            .field(&self[3])
            .finish()
    }
}

impl fmt::Display for V4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "vec4({}, {}, {}, {})", self.x(), self.y(), self.z(), self.w())
    }
}

#[inline]
pub fn vec4(x: f32, y: f32, z: f32, w: f32) -> V4 {
    V4::new(x, y, z, w)
}

impl V4 {
    #[inline]
    fn fold(self, f: impl Fn(f32, f32) -> f32) -> f32 {
        f(f(f(self.x(), self.y()), self.z()), self.w())
    }

    #[inline]
    fn fold2_init<T>(self, o: Self, init: T, f: impl Fn(T, f32, f32) -> T) -> T {
        f(
            f(f(f(init, o.x(), self.x()), o.y(), self.y()), o.z(), self.z()),
            o.w(),
            self.w(),
        )
    }

    // #[inline]
    // fn map2<F>(self, o: Self, f: F) -> Self
    // where
    //     F: Fn(f32, f32) -> f32,
    // {
    //     self.map3(o, self, |a, b, _| f(a, b))
    // }

    // #[inline]
    // fn map<F>(self, f: F) -> Self
    // where
    //     F: Fn(f32) -> f32,
    // {
    //     self.map3(self, self, |a, _, _| f(a))
    // }
}

impl PartialEq for V4 {
    #[inline]
    fn eq(&self, o: &V4) -> bool {
        simd_match! {
            "sse2" => unsafe {
                let v = sse::_mm_cmpeq_ps(self.0, o.0);
                let m = sse::_mm_movemask_ps(v);
                m == 0b1111
            },
            _ => {
                self.x == o.x && self.y == o.y && self.z == o.z && self.w == o.w
            }
        }
    }
}
impl AsRef<[f32; 4]> for V4 {
    #[inline]
    fn as_ref(&self) -> &[f32; 4] {
        self.as_array()
    }
}

impl AsMut<[f32; 4]> for V4 {
    #[inline]
    fn as_mut(&mut self) -> &mut [f32; 4] {
        self.as_mut_array()
    }
}

impl AsRef<[f32]> for V4 {
    #[inline]
    fn as_ref(&self) -> &[f32] {
        self.as_slice()
    }
}

impl AsMut<[f32]> for V4 {
    #[inline]
    fn as_mut(&mut self) -> &mut [f32] {
        self.as_mut_slice()
    }
}

impl From<V4> for [f32; 4] {
    #[inline]
    fn from(v: V4) -> [f32; 4] {
        *v.as_array()
    }
}

impl From<V4> for (f32, f32, f32, f32) {
    #[inline]
    fn from(v: V4) -> (f32, f32, f32, f32) {
        (v.x(), v.y(), v.z(), v.w())
    }
}

impl From<[f32; 4]> for V4 {
    #[inline]
    fn from(v: [f32; 4]) -> V4 {
        vec4(v[0], v[1], v[2], v[3])
    }
}

impl From<(f32, f32, f32, f32)> for V4 {
    #[inline]
    fn from(v: (f32, f32, f32, f32)) -> Self {
        vec4(v.0, v.1, v.2, v.3)
    }
}

impl Neg for V4 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        simd_match! {
            "sse2" => unsafe {
                V4(sse::_mm_xor_ps(self.0, SIGN_MASK))
            },
            _ => Self {
                x: -self.x,
                y: -self.y,
                z: -self.z,
                w: -self.w,
            }
        }
    }
}

impl Add for V4 {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        simd_match! {
            "sse2" => unsafe {
                V4(sse::_mm_add_ps(self.0, o.0))
            },
            _ => Self {
                x: self.x + o.x,
                y: self.y + o.y,
                z: self.z + o.z,
                w: self.w + o.w,
            }
        }
    }
}

impl Sub for V4 {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        simd_match! {
            "sse2" => unsafe {
                V4(sse::_mm_sub_ps(self.0, o.0))
            },
            _ => Self {
                x: self.x - o.x,
                y: self.y - o.y,
                z: self.z - o.z,
                w: self.w - o.w,
            }
        }
    }
}

impl Mul for V4 {
    type Output = Self;
    #[inline]
    fn mul(self, o: Self) -> Self {
        simd_match! {
            "sse2" => unsafe {
                V4(sse::_mm_mul_ps(self.0, o.0))
            },
            _ => Self {
                x: self.x * o.x,
                y: self.y * o.y,
                z: self.z * o.z,
                w: self.w * o.w,
            }
        }
    }
}

impl Div for V4 {
    type Output = Self;
    #[inline]
    fn div(self, o: Self) -> Self {
        debug_assert!(!o.any_zero());
        simd_match! {
            "sse2" => unsafe {
                V4(sse::_mm_div_ps(self.0, o.0))
            },
            _ => Self {
                x: self.x / o.x,
                y: self.y / o.y,
                z: self.z / o.z,
                w: self.w / o.w,
            }
        }
    }
}

impl Mul<f32> for V4 {
    type Output = Self;
    #[inline]
    fn mul(self, o: f32) -> Self {
        self * V4::splat(o)
    }
}

impl Mul<V4> for f32 {
    type Output = V4;
    #[inline]
    fn mul(self, v: V4) -> V4 {
        v * self
    }
}

impl Div<f32> for V4 {
    type Output = Self;
    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, o: f32) -> Self {
        debug_assert!(o != 0.0);
        self * (1.0 / o)
    }
}

impl Div<V4> for f32 {
    type Output = V4;
    #[inline]
    fn div(self, v: V4) -> V4 {
        debug_assert!(!v.any_zero());
        V4::splat(self) / v
    }
}

impl MulAssign for V4 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl DivAssign for V4 {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl MulAssign<f32> for V4 {
    #[inline]
    fn mul_assign(&mut self, rhs: f32) {
        *self = *self * rhs;
    }
}

impl DivAssign<f32> for V4 {
    #[inline]
    fn div_assign(&mut self, rhs: f32) {
        *self = *self / rhs;
    }
}

impl AddAssign for V4 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for V4 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

#[rustfmt::skip]
impl V4 {
    pub const ZERO: V4 = vec4_const![0.0, 0.0, 0.0, 0.];
    pub const ONES: V4 = vec4_const![1.0, 1.0, 1.0, 1.0];
    pub const NEG_ONES: V4 = vec4_const![-1.0, -1.0, -1.0, -1.0];

    pub const POS_X: V4 = vec4_const![1.0, 0.0, 0.0, 0.0];
    pub const POS_Y: V4 = vec4_const![0.0, 1.0, 0.0, 0.0];
    pub const POS_Z: V4 = vec4_const![0.0, 0.0, 1.0, 0.0];
    pub const POS_W: V4 = vec4_const![0.0, 0.0, 0.0, 1.0];

    pub const NEG_X: V4 = vec4_const![-1.0, 0.0, 0.0, 0.0];
    pub const NEG_Y: V4 = vec4_const![0.0, -1.0, 0.0, 0.0];
    pub const NEG_Z: V4 = vec4_const![0.0, 0.0, -1.0, 0.0];
    pub const NEG_W: V4 = vec4_const![0.0, 0.0, 0.0, -1.0];

    #[inline(always)] pub fn x(&self) -> f32 { self[0] }
    #[inline(always)] pub fn y(&self) -> f32 { self[1] }
    #[inline(always)] pub fn z(&self) -> f32 { self[2] }
    #[inline(always)] pub fn w(&self) -> f32 { self[3] }

    #[inline(always)] pub fn mx(&mut self) -> &mut f32 { &mut self[0] }
    #[inline(always)] pub fn my(&mut self) -> &mut f32 { &mut self[1] }
    #[inline(always)] pub fn mz(&mut self) -> &mut f32 { &mut self[2] }
    #[inline(always)] pub fn mw(&mut self) -> &mut f32 { &mut self[3] }

    #[inline(always)] pub fn set_x(&mut self, v: f32) { self[0] = v; }
    #[inline(always)] pub fn set_y(&mut self, v: f32) { self[1] = v; }
    #[inline(always)] pub fn set_z(&mut self, v: f32) { self[2] = v; }
    #[inline(always)] pub fn set_w(&mut self, v: f32) { self[3] = v; }

    #[inline(always)] pub fn with_x(mut self, v: f32) -> V4 { self[0] = v; self }
    #[inline(always)] pub fn with_y(mut self, v: f32) -> V4 { self[1] = v; self }
    #[inline(always)] pub fn with_z(mut self, v: f32) -> V4 { self[2] = v; self }
    #[inline(always)] pub fn with_w(mut self, v: f32) -> V4 { self[3] = v; self }

    #[inline]
    pub fn arr(self) -> [f32; 4] {
        self.into()
    }
}

impl V4 {
    #[inline]
    pub const fn zero() -> Self {
        Self::ZERO
    }

    #[inline]
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        simd_match! {
            "sse2" => unsafe { V4(sse::_mm_set_ps(w, z, y, x)) },
            _ => Self { x, y, z, w },
        }
    }

    #[inline]
    pub fn splat(v: f32) -> Self {
        simd_match! {
            "sse2" => unsafe {
                let v = sse::_mm_set_ss(v);
                V4(sse::_mm_shuffle_ps(v, v, shuf![0, 0, 0, 0]))
            },
            _ => Self::new(v, v, v, v),
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        self.as_array()
    }
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        self.as_mut_array()
    }

    #[inline]
    pub fn as_ptr(&self) -> *const f32 {
        self as *const V4 as *const f32
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self as *mut V4 as *mut f32
    }

    #[inline]
    pub fn as_array(&self) -> &[f32; 4] {
        unsafe { &*(self as *const V4 as *const [f32; 4]) }
    }

    #[inline]
    pub fn as_mut_array(&mut self) -> &mut [f32; 4] {
        unsafe { &mut *(self as *mut V4 as *mut [f32; 4]) }
    }

    #[inline]
    pub fn tup(self) -> (f32, f32, f32, f32) {
        (self[0], self[1], self[2], self[3])
    }

    #[inline]
    pub fn len(&self) -> usize {
        4
    }

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
        4
    }

    #[inline]
    pub fn max_elem(self) -> f32 {
        self.fold(|a, b| a.max(b))
    }
    #[inline]
    pub fn min_elem(self) -> f32 {
        self.fold(|a, b| a.min(b))
    }

    #[inline]
    pub fn abs(self) -> Self {
        simd_match! {
            "sse2" => unsafe { V4(sse::_mm_and_ps(self.0, ABS_MASK)) },
            _ => self.map(|x| x.abs()),
        }
    }

    #[inline]
    pub fn floor(self) -> Self {
        simd_match! {
            "sse4.1" => unsafe { V4(sse::_mm_floor_ps(self.0)) },
            _ => self.map(|x| x.floor()),
        }
    }
    #[inline]
    pub fn ceil(self) -> Self {
        simd_match! {
            "sse4.1" => unsafe { V4(sse::_mm_ceil_ps(self.0)) },
            _ => self.map(|x| x.ceil()),
        }
    }
    #[inline]
    pub fn round(self) -> Self {
        simd_match! {
            "sse4.1" => unsafe { V4(sse::_mm_round_ps(self.0, sse::_MM_FROUND_TO_NEAREST_INT)) },
            _ => self.map(|x| x.round()),
        }
    }

    #[inline]
    pub fn min(self, o: Self) -> Self {
        simd_match! {
            "sse2" => unsafe { V4(sse::_mm_min_ps(self.0, o.0)) },
            _ => self.map2(o, |a, b| a.min(b)),
        }
    }
    #[inline]
    pub fn max(self, o: Self) -> Self {
        simd_match! {
            "sse2" => unsafe { V4(sse::_mm_max_ps(self.0, o.0)) },
            _ => self.map2(o, |a, b| a.max(b)),
        }
    }

    #[inline]
    pub fn length_sq(self) -> f32 {
        self.dot(self)
    }
    #[inline]
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

    #[inline]
    pub fn dist_sq(self, o: Self) -> f32 {
        (o - self).length_sq()
    }
    #[inline]
    pub fn dist(self, o: Self) -> f32 {
        (o - self).length()
    }

    #[inline]
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
    pub fn negate_xyz(self) -> V4 {
        simd_match! {
            "sse2" => unsafe {
                const XYZSIGN: __m128 = const_simd_mask![0x8000_0000u32; 3];
                V4(sse::_mm_xor_ps(self.0, XYZSIGN))
            },
            _ => vec4(-self.x, -self.y, -self.z, self.w)
        }
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
        debug_assert!(len_sq != 0.0);
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
        self.dot(self) == 0.0_f32
    }
    #[inline]
    pub fn any_zero(self) -> bool {
        simd_match! {
            "sse2" => unsafe {
                let m = sse::_mm_movemask_ps(
                    sse::_mm_cmpeq_ps(self.0, sse::_mm_setzero_ps()));
                m != 0
            },
            _ => {
                (self.x != 0.0) | (self.y != 0.0) | (self.z != 0.0) | (self.w != 0)
            }
        }
    }

    #[inline]
    pub fn round_to(self, p: f32) -> Self {
        self.map(|v| round_to(v, p))
    }

    #[inline]
    pub fn unit_axis(axis: usize) -> Self {
        chek::debug_lt!(axis, 4, "Invalid axis");
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
    pub fn norm_or(self, x: f32, y: f32, z: f32, w: f32) -> Self {
        self.norm_or_v(vec4(x, y, z, w))
    }

    // #[inline]
    // pub fn identity() -> Self {
    //     V4 { $($field: 0.0),+ }
    // }
}
impl ApproxEq for V4 {
    #[inline]
    fn approx_zero_e(&self, e: f32) -> bool {
        self.x().approx_zero_e(e) && self.y().approx_zero_e(e) && self.z().approx_zero_e(e) && self.w().approx_zero_e(e)
    }

    #[inline]
    fn approx_eq_e(&self, o: &Self, e: f32) -> bool {
        self.fold2_init(*o, true, |cnd, l, r| cnd && l.approx_eq_e(&r, e))
    }
}

impl V4 {
    #[inline]
    fn map3(self, a: Self, b: Self, f: impl Fn(f32, f32, f32) -> f32) -> Self {
        vec4(
            f(self.x(), a.x(), b.x()),
            f(self.y(), a.y(), b.y()),
            f(self.z(), a.z(), b.z()),
            f(self.w(), a.w(), b.w()),
        )
    }

    // #[inline]
    // fn map2(self, o: Self, f: impl Fn(f32, f32) -> f32) -> Self {
    //     self.map3(o, self, |a, b, _| f(a, b))
    // }
    fn map(self, f: impl Fn(f32) -> f32) -> Self {
        self.map3(self, self, |a, _, _| f(a))
    }
}

impl V4 {
    #[inline]
    pub fn expand(v: V3, w: f32) -> V4 {
        simd_match! {
            "sse2" => V4(v.0).with_w(w),
            _ => V4 {
                x: v.x(),
                y: v.y(),
                z: v.z(),
                w,
            }

        }
    }
    #[inline]
    pub fn xyz(self) -> V3 {
        vec3(self.x(), self.y(), self.z())
    }
    #[inline]
    pub fn norm_or_unit(self) -> V4 {
        self.norm_or(0.0, 0.0, 0.0, 1.0)
    }
    #[inline]
    pub fn to_arr(self) -> [f32; 4] {
        self.into()
    }

    #[inline]
    pub fn dot(self, o: V4) -> f32 {
        self.x() * o.x() + self.y() * o.y() + self.z() * o.z() + self.w() * o.w()
    }

    #[inline]
    pub fn outer_prod(self, o: V4) -> M4x4 {
        M4x4 {
            x: self * o.x(),
            y: self * o.y(),
            z: self * o.z(),
            w: self * o.w(),
        }
    }
    #[inline]
    pub fn to_arr8(self) -> [u8; 4] {
        [
            (self.x() * 255.0).trunc() as u8,
            (self.y() * 255.0).trunc() as u8,
            (self.z() * 255.0).trunc() as u8,
            (self.w() * 255.0).trunc() as u8,
        ]
    }

    #[inline]
    #[allow(clippy::collapsible_if)]
    #[rustfmt::skip]
    pub fn max_index(&self) -> usize {
        let [x, y, z, w] = self.arr();
        if x > y {
            // y out
            if x > z {
                if x > w { 0 } else { 3 }
            }
            // z out
            else {
                if z > w { 2 } else { 3 }
            } // x out
        } else {
            // x out
            if y > z {
                if y > w { 1 } else { 3 }
            }
            // z out
            else {
                if z > w { 2 } else { 3 }
            } // y out
        }
    }

    #[inline]
    #[allow(clippy::collapsible_if)]
    #[rustfmt::skip]
    pub fn min_index(&self) -> usize {
        let [x, y, z, w] = self.arr();
        if x < y {
            // y out
            if x < z {
                if x < w { 0 } else { 3 }
            }
            // z out
            else {
                if z < w { 2 } else { 3 }
            } // x out
        } else {
            // x out
            if y < z {
                if y < w { 1 } else { 3 }
            }
            // z out
            else {
                if z < w { 2 } else { 3 }
            } // y out
        }
    }
}

#[cfg(target_feature = "sse2")]
impl V4 {
    #[inline(always)]
    pub const fn to_simd(self) -> __m128 {
        self.0
    }
    #[inline(always)]
    pub const fn from_simd(v: __m128) -> Self {
        Self(v)
    }
}

use super::*;

#[derive(Copy, Clone, Default)]
#[repr(C, align(16))]
pub struct V3 {
    x: f32,
    y: f32,
    z: f32,
    // only exists for simd...
    w: f32,
}

#[inline]
pub const fn vec3(x: f32, y: f32, z: f32) -> V3 {
    V3 { x, y, z, w: 0.0 }
}

impl AsRef<[f32; 3]> for V3 {
    #[inline]
    fn as_ref(&self) -> &[f32; 3] {
        self.as_array()
    }
}

impl AsMut<[f32; 3]> for V3 {
    #[inline]
    fn as_mut(&mut self) -> &mut [f32; 3] {
        self.as_mut_array()
    }
}

impl AsRef<[f32]> for V3 {
    #[inline]
    fn as_ref(&self) -> &[f32] {
        self.as_slice()
    }
}

impl AsMut<[f32]> for V3 {
    #[inline]
    fn as_mut(&mut self) -> &mut [f32] {
        self.as_mut_slice()
    }
}

impl From<V3> for [f32; 3] {
    #[inline]
    fn from(v: V3) -> [f32; 3] {
        [v.x, v.y, v.z]
    }
}

impl From<V3> for (f32, f32, f32) {
    #[inline]
    fn from(v: V3) -> (f32, f32, f32) {
        (v.x, v.y, v.z)
    }
}

impl From<[f32; 3]> for V3 {
    #[inline]
    fn from(v: [f32; 3]) -> V3 {
        V3 {
            x: v[0],
            y: v[1],
            z: v[2],
            w: 0.0,
        }
    }
}

impl From<(f32, f32, f32)> for V3 {
    #[inline]
    fn from(v: (f32, f32, f32)) -> Self {
        V3 {
            x: v.0,
            y: v.1,
            z: v.2,
            w: 0.0,
        }
    }
}

// impl From<f32> for V3 {
//     #[inline] fn from(v: f32) -> Self { Self { $($field: v),+ } }
// }

impl Neg for V3 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: 0.0,
        }
    }
}

impl Add for V3 {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        Self {
            x: self.x + o.x,
            y: self.y + o.y,
            z: self.z + o.z,
            w: 0.0,
        }
    }
}

impl Sub for V3 {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        Self {
            x: self.x - o.x,
            y: self.y - o.y,
            z: self.z - o.z,
            w: 0.0,
        }
    }
}

impl Mul for V3 {
    type Output = Self;
    #[inline]
    fn mul(self, o: Self) -> Self {
        Self {
            x: self.x * o.x,
            y: self.y * o.y,
            z: self.z * o.z,
            w: 0.0,
        }
        // Self { $($field: (self.$field * o.$field)),+ }
    }
}

impl Div for V3 {
    type Output = Self;
    #[inline]
    fn div(self, o: Self) -> Self {
        // debug_assert!(o.x != 0.0);
        // debug_assert!(o.y != 0.0);
        // debug_assert!(o.z != 0.0);
        // debug_assert!(o.w != 0.0);
        Self {
            x: self.x / o.x,
            y: self.y / o.y,
            z: self.z / o.z,
            w: 0.0,
        }
        // Self { $($field: (self.$field / o.$field)),+ }
    }
}

impl Mul<f32> for V3 {
    type Output = Self;
    #[inline]
    fn mul(self, o: f32) -> Self {
        Self {
            x: self.x * o,
            y: self.y * o,
            z: self.z * o,
            w: 0.0,
        }
        // Self { $($field: (self.$field * o)),+ }
    }
}

impl Mul<V3> for f32 {
    type Output = V3;
    #[inline]
    fn mul(self, v: V3) -> V3 {
        v * self
    }
}

impl Div<f32> for V3 {
    type Output = Self;
    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, o: f32) -> Self {
        debug_assert_ne!(o, 0.0);
        let inv = 1.0 / o;
        self * inv
    }
}

impl Div<V3> for f32 {
    type Output = V3;
    #[inline]
    fn div(self, v: V3) -> V3 {
        // debug_assert!(self.x != 0.0);
        // debug_assert!(self.y != 0.0);
        // debug_assert!(self.z != 0.0);
        // debug_assert!(self.w != 0.0);
        V3 {
            x: self / v.x,
            y: self / v.y,
            z: self / v.z,
            w: 0.0,
        }
    }
}

impl MulAssign for V3 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        self.x *= rhs.x;
        self.y *= rhs.y;
        self.z *= rhs.z;
        self.w = 0.0;
    }
}

impl DivAssign for V3 {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        self.x /= rhs.x;
        self.y /= rhs.y;
        self.z /= rhs.z;
        // self.w /= rhs.w;
    }
}

impl MulAssign<f32> for V3 {
    #[inline]
    fn mul_assign(&mut self, rhs: f32) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
        self.w = 0.0;
    }
}

impl DivAssign<f32> for V3 {
    #[inline]
    fn div_assign(&mut self, rhs: f32) {
        let inv = 1.0 / rhs;
        self.x *= inv;
        self.y *= inv;
        self.z *= inv;
        self.w = 0.0;
        // $(self.$field *= inv;)+
    }
}

impl AddAssign for V3 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
        self.w = 0.0;
        // $(self.$field += rhs.$field;)+
    }
}

impl SubAssign for V3 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
        self.w = 0.0;
        // $(self.$field -= rhs.$field;)+
    }
}

impl V3 {
    #[inline(always)]
    pub fn x(&self) -> f32 {
        self.x
    }
    #[inline(always)]
    pub fn y(&self) -> f32 {
        self.y
    }
    #[inline(always)]
    pub fn z(&self) -> f32 {
        self.z
    }

    #[inline(always)]
    pub fn mx(&mut self) -> &mut f32 {
        &mut self.x
    }
    #[inline(always)]
    pub fn my(&mut self) -> &mut f32 {
        &mut self.y
    }
    #[inline(always)]
    pub fn mz(&mut self) -> &mut f32 {
        &mut self.z
    }

    #[inline(always)]
    pub fn set_x(&mut self, v: f32) {
        self.x = v;
    }

    #[inline(always)]
    pub fn set_y(&mut self, v: f32) {
        self.y = v;
    }

    #[inline(always)]
    pub fn set_z(&mut self, v: f32) {
        self.z = v;
    }

    #[inline(always)]
    pub fn with_x(mut self, v: f32) -> V3 {
        self.x = v;
        self
    }

    #[inline(always)]
    pub fn with_y(mut self, v: f32) -> V3 {
        self.y = v;
        self
    }

    #[inline(always)]
    pub fn with_z(mut self, v: f32) -> V3 {
        self.z = v;
        self
    }

    pub const ZERO: V3 = V3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 0.0,
    };

    #[inline]
    pub const fn zero() -> Self {
        Self::ZERO
    }

    #[inline]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z, w: 0.0 }
    }

    #[inline]
    pub const fn splat(v: f32) -> Self {
        Self {
            x: v,
            y: v,
            z: v,
            w: 0.0,
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
        &self.x as *const f32
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        &mut self.x as *mut f32
    }

    #[inline]
    pub fn as_array(&self) -> &[f32; 3] {
        unsafe { &*(self as *const V3 as *const [f32; 3]) }
    }

    #[inline]
    pub fn as_mut_array(&mut self) -> &mut [f32; 3] {
        unsafe { &mut *(self as *mut V3 as *mut [f32; 3]) }
    }

    #[inline]
    pub fn as_array4(&self) -> &[f32; 4] {
        unsafe { &*(self as *const V3 as *const [f32; 4]) }
    }

    #[inline]
    pub fn as_mut_array4(&mut self) -> &mut [f32; 4] {
        unsafe { &mut *(self as *mut V3 as *mut [f32; 4]) }
    }

    #[inline]
    pub fn tup(self) -> (f32, f32, f32) {
        self.into()
    }
    #[inline]
    pub fn arr(self) -> [f32; 3] {
        self.into()
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
        self.fold(|a, b| a.max(b))
    }
    #[inline]
    pub fn min_elem(self) -> f32 {
        self.fold(|a, b| a.min(b))
    }

    #[inline]
    pub fn abs(self) -> Self {
        self.map(|x| x.abs())
    }

    #[inline]
    pub fn floor(self) -> Self {
        self.map(|x| x.floor())
    }
    #[inline]
    pub fn ceil(self) -> Self {
        self.map(|x| x.ceil())
    }
    #[inline]
    pub fn round(self) -> Self {
        self.map(|x| x.round())
    }

    #[inline]
    pub fn min(self, o: Self) -> Self {
        self.map2(o, |a, b| a.min(b))
    }
    #[inline]
    pub fn max(self, o: Self) -> Self {
        self.map2(o, |a, b| a.max(b))
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
        self.map3(min, max, clamp)
    }

    #[inline]
    pub fn lerp(self, b: Self, t: f32) -> Self {
        self.map2(b, |x, y| x.lerp(y, t))
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
        self.dot(self) == 0.0_f32
    }

    #[inline]
    pub fn hadamard(self, o: Self) -> Self {
        self.map2(o, |a, b| a * b)
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
        self.x() != 0.0 && self.y() != 0.0 && self.z() != 0.0
    }

    #[inline]
    pub fn inverse(self) -> Self {
        debug_assert!(self.no_zeroes());
        vec3(1.0 / self.x, 1.0 / self.y, 1.0 / self.z)
    }

    #[inline]
    pub fn norm_or(self, x: f32, y: f32, z: f32) -> Self {
        self.norm_or_v(Self { x, y, z, w: 0.0 })
    }
    #[inline]
    pub fn dot3(self, a: V3, b: V3, c: V3) -> V3 {
        simd_match! {
            "sse2" => {
                crate::math::simd::dot3(self, a, b, c)
            },
            _ => {
                self.naive_dot3(a, b, c)
            }
        }
    }

    #[inline]
    #[allow(dead_code)]
    pub(crate) fn naive_dot3(self, a: V3, b: V3, c: V3) -> V3 {
        let va = self.x * a.x + self.y * a.y + self.z * a.z;
        let vb = self.x * b.x + self.y * b.y + self.z * b.z;
        let vc = self.x * c.x + self.y * c.y + self.z * c.z;
        vec3(va, vb, vc)
    }
    // #[inline]
    // pub fn identity() -> Self {
    //     V3 { $($field: 0.0),+ }
    // }

    // #[inline]
    // pub fn expand(v: V2, z: f32) -> V3 {
    //     vec3(v.x, v.y, z)
    // }

    #[inline]
    pub fn dot(self, o: V3) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    #[inline]
    pub fn outer_prod(self, o: V3) -> M3x3 {
        M3x3 {
            x: self * o.x,
            y: self * o.y,
            z: self * o.z,
        }
    }

    #[allow(dead_code)]
    #[inline]
    pub(crate) fn naive_cross(self: V3, b: V3) -> V3 {
        vec3(
            self.y * b.z - self.z * b.y,
            self.z * b.x - self.x * b.z,
            self.x * b.y - self.y * b.x,
        )
    }

    #[inline]
    pub fn cross(self, b: V3) -> V3 {
        simd_match! {
            "sse2" => {
                crate::math::simd::v3_cross(self, b)
            },
            _ => {
                self.naive_cross(b)
            }
        }
    }

    #[inline]
    pub fn orth(self) -> V3 {
        let abs_v = self.abs();
        let mut u = V3::splat(1.0);
        u[abs_v.max_index()] = 0.0;
        // let u = u;
        u.cross(self).norm_or(1.0, 0.0, 0.0)
    }

    #[inline]
    pub fn basis(self) -> (V3, V3, V3) {
        let a = self.norm_or(1.0, 0.0, 0.0);
        let bu = if self.x.abs() > 0.57735 {
            // sqrt(1/3)
            vec3(self.y, -self.x, 0.0)
        } else {
            vec3(0.0, self.z, -self.y)
        };
        // should never need normalizing, but there may be degenerate cases...
        let b = bu.norm_or(0.0, -1.0, 0.0);
        let c = a.cross(b);
        (a, b, c)
    }

    #[inline]
    pub fn norm_or_unit(self) -> V3 {
        self.norm_or(0.0, 0.0, 1.0)
    }
    #[inline]
    pub fn to_arr(self) -> [f32; 3] {
        self.into()
    }

    #[inline]
    #[allow(clippy::collapsible_if)]
    #[rustfmt::skip]
    pub fn max_index(&self) -> usize {
        if self.x > self.y {
            if self.x > self.z { 0 } else { 2 }
        } else {
            if self.y > self.z { 1 } else { 2 }
        }
    }

    #[inline]
    #[allow(clippy::collapsible_if)]
    #[rustfmt::skip]
    pub fn min_index(&self) -> usize {
        if self.x < self.y {
            if self.x < self.z { 0 } else { 2 }
        } else {
            if self.y < self.z { 1 } else { 2 }
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

impl Map for V3 {
    #[inline]
    fn map3<F: Fn(f32, f32, f32) -> f32>(self, a: Self, b: Self, f: F) -> Self {
        Self {
            x: f(self.x, a.x, b.x),
            y: f(self.y, a.y, b.y),
            z: f(self.z, a.z, b.z),
            w: 0.0,
        }
    }
}

impl VecType for V3 {
    const SIZE: usize = 3;

    #[inline]
    fn splat(v: f32) -> Self {
        V3::splat(v)
    }
    #[inline]
    fn dot(self, o: Self) -> f32 {
        V3::dot(self, o)
    }
}

impl std::fmt::Debug for V3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("vec3")
            .field(&self.x)
            .field(&self.y)
            .field(&self.z)
            .finish()
    }
}

impl PartialEq for V3 {
    #[inline]
    fn eq(&self, o: &V3) -> bool {
        self.x == o.x && self.y == o.y && self.z == o.z
    }
}
impl From<V2> for V3 {
    #[inline]
    fn from(v: V2) -> V3 {
        V3 {
            x: v.x,
            y: v.y,
            z: 0.0,
            w: 0.0,
        }
    }
}

impl From<V4> for V3 {
    #[inline]
    fn from(v: V4) -> V3 {
        V3 {
            x: v.x,
            y: v.y,
            z: v.z,
            w: 0.0,
        }
    }
}

impl IntoIterator for V3 {
    type Item = f32;
    type IntoIter = VecIter<f32>;
    #[inline]
    fn into_iter(self) -> VecIter<f32> {
        VecIter::new([self.x, self.y, self.z, 0.0], 3)
    }
}
impl Fold for V3 {
    #[inline]
    fn fold(self, f: impl Fn(f32, f32) -> f32) -> f32 {
        f(f(self.x, self.y), self.z)
    }

    #[inline]
    fn fold2_init<T>(self, o: Self, init: T, f: impl Fn(T, f32, f32) -> T) -> T {
        f(f(f(init, o.x, self.x), o.y, self.y), o.z, self.z)
    }
}

impl fmt::Display for V3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "vec3({}, {}, {})", self.x, self.y, self.z)
    }
}

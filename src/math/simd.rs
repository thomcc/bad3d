
use std::arch::x86_64::{self, __m128, __m128i};
use super::{quat::Quat, vec::{V3, V4}};

static_assert_usize_eq!(QUAT_SIMD_ALIGN, std::mem::align_of::<Quat>(), std::mem::align_of::<__m128>());
static_assert_usize_eq!(V3_SIMD_ALIGN, std::mem::align_of::<V3>(), std::mem::align_of::<__m128>());
static_assert_usize_eq!(V4_SIMD_ALIGN, std::mem::align_of::<V4>(), std::mem::align_of::<__m128>());

impl Quat {
    #[inline(always)]
    fn as_x86(&self) -> &__m128 {
        unsafe { &*(self as *const Quat as *const __m128) }
    }
}
impl V4 {
    #[inline(always)]
    fn as_x86(&self) -> &__m128 {
        unsafe { &*(self as *const V4 as *const __m128) }
    }
}

// #[inline]
// pub fn quat_mul_v3(q: &super::quat::Quat, )




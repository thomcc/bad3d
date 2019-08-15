use super::prelude::*;
use crate::util::{Align16, ConstTransmuter};
use std::arch::x86_64::{self as sse, __m128};

static_assert_usize_eq!(
    QUAT_SIMD_ALIGN,
    std::mem::align_of::<Quat>(),
    std::mem::align_of::<__m128>()
);
static_assert_usize_eq!(
    V3_SIMD_ALIGN,
    std::mem::align_of::<V3>(),
    std::mem::align_of::<__m128>()
);
static_assert_usize_eq!(
    V4_SIMD_ALIGN,
    std::mem::align_of::<V4>(),
    std::mem::align_of::<__m128>()
);

static_assert_usize_eq!(
    QUAT_SIMD_SIZE,
    std::mem::size_of::<Quat>(),
    std::mem::size_of::<__m128>()
);
static_assert_usize_eq!(V3_SIMD_SIZE, std::mem::size_of::<V3>(), std::mem::size_of::<__m128>());
static_assert_usize_eq!(V4_SIMD_SIZE, std::mem::size_of::<V4>(), std::mem::size_of::<__m128>());

impl Quat {
    #[inline(always)]
    fn as_x86(&self) -> &__m128 {
        unsafe { &*(self as *const Quat as *const __m128) }
    }
    #[inline(always)]
    fn into_x86(self) -> __m128 {
        *self.as_x86()
    }

    #[inline(always)]
    fn from_x86(m: __m128) -> Self {
        unsafe { *(&m as *const __m128 as *const Quat) }
    }
}

impl V3 {
    #[inline(always)]
    fn into_x86(self) -> __m128 {
        self.0
    }
    #[inline(always)]
    fn from_x86(m: __m128) -> Self {
        Self(m)
    }
}

macro_rules! shuf {
    ($A:expr, $B:expr, $C:expr, $D:expr) => {
        (($D << 6) | ($C << 4) | ($B << 2) | $A) & 0xff
    };
}
const SIMD_W_SIGNMASK: __m128 = unsafe {
    ConstTransmuter::<Align16<[u32; 4]>, __m128> {
        from: Align16([0, 0, 0, 0x8000_0000u32]),
    }
    .to
};
// const SIMD_XYZ_SIGNMASK: __m128 = unsafe {
//     Transmuter::<Align16<[u32; 4]>, __m128> {
//         from: Align16([0x8000_0000u32, 0x8000_0000u32, 0x8000_0000u32, 0]),
//     }
//     .to
// };

#[inline]
unsafe fn quatmul3(q: __m128, v: __m128) -> __m128 {
    //  qw * vx + qy * vz - qz * vy;
    //  qw * vy + qz * vx - qx * vz;
    //  qw * vz + qx * vy - qy * vx;
    // -qx * vx - qy * vy - qz * vz;
    //  \--a--/   \--b--/   \--c--/

    let q_wwwx = sse::_mm_shuffle_ps(q, q, shuf![3, 3, 3, 0]);
    let v_xyzx = sse::_mm_shuffle_ps(v, v, shuf![0, 1, 2, 0]);
    let a = sse::_mm_mul_ps(q_wwwx, v_xyzx);

    let q_yzxy = sse::_mm_shuffle_ps(q, q, shuf![1, 2, 0, 1]);
    let v_zxyy = sse::_mm_shuffle_ps(v, v, shuf![2, 0, 1, 1]);
    let b = sse::_mm_mul_ps(q_yzxy, v_zxyy);

    let q_zxyz = sse::_mm_shuffle_ps(q, q, shuf![2, 0, 1, 2]);
    let v_yzxz = sse::_mm_shuffle_ps(v, v, shuf![1, 2, 0, 2]);
    let c = sse::_mm_mul_ps(q_zxyz, v_yzxz);

    // assemble them so we don't need to negate too much
    let ab = sse::_mm_add_ps(a, b);
    let ab_w = sse::_mm_xor_ps(ab, SIMD_W_SIGNMASK);

    sse::_mm_sub_ps(ab_w, c)
}
#[inline]
unsafe fn quatmulq(q0: __m128, q1: __m128) -> __m128 {
    // similar to above, same a, b, and c
    // x0 * w1 + w0 * x1 + y0 * z1 - z0 * y1,
    // y0 * w1 + w0 * y1 + z0 * x1 - x0 * z1,
    // z0 * w1 + w0 * z1 + x0 * y1 - y0 * x1,
    // w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
    // \--d--/   \--a--/   \--b--/   \--c--/

    let xyzw_0 = q0;
    let wwww_1 = sse::_mm_shuffle_ps(q1, q1, shuf![3, 3, 3, 3]);
    let d = sse::_mm_mul_ps(xyzw_0, wwww_1);

    let wwwx_0 = sse::_mm_shuffle_ps(q0, q0, shuf![3, 3, 3, 0]);
    let xyzx_1 = sse::_mm_shuffle_ps(q1, q1, shuf![0, 1, 2, 0]);
    let a = sse::_mm_mul_ps(wwwx_0, xyzx_1);

    let yzxy_0 = sse::_mm_shuffle_ps(q0, q0, shuf![1, 2, 0, 1]);
    let zxyy_1 = sse::_mm_shuffle_ps(q1, q1, shuf![2, 0, 1, 1]);
    let b = sse::_mm_mul_ps(yzxy_0, zxyy_1);

    let zxyz_0 = sse::_mm_shuffle_ps(q0, q0, shuf![2, 0, 1, 2]);
    let yzxz_1 = sse::_mm_shuffle_ps(q1, q1, shuf![1, 2, 0, 2]);
    let c = sse::_mm_mul_ps(zxyz_0, yzxz_1);
    // assemble them so we don't need to negate too much
    let ab = sse::_mm_add_ps(a, b);
    let cd = sse::_mm_sub_ps(d, c);

    let ab = sse::_mm_xor_ps(ab, SIMD_W_SIGNMASK);
    sse::_mm_add_ps(cd, ab)
}

#[inline]
unsafe fn quatrot3(q: __m128, v: __m128) -> __m128 {
    // let ix =  qw * vx + qy * vz - qz * vy;
    // let iy =  qw * vy + qz * vx - qx * vz;
    // let iz =  qw * vz + qx * vy - qy * vx;
    // let iw = -qx * vx - qy * vy - qz * vz;
    //           \--a--/   \--b--/   \--c--/
    //
    // ix * qw - iw * qx - iy * qz + iz * qy,
    // iy * qw - iw * qy - iz * qx + ix * qz,
    // iz * qw - iw * qz - ix * qy + iy * qx,
    // \--d--/   \--e--/   \--f--/   \--g--/
    let i = quatmul3(q, v);

    let q_wwww = sse::_mm_shuffle_ps(q, q, shuf![3, 3, 3, 3]);
    let d = sse::_mm_mul_ps(i, q_wwww);

    let i_wwww = sse::_mm_shuffle_ps(i, i, shuf![3, 3, 3, 3]);
    let e = sse::_mm_mul_ps(i_wwww, q);

    let i_yzxw = sse::_mm_shuffle_ps(i, i, shuf![1, 2, 0, 3]);
    let q_zxyw = sse::_mm_shuffle_ps(q, q, shuf![2, 0, 1, 3]);
    let f = sse::_mm_mul_ps(i_yzxw, q_zxyw);

    let i_zxyw = sse::_mm_shuffle_ps(i, i, shuf![2, 0, 1, 3]);
    let q_yzxw = sse::_mm_shuffle_ps(q, q, shuf![1, 2, 0, 3]);
    let g = sse::_mm_mul_ps(i_zxyw, q_yzxw);

    let dg = sse::_mm_add_ps(d, g);
    let ef = sse::_mm_add_ps(f, e);
    let defg = sse::_mm_sub_ps(dg, ef);

    clear_w(defg)
}

const CLEARW_MASK: __m128 = unsafe {
    ConstTransmuter::<Align16<[u32; 4]>, __m128> {
        from: Align16([0xffff_ffff, 0xffff_ffff, 0xffff_ffff, 0]),
    }
    .to
};
#[inline(always)]
unsafe fn clear_w(v: __m128) -> __m128 {
    sse::_mm_and_ps(v, CLEARW_MASK)
}

#[inline]
pub fn quat_rot3(rot: Quat, v: V3) -> V3 {
    unsafe {
        let rot = rot.into_x86();
        let v = v.0;
        V3(quatrot3(rot, v))
    }
}

#[inline]
pub fn quat_mul_quat(a: Quat, b: Quat) -> Quat {
    unsafe { Quat::from_x86(quatmulq(a.into_x86(), b.into_x86())) }
}

#[inline]
pub fn v3_cross(a: V3, b: V3) -> V3 {
    unsafe {
        let a = a.into_x86();
        let b = b.into_x86();
        let a_yzx = sse::_mm_shuffle_ps(a, a, shuf![1, 2, 0, 3]);
        let b_yzx = sse::_mm_shuffle_ps(b, b, shuf![1, 2, 0, 3]);

        let l = sse::_mm_mul_ps(b_yzx, a);
        let r = sse::_mm_mul_ps(a_yzx, b);
        let v = sse::_mm_sub_ps(l, r);
        V3::from_x86(sse::_mm_shuffle_ps(v, v, shuf![1, 2, 0, 3]))
    }
}

#[inline(always)]
unsafe fn zwxy(v0: __m128, v1: __m128) -> __m128 {
    sse::_mm_castpd_ps(sse::_mm_move_sd(sse::_mm_castps_pd(v0), sse::_mm_castps_pd(v1)))
}

#[inline]
unsafe fn do_dot3(v: __m128, a: __m128, b: __m128, c: __m128) -> __m128 {
    let va = sse::_mm_mul_ps(v, a);
    let vb = sse::_mm_mul_ps(v, b);
    let vc = sse::_mm_mul_ps(v, c);

    let abl = sse::_mm_unpacklo_ps(va, vb);
    let abh = sse::_mm_unpackhi_ps(va, vb);
    let vc0 = sse::_mm_unpacklo_ps(vc, sse::_mm_setzero_ps());

    let hsum = sse::_mm_movelh_ps(abl, vc0);
    let hsum = sse::_mm_add_ps(hsum, sse::_mm_movehl_ps(vc0, abl));

    let vc = clear_w(vc);
    sse::_mm_add_ps(hsum, zwxy(vc, abh))
}

pub fn dot3(v: V3, a: V3, b: V3, c: V3) -> V3 {
    unsafe { V3(do_dot3(v.0, a.0, b.0, c.0)) }
}

pub fn maxdot_i(v: V3, vs: &[V3]) -> usize {
    unsafe {
        assert!(vs.len() != 0);
        let mut i = 0;
        // We have an assert above that both the size and alignment match.
        let mut best = sse::_mm_set1_ps(std::f32::MIN);
        let mut best4 = sse::_mm_set1_ps(std::f32::MIN);
        let dir = v.0;

        let d_xyxy = sse::_mm_movelh_ps(dir, dir);
        let d_zzzz = sse::_mm_shuffle_ps(dir, dir, shuf![2, 2, 2, 2]);
        let mut best_start: usize = 0;
        let mut sp = vs.as_ptr() as *const __m128;
        let ep = sp.add(vs.len());
        while i + 3 < vs.len() {
            debug_assert!((sp as usize + 3) < (ep as usize));

            let p0 = *sp.add(0);
            let p1 = *sp.add(1);
            let p2 = *sp.add(2);
            let p3 = *sp.add(3);

            // Compute 4 dot products at a time, without w
            let l0 = sse::_mm_movelh_ps(p0, p1);
            let h0 = sse::_mm_movehl_ps(p1, p0);

            let l1 = sse::_mm_movelh_ps(p2, p3);
            let h1 = sse::_mm_movehl_ps(p3, p2);

            let l0 = sse::_mm_mul_ps(l0, d_xyxy);
            let l1 = sse::_mm_mul_ps(l1, d_xyxy);

            let z = sse::_mm_shuffle_ps(h0, h1, shuf![0, 2, 0, 2]);
            let x = sse::_mm_shuffle_ps(l0, l1, shuf![0, 2, 0, 2]);
            let y = sse::_mm_shuffle_ps(l0, l1, shuf![1, 3, 1, 3]);

            let z = sse::_mm_mul_ps(z, d_zzzz);
            let x = sse::_mm_add_ps(x, y);
            let dots = sse::_mm_add_ps(x, z);
            // compare the result against `best`, which has best seen dot
            // product in all 4 lanes.
            let gti = sse::_mm_cmpgt_ps(dots, best);
            if sse::_mm_movemask_ps(gti) != 0 {
                // Just mark the start index, we'll sort it out later.
                best4 = dots;
                best_start = i;
                best = sse::_mm_max_ps(best, dots);
                // expand the new maximum to all 4 lanes of `best`.
                best = sse::_mm_max_ps(best, sse::_mm_shuffle_ps(best, best, shuf![1, 0, 3, 2]));
                best = sse::_mm_max_ps(best, sse::_mm_shuffle_ps(best, best, shuf![2, 3, 0, 1]));
            }
            sp = sp.add(4);
            i += 4;
        }
        if i != vs.len() {
            let (x, y, z) = match vs.len() - i {
                1 => {
                    debug_assert!((sp as usize + 1) == (ep as usize));
                    let xy = *sp;
                    let z = sse::_mm_shuffle_ps(xy, xy, shuf![2, 2, 2, 2]);
                    let xy = sse::_mm_mul_ps(xy, d_xyxy);
                    let z = sse::_mm_mul_ps(z, d_zzzz);
                    let x = sse::_mm_shuffle_ps(xy, xy, shuf![0, 0, 0, 0]);
                    let y = sse::_mm_shuffle_ps(xy, xy, shuf![1, 1, 1, 1]);
                    (x, y, z)
                }
                2 => {
                    debug_assert!((sp as usize + 2) == (ep as usize));
                    let v0 = *sp;
                    let v1 = *sp.add(1);
                    let xy = sse::_mm_movelh_ps(v0, v1);
                    let z = sse::_mm_movehl_ps(v1, v0);
                    let xy = sse::_mm_mul_ps(xy, d_xyxy);
                    let z = sse::_mm_shuffle_ps(z, z, shuf![0, 2, 2, 2]);
                    let x = sse::_mm_shuffle_ps(xy, xy, shuf![0, 2, 2, 2]);
                    let y = sse::_mm_shuffle_ps(xy, xy, shuf![1, 3, 3, 3]);
                    let z = sse::_mm_mul_ps(z, d_zzzz);
                    (x, y, z)
                }
                3 => {
                    debug_assert!((sp as usize + 3) == (ep as usize));
                    let v0 = *sp;
                    let v1 = *sp.add(1);
                    let v2 = *sp.add(2);
                    // v2 gets repeated but ignored, we basically do the 4-wide
                    // dot loop from above but with v0122
                    let l0 = sse::_mm_movelh_ps(v0, v1);
                    let h0 = sse::_mm_movehl_ps(v1, v0);

                    let l0 = sse::_mm_mul_ps(l0, d_xyxy);
                    let z = sse::_mm_shuffle_ps(h0, v2, shuf![0, 2, 2, 2]);
                    let z = sse::_mm_mul_ps(z, d_zzzz);

                    let l1 = sse::_mm_movelh_ps(v2, v2);
                    let l1 = sse::_mm_mul_ps(l1, d_xyxy);

                    let x = sse::_mm_shuffle_ps(l0, l1, shuf![0, 2, 0, 2]);
                    let y = sse::_mm_shuffle_ps(l0, l1, shuf![1, 3, 1, 3]);
                    (x, y, z)
                }
                _ => {
                    unreachable!();
                }
            };
            let dot = sse::_mm_add_ps(x, y);
            let dot = sse::_mm_add_ps(dot, z);
            let gti = sse::_mm_cmpgt_ps(dot, best);
            if sse::_mm_movemask_ps(gti) != 0 {
                // Just mark the start index, we'll sort it out later.
                best_start = i;
                // expand the new maximum to all 4 lanes of `best`.
                best4 = dot;
                best = sse::_mm_max_ps(best, dot);
                best = sse::_mm_max_ps(best, sse::_mm_shuffle_ps(best, best, shuf![1, 0, 3, 2]));
                best = sse::_mm_max_ps(best, sse::_mm_shuffle_ps(best, best, shuf![2, 3, 0, 1]));
            }
        }
        // we need to see which index `best` is in `best4` to see how much we
        // should add to `best_start` to return. we compute the mask, and then
        // use that mask to index intota lookup table that says which value to
        // use
        let mask = sse::_mm_movemask_ps(sse::_mm_cmpeq_ps(best, best4)) & 0xf;
        debug_assert_ne!(mask, 0);
        const MASK_TO_FIRST_INDEX: [u8; 16] = [0, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0];
        best_start + (MASK_TO_FIRST_INDEX[mask as usize] as usize)
    }
}
#[inline]
pub fn maxdot(v: V3, vs: &[V3]) -> V3 {
    let i = maxdot_i(v, vs);
    vs[i]
}
#[cfg(test)]
mod test {
    use super::*;
    fn naive_maxdot(dir: V3, s: &[V3]) -> V3 {
        // geom::max_dir_iter(dir, s.iter().copied());
        let mut best = s[0];
        let mut best_dot = best.x() * dir.x() + best.y() * dir.y() + best.z() * dir.z();
        for &v in s[1..].iter() {
            let new_dot = v.x() * dir.x() + v.y() * dir.y() + v.z() * dir.z();
            if new_dot > best_dot {
                best = v;
                best_dot = new_dot;
            }
        }
        best
    }
    quickcheck::quickcheck! {
        fn prop_maxdot_eq_naive(dir: V3, s: Vec<V3>) -> bool {
            s.is_empty() || (naive_maxdot(dir, &s) == maxdot(dir, &s))
        }
        fn prop_dot_eq_naive(a: V3, b: V3) -> bool {
            v3_cross(a, b) == a.naive_cross(b)
        }
        fn prop_qrot_eq_naive(a: Quat, b: V3) -> bool {
            quat_rot3(a, b).approx_eq(&super::super::quat::naive_quat_rot3(a, b))
        }
        fn prop_qmul_eq_naive(a: Quat, b: Quat) -> bool {
            quat_mul_quat(a, b).approx_eq(&super::super::quat::naive_quat_mul_quat(a, b))
        }
        fn prop_dot3_eq_naive(a: V3, b: V3, c: V3, d: V3) -> bool {
            dot3(a, b, c, d).approx_eq(&V3::naive_dot3(a, b, c, d))
        }
    }

    #[test]
    #[allow(clippy::unreadable_literal)]
    fn quickcheck_regressions() {
        let a = Quat::new(0.053105693, -0.5909659, -0.80490583, 0.008106291);
        let b = Quat::new(-0.066779576, 0.46287408, 0.883806, 0.013232946);
        let r0 = super::super::quat::naive_quat_mul_quat(a, b);
        let r1 = quat_mul_quat(a, b);
        assert!(r0.approx_eq(&r1), "{}, {}", r0, r1);

        let a = vec3(0.0, 0.0, 1.0);
        let b = vec3(0.0, 0.0, 0.0);
        let c = vec3(0.0, 0.0, 0.0);
        let d = vec3(0.0, 0.0, 1.0);
        let r0 = V3::naive_dot3(a, b, c, d);
        let r1 = dot3(a, b, c, d);
        assert!(r0.approx_eq(&r1), "{}, {}", r0, r1);
        let d0 = naive_maxdot(vec3(0.0, 0.0, 1.0), &[vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0)]);
        let d1 = super::maxdot(vec3(0.0, 0.0, 1.0), &[vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0)]);

        assert_eq!(d0, d1);
        assert_eq!(
            super::maxdot_i(vec3(0.0, 0.0, 1.0), &[vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0)]),
            1
        );
        let ds = vec3(42.0, 55.417053, -40.881634);
        let dv = &[vec3(-46.167755, -98.00792, 64.66603)];
        assert_eq!(naive_maxdot(ds, dv), super::maxdot(ds, dv));
    }

}

/// Allows
/// ```
/// # #[allow(dead_code)] fn code_requiring_fma_and_avx() {}
/// # #[allow(dead_code)] fn code_requiring_sse41() {}
/// # #[allow(dead_code)] fn code_requiring_sse2() {}
/// # #[allow(dead_code)] fn fallback_code() {}
/// # fn main() {
/// let thing = t3m::simd_match! {
///     // Comma separate required target features to require both.
///     "fma", "avx" => code_requiring_fma_and_avx(),
///     "sse4.1" => code_requiring_sse41(),
///     "sse2" => code_requiring_sse2(),
///     // If a default case is ommitted, we'll emit a
///     // compile error when compiling for a platform
///     // that doesn't match something above.
///     _ => fallback_code(),
/// };
/// # let _ = thing;
/// # }
/// ```
/// Note that (like match) these must be in MOST to LEAST specific. Specifying
/// `sse` before `avx` will mean the AVX one will never be used! I'd like to fix
/// this, but am unsure how.
#[macro_export(local_inner_macros)]
macro_rules! simd_match {
    ($($feat:literal),+ => $it:expr, $($rest:tt)*) => {{
        // Create a dummy variable that the bodies initialize.
        // and allows the match arms to be exprs without rustc
        // complaining (even fabricating a block inside the macro
        // hits an error)
        let _simd_match_val;
        simd_match! {
            @__scan;
            _simd_match_val;
            ();
            $($feat),+ => $it,
            $($rest)*
        };
        _simd_match_val
    }};
    (@__scan;
        $target:ident ;
        ($($nots:meta),* $(,)?) ;
        $($feat:literal),+ => $block:expr,
        $($rest:tt)*
    ) => {
        simd_match!{
            @__scan;
            $target;
            (all($(target_feature = $feat),+), $($nots),*);
            $($rest)*
        };
        #[cfg(all(
            not(any($($nots),*)),
            $(target_feature = $feat),+
        ))] { $target = $block; }
    };
    // hacky duplicated case to make trailing comma optional here...
    // can't use $(,)? before a $($rest:tt)*, and can't leave
    // the comma out or we'd have an `expr` followed by a tt, which
    // is invalid.
    (@__scan;
        $target:ident ;
        ($($nots:meta),* $(,)?) ;
        $($feat:literal),+ => $block:expr
    ) => {
        // add the comma and recurse.
        simd_match!{
            @__scan;
            $target ;
            (all($(target_feature = $feat),+), $($nots),*) ;
            $($feat),+ => $block,
        }
    };
    (@__scan; $target:ident; ($($nots:meta),* $(,)?) ; $(,)*) => {
        #[cfg(not(any($($nots),*)))] {
            compile_error!(
                concat!(
                    "Unhandled case in simd_match! Expected one of the following cfg's to match: ",
                    stringify!($($nots),+)
                )
            );
        }
    };
    (@__scan;
        $target:ident;
        ($($nots:meta),* $(,)?) ;
        _ => $fallback:expr $(,)?
    ) => {
        #[cfg(not(any($($nots),*)))] { $target = $fallback; };
    };
}

/// Construct a swizzle mask
#[macro_export]
macro_rules! shuf {
    ($A:expr, $B:expr, $C:expr, $D:expr) => {
        (($D << 6) | ($C << 4) | ($B << 2) | $A) & 0xff
    };
}

const _ALIGN_SANITY_CHECK_V3: [(); 16] = [(); ::core::mem::align_of::<crate::vec::V3>()];
const _SIZE_SANITY_CHECK_V3: [(); 16] = [(); ::core::mem::size_of::<crate::vec::V3>()];
const _ALIGN_SANITY_CHECK_V4: [(); 16] = [(); ::core::mem::align_of::<crate::vec::V4>()];
const _SIZE_SANITY_CHECK_V4: [(); 16] = [(); ::core::mem::size_of::<crate::vec::V4>()];

cfg_if::cfg_if! {
    if #[cfg(target_feature = "sse2")] {

        const _SIZE_CHECK_V4: [(); ::core::mem::size_of::<::core::arch::x86_64::__m128>()] =
            [(); ::core::mem::size_of::<crate::vec::V4>()];

        const _SIZE_CHECK_V3: [(); ::core::mem::size_of::<::core::arch::x86_64::__m128>()] =
            [(); ::core::mem::size_of::<crate::vec::V3>()];

        const _ALIGN_CHECK_V3: [(); ::core::mem::align_of::<::core::arch::x86_64::__m128>()] =
            [(); ::core::mem::align_of::<crate::vec::V3>()];
        const _ALIGN_CHECK_V4: [(); ::core::mem::align_of::<::core::arch::x86_64::__m128>()] =
            [(); ::core::mem::align_of::<crate::vec::V4>()];

        /// Due to the limitations of SIMD in rust, you can't create a constant
        /// V4. This can be used as a workaround.
        /// ```
        /// const EXAMPLE: t3m::V4 = t3m::vec4_const![1.0, 2.0, 3.0, 4.0];
        /// println!("{}", EXAMPLE);
        /// ```
        #[macro_export]
        macro_rules! vec4_const {
            ($x:expr; 4) => {
                vec4_const![$x, $x, $x, $x]
            };
            ($x:expr, $y:expr, $z:expr, $w:expr) => {{
                const ARR: [f32; 4] = [$x, $y, $z, $w];
                const VV: $crate::vec::V4 = unsafe {
                    $crate::util::ConstTransmuter::<[f32; 4], $crate::vec::V4> { from: ARR }.to
                };
                VV
            }};
        }

        /// Due to the limitations of SIMD in rust, you can't create a constant
        /// V3. This can be used as a workaround.
        /// ```
        /// const GRAVITY: t3m::V3 = t3m::vec3_const![0.0, 0.0, -9.8];
        /// println!("{:?}", GRAVITY);
        /// ```
        #[macro_export]
        macro_rules! vec3_const {
            ($x:expr; 3) => {
                vec3_const![$x, $x, $x]
            };
            ($x:expr, $y:expr, $z:expr) => {{
                const ARR: [f32; 4] = [$x, $y, $z, 0.0];
                const VV: $crate::vec::V3 = unsafe {
                    $crate::util::ConstTransmuter::<[f32; 4], $crate::vec::V3> { from: ARR }.to
                };
                VV
            }};
        }

        #[cfg(target_feature = "sse2")]
        #[macro_export]
        macro_rules! const_simd_mask {
            ($x:expr; 4) => {
                const_simd_mask![$x, $x, $x, $x]
            };
            ($x:expr; 3) => {
                const_simd_mask![$x, $x, $x, 0u32]
            };
            ($x:expr, $y:expr, $z:expr, $w:expr) => {{
                const MASKARR: [u32; 4] = [$x, $y, $z, $w];
                const MASK: ::std::arch::x86_64::__m128 = unsafe {
                    $crate::util::ConstTransmuter::<[u32; 4], ::std::arch::x86_64::__m128> {
                        from: MASKARR
                    }.to
                };
                MASK
            }};
        }
    } else {
        /// Due to the limitations of SIMD in rust, you can't create a constant
        /// V3. This can be used as a workaround.
        /// ```
        /// const GRAVITY: t3m::V3 = t3m::vec3_const![0.0, 0.0, -9.8];
        /// println!("{:?}", GRAVITY);
        /// ```
        #[macro_export]
        macro_rules! vec3_const {
            ($x:expr; 3) => {
                vec3_const![$x, $x, $x]
            };
            ($x:expr, $y:expr, $z:expr) => {{
                const VV: V3 = $crate::vec::__v3_const($x, $y, $z);
                VV
            }};
        }
        /// Due to the limitations of SIMD in rust, you can't create a constant
        /// V4. This can be used as a workaround.
        /// ```
        /// const EXAMPLE: t3m::V4 = t3m::vec4_const![1.0, 2.0, 3.0, 4.0];
        /// println!("{}", EXAMPLE);
        /// ```
        #[macro_export]
        macro_rules! vec4_const {
            ($x:expr; 4) => {
                vec4_const![$x, $x, $x, $x]
            };
            ($x:expr, $y:expr, $z:expr, $w:expr) => {{
                const VV: V4 = $crate::vec::__v4_const($x, $y, $z, $w);
                VV
            }};
        }
    }
}

#[macro_export(local_inner_macros)]
macro_rules! max {
    ($e:expr $(,)?) => ($e);
    ($a:expr, $b:expr $(,)?) => ({
        let (a, b) = ($a, $b);
        if a < b {
            b
        } else {
            a
        }
    });
    ($a:expr, $b:expr, $($rest:expr),+ $(,)?) => (max!($a, max!($b, $($rest),+)))
}

#[macro_export(local_inner_macros)]
macro_rules! min {
    ($e:expr $(,)?) => ($e);
    ($a:expr, $b:expr $(,)?) => ({
        let (a, b) = ($a, $b);
        if a < b {
            a
        } else {
            b
        }
    });
    ($a:expr, $b:expr, $($rest:expr),+ $(,)?) => (min!($a, min!($b, $($rest),+)))
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_min_max() {
        assert_eq!(min!(3), 3);
        assert_eq!(min!(4.0, 5.0), 4.0);
        assert_eq!(min!(4.0, -1.0, 52.0, 5.0), -1.0);
        assert_eq!(min!(-0.4, -1.0, -2.0, -5.0), -5.0);

        assert_eq!(max!(3), 3);
        assert_eq!(max!(4.0, 5.0), 5.0);
        assert_eq!(max!(4.0, -1.0, 52.0, 5.0), 52.0);
        assert_eq!(max!(-0.4, -1.0, -2.0, -5.0), -0.4);
    }
}

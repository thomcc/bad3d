/// Allows
/// ```
/// # #[allow(dead_code)] fn code_requiring_fma_and_avx() {}
/// # #[allow(dead_code)] fn code_requiring_sse41() {}
/// # #[allow(dead_code)] fn code_requiring_sse2() {}
/// # #[allow(dead_code)] fn fallback_code() {}
/// simd_match {
///     // Comma separate required target features to require both.
///     "fma", "avx" => code_requiring_fma_and_avx(),
///     "sse4.1" => code_requiring_sse41(),
///     "sse2" => code_sse2(),
///     // If a default case is ommitted, we'll emit a
///     // compile error when compiling for a platform
///     // that doesn't match something above.
///     _ => fallback_code(),
/// }
/// ```
/// Note that (like match) these must be in MOST to LEAST specific. Specifying
/// `sse` before `avx` will mean the AVX one will never be used! I'd like to fix
/// this, but am unsure how.
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

#[macro_use]
pub mod vec;

pub mod geom;
pub mod mat;
pub mod plane;
pub mod pose;
pub mod quat;
pub mod scalar;
pub mod traits;
pub mod tri;

pub mod prelude;
pub use self::prelude::*;

#[cfg(test)]
mod test_traits;

#[cfg(target_feature = "sse2")]
pub(crate) mod simd;

/// Lets us do
/// ```
/// simd_match {
///     "fma", "avx" => avx impl,
///     "sse2" => sse2 impl,
///     "sse" => sse impl,
///     _ => fallback,
/// }
/// ```
/// Note that (like match) these must be in MOST to LEAST specific. Specifying
/// `sse` before `avx` will mean the AVX one will never be used.
macro_rules! simd_match {
    (@__scan ($($nots:meta),* $(,)?) ; $(,)?) => {};
    (@__scan
        ($($nots:meta),* $(,)?) ;
        $($feat:literal),+ => $block:block,
        $($rest:tt)*
    ) => {{
        simd_match!{
            @__scan
            ( all($(target_feature = $feat),+), $($nots),*) ;
            $($rest)*
        }
        #[cfg(all(
            not(any($($nots),*)),
            $(target_feature = $feat),+
        ))] $block
    }};
    (@__scan
        ($($nots:meta),* $(,)?) ;
        _ => $fallback:block $(,)?
    ) => {
        #[cfg(not(any($($nots),*)))] $fallback
    };
    ($($feat:literal),+ => $it:block, $($rest:tt)*) => {
        simd_match! {
            @__scan
            () ;
            $($feat),+ => $it,
            $($rest)*
        }
    };
}
pub mod geom;
pub mod mat;
pub mod plane;
pub mod pose;
pub mod quat;
pub mod scalar;
pub mod traits;
pub mod tri;
pub mod vec;

pub mod prelude;
pub use self::prelude::*;

#[cfg(target_feature = "sse2")]
pub(crate) mod simd;

#[cfg(test)]
mod test_traits;

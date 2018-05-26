
macro_rules! cmp_assert_impl {
    ($li:ident, $ri:ident, $cmp:expr, $le:expr, $re:expr) => ({
        let ($li, $ri) = (&($le), &($re));
        if !($cmp) {
            panic!("assertion failed: `{}`\n  left: `{:?}`,\n right: `{:?}`",
                   stringify!($cmp), $li, $ri);
        }
    });
    ($li:ident, $ri:ident, $cmp:expr, $le:expr, $re:expr, ) => ({
        cmp_assert_impl!($li, $ri, $cmp, $le, $re);
    });
    ($li:ident, $ri:ident, $cmp:expr, $le:expr, $re:expr, $($a:tt)+) => ({
        let ($li, $ri) = (&($le), &($re));
        if !($cmp) {
            panic!("assertion failed: `{}`\n  left: `{:?}`,\n right: `{:?}`: {}",
                   stringify!($cmp), $li, $ri, format_args!($($a)+));
        }
    })
}

#[macro_export]
macro_rules! assert_lt {
    ($($a:tt)+) => ({ cmp_assert_impl!(left, right, *left < *right, $($a)+); });
}

#[macro_export]
macro_rules! assert_gt {
    ($($a:tt)+) => ({ cmp_assert_impl!(left, right, *left > *right, $($a)+); });
}

#[macro_export]
macro_rules! assert_le {
    ($($a:tt)+) => ({ cmp_assert_impl!(left, right, *left <= *right, $($a)+); });
}

#[macro_export]
macro_rules! assert_ge {
    ($($a:tt)+) => ({ cmp_assert_impl!(left, right, *left >= *right, $($a)+); });
}

#[macro_export]
macro_rules! debug_assert_lt {
    ($($arg:tt)+) => { if cfg!(debug_assertions) { assert_lt!($($arg)+); } }
}

#[macro_export]
macro_rules! debug_assert_le {
    ($($arg:tt)+) => { if cfg!(debug_assertions) { assert_le!($($arg)+); } }
}

#[macro_export]
macro_rules! debug_assert_gt {
    ($($arg:tt)+) => { if cfg!(debug_assertions) { assert_gt!($($arg)+); } }
}

#[macro_export]
macro_rules! debug_assert_ge {
    ($($arg:tt)+) => { if cfg!(debug_assertions) { assert_ge!($($arg)+); } }
}

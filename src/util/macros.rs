#[macro_export]
macro_rules! static_assert {
    ($NAME:ident, $test:expr) => {
        #[allow(dead_code, nonstandard_style)]
        const $NAME: [(); 0] = {
            const ASSERT_TEST: bool = $test;
            [(); (!ASSERT_TEST as usize)]
        };
    };
}

#[macro_export]
macro_rules! static_assert_usize_eq {
    ($NAME:ident, $first:expr, $second:expr) => {
        #[allow(dead_code, nonstandard_style)]
        const $NAME: [(); $first] = [(); $second];
    };
}

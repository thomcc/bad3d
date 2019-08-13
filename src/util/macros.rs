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

#[macro_export]
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

#[macro_export]
macro_rules! warn_if {
    ($expr:expr) => ({
        if !($expr) {
            log::warn!("warn_if!({}) failed!", stringify!($expr));
        }
    });
    ($expr:expr, ) => ({
        warn_if!($expr);
    });
    ($expr:expr, $($msg_args:tt)+) => ({
        if !($expr) {
            log::warn!("warn_if!({}) failed: {}", stringify!($expr), format_args!($($msg_args)+));
        }
    });
}

#[macro_export]
macro_rules! debug_warn_if {
    ($expr:expr) => ({
        if cfg!(debug_assertions) && !($expr) {
            log::warn!("debug_warn_if!({}) failed!", stringify!($expr));
        }
    });
    ($expr:expr, ) => ({
        debug_warn_if!($expr);
    });
    ($expr:expr, $($msg_args:tt)+) => ({
        if cfg!(debug_assertions) && !($expr) {
            log::warn!("debug_warn_if!({}) failed: {}", stringify!($expr), format_args!($($msg_args)+));
        }
    });
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

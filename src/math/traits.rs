
pub trait ApproxEq {
    fn approx_zero_e(&self, e: f32) -> bool;
    fn approx_eq_e(&self, o: &Self, e: f32) -> bool;
    // fn approx_eq_ra(&self, o: &Self, rel_tol: f32, abs_tol: f32) -> bool;

    #[inline]
    fn default_epsilon() -> f32 {
        1.0e-6_f32
    }

    #[inline]
    fn approx_zero(&self) -> bool {
        self.approx_zero_e(Self::default_epsilon())
    }

    #[inline]
    fn approx_eq(&self, o: &Self) -> bool {
        self.approx_eq_e(o, Self::default_epsilon())
    }
}

#[inline] pub fn approx_zero<T: ApproxEq>(n: T) -> bool { n.approx_zero() }
#[inline] pub fn approx_eq<T: ApproxEq>(a: T, b: T) -> bool { a.approx_eq(&b) }
#[inline] pub fn approx_zero_e<T: ApproxEq>(n: T, e: f32) -> bool { n.approx_zero_e(e) }
#[inline] pub fn approx_eq_e<T: ApproxEq>(a: T, b: T, e: f32) -> bool { a.approx_eq_e(&b, e) }

pub trait Lerp: Copy + Clone {
    fn lerp(self, Self, f32) -> Self;
}

pub trait Clamp: Copy + Clone {
    fn clamp(self, min: Self, max: Self) -> Self;
}

pub trait Identity: Copy + Clone {
    fn identity() -> Self;
}

pub trait Fold: Copy + Clone {
    fn fold<F>(self, f: F) -> f32
            where F: Fn(f32, f32) -> f32;

    fn fold2_init<T, F>(self, Self, init: T, f: F) -> T
            where F: Fn(T, f32, f32) -> T;

    fn fold_init<T, F>(self, init: T, f: F) -> T
            where F: Fn(T, f32) -> T {
        self.fold2_init(self, init, |acc, v, _| f(acc, v))
    }
}

pub trait Map: Copy + Clone {

    fn map3<F>(self, a: Self, b: Self, f: F) -> Self
            where F: Fn(f32, f32, f32) -> f32;

    #[inline]
    fn map2<F>(self, o: Self, f: F) -> Self
            where F: Fn(f32, f32) -> f32 {
        self.map3(o, self, |a, b, _| f(a, b))
    }

    #[inline]
    fn map<F>(self, f: F) -> Self
            where F: Fn(f32) -> f32 {
        self.map3(self, self, |a, _, _| f(a))
    }
}

#[inline]
pub fn dot<T: Map + Fold>(a: T, b: T) -> f32 {
    a.map2(b, |x, y| x * y).fold(|a, b| a + b)
}

#[inline]
pub fn clamp<T: Clamp>(a: T, min: T, max: T) -> T {
    a.clamp(min, max)
}


#[inline]
pub fn lerp<T: Lerp>(a: T, b: T, t: f32) -> T {
    a.lerp(b, t)
}

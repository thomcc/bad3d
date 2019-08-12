#[derive(Debug, Copy, Clone)]
pub struct PairIndices {
    pos: usize,
    end: usize,
}

#[derive(Debug, Copy, Clone)]
pub struct TripleIndices {
    pos: usize,
    end: usize,
}

impl PairIndices {
    #[inline]
    pub fn len(&self) -> usize {
        self.end.wrapping_sub(self.pos)
    }
}

impl TripleIndices {
    #[inline]
    pub fn len(&self) -> usize {
        self.end.wrapping_sub(self.pos)
    }
}

impl Iterator for PairIndices {
    type Item = (usize, usize);

    #[inline]
    fn next(&mut self) -> Option<<Self as Iterator>::Item> {
        if self.pos >= self.end {
            None
        } else {
            let p = self.pos;
            self.pos += 1;
            Some((p, self.pos % self.end))
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}
impl Iterator for TripleIndices {
    type Item = (usize, usize, usize);
    #[inline]
    fn next(&mut self) -> Option<<Self as Iterator>::Item> {
        if self.pos >= self.end {
            None
        } else {
            let p = self.pos;
            self.pos += 1;
            Some((p, self.pos % self.end, (self.pos + 1) % self.end))
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}

impl std::iter::ExactSizeIterator for PairIndices {}
impl std::iter::ExactSizeIterator for TripleIndices {}

#[inline]
pub fn pair_indices(len: usize) -> PairIndices {
    if len < 2 {
        PairIndices { pos: 2, end: len }
    } else {
        PairIndices { pos: 0, end: len }
    }
}

#[inline]
pub fn triple_indices(len: usize) -> TripleIndices {
    if len < 3 {
        TripleIndices { pos: 3, end: len }
    } else {
        TripleIndices { pos: 0, end: len }
    }
}

pub struct Pairs<'a, T> {
    pos: usize,
    slice: &'a [T],
}
pub struct Triples<'a, T> {
    pos: usize,
    slice: &'a [T],
}

pub struct ClonedPairs<'a, T> {
    pos: usize,
    slice: &'a [T],
}
pub struct ClonedTriples<'a, T: Clone> {
    pos: usize,
    slice: &'a [T],
}

#[inline]
pub fn pairs<T>(slice: &[T]) -> Pairs<'_, T> {
    Pairs::new(slice)
}

#[inline]
pub fn triples<T>(slice: &[T]) -> Triples<'_, T> {
    Triples::new(slice)
}

impl<'a, T> Pairs<'a, T> {
    #[inline]
    pub fn new(slice: &'a [T]) -> Self {
        debug_warn_if!(slice.len() < 2, "slice is too small: {}", slice.len());
        if slice.len() < 2 {
            Self {
                pos: slice.len(),
                slice,
            }
        } else {
            Self { pos: 0, slice }
        }
    }
    #[inline]
    unsafe fn get_at(&self, p0: usize) -> (&'a T, &'a T) {
        let len = self.slice.len();
        debug_assert_lt!(p0, len);
        let p1 = if p0 + 1 == len { 0 } else { p0 + 1 };
        (self.slice.get_unchecked(p0), self.slice.get_unchecked(p1))
    }
}

impl<'a, T: Clone> Pairs<'a, T> {
    #[inline]
    pub fn cloned(self) -> ClonedPairs<'a, T> {
        ClonedPairs {
            pos: self.pos,
            slice: self.slice,
        }
    }
}

impl<'a, T> Triples<'a, T> {
    #[inline]
    pub fn new(slice: &'a [T]) -> Self {
        debug_warn_if!(slice.len() < 3, "slice is too small: {}", slice.len());
        if slice.len() < 3 {
            Self {
                pos: slice.len(),
                slice,
            }
        } else {
            Self { pos: 0, slice }
        }
    }

    #[inline]
    unsafe fn get_at(&self, p0: usize) -> (&'a T, &'a T, &'a T) {
        let len = self.slice.len();
        debug_assert_lt!(p0, len);
        let p1 = if p0 + 1 == len { 0 } else { p0 + 1 };
        let p2 = if p1 + 1 == len { 0 } else { p1 + 1 };
        (
            self.slice.get_unchecked(p0),
            self.slice.get_unchecked(p1),
            self.slice.get_unchecked(p2),
        )
    }
}

impl<'a, T: Clone> Triples<'a, T> {
    #[inline]
    pub fn cloned(self) -> ClonedTriples<'a, T> {
        ClonedTriples {
            pos: self.pos,
            slice: self.slice,
        }
    }
}

impl<'a, T> ClonedPairs<'a, T>
where
    T: Clone,
{
    #[inline]
    unsafe fn get_at(&self, p0: usize) -> (T, T) {
        let len = self.slice.len();
        debug_assert_lt!(p0, len);
        let p1 = if p0 + 1 == len { 0 } else { p0 + 1 };
        (
            self.slice.get_unchecked(p0).clone(),
            self.slice.get_unchecked(p1).clone(),
        )
    }
}

impl<'a, T> ClonedTriples<'a, T>
where
    T: Clone,
{
    #[inline]
    unsafe fn get_at(&self, p0: usize) -> (T, T, T) {
        let len = self.slice.len();
        debug_assert_lt!(p0, len);
        let p1 = if p0 + 1 == len { 0 } else { p0 + 1 };
        let p2 = if p1 + 1 == len { 0 } else { p1 + 1 };
        (
            self.slice.get_unchecked(p0).clone(),
            self.slice.get_unchecked(p1).clone(),
            self.slice.get_unchecked(p2).clone(),
        )
    }
}

macro_rules! impl_wrap_iter {
    ($name:ty, $item:ty, $( $clause:tt )*) => {
        impl<'a, T> $name $($clause)* {
            #[inline]
            pub fn len(&self) -> usize {
                self.slice.len().saturating_sub(self.pos)
            }
        }

        impl<'a, T> Iterator for $name $($clause)* {
            type Item = $item;

            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                let curr = self.pos;
                if curr >= self.slice.len() {
                    None
                } else {
                    self.pos += 1;
                    Some(unsafe { self.get_at(curr) })
                }
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                (self.len(), Some(self.len()))
            }

            #[inline]
            fn count(self) -> usize {
                self.len()
            }

            #[inline]
            fn last(self) -> Option<Self::Item> {
                let len = self.slice.len();
                if self.pos >= self.slice.len() {
                    None
                } else {
                    Some(unsafe { self.get_at(len - 1) })
                }
            }

            #[inline]
            fn nth(&mut self, n: usize) -> Option<Self::Item> {
                if n + self.pos >= self.slice.len() {
                    self.pos = self.slice.len();
                    None
                } else {
                    let curr = self.pos + n;
                    self.pos = curr + 1;
                    Some(unsafe { self.get_at(curr) })
                }
            }
        }
        impl<'a, T> std::iter::ExactSizeIterator for $name $($clause)*  {}
    };
}

impl_wrap_iter!(Pairs<'a, T>, (&'a T, &'a T),);
impl_wrap_iter!(Triples<'a, T>, (&'a T, &'a T, &'a T),);

// impl_wrap_iter!(PairsMut, (&'a mut T, &'a mut T),);
// impl_wrap_iter!(TriplesMut, (&'a mut T, &'a mut T, &'a mut T),);

impl_wrap_iter!(ClonedPairs<'a, T>, (T, T), where T: Clone);
impl_wrap_iter!(ClonedTriples<'a, T>, (T, T, T), where T: Clone);

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_pairs() {
        let s0 = &[1, 2];
        let p0 = pairs(s0).collect::<Vec<_>>();
        assert_eq!(&p0, &[(&1, &2), (&2, &1)]);

        let s1 = &[1, 2, 3, 4, 5];
        let p1 = pairs(s1).cloned().collect::<Vec<_>>();
        assert_eq!(&p1, &[(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)]);

        let s2 = &[1];
        let p2 = pairs(s2).collect::<Vec<_>>();
        assert_eq!(&p2, &[]);
    }

    #[test]
    fn test_triples() {
        let s0 = &[1, 2, 3];
        let p0 = triples(s0).collect::<Vec<_>>();
        assert_eq!(&p0, &[(&1, &2, &3), (&2, &3, &1), (&3, &1, &2)]);

        let s1 = &[1, 2, 3, 4, 5];
        let p1 = triples(s1).cloned().collect::<Vec<_>>();
        assert_eq!(
            &p1,
            &[(1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 1), (5, 1, 2)]
        );

        let s2 = &[1, 2];
        let p2 = triples(s2).cloned().collect::<Vec<_>>();
        assert_eq!(&p2, &[]);
    }

    #[test]
    fn test_indices() {
        let p0 = pair_indices(5).collect::<Vec<_>>();
        assert_eq!(&p0, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]);

        let p1 = triple_indices(5).collect::<Vec<_>>();
        assert_eq!(
            &p1,
            &[(0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 0), (4, 0, 1)]
        );

        let p2 = pair_indices(1).collect::<Vec<_>>();
        assert_eq!(&p2, &[]);

        let p3 = triple_indices(2).collect::<Vec<_>>();
        assert_eq!(&p3, &[]);
    }
}

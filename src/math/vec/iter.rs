const MAX_VEC_SIZE: usize = 4;

#[derive(Debug, Clone, PartialEq)]
pub struct VecIter<T: Copy> {
    items: [T; MAX_VEC_SIZE],
    end: usize,
    pos: usize,
}

impl<T: Copy> VecIter<T> {
    #[inline]
    pub(crate) fn new(items: [T; MAX_VEC_SIZE], num_items: usize) -> Self {
        debug_assert!(num_items <= MAX_VEC_SIZE);
        Self {
            items,
            end: num_items,
            pos: 0,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        debug_assert!(self.pos <= self.end);
        self.end.saturating_sub(self.pos)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.pos >= self.end
    }
}

impl<T: Copy> Iterator for VecIter<T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.end {
            return None;
        }
        let p = self.pos;
        self.pos += 1;
        Some(self.items[p])
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        if self.pos >= self.end {
            None
        } else {
            Some(self.items[self.end - 1])
        }
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        if n > MAX_VEC_SIZE || self.pos + n >= self.end {
            self.pos = self.end;
            return None;
        }
        self.pos += n;
        let p = self.pos;
        self.pos += 1;
        Some(self.items[p])
    }
}

impl<T: Copy> std::iter::ExactSizeIterator for VecIter<T> {}

impl<T: Copy> std::iter::DoubleEndedIterator for VecIter<T> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        if self.pos >= self.end {
            return None;
        }
        self.end -= 1;
        Some(self.items[self.end])
    }
}

impl<T: Copy> std::iter::FusedIterator for VecIter<T> {}

#[cfg(test)]
#[allow(clippy::cognitive_complexity)]
mod tests {
    use super::super::*;
    // test each size separately since when they're this small they all have
    // edge-cases...
    #[test]
    fn test_vec_iter2() {
        let v2i = vec2(100.0, 4.0).into_iter();
        assert_eq!(v2i.len(), 2);
        assert_eq!(&v2i.clone().collect::<Vec<_>>(), &[100.0, 4.0]);
        assert_eq!(&v2i.clone().rev().collect::<Vec<_>>(), &[4.0, 100.0]);
        {
            let mut v2ia = v2i.clone();
            assert_eq!(v2ia.next(), Some(100.0));
            assert_eq!(v2ia.len(), 1);
            assert_eq!(v2ia.next_back(), Some(4.0));
            assert_eq!(v2ia.len(), 0);
            assert_eq!(v2ia.next_back(), None);
        }

        {
            let mut v2ia = v2i.clone();
            assert_eq!(v2ia.nth(1), Some(4.0));
            assert_eq!(v2ia.len(), 0);
            assert_eq!(v2ia.next(), None);
        }

        {
            let mut v2ia = v2i.clone();
            assert_eq!(v2ia.nth(1), Some(4.0));
            assert_eq!(v2ia.len(), 0);
            assert_eq!(v2ia.next_back(), None);
        }
    }

    #[test]
    fn test_vec_iter3() {
        let vi = vec3(100.0, 4.0, 6.0).into_iter();
        assert_eq!(vi.len(), 3);
        assert_eq!(&vi.clone().collect::<Vec<_>>(), &[100.0, 4.0, 6.0]);
        assert_eq!(&vi.clone().rev().collect::<Vec<_>>(), &[6.0, 4.0, 100.0]);
        {
            let mut via = vi.clone();
            assert_eq!(via.next(), Some(100.0));
            assert_eq!(via.len(), 2);
            assert_eq!(via.next_back(), Some(6.0));
            assert_eq!(via.len(), 1);

            let mut vib = via.clone();
            assert_eq!(vib.next_back(), Some(4.0));
            assert_eq!(vib.len(), 0);
            assert_eq!(vib.next_back(), None);

            let mut vib = via.clone();
            assert_eq!(vib.next(), Some(4.0));
            assert_eq!(vib.len(), 0);
            assert_eq!(vib.next(), None);
        }

        {
            let mut via = vi.clone();
            assert_eq!(via.nth(1), Some(4.0));
            assert_eq!(via.len(), 1);

            let mut vib = via.clone();
            assert_eq!(vib.nth(1), None);
            assert_eq!(vib.len(), 0);

            let mut vib = via.clone();
            assert_eq!(vib.next(), Some(6.0));
            assert_eq!(vib.len(), 0);

            let mut vib = via.clone();
            assert_eq!(vib.next_back(), Some(6.0));
            assert_eq!(vib.len(), 0);
        }
    }

    #[test]
    fn test_vec_iter4() {
        let vi = vec4(100.0, 4.0, 6.0, 15.0).into_iter();
        assert_eq!(vi.len(), 4);
        assert_eq!(&vi.clone().collect::<Vec<_>>(), &[100.0, 4.0, 6.0, 15.0]);
        assert_eq!(&vi.clone().rev().collect::<Vec<_>>(), &[15.0, 6.0, 4.0, 100.0]);
        {
            let mut via = vi.clone();
            assert_eq!(via.next(), Some(100.0));
            assert_eq!(via.len(), 3);
            assert_eq!(via.next_back(), Some(15.0));
            assert_eq!(via.len(), 2);

            let mut vib = via.clone();
            assert_eq!(vib.next_back(), Some(6.0));
            assert_eq!(vib.len(), 1);
            assert_eq!(vib.nth(0), Some(4.0));
            assert_eq!(vib.len(), 0);
            assert_eq!(vib.next_back(), None);

            let mut vib = via.clone();
            assert_eq!(vib.next(), Some(4.0));
            assert_eq!(vib.len(), 1);
            assert_eq!(vib.next(), Some(6.0));
            assert_eq!(vib.len(), 0);
            assert_eq!(vib.next(), None);
        }

        {
            let mut via = vi.clone();
            assert_eq!(via.nth(1), Some(4.0));
            assert_eq!(via.len(), 2);

            let mut vib = via.clone();
            assert_eq!(vib.nth(1), Some(15.0));
            assert_eq!(vib.len(), 0);

            let mut vib = via.clone();
            assert_eq!(vib.next(), Some(6.0));
            assert_eq!(vib.len(), 1);
            assert_eq!(vib.next_back(), Some(15.0));
            assert_eq!(vib.len(), 0);
        }
    }
}

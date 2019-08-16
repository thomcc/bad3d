type Index = u16;

#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Idx3(pub Index, pub Index, pub Index);

const _SIZE_CHECK: [(); core::mem::size_of::<[u16; 3]>()] = [(); core::mem::size_of::<Idx3>()];
const _ALIGN_CHECK: [(); core::mem::align_of::<[u16; 3]>()] = [(); core::mem::align_of::<Idx3>()];

const MAX_IDX: usize = Index::max_value() as usize;

impl Idx3 {
    pub const MAX_IDX: usize = MAX_IDX;
    #[inline(always)]
    pub fn as_array(&self) -> &[Index; 3] {
        unsafe { &*(self as *const Idx3 as *const [Index; 3]) }
    }

    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut [Index; 3] {
        unsafe { &mut *(self as *mut Idx3 as *mut [Index; 3]) }
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[Index] {
        self.as_array()
    }

    #[inline(always)]
    pub fn as_slice_mut(&mut self) -> &mut [Index] {
        self.as_array_mut()
    }

    #[inline]
    pub const fn indices(self) -> [usize; 3] {
        [self.0 as usize, self.1 as usize, self.2 as usize]
    }

    #[inline] // legacy
    pub const fn tri_indices(self) -> (usize, usize, usize) {
        (self.0 as usize, self.1 as usize, self.2 as usize)
    }

    #[inline]
    pub fn tri_refs<V>(self, vs: &[V]) -> (&V, &V, &V) {
        let [a, b, c] = self.indices();
        (&vs[a], &vs[b], &vs[c])
    }

    #[inline]
    pub fn tri_verts<V: Copy>(self, vs: &[V]) -> (V, V, V) {
        let [a, b, c] = self.indices();
        (vs[a], vs[b], vs[c])
    }

    #[inline]
    pub fn tri_verts_opt<V: Copy>(self, vs: &[V]) -> Option<(V, V, V)> {
        if let Some((a, b, c)) = self.inbounds_indices(vs.len()) {
            Some((vs[a], vs[b], vs[c]))
        } else {
            None
        }
    }

    #[inline]
    pub fn tri_vert_ref<V>(self, vs: &[V]) -> (&V, &V, &V) {
        let (a, b, c) = self.tri_indices();
        (&vs[a], &vs[b], &vs[c])
    }

    #[inline]
    pub fn tri_verts_mut<V>(self, vs: &mut [V]) -> Option<(&mut V, &mut V, &mut V)> {
        let (a, b, c) = self.inbounds_distinct_indices(vs.len())?;
        debug_assert!(a < vs.len() && b < vs.len() && c < vs.len());
        debug_assert!(a != b && b != c && c != a);
        unsafe {
            let x = &mut *vs.as_mut_ptr().add(a);
            let y = &mut *vs.as_mut_ptr().add(b);
            let z = &mut *vs.as_mut_ptr().add(c);
            Some((x, y, z))
        }
    }

    #[inline]
    pub fn distinct(self) -> bool {
        (self.0 != self.1) & (self.1 != self.2) & (self.2 != self.0)
    }

    #[inline]
    pub fn inbounds(self, bounds: usize) -> bool {
        ((self.0 as usize) < bounds) & ((self.1 as usize) < bounds) & ((self.2 as usize) < bounds)
    }

    #[inline]
    pub fn inbounds_distinct_indices(self, bounds: usize) -> Option<(usize, usize, usize)> {
        let i0 = self.0 as usize;
        let i1 = self.1 as usize;
        let i2 = self.2 as usize;
        let inbounds = (i0 < bounds) & (i1 < bounds) & (i2 < bounds);
        let distinct = (i0 != i1) & (i1 != i2) & (i2 != i0);
        if inbounds & distinct {
            Some((i0, i1, i2))
        } else {
            None
        }
    }

    #[inline]
    pub fn inbounds_indices(self, bounds: usize) -> Option<(usize, usize, usize)> {
        let i0 = self.0 as usize;
        let i1 = self.1 as usize;
        let i2 = self.2 as usize;
        if (i0 < bounds) & (i1 < bounds) & (i2 < bounds) {
            Some((i0, i1, i2))
        } else {
            None
        }
    }

    #[inline]
    pub fn distinct_inbounds(self, bounds: usize) -> bool {
        self.inbounds(bounds) & self.distinct()
    }
}

impl std::ops::Index<usize> for Idx3 {
    type Output = u16;
    #[inline(always)]
    fn index(&self, u: usize) -> &u16 {
        &self.as_array()[u]
    }
}

impl std::ops::IndexMut<usize> for Idx3 {
    #[inline(always)]
    fn index_mut(&mut self, u: usize) -> &mut u16 {
        &mut self.as_array_mut()[u]
    }
}

impl From<(usize, usize, usize)> for Idx3 {
    #[inline]
    fn from(t: (usize, usize, usize)) -> Self {
        debug_assert!(t.0 < MAX_IDX && t.1 < MAX_IDX && t.2 < MAX_IDX, "{:?}", t);
        Self(t.0 as Index, t.1 as Index, t.2 as Index)
    }
}

impl From<(i32, i32, i32)> for Idx3 {
    #[inline]
    fn from(t: (i32, i32, i32)) -> Self {
        debug_assert!(
            ((t.0 as usize) < MAX_IDX) && ((t.1 as usize) < MAX_IDX) && ((t.2 as usize) < MAX_IDX),
            "{:?}",
            t
        );
        Self(t.0 as Index, t.1 as Index, t.2 as Index)
    }
}

impl From<(u16, u16, u16)> for Idx3 {
    #[inline]
    fn from(t: (u16, u16, u16)) -> Self {
        Self(t.0, t.1, t.2)
    }
}

impl From<[u16; 3]> for Idx3 {
    #[inline]
    fn from(t: [u16; 3]) -> Self {
        Self(t[0], t[1], t[2])
    }
}

impl From<[i32; 3]> for Idx3 {
    #[inline]
    fn from(t: [i32; 3]) -> Self {
        debug_assert!(
            ((t[0] as usize) < MAX_IDX) && ((t[1] as usize) < MAX_IDX) && ((t[2] as usize) < MAX_IDX),
            "{:?}",
            t
        );
        Self(t[0] as Index, t[1] as Index, t[2] as Index)
    }
}

impl From<[usize; 3]> for Idx3 {
    #[inline]
    fn from(t: [usize; 3]) -> Self {
        debug_assert!(t[0] < MAX_IDX && t[1] < MAX_IDX && t[2] < MAX_IDX, "{:?}", t);
        Self(t[0] as Index, t[1] as Index, t[2] as Index)
    }
}

impl From<Idx3> for (usize, usize, usize) {
    #[inline]
    fn from(t: Idx3) -> (usize, usize, usize) {
        (t.0 as usize, t.1 as usize, t.2 as usize)
    }
}

impl From<Idx3> for (i32, i32, i32) {
    #[inline]
    fn from(t: Idx3) -> (i32, i32, i32) {
        (t.0 as i32, t.1 as i32, t.2 as i32)
    }
}

impl From<Idx3> for (u16, u16, u16) {
    #[inline]
    fn from(t: Idx3) -> (u16, u16, u16) {
        (t.0, t.1, t.2)
    }
}

impl From<Idx3> for [u16; 3] {
    #[inline]
    fn from(t: Idx3) -> [u16; 3] {
        [t[0], t[1], t[2]]
    }
}

impl From<Idx3> for [i32; 3] {
    #[inline]
    fn from(t: Idx3) -> [i32; 3] {
        [t.0 as i32, t.1 as i32, t.2 as i32]
    }
}

impl From<Idx3> for [usize; 3] {
    #[inline]
    fn from(t: Idx3) -> [usize; 3] {
        [t.0 as usize, t.1 as usize, t.2 as usize]
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_inbounds() {
        assert_eq!(Idx3(1, 2, 3).inbounds(5), true);
        assert_eq!(Idx3(1, 2, 3).inbounds(3), false);
        assert_eq!(Idx3(1, 2, 3).inbounds(4), true);
        assert_eq!(Idx3(1, 2, 0).inbounds(3), true);
        assert_eq!(Idx3(1, 2, 0).inbounds(1), false);
        assert_eq!(Idx3(1, 2, 0).inbounds(2), false);
        assert_eq!(Idx3(0, 0, 0).inbounds(1), true);
        assert_eq!(Idx3(0, 0, 0).inbounds(2), true);
        assert_eq!(Idx3(0, 0, 0).inbounds(u16::max_value() as usize), true);
        assert_eq!(Idx3(0, 0, 0).inbounds(0), false);
        assert_eq!(
            Idx3(u16::max_value(), u16::max_value(), u16::max_value()).inbounds(0),
            false
        );
    }

    #[test]
    fn test_naib() {
        assert_eq!(Idx3(1, 2, 3).distinct_inbounds(5), true);
        assert_eq!(Idx3(1, 2, 3).distinct_inbounds(3), false);
        assert_eq!(Idx3(1, 2, 3).distinct_inbounds(4), true);
        assert_eq!(Idx3(1, 2, 0).distinct_inbounds(3), true);
        assert_eq!(Idx3(1, 2, 0).distinct_inbounds(1), false);
        // assert_eq!(Idx3(0, 0, 0).distinct_inbounds(0), false);
        assert_eq!(
            Idx3(u16::max_value(), u16::max_value() - 1, u16::max_value() - 2).distinct_inbounds(0),
            false
        );
    }
}

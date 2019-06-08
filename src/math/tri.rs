use crate::math::geom;
use crate::math::mat::*;
use crate::math::plane::*;
use crate::math::traits::*;
use crate::math::vec::*;

use std::iter;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Tri(pub V3, pub V3, pub V3);

impl Tri {
    #[inline]
    pub fn from_index(vs: &[V3], i: impl TriIndices) -> Self {
        let (a, b, c) = i.tri_indices();
        Tri(vs[a], vs[b], vs[c])
    }

    #[inline]
    pub fn tup(self) -> (V3, V3, V3) {
        (self.0, self.1, self.2)
    }

    #[inline]
    pub fn face_dir(&self) -> V3 {
        geom::tri_face_dir(self.0, self.1, self.2)
    }

    #[inline]
    pub fn normal(&self) -> V3 {
        geom::tri_normal(self.0, self.1, self.2)
    }

    #[inline]
    pub fn area(&self) -> f32 {
        geom::tri_area(self.0, self.1, self.2)
    }

    #[inline]
    pub fn is_degenerate(&self, epsilon: f32) -> bool {
        self.area() < epsilon
    }

    #[inline]
    pub fn to_barycentric(&self, p: V3) -> V3 {
        geom::barycentric(self.0, self.1, self.2, p)
    }

    #[inline]
    pub fn project(&self, p: V3) -> V3 {
        geom::tri_project(self.0, self.1, self.2, p)
    }

    #[inline]
    pub fn matrix(&self) -> M3x3 {
        M3x3::from_cols(self.0, self.1, self.2)
    }

    #[inline]
    pub fn closest_point_to(self, p: V3) -> V3 {
        geom::closest_point_on_triangle(self.0, self.1, self.2, p)
    }

    #[inline]
    pub fn plane(self) -> Plane {
        Plane::from_tri(self.0, self.1, self.2)
    }

    #[inline]
    pub fn gradient(self, t0: f32, t1: f32, t2: f32) -> V3 {
        geom::gradient(self.0, self.1, self.2, t0, t1, t2)
    }
}

impl From<Tri> for (V3, V3, V3) {
    #[inline]
    fn from(t: Tri) -> Self {
        t.tup()
    }
}

impl From<(V3, V3, V3)> for Tri {
    #[inline]
    fn from(t: (V3, V3, V3)) -> Self {
        Tri(t.0, t.1, t.2)
    }
}

#[derive(Debug, Clone)]
pub struct TriIter<'a, V: 'a, I> {
    verts: &'a [V],
    wrapped: I,
}

impl<'a, Tri: TriIndices, V, I: Iterator<Item = Tri>> TriIter<'a, V, I> {
    #[inline]
    pub fn new(vs: &'a [V], iter: I) -> Self {
        Self {
            verts: vs,
            wrapped: iter,
        }
    }
}

impl<'a, Tri: TriIndices, V: Clone, I: Iterator<Item = Tri> + 'a> TriIter<'a, V, I> {
    #[inline]
    pub fn copied(self) -> impl Iterator<Item = (V, V, V)> + 'a {
        self.map(|(a, b, c)| (a.clone(), b.clone(), c.clone()))
    }
}

#[inline]
pub fn de_index<V: Clone, I: TriIndices>(verts: &[V], tris: &[I]) -> Vec<(V, V, V)> {
    tri_iter(verts, tris).collect()
}

#[inline]
pub fn tri_ref_iter<'a, 'b, V, I: TriIndices>(
    verts: &'a [V],
    tris: &'b [I],
) -> TriIter<'a, V, impl Iterator<Item = I> + 'b> {
    TriIter::new(verts, tris.iter().cloned())
}

#[inline]
pub fn tri_iter<'a, V: Clone, I: TriIndices>(
    verts: &'a [V],
    tris: &'a [I],
) -> impl Iterator<Item = (V, V, V)> + 'a {
    TriIter::new(verts, tris.iter().cloned()).copied()
}

impl<'a, Vert, Tri, Iter> Iterator for TriIter<'a, Vert, Iter>
where
    Tri: TriIndices,
    Iter: Iterator<Item = Tri>,
{
    type Item = (&'a Vert, &'a Vert, &'a Vert);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.wrapped.next().map(|tri| tri.tri_vert_ref(self.verts))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.wrapped.size_hint()
    }

    #[inline]
    fn count(self) -> usize {
        self.wrapped.count()
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        let TriIter { wrapped, verts } = self;
        wrapped.last().map(|tri| tri.tri_vert_ref(verts))
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.wrapped.nth(n).map(|tri| tri.tri_vert_ref(self.verts))
    }
}

impl<'a, Vert, Tri, Iter> iter::ExactSizeIterator for TriIter<'a, Vert, Iter>
where
    Tri: TriIndices,
    Iter: Iterator<Item = Tri> + iter::ExactSizeIterator,
{
    #[inline]
    fn len(&self) -> usize {
        self.wrapped.len()
    }
}

impl<'a, Vert, Tri, Iter> iter::DoubleEndedIterator for TriIter<'a, Vert, Iter>
where
    Tri: TriIndices,
    Iter: Iterator<Item = Tri> + iter::DoubleEndedIterator,
{
    #[inline]
    fn next_back(&mut self) -> Option<<Self as Iterator>::Item> {
        self.wrapped
            .next_back()
            .map(|tri| tri.tri_vert_ref(self.verts))
    }
}

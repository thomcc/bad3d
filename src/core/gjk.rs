use crate::core::{
    hull,
    support::{Support, TransformedSupport},
};
use crate::math::prelude::*;
use std::f32;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum ContactType {
    Unknown,
    PtPlane,
    EdgeEdge,
    PlanePt,
}

impl Default for ContactType {
    #[inline]
    fn default() -> ContactType {
        ContactType::Unknown
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct ContactInfo {
    pub ty: ContactType,
    pub simplex: [V3; 4],
    pub plane: Plane,
    pub impact: V3,
    pub separation: f32,
    pub points: (V3, V3),
    pub time: f32,
}

impl ContactInfo {
    fn fill_simplex(&mut self, src: &Simplex) {
        self.ty = ContactType::Unknown;

        if src.points[0].a == src.points[1].a && src.points[0].a == src.points[2].a {
            self.ty = ContactType::PtPlane;
            self.simplex[0] = src.points[0].a;
            self.simplex[1] = src.points[0].b;
            self.simplex[2] = src.points[1].b;
            self.simplex[3] = src.points[2].b;
            if dot(
                self.plane.normal,
                cross(
                    src.points[1].p - src.points[0].p,
                    src.points[2].p - src.points[0].p,
                ),
            ) < 0.0
            {
                self.simplex.swap(1, 2);
            }
        } else if src.points[0].b == src.points[1].b && src.points[0].b == src.points[2].b {
            self.ty = ContactType::PlanePt;
            self.simplex[0] = src.points[0].a;
            self.simplex[1] = src.points[1].a;
            self.simplex[2] = src.points[2].a;
            self.simplex[3] = src.points[2].b;
            if dot(
                self.plane.normal,
                cross(
                    src.points[1].p - src.points[0].p,
                    src.points[2].p - src.points[0].p,
                ),
            ) < 0.0
            {
                self.simplex.swap(1, 2);
            }
        } else if (src.points[0].a == src.points[1].a
            || src.points[0].a == src.points[2].a
            || src.points[1].a == src.points[2].a)
            && (src.points[0].b == src.points[1].b
                || src.points[0].b == src.points[2].b
                || src.points[1].b == src.points[2].b)
        {
            self.ty = ContactType::EdgeEdge;

            self.simplex[0] = src.points[0].a;
            self.simplex[1] = if src.points[1].a != src.points[0].a {
                src.points[1].a
            } else {
                src.points[2].a
            };
            self.simplex[2] = src.points[0].b;
            self.simplex[3] = if src.points[1].b != src.points[0].b {
                src.points[1].b
            } else {
                src.points[2].b
            };

            let dp = dot(
                self.plane.normal,
                cross(
                    src.points[1].p - src.points[0].p,
                    src.points[2].p - src.points[0].p,
                ),
            );

            if (dp < 0.0 && src.points[1].a != src.points[0].a)
                || (dp > 0.0 && src.points[1].a == src.points[0].a)
            {
                self.simplex.swap(2, 3);
            }
        }
    }
}

#[derive(Copy, Clone, Debug, Default)]
struct Point {
    a: V3,
    b: V3,
    p: V3,
    t: f32,
}

impl Point {
    fn new(a: V3, b: V3, p: V3) -> Point {
        Point {
            a: a,
            b: b,
            p: p,
            t: 0.0,
        }
    }

    fn on_sum(a: &dyn Support, b: &dyn Support, n: V3) -> Point {
        let pa = a.support(n);
        let pb = b.support(-n);
        Point::new(pa, pb, pa - pb)
    }

    #[inline]
    fn with_t(&self, t: f32) -> Point {
        Point { t: t, ..*self }
    }
}

#[derive(Copy, Clone, Debug, Default)]
struct Simplex {
    v: V3,
    size: u32, // how many points actually exist
    points: [Point; 4],
}

#[inline]
fn towards_origin(a: V3, b: V3) -> bool {
    dot(-a, b - a) > 0.0
}

impl Simplex {
    fn set(&self, v: V3, pts: &[Point]) -> Simplex {
        let mut res = *self;
        res.v = v;
        for (i, p) in pts.iter().enumerate() {
            res.points[i] = *p;
        }
        res.size = pts.len() as u32;
        res
    }

    fn initial(v: V3) -> Simplex {
        Simplex {
            v: v,
            size: 0,
            ..Default::default()
        }
    }

    fn finish(&self, w: &Point) -> Simplex {
        debug_assert!(self.size == 3);
        let mut result = *self;
        result.points[3] = *w;
        result.size = 4;
        result.v = V3::zero();
        result
    }

    fn next1(&self, w: &Point) -> Simplex {
        let s = self.points[0];
        let t = geom::line_project_time(w.p, s.p, V3::zero());
        if t < 0.0 {
            self.set(w.p, &[w.with_t(1.0)])
        } else {
            self.set(
                w.p + (self.points[0].p - w.p) * t,
                &[s.with_t(t), w.with_t(1.0 - t)],
            )
        }
    }

    fn next2(&self, w: &Point) -> Simplex {
        let s0 = self.points[0];
        let s1 = self.points[1];

        let t0 = geom::line_project_time(w.p, s0.p, V3::zero());
        let t1 = geom::line_project_time(w.p, s1.p, V3::zero());

        let v0 = w.p.lerp(s0.p, t0);
        let v1 = w.p.lerp(s1.p, t1);

        let in_edge0 = towards_origin(v0, s1.p);
        let in_edge1 = towards_origin(v1, s0.p);

        if in_edge0 && in_edge1 {
            self.set(
                geom::tri_project(s0.p, s1.p, w.p, V3::zero()),
                &[s0, s1, *w],
            )
        } else if !in_edge0 && t0 > 0.0 {
            self.set(v0, &[s0.with_t(t0), w.with_t(1.0 - t0)])
        } else if !in_edge1 && t1 > 0.0 {
            self.set(v1, &[s1.with_t(t1), w.with_t(1.0 - t1)])
        } else {
            self.set(w.p, &[w.with_t(1.0)])
        }
    }

    fn next3(&self, w: &Point) -> Simplex {
        let s0 = self.points[0];
        let s1 = self.points[1];
        let s2 = self.points[2];
        let t0 = geom::line_project_time(w.p, s0.p, V3::zero());
        let t1 = geom::line_project_time(w.p, s1.p, V3::zero());
        let t2 = geom::line_project_time(w.p, s2.p, V3::zero());
        let v0 = w.p + (s0.p - w.p) * t0;
        let v1 = w.p + (s1.p - w.p) * t1;
        let v2 = w.p + (s2.p - w.p) * t2;
        let c0 = geom::tri_project(w.p, s1.p, s2.p, V3::zero());
        let c1 = geom::tri_project(w.p, s2.p, s0.p, V3::zero());
        let c2 = geom::tri_project(w.p, s0.p, s1.p, V3::zero());

        let inp0 = towards_origin(c0, s0.p);
        let inp1 = towards_origin(c1, s1.p);
        let inp2 = towards_origin(c2, s2.p);

        let inp2e0 = towards_origin(v0, s1.p);
        let inp2e1 = towards_origin(v1, s0.p);

        let inp0e1 = towards_origin(v1, s2.p);
        let inp0e2 = towards_origin(v2, s1.p);

        let inp1e2 = towards_origin(v2, s0.p);
        let inp1e0 = towards_origin(v0, s2.p);
        if inp0 && inp1 && inp2 {
            self.finish(w) /* terminated */
        } else if !inp2 && inp2e0 && inp2e1 {
            self.set(
                geom::tri_project(s0.p, s1.p, w.p, V3::zero()),
                &[self.points[0], self.points[1], *w],
            )
        } else if !inp0 && inp0e1 && inp0e2 {
            self.set(
                geom::tri_project(s1.p, s2.p, w.p, V3::zero()),
                &[self.points[1], self.points[2], *w],
            )
        } else if !inp1 && inp1e2 && inp1e0 {
            self.set(
                geom::tri_project(s2.p, s0.p, w.p, V3::zero()),
                &[self.points[2], self.points[0], *w],
            )
        } else if !inp1e0 && !inp2e0 && t0 > 0.0 {
            self.set(v0, &[self.points[0].with_t(t0), w.with_t(1.0 - t0)])
        } else if !inp2e1 && !inp0e1 && t1 > 0.0 {
            self.set(v1, &[self.points[1].with_t(t1), w.with_t(1.0 - t1)])
        } else if !inp0e2 && !inp1e2 && t2 > 0.0 {
            self.set(v2, &[self.points[2].with_t(t2), w.with_t(1.0 - t2)])
        } else {
            self.set(w.p, &[w.with_t(1.0)])
        }
    }

    fn next(&self, w: &Point) -> Simplex {
        match self.size {
            0 => self.set(w.p, &[w.with_t(1.0)]),
            1 => self.next1(w),
            2 => self.next2(w),
            3 => self.next3(w),
            _ => unreachable!(),
        }
    }

    fn compute_points(&mut self) -> ContactInfo {
        assert!(self.size > 0);
        if self.size == 3 {
            let b = geom::barycentric(self.points[0].p, self.points[1].p, self.points[2].p, self.v);
            self.points[0].t = b.x;
            self.points[1].t = b.y;
            self.points[2].t = b.z;
        }

        let mut pa = V3::zero();
        let mut pb = V3::zero();
        for pt in self.points[0..(self.size as usize)].iter() {
            pa += pt.t * pt.a;
            pb += pt.t * pt.b;
        }

        let (pa, pb) = self.points[0..(self.size as usize)]
            .iter()
            .fold((V3::zero(), V3::zero()), |(pa, pb), p| {
                (pa + p.t * p.a, pb + p.t * p.b)
            });

        let norm = self.v.norm_or(0.0, 0.0, 1.0);
        let impact = (pa + pb) * 0.5;
        let mut hit_info = ContactInfo {
            points: (pa, pb),
            impact: impact,
            separation: pa.dist(pb) + f32::MIN_POSITIVE,
            plane: Plane::from_norm_and_point(norm, impact),
            ..Default::default()
        };
        hit_info.fill_simplex(self);

        assert!(hit_info.separation > 0.0);
        hit_info
    }
}

pub fn separated(a: &dyn Support, b: &dyn Support, find_closest: bool) -> ContactInfo {
    let eps = 0.00001_f32;

    let mut v = Point::on_sum(a, b, vec3(0.0, 0.0, 1.0)).p;
    let mut last = Simplex::initial(v);

    let mut w = Point::on_sum(a, b, -v);
    let mut next = last.next(&w); //Simplex::from_point(&w);

    let mut iter = 0;
    while iter == 0 || (dot(w.p, v) < dot(v, v) - eps) && iter < 100 {
        iter += 1;
        last = next;
        v = last.v;
        w = Point::on_sum(a, b, -v);
        if dot(w.p, v) >= dot(v, v) - (eps + eps * dot(v, v)) {
            break;
        }
        if !find_closest && dot(w.p, v) >= 0.0 {
            break;
        }

        next = last.next(&w);
        if next.v.is_zero() {
            if next.size == 2 {
                last = next;
                let n = (next.points[0].p - next.points[1].p).orth();
                next.points[2] = Point::on_sum(a, b, n);
                next.size = 3;
            }
            if next.size == 3 {
                last = next;
                let n = geom::tri_normal(next.points[0].p, next.points[1].p, next.points[2].p);
                next.points[3] = Point::on_sum(a, b, n);
                next.size = 4;
            }
            assert!(next.size == 4);
            let min_penetration_plane = hull::furthest_plane_epa(
                (
                    next.points[0].p,
                    next.points[1].p,
                    next.points[2].p,
                    next.points[3].p,
                ),
                |v| a.support(v) - b.support(-v),
            );

            let mp = M4x4::from_cols(
                V4::expand(next.points[0].p, 1.0),
                V4::expand(next.points[1].p, 1.0),
                V4::expand(next.points[2].p, 1.0),
                V4::expand(next.points[3].p, 1.0),
            );

            let ma = M4x4::from_cols(
                V4::expand(next.points[0].a, 1.0),
                V4::expand(next.points[1].a, 1.0),
                V4::expand(next.points[2].a, 1.0),
                V4::expand(next.points[3].a, 1.0),
            );

            let mb = M4x4::from_cols(
                V4::expand(next.points[0].b, 1.0),
                V4::expand(next.points[1].b, 1.0),
                V4::expand(next.points[2].b, 1.0),
                V4::expand(next.points[3].b, 1.0),
            );

            let b = mp.inverse().unwrap().w; // just unwrap directly? (no chance this can't be inverted, right?)

            let p0 = (ma * b).xyz();
            let p1 = (mb * b).xyz();

            let mut hit_info = ContactInfo {
                plane: Plane::from_v4(-min_penetration_plane.to_v4()), // ... flip?
                separation: min_penetration_plane.offset.min(0.0),
                points: (p0, p1),
                impact: (p0 + p1) * 0.5,
                ..Default::default()
            };
            hit_info.fill_simplex(&last);
            assert!(hit_info.separation <= 0.0);
            return hit_info;
        }
        if dot(next.v, next.v) >= dot(last.v, last.v) {
            //println!("Warning: GJK Robustness error (i = {}, n = {}, l = {}): {:?} >= {:?}", iter, dot(next.v, next.v), dot(last.v, last.v), next.v, last.v);
            break;
        }
    }
    // println!("EXIT (fail)");
    debug_assert!(iter < 100); // ...
    last.compute_points()
}

#[derive(Copy, Clone, Debug, Default)]
pub struct ContactPatch {
    pub hit_info: [ContactInfo; 5],
    pub count: usize,
}

impl ContactPatch {
    pub fn new(s0: &dyn Support, s1: &dyn Support, max_sep: f32) -> ContactPatch {
        let mut result: ContactPatch = Default::default();
        result.hit_info[0] = separated(s0, s1, true);
        if result.hit_info[0].separation > max_sep {
            return result;
        }

        let n = result.hit_info[0].plane.normal;
        result.count += 1;
        // return result;
        // let tan = n.orth();
        // let bit = cross(n, tan);
        let qs = Quat::shortest_arc(n, vec3(0.0, 0.0, 1.0));
        let tan = qs.x_dir();
        let bit = qs.y_dir();
        let roll_axes = [tan, bit, -tan, -bit];
        for &raxis in &roll_axes {
            let wiggle_angle = 4.0f32.to_radians();
            let wiggle = Quat::from_axis_angle(raxis, wiggle_angle).must_norm();
            let pivot = result.hit_info[0].points.0;
            let ar = Pose::new(n * 0.2, quat(0.0, 0.0, 0.0, 1.0))
                * Pose::new(-pivot, quat(0.0, 0.0, 0.0, 1.0))
                * Pose::new(vec3(0.0, 0.0, 0.0), wiggle)
                * Pose::new(pivot, quat(0.0, 0.0, 0.0, 1.0));

            let mut next = separated(
                &TransformedSupport {
                    pose: ar,
                    object: s0,
                },
                s1,
                true,
            );

            next.plane.normal = n;
            {
                let p0 = next.points.0;
                next.points.0 = ar.inverse() * p0;
            }
            next.separation = dot(n, next.points.0 - next.points.1);

            let mut matched = false;
            for j in 0..result.count {
                if (next.points.0 - result.hit_info[j].points.0).length() < 0.05
                    || (next.points.1 - result.hit_info[j].points.1).length() < 0.05
                {
                    matched = true;
                    break;
                }
            }

            // let matched = result.hit_info[0..result.count].iter().find(|item|
            //     next.points.0.dist(item.points.0) < 0.05 ||
            //     next.points.1.dist(item.points.1) < 0.05).is_some();

            if !matched {
                let c = result.count;
                result.hit_info[c] = next;
                result.count += 1;
            }
        }
        result
    }
}

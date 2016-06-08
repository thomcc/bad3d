use math::*;
use hull;
use std::default::Default;
// use math::geom;
use support::{TransformedSupport, Support};

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum ContactType {
    Unknown,
    PtPlane,
    EdgeEdge,
    PlanePt,
}

impl Default for ContactType {
    #[inline] fn default() -> ContactType { ContactType::Unknown }
}


#[derive(Copy, Clone, Debug, Default)]
pub struct ContactInfo {
    pub ty: ContactType,
    pub simplex: [V3; 4],
    pub plane: geom::Plane,
    pub impact: V3,
    pub separation: f32,
    pub points: (V3, V3),
    pub time: f32,
}

impl ContactInfo {
    fn fill_simplex(&mut self, src: &Simplex) {
        self.ty = ContactType::Unknown;
        if src.size != 3 {
            return;
        }
        let face_norm = geom::tri_normal(src.points[0].p, src.points[1].p, src.points[2].p);
        let face_proj = dot(self.plane.normal, face_norm);
        let (p0a, p1a, p2a) = (src.points[0].a, src.points[1].a, src.points[2].a);
        let (p0b, p1b, p2b) = (src.points[0].b, src.points[1].b, src.points[2].b);

        if p0a == p1a && p0a == p2a {
            self.ty = ContactType::PtPlane;
            self.simplex = [p0a, p0b, p1b, p2b];
            if face_proj < 0.0 {
                self.simplex.swap(1, 2);
            }
        }
        else if p0b == p1b && p0b == p2b {
            self.ty = ContactType::PlanePt;
            self.simplex = [p0a, p1a, p2a, p2b];
            if face_proj < 0.0 {
                self.simplex.swap(1, 2);
            }
        }
        else if (p0a == p1a || p0a == p2a || p1a == p2a) &&
                (p0b == p1b || p0b == p2b || p1b == p2b) {
            self.ty = ContactType::EdgeEdge;
            self.simplex = [p0a, if p1a != p0a { p1a } else { p2a },
                            p0b, if p1b != p0b { p1b } else { p2b }];
            if (face_proj < 0.0 && p1a != p0a) || (face_proj > 0.0 && p1a == p0a) {
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
        Point { a: a, b: b, p: p, t: 0.0 }
    }

    fn on_sum(a: &Support, b: &Support, n: V3) -> Point {
        let pa = a.support(n);
        let pb = b.support(-n);
        Point::new(pa, pb, pa-pb)
    }
    #[inline]
    fn with_t(&self, t: f32) -> Point {
        Point { t: t, .. *self }
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
    // fn make(v: V3, pts: &[Point]) -> Simplex {
    //     debug_assert!(pts.len() < 4 && pts.len() > 0);
    //     let mut arr: [Point; 4] = Default::default();
    //     for (&a, b) in pts.iter().zip(arr.iter_mut()) {
    //         *b = a;
    //     }
    //     Simplex { v: v, points: arr, size: pts.len() as u32 }
    // }

    fn make2(v: V3, p0: Point, p1: Point) -> Simplex {
        Simplex {
            v: v,
            size: 2,
            points: [p0, p1, Default::default(), Default::default()]
        }
        // debug_assert!(pts.len() < 4 && pts.len() > 0);
        // let mut arr: [Point; 4] = Default::default();
        // for (&a, b) in pts.iter().zip(arr.iter_mut()) {
        //     *b = a;
        // }
        // Simplex { v: v, points: arr, size: pts.len() as u32 }
    }
    fn make3(v: V3, p0: Point, p1: Point, p2: Point) -> Simplex {
        Simplex {
            v: v,
            size: 3,
            points: [p0, p1, p2, Default::default()]
        }
        // debug_assert!(pts.len() < 4 && pts.len() > 0);
        // let mut arr: [Point; 4] = Default::default();
        // for (&a, b) in pts.iter().zip(arr.iter_mut()) {
        //     *b = a;
        // }
        // Simplex { v: v, points: arr, size: pts.len() as u32 }
    }

    fn from_point(w: &Point) -> Simplex {
        Simplex {
            v: w.p,
            size: 1,
            points: [
                w.with_t(1.0),
                Default::default(),
                Default::default(),
                Default::default()
            ]
        }
    }

    fn initial(v: V3) -> Simplex {
        Simplex { v: v, size: 0, .. Default::default() }
    }

    fn finish(&self, w: &Point) -> Simplex {
        debug_assert!(self.size == 3);
        let mut result = Simplex{ v: V3::zero(), size: 4, .. *self };
        result.points[3] = *w;
        result
    }

    fn next1(&self, w: &Point) -> Simplex {
        /* println!("NEXT1"); */
        let s = self.points[0];
        let t = geom::line_project_time(w.p, s.p, V3::zero());
        if t < 0.0 { /* println!("1:T<0"); */ Simplex::from_point(w) }
        else { /* println!("1:ELSE"); */ Simplex::make2(w.p.lerp(s.p, t), s.with_t(t), w.with_t(1.0-t)) }
    }

    fn next2(&self, w: &Point) -> Simplex {
        /* println!("NEXT2"); */
        let s0 = self.points[0];
        let s1 = self.points[1];

        let t0 = geom::line_project_time(w.p, s0.p, V3::zero());
        let t1 = geom::line_project_time(w.p, s1.p, V3::zero());

        let v0 = w.p.lerp(s0.p, t0);
        let v1 = w.p.lerp(s1.p, t1);

        let in_edge0 = towards_origin(v0, s1.p);
        let in_edge1 = towards_origin(v1, s0.p);

        if in_edge0 && in_edge1 { /* println!("2:IN E0 E2"); */ Simplex::make3(geom::tri_project(s0.p, s1.p, w.p, V3::zero()), s0, s1, *w) }
        else if !in_edge0 && t0 > 0.0 { /* println!("2:!IN E0 & T0 > 0"); */ Simplex::make2(v0, s0.with_t(t0), w.with_t(1.0 - t0)) }
        else if !in_edge1 && t1 > 0.0 { /* println!("2:!IN E0 & T0 > 0"); */ Simplex::make2(v1, s1.with_t(t1), w.with_t(1.0 - t1)) }
        else { /* println!("2:ELSE"); */ Simplex::from_point(w) }
    }

    fn next3(&self, w: &Point) -> Simplex {
        /* println!("NEXT3"); */
        let s0 = self.points[0];
        let s1 = self.points[1];
        let s2 = self.points[2];

        let t0 = geom::line_project_time(w.p, s0.p, V3::zero());
        let t1 = geom::line_project_time(w.p, s1.p, V3::zero());
        let t2 = geom::line_project_time(w.p, s2.p, V3::zero());

        let v0 = w.p.lerp(s0.p, t0);
        let v1 = w.p.lerp(s1.p, t1);
        let v2 = w.p.lerp(s2.p, t2);

        let c0 = geom::tri_project(w.p, s1.p, s2.p, V3::zero());
        let c1 = geom::tri_project(w.p, s2.p, s0.p, V3::zero());
        let c2 = geom::tri_project(w.p, s0.p, s1.p, V3::zero());

        let in_p0 = towards_origin(c0, s0.p);
        let in_p1 = towards_origin(c1, s1.p);
        let in_p2 = towards_origin(c2, s2.p);

        let in_p2_e0 = towards_origin(v0, s1.p);
        let in_p2_e1 = towards_origin(v1, s0.p);

        let in_p0_e1 = towards_origin(v1, s2.p);
        let in_p0_e2 = towards_origin(v2, s1.p);

        let in_p1_e2 = towards_origin(v2, s0.p);
        let in_p1_e0 = towards_origin(v0, s2.p);

        if in_p0 && in_p1 && in_p2 { /* println!("3:FINISH"); */ self.finish(w) /* terminated */ }
        else if !in_p2 && in_p2_e0 && in_p2_e1 { /* println!("3:!in_p2 && in_p2_e0 && in_p2_e1 "); */ Simplex::make3(geom::tri_project(s0.p, s1.p, w.p, V3::zero()), s0, s1, *w) }
        else if !in_p0 && in_p0_e1 && in_p0_e2 { /* println!("3:!in_p0 && in_p0_e1 && in_p0_e2 "); */ Simplex::make3(geom::tri_project(s1.p, s2.p, w.p, V3::zero()), s1, s2, *w) }
        else if !in_p1 && in_p1_e2 && in_p1_e0 { /* println!("3:!in_p1 && in_p1_e2 && in_p1_e0 "); */ Simplex::make3(geom::tri_project(s2.p, s0.p, w.p, V3::zero()), s2, s0, *w) }
        else if !in_p1_e0 && !in_p2_e0 && t0 > 0.0 { /* println!("3:!in_p1_e0 && !in_p2_e0 && t0 > 0.0 "); */ Simplex::make2(v0, s0.with_t(t0), w.with_t(1.0 - t0)) }
        else if !in_p2_e1 && !in_p0_e1 && t1 > 0.0 { /* println!("3:!in_p2_e1 && !in_p0_e1 && t1 > 0.0 "); */ Simplex::make2(v1, s1.with_t(t1), w.with_t(1.0 - t1)) }
        else if !in_p0_e2 && !in_p1_e2 && t2 > 0.0 { /* println!("3:!in_p0_e2 && !in_p1_e2 && t2 > 0.0 "); */ Simplex::make2(v2, s2.with_t(t2), w.with_t(1.0 - t2)) }
        else { /* println!("3:ELSE"); */ Simplex::from_point(w) }
    }

    fn next(&self, w: &Point) -> Simplex {
        match self.size {
            0 => { /* println!("NEXT0"); */ Simplex::from_point(w) },
            1 => self.next1(w),
            2 => self.next2(w),
            3 => self.next3(w),
            _ => unreachable!()
        }
    }

    fn compute_points(&self) -> ContactInfo {
        assert!(self.size > 0);
        {
            let mut pts = self.points;
            if self.size == 3 {
                let b = geom::barycentric(pts[0].p, pts[1].p, pts[2].p, self.v);
                pts[0].t = b.x;
                pts[1].t = b.y;
                pts[2].t = b.z;
            }
        }
        let (pa, pb) = self.points[0..(self.size as usize)].iter().fold((V3::zero(), V3::zero()),
            |(pa, pb), p| (pa + p.t*p.a, pb + p.t*p.b));


        let norm = self.v.norm_or(0.0, 0.0, 1.0);
        let impact = (pa + pb) * 0.5;
        let mut hit_info = ContactInfo {
            points: (pa, pb),
            impact: impact,
            separation: pa.dist(pb) + 1.0e-30_f32,
            plane: geom::Plane::from_norm_and_point(norm, impact),
            .. Default::default()
        };
        hit_info.fill_simplex(self);
        hit_info
    }
}

fn epa_tri(w0: usize, w1: usize, w2: usize, pts: &[Point]) -> (usize, usize, usize, V3) {
    let mut v = tri_project(pts[w0].p, pts[w1].p, pts[w2].p, V3::zero());
    let b = barycentric(pts[w0].p, pts[w1].p, pts[w2].p, v);
    if b.x < 0.0 || b.x > 1.0 || b.y < 0.0 || b.y > 1.0 || b.z < 0.0 || b.z > 1.0 {
        v = (pts[w0].p + pts[w1].p + pts[w2].p) / 3.0;
    }
    (w0, w1, w2, v)
}

fn do_hacky_epa(next: &Simplex, last: &mut Simplex, a: &Support, b: &Support) {
    let mut points: Vec<Point> = Vec::from(&next.points[..]);
    let mut tris: Vec<(usize, usize, usize, V3)> = Vec::new();

    tris.push(epa_tri(0, 1, 2, points));
    tris.push(epa_tri(1, 0, 3, points));
    tris.push(epa_tri(2, 1, 3, points));
    tris.push(epa_tri(0, 2, 3, points));

    let mut


}


pub fn separated(a: &Support, b: &Support, find_closest: bool) -> ContactInfo {
    let eps = 0.00001_f32;

    let mut v = Point::on_sum(a, b, vec3(0.0, 0.0, 1.0)).p;
    let mut last = Simplex::initial(v);

    let mut w = Point::on_sum(a, b, -v);
    let mut next = Simplex::from_point(&w);

    let mut iter = 0;
    // println!("ENTER");
    while iter == 0 || (dot(w.p, v) < dot(v, v) - eps) && iter < 100 {
        iter += 1;
        last = next;
        v = last.v;
        w = Point::on_sum(a, b, -v);
        if dot(w.p, v) >= dot(v, v) - (eps + eps * dot(v, v)) { break; }
        if !find_closest && dot(w.p, v) >= 0.0 { break; }

        next = last.next(&w);
        // println!("NEXT COUNT {}, v {:?}", next.size, next.v);
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
                (next.points[0].p, next.points[1].p, next.points[2].p, next.points[3].p),
                |v| a.support(v) - b.support(-v));

            let mp = M4x4::from_cols(V4::expand(next.points[0].p, 1.0), V4::expand(next.points[1].p, 1.0),
                                     V4::expand(next.points[2].p, 1.0), V4::expand(next.points[3].p, 1.0));

            let ma = M4x4::from_cols(V4::expand(next.points[0].a, 1.0), V4::expand(next.points[1].a, 1.0),
                                     V4::expand(next.points[2].a, 1.0), V4::expand(next.points[3].a, 1.0));

            let mb = M4x4::from_cols(V4::expand(next.points[0].b, 1.0), V4::expand(next.points[1].b, 1.0),
                                     V4::expand(next.points[2].b, 1.0), V4::expand(next.points[3].b, 1.0));

            let b = mp.inverse().unwrap().w; // just unwrap directly? (no chance this can't be inverted, right?)

            let p0 = (ma * b).xyz();
            let p1 = (mb * b).xyz();

            let mut hit_info = ContactInfo {
                plane: geom::Plane::from_v4(-min_penetration_plane.to_v4()), // ... flip?
                separation: min_penetration_plane.offset.min(0.0),
                points: (p0, p1),
                impact: (p0 + p1) * 0.5,
                .. Default::default()
            };
            hit_info.fill_simplex(&last);
            assert!(hit_info.separation <= 0.0);
            // println!("EXIT (pass)");
            return hit_info;
        }
        if dot(next.v, next.v) >= dot(last.v, last.v) {
            println!("Warning: GJK Robustness error (i = {}, n = {}, l = {}): {:?} >= {:?}", iter, dot(next.v, next.v), dot(last.v, last.v), next.v, last.v);

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
    pub fn new(s0: &Support, s1: &Support, max_sep: f32) -> ContactPatch {
        let mut result: ContactPatch = Default::default();
        result.hit_info[0] = separated(s0, s1, true);
        if result.hit_info[0].separation > max_sep {
            return result;
        }

        let n = result.hit_info[0].plane.normal;
        result.count += 1;
        // return result;

        let qs = Quat::shortest_arc(n, vec3(0.0, 0.0, 1.0));
        let tan = qs.x_dir();
        let bit = qs.y_dir();
        let roll_axes = [tan, bit, -tan, -bit];
        for &raxis in &roll_axes {
            let wiggle_angle = 4.0f32.to_radians();
            let wiggle = Quat::from_axis_angle(raxis, wiggle_angle).must_norm(); //Quat(V4::expand(raxis * (wiggle_angle * 0.5).sin(), 1.0).must_norm());
            let pivot = result.hit_info[0].points.0;
            let ar = pose::Pose::new(n * 0.2, quat(0.0, 0.0, 0.0, 1.0)) *
                     pose::Pose::new(-pivot, quat(0.0, 0.0, 0.0, 1.0)) *
                     pose::Pose::new(vec3(0.0, 0.0, 0.0), wiggle) *
                     pose::Pose::new(pivot, quat(0.0, 0.0, 0.0, 1.0));

            let mut next = separated(
                &TransformedSupport { pose: ar, object: s0 },
                s1,
                true);

            next.plane.normal = n;
            {
                let p0 = next.points.0;
                next.points.0 = ar.inverse() * p0;
            }

            next.separation = dot(n, next.points.0 - next.points.1);

            let matched = result.hit_info[0..result.count].iter().find(|item|
                next.points.0.dist(item.points.0) < 0.05 ||
                next.points.1.dist(item.points.1) < 0.05).is_some();

            if !matched {
                let c = result.count;
                result.hit_info[c] = next;
                result.count += 1;
            }
        }
        result
    }
}



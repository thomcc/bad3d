use math::*;

use hull;
use wingmesh;
use gjk;
use support::{TransformedSupport};
use std::sync::atomic::{AtomicUsize, Ordering, ATOMIC_USIZE_INIT};
use std::cell::{Ref, RefCell};
use std::collections::{HashSet};
use std::rc::Rc;
use std::f32;

// pub const DELTA_T: f32 = 1.0 / 60.0;
pub const RESTITUTION: f32 = 0.4;
pub const GRAVITY: V3 = V3 { x: 0.0, y: 0.0, z: -10.0 };
pub const COLOUMB: f32 = 0.6;

pub const BIAS_FACTOR_JOINT: f32 = 0.3;
pub const BIAS_FACTOR_POS: f32 = 0.3;
pub const BIAS_FACTOR_NEG: f32 = 0.3;

pub const BALLISTIC_FALLTIME: f32 = 0.2;

pub const MAX_DRIFT: f32 = 0.03;
pub const DAMPING: f32 = 0.15;
pub const EULER_PHYSICS: bool = false;

#[derive(Debug, Clone, Default)]
pub struct Shape {
    pub vertices: Vec<V3>,
    pub tris: Vec<[u16; 3]>,
}

impl Shape {
    #[inline]
    pub fn new(vertices: Vec<V3>, tris: Vec<[u16; 3]>) -> Self {
        Self { vertices, tris }
    }

    #[inline]
    pub fn from_winged(wm: &wingmesh::WingMesh) -> Shape {
        let tris = wm.generate_tris();
        Shape::new(wm.verts.clone(), tris)
    }

    #[inline]
    pub fn new_hull(mut vertices: Vec<V3>) -> Option<Self> {
        if let Some(tris) = hull::compute_hull_trunc(&mut vertices, None) {
            Some(Self { vertices, tris})
        } else {
            None
        }
    }

    #[inline]
    pub fn new_box_at(radii: V3, com: V3) -> Self {
        let size = radii.abs().map(|x| x.max(1.0e-3));
        let mut vertices = Vec::with_capacity(8);
        for &z in &[-size.z, size.z] {
            for &y in &[-size.y, size.y] {
                for &x in &[-size.x, size.x] {
                    vertices.push(vec3(x, y, z) + com)
                }
            }
        }
        // This is so lazy...
        Shape::new_hull(vertices).unwrap()
    }

    #[inline]
    pub fn new_box(radii: V3) -> Self {
        Shape::new_box_at(radii, V3::zero())
    }

    #[inline]
    pub fn new_aabb(min: V3, max: V3) -> Self {
        Shape::new_box_at((max - min) / 2.0, (max + min) / 2.0)
    }

    #[inline]
    pub fn new_octa(radii: V3) -> Self {
        let size = radii.abs().map(|x| x.max(1.0e-3));
        let vertices = vec![
            vec3(-size.x, 0.0, 0.0),
            vec3( size.x, 0.0, 0.0),
            vec3(0.0, -size.y, 0.0),
            vec3(0.0,  size.y, 0.0),
            vec3(0.0, 0.0, -size.z),
            vec3(0.0, 0.0,  size.z),
        ];
        // This is so lazy...
        Shape::new_hull(vertices).unwrap()
    }

    #[inline]
    pub fn volume(&self) -> f32 {
        geom::volume(&self.vertices, &self.tris)
    }

    #[inline]
    pub fn center_of_mass(&self) -> V3 {
        geom::center_of_mass(&self.vertices, &self.tris)
    }

    #[inline]
    pub fn inertia(&self, com: V3) -> M3x3 {
        geom::inertia(&self.vertices, &self.tris, com)
    }
}

impl From<wingmesh::WingMesh> for Shape {
    #[inline]
    fn from(wm: wingmesh::WingMesh) -> Shape {
        let tris = wm.generate_tris();
        Shape::new(wm.verts, tris)
    }
}

pub fn combined_volume(shapes: &[Shape]) -> f32 {
    shapes.iter().fold(0.0, |sum, shape| sum + shape.volume())
}

pub fn combined_center_of_mass(shapes: &[Shape]) -> V3 {
    let mut com = V3::zero();
    let mut vol = 0.0f32;
    for mesh in shapes {
        let v = mesh.volume();
        let c = mesh.center_of_mass();
        vol += v;
        com += c*v;
    }
    com / vol
}

pub fn combined_inertia(shapes: &[Shape], com: V3) -> M3x3 {
    let mut vol = 0.0f32;
    let mut inertia = mat3(0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0);
    for mesh in shapes {
        let v = mesh.volume();
        let i = mesh.inertia(com);
        vol += v;
        inertia += i * v;
    }
    inertia / vol
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct RigidBodyID(usize);

#[derive(Debug)]
pub struct RigidBody {
    pub id: RigidBodyID,

    pub pose: Pose,

    pub mass: f32,
    pub inv_mass: f32,

    pub linear_momentum: V3,
    pub angular_momentum: V3,

    pub inv_tensor_massless: M3x3,
    pub inv_tensor: M3x3,

    pub radius: f32,
    pub inner_radius: f32,

    pub bounds: (V3, V3),

    pub next_pose: Pose,
    pub old_pose: Pose,
    pub start_pose: Pose,

    pub damping: f32,
    pub gravity_scale: f32,
    pub friction: f32,

    pub old_state: (Pose, V3, V3),

    pub collides_with_body: bool,
    pub collides_with_world: bool,

    pub center: V3,

    pub shapes: Vec<Shape>,

    pub ignored: HashSet<RigidBodyID>,
}

pub type RigidBodyRef = Rc<RefCell<RigidBody>>;

static RIGIDBODY_ID_COUNTER: AtomicUsize = ATOMIC_USIZE_INIT;

impl RigidBody {
    #[inline]
    pub fn new_ref(shapes: Vec<Shape>, posn: V3, mass: f32) -> RigidBodyRef {
        Rc::new(RefCell::new(RigidBody::new(shapes, posn, mass)))
    }

    pub fn new(shapes: Vec<Shape>, posn: V3, mass: f32) -> RigidBody {
        let mut res = RigidBody {
            id: RigidBodyID(RIGIDBODY_ID_COUNTER.fetch_add(1, Ordering::SeqCst)),
            pose: Pose::from_translation(posn),
            mass: mass,
            inv_mass: safe_div0(1.0, mass),

            linear_momentum: V3::zero(),
            angular_momentum: V3::zero(),

            inv_tensor_massless: M3x3::identity(),
            inv_tensor: M3x3::identity(),

            radius: 0.0,
            inner_radius: 0.0,
            bounds: (V3::zero(), V3::zero()),

            next_pose: Pose::from_translation(posn),
            old_pose: Pose::from_translation(posn),
            start_pose: Pose::from_translation(posn),

            damping: 0.6,
            friction: COLOUMB,
            gravity_scale: 1.0,

            ignored: HashSet::new(),


            old_state: (Pose::from_translation(posn), V3::zero(), V3::zero()),
            collides_with_body: true,
            collides_with_world: true,

            center: V3::zero(),
            shapes: shapes,
        };

        if res.shapes.len() == 0 {
            res.collides_with_world = false;
            res.collides_with_body = false;
        }

        let com = combined_center_of_mass(&res.shapes[..]);

        for s in res.shapes.iter_mut() {
            for v in s.vertices.iter_mut() {
                *v -= com;
            }
        }

        res.pose.position += com;
        res.next_pose.position = res.pose.position;
        res.old_pose.position = res.pose.position;
        res.start_pose.position = res.pose.position;
        res.old_state.0.position = res.pose.position;

        let inertia = combined_inertia(&res.shapes[..], V3::zero());
        let mi = res.inv_mass;
        res.inv_tensor_massless = inertia.inverse().unwrap();
        res.inv_tensor = res.inv_tensor_massless * mi;

        let bounds = compute_bounds_i(
            &mut res.shapes.iter().flat_map(|shape|
                shape.vertices.iter().map(|v| *v))).unwrap();

        let radius = res.shapes.iter()
            .flat_map(|shape| shape.vertices.iter())
            .fold(0.0, |a, &b| f32::max(a, b.dot(b)));

        res.radius = radius;
        res.bounds = bounds;
        res.center = com;
        res
    }

    #[inline]
    pub fn scale_mass(&mut self, mass: f32) {
        let inv_mass = 1.0 / mass;
        self.mass *= mass;
        self.linear_momentum *= mass;
        self.inv_mass *= inv_mass;
        self.angular_momentum *= mass;
        self.inv_tensor *= inv_mass;
    }

    #[inline]
    pub fn spin(&self) -> V3 {
        self.inv_tensor * self.angular_momentum
    }

    #[inline]
    pub fn position_user(&self) -> V3 {
        self.pose * -self.center
    }

    #[inline]
    pub fn apply_impulse(&mut self, r: V3, impulse: V3) {
        self.linear_momentum += impulse;
        self.angular_momentum += r.cross(impulse);
    }

    #[inline]
    fn init_velocity(&mut self, dt: f32) {
        self.old_state = (self.pose, self.linear_momentum, self.angular_momentum);
        let damp = (1.0 - self.damping.max(DAMPING)).powf(dt);
        self.linear_momentum *= damp;
        self.angular_momentum *= damp;

        let force = GRAVITY*self.mass*self.gravity_scale;
        let torque = V3::zero();
        self.linear_momentum += force*dt;
        self.angular_momentum += torque*dt;

        let om3 = self.pose.orientation.to_mat3();
        self.inv_tensor = (om3 * (self.inv_tensor_massless * self.inv_mass)) * om3.transpose();
    }

    #[inline]
    fn calc_next_pose(&mut self, dt: f32) {
        self.next_pose.position = self.pose.position + self.linear_momentum * self.inv_mass * dt;
        self.next_pose.orientation = if EULER_PHYSICS {
            (self.pose.orientation + diff_q(self.pose.orientation, &(self.inv_tensor_massless*self.inv_mass), self.angular_momentum) * dt).must_norm()
        } else {
            rk_update(self.pose.orientation,
                      self.inv_tensor_massless*self.inv_mass,
                      self.angular_momentum,
                      dt)
        };
        for i in 0..4 {
            if self.next_pose.orientation.0[i].abs() < f32::EPSILON / 4.0 {
                self.next_pose.orientation.0[i] = 0.0;
            }
        }

    }

    #[inline]
    fn update_pose(&mut self) {
        self.old_pose = self.pose;
        self.pose = self.next_pose;
        let om3 = self.pose.orientation.to_mat3();
        self.inv_tensor = (om3 * (self.inv_tensor_massless * self.inv_mass)) * om3.transpose();
    }
}

#[inline]
fn diff_q(orientation: Quat, tensor_inv: &M3x3, angular: V3) -> Quat {
    let norm_orient = orientation.norm_or_identity();
    let orient_matrix = norm_orient.to_mat3();
    let i_inv = orient_matrix * tensor_inv * orient_matrix.transpose();
    let half_spin = (i_inv * angular) * 0.5;
    Quat(V4::expand(half_spin, 0.0)) * norm_orient
}

#[inline]
fn rk_update(s: Quat, tensor_inv: M3x3, angular: V3, dt: f32) -> Quat {
    let (ht, tt, st) = (dt/2.0, dt/3.0, dt/6.0);
    let d1 = diff_q(s,           &tensor_inv, angular);
    let d2 = diff_q(s + d1 * ht, &tensor_inv, angular);
    let d3 = diff_q(s + d2 * ht, &tensor_inv, angular);
    let d4 = diff_q(s + d3 * dt, &tensor_inv, angular);
    (s + d1*st + d2*tt + d3*tt + d4*st).must_norm()//norm_or_q(s)
}

#[inline]
fn rb_map_or<F, T: Copy + Clone>(o: &Option<RigidBodyRef>, default: T, f: F) -> T
    where F: Fn(&Ref<RigidBody>) -> T
{
    if let &Some(ref a) = o {
        f(&a.borrow())
    } else {
        default
    }
}


pub struct AngularConstraint {
    pub bodies: (Option<RigidBodyRef>, Option<RigidBodyRef>),
    pub axis: V3,
    pub torque: f32,
    pub target_spin: f32,
    pub torque_bounds: (f32, f32),
}

impl AngularConstraint {
    #[inline]
    pub fn new(bodies: (Option<RigidBodyRef>, Option<RigidBodyRef>), axis: V3, target_spin: f32, torque_bounds: (f32, f32))
                -> AngularConstraint {
        AngularConstraint {
            bodies: bodies,
            axis: axis,
            target_spin: target_spin,
            torque_bounds: torque_bounds,
            torque: 0.0,
        }
    }

    #[inline]
    fn remove_bias(&mut self) {
        let target_spin = if self.torque_bounds.0 < 0.0 { 0.0 }
                          else { self.target_spin.min(0.0) };
        self.target_spin = target_spin;
    }

    #[inline]
    fn solve(&mut self, dt: f32) {
        if self.target_spin == -f32::MAX {
            return;
        }

        let current_spin =
            rb_map_or(&self.bodies.1, 0.0, |rbr| dot(rbr.spin(), self.axis)) -
            rb_map_or(&self.bodies.0, 0.0, |rbr| dot(rbr.spin(), self.axis));

        let spin_to_torque_inv =
            rb_map_or(&self.bodies.0, 0.0, |rbr| dot(self.axis, rbr.inv_tensor * self.axis)) +
            rb_map_or(&self.bodies.1, 0.0, |rbr| dot(self.axis, rbr.inv_tensor * self.axis));

        debug_assert!(spin_to_torque_inv != 0.0);

        let spin_to_torque = safe_div0(1.0, spin_to_torque_inv);

        let delta_spin = self.target_spin - current_spin;

        let delta_torque = (delta_spin * spin_to_torque).clamp(
            dt*self.torque_bounds.0-self.torque, dt*self.torque_bounds.1-self.torque);

        if let Some(ref mut rb0) = self.bodies.0 {
            rb0.borrow_mut().angular_momentum -= self.axis*delta_torque;
        }

        if let Some(ref mut rb1) = self.bodies.1 {
            rb1.borrow_mut().angular_momentum += self.axis*delta_torque;
        }

        self.torque += delta_torque;
    }

}

pub struct LinearConstraint {
    pub bodies: (Option<RigidBodyRef>, Option<RigidBodyRef>),

    pub positions: (V3, V3),
    pub normal: V3,

    pub target_dist: f32,

    pub unbiased_target_speed: f32,
    pub target_speed: f32,

    pub force_limit: (f32, f32),
    pub impulse_sum: f32,

    pub friction_controller: isize,
}

impl LinearConstraint {
    #[inline]
    pub fn new(
        bodies: (Option<RigidBodyRef>, Option<RigidBodyRef>),
        positions: (V3, V3),
        normal: V3,
        dist: f32,
        targ_speed_nobias: Option<f32>,
        force_lim: Option<(f32, f32)>
    ) -> LinearConstraint {
        let lim = force_lim.unwrap_or((-f32::MAX, f32::MAX));
        LinearConstraint {
            bodies: bodies,
            positions: positions,
            normal: normal,
            target_dist: dist,
            unbiased_target_speed: targ_speed_nobias.unwrap_or(0.0),
            force_limit: (lim.0.min(lim.1), lim.0.max(lim.1)),
            target_speed: 0.0,
            friction_controller: 0,
            impulse_sum: 0.0,
        }
    }

    #[inline]
    pub fn friction_control(&mut self, offset: isize) -> &mut LinearConstraint {
        self.friction_controller = offset;
        self
    }

    #[inline]
    fn remove_bias(&mut self) {
        let target_speed = self.target_speed.min(self.unbiased_target_speed);
        self.target_speed = target_speed;
    }

    #[inline]
    fn controller_impulse(&self, idx: usize, entries: &[LinearConstraint]) -> Option<f32> {
        if self.friction_controller == 0 {
            None
        } else {
            let iidx = (idx as isize) + self.friction_controller;
            Some(entries[iidx as usize].impulse_sum)
        }
    }

    #[inline]
    fn solve(&mut self, dt: f32, controller_impulse: Option<f32>) {
        if let Some(friction_impulse) = controller_impulse {
            let f0 = rb_map_or(&self.bodies.0, 0.0, |rb| rb.friction);
            let f1 = rb_map_or(&self.bodies.1, 0.0, |rb| rb.friction);
            let limit = f0.max(f1) * friction_impulse / dt;
            self.force_limit.1 =  limit;
            self.force_limit.0 = -limit;
        }

        let r0 = rb_map_or(&self.bodies.0, self.positions.0, |rb| rb.pose.orientation * self.positions.0);
        let r1 = rb_map_or(&self.bodies.1, self.positions.1, |rb| rb.pose.orientation * self.positions.1);

        let v0 = rb_map_or(&self.bodies.0, V3::zero(), |rb|
            cross(rb.spin(), r0) + rb.linear_momentum*rb.inv_mass);

        let v1 = rb_map_or(&self.bodies.1, V3::zero(), |rb|
            cross(rb.spin(), r1) + rb.linear_momentum*rb.inv_mass);


        let vn = dot(v1-v0, self.normal);

        let impulse_n  = -self.target_speed - vn;

        let impulse_d0 = rb_map_or(&self.bodies.0, 0.0, |rb|
            rb.inv_mass + dot(cross(rb.inv_tensor * cross(r0, self.normal), r0), self.normal));

        let impulse_d1 = rb_map_or(&self.bodies.1, 0.0, |rb|
            rb.inv_mass + dot(cross(rb.inv_tensor * cross(r1, self.normal), r1), self.normal));

        let impulse_d = impulse_d0 + impulse_d1;

        let impulse = safe_div0(impulse_n, impulse_d).clamp(
            self.force_limit.0*dt - self.impulse_sum,
            self.force_limit.1*dt - self.impulse_sum);

        if let Some(ref mut rb0) = self.bodies.0 {
            rb0.borrow_mut().apply_impulse(r0, self.normal * -impulse);
        }

        if let Some(ref mut rb1) = self.bodies.1 {
            rb1.borrow_mut().apply_impulse(r1, self.normal *  impulse);
        }

        self.impulse_sum += impulse;
    }

}

pub struct ConstraintSet {
    pub linears: Vec<LinearConstraint>,
    pub angulars: Vec<AngularConstraint>,
    pub dt: f32
}

impl ConstraintSet {
    #[inline]
    pub fn new(dt: f32) -> ConstraintSet {
        ConstraintSet {
            linears: Vec::new(),
            angulars: Vec::new(),
            dt: if dt == 0.0 { 1.0 / 60.0 } else { dt }
        }
    }

    #[inline]
    pub fn linear(&mut self, c: LinearConstraint) -> &mut ConstraintSet {
        self.linears.push(c);
        self
    }

    #[inline]
    pub fn angular(&mut self, c: AngularConstraint) -> &mut ConstraintSet {
        self.angulars.push(c);
        self
    }

    #[inline]
    pub fn along_direction(&mut self,
                           r0: Option<RigidBodyRef>, p0: V3,
                           r1: Option<RigidBodyRef>, p1: V3,
                           axis: V3,
                           force_bounds: (f32, f32)) -> &mut ConstraintSet
    {
        let target = rb_map_or(&r1, p1, |rb| rb.pose*p1) -
                     rb_map_or(&r0, p0, |rb| rb.pose*p0);
        self.linear(LinearConstraint::new((r0, r1), (p0, p1), axis, dot(target, axis), None, Some(force_bounds)))
    }

    #[inline]
    pub fn under_plane(&mut self, rbr: RigidBodyRef, plane: geom::Plane, max_force: Option<f32>)
                       -> &mut ConstraintSet {
        // @@TODO: all vertices
        let pos = {
            let rbb = rbr.borrow();
            let crot = rbb.pose.orientation.conj();
            let dir = crot * plane.normal;
            max_dir(&rbb.shapes[0].vertices[..], dir).unwrap()
        };

        self.along_direction(
            None, plane.normal*-plane.offset,
            Some(rbr), pos,
            -plane.normal,
            (0.0, max_force.unwrap_or(f32::MAX)))
    }

    #[inline]
    pub fn nail(&mut self, r0: Option<RigidBodyRef>, p0: V3, r1: Option<RigidBodyRef>, p1: V3)
                -> &mut ConstraintSet {
        let d = rb_map_or(&r1, p1, |rb| rb.pose*p1) -
                rb_map_or(&r0, p0, |rb| rb.pose*p0);
        self.linear(LinearConstraint::new((r0.clone(), r1.clone()), (p0, p1), vec3(1.0, 0.0, 0.0), d.x, None, None));
        self.linear(LinearConstraint::new((r0.clone(), r1.clone()), (p0, p1), vec3(0.0, 1.0, 0.0), d.y, None, None));
        self.linear(LinearConstraint::new((r0,         r1),         (p0, p1), vec3(0.0, 0.0, 1.0), d.z, None, None))
    }

    pub fn range_w(&mut self,
                   rb0: Option<RigidBodyRef>, jb0: Quat,
                   rb1: Option<RigidBodyRef>, jb1: Quat,
                   joint_min: V3,
                   joint_max: V3) -> &mut ConstraintSet
    {
        let joint_min = joint_min.map(|f| f.to_radians());
        let joint_max = joint_max.map(|f| f.to_radians());

        if joint_min.x == 0.0 && joint_max.x == 0.0 &&
           joint_min.y == 0.0 && joint_max.y == 0.0 &&
           joint_min.z < joint_max.z {
            let cb = quat(0.0, -1.0, 0.0, 1.0).must_norm();
            return self.range_w(rb0, jb0*cb, rb1, jb1*cb,
                vec3(joint_min.z.to_degrees(), 0.0, 0.0),
                vec3(joint_max.z.to_degrees(), 0.0, 0.0));
        }

        let r = jb0.conj() * jb1;
        let s = Quat::shortest_arc(vec3(0.0, 0.0, 1.0), r.z_dir());
        let t = s.conj() * r;

        let M3x3 { x: xd1, y: yd1, z: zd1 } = jb1.to_mat3();
        let inv_dt = 1.0 / self.dt;

        if joint_max.x == joint_min.x {
            let spin = 2.0 * (-s.0.x + (joint_min.x*0.5).sin()) * inv_dt;
            self.angular(AngularConstraint::new((rb0.clone(), rb1.clone()),  xd1,  spin, (-f32::MAX, f32::MAX)));
        } else if joint_max.x - joint_min.x < 360.0_f32.to_radians() {
            self.angular(AngularConstraint::new((rb0.clone(), rb1.clone()),  xd1, 2.0 * (-s.0.x + (joint_min.x*0.5).sin()) * inv_dt, (0.0, f32::MAX)));
            self.angular(AngularConstraint::new((rb0.clone(), rb1.clone()), -xd1, 2.0 * ( s.0.x - (joint_max.x*0.5).sin()) * inv_dt, (0.0, f32::MAX)));
        }

        if joint_max.y == joint_min.y {
            self.angular(AngularConstraint::new(
                (rb0.clone(), rb1.clone()), yd1,
                BIAS_FACTOR_JOINT * 2.0 * (-s.0.y + joint_min.y) * inv_dt,
                (-f32::MAX, f32::MAX)));
        } else {
            self.angular(AngularConstraint::new((rb0.clone(), rb1.clone()),  yd1, 2.0 * (-s.0.y + (joint_min.y*0.5).sin()) * inv_dt, (0.0, f32::MAX)));
            self.angular(AngularConstraint::new((rb0.clone(), rb1.clone()), -yd1, 2.0 * ( s.0.y - (joint_max.y*0.5).sin()) * inv_dt, (0.0, f32::MAX)));
        }
        self.angular(AngularConstraint::new((rb0, rb1), zd1,
            BIAS_FACTOR_JOINT * 2.0 * -t.0.z * inv_dt, (-f32::MAX, f32::MAX)))
    }

    pub fn range(&mut self, rb0: Option<RigidBodyRef>, rb1: Option<RigidBodyRef>,
                 frame: Quat, min_lim: V3, max_lim: V3) -> &mut ConstraintSet {
        let q0 = rb_map_or(&rb0, frame, |rb| rb.pose.orientation*frame);
        let q1 = rb_map_or(&rb1, Quat::identity(), |rb| rb.pose.orientation);
        self.range_w(rb0, q0, rb1, q1, min_lim, max_lim)
    }

    pub fn powered_angle(&mut self, r0: Option<RigidBodyRef>, r1: Option<RigidBodyRef>,
                         target: Quat, max_torque: f32) -> &mut ConstraintSet {
        let q0 = rb_map_or(&r0, Quat::identity(), |rb| rb.pose.orientation);
        let q1 = rb_map_or(&r1, Quat::identity(), |rb| rb.pose.orientation);
        let dq = {
            let r = q1 * (q0 * target).conj();
            if r.0.w < 0.0 { -r } else { r }
        };
        // hm... should this be the actual basis for the quat instead?

        let axis = dq.0.xyz().norm_or(0.0, 0.0, 1.0);
        let binormal = axis.orth();
        let normal = cross(axis, binormal);
        // let (axis, binormal, normal) = dq.0.xyz().basis();
        let inv_dt = 1.0 / self.dt;
        self.angular(AngularConstraint::new(
            (r0.clone(), r1.clone()), axis,
            -BIAS_FACTOR_JOINT*dq.0.w.clamp(-1.0, 1.0).acos()*2.0 * inv_dt,
            (-max_torque, max_torque)));

        self.angular(AngularConstraint::new(
            (r0.clone(), r1.clone()), binormal, 0.0, (-max_torque, max_torque)));

        self.angular(AngularConstraint::new(
            (r0.clone(), r1.clone()), normal, 0.0, (-max_torque, max_torque)))
    }


    pub fn cone_angle(&mut self, r0: Option<RigidBodyRef>, n0: V3, r1: RigidBodyRef, n1: V3,
                      angle_degrees: f32) -> &mut ConstraintSet {
        let equality = angle_degrees == 0.0;
        let a0 = rb_map_or(&r0, n0, |rb| rb.pose.orientation * n0);
        let a1 = r1.borrow().pose.orientation * n1;
        let axis = cross(a1, a0).norm_or(0.0, 0.0, 1.0);
        let rb_angle = clamp(dot(a0, a1), 0.0, 1.0).acos();
        let delta_angle = rb_angle - angle_degrees.to_radians();
        let target_spin = (if equality { BIAS_FACTOR_JOINT } else { 1.0 }) * delta_angle / self.dt;
        let torque_min = if angle_degrees > 0.0 { 0.0 } else { -f32::MAX };
        self.angular(AngularConstraint::new(
            (r0, Some(r1)), axis, target_spin, (torque_min, f32::MAX)))
    }

    fn constrain_contacts(&mut self, contacts: &[PhysicsContact]) {
        for c in contacts {
            let cc = c.contact;

            let r0 = rb_map_or(&c.bodies.0, V3::zero(), |rb| cc.points.0 - rb.pose.position);
            let r1 = rb_map_or(&c.bodies.1, V3::zero(), |rb| cc.points.1 - rb.pose.position);

            let v0 = rb_map_or(&c.bodies.0, V3::zero(), |rb| cross(rb.spin(), r0) + rb.linear_momentum * rb.inv_mass);
            let v1 = rb_map_or(&c.bodies.1, V3::zero(), |rb| cross(rb.spin(), r1) + rb.linear_momentum * rb.inv_mass);

            let v = v0 - v1;

            let min_sep = MAX_DRIFT * 0.25;
            let sep = cc.separation;

            let bounce_vel = 0.0f32.max(
                (-dot(cc.plane.normal, v) - GRAVITY.length() * BALLISTIC_FALLTIME) * RESTITUTION);

            let q = Quat::shortest_arc(vec3(0.0, 0.0, 1.0), -cc.plane.normal);


            // let normal = q.z_dir();
            let tangent = q.x_dir();
            let binormal = q.y_dir();

            self.linears.push(LinearConstraint::new(
                c.bodies.clone(),
                c.positions,
                -cc.plane.normal,
                sep.min((sep - min_sep) * BIAS_FACTOR_POS),
                Some(-bounce_vel),
                Some((0.0, f32::MAX))
            ));


            self.linears.push(LinearConstraint::new(c.bodies.clone(), c.positions, binormal, 0.0, Some(0.0), Some((0.0, 0.0))));
            self.linears.last_mut().unwrap().friction_control(-1);

            self.linears.push(LinearConstraint::new(c.bodies.clone(), c.positions, tangent, 0.0, Some(0.0), Some((0.0, 0.0))));
            self.linears.last_mut().unwrap().friction_control(-2);
        }
    }
}

#[derive(Clone)]
pub struct PhysicsContact {
    pub contact: gjk::ContactInfo,
    pub bodies: (Option<RigidBodyRef>, Option<RigidBodyRef>),
    pub positions: (V3, V3),
}

impl PhysicsContact {
    pub fn new(b0: Option<RigidBodyRef>, b1: Option<RigidBodyRef>, c: gjk::ContactInfo) -> PhysicsContact {
        let p0 = rb_map_or(&b0, c.points.0, |rb| rb.pose.inverse() * c.points.0);
        let p1 = rb_map_or(&b1, c.points.1, |rb| rb.pose.inverse() * c.points.1);
        PhysicsContact {
            bodies: (b0, b1),
            positions: (p0, p1),
            contact: c,
        }
    }
}

fn find_world_contacts(
    bodies: &[RigidBodyRef],
    world_cells: &[Vec<V3>],
    dt: f32
) -> Vec<PhysicsContact> {
    let mut result = Vec::new();
    for body_ref in bodies.iter() {
        let body = body_ref.borrow();
        if !body.collides_with_world {
            continue;
        }
        let distance_range = MAX_DRIFT.max(body.linear_momentum.length() * dt * body.inv_mass);
        for shape in body.shapes.iter() {
            for cell in world_cells.iter() {
                let patch = gjk::ContactPatch::new(
                    &TransformedSupport{
                        pose: body.pose,
                        object: &&shape.vertices[..]
                    },
                    &&cell[..],
                    distance_range
                );
                for contact in &patch.hit_info[0..patch.count] {
                    result.push(PhysicsContact::new(Some(body_ref.clone()), None, *contact));
                }
            }
        }
    }
    result
}


fn find_body_contacts(bodies: &[RigidBodyRef], dt: f32) -> Vec<PhysicsContact> {
    let mut result = Vec::new();
    for i in 0..bodies.len() {
        let b0 = bodies[i].borrow();
        if !b0.collides_with_body {
            continue;
        }
        for j in i+1..bodies.len() {
            let b1 = bodies[j].borrow();
            if !b1.collides_with_body {
                continue;
            }
            if b0.pose.position.dist(b1.pose.position) > b0.radius + b1.radius {
                continue;
            }
            if b0.ignored.contains(&b1.id) || b1.ignored.contains(&b0.id) {
                continue;
            }
            let distance_range = MAX_DRIFT.max(b0.linear_momentum.length() * dt * b0.inv_mass)
                                          .max(b1.linear_momentum.length() * dt * b1.inv_mass);
            for s0 in b0.shapes.iter() {
                for s1 in b1.shapes.iter() {
                    let patch = gjk::ContactPatch::new(
                        &TransformedSupport { pose: b0.pose, object: &&s0.vertices[..] },
                        &TransformedSupport { pose: b1.pose, object: &&s1.vertices[..] },
                        distance_range);
                    for contact in &patch.hit_info[0..patch.count] {
                        result.push(PhysicsContact::new(
                            Some(bodies[i].clone()),
                            Some(bodies[j].clone()),
                            *contact));
                    }
                }
            }
        }
    }
    result
}

const PHYS_ITER: usize = 16;
const PHYS_POST_ITER: usize = 8;

pub fn update_physics(rbs: &mut [RigidBodyRef],
                      constraints: &mut ConstraintSet,
                      world_geom: &[Vec<V3>],
                      dt: f32)
{
    for rb in rbs.iter_mut() {
        rb.borrow_mut().init_velocity(dt);
    }

    let world_contacts = find_world_contacts(rbs, world_geom, dt);
    let body_contacts = find_body_contacts(rbs, dt);

    constraints.constrain_contacts(&world_contacts[..]);
    constraints.constrain_contacts(&body_contacts[..]);

    for c in constraints.linears.iter_mut() {
        c.target_speed = c.target_dist / dt;
    }

    for _ in 0..PHYS_ITER {
        for i in 0..constraints.linears.len() {
            let ci = constraints.linears[i].controller_impulse(i, &constraints.linears[..]);
            constraints.linears[i].solve(dt, ci);
        }
        for c in constraints.angulars.iter_mut() {
            c.solve(dt);
        }
    }

    for rb in rbs.iter_mut() {
        rb.borrow_mut().calc_next_pose(dt);
    }

    for c in constraints.linears.iter_mut() {
        c.remove_bias();
    }

    for c in constraints.angulars.iter_mut() {
        c.remove_bias();
    }

    for _ in 0..PHYS_POST_ITER {
        for i in 0..constraints.linears.len() {
            let ci = constraints.linears[i].controller_impulse(i, &constraints.linears[..]);
            constraints.linears[i].solve(dt, ci);
        }
        for c in constraints.angulars.iter_mut() {
            c.solve(dt);
        }
    }

    for rb in rbs.iter_mut() {
        rb.borrow_mut().update_pose();
    }
}




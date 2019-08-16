use crate::core::{gjk, shape};
use crate::gjk::CollisionDetector;
use crate::math::prelude::*;
// use std::cell::RefCell;
// use std::collections::HashSet;
use std::f32;
// use std::rc::Rc;
// use std::sync::atomic::{AtomicUsize, Ordering};
pub use crate::core::shape::Shape;

use handy::{Handle, HandleMap};
use smallvec::{smallvec, SmallVec};

pub use scene::*;

mod scene;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhysParams {
    pub restitution: f32,
    pub gravity: V3,
    pub ballistic_response: f32,
    pub pos_bias: f32,
    pub neg_bias: f32,
    pub joint_bias: f32,
    pub max_drift: f32,
    pub damping: f32,
    pub solver_iterations: usize,
    pub post_solver_iterations: usize,
    pub use_rk4: bool,
}

impl Default for PhysParams {
    fn default() -> Self {
        Self {
            restitution: 0.4,
            gravity: vec3(0.0, 0.0, -9.8),
            pos_bias: 0.3,
            neg_bias: 0.3,
            joint_bias: 0.3,
            damping: 0.15,
            max_drift: 0.03,
            solver_iterations: 16,
            ballistic_response: 0.2,
            post_solver_iterations: 4,
            use_rk4: true,
        }
    }
}

pub const DEFAULT_FRICTION: f32 = 0.6;

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub enum RbMass {
    Specific(f32),
    FromVolume,
    Infinite,
}

impl From<f32> for RbMass {
    #[inline]
    fn from(f: f32) -> Self {
        RbMass::Specific(f)
    }
}

pub const NUM_INLINE_SHAPES: usize = 4;
pub type ShapeVec = SmallVec<[Shape; 4]>;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct RigidBodyID(usize);

#[derive(Debug, Clone)]
pub struct RigidBody {
    pub handle: Handle,
    // pub id: RigidBodyID,
    pub pose: Pose,

    pub mass: f32,
    pub inv_mass: f32,

    pub linear_momentum: V3,
    pub angular_momentum: V3,

    pub inv_tensor_massless: M3x3,
    pub inv_tensor: M3x3,

    pub radius: f32,

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

    pub shapes: ShapeVec,

    pub ignored: SmallVec<[handy::Handle; 4]>,
}

// pub type RigidBodyRef = Rc<RefCell<RigidBody>>;

impl RigidBody {
    // #[inline]
    // pub fn new_ref(shapes: impl Into<ShapeVec>, posn: V3, mass: RbMass) -> RigidBodyRef {
    //     Rc::new(RefCell::new(RigidBody::new_with_pose(shapes.into(), posn.into(), mass)))
    // }

    pub fn new(shapes: impl Into<ShapeVec>, posn: impl Into<Pose>, mass: impl Into<RbMass>) -> RigidBody {
        Self::new_with_pose(shapes.into(), posn.into(), mass.into())
    }

    fn new_with_pose(shapes: ShapeVec, pose: Pose, mass: RbMass) -> RigidBody {
        Self::new_with_options(RigidBodyOptions {
            pose,
            shapes,
            mass,
            ..Default::default()
        })
    }

    pub fn new_with_options(o: RigidBodyOptions) -> RigidBody {
        let mass = match o.mass {
            RbMass::Specific(m) => m,
            RbMass::Infinite => 0.0,
            RbMass::FromVolume => shape::combined_volume(&o.shapes),
        } * o.mass_scale;

        let mut res = RigidBody {
            handle: Handle::EMPTY,
            pose: o.pose,
            mass,
            inv_mass: safe_div0(1.0, mass),

            linear_momentum: V3::zero(),
            angular_momentum: V3::zero(),

            inv_tensor_massless: M3x3::identity(),
            inv_tensor: M3x3::identity(),

            radius: 0.0,
            bounds: (V3::zero(), V3::zero()),

            next_pose: o.pose,
            old_pose: o.pose,
            start_pose: o.pose,

            damping: o.damping,
            friction: o.friction,
            gravity_scale: o.gravity_scale,

            ignored: o.ignored,

            old_state: (o.pose, V3::zero(), V3::zero()),
            collides_with_body: o.collides_with_bodies,
            collides_with_world: o.collides_with_world,

            center: V3::zero(),
            shapes: o.shapes,
        };

        if res.shapes.is_empty() {
            res.collides_with_world = false;
            res.collides_with_body = false;
        }

        let com = shape::combined_center_of_mass(&res.shapes);

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

        let inertia = shape::combined_inertia(&res.shapes[..], V3::zero());
        let mi = res.inv_mass;
        res.inv_tensor_massless = inertia.inverse().unwrap();
        res.inv_tensor = res.inv_tensor_massless * mi;

        let bounds =
            geom::compute_bounds_iter(res.shapes.iter().flat_map(|shape| shape.vertices.iter().copied())).unwrap();

        let radius = res
            .shapes
            .iter()
            .flat_map(|shape| shape.vertices.iter())
            .fold(0.0, |a, &b| f32::max(a, b.length()));

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
    fn init_velocity(&mut self, dt: f32, params: &PhysParams) {
        self.old_state = (self.pose, self.linear_momentum, self.angular_momentum);
        let damp = (1.0 - self.damping.max(params.damping)).powf(dt);
        self.linear_momentum *= damp;
        self.angular_momentum *= damp;

        let force = params.gravity * self.mass * self.gravity_scale;
        let torque = V3::zero();
        self.linear_momentum += force * dt;
        self.angular_momentum += torque * dt;

        let om3 = self.pose.orientation.to_mat3();
        self.inv_tensor = (om3 * (self.inv_tensor_massless * self.inv_mass)) * om3.transpose();
    }

    #[inline]
    fn calc_next_pose(&mut self, dt: f32, params: &PhysParams) {
        self.next_pose.position = self.pose.position + self.linear_momentum * self.inv_mass * dt;
        self.next_pose.orientation = if !params.use_rk4 {
            (self.pose.orientation
                + diff_q(
                    self.pose.orientation,
                    &(self.inv_tensor_massless * self.inv_mass),
                    self.angular_momentum,
                ) * dt)
                .norm_or_identity()
        } else {
            rk_update(
                self.pose.orientation,
                self.inv_tensor_massless * self.inv_mass,
                self.angular_momentum,
                dt,
            )
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
    let (ht, tt, st) = (dt / 2.0, dt / 3.0, dt / 6.0);
    let d1 = diff_q(s, &tensor_inv, angular);
    let d2 = diff_q(s + d1 * ht, &tensor_inv, angular);
    let d3 = diff_q(s + d2 * ht, &tensor_inv, angular);
    let d4 = diff_q(s + d3 * dt, &tensor_inv, angular);
    (s + d1 * st + d2 * tt + d3 * tt + d4 * st).norm_or_identity() //norm_or_q(s)
}

// #[inline]
// fn rb_map_or<T>(o: &Option<RigidBodyRef>, default: T, f: impl Fn(&RigidBody) -> T) -> T {
//     if let Some(a) = o {
//         f(&*a.borrow())
//     } else {
//         default
//     }
// }

#[derive(Clone, Debug)]
pub struct AngularConstraint {
    pub bodies: (Option<Handle>, Option<Handle>),
    pub axis: V3,
    pub torque: f32,
    pub target_spin: f32,
    pub torque_bounds: (f32, f32),
}

impl AngularConstraint {
    #[inline]
    pub fn new(
        bodies: (Option<Handle>, Option<Handle>),
        axis: V3,
        target_spin: f32,
        torque_bounds: (f32, f32),
    ) -> AngularConstraint {
        AngularConstraint {
            bodies,
            axis,
            target_spin,
            torque_bounds,
            torque: 0.0,
        }
    }

    #[inline]
    fn remove_bias(&mut self) {
        let target_spin = if self.torque_bounds.0 < 0.0 {
            0.0
        } else {
            self.target_spin.min(0.0)
        };
        self.target_spin = target_spin;
    }

    #[inline]
    fn solve(&mut self, bodies: &mut HandleMap<RigidBody>, dt: f32) {
        if self.target_spin == -f32::MAX {
            return;
        }

        let (s0, t0) = self
            .bodies
            .0
            .and_then(|h| bodies.get(h))
            .map(|rb| (dot(rb.spin(), self.axis), dot(self.axis, rb.inv_tensor * self.axis)))
            .unwrap_or_default();

        let (s1, t1) = self
            .bodies
            .1
            .and_then(|h| bodies.get(h))
            .map(|rb| (dot(rb.spin(), self.axis), dot(self.axis, rb.inv_tensor * self.axis)))
            .unwrap_or_default();

        let current_spin = s1 - s0;
        let spin_to_torque_inv = t0 + t1;

        // let current_spin = rb_map_or(&self.bodies.1, 0.0, |rbr| dot(rbr.spin(), self.axis))
        //     - rb_map_or(&self.bodies.0, 0.0, |rbr| dot(rbr.spin(), self.axis));

        // let spin_to_torque_inv = rb_map_or(&self.bodies.0, 0.0, |rbr| dot(self.axis, rbr.inv_tensor * self.axis))
        //     + rb_map_or(&self.bodies.1, 0.0, |rbr| dot(self.axis, rbr.inv_tensor * self.axis));

        debug_assert!(spin_to_torque_inv != 0.0);

        let spin_to_torque = safe_div0(1.0, spin_to_torque_inv);

        let delta_spin = self.target_spin - current_spin;

        let delta_torque = clamp(
            delta_spin * spin_to_torque,
            dt * self.torque_bounds.0 - self.torque,
            dt * self.torque_bounds.1 - self.torque,
        );

        if let Some(rb0) = self.bodies.0.and_then(|h| bodies.get_mut(h)) {
            rb0.angular_momentum -= self.axis * delta_torque;
        }

        if let Some(rb1) = self.bodies.1.and_then(|h| bodies.get_mut(h)) {
            rb1.angular_momentum += self.axis * delta_torque;
        }

        self.torque += delta_torque;
    }
}

#[derive(Clone, Debug)]
pub struct LinearConstraint {
    pub bodies: (Option<Handle>, Option<Handle>),

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
        bodies: (Option<Handle>, Option<Handle>),
        positions: (V3, V3),
        normal: V3,
        dist: f32,
        targ_speed_nobias: Option<f32>,
        force_lim: Option<(f32, f32)>,
    ) -> LinearConstraint {
        let lim = force_lim.unwrap_or((-f32::MAX, f32::MAX));
        LinearConstraint {
            bodies,
            positions,
            normal,
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
        self.target_speed = self.target_speed.min(self.unbiased_target_speed);
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
    fn solve(&mut self, bodies: &mut HandleMap<RigidBody>, dt: f32, controller_impulse: Option<f32>) {
        if let Some(friction_impulse) = controller_impulse {
            let f0 = self
                .bodies
                .0
                .and_then(|h| bodies.get(h))
                .map(|rb| rb.friction)
                .unwrap_or(0.0);
            let f1 = self
                .bodies
                .1
                .and_then(|h| bodies.get(h))
                .map(|rb| rb.friction)
                .unwrap_or(0.0);
            let limit = f0.max(f1) * friction_impulse / dt;
            self.force_limit.1 = limit;
            self.force_limit.0 = -limit;
        }

        let (r0, v0, impulse_d0) = self
            .bodies
            .0
            .and_then(|h| bodies.get(h))
            .map(|rb| {
                let r0 = rb.pose.orientation * self.positions.0;
                let v1 = cross(rb.spin(), r0) + rb.linear_momentum * rb.inv_mass;
                let impulse_d0 = rb.inv_mass + dot(cross(rb.inv_tensor * cross(r0, self.normal), r0), self.normal);
                (r0, v1, impulse_d0)
            })
            .unwrap_or((self.positions.0, V3::ZERO, 0.0));

        let (r1, v1, impulse_d1) = self
            .bodies
            .1
            .and_then(|h| bodies.get(h))
            .map(|rb| {
                let r1 = rb.pose.orientation * self.positions.1;
                let v1 = cross(rb.spin(), r1) + rb.linear_momentum * rb.inv_mass;
                let impulse_d1 = rb.inv_mass + dot(cross(rb.inv_tensor * cross(r1, self.normal), r1), self.normal);
                (r1, v1, impulse_d1)
            })
            .unwrap_or((self.positions.1, V3::ZERO, 0.0));

        // let r0 = rb_map_or(&self.bodies.0, self.positions.0, |rb| {
        //     rb.pose.orientation * self.positions.0
        // });
        // let r1 = rb_map_or(&self.bodies.1, self.positions.1, |rb| {
        //     rb.pose.orientation * self.positions.1
        // });

        // let v0 = rb_map_or(&self.bodies.0, V3::zero(), |rb| {
        //     cross(rb.spin(), r0) + rb.linear_momentum * rb.inv_mass
        // });

        // let v1 = rb_map_or(&self.bodies.1, V3::zero(), |rb| {
        //     cross(rb.spin(), r1) + rb.linear_momentum * rb.inv_mass
        // });

        let vn = dot(v1 - v0, self.normal);

        let impulse_n = -self.target_speed - vn;

        // let impulse_d0 = rb_map_or(&self.bodies.0, 0.0, |rb| {
        //     rb.inv_mass + dot(cross(rb.inv_tensor * cross(r0, self.normal), r0), self.normal)
        // });

        // let impulse_d1 = rb_map_or(&self.bodies.1, 0.0, |rb| {
        //     rb.inv_mass + dot(cross(rb.inv_tensor * cross(r1, self.normal), r1), self.normal)
        // });

        let impulse_d = impulse_d0 + impulse_d1;

        let impulse = clamp(
            safe_div0(impulse_n, impulse_d),
            self.force_limit.0 * dt - self.impulse_sum,
            self.force_limit.1 * dt - self.impulse_sum,
        );

        if let Some(rb0) = self.bodies.0.and_then(|h| bodies.get_mut(h)) {
            rb0.apply_impulse(r0, self.normal * -impulse);
        }

        if let Some(rb1) = self.bodies.1.and_then(|h| bodies.get_mut(h)) {
            rb1.apply_impulse(r1, self.normal * impulse);
        }

        self.impulse_sum += impulse;
    }
}

#[derive(Clone, Debug)]
pub struct ConstraintSet {
    pub linears: Vec<LinearConstraint>,
    pub angulars: Vec<AngularConstraint>,
    pub dt: f32,
    pub params: PhysParams,
}

impl ConstraintSet {
    #[inline]
    pub fn new(dt: f32) -> ConstraintSet {
        Self::new_with_params(dt, PhysParams::default())
    }

    #[inline]
    pub fn begin(&mut self, dt: f32, params: PhysParams) {
        self.clear();
        self.dt = if dt == 0.0 { 1.0 / 60.0 } else { dt };
        self.params = params;
    }

    #[inline]
    pub fn new_with_params(dt: f32, params: PhysParams) -> ConstraintSet {
        ConstraintSet {
            linears: Vec::new(),
            angulars: Vec::new(),
            dt: if dt == 0.0 { 1.0 / 60.0 } else { dt },
            params,
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        self.linears.clear();
        self.angulars.clear();
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
    pub fn along_direction(
        &mut self,
        r0: Option<&RigidBody>,
        p0: V3,
        r1: Option<&RigidBody>,
        p1: V3,
        axis: V3,
        force_bounds: (f32, f32),
    ) -> &mut ConstraintSet {
        let target = r1.map(|rb| rb.pose * p1).unwrap_or(p1) - r0.map(|rb| rb.pose * p0).unwrap_or(p0);
        self.linear(LinearConstraint::new(
            (r0.map(|rb| rb.handle), r1.map(|rb| rb.handle)),
            (p0, p1),
            axis,
            dot(target, axis),
            None,
            Some(force_bounds),
        ))
    }

    #[inline]
    pub fn under_plane(&mut self, rb: &RigidBody, plane: Plane, max_force: Option<f32>) -> &mut ConstraintSet {
        // @@TODO: all vertices
        let pos = {
            let crot = rb.pose.orientation.conj();
            let dir = crot * plane.normal;
            geom::max_dir(dir, &rb.shapes[0].vertices).unwrap()
        };

        self.along_direction(
            None,
            plane.normal * -plane.offset,
            Some(rb),
            pos,
            -plane.normal,
            (0.0, max_force.unwrap_or(f32::MAX)),
        )
    }

    #[inline]
    pub fn nail(&mut self, r0: Option<&RigidBody>, p0: V3, r1: Option<&RigidBody>, p1: V3) -> &mut ConstraintSet {
        let d = r1.map(|rb| rb.pose * p1).unwrap_or(p1) - r0.map(|rb| rb.pose * p0).unwrap_or(p0);
        let r0r1 = (r0.map(|rb| rb.handle), r1.map(|rb| rb.handle));
        // let d = rb_map_or(&r1, p1, |rb| rb.pose * p1) - rb_map_or(&r0, p0, |rb| rb.pose * p0);
        self.linear(LinearConstraint::new(
            r0r1,
            (p0, p1),
            vec3(1.0, 0.0, 0.0),
            d.x(),
            None,
            None,
        ));
        self.linear(LinearConstraint::new(
            r0r1,
            (p0, p1),
            vec3(0.0, 1.0, 0.0),
            d.y(),
            None,
            None,
        ));
        self.linear(LinearConstraint::new(
            r0r1,
            (p0, p1),
            vec3(0.0, 0.0, 1.0),
            d.z(),
            None,
            None,
        ))
    }

    pub fn range_w(
        &mut self,
        rb0: Option<&RigidBody>,
        jb0: Quat,
        rb1: Option<&RigidBody>,
        jb1: Quat,
        joint_min: V3,
        joint_max: V3,
    ) -> &mut ConstraintSet {
        let joint_min = joint_min.map(|f| f.to_radians());
        let joint_max = joint_max.map(|f| f.to_radians());

        if joint_min.x() == 0.0
            && joint_max.x() == 0.0
            && joint_min.y() == 0.0
            && joint_max.y() == 0.0
            && joint_min.z() < joint_max.z()
        {
            let cb = quat(0.0, -1.0, 0.0, 1.0).must_norm();
            return self.range_w(
                rb0,
                jb0 * cb,
                rb1,
                jb1 * cb,
                vec3(joint_min.z().to_degrees(), 0.0, 0.0),
                vec3(joint_max.z().to_degrees(), 0.0, 0.0),
            );
        }
        let handles = (rb0.map(|rb| rb.handle), rb1.map(|rb| rb.handle));

        let r = jb0.conj() * jb1;
        let s = Quat::shortest_arc(vec3(0.0, 0.0, 1.0), r.z_dir());
        let t = s.conj() * r;

        let M3x3 { x: xd1, y: yd1, z: zd1 } = jb1.to_mat3();
        let inv_dt = 1.0 / self.dt;

        if joint_max.x() == joint_min.x() {
            let spin = 2.0 * (-s.0.x() + (joint_min.x() * 0.5).sin()) * inv_dt;
            self.angular(AngularConstraint::new(handles, xd1, spin, (-f32::MAX, f32::MAX)));
        } else if joint_max.x() - joint_min.x() < 360.0_f32.to_radians() {
            self.angular(AngularConstraint::new(
                handles,
                xd1,
                2.0 * (-s.0.x() + (joint_min.x() * 0.5).sin()) * inv_dt,
                (0.0, f32::MAX),
            ));
            self.angular(AngularConstraint::new(
                handles,
                -xd1,
                2.0 * (s.0.x() - (joint_max.x() * 0.5).sin()) * inv_dt,
                (0.0, f32::MAX),
            ));
        }

        if joint_max.y() == joint_min.y() {
            self.angular(AngularConstraint::new(
                handles,
                yd1,
                self.params.joint_bias * 2.0 * (-s.0.y() + joint_min.y()) * inv_dt,
                (-f32::MAX, f32::MAX),
            ));
        } else {
            self.angular(AngularConstraint::new(
                handles,
                yd1,
                2.0 * (-s.0.y() + (joint_min.y() * 0.5).sin()) * inv_dt,
                (0.0, f32::MAX),
            ));
            self.angular(AngularConstraint::new(
                handles,
                -yd1,
                2.0 * (s.0.y() - (joint_max.y() * 0.5).sin()) * inv_dt,
                (0.0, f32::MAX),
            ));
        }
        self.angular(AngularConstraint::new(
            handles,
            zd1,
            self.params.joint_bias * 2.0 * -t.0.z() * inv_dt,
            (-f32::MAX, f32::MAX),
        ))
    }

    pub fn range(
        &mut self,
        rb0: Option<&RigidBody>,
        rb1: Option<&RigidBody>,
        frame: Quat,
        min_lim: V3,
        max_lim: V3,
    ) -> &mut ConstraintSet {
        let q0 = rb0.map(|rb| rb.pose.orientation * frame).unwrap_or(frame);
        let q1 = rb1.map(|rb| rb.pose.orientation).unwrap_or_default();
        self.range_w(rb0, q0, rb1, q1, min_lim, max_lim)
    }

    pub fn powered_angle(
        &mut self,
        r0: Option<&RigidBody>,
        r1: Option<&RigidBody>,
        target: Quat,
        max_torque: f32,
    ) -> &mut ConstraintSet {
        let q0 = r0.map(|rb| rb.pose.orientation).unwrap_or_default();
        let q1 = r1.map(|rb| rb.pose.orientation).unwrap_or_default();
        let dq = {
            let r = q1 * (q0 * target).conj();
            if r.0.w() < 0.0 {
                -r
            } else {
                r
            }
        };
        let handles = (r0.map(|rb| rb.handle), r1.map(|rb| rb.handle));
        // hm... should this be the actual basis for the quat instead?

        let axis = dq.0.xyz().norm_or(0.0, 0.0, 1.0);
        let binormal = axis.orth();
        let normal = cross(axis, binormal);
        // let (axis, binormal, normal) = dq.0.xyz().basis();
        let inv_dt = 1.0 / self.dt;
        self.angular(AngularConstraint::new(
            handles,
            axis,
            -self.params.joint_bias * clamp(dq.0.w(), -1.0, 1.0).acos() * 2.0 * inv_dt,
            (-max_torque, max_torque),
        ));

        self.angular(AngularConstraint::new(
            handles,
            binormal,
            0.0,
            (-max_torque, max_torque),
        ));

        self.angular(AngularConstraint::new(handles, normal, 0.0, (-max_torque, max_torque)))
    }

    pub fn cone_angle(
        &mut self,
        r0: Option<&RigidBody>,
        n0: V3,
        r1: &RigidBody,
        n1: V3,
        angle_degrees: f32,
    ) -> &mut ConstraintSet {
        let equality = angle_degrees == 0.0;
        let a0 = r0.map(|rb| rb.pose.orientation * n0).unwrap_or(n0);
        let a1 = r1.pose.orientation * n1;
        let axis = cross(a1, a0).norm_or(0.0, 0.0, 1.0);
        let rb_angle = clamp01(dot(a0, a1)).acos();
        let delta_angle = rb_angle - angle_degrees.to_radians();
        let target_spin = (if equality { self.params.joint_bias } else { 1.0 }) * delta_angle / self.dt;
        let torque_min = if angle_degrees > 0.0 { 0.0 } else { -f32::MAX };
        self.angular(AngularConstraint::new(
            (r0.map(|rb| rb.handle), Some(r1.handle)),
            axis,
            target_spin,
            (torque_min, f32::MAX),
        ))
    }

    fn constrain_contacts(&mut self, bodies: &mut HandleMap<RigidBody>, contacts: &[PhysicsContact]) {
        for c in contacts {
            let cc = c.contact;

            let v0 = c
                .bodies
                .0
                .and_then(|h| bodies.get(h))
                .map(|rb| {
                    let r0 = cc.points.0 - rb.pose.position;
                    cross(rb.spin(), r0) + rb.linear_momentum * rb.inv_mass
                })
                .unwrap_or_default();

            let v1 = c
                .bodies
                .1
                .and_then(|h| bodies.get(h))
                .map(|rb| {
                    let r1 = cc.points.1 - rb.pose.position;
                    cross(rb.spin(), r1) + rb.linear_momentum * rb.inv_mass
                })
                .unwrap_or_default();

            // let v0 = c.bodies.0.and_then(|h| bodies.get(h)).map(|rb| cross(rb.spin(), r0) + rb.linear_momentum * rb.inv_mass).unwrap_or_default();
            // let v1 = c.bodies.1.and_then(|h| bodies.get(h)).map(|rb| cross(rb.spin(), r1) + rb.linear_momentum * rb.inv_mass).unwrap_or_default();

            //  rb_map_or(&c.bodies.0, V3::zero(), |rb| cc.points.0 - rb.pose.position);
            // let r1 = rb_map_or(&c.bodies.1, V3::zero(), |rb| cc.points.1 - rb.pose.position);

            // let v0 = rb_map_or(&c.bodies.0, V3::zero(), |rb| {
            //     cross(rb.spin(), r0) + rb.linear_momentum * rb.inv_mass
            // });
            // let v1 = rb_map_or(&c.bodies.1, V3::zero(), |rb| {
            //     cross(rb.spin(), r1) + rb.linear_momentum * rb.inv_mass
            // });

            let v = v0 - v1;

            let min_sep = self.params.max_drift * 0.25;
            let sep = cc.separation;

            let bounce_vel = 0.0f32.max(
                (-dot(cc.plane.normal, v) - self.params.gravity.length() * self.params.ballistic_response)
                    * self.params.restitution,
            );

            let q = Quat::shortest_arc(vec3(0.0, 0.0, 1.0), -cc.plane.normal);

            let normal = q.z_dir();
            let tangent = q.x_dir();
            let binormal = q.y_dir();

            self.linears.push(LinearConstraint::new(
                c.bodies,
                c.positions,
                normal,
                sep.min((sep - min_sep) * self.params.pos_bias),
                Some(-bounce_vel),
                Some((0.0, f32::MAX)),
            ));

            self.linears.push(LinearConstraint::new(
                c.bodies.clone(),
                c.positions,
                binormal,
                0.0,
                Some(0.0),
                Some((0.0, 0.0)),
            ));
            self.linears.last_mut().unwrap().friction_control(-1);

            self.linears.push(LinearConstraint::new(
                c.bodies.clone(),
                c.positions,
                tangent,
                0.0,
                Some(0.0),
                Some((0.0, 0.0)),
            ));
            self.linears.last_mut().unwrap().friction_control(-2);
        }
    }
}

#[derive(Clone)]
pub struct PhysicsContact {
    pub contact: gjk::ContactInfo,
    pub bodies: (Option<Handle>, Option<Handle>),
    pub positions: (V3, V3),
    pub world: Vec<Shape>,
}

impl PhysicsContact {
    pub fn new(b0: Option<Handle>, b1: Option<Handle>, pos: (V3, V3), c: gjk::ContactInfo) -> PhysicsContact {
        // let p0 = rb_map_or(&b0, c.points.0, |rb| rb.pose.inverse() * c.points.0);
        // let p1 = rb_map_or(&b1, c.points.1, |rb| rb.pose.inverse() * c.points.1);
        PhysicsContact {
            bodies: (b0, b1),
            positions: pos,
            // positions: (p0, p1),
            contact: c,
            world: vec![],
        }
    }
}

fn find_world_contacts(
    cd: &mut CollisionDetector,
    bodies: &[&RigidBody],
    world_cells: &[Shape],
    dt: f32,
    params: &PhysParams,
) -> Vec<PhysicsContact> {
    let mut result = Vec::new();
    for body in bodies {
        if !body.collides_with_world {
            continue;
        }
        let distance_range = params.max_drift.max(body.linear_momentum.length() * dt * body.inv_mass);
        for shape in body.shapes.iter() {
            for cell in world_cells.iter() {
                let patch = cd.find_contact(
                    &shape.vertices[..],
                    body.pose,
                    &cell.vertices[..],
                    Pose::identity(),
                    distance_range,
                );

                for contact in &patch.hit_info[0..patch.count] {
                    result.push(PhysicsContact::new(
                        Some(body.handle),
                        None,
                        (body.pose.inverse() * contact.points.0, contact.points.1),
                        *contact,
                    ));
                }
            }
        }
    }
    result
}

fn find_body_contacts(
    cd: &mut CollisionDetector,
    bodies: &[&RigidBody],
    _dt: f32,
    params: &PhysParams,
) -> Vec<PhysicsContact> {
    let mut result = Vec::new();
    for (i, b0) in bodies.iter().enumerate() {
        if !b0.collides_with_body {
            continue;
        }
        for b1 in &bodies[(i + 1)..] {
            if !b1.collides_with_body {
                continue;
            }
            if b0.pose.position.dist(b1.pose.position) > (b0.radius + b1.radius) * (b0.radius + b1.radius) {
                continue;
            }
            if b0.ignored.contains(&b1.handle) || b1.ignored.contains(&b0.handle) {
                continue;
            }
            // let distance_range = params
            //     .max_drift
            //     .max(b0.linear_momentum.length() * dt * b0.inv_mass)
            //     .max(b1.linear_momentum.length() * dt * b1.inv_mass);
            for s0 in b0.shapes.iter() {
                for s1 in b1.shapes.iter() {
                    let patch = cd.find_contact(
                        &s0.vertices[..],
                        b0.pose,
                        &s1.vertices[..],
                        b1.pose,
                        params.max_drift,
                        // distance_range,
                    );
                    for contact in &patch.hit_info[0..patch.count] {
                        result.push(PhysicsContact::new(
                            Some(b0.handle),
                            Some(b1.handle),
                            (
                                b0.pose.inverse() * contact.points.0,
                                b1.pose.inverse() * contact.points.1,
                            ),
                            *contact,
                        ));
                    }
                }
            }
        }
    }
    result
}

pub fn update_physics(
    bodies: &mut handy::HandleMap<RigidBody>,
    constraints: &mut ConstraintSet,
    world_geom: &[Shape],
    perf: &crate::util::PerfLog,
    cd: &mut CollisionDetector,
) {
    let dt = constraints.dt;
    let _g = perf.begin("physics");
    {
        let _g = perf.begin("  before sim");
        for rb in bodies.iter_mut() {
            rb.init_velocity(dt, &constraints.params);
        }
    }
    let world_contacts;
    let body_contacts;
    {
        let _g = perf.begin("  find contacts");
        let bodies = bodies.iter().collect::<Vec<_>>();
        {
            let _g = perf.begin("    find contacts: world");
            world_contacts = find_world_contacts(cd, &bodies, world_geom, dt, &constraints.params);
        }
        {
            let _g = perf.begin("    find contacts: bodies");
            body_contacts = find_body_contacts(cd, &bodies, dt, &constraints.params);
        }
    }
    {
        let _g = perf.begin("  constrain contacts");
        constraints.constrain_contacts(bodies, &world_contacts[..]);
        constraints.constrain_contacts(bodies, &body_contacts[..]);
    }

    for c in constraints.linears.iter_mut() {
        c.target_speed = c.target_dist / dt;
    }

    {
        let _g = perf.begin("  run solvers: main");
        for _ in 0..constraints.params.solver_iterations {
            for i in 0..constraints.linears.len() {
                let ci = constraints.linears[i].controller_impulse(i, &constraints.linears[..]);
                constraints.linears[i].solve(bodies, dt, ci);
            }
            for c in constraints.angulars.iter_mut() {
                c.solve(bodies, dt);
            }
        }
    }

    {
        let _g = perf.begin("  remove constraint biases");
        for rb in bodies.iter_mut() {
            rb.calc_next_pose(dt, &constraints.params);
        }

        for c in constraints.linears.iter_mut() {
            c.remove_bias();
            // c.target_speed = 0.0;
        }

        for c in constraints.angulars.iter_mut() {
            c.remove_bias();
        }
    }
    {
        let _g = perf.begin("  run solvers: post");
        for _ in 0..constraints.params.post_solver_iterations {
            for i in 0..constraints.linears.len() {
                let ci = constraints.linears[i].controller_impulse(i, &constraints.linears[..]);
                constraints.linears[i].solve(bodies, dt, ci);
            }
            for c in constraints.angulars.iter_mut() {
                c.solve(bodies, dt);
            }
        }
    }
    {
        let _g = perf.begin("  after sim");
        for rb in bodies.iter_mut() {
            rb.update_pose();
        }
    }
}

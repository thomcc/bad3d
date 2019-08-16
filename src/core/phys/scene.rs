use super::*;
use handy::{Handle, HandleMap};

pub struct PhysScene {
    pub bodies: HandleMap<RigidBody>,
    pub world: Vec<Shape>,
    pub constraints: ConstraintSet,
    pub params: PhysParams,
    pub log: crate::util::PerfLog,
    pub coll_api: crate::gjk::CollisionDetector,
}

impl Default for PhysScene {
    fn default() -> Self {
        Self::new(PhysParams::default())
    }
}

impl PhysScene {
    pub fn new(params: PhysParams) -> Self {
        Self {
            bodies: HandleMap::new(),
            constraints: ConstraintSet::new(0.0),
            params,
            world: vec![],
            log: Default::default(),
            coll_api: Default::default(),
        }
    }

    pub fn begin(&mut self, dt: f32) -> &mut ConstraintSet {
        self.constraints.begin(dt, self.params);
        self.log.sections.lock().unwrap().clear();
        &mut self.constraints
    }

    pub fn simulate(&mut self) {
        update_physics(
            &mut self.bodies,
            &mut self.constraints,
            &self.world,
            &self.log,
            &mut self.coll_api,
        );
    }

    #[must_use]
    pub fn add(&mut self) -> BodyBuilder<'_> {
        BodyBuilder::new(self)
    }

    fn insert(&mut self, b: RigidBody) -> Handle {
        let h = self.bodies.insert(b);
        self.bodies[h].handle = h;
        h
    }

    pub fn add_shape(&mut self, pos: impl Into<Pose>, shape: impl Into<Shape>) -> Handle {
        self.insert(RigidBody::new(smallvec![shape.into()], pos, RbMass::FromVolume))
    }

    pub fn add_cube(&mut self, pos: impl Into<Pose>, size: f32) -> Handle {
        chek::gt!(size, 0.0);
        self.add_shape(pos.into(), Shape::new_box(V3::splat(size)))
    }

    pub fn add_box(&mut self, pos: impl Into<Pose>, extents: V3) -> Handle {
        self.add_shape(pos.into(), Shape::new_box(extents))
    }
}

#[derive(Clone, Debug)]
pub struct RigidBodyOptions {
    pub mass: RbMass,
    pub mass_scale: f32,
    pub pose: Pose,
    pub shapes: ShapeVec,
    pub gravity_scale: f32,
    pub friction: f32,
    pub damping: f32,
    pub collides_with_world: bool,
    pub collides_with_bodies: bool,
    pub ignored: SmallVec<[Handle; 4]>,
}

impl Default for RigidBodyOptions {
    #[inline]
    fn default() -> Self {
        Self {
            mass: RbMass::FromVolume,
            mass_scale: 1.0,
            pose: Pose::IDENTITY,
            shapes: smallvec![],
            gravity_scale: 1.0,
            friction: super::DEFAULT_FRICTION,
            damping: 0.6,
            collides_with_world: true,
            collides_with_bodies: true,
            ignored: smallvec![],
        }
    }
}

pub struct BodyBuilder<'a> {
    scene: &'a mut PhysScene,
    pub options: RigidBodyOptions,
}

impl<'a> BodyBuilder<'a> {
    fn new(scene: &'a mut PhysScene) -> Self {
        Self {
            scene,
            options: Default::default(),
        }
    }

    pub fn build(self) -> Handle {
        self.build_with(|_| {})
    }

    pub fn build_with(self, cb: impl FnOnce(&mut RigidBody)) -> Handle {
        let rb = RigidBody::new_with_options(self.options);
        let h = self.scene.insert(rb);
        cb(&mut self.scene.bodies[h]);
        h
    }

    pub fn at<P: Into<Pose>>(mut self, p: P) -> Self {
        self.options.pose = p.into();
        self
    }

    pub fn position(mut self, p: V3) -> Self {
        self.options.pose.position = p;
        self
    }

    pub fn orientation(mut self, o: Quat) -> Self {
        self.options.pose.orientation = o.must_norm();
        self
    }

    pub fn box_collider(mut self, r: V3) -> Self {
        self.options.shapes.push(Shape::new_box(r));
        self
    }

    pub fn cube_collider(mut self, r: f32) -> Self {
        self.options.shapes.push(Shape::new_box(V3::splat(r)));
        self
    }

    pub fn collider<S: Into<Shape>>(mut self, s: S) -> Self {
        self.options.shapes.push(s.into());
        self
    }

    pub fn colliders<I: IntoIterator<Item = Shape>>(mut self, s: I) -> Self {
        self.options.shapes.extend(s);
        self
    }

    pub fn mass<M: Into<RbMass>>(mut self, mass: M) -> Self {
        self.options.mass = mass.into();
        self
    }

    pub fn scale_mass(mut self, s: f32) -> Self {
        self.options.mass_scale = s;
        self
    }

    pub fn infinite_mass(mut self) -> Self {
        self.options.mass = RbMass::Infinite;
        self
    }

    pub fn gravity_scale(mut self, gs: f32) -> Self {
        self.options.gravity_scale = gs;
        self
    }

    pub fn friction(mut self, friction: f32) -> Self {
        self.options.friction = friction;
        self
    }

    pub fn damping(mut self, damping: f32) -> Self {
        self.options.damping = damping;
        self
    }

    pub fn ignores_collisions(mut self, with_world: bool, with_bodies: bool) -> Self {
        self.options.collides_with_world = !with_world;
        self.options.collides_with_bodies = !with_bodies;
        self
    }

    pub fn ignore_body(mut self, h: Handle) -> Self {
        self.options.ignored.push(h);
        self
    }
}

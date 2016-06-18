use math::*;
use math::pose::Pose;
use math::geom::Plane;
use std::Default;

struct Camera {
    pub pose: Pose,
    pub view_angle: f32,
    pub clip_near: f32,
    pub clip_far: f32,
    pub view_dist: f32,
    pub viewport: [i32; 4],// left, bottom, width, height
    pub frustum: [Plane; 6],
    pub view_inv: M4x4,
    pub view: M4x4,
    pub projection: M4x4,
    pub view_projection: M4x4,
}

impl Default for Camera {
    pub fn default() -> Camera {
        Camera {
            pose: Pose::default(),
            view_angle: 60.0,
            clip_far: 512.0,
            clip_near: 0.1,
            view_dist: 0.0,
            viewport: [0, 0, 0, 0],
            frustum: [Default::default(); 4],
            view_inv: M4x4::identity(),
            view: M4x4::identity(),
            projection: M4x4::identity(),
            view_projection: M4x4::identity(),
        }
    }
}

impl Camera {

    pub fn set_size(&mut self, w: i32, h: i32) {
        self.viewport[2] = w;
        self.viewport[3] = h;
    }

    pub fn setup_render(&mut self) {
        self.aspect = if self.viewport[2] != 0 || self.viewport[3] != 0 {
            (self.viewport[2] as f32) / (self.viewport[3] as f32)
        } else {
            // well... hope for the best...
            16.0 / 9.0
        };

        assert_gt!(self.view_angle, 0.0);

        self.projection = M4x4::perspective(self.view_angle.to_radians(),
            self.aspect, self.clip_near, self.clip_far);

        self.view_inv = M4x4::from_pose(self.pose.position, self.pose.orientation);

        self.view = self.view_inv.inverse().expect("non-invertible view matrix?");

        self.view_projection = self.view * self.projection;

        let ha = self.view_angle.to_radians()*0.5;
        let ta = ha.tan();
        let pxl = -ta*aspect;
        let pxr =  ta*aspect;
        let pyb = -ta;
        let pya =  ta;

        let planes = [
            Plane::new(vec3(0.0, 0.0, 1.0), self.clip_near), // near
            Plane::new(vec3(0.0, 0.0,-1.0), -self.clip_far), // far
            Plane::new(geom::tri_normal(V3::zero(), vec3(pxl, pyt, -1.0), vec3(pxl, pyb, -1.0)), 0.0); // left
            Plane::new(geom::tri_normal(V3::zero(), vec3(pxr, pyb, -1.0), vec3(pxr, pyt, -1.0)), 0.0); // right
            Plane::new(geom::tri_normal(V3::zero(), vec3(pxr, pyt, -1.0), vec3(pxl, pyt, -1.0)), 0.0); // bottom
            Plane::new(geom::tri_normal(V3::zero(), vec3(pxl, pyb, -1.0), vec3(pxr, pyb, -1.0)), 0.0); // top
        ];

        for (dst, plane) in self.frustum.iter_mut().zip(planes.iter()) {
            let norm = self.pose.orientation * plane.normal;
            let offset = plane.offset - dot(norm, self.pose.position);
            *dst = Plane::new(norm, offset);
        }
    }

    pub fn test_sphere(&self, c: V3, r: f32) -> bool {
        for
    }

}







use bad3d::{self, math::*};
use shared::input;
use glium::glutin::{VirtualKeyCode as Key};

#[derive(Clone, Debug, Copy)]
pub struct CamUpdate {
    pub dt: f32,

    pub forward_held: bool,
    pub back_held: bool,

    pub ascend_held: bool,
    pub descend_held: bool,

    pub left_held: bool,
    pub right_held: bool,

    pub cursor_delta: V2,
}

impl CamUpdate {
    pub fn from_input(i: &input::InputState) -> Self {
        Self {
            dt: i.dt,
            forward_held: i.key_held(Key::W),
            back_held: i.key_held(Key::S),

            left_held: i.key_held(Key::A),
            right_held: i.key_held(Key::D),

            ascend_held: i.key_held(Key::Q),
            descend_held: i.key_held(Key::E),

            cursor_delta: i.mouse_delta(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct FlyCam {
    pub eye: V3,
    pub look: V3,
    pub up: V3,
    pub view: M4x4,
    pub eye_speed: f32,
    // TODO: `struct Degree(pub f32)`?
    pub mouse_speed: f32,
}

impl FlyCam {
    pub fn update(&mut self, update: CamUpdate) {
        let look_len = self.look.length();
        let up_len = self.up.length();
        debug_assert!(approx_eq_e(look_len, 1.0, 1.0e-6),
                      "look not normalized: {} (len = {})", self.look, look_len);
        debug_assert!(approx_eq_e(up_len, 1.0, 1.0e-6),
                      "up not normalized: {} (len = {})", self.up, look_len);

        let fwd = self.look / look_len;
        let up_norm = self.up / up_len;

        let across = cross(fwd, up_norm).fast_norm();

        // let upward = cross(across, fwd).fast_norm();

        if (update.right_held != update.left_held) || (update.forward_held != update.back_held) {
            let x_mul = (update.right_held as i32 as f32)  - (update.left_held as i32 as f32);
            let z_mul = (update.forward_held as i32 as f32)  - (update.back_held as i32 as f32);
            let xz = (across * x_mul + fwd * z_mul).fast_norm();
            self.eye += xz * (self.eye_speed * update.dt);
        }

        if update.ascend_held != update.descend_held {
            let y_mul = (update.ascend_held as i32 as f32) - (update.descend_held as i32 as f32);
            let y_mvmt = (self.up * y_mul).fast_norm();
            self.eye += y_mvmt * (self.eye_speed * update.dt);
        }

        if update.cursor_delta.x != 0.0 {
            let (ys, yc) = (-update.cursor_delta.x * self.mouse_speed).to_radians().sin_cos();
            let ym = 1.0 - yc;
            let (ux, uy, uz) = up_norm.tup();
            // todo: figure out what's backwards with quat::yaw for this case
            let (uxx, uxy, uxz) = (ux * ux, ux * uy, ux * uz);
            let (uyy, uyz)      = (uy * uy, uy * uz);
            let uzz             = uz * uz;
            let (lx, ly, lz) = self.look.tup();
            self.look = vec3(
                (ym * uxx + yc     ) * lx + (ym * uxy - ys * uz) * ly + (ym * uxz + ys * uy) * lz,
                (ym * uxy + ys * uz) * lx + (ym * uyy + yc     ) * ly + (ym * uyz - ys * ux) * lz,
                (ym * uxz - ys * uy) * lx + (ym * uyz + ys * ux) * ly + (ym * uzz + yc     ) * lz
            ).fast_norm();
        }

        if update.cursor_delta.y != 0.0 {
            let (ps, pc) = (update.cursor_delta.y * self.mouse_speed).to_radians().sin_cos();
            let pm = 1.0 - pc;

            let (ax, ay, az) = across.tup();

            let (axx, axy, axz) = (ax * ax, ax * ay, ax * az);
            let (ayy, ayz)      = (ay * ay, ay * az);
            let azz             = az * az;

            let (lx, ly, lz) = self.look.tup();

            self.look = vec3(
                (pm * axx + pc     ) * lx + (pm * axy + ps * az) * ly + (pm * axz - ps * ay) * lz,
                (pm * axy - ps * az) * lx + (pm * ayy + pc     ) * ly + (pm * ayz + ps * ax) * lz,
                (pm * axz + ps * ay) * lx + (pm * ayz - ps * ax) * ly + (pm * azz + pc     ) * lz,
            ).must_norm();
        }

        let up_norm = self.up.fast_norm();
        let mut f = self.look.fast_norm();

        let s = cross(f, up_norm).fast_norm();

        let u = cross(s, f).fast_norm();

        // negatives for opengl handedness...
        f = -f;
        // let t = M3x3::from_cols(s, u, f) * -self.eye;
        let t = vec3(
            s.x * -self.eye.x + s.y * -self.eye.y + s.z * -self.eye.z,
            u.x * -self.eye.x + u.y * -self.eye.y + u.z * -self.eye.z,
            f.x * -self.eye.x + f.y * -self.eye.y + f.z * -self.eye.z
        );

        self.view = mat4(
            s.x, u.x, f.x, 0.0,
            s.y, u.y, f.y, 0.0,
            s.z, u.z, f.z, 0.0,
            t.x, t.y, t.z, 1.0
        );
    }
}

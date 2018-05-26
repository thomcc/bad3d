
use std::collections::HashSet;
use bad3d;
use bad3d::math::*;
use glium::glutin::{
    VirtualKeyCode,
    EventsLoop,
    Event,
    ElementState,
    KeyboardInput,
    MouseButton,
    WindowEvent,
};

pub struct InputState {
    pub mouse_pos: V2,
    pub mouse_pos_prev: V2,
    pub mouse_vec: V3,
    pub mouse_vec_prev: V3,
    pub view_angle: f32,
    pub size: (u32, u32),
    pub mouse_down: bool,
    pub keys_down: HashSet<VirtualKeyCode>,
    pub key_changes: Vec<(VirtualKeyCode, bool)>,
    pub closed: bool,
}

impl InputState {
    pub fn new(size: (u32, u32), view_angle: f32) -> InputState {
        InputState {
            mouse_pos: vec2(0.0, 0.0),
            mouse_pos_prev: vec2(0.0, 0.0),
            mouse_vec: vec3(0.0, 0.0, 0.0),
            mouse_vec_prev: vec3(0.0, 0.0, 0.0),
            view_angle: view_angle,
            size,
            mouse_down: false,
            keys_down: HashSet::new(),
            key_changes: Vec::new(),
            closed: false
        }
    }

    #[inline]
    pub fn get_projection_matrix(&self, near: f32, far: f32) -> M4x4 {
        M4x4::perspective(self.view_angle.to_radians(),
                          self.size.0 as f32 / self.size.1 as f32, near, far)
    }

    #[inline]
    pub fn mouse_delta(&self) -> V2 {
        self.mouse_pos - self.mouse_pos_prev
    }

    #[inline]
    pub fn dims(&self) -> V2 {
        vec2(self.size.0 as f32, self.size.1 as f32)
    }

    // #[inline]
    // pub fn scaled_mouse_delta(&self) -> V2 {
    //     (self.mouse_pos - self.mouse_pos_prev) / self.dims() * 0.5
    // }

    #[inline]
    pub fn keys_dir(&self, k1: VirtualKeyCode, k2: VirtualKeyCode) -> f32 {
        let v1 = if self.keys_down.contains(&k1) { 1 } else { 0 };
        let v2 = if self.keys_down.contains(&k2) { 1 } else { 0 };
        (v2 - v1) as f32
    }

    #[inline]
    pub fn screen_pos_to_vec(&self, p: V2) -> V3 {
        let spread = (self.view_angle.to_radians() * 0.5).tan();
        let (w, h) = (self.size.0 as f32, self.size.1 as f32);
        let hh = h * 0.5;
        let y = spread * (h - p.y - hh) / hh;
        let x = spread * (p.x - w * 0.5) / hh;
        vec3(x, y, -1.0).normalize().unwrap()
    }

    pub fn update(&mut self, events: &mut EventsLoop) -> bool {
        if self.closed {
            return false;
        }
        let mouse_pos = self.mouse_pos;
        let mouse_vec = self.mouse_vec;
        self.key_changes.clear();

        self.mouse_pos_prev = mouse_pos;
        self.mouse_vec_prev = mouse_vec;

        events.poll_events(|ev| {
            if let Event::WindowEvent { event, .. } = ev {
                match event {
                    WindowEvent::Closed => {
                        self.closed = true;
                    },
                    WindowEvent::Resized(w, h) => {
                        self.size = (w, h);
                    },
                    WindowEvent::Focused(true) => {
                        self.keys_down.clear()
                    },
                    WindowEvent::KeyboardInput {
                        input: KeyboardInput {
                            state,
                            virtual_keycode: Some(vk),
                            ..
                        }, ..
                    } => {
                        let was_pressed = match state {
                            ElementState::Pressed => {
                                self.keys_down.insert(vk);
                                true
                            },
                            ElementState::Released => {
                                self.keys_down.remove(&vk);
                                false
                            }
                        };
                        self.key_changes.push((vk, was_pressed));
                    },
                    WindowEvent::CursorMoved { position, .. } => {
                        self.mouse_pos = vec2(position.0 as f32, position.1 as f32);
                    },
                    WindowEvent::MouseInput { button: MouseButton::Left, state, .. } => {
                        self.mouse_down = state == ElementState::Pressed
                    },
                    _ => {},
                }
            }
        });

        self.mouse_vec = self.screen_pos_to_vec(self.mouse_pos);
        !self.closed
    }
}

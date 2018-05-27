
use std::collections::HashSet;
use std::rc::Rc;
use std::cell::RefCell;

use bad3d;
use bad3d::math::*;

use imgui::{ImGui, ImGuiKey, Ui};

use glium::glutin::{
    VirtualKeyCode,
    EventsLoop,
    Event,
    ElementState,
    KeyboardInput,
    MouseButton,
    WindowEvent,
    TouchPhase,
    MouseScrollDelta,
};

#[derive(Debug, Copy, Clone, Default)]
pub struct Mouse {
    pub pos: V2,
    pub vec: V3,
    pub down: (bool, bool, bool),
    pub wheel: f32,
    pub total_scroll: f32,
}

pub struct InputState {
    pub view_angle: f32,
    pub size: (u32, u32),
    pub mouse: Mouse,
    pub mouse_prev: Mouse,
    pub keys_down: HashSet<VirtualKeyCode>,
    pub key_changes: Vec<(VirtualKeyCode, bool)>,
    pub closed: bool,
    pub gui: Rc<RefCell<ImGui>>,
}

fn init_gui(gui: &mut ImGui) {
    gui.set_imgui_key(ImGuiKey::Tab, 0);
    gui.set_imgui_key(ImGuiKey::LeftArrow, 1);
    gui.set_imgui_key(ImGuiKey::RightArrow, 2);
    gui.set_imgui_key(ImGuiKey::UpArrow, 3);
    gui.set_imgui_key(ImGuiKey::DownArrow, 4);
    gui.set_imgui_key(ImGuiKey::PageUp, 5);
    gui.set_imgui_key(ImGuiKey::PageDown, 6);
    gui.set_imgui_key(ImGuiKey::Home, 7);
    gui.set_imgui_key(ImGuiKey::End, 8);
    gui.set_imgui_key(ImGuiKey::Delete, 9);
    gui.set_imgui_key(ImGuiKey::Backspace, 10);
    gui.set_imgui_key(ImGuiKey::Enter, 11);
    gui.set_imgui_key(ImGuiKey::Escape, 12);
    gui.set_imgui_key(ImGuiKey::A, 13);
    gui.set_imgui_key(ImGuiKey::C, 14);
    gui.set_imgui_key(ImGuiKey::V, 15);
    gui.set_imgui_key(ImGuiKey::X, 16);
    gui.set_imgui_key(ImGuiKey::Y, 17);
    gui.set_imgui_key(ImGuiKey::Z, 18);
}

fn update_keyboard(imgui: &mut ImGui, vk: VirtualKeyCode, pressed: bool) {
    use glium::glutin::VirtualKeyCode as Key;
    match vk {
        Key::Tab => imgui.set_key(0, pressed),
        Key::Left => imgui.set_key(1, pressed),
        Key::Right => imgui.set_key(2, pressed),
        Key::Up => imgui.set_key(3, pressed),
        Key::Down => imgui.set_key(4, pressed),
        Key::PageUp => imgui.set_key(5, pressed),
        Key::PageDown => imgui.set_key(6, pressed),
        Key::Home => imgui.set_key(7, pressed),
        Key::End => imgui.set_key(8, pressed),
        Key::Delete => imgui.set_key(9, pressed),
        Key::Back => imgui.set_key(10, pressed),
        Key::Return => imgui.set_key(11, pressed),
        Key::Escape => imgui.set_key(12, pressed),
        Key::A => imgui.set_key(13, pressed),
        Key::C => imgui.set_key(14, pressed),
        Key::V => imgui.set_key(15, pressed),
        Key::X => imgui.set_key(16, pressed),
        Key::Y => imgui.set_key(17, pressed),
        Key::Z => imgui.set_key(18, pressed),
        Key::LControl | Key::RControl => imgui.set_key_ctrl(pressed),
        Key::LShift | Key::RShift => imgui.set_key_shift(pressed),
        Key::LAlt | Key::RAlt => imgui.set_key_alt(pressed),
        Key::LWin | Key::RWin => imgui.set_key_super(pressed),
        _ => {}
    }
}

impl InputState {
    pub fn new(size: (u32, u32), view_angle: f32, gui: Rc<RefCell<ImGui>>) -> InputState {
        init_gui(&mut gui.borrow_mut());
        InputState {
            mouse: Default::default(),
            mouse_prev: Default::default(),
            view_angle: view_angle,
            size,
            keys_down: HashSet::new(),
            key_changes: Vec::new(),
            closed: false,
            gui
        }
    }

    #[inline]
    pub fn get_projection_matrix(&self, near: f32, far: f32) -> M4x4 {
        M4x4::perspective(self.view_angle.to_radians(),
                          self.size.0 as f32 / self.size.1 as f32, near, far)
    }

    #[inline]
    pub fn mouse_delta(&self) -> V2 {
        self.mouse.pos - self.mouse_prev.pos
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

    pub fn shift_down(&self) -> bool {
        self.keys_down.contains(&VirtualKeyCode::LShift) ||
        self.keys_down.contains(&VirtualKeyCode::RShift)
    }

    pub fn ctrl_down(&self) -> bool {
        self.keys_down.contains(&VirtualKeyCode::LControl) ||
        self.keys_down.contains(&VirtualKeyCode::RControl)
    }

    pub fn update(&mut self, events: &mut EventsLoop) -> bool {
        if self.closed {
            return false;
        }

        self.key_changes.clear();
        self.mouse_prev = self.mouse;
        self.mouse.wheel = 0.0;

        events.poll_events(|ev| {
            if let Event::WindowEvent { event, .. } = ev {
                match event {
                    WindowEvent::Closed => {
                        self.closed = true;
                    }
                    WindowEvent::Resized(w, h) => {
                        self.size = (w, h);
                    }
                    WindowEvent::Focused(true) => {
                        self.keys_down.clear()
                    }
                    WindowEvent::KeyboardInput { input, .. } => {
                        let vk = if let Some(kc) = input.virtual_keycode { kc } else {
                            return;
                        };
                        let was_pressed = match input.state {
                            ElementState::Pressed => {
                                self.keys_down.insert(vk);
                                true
                            }
                            ElementState::Released => {
                                self.keys_down.remove(&vk);
                                false
                            }
                        };
                        self.key_changes.push((vk, was_pressed));
                        update_keyboard(&mut self.gui.borrow_mut(), vk, was_pressed);
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        self.mouse.pos = vec2(position.0 as f32, position.1 as f32);
                    }
                    WindowEvent::MouseInput { button, state, .. } => {
                        let pressed = state == ElementState::Pressed;
                        match button {
                            MouseButton::Left => { self.mouse.down.0 = pressed; }
                            MouseButton::Middle => { self.mouse.down.1 = pressed; }
                            MouseButton::Right => { self.mouse.down.2 = pressed; }
                            _ => {}
                        }
                    }
                    WindowEvent::ReceivedCharacter(c) => {
                        self.gui.borrow_mut().add_input_character(c);
                    }
                    WindowEvent::MouseWheel { delta, phase: TouchPhase::Moved, .. } => {
                        match delta {
                            MouseScrollDelta::LineDelta(_, y) => {
                                self.mouse.wheel = y * 10.0;
                            }
                            MouseScrollDelta::PixelDelta(_, y) => {
                                self.mouse.wheel = y;
                            }
                        }
                    }
                    _ => {}
                }
            }
        });
        self.mouse.vec = self.screen_pos_to_vec(self.mouse.pos);
        let mut gui = self.gui.borrow_mut();
        let scale = gui.display_framebuffer_scale();
        gui.set_mouse_pos(self.mouse.pos.x / scale.0,
                          self.mouse.pos.y / scale.1);
        gui.set_mouse_down(&[self.mouse.down.0,
                             self.mouse.down.1,
                             self.mouse.down.2,
                             false, false]);
        gui.set_mouse_wheel(self.mouse.wheel / scale.1);
        self.mouse.total_scroll += self.mouse.wheel / scale.1;
        !self.closed
    }
}

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use bad3d::{self, prelude::*};

use crate::shared::DemoWindow;
use imgui::{ImGui, ImGuiKey, Ui};

use glium::glutin::{
    ElementState, Event, EventsLoop, KeyboardInput, MouseButton, MouseScrollDelta, TouchPhase,
    VirtualKeyCode, WindowEvent,
};
use glium::Display;

#[derive(Debug, Copy, Clone, Default)]
pub struct Mouse {
    pub pos: V2,
    pub vec: V3,
    pub down: (bool, bool, bool),
    pub wheel: f32,
    pub total_scroll: f32,
}

#[derive(Debug, Copy, Clone, Default)]
pub struct KeyState {
    pub down: bool,
    pub changed: bool,
}

pub struct InputState {
    pub view_angle: f32,
    pub size: (u32, u32),
    pub mouse: Mouse,
    pub mouse_prev: Mouse,
    pub keys: HashMap<VirtualKeyCode, KeyState>,
    pub key_changes: Vec<(VirtualKeyCode, bool)>,
    pub closed: bool,
    pub gui: Rc<RefCell<ImGui>>,
    pub mouse_grabbed: bool,
    pub dt: f32,
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
    use imgui::{ImGuiCol, ImVec2, ImVec4};

    let style = gui.style_mut();
    // style.child_window_rounding = 3.0;
    style.window_rounding = 0.0;
    style.grab_rounding = 0.0;
    style.scrollbar_rounding = 3.0;
    style.frame_rounding = 3.0;
    style.window_title_align = ImVec2::new(0.5, 0.5);
    style.colors[ImGuiCol::Text as usize] = ImVec4::new(0.73, 0.73, 0.73, 1.00);
    style.colors[ImGuiCol::TextDisabled as usize] = ImVec4::new(0.50, 0.50, 0.50, 1.00);
    style.colors[ImGuiCol::WindowBg as usize] = ImVec4::new(0.26, 0.26, 0.26, 0.95);
    // style.colors[ImGuiCol::ChildWindowBg as usize]        = ImVec4::new(0.28, 0.28, 0.28, 1.00);
    style.colors[ImGuiCol::PopupBg as usize] = ImVec4::new(0.26, 0.26, 0.26, 1.00);
    style.colors[ImGuiCol::Border as usize] = ImVec4::new(0.26, 0.26, 0.26, 1.00);
    style.colors[ImGuiCol::BorderShadow as usize] = ImVec4::new(0.26, 0.26, 0.26, 1.00);
    style.colors[ImGuiCol::FrameBg as usize] = ImVec4::new(0.16, 0.16, 0.16, 1.00);
    style.colors[ImGuiCol::FrameBgHovered as usize] = ImVec4::new(0.16, 0.16, 0.16, 1.00);
    style.colors[ImGuiCol::FrameBgActive as usize] = ImVec4::new(0.16, 0.16, 0.16, 1.00);
    style.colors[ImGuiCol::TitleBg as usize] = ImVec4::new(0.36, 0.36, 0.36, 1.00);
    style.colors[ImGuiCol::TitleBgCollapsed as usize] = ImVec4::new(0.36, 0.36, 0.36, 1.00);
    style.colors[ImGuiCol::TitleBgActive as usize] = ImVec4::new(0.36, 0.36, 0.36, 1.00);
    style.colors[ImGuiCol::MenuBarBg as usize] = ImVec4::new(0.26, 0.26, 0.26, 1.00);
    style.colors[ImGuiCol::ScrollbarBg as usize] = ImVec4::new(0.21, 0.21, 0.21, 1.00);
    style.colors[ImGuiCol::ScrollbarGrab as usize] = ImVec4::new(0.36, 0.36, 0.36, 1.00);
    style.colors[ImGuiCol::ScrollbarGrabHovered as usize] = ImVec4::new(0.36, 0.36, 0.36, 1.00);
    style.colors[ImGuiCol::ScrollbarGrabActive as usize] = ImVec4::new(0.36, 0.36, 0.36, 1.00);
    // style.colors[ImGuiCol::ComboBg as usize]              = ImVec4::new(0.32, 0.32, 0.32, 1.00);
    style.colors[ImGuiCol::CheckMark as usize] = ImVec4::new(0.78, 0.78, 0.78, 1.00);
    style.colors[ImGuiCol::SliderGrab as usize] = ImVec4::new(0.74, 0.74, 0.74, 1.00);
    style.colors[ImGuiCol::SliderGrabActive as usize] = ImVec4::new(0.74, 0.74, 0.74, 1.00);
    style.colors[ImGuiCol::Button as usize] = ImVec4::new(0.36, 0.36, 0.36, 1.00);
    style.colors[ImGuiCol::ButtonHovered as usize] = ImVec4::new(0.43, 0.43, 0.43, 1.00);
    style.colors[ImGuiCol::ButtonActive as usize] = ImVec4::new(0.11, 0.11, 0.11, 1.00);
    style.colors[ImGuiCol::Header as usize] = ImVec4::new(0.36, 0.36, 0.36, 1.00);
    style.colors[ImGuiCol::HeaderHovered as usize] = ImVec4::new(0.36, 0.36, 0.36, 1.00);
    style.colors[ImGuiCol::HeaderActive as usize] = ImVec4::new(0.36, 0.36, 0.36, 1.00);
    style.colors[ImGuiCol::Separator as usize] = ImVec4::new(0.39, 0.39, 0.39, 1.00);
    style.colors[ImGuiCol::SeparatorHovered as usize] = ImVec4::new(0.26, 0.59, 0.98, 1.00);
    style.colors[ImGuiCol::SeparatorActive as usize] = ImVec4::new(0.26, 0.59, 0.98, 1.00);
    style.colors[ImGuiCol::ResizeGrip as usize] = ImVec4::new(0.36, 0.36, 0.36, 1.00);
    style.colors[ImGuiCol::ResizeGripHovered as usize] = ImVec4::new(0.26, 0.59, 0.98, 1.00);
    style.colors[ImGuiCol::ResizeGripActive as usize] = ImVec4::new(0.26, 0.59, 0.98, 1.00);
    style.colors[ImGuiCol::CloseButton as usize] = ImVec4::new(0.59, 0.59, 0.59, 1.00);
    style.colors[ImGuiCol::CloseButtonHovered as usize] = ImVec4::new(0.98, 0.39, 0.36, 1.00);
    style.colors[ImGuiCol::CloseButtonActive as usize] = ImVec4::new(0.98, 0.39, 0.36, 1.00);
    style.colors[ImGuiCol::PlotLines as usize] = ImVec4::new(0.39, 0.39, 0.39, 1.00);
    style.colors[ImGuiCol::PlotLinesHovered as usize] = ImVec4::new(1.00, 0.43, 0.35, 1.00);
    style.colors[ImGuiCol::PlotHistogram as usize] = ImVec4::new(0.90, 0.70, 0.00, 1.00);
    style.colors[ImGuiCol::PlotHistogramHovered as usize] = ImVec4::new(1.00, 0.60, 0.00, 1.00);
    style.colors[ImGuiCol::TextSelectedBg as usize] = ImVec4::new(0.32, 0.52, 0.65, 1.00);
    style.colors[ImGuiCol::ModalWindowDarkening as usize] = ImVec4::new(0.20, 0.20, 0.20, 0.50);
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
            view_angle,
            size,
            keys: HashMap::new(),
            key_changes: Vec::new(),
            closed: false,
            gui,
            mouse_grabbed: false,
            dt: 1.0 / 60.0,
        }
    }

    #[inline]
    pub fn get_projection_matrix(&self, near: f32, far: f32) -> M4x4 {
        M4x4::perspective(
            self.view_angle.to_radians(),
            self.size.0 as f32 / self.size.1 as f32,
            near,
            far,
        )
    }

    #[inline]
    pub fn mouse_delta(&self) -> V2 {
        if self.mouse_grabbed {
            self.mouse.pos
        } else {
            self.mouse.pos - self.mouse_prev.pos
        }
    }

    #[inline]
    pub fn dims(&self) -> V2 {
        vec2(self.size.0 as f32, self.size.1 as f32)
    }

    #[inline]
    pub fn scaled_mouse_delta(&self) -> V2 {
        self.mouse_delta() / (self.dims() * 0.5)
        // if self.mouse_grabbed {
        //     self.mouse.pos / (self.dims() * 0.5)
        // } else {
        //     (self.mouse.pos - self.mouse_prev.pos) / (self.dims() * 0.5)
        // }
    }

    #[inline]
    pub fn key_state(&self, k: VirtualKeyCode) -> KeyState {
        if let Some(k) = self.keys.get(&k) {
            *k
        } else {
            KeyState {
                down: false,
                changed: false,
            }
        }
    }

    #[inline]
    pub fn key_state_mut(&mut self, k: VirtualKeyCode) -> &mut KeyState {
        self.keys.entry(k).or_insert_with(|| Default::default())
    }

    #[inline]
    pub fn key_held(&self, k: VirtualKeyCode) -> bool {
        self.key_state(k).down
    }

    #[inline]
    pub fn key_hit(&self, k: VirtualKeyCode) -> bool {
        let s = self.key_state(k);
        s.changed && s.down
    }

    #[inline]
    pub fn key_released(&self, k: VirtualKeyCode) -> bool {
        let s = self.key_state(k);
        s.changed && !s.down
    }

    #[inline]
    pub fn keys_dir(&self, k1: VirtualKeyCode, k2: VirtualKeyCode) -> f32 {
        let v1 = if self.key_held(k1) { 1 } else { 0 };
        let v2 = if self.key_held(k2) { 1 } else { 0 };
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
        self.key_held(VirtualKeyCode::LShift) || self.key_held(VirtualKeyCode::RShift)
    }

    pub fn ctrl_down(&self) -> bool {
        self.key_held(VirtualKeyCode::LControl) || self.key_held(VirtualKeyCode::RControl)
    }

    pub fn update(&mut self, events: &mut EventsLoop, display: &mut Display, dt: f32) -> bool {
        if self.closed {
            return false;
        }

        self.dt = dt;
        self.key_changes.clear();
        let last_pos = self.mouse_prev.pos;
        self.mouse_prev = self.mouse;
        self.mouse.wheel = 0.0;
        let mut moved_mouse = false;

        for (_, state) in self.keys.iter_mut() {
            state.changed = false;
        }

        events.poll_events(|ev| {
            if let Event::WindowEvent { event, .. } = ev {
                match event {
                    WindowEvent::CloseRequested => {
                        self.closed = true;
                    }
                    WindowEvent::Resized(glium::glutin::dpi::LogicalSize { width, height }) => {
                        self.size = (width as u32, height as u32);
                    }
                    WindowEvent::Focused(true) => self.keys.clear(),
                    WindowEvent::KeyboardInput { input, .. } => {
                        let vk = if let Some(kc) = input.virtual_keycode {
                            kc
                        } else {
                            return;
                        };
                        let was_hit = input.state == ElementState::Pressed;
                        self.key_changes.push((vk, was_hit));
                        update_keyboard(&mut self.gui.borrow_mut(), vk, was_hit);
                        let mut k = self.key_state_mut(vk);
                        if !k.changed {
                            k.changed = k.down != was_hit;
                        }
                        k.down = was_hit;
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        let pos = vec2(position.x as f32, position.y as f32);
                        if self.mouse_grabbed {
                            let gl_window = display.gl_window();
                            // let dpi = gl_window.get_hidpi_factor();
                            let dims = self.dims();
                            // let dpi_dims = dims / dpi;
                            gl_window
                                .set_cursor_position(glium::glutin::dpi::LogicalPosition {
                                    x: (dims.x / 2.0).trunc() as f64,
                                    y: (dims.y / 2.0).trunc() as f64,
                                })
                                .ok()
                                .expect("Could not set mouse cursor position");
                            self.mouse.pos = pos - last_pos;
                            self.mouse_prev.pos = dims / 2.0;
                            moved_mouse = true;
                        } else {
                            self.mouse.pos = pos;
                        }
                    }
                    WindowEvent::MouseInput { button, state, .. } => {
                        let pressed = state == ElementState::Pressed;
                        match button {
                            MouseButton::Left => {
                                self.mouse.down.0 = pressed;
                            }
                            MouseButton::Middle => {
                                self.mouse.down.1 = pressed;
                            }
                            MouseButton::Right => {
                                self.mouse.down.2 = pressed;
                            }
                            _ => {}
                        }
                    }
                    WindowEvent::ReceivedCharacter(c) => {
                        self.gui.borrow_mut().add_input_character(c);
                    }
                    WindowEvent::MouseWheel {
                        delta,
                        phase: TouchPhase::Moved,
                        ..
                    } => match delta {
                        MouseScrollDelta::LineDelta(_, y) => {
                            self.mouse.wheel = y * 10.0;
                        }
                        MouseScrollDelta::PixelDelta(pos) => {
                            self.mouse.wheel = pos.y as f32;
                        }
                    },
                    _ => {}
                }
            }
        });
        if !moved_mouse && self.mouse_grabbed {
            let gl_window = display.gl_window();
            // let dpi = gl_window.get_hidpi_factor();
            let dims = self.dims(); // / dpi;
            gl_window
                .set_cursor_position(glium::glutin::dpi::LogicalPosition {
                    x: (dims.x / 2.0).trunc() as f64,
                    y: (dims.y / 2.0).trunc() as f64,
                })
                .ok()
                .expect("Could not set mouse cursor position");
            self.mouse.pos = vec2(0.0, 0.0);
            self.mouse_prev.pos = self.dims() / 2.0;
        }

        self.mouse.vec = self.screen_pos_to_vec(self.mouse.pos);
        let mut gui = self.gui.borrow_mut();
        let scale = gui.display_framebuffer_scale();

        gui.set_mouse_pos(self.mouse.pos.x / scale.0, self.mouse.pos.y / scale.1);

        gui.set_mouse_down([
            self.mouse.down.0,
            self.mouse.down.1,
            self.mouse.down.2,
            false,
            false,
        ]);

        gui.set_mouse_wheel(self.mouse.wheel / scale.1);

        self.mouse.total_scroll += self.mouse.wheel / scale.1;
        !self.closed
    }
}

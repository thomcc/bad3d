use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use bad3d::{self, prelude::*};

use crate::shared::DemoWindow;
use imgui::{Context, StyleColor, Ui};
use imgui_winit_support::WinitPlatform;

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
    pub gui: Rc<RefCell<Context>>,
    pub mouse_grabbed: bool,
    pub dt: f32,
    pub platform: WinitPlatform,
}

fn init_style(gui: &mut Context) {
    {
        let io = &mut gui.io_mut();
        io[imgui::Key::Tab] = 0;
        io[imgui::Key::LeftArrow] = 1;
        io[imgui::Key::RightArrow] = 2;
        io[imgui::Key::UpArrow] = 3;
        io[imgui::Key::DownArrow] = 4;
        io[imgui::Key::PageUp] = 5;
        io[imgui::Key::PageDown] = 6;
        io[imgui::Key::Home] = 7;
        io[imgui::Key::End] = 8;
        io[imgui::Key::Delete] = 9;
        io[imgui::Key::Backspace] = 10;
        io[imgui::Key::Enter] = 11;
        io[imgui::Key::Escape] = 12;
        io[imgui::Key::A] = 13;
        io[imgui::Key::C] = 14;
        io[imgui::Key::V] = 15;
        io[imgui::Key::X] = 16;
        io[imgui::Key::Y] = 17;
        io[imgui::Key::Z] = 18;
    }

    let style = gui.style_mut();
    // style.child_window_rounding = 3.0;
    style.window_rounding = 0.0;
    style.grab_rounding = 0.0;
    style.scrollbar_rounding = 3.0;
    style.frame_rounding = 3.0;
    style.window_title_align = [0.5, 0.5];
    style.colors[StyleColor::Text as usize] = [0.73, 0.73, 0.73, 1.00];
    style.colors[StyleColor::TextDisabled as usize] = [0.50, 0.50, 0.50, 1.00];
    style.colors[StyleColor::WindowBg as usize] = [0.26, 0.26, 0.26, 0.95];
    // style.colors[StyleColor::ChildWindowBg as usize]        = [0.28, 0.28, 0.28, 1.00];
    style.colors[StyleColor::PopupBg as usize] = [0.26, 0.26, 0.26, 1.00];
    style.colors[StyleColor::Border as usize] = [0.26, 0.26, 0.26, 1.00];
    style.colors[StyleColor::BorderShadow as usize] = [0.26, 0.26, 0.26, 1.00];
    style.colors[StyleColor::FrameBg as usize] = [0.16, 0.16, 0.16, 1.00];
    style.colors[StyleColor::FrameBgHovered as usize] = [0.16, 0.16, 0.16, 1.00];
    style.colors[StyleColor::FrameBgActive as usize] = [0.16, 0.16, 0.16, 1.00];
    style.colors[StyleColor::TitleBg as usize] = [0.36, 0.36, 0.36, 1.00];
    style.colors[StyleColor::TitleBgCollapsed as usize] = [0.36, 0.36, 0.36, 1.00];
    style.colors[StyleColor::TitleBgActive as usize] = [0.36, 0.36, 0.36, 1.00];
    style.colors[StyleColor::MenuBarBg as usize] = [0.26, 0.26, 0.26, 1.00];
    style.colors[StyleColor::ScrollbarBg as usize] = [0.21, 0.21, 0.21, 1.00];
    style.colors[StyleColor::ScrollbarGrab as usize] = [0.36, 0.36, 0.36, 1.00];
    style.colors[StyleColor::ScrollbarGrabHovered as usize] = [0.36, 0.36, 0.36, 1.00];
    style.colors[StyleColor::ScrollbarGrabActive as usize] = [0.36, 0.36, 0.36, 1.00];
    // style.colors[StyleColor::ComboBg as usize]              = [0.32, 0.32, 0.32, 1.00];
    style.colors[StyleColor::CheckMark as usize] = [0.78, 0.78, 0.78, 1.00];
    style.colors[StyleColor::SliderGrab as usize] = [0.74, 0.74, 0.74, 1.00];
    style.colors[StyleColor::SliderGrabActive as usize] = [0.74, 0.74, 0.74, 1.00];
    style.colors[StyleColor::Button as usize] = [0.36, 0.36, 0.36, 1.00];
    style.colors[StyleColor::ButtonHovered as usize] = [0.43, 0.43, 0.43, 1.00];
    style.colors[StyleColor::ButtonActive as usize] = [0.11, 0.11, 0.11, 1.00];
    style.colors[StyleColor::Header as usize] = [0.36, 0.36, 0.36, 1.00];
    style.colors[StyleColor::HeaderHovered as usize] = [0.36, 0.36, 0.36, 1.00];
    style.colors[StyleColor::HeaderActive as usize] = [0.36, 0.36, 0.36, 1.00];
    style.colors[StyleColor::Separator as usize] = [0.39, 0.39, 0.39, 1.00];
    style.colors[StyleColor::SeparatorHovered as usize] = [0.26, 0.59, 0.98, 1.00];
    style.colors[StyleColor::SeparatorActive as usize] = [0.26, 0.59, 0.98, 1.00];
    style.colors[StyleColor::ResizeGrip as usize] = [0.36, 0.36, 0.36, 1.00];
    style.colors[StyleColor::ResizeGripHovered as usize] = [0.26, 0.59, 0.98, 1.00];
    style.colors[StyleColor::ResizeGripActive as usize] = [0.26, 0.59, 0.98, 1.00];
    // style.colors[StyleColor::CloseButton as usize] = [0.59, 0.59, 0.59, 1.00];
    // style.colors[StyleColor::CloseButtonHovered as usize] = [0.98, 0.39, 0.36, 1.00];
    // style.colors[StyleColor::CloseButtonActive as usize] = [0.98, 0.39, 0.36, 1.00];
    style.colors[StyleColor::PlotLines as usize] = [0.39, 0.39, 0.39, 1.00];
    style.colors[StyleColor::PlotLinesHovered as usize] = [1.00, 0.43, 0.35, 1.00];
    style.colors[StyleColor::PlotHistogram as usize] = [0.90, 0.70, 0.00, 1.00];
    style.colors[StyleColor::PlotHistogramHovered as usize] = [1.00, 0.60, 0.00, 1.00];
    style.colors[StyleColor::TextSelectedBg as usize] = [0.32, 0.52, 0.65, 1.00];
    // style.colors[StyleColor::ModalWindowDarkening as usize] = [0.20, 0.20, 0.20, 0.50];
}

fn init_gui(gui: &mut Context) -> WinitPlatform {
    init_style(gui);
    WinitPlatform::init(gui)
}

// fn update_keyboard(imgui: &mut Context, vk: VirtualKeyCode, pressed: bool) {
//     use glium::glutin::VirtualKeyCode as Key;
//     let io = imgui.io_mut();
//     match vk {
//         Key::Tab => io.keys_down[0] = pressed,
//         Key::Left => io.keys_down[1] = pressed,
//         Key::Right => io.keys_down[2] = pressed,
//         Key::Up => io.keys_down[3] = pressed,
//         Key::Down => io.keys_down[4] = pressed,
//         Key::PageUp => io.keys_down[5] = pressed,
//         Key::PageDown => io.keys_down[6] = pressed,
//         Key::Home => io.keys_down[7] = pressed,
//         Key::End => io.keys_down[8] = pressed,
//         Key::Delete => io.keys_down[9] = pressed,
//         Key::Back => io.keys_down[10] = pressed,
//         Key::Return => io.keys_down[11] = pressed,
//         Key::Escape => io.keys_down[12] = pressed,
//         Key::A => io.keys_down[13] = pressed,
//         Key::C => io.keys_down[14] = pressed,
//         Key::V => io.keys_down[15] = pressed,
//         Key::X => io.keys_down[16] = pressed,
//         Key::Y => io.keys_down[17] = pressed,
//         Key::Z => io.keys_down[18] = pressed,
//         Key::LControl | Key::RControl => io.key_ctrl = pressed,
//         Key::LShift | Key::RShift => io.key_shift = pressed,
//         Key::LAlt | Key::RAlt => io.key_alt = pressed,
//         Key::LWin | Key::RWin => io.key_super = pressed,
//         _ => {}
//     }
// }

impl InputState {
    pub fn new(size: (u32, u32), view_angle: f32, gui: Rc<RefCell<Context>>) -> InputState {
        let platform = init_gui(&mut gui.borrow_mut());
        InputState {
            mouse: Default::default(),
            mouse_prev: Default::default(),
            view_angle,
            size,
            keys: HashMap::new(),
            key_changes: Vec::new(),
            closed: false,
            gui,
            platform,
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
        self.keys.entry(k).or_insert_with(Default::default)
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
            {
                let disp = display.gl_window();
                self.platform
                    .handle_event(self.gui.borrow_mut().io_mut(), disp.window(), &ev);
            }
            let gui = self.gui.borrow_mut();
            let io = gui.io();
            if let Event::WindowEvent { event, .. } = ev {
                match event {
                    WindowEvent::CloseRequested => {
                        self.closed = true;
                    }
                    WindowEvent::Resized(glium::glutin::dpi::LogicalSize { width, height }) => {
                        self.size = (width as u32, height as u32);
                    }
                    WindowEvent::Focused(true) => self.keys.clear(),
                    WindowEvent::KeyboardInput { input, .. } if !io.want_capture_keyboard => {
                        let vk = if let Some(kc) = input.virtual_keycode {
                            kc
                        } else {
                            return;
                        };
                        let was_hit = input.state == ElementState::Pressed;
                        self.key_changes.push((vk, was_hit));
                        // update_keyboard(&mut self.gui.borrow_mut(), vk, was_hit);
                        let mut k = self.keys.entry(vk).or_insert_with(Default::default);
                        if !k.changed {
                            k.changed = k.down != was_hit;
                        }
                        k.down = was_hit;
                    }
                    WindowEvent::CursorMoved { position, .. } if !io.want_capture_mouse => {
                        let pos = vec2(position.x as f32, position.y as f32);
                        if self.mouse_grabbed {
                            let gl_window = display.gl_window();
                            // let dpi = gl_window.get_hidpi_factor();
                            let dims = self.dims();
                            // let dpi_dims = dims / dpi;
                            gl_window
                                .window()
                                .set_cursor_position(glium::glutin::dpi::LogicalPosition {
                                    x: (dims.x / 2.0).trunc() as f64,
                                    y: (dims.y / 2.0).trunc() as f64,
                                })
                                .expect("Could not set mouse cursor position");
                            self.mouse.pos = pos - last_pos;
                            self.mouse_prev.pos = dims / 2.0;
                            moved_mouse = true;
                        } else {
                            self.mouse.pos = pos;
                        }
                    }
                    WindowEvent::MouseInput { button, state, .. } if !io.want_capture_mouse => {
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
                    WindowEvent::MouseWheel {
                        delta,
                        phase: TouchPhase::Moved,
                        ..
                    } if !io.want_capture_mouse => match delta {
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
                .window()
                .set_cursor_position(glium::glutin::dpi::LogicalPosition {
                    x: (dims.x / 2.0).trunc() as f64,
                    y: (dims.y / 2.0).trunc() as f64,
                })
                .expect("Could not set mouse cursor position");
            self.mouse.pos = vec2(0.0, 0.0);
            self.mouse_prev.pos = self.dims() / 2.0;
        }

        self.mouse.vec = self.screen_pos_to_vec(self.mouse.pos);
        let mut gui = self.gui.borrow_mut();
        let scale = gui.io().display_framebuffer_scale;

        self.mouse.total_scroll += self.mouse.wheel / scale[1];
        if self.closed {
            return false;
        }

        let glw = display.gl_window();
        if let Err(e) = self.platform.prepare_frame(gui.io_mut(), glw.window()) {
            eprintln!("gui error: {:?}", e);
        }
        true
    }
}

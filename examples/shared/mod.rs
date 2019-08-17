pub mod cam;
pub mod demo_window;
pub mod input;
pub mod object;
mod shader_lib;
mod watched_file;
pub use self::demo_window::{DemoOptions, DemoWindow};
pub use self::input::InputState;
pub use self::object::DemoMesh;

pub type Result<T> = ::std::result::Result<T, ::failure::Error>;

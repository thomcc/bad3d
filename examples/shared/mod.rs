
pub mod input;
pub mod demo_window;
pub mod object;

pub use self::input::InputState;
pub use self::demo_window::{DemoWindow, DemoOptions};
pub use self::object::{DemoMesh, DemoObject};


pub type Result<T> = ::std::result::Result<T, ::failure::Error>;


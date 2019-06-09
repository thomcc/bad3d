# a toy 3d physics engine in rust

this is riddled with numerical robustness problems, logic errors, and design mistakes, you're really best moving on.

if you're brave, some nice demos are in the examples. i recommend `cargo run --release --example phys`.

this is a pretty early rust project of mine, and much of it suffers from not being very familiar with rust. it's full of `Rc<RefCell<RigidBody>>`, for example, when it should just use indices, handles, or some other kind of stable identifier.

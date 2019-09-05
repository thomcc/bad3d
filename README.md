# a toy 3d physics engine in rust

eventually this might be a good physics engine. at the moment, i'm still working out a bunch of it's bugs. caveat emptor.

if you're brave, some nice demos are in the examples. i recommend `cargo run --release --example phys`. (TODO: screenshots)

this was a pretty early rust project of mine, and i abandoned it for a while, and am only now getting back to making it not embarassing. it's... still getting there.

## features

- 3d collision detection of arbitrary convex solids
    - via GJK + EPA.
- 3d rigidbody simulation with a vareity of constraints
    - immediate-mode api for specifying constraints (i'm still deciding if this is worth the trouble)
- convex decomposition.
    - TODO: make the API for this less roundabout
- constructive solid geometry (CSG) via BSP trees
    - TODO: better integrate this with the rest of the API
- convex hull computation (both exact and approximate)
- many routines are SIMD accelerated when on x86
- a variety of exciting bugs
- ...











fn wm_faces(wm: &WingMesh) -> Vec<Face> {
    let res = Vec::with_capacity(wm.faces.len());
    assert!(wm.is_packed);
    assert!(wm.fback.len() == wm.faces.len());
    for i in 0..wm.faces.len() {
        let mut f = Face::new();
        f.plane = wm.faces[i];
        f.vertex = wm.face_verts(i);
        res.push(f);
    }
    res
}
fn wm_big_box(r: f32) -> WingMesh {
    WingMesh::
}
fn make_bsp(wm0: &WingMesh, wm1: WingMesh) -> {
    Box::new(bsp::compile(wm_faces(wm0), wm1))
}

fn run_play_test() {
    let display = glium::glutin::WindowBuilder::new()
                        .with_depth_buffer(24)
                        .with_vsync()
                        .build_glium()
                        .unwrap();
    let mut input_state = {
        let (win_w, win_h) = display.get_window().unwrap()
            .get_inner_size_pixels().unwrap();
        InputState::new(win_w, win_h, 75.0)
    };

    let mut ground_verts = create_box_verts(vec3(-10.0, -10.0, -5.0), vec3(10.0, 10.0, -2.0));
    let ground_tris = hull::compute_hull(&mut ground_verts[..]).unwrap();
    ground_verts.truncate((ground_tris.iter().flat_map(|n| n.iter()).max().unwrap() + 1) as usize);

    let arena = WingMesh::new_box(vec3(-10.0, -10.0, -5.0) vec3(10.0, 10.0, 5.0));
    let mut bsp_geom = make_bsp(&arena, WingMesh::new_cube(32.0));
    bsp_geom.negate();

    let boxes = [
        (vec3(-11.0, -11.0,-0.25), vec3(11.0, 11.0, 0.0)),
        (vec3(  4.0, -11.0, -6.0), vec3( 4.5, 11.0, 6.0)),
        (vec3( -4.5, -11.0, -6.0), vec3(-4.0, 11.0, 6.0)),
        (vec3(-11.0,   4.0, -6.0), vec3(11.0,  4.5, 6.0)),
        (vec3(-11.0,  -4.5, -6.0), vec3(11.0, -4.0, 6.0)),
        (vec3(  2.5,   1.5,  2.0), vec3( 3.5,  3.5, 4.5)),
    ];
    let door_xs = [-7.0f32, 0.0, 7.0];
    let door_ys = [-7.0f32, 7.0];


    for b in boxes.iter() {
        bsp_geom = bsp::union(make_bsp(&WingMesh::new_box(b.0, b.1), WingMesh::new_cube(16.0)), bsp_geom);
    }

    for x in door_xs.iter() {
        let dx = make_bsp(WingMesh::new_box(vec3(x-1.0, 9.0, 0.1), vec3(x+1.0, 9.0, 2.5)),
                          WingMesh::new_cube(16.0));
        dx.negate();
        bsp_geom = bsp::intersect(dx, bsp_geom);
    }

    for y in door_ys.iter() {
        let dy = make_bsp(WingMesh::new_box(vec3(-9.0, y-1.0, 0.1), vec3(9.0, y+1.0, 2.5)),
                          WingMesh::new_cube(16.0));
        dx.negate();
        bsp_geom = bsp::intersect(dx, bsp_geom);
    }

    let brep = bsp_geom.rip_brep();

    bsp_geom.make_brep(brep, 0);

    let mut plane_view = false;
    let mut cam_move = false;
    let mut cam_snap = true;







}




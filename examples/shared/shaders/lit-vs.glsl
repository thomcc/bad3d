
in vec4 position;
out vec3 v_view_pos;
out vec3 v_world_pos;

uniform mat4 perspective;
uniform mat4 view;
uniform mat4 model;

void main() {
    vec4 wp = model * vec4(position.xyz, 1.0);
    vec4 vp = view * wp;
    gl_Position = perspective * vp;
    v_view_pos = vp.xyz;
    v_world_pos = wp.xyz;
}

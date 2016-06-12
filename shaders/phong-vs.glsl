#version 140
in vec3 position;
out vec3 v_position;
out vec3 v_viewpos;
uniform mat4 perspective;
uniform mat4 view;
uniform mat4 model;

void main() {
    mat4 modelview = view * model;
    vec4 mpos = modelview * vec4(position, 1.0);
    gl_Position = perspective * mpos;
    v_position = gl_Position.xyz / gl_Position.w;
    v_viewpos = -mpos.xyz;
}

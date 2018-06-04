
in vec3 position;
in vec2 texcoord;
in vec4 color;

out vec3 v_model_pos;
out vec3 v_view_pos;
out vec2 v_texcoord;
out vec4 v_color;

uniform mat4 u_perspective;
uniform mat4 u_view;
uniform mat4 u_model;

void main() {
    mat4 modelview = view * model;
    vec4 mpos = modelview * vec4(position, 1.0);
    v_texcoord = texcoord;
    v_color = color;
    gl_Position = perspective * mpos;
    v_view_pos = mpos.xyz;
}

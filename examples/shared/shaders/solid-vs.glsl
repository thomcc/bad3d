
in vec4 position;
uniform mat4 perspective, view, model;
void main() {
    mat4 modelview = view * model;
    vec4 mpos = modelview * vec4(position.xyz, 1.0);
    gl_Position = perspective * mpos;
}

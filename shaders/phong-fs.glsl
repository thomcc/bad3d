#version 140

in vec3 v_position;
in vec3 v_viewpos;

out vec4 color;
uniform vec3 u_light;
uniform vec4 u_color;

const vec3 ambient_color = vec3(0.1, 0.1, 0.1);
// const vec3 diffuse_color = vec3(0.5, 0.5, 0.5);
const vec3 specular_color = vec3(1.0, 1.0, 1.0);

void main() {
    vec3 normal = normalize(cross(dFdx(v_viewpos), dFdy(v_viewpos)));
    float diffuse = max(dot(normal, normalize(u_light)), 0.0);
    vec3 camera_dir = normalize(-v_position);
    vec3 half_direction = normalize(normalize(u_light) + camera_dir);
    float specular = pow(max(dot(half_direction, normal), 0.0), 16.0);
    color = vec4(ambient_color + diffuse*u_color.rgb + specular * specular_color, u_color.a);
}

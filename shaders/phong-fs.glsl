#version 140

in vec3 v_viewpos;

out vec4 color;
uniform vec3 u_light;
uniform vec4 u_color;

const vec3 ambient_color = vec3(0.1, 0.1, 0.1);
// const vec3 diffuse_color = vec3(0.5, 0.5, 0.5);
const vec3 specular_color = vec3(1.0, 1.0, 1.0);

void main() {
    vec3 lightvec = u_light - v_viewpos;
    vec3 lightdir = normalize(lightvec);
    vec3 normal = normalize(cross(dFdx(v_viewpos), dFdy(v_viewpos)));
    float diffuse = max(dot(normal, lightdir), 0.0);
    vec3 camera_dir = normalize(-v_viewpos);
    vec3 half_direction = normalize(lightdir + camera_dir);
    float specular = pow(max(dot(half_direction, normal), 0.0), 16.0);
    color = vec4(ambient_color + diffuse*u_color.rgb + specular * specular_color, u_color.a);
}

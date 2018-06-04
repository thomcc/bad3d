
in vec3 v_viewpos;
in vec2 v_texcoord;
in vec4 v_color;

out vec4 color;

uniform vec3 u_light_dir;
uniform vec3 u_ambient_color;
uniform vec3 u_specular_color;

const vec3 ambient_color = vec3(0.1, 0.1, 0.1);
const vec3 specular_color = vec3(1.0, 1.0, 1.0);


void main() {
    vec3 lightvec = u_light - v_viewpos;
    vec3 lightdir = normalize(lightvec);
    vec3 camera_dir = normalize(-v_viewpos);
    float roughness = 0.8;
    float albedo = 0.9;
    float specularScale = 0.4;
    vec3 normal = normalize(cross(dFdx(v_viewpos), dFdy(v_viewpos)));

    float specular = computeSpecular(lightdir, camera_dir, normal, roughness) * specularScale;

    float diffuse = computeDiffuse(lightdir, camera_dir, normal, roughness, albedo);

    vec3 c = ambient_color * u_color.rgb + diffuse * u_color.rgb + specular * specular_color;
    color = vec4(max(min(c, vec3(1.0)), vec3(0.0)), u_color.a);
}

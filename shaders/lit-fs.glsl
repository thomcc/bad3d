#version 140
#define PI 3.14159
#define INV_PI (1.0 / PI)

in vec3 v_viewpos;

out vec4 color;
uniform vec3 u_light;
uniform vec4 u_color;

const vec3 ambient_color = vec3(0.1, 0.1, 0.1);
// const vec3 diffuse_color = vec3(0.5, 0.5, 0.5);
const vec3 specular_color = vec3(1.0, 1.0, 1.0);

float computeSpecular(vec3 lightDir, vec3 viewDir, vec3 normal, float rough) {
    vec3 h = normalize(lightDir + viewDir); // average of lightDir and viewDir...
    // compute beckmann distribution
    float ndh = dot(normal, h);
    float pndh = max(ndh, 0.0001);
    float c2a = pndh * pndh;
    float ic2a = 1.0 / c2a;
    float t2a = (c2a - 1.0) * ic2a;
    float irsq = 1.0 / (rough * rough);
    float mul = INV_PI * irsq * ic2a * ic2a;
    return exp(t2a * irsq) * mul;
}

float computeDiffuse(vec3 lightDir, vec3 viewDir, vec3 normal, float rough, float albedo) {
    float ldv = dot(lightDir, viewDir);
    float ndl = dot(lightDir, normal);
    float ndv = dot(normal, viewDir);

    float s = ldv - ndl * ndv;
    float t = mix(1.0, max(ndl, ndv), step(0.0, s));

    float rSq = rough * rough;
    float a = 1.0 + rSq * (albedo / (rSq + 0.13) + 0.5 / (rSq + 0.33));
    float b = 0.45 * rSq / (rSq + 0.09);

    return albedo * max(0.0, ndl) * (a + b * s / t) * INV_PI;
}

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

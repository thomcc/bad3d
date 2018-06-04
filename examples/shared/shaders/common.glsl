#version 140

#define PI 3.14159
#define INV_PI (1.0 / PI)

#define lerp mix
#define saturate(v) clamp((v), 0.0, 1.0)

float compute_specular(vec3 light_dir, vec3 view_dir, vec3 normal, float rough) {
    vec3 h = normalize(light_dir + view_dir); // average of light_dir and view_dir...
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

float compute_diffuse(vec3 light_dir, vec3 view_dir, vec3 normal, float rough, float albedo) {
    float ldv = dot(light_dir, view_dir);
    float ndl = dot(light_dir, normal);
    float ndv = dot(normal, view_dir);

    float s = ldv - ndl * ndv;
    float t = mix(1.0, max(ndl, ndv), step(0.0, s));

    float rSq = rough * rough;
    float a = 1.0 + rSq * (albedo / (rSq + 0.13) + 0.5 / (rSq + 0.33));
    float b = 0.45 * rSq / (rSq + 0.09);

    return albedo * max(0.0, ndl) * (a + b * s / t) * INV_PI;
}



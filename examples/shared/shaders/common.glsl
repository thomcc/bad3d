#version 140

#define PI 3.14159
#define INV_PI (1.0 / PI)

#define lerp mix
#define saturate(v) clamp((v), 0.0, 1.0)

vec4 quat_conj(vec4 q) {
	return vec4(-q.x, -q.y, -q.z, q.w);
}

vec4 quat_mul(vec4 a, vec4 b) {
	return vec4(
		a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
		a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
		a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
		a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w
	);
}

vec3 quat_rotate(vec4 q, vec3 v) {
	return quat_mul(quat_mul(q, vec4(v, 0.0)), quat_conj(q)).xyz;
}


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

highp float rand(const in vec2 uv) {
	const highp float a = 12.9898, b = 78.233, c = 43758.5453;
	highp float dt = dot(uv.xy, vec2(a, b)), sn = mod(dt, PI);
	return fract(sin(sn) * c);
}
highp float rand3(const in vec3 uv) {
	const highp float a = 12.9898, b = 78.233, c = 43758.5453, d = 92.414;
	highp float dt = dot(uv.xyz, vec3(a, b, d)), sn = mod(dt, PI);
	return fract(sin(sn) * c);
}

highp float smooth_rand(const in vec2 st) {
    return fract(sin(dot(st.xy,
                         vec2(12.9898,78.233)))
                 * 43758.5453123);
}

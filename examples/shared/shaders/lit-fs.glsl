
in vec3 v_view_pos;
in vec3 v_world_pos;


out vec4 color;

uniform vec3 u_light;
uniform vec4 u_color;

uniform float u_fog;
uniform float u_nearfar_dist;
uniform vec3 u_camera_pos;

void main() {
    vec3 lightvec = u_light;
    vec3 lightdir = normalize(lightvec);
    vec3 camera_dir = normalize(v_view_pos);
    vec3 normal = normalize(cross(dFdx(v_view_pos), dFdy(v_view_pos)));
    vec3 col = u_color.rgb;

    vec3 hl_ground_color = vec3(0.0, 0.6666, 1.0);
    vec3 hl_sky_color = vec3(1.0, 0.6666, 0.0);
    vec3 hl_direction = vec3(0.0, 1.0, 0.0);
    float hl_intensity = 0.3;
    float ndl = dot(normal, hl_direction);
    vec3 hemilight = mix(hl_ground_color, hl_sky_color, 0.5 * ndl + 0.5) * hl_intensity;

    vec3 c = hemilight * col;



    const float shininess = 8.0;
    // vec3 H = normalize(lightdir + v_view_pos);
    // float theta = acos(dot(H, normal));
    // float w = theta / shininess;
    // float specular = exp(-w*w);
    // vec3 R = -reflect(normalize(u_light - v_world_pos), normal);
    // float specular = pow(max(0.0, dot(-camera_dir, R)), shininess);
    float specular = compute_specular(normalize(u_light - v_world_pos), camera_dir, normal, 0.2) * 0.4;

    float diffuse = saturate(dot(lightdir, normal));//compute_diffuse(lightdir, camera_dir, normal, roughness, albedo);
    c += diffuse * col;
    c += specular * vec3(1.0);
    // c += specular * vec3(1.0);
    // c += normal / 10.0;

    if (u_fog > 0.0) {
        float dist = length(v_view_pos);//(gl_FragCoord.z / gl_FragCoord.w);// abs(v_view_pos.z);
        float fogAmount=1.-exp(-dist*(u_fog / u_nearfar_dist));
        vec3 fogColor=vec3(.5,.6,.7);
        // float fog = saturate((u_fog - dist) / u_fog);
        c = lerp(c, fogColor, fogAmount);
    }
    color = vec4(max(min(c, vec3(1.0)), vec3(0.0)), u_color.a);
}

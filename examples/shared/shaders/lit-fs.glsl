
in vec3 v_view_pos;
in vec3 v_world_pos;


out vec4 color;

uniform vec3 u_light;
uniform vec4 u_color;

uniform float u_fog;
uniform float u_nearfar_dist;
uniform vec3 u_camera_pos;
uniform vec4 u_camera_q;

uniform vec3 u_hl_ground_color = vec3(0.0, 0.6666, 1.0);
uniform vec3 u_hl_sky_color = vec3(1.0, 0.6666, 0.0);
uniform vec3 u_hl_direction = vec3(0.0, 1.0, 0.0);
uniform float u_hl_intensity = 0.2;
void main() {
    vec3 lightdir =  normalize(u_light);
    // vec3 lightdir = normalize();
    vec3 view_dir = normalize(v_view_pos - v_world_pos);//quat_rotate(u_camera_q, vec3(1.0, 0.0, 0.0)); //normalize(v_view_pos);
    vec3 normal = normalize(cross(dFdx(v_view_pos), dFdy(v_view_pos)));
    vec3 col = u_color.rgb;// * dot(normal, view_dir);

    float ndl = dot(normal, u_hl_direction);
    vec3 hemilight = mix(u_hl_ground_color, u_hl_sky_color, 0.5 * ndl + 0.5) * u_hl_intensity;

    vec3 c = hemilight * col;

    const float shininess = 8.0;
    // vec3 H = normalize(lightdir + v_view_pos);
    // float theta = acos(dot(H, normal));
    // float w = theta / shininess;
    // float specular = exp(-w*w);
    // vec3 R = -reflect(normalize(u_light), normal);
    // float specular = pow(max(0.0, dot(view_dir, R)), shininess);
    float specular = compute_specular(lightdir, view_dir, normal, 0.6) * 0.2;

    float diffuse = saturate(dot(lightdir, normal));
    // diffuse += saturate(dot(-lightdir, normal));//compute_diffuse(lightdir, camera_dir, normal, roughness, albedo);
    c += diffuse * col + vec3(specular);
    // c += specular * vec3(1.0);
    // c += specular * vec3(1.0);
    // c += normal / 10.0;

    // if (u_fog > 0.0) {
    //     float dist = length(v_view_pos);//(gl_FragCoord.z / gl_FragCoord.w);// abs(v_view_pos.z);
    //     float fogAmount=1.-exp(-dist*(u_fog / u_nearfar_dist));
    //     vec3 fogColor=vec3(.5,.6,.7);
    //     // float fog = saturate((u_fog - dist) / u_fog);
    //     c = lerp(c, fogColor, fogAmount);
    // }
    color = vec4(max(min(c, vec3(1.0)), vec3(0.0)), u_color.a);
}

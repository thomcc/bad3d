
in vec3 v_viewpos;

out vec4 color;

uniform vec3 u_light;
uniform vec4 u_color;

uniform float u_fog;
uniform float u_nearfar_dist;

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

    float specular = compute_specular(lightdir, camera_dir, normal, roughness) * specularScale;

    float diffuse = compute_diffuse(lightdir, camera_dir, normal, roughness, albedo);

    vec3 c = ambient_color * u_color.rgb + diffuse * u_color.rgb + specular * specular_color;
    // c += normal / 10.0;

    if (u_fog > 0.0) {
        float dist = length(v_viewpos);//(gl_FragCoord.z / gl_FragCoord.w);// abs(v_viewpos.z);
        float fogAmount=1.-exp(-dist*(u_fog / u_nearfar_dist));
        vec3 fogColor=vec3(.5,.6,.7);
        // float fog = saturate((u_fog - dist) / u_fog);
        c = lerp(c, fogColor, fogAmount);
    }

    color = vec4(max(min(c, vec3(1.0)), vec3(0.0)), u_color.a);
}

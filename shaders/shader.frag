#version 450

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;

layout (location = 0) out vec4 final_color;

layout (binding = 1, std140) uniform ModelUniforms {
	mat4 model;
	vec3 albedo_color;
};

void main() {
	vec3 normal = normalize(f_normal);
	
	vec3 light_dir = normalize(vec3(1.0, 1.0, 0.5));
	
	float diffuse = max(dot(normal, light_dir), 0.0);
	
	float ambient = 0.3;
	
	float lighting = ambient + diffuse * 0.7;
	
	final_color = vec4(albedo_color * lighting, 1.0f);
}

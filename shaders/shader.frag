#version 450

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;

layout (location = 0) out vec4 final_color;

layout (binding = 0, std140) uniform SceneUniforms {
	mat4 view_projection;
	vec3 camera_position;
	vec3 light_direction;
	vec3 light_ambient;
	vec3 light_diffuse;
	vec3 light_specular;
};

layout (binding = 1, std140) uniform ModelUniforms {
	mat4 model;
	vec3 albedo_color;
	vec3 specular_color;
	float shininess;
};

void main() {
	vec3 normal = normalize(f_normal);
	vec3 light_dir = normalize(-light_direction);
	vec3 view_dir = normalize(camera_position - f_position);
	vec3 half_dir = normalize(light_dir + view_dir);

	vec3 ambient = light_ambient * albedo_color;
	float diff = max(dot(normal, light_dir), 0.0);
	vec3 diffuse = light_diffuse * diff * albedo_color;
	float spec = pow(max(dot(normal, half_dir), 0.0), shininess);
	vec3 specular = light_specular * spec * specular_color;

	vec3 result = ambient + diffuse + specular;
	final_color = vec4(result, 1.0);
}

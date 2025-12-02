#version 450

layout (location = 0) in vec3 v_position;
layout (location = 1) in vec3 v_normal;
layout (location = 2) in vec2 v_uv;

layout (location = 0) out vec3 f_position;
layout (location = 1) out vec3 f_normal;
layout (location = 2) out vec2 f_uv;

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
	vec4 position = model * vec4(v_position, 1.0f);
	mat3 normal_matrix = transpose(inverse(mat3(model)));
	vec3 normal = normal_matrix * v_normal;

	gl_Position = view_projection * position;

	f_position = position.xyz;
	f_normal = normalize(normal);
	f_uv = v_uv;
}

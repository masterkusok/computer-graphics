#version 450

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;

layout (location = 0) out vec4 final_color;

struct DirectionalLight {
	vec3 direction;
	float _pad0;
	vec3 ambient;
	float _pad1;
	vec3 diffuse;
	float _pad2;
	vec3 specular;
	float _pad3;
};

struct PointLight {
	vec3 position;
	float _pad0;
	vec3 ambient;
	float _pad1;
	vec3 diffuse;
	float _pad2;
	vec3 specular;
	float _pad3;
};



layout (binding = 0, std140) uniform SceneUniforms {
	mat4 view_projection;
	vec3 view_pos;
	float _pad0;
	DirectionalLight dir_light;
	int num_point_lights;
	float _pad1;
	float _pad2;
	float _pad3;
};

layout (binding = 1, std140) uniform ModelUniforms {
	mat4 model;
	vec3 albedo_color;
	float _pad_m0;
	vec3 specular_color;
	float _pad_m1;
	float shininess;
	float _pad_m2;
	float _pad_m3;
	float _pad_m4;
};

layout (binding = 2, std430) readonly buffer PointLightsBuffer {
	PointLight point_lights[];
};

layout (binding = 3) uniform sampler2D u_texture;



vec3 calcDirectionalLight(DirectionalLight light, vec3 normal, vec3 viewDir, vec3 texColor) {
	vec3 lightDir = normalize(-light.direction);
	
	float diff = max(dot(normal, lightDir), 0.0);
	
	vec3 halfwayDir = normalize(lightDir + viewDir);
	float spec = pow(max(dot(normal, halfwayDir), 0.0), shininess);
	
	vec3 ambient = light.ambient * texColor;
	vec3 diffuse = light.diffuse * diff * texColor;
	vec3 specular = light.specular * spec * specular_color;
	
	return ambient + diffuse + specular;
}

vec3 calcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir, vec3 texColor) {
	vec3 lightDir = normalize(light.position - fragPos);
	
	float diff = max(dot(normal, lightDir), 0.0);
	
	vec3 halfwayDir = normalize(lightDir + viewDir);
	float spec = pow(max(dot(normal, halfwayDir), 0.0), shininess);
	
	float distance = length(light.position - fragPos);
	float attenuation = 1.0 / (distance * distance);
	
	vec3 ambient = light.ambient * texColor;
	vec3 diffuse = light.diffuse * diff * texColor;
	vec3 specular = light.specular * spec * specular_color;
	
	ambient *= attenuation;
	diffuse *= attenuation;
	specular *= attenuation;
	
	return ambient + diffuse + specular;
}



void main() {
	vec3 norm = normalize(f_normal);
	vec3 viewDir = normalize(view_pos - f_position);
	vec3 texColor = texture(u_texture, f_uv).rgb * albedo_color;
	
	vec3 result = calcDirectionalLight(dir_light, norm, viewDir, texColor);
	
	for (int i = 0; i < num_point_lights; i++) {
		result += calcPointLight(point_lights[i], norm, f_position, viewDir, texColor);
	}
		
	final_color = vec4(result, 1.0);
}

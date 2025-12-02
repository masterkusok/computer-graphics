#include "veekay/input.hpp"
#include <cstdint>
#include <climits>
#include <vector>
#include <iostream>
#include <fstream>

#define _USE_MATH_DEFINES
#include <math.h>

#include <veekay/veekay.hpp>

#include <imgui.h>
#include <vulkan/vulkan_core.h>

#include <lodepng.h>

namespace {

constexpr uint32_t max_models = 1024;

struct Vertex {
	veekay::vec3 position;
	veekay::vec3 normal;
	veekay::vec2 uv;
	// NOTE: You can add more attributes
};

struct DirectionalLight {
	veekay::vec3 direction; float _pad0;
	veekay::vec3 ambient; float _pad1;
	veekay::vec3 diffuse; float _pad2;
	veekay::vec3 specular; float _pad3;
};

struct PointLight {
	veekay::vec3 position; float _pad0;
	veekay::vec3 ambient; float _pad1;
	veekay::vec3 diffuse; float _pad2;
	veekay::vec3 specular; float _pad3;
	float constant; float linear; float quadratic; float _pad4;
};

struct SpotLight {
	veekay::vec3 position; float _pad0;
	veekay::vec3 direction; float _pad1;
	veekay::vec3 ambient; float _pad2;
	veekay::vec3 diffuse; float _pad3;
	veekay::vec3 specular; float _pad4;
	float cutOff; float outerCutOff; float constant; float linear;
	float quadratic; float _pad5; float _pad6; float _pad7;
};

struct SceneUniforms {
	veekay::mat4 view_projection;
	veekay::vec3 camera_position; float _pad0;
	veekay::vec3 light_direction; float _pad1;
	veekay::vec3 light_ambient; float _pad2;
	veekay::vec3 light_diffuse; float _pad3;
	veekay::vec3 light_specular; float _pad4;
};

struct ModelUniforms {
	veekay::mat4 model;
	veekay::vec3 albedo_color; float _pad0;
	veekay::vec3 specular_color; float shininess;
};

struct Mesh {
	veekay::graphics::Buffer* vertex_buffer;
	veekay::graphics::Buffer* index_buffer;
	uint32_t indices;
};

struct Transform {
	veekay::vec3 position = {};
	veekay::vec3 scale = {1.0f, 1.0f, 1.0f};
	veekay::vec3 rotation = {};

	// NOTE: Model matrix (translation, rotation and scaling)
	veekay::mat4 matrix() const;
};

struct Model {
	Mesh mesh;
	Transform transform;
	veekay::vec3 albedo_color;
};

struct Camera {
	constexpr static float default_fov = 60.0f;
	constexpr static float default_near_plane = 0.01f;
	constexpr static float default_far_plane = 100.0f;

	veekay::vec3 position = {};
	veekay::vec3 rotation = {};

	float fov = default_fov;
	float near_plane = default_near_plane;
	float far_plane = default_far_plane;

	// NOTE: View matrix of camera (inverse of a transform)
	veekay::mat4 view() const;

	// NOTE: View and projection composition
	veekay::mat4 view_projection(float aspect_ratio) const;
};

// NOTE: Scene objects
float toRadians(float degrees);

inline namespace {
	Camera camera{
		.position = {0.0f, 0.0f, -5.0f},
		.rotation = {0.0f, 0.0f, 0.0f}
	};

	float rotation_speed = 1.0f;
	float rotation_x = 0.0f;
	float rotation_y = 0.0f;

	float camera_speed = 2.0f;
	float mouse_sensitivity = 0.1f;
	double last_mouse_x = 0.0;
	double last_mouse_y = 0.0;
	bool first_mouse = true;

	DirectionalLight dir_light{
		.direction = {-0.2f, -1.0f, -0.3f},
		.ambient = {0.3f, 0.3f, 0.3f},
		.diffuse = {0.8f, 0.8f, 0.8f},
		.specular = {1.0f, 1.0f, 1.0f}
	};

	constexpr int MAX_POINT_LIGHTS = 4;
	constexpr int MAX_SPOT_LIGHTS = 4;
	int num_point_lights = 1;
	int num_spot_lights = 1;

	PointLight point_lights[MAX_POINT_LIGHTS] = {
		{.position = {1.2f, 1.0f, 2.0f}, .ambient = {0.05f, 0.05f, 0.05f},
		 .diffuse = {0.8f, 0.8f, 0.8f}, .specular = {1.0f, 1.0f, 1.0f},
		 .constant = 1.0f, .linear = 0.09f, .quadratic = 0.032f}
	};

	SpotLight spot_lights[MAX_SPOT_LIGHTS] = {
		{.position = {0.0f, 0.0f, 3.0f}, .direction = {0.0f, 0.0f, -1.0f},
		 .ambient = {0.0f, 0.0f, 0.0f}, .diffuse = {1.0f, 1.0f, 1.0f},
		 .specular = {1.0f, 1.0f, 1.0f}, .cutOff = 0.9781476f,
		 .outerCutOff = 0.9537169f, .constant = 1.0f,
		 .linear = 0.09f, .quadratic = 0.032f}
	};
}

// NOTE: Vulkan objects
inline namespace {
	VkShaderModule vertex_shader_module;
	VkShaderModule fragment_shader_module;

	VkDescriptorPool descriptor_pool;
	VkDescriptorSetLayout descriptor_set_layout;
	VkDescriptorSet descriptor_set;

	VkPipelineLayout pipeline_layout;
	VkPipeline pipeline;

	veekay::graphics::Buffer* scene_uniforms_buffer;
	veekay::graphics::Buffer* model_uniforms_buffer;
	veekay::graphics::Buffer* point_lights_buffer;
	veekay::graphics::Buffer* spot_lights_buffer;

	Mesh octahedron_mesh;
}

float toRadians(float degrees) {
	return degrees * float(M_PI) / 180.0f;
}

veekay::mat4 Transform::matrix() const {
	auto t = veekay::mat4::translation(position);
	auto rx = veekay::mat4::rotation({1.0f, 0.0f, 0.0f}, toRadians(rotation.x));
	auto ry = veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, toRadians(rotation.y));
	auto rz = veekay::mat4::rotation({0.0f, 0.0f, 1.0f}, toRadians(rotation.z));
	auto s = veekay::mat4::scaling(scale);

	return t * rz * ry * rx * s;
}

veekay::mat4 Camera::view() const {
	auto t = veekay::mat4::translation(-position);
	auto rx = veekay::mat4::rotation({1.0f, 0.0f, 0.0f}, toRadians(rotation.x));
	auto ry = veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, toRadians(rotation.y));
	auto rz = veekay::mat4::rotation({0.0f, 0.0f, 1.0f}, toRadians(rotation.z));

	return t * rx * ry * rz;
}

veekay::mat4 Camera::view_projection(float aspect_ratio) const {
	auto projection = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);
	auto v = view();

	return v * projection;
}

// NOTE: Loads shader byte code from file
// NOTE: Your shaders are compiled via CMake with this code too, look it up
VkShaderModule loadShaderModule(const char* path) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	size_t size = file.tellg();
	std::vector<uint32_t> buffer(size / sizeof(uint32_t));
	file.seekg(0);
	file.read(reinterpret_cast<char*>(buffer.data()), size);
	file.close();

	VkShaderModuleCreateInfo info{
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size,
		.pCode = buffer.data(),
	};

	VkShaderModule result;
	if (vkCreateShaderModule(veekay::app.vk_device, &info, nullptr, &result) != VK_SUCCESS) {
		return nullptr;
	}

	return result;
}
Mesh createOctahedron() {
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;

	veekay::vec2 uv = {0, 0};

	veekay::vec3 top    = {0.0f,  1.0f, 0.0f};
	veekay::vec3 right  = {1.0f,  0.0f, 0.0f};
	veekay::vec3 front  = {0.0f,  0.0f, 1.0f};
	veekay::vec3 left   = {-1.0f, 0.0f, 0.0f};
	veekay::vec3 back   = {0.0f,  0.0f,-1.0f};
	veekay::vec3 bottom = {0.0f, -1.0f, 0.0f};

	auto add_tri = [&](veekay::vec3 a, veekay::vec3 b, veekay::vec3 c) {
		veekay::vec3 normal = veekay::vec3::normalized(veekay::vec3::cross(b - a, c - a));
		uint32_t base = vertices.size();
		vertices.push_back({a, normal, uv});
		vertices.push_back({b, normal, uv});
		vertices.push_back({c, normal, uv});
		indices.insert(indices.end(), {base, base+1, base+2});
	};

	add_tri(top, front, right);
	add_tri(top, left, front);
	add_tri(top, back, left);
	add_tri(top, right, back);
	add_tri(bottom, right, front);
	add_tri(bottom, front, left);
	add_tri(bottom, left, back);
	add_tri(bottom, back, right);

	Mesh mesh;
	mesh.indices = indices.size();

	mesh.vertex_buffer = new veekay::graphics::Buffer(
		vertices.size() * sizeof(Vertex),
		vertices.data(),
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
	);

	mesh.index_buffer = new veekay::graphics::Buffer(
		indices.size() * sizeof(uint32_t),
		indices.data(),
		VK_BUFFER_USAGE_INDEX_BUFFER_BIT
	);

	return mesh;
}

void initialize(VkCommandBuffer cmd) {
	VkDevice& device = veekay::app.vk_device;

	{ // NOTE: Build graphics pipeline
		vertex_shader_module = loadShaderModule("./shaders/shader.vert.spv");
		if (!vertex_shader_module) {
			std::cerr << "Failed to load Vulkan vertex shader from file\n";
			veekay::app.running = false;
			return;
		}

		fragment_shader_module = loadShaderModule("./shaders/shader.frag.spv");
		if (!fragment_shader_module) {
			std::cerr << "Failed to load Vulkan fragment shader from file\n";
			veekay::app.running = false;
			return;
		}

		VkPipelineShaderStageCreateInfo stage_infos[2];

		// NOTE: Vertex shader stage
		stage_infos[0] = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertex_shader_module,
			.pName = "main",
		};
		stage_infos[1] = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragment_shader_module,
			.pName = "main",
		};

		// NOTE: How many bytes does a vertex take?
		VkVertexInputBindingDescription buffer_binding{
			.binding = 0,
			.stride = sizeof(Vertex),
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
		};

		// NOTE: Declare vertex attributes
		VkVertexInputAttributeDescription attributes[] = {
			{
				.location = 0, // NOTE: First attribute
				.binding = 0, // NOTE: First vertex buffer
				.format = VK_FORMAT_R32G32B32_SFLOAT, // NOTE: 3-component vector of floats
				.offset = offsetof(Vertex, position), // NOTE: Offset of "position" field in a Vertex struct
			},
			{
				.location = 1,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Vertex, normal),
			},
			{
				.location = 2,
				.binding = 0,
				.format = VK_FORMAT_R32G32_SFLOAT,
				.offset = offsetof(Vertex, uv),
			},
		};

		// NOTE: Describe inputs
		VkPipelineVertexInputStateCreateInfo input_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			.vertexBindingDescriptionCount = 1,
			.pVertexBindingDescriptions = &buffer_binding,
			.vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
			.pVertexAttributeDescriptions = attributes,
		};

		// NOTE: Every three vertices make up a triangle,
		//       so our vertex buffer contains a "list of triangles"
		VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		};

		// NOTE: Declare clockwise triangle order as front-facing
		//       Discard triangles that are facing away
		//       Fill triangles, don't draw lines instaed
		VkPipelineRasterizationStateCreateInfo raster_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.polygonMode = VK_POLYGON_MODE_FILL,
			.cullMode = VK_CULL_MODE_NONE,
			.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
			.lineWidth = 1.0f,
		};

		// NOTE: Use 1 sample per pixel
		VkPipelineMultisampleStateCreateInfo sample_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
			.sampleShadingEnable = false,
			.minSampleShading = 1.0f,
		};

		VkViewport viewport{
			.x = 0.0f,
			.y = 0.0f,
			.width = static_cast<float>(veekay::app.window_width),
			.height = static_cast<float>(veekay::app.window_height),
			.minDepth = 0.0f,
			.maxDepth = 1.0f,
		};

		VkRect2D scissor{
			.offset = {0, 0},
			.extent = {veekay::app.window_width, veekay::app.window_height},
		};

		// NOTE: Let rasterizer draw on the entire window
		VkPipelineViewportStateCreateInfo viewport_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,

			.viewportCount = 1,
			.pViewports = &viewport,

			.scissorCount = 1,
			.pScissors = &scissor,
		};

		// NOTE: Let rasterizer perform depth-testing and overwrite depth values on condition pass
		VkPipelineDepthStencilStateCreateInfo depth_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
			.depthTestEnable = true,
			.depthWriteEnable = true,
			.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
		};

		// NOTE: Let fragment shader write all the color channels
		VkPipelineColorBlendAttachmentState attachment_info{
			.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
			                  VK_COLOR_COMPONENT_G_BIT |
			                  VK_COLOR_COMPONENT_B_BIT |
			                  VK_COLOR_COMPONENT_A_BIT,
		};

		// NOTE: Let rasterizer just copy resulting pixels onto a buffer, don't blend
		VkPipelineColorBlendStateCreateInfo blend_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,

			.logicOpEnable = false,
			.logicOp = VK_LOGIC_OP_COPY,

			.attachmentCount = 1,
			.pAttachments = &attachment_info
		};

		{
			VkDescriptorPoolSize pools[] = {
				{.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 8},
				{.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 8}
			};
			
			VkDescriptorPoolCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
				.maxSets = 1,
				.poolSizeCount = sizeof(pools) / sizeof(pools[0]),
				.pPoolSizes = pools,
			};

			if (vkCreateDescriptorPool(device, &info, nullptr,
			                           &descriptor_pool) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor pool\n";
				veekay::app.running = false;
				return;
			}
		}

		// NOTE: Descriptor set layout specification
		{
			VkDescriptorSetLayoutBinding bindings[] = {
				{
					.binding = 0,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{
					.binding = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{
					.binding = 2,
					.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{
					.binding = 3,
					.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
				}
			};
		
			VkDescriptorSetLayoutCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
				.bindingCount = sizeof(bindings) / sizeof(bindings[0]),
				.pBindings = bindings,
			};

			if (vkCreateDescriptorSetLayout(device, &info, nullptr,
			                                &descriptor_set_layout) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor set layout\n";
				veekay::app.running = false;
				return;
			}
		}

		{
			VkDescriptorSetAllocateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &descriptor_set_layout,
			};

			if (vkAllocateDescriptorSets(device, &info, &descriptor_set) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor set\n";
				veekay::app.running = false;
				return;
			}
		}

		// NOTE: Declare external data sources, only push constants this time
		VkPipelineLayoutCreateInfo layout_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 1,
			.pSetLayouts = &descriptor_set_layout,
		};

		// NOTE: Create pipeline layout
		if (vkCreatePipelineLayout(device, &layout_info,
		                           nullptr, &pipeline_layout) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan pipeline layout\n";
			veekay::app.running = false;
			return;
		}
		
		VkGraphicsPipelineCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			.stageCount = 2,
			.pStages = stage_infos,
			.pVertexInputState = &input_state_info,
			.pInputAssemblyState = &assembly_state_info,
			.pViewportState = &viewport_info,
			.pRasterizationState = &raster_info,
			.pMultisampleState = &sample_info,
			.pDepthStencilState = &depth_info,
			.pColorBlendState = &blend_info,
			.layout = pipeline_layout,
			.renderPass = veekay::app.vk_render_pass,
		};

		// NOTE: Create graphics pipeline
		if (vkCreateGraphicsPipelines(device, nullptr,
		                              1, &info, nullptr, &pipeline) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan pipeline\n";
			veekay::app.running = false;
			return;
		}
	}

	scene_uniforms_buffer = new veekay::graphics::Buffer(
		sizeof(SceneUniforms),
		nullptr,
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

	model_uniforms_buffer = new veekay::graphics::Buffer(
		sizeof(ModelUniforms),
		nullptr,
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

	point_lights_buffer = new veekay::graphics::Buffer(
		sizeof(PointLight) * MAX_POINT_LIGHTS,
		nullptr,
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

	spot_lights_buffer = new veekay::graphics::Buffer(
		sizeof(SpotLight) * MAX_SPOT_LIGHTS,
		nullptr,
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

	octahedron_mesh = createOctahedron();


	{
		VkDescriptorBufferInfo buffer_infos[] = {
			{
				.buffer = scene_uniforms_buffer->buffer,
				.offset = 0,
				.range = sizeof(SceneUniforms),
			},
			{
				.buffer = model_uniforms_buffer->buffer,
				.offset = 0,
				.range = sizeof(ModelUniforms),
			},
			{
				.buffer = point_lights_buffer->buffer,
				.offset = 0,
				.range = sizeof(PointLight) * MAX_POINT_LIGHTS,
			},
			{
				.buffer = spot_lights_buffer->buffer,
				.offset = 0,
				.range = sizeof(SpotLight) * MAX_SPOT_LIGHTS,
			},
		};

		VkWriteDescriptorSet write_infos[] = {
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.pBufferInfo = &buffer_infos[0],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 1,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.pBufferInfo = &buffer_infos[1],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 2,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.pBufferInfo = &buffer_infos[2],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 3,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.pBufferInfo = &buffer_infos[3],
			},
		};

		vkUpdateDescriptorSets(device, sizeof(write_infos) / sizeof(write_infos[0]),
		                       write_infos, 0, nullptr);
	}
}

void update(double time) {
	static double last_time = 0.0;
	float dt = static_cast<float>(time - last_time);
	last_time = time;

	// Camera controls
	if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::w)) {
		veekay::vec3 forward = {sinf(toRadians(camera.rotation.y)), 0, cosf(toRadians(camera.rotation.y))};
		camera.position = camera.position + forward * camera_speed * dt;
	}
	if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::s)) {
		veekay::vec3 forward = {sinf(toRadians(camera.rotation.y)), 0, cosf(toRadians(camera.rotation.y))};
		camera.position = camera.position - forward * camera_speed * dt;
	}
	if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::a)) {
		veekay::vec3 right = {cosf(toRadians(camera.rotation.y)), 0, -sinf(toRadians(camera.rotation.y))};
		camera.position = camera.position - right * camera_speed * dt;
	}
	if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::d)) {
		veekay::vec3 right = {cosf(toRadians(camera.rotation.y)), 0, -sinf(toRadians(camera.rotation.y))};
		camera.position = camera.position + right * camera_speed * dt;
	}
	if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::space)) {
		camera.position.y += camera_speed * dt;
	}
	if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::left_shift)) {
		camera.position.y -= camera_speed * dt;
	}

	if (veekay::input::mouse::isButtonDown(veekay::input::mouse::Button::right)) {
		veekay::vec2 mouse_pos = veekay::input::mouse::cursorPosition();
		if (first_mouse) {
			last_mouse_x = mouse_pos.x;
			last_mouse_y = mouse_pos.y;
			first_mouse = false;
		}
		double dx = mouse_pos.x - last_mouse_x;
		double dy = mouse_pos.y - last_mouse_y;
		last_mouse_x = mouse_pos.x;
		last_mouse_y = mouse_pos.y;
		camera.rotation.y += dx * mouse_sensitivity;
		camera.rotation.x += dy * mouse_sensitivity;
		if (camera.rotation.x > 89.0f) camera.rotation.x = 89.0f;
		if (camera.rotation.x < -89.0f) camera.rotation.x = -89.0f;
	} else {
		first_mouse = true;
	}

	rotation_x += rotation_speed * dt * 45.0f;
	rotation_y += rotation_speed * dt * 30.0f;
	if (rotation_x > 360.0f) rotation_x -= 360.0f;
	if (rotation_y > 360.0f) rotation_y -= 360.0f;

	SceneUniforms scene_uniforms;
	scene_uniforms.view_projection = camera.view_projection(
		static_cast<float>(veekay::app.window_width) / static_cast<float>(veekay::app.window_height)
	);
	scene_uniforms.camera_position = camera.position;
	scene_uniforms.light_direction = dir_light.direction;
	scene_uniforms.light_ambient = dir_light.ambient;
	scene_uniforms.light_diffuse = dir_light.diffuse;
	scene_uniforms.light_specular = dir_light.specular;
	memcpy(scene_uniforms_buffer->mapped_region, &scene_uniforms, sizeof(SceneUniforms));

	static int frame_count = 0;
	if (frame_count++ % 60 == 0) {
		std::cout << "Camera: (" << camera.position.x << ", " << camera.position.y << ", " << camera.position.z << ")";
		std::cout << " Rot: (" << camera.rotation.x << ", " << camera.rotation.y << ")\n";
	}

	Transform octahedron_transform;
	octahedron_transform.position = {0.0f, 0.0f, 0.0f};
	octahedron_transform.rotation = {rotation_x, rotation_y, 0.0f};
	octahedron_transform.scale = {1.0f, 1.0f, 1.0f};

	ModelUniforms model_uniforms;
	model_uniforms.model = octahedron_transform.matrix();
	model_uniforms.albedo_color = {1.0f, 0.5f, 0.2f};
	model_uniforms.specular_color = {1.0f, 1.0f, 1.0f};
	model_uniforms.shininess = 32.0f;
	memcpy(model_uniforms_buffer->mapped_region, &model_uniforms, sizeof(ModelUniforms));

	memcpy(point_lights_buffer->mapped_region, point_lights, sizeof(PointLight) * MAX_POINT_LIGHTS);
	memcpy(spot_lights_buffer->mapped_region, spot_lights, sizeof(SpotLight) * MAX_SPOT_LIGHTS);

	ImGui::Begin("Lighting Controls");
	if (ImGui::CollapsingHeader("Directional Light")) {
		ImGui::DragFloat3("Direction", &dir_light.direction.x, 0.01f);
		ImGui::ColorEdit3("Ambient", &dir_light.ambient.x);
		ImGui::ColorEdit3("Diffuse", &dir_light.diffuse.x);
		ImGui::ColorEdit3("Specular", &dir_light.specular.x);
	}
	if (ImGui::CollapsingHeader("Point Lights")) {
		ImGui::SliderInt("Count", &num_point_lights, 0, MAX_POINT_LIGHTS);
		for (int i = 0; i < num_point_lights; i++) {
			ImGui::PushID(i);
			if (ImGui::TreeNode("", "Point Light %d", i)) {
				ImGui::DragFloat3("Position", &point_lights[i].position.x, 0.1f);
				ImGui::ColorEdit3("Diffuse", &point_lights[i].diffuse.x);
				ImGui::DragFloat("Linear", &point_lights[i].linear, 0.01f, 0.0f, 1.0f);
				ImGui::DragFloat("Quadratic", &point_lights[i].quadratic, 0.001f, 0.0f, 1.0f);
				ImGui::TreePop();
			}
			ImGui::PopID();
		}
	}
	if (ImGui::CollapsingHeader("Spot Lights")) {
		ImGui::SliderInt("Count", &num_spot_lights, 0, MAX_SPOT_LIGHTS);
		for (int i = 0; i < num_spot_lights; i++) {
			ImGui::PushID(i + 100);
			if (ImGui::TreeNode("", "Spot Light %d", i)) {
				ImGui::DragFloat3("Position", &spot_lights[i].position.x, 0.1f);
				ImGui::DragFloat3("Direction", &spot_lights[i].direction.x, 0.01f);
				ImGui::ColorEdit3("Diffuse", &spot_lights[i].diffuse.x);
				float cutoff_deg = acosf(spot_lights[i].cutOff) * 180.0f / M_PI;
				float outer_deg = acosf(spot_lights[i].outerCutOff) * 180.0f / M_PI;
				if (ImGui::SliderFloat("Inner Angle", &cutoff_deg, 0.0f, 90.0f)) {
					spot_lights[i].cutOff = cosf(cutoff_deg * M_PI / 180.0f);
				}
				if (ImGui::SliderFloat("Outer Angle", &outer_deg, 0.0f, 90.0f)) {
					spot_lights[i].outerCutOff = cosf(outer_deg * M_PI / 180.0f);
				}
				ImGui::TreePop();
			}
			ImGui::PopID();
		}
	}
	ImGui::End();

	ImGui::Begin("Camera");
	ImGui::Text("Position: (%.1f, %.1f, %.1f)", camera.position.x, camera.position.y, camera.position.z);
	ImGui::Text("Rotation: (%.1f, %.1f)", camera.rotation.x, camera.rotation.y);
	ImGui::SliderFloat("Speed", &camera_speed, 0.5f, 10.0f);
	ImGui::Text("Controls: WASD - move, Space/Shift - up/down");
	ImGui::Text("Right Mouse - look around");
	ImGui::End();

	ImGui::Begin("Octahedron");
	ImGui::SliderFloat("Rotation Speed", &rotation_speed, 0.0f, 3.0f);
	ImGui::Text("Rotation X: %.1f degrees", rotation_x);
	ImGui::Text("Rotation Y: %.1f degrees", rotation_y);
	if (ImGui::Button("Reset Rotation")) {
		rotation_x = 0.0f;
		rotation_y = 0.0f;
	}
	ImGui::End();
}

void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
	vkResetCommandBuffer(cmd, 0);

	VkCommandBufferBeginInfo begin_info{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vkBeginCommandBuffer(cmd, &begin_info);

	VkClearValue clear_values[2];
	clear_values[0].color = {{0.2f, 0.3f, 0.4f, 1.0f}};
	clear_values[1].depthStencil = {1.0f, 0};

	VkRenderPassBeginInfo render_pass_info{
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = veekay::app.vk_render_pass,
		.framebuffer = framebuffer,
		.renderArea = {
			.offset = {0, 0},
			.extent = {veekay::app.window_width, veekay::app.window_height},
		},
		.clearValueCount = 2,
		.pClearValues = clear_values,
	};

	vkCmdBeginRenderPass(cmd, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);

	VkBuffer vertex_buffers[] = {octahedron_mesh.vertex_buffer->buffer};
	VkDeviceSize offsets[] = {0};
	vkCmdBindVertexBuffers(cmd, 0, 1, vertex_buffers, offsets);
	vkCmdBindIndexBuffer(cmd, octahedron_mesh.index_buffer->buffer, 0, VK_INDEX_TYPE_UINT32);

	vkCmdDrawIndexed(cmd, octahedron_mesh.indices, 1, 0, 0, 0);

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

void shutdown() {
	VkDevice& device = veekay::app.vk_device;

	if (octahedron_mesh.vertex_buffer) {
		delete octahedron_mesh.vertex_buffer;
	}
	if (octahedron_mesh.index_buffer) {
		delete octahedron_mesh.index_buffer;
	}
	if (scene_uniforms_buffer) {
		delete scene_uniforms_buffer;
	}
	if (model_uniforms_buffer) {
		delete model_uniforms_buffer;
	}
	if (point_lights_buffer) {
		delete point_lights_buffer;
	}
	if (spot_lights_buffer) {
		delete spot_lights_buffer;
	}

	if (pipeline) {
		vkDestroyPipeline(device, pipeline, nullptr);
	}
	if (pipeline_layout) {
		vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
	}
	if (descriptor_set_layout) {
		vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
	}
	if (descriptor_pool) {
		vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
	}
	if (vertex_shader_module) {
		vkDestroyShaderModule(device, vertex_shader_module, nullptr);
	}
	if (fragment_shader_module) {
		vkDestroyShaderModule(device, fragment_shader_module, nullptr);
	}
}

} // namespace

int main() {
	veekay::ApplicationInfo app_info{
		.init = initialize,
		.shutdown = shutdown,
		.update = update,
		.render = render,
	};

	return veekay::run(app_info);
}
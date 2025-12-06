#include "veekay/input.hpp"
#include <cstdint>
#include <climits>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

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
};

struct SceneUniforms {
	veekay::mat4 view_projection;
	veekay::vec3 view_pos; float _pad0;
	DirectionalLight dir_light;
	int num_point_lights;
	float _pad1;
	float _pad2;
	float _pad3;
};

struct ModelUniforms {
	veekay::mat4 model;
	veekay::vec3 albedo_color; float _pad0;
	veekay::vec3 specular_color; float _pad1;
	float shininess; float _pad2; float _pad3; float _pad4;
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
	veekay::vec3 specular_color = {1.0f, 1.0f, 1.0f};
	float shininess = 32.0f;
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
inline namespace {
	Camera camera{
		.position = {0.0f, -2.0f, -5.0f}
	};

	std::vector<Model> models;

	DirectionalLight dir_light{
		.direction = {-0.2f, 1.0f, -0.3f},
		.ambient = {0.2f, 0.2f, 0.2f},
		.diffuse = {0.5f, 0.5f, 0.5f},
		.specular = {1.0f, 1.0f, 1.0f}
	};

	std::vector<PointLight> point_lights;
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



	veekay::graphics::Texture* missing_texture;
	VkSampler missing_texture_sampler;

	veekay::graphics::Texture* texture;
	VkSampler texture_sampler;
}

float toRadians(float degrees) {
	return degrees * float(M_PI) / 180.0f;
}

veekay::mat4 Transform::matrix() const {
	auto t = veekay::mat4::translation(position);
	auto s = veekay::mat4::scaling(scale);
	auto rx = veekay::mat4::rotation({1.0f, 0.0f, 0.0f}, rotation.x);
	auto ry = veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, rotation.y);
	auto rz = veekay::mat4::rotation({0.0f, 0.0f, 1.0f}, rotation.z);

	return t * ry * rx * rz * s;
}

veekay::mat4 Camera::view() const {
	float cy = cosf(rotation.y);
	float sy = sinf(rotation.y);
	float cx = cosf(rotation.x);
	float sx = sinf(rotation.x);

	veekay::vec3 forward = {-sy * cx, sx, -cy * cx};
	veekay::vec3 right = {cy, 0.0f, -sy};
	veekay::vec3 up = veekay::vec3::cross(right, forward);

	veekay::mat4 result = veekay::mat4::identity();
	result[0][0] = right.x;
	result[1][0] = right.y;
	result[2][0] = right.z;
	result[0][1] = up.x;
	result[1][1] = up.y;
	result[2][1] = up.z;
	result[0][2] = -forward.x;
	result[1][2] = -forward.y;
	result[2][2] = -forward.z;
	result[3][0] = -veekay::vec3::dot(right, position);
	result[3][1] = -veekay::vec3::dot(up, position);
	result[3][2] = veekay::vec3::dot(forward, position);

	return result;
}

veekay::mat4 Camera::view_projection(float aspect_ratio) const {
	auto projection = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);

	return view() * projection;
}

// NOTE: Loads shader byte code from file
// NOTE: Your shaders are compiled via CMake with this code too, look it up
Mesh loadObjMesh(const char* path) {
	std::ifstream file(path);
	if (!file.is_open()) {
		std::cerr << "Failed to open OBJ file: " << path << "\n";
		return {};
	}

	std::vector<veekay::vec3> positions;
	std::vector<veekay::vec3> normals;
	std::vector<veekay::vec2> uvs;
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;

	std::string line;
	while (std::getline(file, line)) {
		if (line.empty() || line[0] == '#') continue;

		std::istringstream iss(line);
		std::string prefix;
		iss >> prefix;

		if (prefix == "v") {
			veekay::vec3 pos;
			iss >> pos.x >> pos.y >> pos.z;
			positions.push_back(pos);
		} else if (prefix == "vt") {
			veekay::vec2 uv;
			iss >> uv.x >> uv.y;
			uvs.push_back(uv);
		} else if (prefix == "vn") {
			veekay::vec3 normal;
			iss >> normal.x >> normal.y >> normal.z;
			normals.push_back(normal);
		} else if (prefix == "f") {
			std::vector<std::string> face_verts;
			std::string vertex_str;
			while (iss >> vertex_str) {
				face_verts.push_back(vertex_str);
			}
			
			for (size_t i = 0; i < face_verts.size(); i++) {
				int pos_idx = 0, uv_idx = 0, norm_idx = 0;
				int matched = sscanf(face_verts[i].c_str(), "%d/%d/%d", &pos_idx, &uv_idx, &norm_idx);
				if (matched < 3) {
					matched = sscanf(face_verts[i].c_str(), "%d//%d", &pos_idx, &norm_idx);
					if (matched < 2) {
						sscanf(face_verts[i].c_str(), "%d", &pos_idx);
					}
				}

				Vertex v;
				v.position = (pos_idx > 0 && pos_idx <= positions.size()) ? positions[pos_idx - 1] : veekay::vec3{0, 0, 0};
				v.uv = (uv_idx > 0 && uv_idx <= uvs.size()) ? uvs[uv_idx - 1] : veekay::vec2{0, 0};
				v.normal = (norm_idx > 0 && norm_idx <= normals.size()) ? normals[norm_idx - 1] : veekay::vec3{0, 1, 0};
				
				if (i < 3) {
					indices.push_back(vertices.size());
					vertices.push_back(v);
				} else {
					indices.push_back(vertices.size() - face_verts.size());
					indices.push_back(vertices.size() - 1);
					indices.push_back(vertices.size());
					vertices.push_back(v);
				}
			}
		}
	}

	Mesh mesh;
	mesh.vertex_buffer = new veekay::graphics::Buffer(
		vertices.size() * sizeof(Vertex), vertices.data(),
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
	mesh.index_buffer = new veekay::graphics::Buffer(
		indices.size() * sizeof(uint32_t), indices.data(),
		VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
	mesh.indices = uint32_t(indices.size());

	return mesh;
}

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
	if (vkCreateShaderModule(veekay::app.vk_device, &
	                         info, nullptr, &result) != VK_SUCCESS) {
		return nullptr;
	}

	return result;
}

void initialize(VkCommandBuffer cmd) {
	VkDevice& device = veekay::app.vk_device;
	VkPhysicalDevice& physical_device = veekay::app.vk_physical_device;

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

		// NOTE: Fragment shader stage
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
			.cullMode = VK_CULL_MODE_BACK_BIT,
			.frontFace = VK_FRONT_FACE_CLOCKWISE,
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
				{
					.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.descriptorCount = 8,
				},
				{
					.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
					.descriptorCount = 8,
				},
				{
					.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.descriptorCount = 8,
				},
				{
					.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					.descriptorCount = 8,
				}
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
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
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
					.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
				},
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
		max_models * sizeof(ModelUniforms),
		nullptr,
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

	point_lights_buffer = new veekay::graphics::Buffer(
		16 * sizeof(PointLight),
		nullptr,
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

	// NOTE: This texture and sampler is used when texture could not be loaded
	{
		VkSamplerCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		};

		if (vkCreateSampler(device, &info, nullptr, &missing_texture_sampler) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan texture sampler\n";
			veekay::app.running = false;
			return;
		}

		uint32_t pixels[] = {
			0xff000000, 0xffff00ff,
			0xffff00ff, 0xff000000,
		};

		missing_texture = new veekay::graphics::Texture(cmd, 2, 2,
		                                                VK_FORMAT_B8G8R8A8_UNORM,
		                                                pixels);
	}

	// NOTE: Load texture from file
	{
		std::vector<unsigned char> image_data;
		unsigned width, height;
		unsigned error = lodepng::decode(image_data, width, height, "./assets/texture.png");
		
		if (error) {
			std::cerr << "Failed to load texture: " << lodepng_error_text(error) << "\n";
			texture = missing_texture;
			texture_sampler = missing_texture_sampler;
		} else {
			texture = new veekay::graphics::Texture(cmd, width, height,
			                                        VK_FORMAT_R8G8B8A8_UNORM,
			                                        image_data.data());
			
			VkSamplerCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
				.magFilter = VK_FILTER_LINEAR,
				.minFilter = VK_FILTER_LINEAR,
				.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
				.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
				.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
				.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
				.maxLod = VK_LOD_CLAMP_NONE,
			};
			
			if (vkCreateSampler(device, &info, nullptr, &texture_sampler) != VK_SUCCESS) {
				std::cerr << "Failed to create texture sampler\n";
				veekay::app.running = false;
				return;
			}
		}
	}

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
				.range = 16 * sizeof(PointLight),
			},
		};

		VkDescriptorImageInfo image_info{
			.sampler = texture_sampler,
			.imageView = texture->view,
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
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
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
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
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = &image_info,
			},
		};

		vkUpdateDescriptorSets(device, sizeof(write_infos) / sizeof(write_infos[0]),
		                       write_infos, 0, nullptr);
	}

	// NOTE: Scene starts empty

	// NOTE: Initialize lights
	point_lights.push_back(PointLight{
		.position = {-2.0f, 1.5f, 0.0f},
		.ambient = {0.1f, 0.0f, 0.0f},
		.diffuse = {2.0f, 0.0f, 0.0f},
		.specular = {1.0f, 1.0f, 1.0f}
	});

	point_lights.push_back(PointLight{
		.position = {2.0f, 1.5f, 0.0f},
		.ambient = {0.0f, 0.1f, 0.0f},
		.diffuse = {0.0f, 2.0f, 0.0f},
		.specular = {1.0f, 1.0f, 1.0f}
	});
}

// NOTE: Destroy resources here, do not cause leaks in your program!
void shutdown() {
	VkDevice& device = veekay::app.vk_device;

	if (texture != missing_texture) {
		vkDestroySampler(device, texture_sampler, nullptr);
		delete texture;
	}
	vkDestroySampler(device, missing_texture_sampler, nullptr);
	delete missing_texture;

	for (auto& model : models) {
		if (model.mesh.index_buffer) delete model.mesh.index_buffer;
		if (model.mesh.vertex_buffer) delete model.mesh.vertex_buffer;
	}

	delete point_lights_buffer;
	delete model_uniforms_buffer;
	delete scene_uniforms_buffer;

	vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
	vkDestroyDescriptorPool(device, descriptor_pool, nullptr);

	vkDestroyPipeline(device, pipeline, nullptr);
	vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
	vkDestroyShaderModule(device, fragment_shader_module, nullptr);
	vkDestroyShaderModule(device, vertex_shader_module, nullptr);
}

void update(double time) {
	ImGui::Begin("Controls:");
	
	if (ImGui::CollapsingHeader("Models", ImGuiTreeNodeFlags_DefaultOpen)) {
		static char obj_path[256] = "./assets/model.obj";
		ImGui::InputText("OBJ Path", obj_path, sizeof(obj_path));
		
		if (ImGui::Button("Load OBJ")) {
			Mesh mesh = loadObjMesh(obj_path);
			if (mesh.indices > 0) {
				models.emplace_back(Model{
					.mesh = mesh,
					.transform = Transform{},
					.albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f},
					.specular_color = veekay::vec3{1.0f, 1.0f, 1.0f},
					.shininess = 64.0f
				});
			}
		}
		
		for (size_t i = 0; i < models.size(); ++i) {
			ImGui::PushID(i);
			if (ImGui::TreeNode("", "Model %zu", i)) {
				ImGui::DragFloat3("Position", &models[i].transform.position.x, 0.1f);
				ImGui::DragFloat3("Rotation", &models[i].transform.rotation.x, 0.01f);
				ImGui::DragFloat3("Scale", &models[i].transform.scale.x, 0.1f);
				ImGui::ColorEdit3("Albedo", &models[i].albedo_color.x);
				ImGui::ColorEdit3("Specular", &models[i].specular_color.x);
				ImGui::DragFloat("Shininess", &models[i].shininess, 1.0f, 1.0f, 256.0f);
				
				if (ImGui::Button("Remove")) {
					if (models[i].mesh.index_buffer) delete models[i].mesh.index_buffer;
					if (models[i].mesh.vertex_buffer) delete models[i].mesh.vertex_buffer;
					models.erase(models.begin() + i);
					ImGui::TreePop();
					ImGui::PopID();
					break;
				}
				ImGui::TreePop();
			}
			ImGui::PopID();
		}
	}

	if (ImGui::CollapsingHeader("Directional Light", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::DragFloat3("Direction##dir", &dir_light.direction.x, 0.01f);
		ImGui::ColorEdit3("Ambient##dir", &dir_light.ambient.x);
		ImGui::ColorEdit3("Diffuse##dir", &dir_light.diffuse.x);
		ImGui::ColorEdit3("Specular##dir", &dir_light.specular.x);
	}

	if (ImGui::CollapsingHeader("Point Lights", ImGuiTreeNodeFlags_DefaultOpen)) {
		if (ImGui::Button("Add Point Light") && point_lights.size() < 16) {
			point_lights.push_back(PointLight{
				.position = {0.0f, 1.5f, 0.0f},
				.ambient = {0.05f, 0.05f, 0.05f},
				.diffuse = {1.0f, 1.0f, 1.0f},
				.specular = {1.0f, 1.0f, 1.0f}
			});
		}
		for (size_t i = 0; i < point_lights.size(); ++i) {
			ImGui::PushID(i);
			if (ImGui::TreeNode("", "Point Light %zu", i)) {
				ImGui::DragFloat3("Position", &point_lights[i].position.x, 0.1f);
				ImGui::ColorEdit3("Ambient", &point_lights[i].ambient.x);
				ImGui::ColorEdit3("Diffuse", &point_lights[i].diffuse.x);
				ImGui::ColorEdit3("Specular", &point_lights[i].specular.x);
				if (ImGui::Button("Remove")) {
					point_lights.erase(point_lights.begin() + i);
					ImGui::TreePop();
					ImGui::PopID();
					break;
				}
				ImGui::TreePop();
			}
			ImGui::PopID();
		}
	}



	ImGui::End();

	bool window_hovered = ImGui::IsAnyItemHovered() || ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow);

	if (!window_hovered) {
		using namespace veekay::input;

		if (mouse::isButtonDown(mouse::Button::right)) {
			auto move_delta = mouse::cursorDelta();

			camera.rotation.y += move_delta.x * 0.005f;
			camera.rotation.x -= move_delta.y * 0.005f;
			camera.rotation.x = std::max(-1.5f, std::min(1.5f, camera.rotation.x));
		}

		float cy = cosf(camera.rotation.y);
		float sy = sinf(camera.rotation.y);
		float cx = cosf(camera.rotation.x);

		veekay::vec3 front = {sy * cx, -sinf(camera.rotation.x), cy * cx};
		veekay::vec3 right = {cy, 0.0f, -sy};
		veekay::vec3 up = {0.0f, 1.0f, 0.0f};

		if (keyboard::isKeyDown(keyboard::Key::w))
			camera.position += front * 0.1f;

		if (keyboard::isKeyDown(keyboard::Key::s))
			camera.position -= front * 0.1f;

		if (keyboard::isKeyDown(keyboard::Key::d))
			camera.position += right * 0.1f;

		if (keyboard::isKeyDown(keyboard::Key::a))
			camera.position -= right * 0.1f;

		if (keyboard::isKeyDown(keyboard::Key::space))
			camera.position += up * 0.1f;

		if (keyboard::isKeyDown(keyboard::Key::left_shift))
			camera.position -= up * 0.1f;
	}

	float aspect_ratio = float(veekay::app.window_width) / float(veekay::app.window_height);
	SceneUniforms scene_uniforms{
		.view_projection = camera.view_projection(aspect_ratio),
		.view_pos = camera.position,
		.dir_light = dir_light,
		.num_point_lights = int(point_lights.size())
	};

	std::vector<ModelUniforms> model_uniforms(models.size());
	for (size_t i = 0, n = models.size(); i < n; ++i) {
		const Model& model = models[i];
		ModelUniforms& uniforms = model_uniforms[i];

		uniforms.model = model.transform.matrix();
		uniforms.albedo_color = model.albedo_color;
		uniforms.specular_color = model.specular_color;
		uniforms.shininess = model.shininess;
	}

	*(SceneUniforms*)scene_uniforms_buffer->mapped_region = scene_uniforms;
	std::copy(model_uniforms.begin(),
	          model_uniforms.end(),
	          static_cast<ModelUniforms*>(model_uniforms_buffer->mapped_region));

	if (!point_lights.empty()) {
		std::copy(point_lights.begin(),
		          point_lights.end(),
		          static_cast<PointLight*>(point_lights_buffer->mapped_region));
	}
}

void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
	vkResetCommandBuffer(cmd, 0);

	{ // NOTE: Start recording rendering commands
		VkCommandBufferBeginInfo info{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		};

		vkBeginCommandBuffer(cmd, &info);
	}

	{ // NOTE: Use current swapchain framebuffer and clear it
		VkClearValue clear_color{.color = {{0.1f, 0.1f, 0.1f, 1.0f}}};
		VkClearValue clear_depth{.depthStencil = {1.0f, 0}};

		VkClearValue clear_values[] = {clear_color, clear_depth};

		VkRenderPassBeginInfo info{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = veekay::app.vk_render_pass,
			.framebuffer = framebuffer,
			.renderArea = {
				.extent = {
					veekay::app.window_width,
					veekay::app.window_height
				},
			},
			.clearValueCount = 2,
			.pClearValues = clear_values,
		};

		vkCmdBeginRenderPass(cmd, &info, VK_SUBPASS_CONTENTS_INLINE);
	}

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	VkDeviceSize zero_offset = 0;

	VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
	VkBuffer current_index_buffer = VK_NULL_HANDLE;

	for (size_t i = 0, n = models.size(); i < n; ++i) {
		const Model& model = models[i];
		const Mesh& mesh = model.mesh;

		if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
			current_vertex_buffer = mesh.vertex_buffer->buffer;
			vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
		}

		if (current_index_buffer != mesh.index_buffer->buffer) {
			current_index_buffer = mesh.index_buffer->buffer;
			vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
		}

		uint32_t offset = i * sizeof(ModelUniforms);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
		                    0, 1, &descriptor_set, 1, &offset);

		vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
	}

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

} // namespace

int main() {
	return veekay::run({
		.init = initialize,
		.shutdown = shutdown,
		.update = update,
		.render = render,
	});
}

// SPDX-License-Identifier: MIT
#include "vulkan_compute.hpp"

#include <vulkan/vulkan.h>

#include <stdexcept>
#include <vector>

namespace manta_improve::gpu {
namespace {

void check(const VkResult result, const char *what)
{
  if (result != VK_SUCCESS) {
    throw std::runtime_error(std::string("Vulkan error in ") + what + ": " +
                             std::to_string(static_cast<int>(result)));
  }
}

int device_rank(const VkPhysicalDeviceType type)
{
  switch (type) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
      return 3;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
      return 2;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
      return 1;
    default:
      return 0;
  }
}

}  // namespace

struct Compute::Impl {
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physical = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  std::uint32_t queue_family = 0;
  VkCommandPool pool = VK_NULL_HANDLE;
  VkCommandBuffer commands = VK_NULL_HANDLE;
  VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
  VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
  VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
  VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
  std::vector<VkPipeline> pipelines;
  std::vector<VkShaderModule> modules;
  Buffer dummy;
  std::string name;
  bool software = false;
};

Compute::Compute() : impl_(std::make_unique<Impl>())
{
  Impl &v = *impl_;

  VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "manta_improve";
  app.apiVersion = VK_API_VERSION_1_1;
  VkInstanceCreateInfo instance_info = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  instance_info.pApplicationInfo = &app;
  check(vkCreateInstance(&instance_info, nullptr, &v.instance), "vkCreateInstance");

  std::uint32_t count = 0;
  vkEnumeratePhysicalDevices(v.instance, &count, nullptr);
  std::vector<VkPhysicalDevice> devices(count);
  vkEnumeratePhysicalDevices(v.instance, &count, devices.data());
  int best_rank = -1;
  for (const VkPhysicalDevice candidate : devices) {
    std::uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(candidate, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(candidate, &family_count, families.data());
    for (std::uint32_t family = 0; family < family_count; ++family) {
      if (!(families[family].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
        continue;
      }
      VkPhysicalDeviceProperties properties;
      vkGetPhysicalDeviceProperties(candidate, &properties);
      const int rank = device_rank(properties.deviceType);
      if (rank > best_rank) {
        best_rank = rank;
        v.physical = candidate;
        v.queue_family = family;
        v.name = properties.deviceName;
        v.software = properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU;
      }
      break;
    }
  }
  if (v.physical == VK_NULL_HANDLE) {
    vkDestroyInstance(v.instance, nullptr);
    throw std::runtime_error("No Vulkan device with a compute queue was found");
  }

  const float priority = 1.0f;
  VkDeviceQueueCreateInfo queue_info = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  queue_info.queueFamilyIndex = v.queue_family;
  queue_info.queueCount = 1;
  queue_info.pQueuePriorities = &priority;
  VkDeviceCreateInfo device_info = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  device_info.queueCreateInfoCount = 1;
  device_info.pQueueCreateInfos = &queue_info;
  check(vkCreateDevice(v.physical, &device_info, nullptr, &v.device), "vkCreateDevice");
  vkGetDeviceQueue(v.device, v.queue_family, 0, &v.queue);

  VkCommandPoolCreateInfo pool_info = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pool_info.queueFamilyIndex = v.queue_family;
  check(vkCreateCommandPool(v.device, &pool_info, nullptr, &v.pool), "vkCreateCommandPool");
  VkCommandBufferAllocateInfo command_info = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  command_info.commandPool = v.pool;
  command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  command_info.commandBufferCount = 1;
  check(vkAllocateCommandBuffers(v.device, &command_info, &v.commands),
        "vkAllocateCommandBuffers");

  std::array<VkDescriptorSetLayoutBinding, binding_count> bindings = {};
  for (std::uint32_t i = 0; i < binding_count; ++i) {
    bindings[i].binding = i;
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo layout_info = {
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  layout_info.bindingCount = static_cast<std::uint32_t>(binding_count);
  layout_info.pBindings = bindings.data();
  check(vkCreateDescriptorSetLayout(v.device, &layout_info, nullptr, &v.set_layout),
        "vkCreateDescriptorSetLayout");

  VkPipelineLayoutCreateInfo pipeline_layout_info = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pipeline_layout_info.setLayoutCount = 1;
  pipeline_layout_info.pSetLayouts = &v.set_layout;
  check(vkCreatePipelineLayout(v.device, &pipeline_layout_info, nullptr, &v.pipeline_layout),
        "vkCreatePipelineLayout");

  VkDescriptorPoolSize pool_size = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                    static_cast<std::uint32_t>(binding_count)};
  VkDescriptorPoolCreateInfo descriptor_pool_info = {
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  descriptor_pool_info.maxSets = 1;
  descriptor_pool_info.poolSizeCount = 1;
  descriptor_pool_info.pPoolSizes = &pool_size;
  check(vkCreateDescriptorPool(v.device, &descriptor_pool_info, nullptr, &v.descriptor_pool),
        "vkCreateDescriptorPool");
  VkDescriptorSetAllocateInfo set_info = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  set_info.descriptorPool = v.descriptor_pool;
  set_info.descriptorSetCount = 1;
  set_info.pSetLayouts = &v.set_layout;
  check(vkAllocateDescriptorSets(v.device, &set_info, &v.descriptor_set),
        "vkAllocateDescriptorSets");

  v.dummy = create_buffer(16);
}

Compute::~Compute()
{
  Impl &v = *impl_;
  vkDeviceWaitIdle(v.device);
  destroy_buffer(v.dummy);
  for (const VkPipeline pipeline : v.pipelines) {
    vkDestroyPipeline(v.device, pipeline, nullptr);
  }
  for (const VkShaderModule module : v.modules) {
    vkDestroyShaderModule(v.device, module, nullptr);
  }
  vkDestroyDescriptorPool(v.device, v.descriptor_pool, nullptr);
  vkDestroyPipelineLayout(v.device, v.pipeline_layout, nullptr);
  vkDestroyDescriptorSetLayout(v.device, v.set_layout, nullptr);
  vkDestroyCommandPool(v.device, v.pool, nullptr);
  vkDestroyDevice(v.device, nullptr);
  vkDestroyInstance(v.instance, nullptr);
}

const std::string &Compute::device_name() const
{
  return impl_->name;
}

bool Compute::is_software_device() const
{
  return impl_->software;
}

Compute::Buffer Compute::create_buffer(const std::size_t bytes)
{
  Impl &v = *impl_;
  Buffer buffer;
  buffer.size = bytes < 16 ? 16 : bytes;

  VkBufferCreateInfo info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  info.size = buffer.size;
  info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VkBuffer handle = VK_NULL_HANDLE;
  check(vkCreateBuffer(v.device, &info, nullptr, &handle), "vkCreateBuffer");

  VkMemoryRequirements requirements;
  vkGetBufferMemoryRequirements(v.device, handle, &requirements);
  VkPhysicalDeviceMemoryProperties memory_properties;
  vkGetPhysicalDeviceMemoryProperties(v.physical, &memory_properties);
  const VkMemoryPropertyFlags wanted = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  std::uint32_t type_index = UINT32_MAX;
  for (std::uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
    if ((requirements.memoryTypeBits & (1u << i)) &&
        (memory_properties.memoryTypes[i].propertyFlags & wanted) == wanted)
    {
      type_index = i;
      break;
    }
  }
  if (type_index == UINT32_MAX) {
    throw std::runtime_error("No host-visible Vulkan memory type for storage buffers");
  }
  VkMemoryAllocateInfo allocate = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  allocate.allocationSize = requirements.size;
  allocate.memoryTypeIndex = type_index;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  check(vkAllocateMemory(v.device, &allocate, nullptr, &memory), "vkAllocateMemory");
  check(vkBindBufferMemory(v.device, handle, memory, 0), "vkBindBufferMemory");
  check(vkMapMemory(v.device, memory, 0, VK_WHOLE_SIZE, 0, &buffer.mapped), "vkMapMemory");

  buffer.handle = reinterpret_cast<std::uint64_t>(handle);
  buffer.memory = reinterpret_cast<std::uint64_t>(memory);
  return buffer;
}

void Compute::destroy_buffer(Buffer &buffer)
{
  if (!buffer.handle) {
    return;
  }
  Impl &v = *impl_;
  vkDestroyBuffer(v.device, reinterpret_cast<VkBuffer>(buffer.handle), nullptr);
  vkFreeMemory(v.device, reinterpret_cast<VkDeviceMemory>(buffer.memory), nullptr);
  buffer = {};
}

int Compute::create_kernel(const std::uint32_t *spirv, const std::size_t word_count)
{
  Impl &v = *impl_;
  VkShaderModuleCreateInfo module_info = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  module_info.codeSize = word_count * sizeof(std::uint32_t);
  module_info.pCode = spirv;
  VkShaderModule module = VK_NULL_HANDLE;
  check(vkCreateShaderModule(v.device, &module_info, nullptr, &module), "vkCreateShaderModule");
  v.modules.push_back(module);

  VkComputePipelineCreateInfo pipeline_info = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipeline_info.stage.module = module;
  pipeline_info.stage.pName = "main";
  pipeline_info.layout = v.pipeline_layout;
  VkPipeline pipeline = VK_NULL_HANDLE;
  check(vkCreateComputePipelines(v.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline),
        "vkCreateComputePipelines");
  v.pipelines.push_back(pipeline);
  return static_cast<int>(v.pipelines.size()) - 1;
}

void Compute::dispatch(const int kernel,
                       const std::array<const Buffer *, binding_count> &buffers,
                       const std::uint32_t group_count)
{
  Impl &v = *impl_;
  if (group_count == 0) {
    return;
  }

  std::array<VkDescriptorBufferInfo, binding_count> buffer_infos = {};
  std::array<VkWriteDescriptorSet, binding_count> writes = {};
  for (std::uint32_t i = 0; i < binding_count; ++i) {
    const Buffer &buffer = buffers[i] ? *buffers[i] : v.dummy;
    buffer_infos[i].buffer = reinterpret_cast<VkBuffer>(buffer.handle);
    buffer_infos[i].range = VK_WHOLE_SIZE;
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = v.descriptor_set;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &buffer_infos[i];
  }
  vkUpdateDescriptorSets(
      v.device, static_cast<std::uint32_t>(binding_count), writes.data(), 0, nullptr);

  VkCommandBufferBeginInfo begin = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  check(vkBeginCommandBuffer(v.commands, &begin), "vkBeginCommandBuffer");
  vkCmdBindPipeline(v.commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                    v.pipelines.at(static_cast<std::size_t>(kernel)));
  vkCmdBindDescriptorSets(v.commands, VK_PIPELINE_BIND_POINT_COMPUTE, v.pipeline_layout, 0, 1,
                          &v.descriptor_set, 0, nullptr);
  vkCmdDispatch(v.commands, group_count, 1, 1);
  check(vkEndCommandBuffer(v.commands), "vkEndCommandBuffer");

  VkSubmitInfo submit = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &v.commands;
  check(vkQueueSubmit(v.queue, 1, &submit, VK_NULL_HANDLE), "vkQueueSubmit");
  check(vkQueueWaitIdle(v.queue), "vkQueueWaitIdle");
}

}  // namespace manta_improve::gpu

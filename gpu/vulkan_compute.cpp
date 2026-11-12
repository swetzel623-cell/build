// SPDX-License-Identifier: MIT
#include "vulkan_compute.hpp"

#include <vulkan/vulkan.h>

#include <cstring>
#include <map>
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

VkBuffer to_vk(const Compute::Buffer &buffer)
{
  return reinterpret_cast<VkBuffer>(buffer.handle);
}

constexpr std::uint32_t sets_per_pool = 256;

}  // namespace

struct Compute::Impl {
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physical = VK_NULL_HANDLE;
  VkPhysicalDeviceMemoryProperties memory_properties{};
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  std::uint32_t queue_family = 0;
  VkCommandPool command_pool = VK_NULL_HANDLE;
  VkCommandBuffer commands = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
  VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
  VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
  std::vector<VkDescriptorPool> descriptor_pools;
  std::map<std::array<std::uint64_t, binding_count>, VkDescriptorSet> descriptor_cache;
  std::vector<VkPipeline> pipelines;
  std::vector<VkShaderModule> modules;
  VkQueryPool timestamps = VK_NULL_HANDLE;
  double timestamp_period_ns = 0.0;
  std::uint64_t timestamp_mask = 0;

  Buffer dummy;
  bool recording = false;

  struct Staging {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void *mapped = nullptr;
  };
  std::vector<Staging> staging;
  struct PendingDownload {
    void *mapped;
    void *out;
    std::size_t bytes;
  };
  std::vector<PendingDownload> downloads;

  std::string name;
  bool software = false;

  std::uint32_t memory_type(const std::uint32_t allowed,
                            const VkMemoryPropertyFlags required,
                            const VkMemoryPropertyFlags preferred) const
  {
    for (const VkMemoryPropertyFlags wanted : {required | preferred, required}) {
      for (std::uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
        if ((allowed & (1u << i)) &&
            (memory_properties.memoryTypes[i].propertyFlags & wanted) == wanted)
        {
          return i;
        }
      }
    }
    throw std::runtime_error("No suitable Vulkan memory type");
  }

  void make_buffer(const std::size_t bytes,
                   const VkBufferUsageFlags usage,
                   const VkMemoryPropertyFlags required,
                   const VkMemoryPropertyFlags preferred,
                   VkBuffer &buffer,
                   VkDeviceMemory &memory) const
  {
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = bytes;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    check(vkCreateBuffer(device, &info, nullptr, &buffer), "vkCreateBuffer");

    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(device, buffer, &requirements);
    VkMemoryAllocateInfo allocate{};
    allocate.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocate.allocationSize = requirements.size;
    allocate.memoryTypeIndex = memory_type(requirements.memoryTypeBits, required, preferred);
    const VkResult result = vkAllocateMemory(device, &allocate, nullptr, &memory);
    if (result != VK_SUCCESS) {
      vkDestroyBuffer(device, buffer, nullptr);
      check(result, "vkAllocateMemory");
    }
    check(vkBindBufferMemory(device, buffer, memory, 0), "vkBindBufferMemory");
  }

  void *staging_buffer(const std::size_t bytes, VkBuffer &buffer)
  {
    Staging entry;
    make_buffer(bytes,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                entry.buffer,
                entry.memory);
    check(vkMapMemory(device, entry.memory, 0, VK_WHOLE_SIZE, 0, &entry.mapped), "vkMapMemory");
    staging.push_back(entry);
    buffer = entry.buffer;
    return entry.mapped;
  }

  void release_staging()
  {
    for (const Staging &entry : staging) {
      vkDestroyBuffer(device, entry.buffer, nullptr);
      vkFreeMemory(device, entry.memory, nullptr);
    }
    staging.clear();
  }

  /** Make every earlier write visible to every later command. */
  void barrier() const
  {
    VkMemoryBarrier memory_barrier{};
    memory_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memory_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    memory_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                                   VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(commands,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0,
                         1,
                         &memory_barrier,
                         0,
                         nullptr,
                         0,
                         nullptr);
  }

  VkDescriptorPool new_descriptor_pool()
  {
    VkDescriptorPoolSize size = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                 sets_per_pool * static_cast<std::uint32_t>(binding_count)};
    VkDescriptorPoolCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    info.maxSets = sets_per_pool;
    info.poolSizeCount = 1;
    info.pPoolSizes = &size;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    check(vkCreateDescriptorPool(device, &info, nullptr, &pool), "vkCreateDescriptorPool");
    descriptor_pools.push_back(pool);
    return pool;
  }

  /** One descriptor set per distinct combination of buffers, reused across submissions. */
  VkDescriptorSet descriptor_set(const Bindings &buffers)
  {
    std::array<std::uint64_t, binding_count> key{};
    for (std::size_t i = 0; i < binding_count; ++i) {
      key[i] = (buffers[i] ? *buffers[i] : dummy).handle;
    }
    const auto found = descriptor_cache.find(key);
    if (found != descriptor_cache.end()) {
      return found->second;
    }

    VkDescriptorSetAllocateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    info.descriptorPool = descriptor_pools.back();
    info.descriptorSetCount = 1;
    info.pSetLayouts = &set_layout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkResult result = vkAllocateDescriptorSets(device, &info, &set);
    if (result == VK_ERROR_OUT_OF_POOL_MEMORY || result == VK_ERROR_FRAGMENTED_POOL) {
      info.descriptorPool = new_descriptor_pool();
      result = vkAllocateDescriptorSets(device, &info, &set);
    }
    check(result, "vkAllocateDescriptorSets");

    std::array<VkDescriptorBufferInfo, binding_count> buffer_infos{};
    std::array<VkWriteDescriptorSet, binding_count> writes{};
    for (std::uint32_t i = 0; i < binding_count; ++i) {
      buffer_infos[i].buffer = reinterpret_cast<VkBuffer>(key[i]);
      buffer_infos[i].range = VK_WHOLE_SIZE;
      writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[i].dstSet = set;
      writes[i].dstBinding = i + 1;
      writes[i].descriptorCount = 1;
      writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[i].pBufferInfo = &buffer_infos[i];
    }
    vkUpdateDescriptorSets(
        device, static_cast<std::uint32_t>(binding_count), writes.data(), 0, nullptr);
    descriptor_cache.emplace(key, set);
    return set;
  }

  /** Forget every cached descriptor set. Their buffers may no longer exist. */
  void clear_descriptor_cache()
  {
    descriptor_cache.clear();
    for (std::size_t i = 1; i < descriptor_pools.size(); ++i) {
      vkDestroyDescriptorPool(device, descriptor_pools[i], nullptr);
    }
    descriptor_pools.resize(1);
    vkResetDescriptorPool(device, descriptor_pools[0], 0);
  }
};

Compute::Compute() : impl_(std::make_unique<Impl>())
{
  Impl &v = *impl_;

  VkApplicationInfo app{};
  app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app.pApplicationName = "manta_improve";
  app.apiVersion = VK_API_VERSION_1_1;
  VkInstanceCreateInfo instance_info{};
  instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instance_info.pApplicationInfo = &app;
  check(vkCreateInstance(&instance_info, nullptr, &v.instance), "vkCreateInstance");

  std::uint32_t count = 0;
  vkEnumeratePhysicalDevices(v.instance, &count, nullptr);
  std::vector<VkPhysicalDevice> devices(count);
  vkEnumeratePhysicalDevices(v.instance, &count, devices.data());
  int best_rank = -1;
  std::uint32_t timestamp_bits = 0;
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
        v.timestamp_period_ns = properties.limits.timestampPeriod;
        timestamp_bits = families[family].timestampValidBits;
      }
      break;
    }
  }
  if (v.physical == VK_NULL_HANDLE) {
    vkDestroyInstance(v.instance, nullptr);
    throw std::runtime_error("No Vulkan device with a compute queue was found");
  }
  vkGetPhysicalDeviceMemoryProperties(v.physical, &v.memory_properties);
  if (timestamp_bits > 0 && v.timestamp_period_ns > 0.0) {
    v.timestamp_mask = timestamp_bits >= 64 ? ~std::uint64_t(0) :
                                              (std::uint64_t(1) << timestamp_bits) - 1;
  }

  const float priority = 1.0f;
  VkDeviceQueueCreateInfo queue_info{};
  queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue_info.queueFamilyIndex = v.queue_family;
  queue_info.queueCount = 1;
  queue_info.pQueuePriorities = &priority;
  VkDeviceCreateInfo device_info{};
  device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  device_info.queueCreateInfoCount = 1;
  device_info.pQueueCreateInfos = &queue_info;
  check(vkCreateDevice(v.physical, &device_info, nullptr, &v.device), "vkCreateDevice");
  vkGetDeviceQueue(v.device, v.queue_family, 0, &v.queue);

  VkCommandPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pool_info.queueFamilyIndex = v.queue_family;
  check(vkCreateCommandPool(v.device, &pool_info, nullptr, &v.command_pool),
        "vkCreateCommandPool");
  VkCommandBufferAllocateInfo command_info{};
  command_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  command_info.commandPool = v.command_pool;
  command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  command_info.commandBufferCount = 1;
  check(vkAllocateCommandBuffers(v.device, &command_info, &v.commands),
        "vkAllocateCommandBuffers");
  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  check(vkCreateFence(v.device, &fence_info, nullptr, &v.fence), "vkCreateFence");

  std::array<VkDescriptorSetLayoutBinding, binding_count> bindings{};
  for (std::uint32_t i = 0; i < binding_count; ++i) {
    bindings[i].binding = i + 1;
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layout_info.bindingCount = static_cast<std::uint32_t>(binding_count);
  layout_info.pBindings = bindings.data();
  check(vkCreateDescriptorSetLayout(v.device, &layout_info, nullptr, &v.set_layout),
        "vkCreateDescriptorSetLayout");

  VkPushConstantRange push_range{};
  push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  push_range.size = static_cast<std::uint32_t>(push_constant_bytes);
  VkPipelineLayoutCreateInfo pipeline_layout_info{};
  pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeline_layout_info.setLayoutCount = 1;
  pipeline_layout_info.pSetLayouts = &v.set_layout;
  pipeline_layout_info.pushConstantRangeCount = 1;
  pipeline_layout_info.pPushConstantRanges = &push_range;
  check(vkCreatePipelineLayout(v.device, &pipeline_layout_info, nullptr, &v.pipeline_layout),
        "vkCreatePipelineLayout");

  v.new_descriptor_pool();

  if (v.timestamp_mask) {
    VkQueryPoolCreateInfo query_info{};
    query_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    query_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    query_info.queryCount = 2;
    check(vkCreateQueryPool(v.device, &query_info, nullptr, &v.timestamps), "vkCreateQueryPool");
  }

  v.dummy = create_buffer(16);
}

Compute::~Compute()
{
  Impl &v = *impl_;
  vkDeviceWaitIdle(v.device);
  v.release_staging();
  destroy_buffer(v.dummy);
  for (const VkPipeline pipeline : v.pipelines) {
    vkDestroyPipeline(v.device, pipeline, nullptr);
  }
  for (const VkShaderModule module : v.modules) {
    vkDestroyShaderModule(v.device, module, nullptr);
  }
  if (v.timestamps) {
    vkDestroyQueryPool(v.device, v.timestamps, nullptr);
  }
  for (const VkDescriptorPool pool : v.descriptor_pools) {
    vkDestroyDescriptorPool(v.device, pool, nullptr);
  }
  vkDestroyPipelineLayout(v.device, v.pipeline_layout, nullptr);
  vkDestroyDescriptorSetLayout(v.device, v.set_layout, nullptr);
  vkDestroyFence(v.device, v.fence, nullptr);
  vkDestroyCommandPool(v.device, v.command_pool, nullptr);
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

bool Compute::has_timestamps() const
{
  return impl_->timestamps != VK_NULL_HANDLE;
}

Compute::Buffer Compute::create_buffer(const std::size_t bytes)
{
  Impl &v = *impl_;
  if (v.recording) {
    throw std::logic_error("create_buffer while recording");
  }
  Buffer buffer;
  buffer.size = bytes < 16 ? 16 : bytes;
  VkBuffer handle = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  v.make_buffer(buffer.size,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                0,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                handle,
                memory);
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
  if (v.recording) {
    throw std::logic_error("destroy_buffer while recording");
  }
  v.clear_descriptor_cache();
  vkDestroyBuffer(v.device, to_vk(buffer), nullptr);
  vkFreeMemory(v.device, reinterpret_cast<VkDeviceMemory>(buffer.memory), nullptr);
  buffer = {};
}

int Compute::create_kernel(const std::uint32_t *spirv, const std::size_t word_count)
{
  Impl &v = *impl_;
  VkShaderModuleCreateInfo module_info{};
  module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  module_info.codeSize = word_count * sizeof(std::uint32_t);
  module_info.pCode = spirv;
  VkShaderModule module = VK_NULL_HANDLE;
  check(vkCreateShaderModule(v.device, &module_info, nullptr, &module), "vkCreateShaderModule");
  v.modules.push_back(module);

  VkComputePipelineCreateInfo pipeline_info{};
  pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
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

void Compute::begin()
{
  Impl &v = *impl_;
  if (v.recording) {
    throw std::logic_error("begin while recording");
  }
  check(vkResetCommandBuffer(v.commands, 0), "vkResetCommandBuffer");
  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  check(vkBeginCommandBuffer(v.commands, &begin_info), "vkBeginCommandBuffer");
  if (v.timestamps) {
    vkCmdResetQueryPool(v.commands, v.timestamps, 0, 2);
    vkCmdWriteTimestamp(v.commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, v.timestamps, 0);
  }
  v.recording = true;
}

void Compute::upload(const Buffer &target, const void *data, const std::size_t bytes)
{
  Impl &v = *impl_;
  if (bytes == 0) {
    return;
  }
  if (bytes > target.size) {
    throw std::logic_error("upload larger than its target buffer");
  }
  VkBuffer staging = VK_NULL_HANDLE;
  std::memcpy(v.staging_buffer(bytes, staging), data, bytes);
  VkBufferCopy region{};
  region.size = bytes;
  vkCmdCopyBuffer(v.commands, staging, to_vk(target), 1, &region);
  v.barrier();
}

void Compute::download(const Buffer &source, void *out, const std::size_t bytes)
{
  Impl &v = *impl_;
  if (bytes == 0) {
    return;
  }
  if (bytes > source.size) {
    throw std::logic_error("download larger than its source buffer");
  }
  VkBuffer staging = VK_NULL_HANDLE;
  void *mapped = v.staging_buffer(bytes, staging);
  VkBufferCopy region{};
  region.size = bytes;
  vkCmdCopyBuffer(v.commands, to_vk(source), staging, 1, &region);
  v.barrier();
  v.downloads.push_back({mapped, out, bytes});
}

void Compute::fill(const Buffer &target, const std::uint32_t value)
{
  Impl &v = *impl_;
  vkCmdFillBuffer(v.commands, to_vk(target), 0, VK_WHOLE_SIZE, value);
  v.barrier();
}

void Compute::dispatch(const int kernel,
                       const void *push_constants,
                       const std::size_t push_bytes,
                       const Bindings &buffers,
                       const std::uint32_t group_count)
{
  Impl &v = *impl_;
  if (group_count == 0) {
    return;
  }
  if (push_bytes > push_constant_bytes) {
    throw std::logic_error("too many push-constant bytes");
  }
  std::array<std::uint8_t, push_constant_bytes> push{};
  if (push_bytes) {
    std::memcpy(push.data(), push_constants, push_bytes);
  }
  const VkDescriptorSet set = v.descriptor_set(buffers);
  vkCmdBindPipeline(v.commands,
                    VK_PIPELINE_BIND_POINT_COMPUTE,
                    v.pipelines.at(static_cast<std::size_t>(kernel)));
  vkCmdBindDescriptorSets(
      v.commands, VK_PIPELINE_BIND_POINT_COMPUTE, v.pipeline_layout, 0, 1, &set, 0, nullptr);
  vkCmdPushConstants(v.commands,
                     v.pipeline_layout,
                     VK_SHADER_STAGE_COMPUTE_BIT,
                     0,
                     static_cast<std::uint32_t>(push_constant_bytes),
                     push.data());
  vkCmdDispatch(v.commands, group_count, 1, 1);
  v.barrier();
}

double Compute::submit()
{
  Impl &v = *impl_;
  if (!v.recording) {
    throw std::logic_error("submit without begin");
  }
  if (v.timestamps) {
    vkCmdWriteTimestamp(v.commands, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, v.timestamps, 1);
  }
  check(vkEndCommandBuffer(v.commands), "vkEndCommandBuffer");
  v.recording = false;

  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &v.commands;
  check(vkResetFences(v.device, 1, &v.fence), "vkResetFences");
  check(vkQueueSubmit(v.queue, 1, &submit_info, v.fence), "vkQueueSubmit");
  check(vkWaitForFences(v.device, 1, &v.fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");

  for (const Impl::PendingDownload &pending : v.downloads) {
    std::memcpy(pending.out, pending.mapped, pending.bytes);
  }
  v.downloads.clear();
  v.release_staging();

  if (!v.timestamps) {
    return -1.0;
  }
  std::uint64_t ticks[2] = {0, 0};
  check(vkGetQueryPoolResults(v.device,
                              v.timestamps,
                              0,
                              2,
                              sizeof(ticks),
                              ticks,
                              sizeof(std::uint64_t),
                              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
        "vkGetQueryPoolResults");
  const std::uint64_t elapsed = (ticks[1] - ticks[0]) & v.timestamp_mask;
  return static_cast<double>(elapsed) * v.timestamp_period_ns * 1.0e-6;
}

}  // namespace manta_improve::gpu

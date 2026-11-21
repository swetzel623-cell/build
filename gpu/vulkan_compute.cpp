// SPDX-License-Identifier: MIT
#include "vulkan_compute.hpp"

/* The Vulkan loader is opened at run time rather than linked, so a program that embeds this
 * code starts on machines without Vulkan, and these function pointers stay separate from any
 * the host program loads for its own use of Vulkan (Blender's volk, for example). */
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <map>
#include <mutex>
#include <stdexcept>
#include <vector>

#ifdef MANTA_PERF_COUNTERS
#  include "perf_counters.hpp"
PerfCounters g_perf;
#  define MANTA_PERF(...) (__VA_ARGS__)
#else
#  define MANTA_PERF(...) ((void)0)
#endif

namespace {

#define MANTA_VK_INSTANCE_FUNCTIONS(X) \
  X(vkAllocateCommandBuffers) \
  X(vkAllocateDescriptorSets) \
  X(vkAllocateMemory) \
  X(vkBeginCommandBuffer) \
  X(vkBindBufferMemory) \
  X(vkCmdBindDescriptorSets) \
  X(vkCmdBindPipeline) \
  X(vkCmdCopyBuffer) \
  X(vkCmdDispatch) \
  X(vkCmdFillBuffer) \
  X(vkCmdPipelineBarrier) \
  X(vkCmdPushConstants) \
  X(vkCmdResetQueryPool) \
  X(vkCmdWriteTimestamp) \
  X(vkCreateBuffer) \
  X(vkCreateCommandPool) \
  X(vkCreateComputePipelines) \
  X(vkCreateDescriptorPool) \
  X(vkCreateDescriptorSetLayout) \
  X(vkCreateDevice) \
  X(vkCreateFence) \
  X(vkCreatePipelineLayout) \
  X(vkCreateQueryPool) \
  X(vkCreateShaderModule) \
  X(vkDestroyBuffer) \
  X(vkDestroyCommandPool) \
  X(vkDestroyDescriptorPool) \
  X(vkDestroyDescriptorSetLayout) \
  X(vkDestroyDevice) \
  X(vkDestroyFence) \
  X(vkDestroyInstance) \
  X(vkDestroyPipeline) \
  X(vkDestroyPipelineLayout) \
  X(vkDestroyQueryPool) \
  X(vkDestroyShaderModule) \
  X(vkDeviceWaitIdle) \
  X(vkEndCommandBuffer) \
  X(vkEnumeratePhysicalDevices) \
  X(vkFreeMemory) \
  X(vkGetBufferMemoryRequirements) \
  X(vkGetDeviceQueue) \
  X(vkGetPhysicalDeviceMemoryProperties) \
  X(vkGetPhysicalDeviceProperties) \
  X(vkGetPhysicalDeviceQueueFamilyProperties) \
  X(vkGetQueryPoolResults) \
  X(vkMapMemory) \
  X(vkQueueSubmit) \
  X(vkResetCommandBuffer) \
  X(vkResetDescriptorPool) \
  X(vkResetFences) \
  X(vkUpdateDescriptorSets) \
  X(vkWaitForFences)

#define MANTA_VK_DECLARE(name) PFN_##name name = nullptr;
PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
PFN_vkCreateInstance vkCreateInstance = nullptr;
MANTA_VK_INSTANCE_FUNCTIONS(MANTA_VK_DECLARE)
#undef MANTA_VK_DECLARE

std::mutex &loader_mutex()
{
  static std::mutex mutex;
  return mutex;
}

/** Open the Vulkan loader once per process. The library is never closed. */
void load_vulkan_loader()
{
  std::lock_guard<std::mutex> lock(loader_mutex());
  if (vkGetInstanceProcAddr) {
    return;
  }
  void *symbol = nullptr;
#ifdef _WIN32
  HMODULE library = LoadLibraryA("vulkan-1.dll");
  if (library) {
    symbol = reinterpret_cast<void *>(GetProcAddress(library, "vkGetInstanceProcAddr"));
  }
#else
#  ifdef __APPLE__
  const char *names[] = {"libvulkan.1.dylib", "libvulkan.dylib", "libMoltenVK.dylib"};
#  else
  const char *names[] = {"libvulkan.so.1", "libvulkan.so"};
#  endif
  for (const char *name : names) {
    if (void *library = dlopen(name, RTLD_NOW | RTLD_LOCAL)) {
      symbol = dlsym(library, "vkGetInstanceProcAddr");
      if (symbol) {
        break;
      }
    }
  }
#endif
  if (!symbol) {
    throw std::runtime_error("No Vulkan loader is installed (a graphics driver with Vulkan support "
                             "provides it)");
  }
  vkGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(symbol);
  vkCreateInstance = reinterpret_cast<PFN_vkCreateInstance>(
      vkGetInstanceProcAddr(nullptr, "vkCreateInstance"));
  if (!vkCreateInstance) {
    vkGetInstanceProcAddr = nullptr;
    throw std::runtime_error("The Vulkan loader has no vkCreateInstance");
  }
}

/**
 * Load every other function through `instance`. The loader's entry points dispatch on the
 * handle they are given, so the pointers stay valid for later instances too.
 */
void load_instance_functions(VkInstance instance)
{
  std::lock_guard<std::mutex> lock(loader_mutex());
#define MANTA_VK_LOAD(name) \
  name = reinterpret_cast<PFN_##name>(vkGetInstanceProcAddr(instance, #name)); \
  if (!name) { \
    throw std::runtime_error("The Vulkan loader has no " #name); \
  }
  MANTA_VK_INSTANCE_FUNCTIONS(MANTA_VK_LOAD)
#undef MANTA_VK_LOAD
}

}  // namespace

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

/** Which bindings (bit i = binding i + 1) a kernel reads and writes. */
struct KernelAccess {
  std::uint32_t reads = 0;
  std::uint32_t writes = 0;
};

/**
 * Find the storage buffers a SPIR-V compute shader loads from and stores to, by following every
 * load, store and atomic back through its access chains to the descriptor variable. A binding the
 * shader declares but never touches is in neither set. Anything not understood counts as both.
 */
KernelAccess reflect_access(const std::uint32_t *code, const std::size_t words)
{
  KernelAccess all;
  all.reads = all.writes = (1u << binding_count) - 1u;
  if (words < 5 || code[0] != 0x07230203u) {
    return all;
  }
  std::unordered_map<std::uint32_t, std::uint32_t> binding_of; /* variable -> binding */
  std::unordered_map<std::uint32_t, std::uint32_t> root;       /* pointer -> variable */
  KernelAccess found;
  const auto touch = [&](const std::uint32_t pointer, const bool read, const bool write) {
    const auto r = root.find(pointer);
    if (r == root.end()) {
      return;
    }
    const auto b = binding_of.find(r->second);
    if (b == binding_of.end() || b->second < 1 || b->second > binding_count) {
      return;
    }
    const std::uint32_t bit = 1u << (b->second - 1);
    if (read) {
      found.reads |= bit;
    }
    if (write) {
      found.writes |= bit;
    }
  };
  for (std::size_t i = 5; i < words;) {
    const std::uint32_t op = code[i] & 0xffffu, count = code[i] >> 16;
    if (count == 0 || i + count > words) {
      return all;
    }
    const std::uint32_t *w = code + i;
    switch (op) {
      case 71: /* OpDecorate target Binding n */
        if (count >= 4 && w[2] == 33) {
          binding_of[w[1]] = w[3];
        }
        break;
      case 59: /* OpVariable */
        root[w[2]] = w[2];
        break;
      case 65: /* OpAccessChain */
      case 66: /* OpInBoundsAccessChain */
      case 67: /* OpPtrAccessChain */
      case 70: /* OpInBoundsPtrAccessChain */
      case 83: /* OpCopyObject */
        if (root.count(w[3])) {
          root[w[2]] = root[w[3]];
        }
        break;
      case 61:  /* OpLoad */
      case 227: /* OpAtomicLoad */
        touch(w[3], true, false);
        break;
      case 62:  /* OpStore */
      case 228: /* OpAtomicStore */
        touch(w[1], false, true);
        break;
      case 63: /* OpCopyMemory */
        touch(w[1], false, true);
        touch(w[2], true, false);
        break;
      default:
        if ((op >= 229 && op <= 242) || op == 6035 /* OpAtomicFAddEXT */) {
          touch(w[3], true, true);
        }
        else if (op == 57 || op == 4 /* OpFunctionCall with pointers, OpCapability Addresses */) {
          if (op == 57) {
            /* Pointers passed to functions: count the buffers as read and written. */
            for (std::uint32_t k = 4; k < count; ++k) {
              touch(w[k], true, true);
            }
          }
        }
        break;
    }
    i += count;
  }
  return found;
}

constexpr std::uint32_t sets_per_pool = 256;
constexpr std::uint32_t profile_capacity = 4096;

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
  /**
   * MANTA_GPU_CHUNK=N (N > 0): after every N recorded commands the command buffer is submitted
   * without waiting and recording continues in the next one, so the device starts on the first
   * commands while the host still records the rest. submit() waits for the last chunk only.
   */
  std::size_t chunk_commands = 0;
  bool half_storage = false;
  std::size_t commands_in_chunk = 0;
  std::vector<VkCommandBuffer> chunks;
  std::size_t chunk_index = 0;
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

  VkQueryPool profile_pool = VK_NULL_HANDLE;
  bool profiling = false;
  std::vector<const char *> profile_labels;
  std::vector<std::pair<std::string, double>> profile;

  Buffer dummy;
  bool recording = false;

  /** Read and write sets of each kernel, from its SPIR-V. */
  std::vector<KernelAccess> access;
  /**
   * Barrier placement. Off (the default): a full barrier after every command. On
   * (MANTA_GPU_TRACKED_BARRIERS=1): a barrier only before a command that reads a buffer written,
   * or writes a buffer read or written, since the last barrier.
   */
  bool tracked = false;
  std::vector<std::uint64_t> pending_reads, pending_writes;
  VkPipelineStageFlags pending_stages = 0;

  static bool contains(const std::vector<std::uint64_t> &set, const std::uint64_t handle)
  {
    return std::find(set.begin(), set.end(), handle) != set.end();
  }

  /** Record the hazard of a command about to be recorded; emits a barrier first if needed. */
  void before_command(const std::uint64_t *reads, const std::size_t read_count,
                      const std::uint64_t *writes, const std::size_t write_count,
                      const VkPipelineStageFlags stage)
  {
    if (!tracked) {
      return;
    }
    bool hazard = false;
    for (std::size_t i = 0; i < read_count && !hazard; ++i) {
      hazard = contains(pending_writes, reads[i]);
    }
    for (std::size_t i = 0; i < write_count && !hazard; ++i) {
      hazard = contains(pending_writes, writes[i]) || contains(pending_reads, writes[i]);
    }
    if (hazard) {
      barrier_between(pending_stages, stage);
      pending_reads.clear();
      pending_writes.clear();
      pending_stages = 0;
    }
    pending_reads.insert(pending_reads.end(), reads, reads + read_count);
    pending_writes.insert(pending_writes.end(), writes, writes + write_count);
    pending_stages |= stage;
  }

  /** After a recorded command: the untracked mode's full barrier, and the chunking. */
  void after_command()
  {
    if (!tracked) {
      barrier();
    }
    if (chunk_commands && ++commands_in_chunk >= chunk_commands) {
      next_chunk();
    }
  }

  /** Submit the commands recorded so far without waiting and continue in a new buffer. */
  void next_chunk()
  {
    flush_pending();
    check(vkEndCommandBuffer(commands), "vkEndCommandBuffer");
    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &commands;
    MANTA_PERF(g_perf.chunk_submits++);
    MANTA_PERF(chunk_index == 0 ? (g_perf.start_delay_ms += std::chrono::duration<double, std::milli>(
                                       std::chrono::steady_clock::now() - g_perf.record_start)
                                       .count()) : 0.0);
    check(vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE), "vkQueueSubmit");
    ++chunk_index;
    if (chunk_index == chunks.size()) {
      VkCommandBufferAllocateInfo info{};
      info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
      info.commandPool = command_pool;
      info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      info.commandBufferCount = 1;
      VkCommandBuffer buffer = VK_NULL_HANDLE;
      check(vkAllocateCommandBuffers(device, &info, &buffer), "vkAllocateCommandBuffers");
      chunks.push_back(buffer);
    }
    commands = chunks[chunk_index];
    check(vkResetCommandBuffer(commands, 0), "vkResetCommandBuffer");
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(commands, &begin_info), "vkBeginCommandBuffer");
    commands_in_chunk = 0;
  }

  /** Before the command buffer ends: order everything against later submissions. */
  void flush_pending()
  {
    if (tracked && !pending_writes.empty()) {
      barrier();
    }
    pending_reads.clear();
    pending_writes.clear();
    pending_stages = 0;
  }

  void barrier_between(const VkPipelineStageFlags src, const VkPipelineStageFlags dst) const
  {
    VkMemoryBarrier memory_barrier{};
    memory_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memory_barrier.srcAccessMask = 0;
    if (src & VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT) {
      memory_barrier.srcAccessMask |= VK_ACCESS_SHADER_WRITE_BIT;
    }
    if (src & VK_PIPELINE_STAGE_TRANSFER_BIT) {
      memory_barrier.srcAccessMask |= VK_ACCESS_TRANSFER_WRITE_BIT;
    }
    memory_barrier.dstAccessMask = 0;
    if (dst & VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT) {
      memory_barrier.dstAccessMask |= VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    }
    if (dst & VK_PIPELINE_STAGE_TRANSFER_BIT) {
      memory_barrier.dstAccessMask |= VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    }
    MANTA_PERF(g_perf.barriers++);
    vkCmdPipelineBarrier(commands, src, dst, 0, 1, &memory_barrier, 0, nullptr, 0, nullptr);
  }

  struct Staging {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void *mapped = nullptr;
    std::size_t size = 0;
  };
  /** Transfers share one host-visible buffer kept between submits; it grows after a submit
   * that needed more, and a transfer that does not fit meanwhile gets a buffer of its own. */
  Staging arena;
  std::size_t arena_used = 0;
  std::size_t arena_wanted = 0;
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

  Staging make_staging(const std::size_t bytes) const
  {
    Staging entry;
    make_buffer(bytes,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                entry.buffer,
                entry.memory);
    check(vkMapMemory(device, entry.memory, 0, VK_WHOLE_SIZE, 0, &entry.mapped), "vkMapMemory");
    entry.size = bytes;
    return entry;
  }

  void free_staging(const Staging &entry) const
  {
    vkDestroyBuffer(device, entry.buffer, nullptr);
    vkFreeMemory(device, entry.memory, nullptr);
  }

  /** Room for one transfer of `bytes`: the buffer, its offset, and the mapped address. */
  void *staging_buffer(const std::size_t bytes, VkBuffer &buffer, VkDeviceSize &offset)
  {
    const std::size_t aligned = (bytes + 255) & ~std::size_t(255);
    arena_wanted += aligned;
    if (arena.buffer && arena_used + aligned <= arena.size) {
      buffer = arena.buffer;
      offset = arena_used;
      arena_used += aligned;
      return static_cast<char *>(arena.mapped) + offset;
    }
    staging.push_back(make_staging(bytes));
    buffer = staging.back().buffer;
    offset = 0;
    return staging.back().mapped;
  }

  /** After a submit has finished (or its commands were dropped). */
  void release_staging()
  {
    for (const Staging &entry : staging) {
      free_staging(entry);
    }
    staging.clear();
    if (arena_wanted > arena.size) {
      if (arena.buffer) {
        free_staging(arena);
        arena = Staging();
      }
      arena = make_staging(std::max<std::size_t>(arena_wanted + arena_wanted / 4, 1 << 20));
    }
    arena_used = 0;
    arena_wanted = 0;
  }

  void destroy_arena()
  {
    if (arena.buffer) {
      free_staging(arena);
      arena = Staging();
    }
  }

  /** Close the timing interval of the command just recorded. */
  void mark(const char *label)
  {
    if (!profiling || profile_labels.size() + 1 >= profile_capacity) {
      return;
    }
    profile_labels.push_back(label);
    vkCmdWriteTimestamp(commands,
                        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                        profile_pool,
                        static_cast<std::uint32_t>(profile_labels.size()));
  }

  /** Make every earlier write visible to every later command. */
  void barrier() const
  {
    MANTA_PERF(g_perf.barriers++);
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
  if (const char *mode = std::getenv("MANTA_GPU_TRACKED_BARRIERS")) {
    v.tracked = mode[0] == '1';
  }
  load_vulkan_loader();

  VkApplicationInfo app{};
  app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app.pApplicationName = "manta_improve";
  app.apiVersion = VK_API_VERSION_1_1;
  VkInstanceCreateInfo instance_info{};
  instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instance_info.pApplicationInfo = &app;
  check(vkCreateInstance(&instance_info, nullptr, &v.instance), "vkCreateInstance");
  load_instance_functions(v.instance);

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
  /* Channels may be stored as fp16 (storageBuffer16BitAccess, Vulkan 1.1 core; arithmetic
   * stays fp32, so shaderFloat16 is not needed). Enabled only where the device has it. */
  VkPhysicalDevice16BitStorageFeatures storage16{};
  storage16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
  const auto get_features2 = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
      vkGetInstanceProcAddr(v.instance, "vkGetPhysicalDeviceFeatures2"));
  if (get_features2) {
    VkPhysicalDeviceFeatures2 features{};
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features.pNext = &storage16;
    get_features2(v.physical, &features);
  }
  v.half_storage = storage16.storageBuffer16BitAccess == VK_TRUE;
  VkPhysicalDevice16BitStorageFeatures enabled16{};
  enabled16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
  enabled16.storageBuffer16BitAccess = VK_TRUE;
  if (v.half_storage) {
    device_info.pNext = &enabled16;
  }
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
  v.chunks.push_back(v.commands);
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
    query_info.queryCount = profile_capacity;
    check(vkCreateQueryPool(v.device, &query_info, nullptr, &v.profile_pool),
          "vkCreateQueryPool");
  }

  v.dummy = create_buffer(16);
}

Compute::~Compute()
{
  Impl &v = *impl_;
  vkDeviceWaitIdle(v.device);
  v.release_staging();
  v.destroy_arena();
  destroy_buffer(v.dummy);
  for (const VkPipeline pipeline : v.pipelines) {
    vkDestroyPipeline(v.device, pipeline, nullptr);
  }
  for (const VkShaderModule module : v.modules) {
    vkDestroyShaderModule(v.device, module, nullptr);
  }
  if (v.timestamps) {
    vkDestroyQueryPool(v.device, v.timestamps, nullptr);
    vkDestroyQueryPool(v.device, v.profile_pool, nullptr);
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

bool Compute::supports_half_storage() const
{
  return impl_->half_storage;
}

void Compute::set_command_batch(const std::size_t dispatches)
{
  impl_->chunk_commands = dispatches;
}

void Compute::set_profiling(const bool enabled)
{
  if (impl_->recording) {
    throw std::logic_error("set_profiling while recording");
  }
  impl_->profiling = enabled && impl_->profile_pool != VK_NULL_HANDLE;
}

const std::vector<std::pair<std::string, double>> &Compute::last_profile() const
{
  return impl_->profile;
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
    /* Only reached while unwinding from an error: drop the unsubmitted commands. */
    vkEndCommandBuffer(v.commands);
    v.recording = false;
    v.downloads.clear();
    v.release_staging();
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
  v.access.push_back(reflect_access(spirv, word_count));
  return static_cast<int>(v.pipelines.size()) - 1;
}

void Compute::begin()
{
  Impl &v = *impl_;
  if (v.recording) {
    throw std::logic_error("begin while recording");
  }
  v.chunk_index = 0;
  v.commands_in_chunk = 0;
  v.commands = v.chunks[0];
  check(vkResetCommandBuffer(v.commands, 0), "vkResetCommandBuffer");
  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  check(vkBeginCommandBuffer(v.commands, &begin_info), "vkBeginCommandBuffer");
  if (v.timestamps) {
    vkCmdResetQueryPool(v.commands, v.timestamps, 0, 2);
    vkCmdWriteTimestamp(v.commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, v.timestamps, 0);
  }
  v.profile_labels.clear();
  if (v.profiling) {
    vkCmdResetQueryPool(v.commands, v.profile_pool, 0, profile_capacity);
    vkCmdWriteTimestamp(v.commands, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, v.profile_pool, 0);
  }
  v.pending_reads.clear();
  v.pending_writes.clear();
  v.pending_stages = 0;
  v.recording = true;
  MANTA_PERF(g_perf.record_start = std::chrono::steady_clock::now());
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
  VkDeviceSize offset = 0;
  std::memcpy(v.staging_buffer(bytes, staging, offset), data, bytes);
  MANTA_PERF((g_perf.uploads++, g_perf.upload_bytes += bytes));
  v.before_command(nullptr, 0, &target.handle, 1, VK_PIPELINE_STAGE_TRANSFER_BIT);
  VkBufferCopy region{};
  region.srcOffset = offset;
  region.size = bytes;
  vkCmdCopyBuffer(v.commands, staging, to_vk(target), 1, &region);
  v.after_command();
  v.mark("upload");
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
  VkDeviceSize offset = 0;
  void *mapped = v.staging_buffer(bytes, staging, offset);
  MANTA_PERF((g_perf.downloads++, g_perf.download_bytes += bytes));
  v.before_command(&source.handle, 1, nullptr, 0, VK_PIPELINE_STAGE_TRANSFER_BIT);
  VkBufferCopy region{};
  region.dstOffset = offset;
  region.size = bytes;
  vkCmdCopyBuffer(v.commands, to_vk(source), staging, 1, &region);
  v.after_command();
  v.mark("download");
  v.downloads.push_back({mapped, out, bytes});
}

void Compute::fill(const Buffer &target, const std::uint32_t value)
{
  Impl &v = *impl_;
  MANTA_PERF(g_perf.fills++);
  v.before_command(nullptr, 0, &target.handle, 1, VK_PIPELINE_STAGE_TRANSFER_BIT);
  vkCmdFillBuffer(v.commands, to_vk(target), 0, VK_WHOLE_SIZE, value);
  v.after_command();
  v.mark("fill");
}

void Compute::dispatch(const int kernel,
                       const void *push_constants,
                       const std::size_t push_bytes,
                       const Bindings &buffers,
                       const std::uint32_t group_count,
                       const char *label)
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
  MANTA_PERF((g_perf.dispatches++, g_perf.groups += group_count,
              g_perf.tiny_dispatches += group_count <= 8 ? 1 : 0));
  if (v.tracked) {
    const KernelAccess &access = v.access.at(static_cast<std::size_t>(kernel));
    std::uint64_t reads[binding_count], writes[binding_count];
    std::size_t read_count = 0, write_count = 0;
    for (std::size_t i = 0; i < binding_count; ++i) {
      const std::uint64_t handle = (buffers[i] ? *buffers[i] : v.dummy).handle;
      if (access.reads & (1u << i)) {
        reads[read_count++] = handle;
      }
      if (access.writes & (1u << i)) {
        writes[write_count++] = handle;
      }
    }
    v.before_command(reads, read_count, writes, write_count, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
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
  v.after_command();
  v.mark(label);
}

double Compute::submit()
{
  Impl &v = *impl_;
  if (!v.recording) {
    throw std::logic_error("submit without begin");
  }
  v.flush_pending();
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
  MANTA_PERF(v.chunk_index == 0 ? (g_perf.start_delay_ms += std::chrono::duration<double, std::milli>(
                                       std::chrono::steady_clock::now() - g_perf.record_start)
                                       .count()) : 0.0);
  MANTA_PERF((g_perf.submits++, g_perf.record_ms += std::chrono::duration<double, std::milli>(
                                     std::chrono::steady_clock::now() - g_perf.record_start)
                                     .count()));
  const auto wait_start = std::chrono::steady_clock::now();
  check(vkQueueSubmit(v.queue, 1, &submit_info, v.fence), "vkQueueSubmit");
  check(vkWaitForFences(v.device, 1, &v.fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
  MANTA_PERF(g_perf.wait_ms += std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - wait_start).count());
  (void)wait_start;

  for (const Impl::PendingDownload &pending : v.downloads) {
    std::memcpy(pending.out, pending.mapped, pending.bytes);
  }
  v.downloads.clear();
  v.release_staging();

  v.profile.clear();
  if (v.profiling && !v.profile_labels.empty()) {
    std::vector<std::uint64_t> ticks(v.profile_labels.size() + 1);
    check(vkGetQueryPoolResults(v.device,
                                v.profile_pool,
                                0,
                                static_cast<std::uint32_t>(ticks.size()),
                                ticks.size() * sizeof(std::uint64_t),
                                ticks.data(),
                                sizeof(std::uint64_t),
                                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
          "vkGetQueryPoolResults");
    for (std::size_t i = 0; i < v.profile_labels.size(); ++i) {
      const double ms = static_cast<double>((ticks[i + 1] - ticks[i]) & v.timestamp_mask) *
                        v.timestamp_period_ns * 1.0e-6;
      const std::string label = v.profile_labels[i];
      auto entry = std::find_if(v.profile.begin(), v.profile.end(), [&](const auto &item) {
        return item.first == label;
      });
      if (entry == v.profile.end()) {
        v.profile.emplace_back(label, ms);
      }
      else {
        entry->second += ms;
      }
    }
  }

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

// MemoryManager.h

#ifndef MEMORYMANAGER_H
#define MEMORYMANAGER_H

#include <cuda_runtime.h>
#include <mutex>
#include <queue>
#include <vector>
#include <unordered_map>
#include "ErrorChecking.h"

// 定义内存类型枚举
enum class MemoryType {
    SubTensor,
    CoreTensor,
    FactorMatrix,
    IntermediateResult
};

// Device Memory Pool 管理
class DeviceMemoryPool {
public:
    DeviceMemoryPool(size_t block_size, size_t num_blocks);
    ~DeviceMemoryPool();

    void* Allocate();
    void Free(void* ptr);

private:
    size_t block_size_;
    size_t total_blocks_;
    std::queue<void*> free_blocks_;
    std::mutex mutex_;
};

// Host Memory Pool 管理（使用Pinned Memory）
class HostMemoryPool {
public:
    HostMemoryPool(size_t block_size, size_t num_blocks);
    ~HostMemoryPool();

    void* Allocate();
    void Free(void* ptr);

private:
    size_t block_size_;
    size_t total_blocks_;
    std::queue<void*> free_blocks_;
    std::mutex mutex_;
};

// 内存池管理器，支持多类型内存池
class MemoryManager {
public:
    MemoryManager(size_t device_block_size, size_t device_num_blocks,
                 size_t host_block_size, size_t host_num_blocks,
                 size_t num_streams);
    ~MemoryManager();

    // 获取一个可用的CUDA流
    cudaStream_t GetStream();

    // 根据内存类型分配设备内存
    void* AllocateDeviceMemory(MemoryType type);
    void FreeDeviceMemory(MemoryType type, void* ptr);

    // 根据内存类型分配主机Pinned内存
    void* AllocateHostMemory(MemoryType type);
    void FreeHostMemory(MemoryType type, void* ptr);

private:
    DeviceMemoryPool device_pool_;
    HostMemoryPool host_pool_;
    std::vector<cudaStream_t> streams_;
    size_t num_streams_;
    size_t current_stream_;
    std::mutex stream_mutex_;
    std::mutex memory_mutex_;
};

#endif // MEMORYMANAGER_H

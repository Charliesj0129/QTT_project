// MemoryManager.cpp

#include "MemoryManager.h"
#include <iostream>

// DeviceMemoryPool 实现
DeviceMemoryPool::DeviceMemoryPool(size_t block_size, size_t num_blocks)
    : block_size_(block_size), total_blocks_(num_blocks) {
    for(size_t i = 0; i < total_blocks_; ++i){
        void* ptr;
        CUDA_CHECK(cudaMalloc(&ptr, block_size_));
        free_blocks_.push(ptr);
    }
}

DeviceMemoryPool::~DeviceMemoryPool(){
    while(!free_blocks_.empty()){
        void* ptr = free_blocks_.front();
        free_blocks_.pop();
        CUDA_CHECK(cudaFree(ptr));
    }
}

void* DeviceMemoryPool::Allocate(){
    std::lock_guard<std::mutex> lock(mutex_);
    if(free_blocks_.empty()){
        // 动态分配更多内存块
        void* ptr;
        CUDA_CHECK(cudaMalloc(&ptr, block_size_));
        return ptr;
    } else {
        void* ptr = free_blocks_.front();
        free_blocks_.pop();
        return ptr;
    }
}

void DeviceMemoryPool::Free(void* ptr){
    std::lock_guard<std::mutex> lock(mutex_);
    free_blocks_.push(ptr);
}

// HostMemoryPool 实现
HostMemoryPool::HostMemoryPool(size_t block_size, size_t num_blocks)
    : block_size_(block_size), total_blocks_(num_blocks) {
    for(size_t i = 0; i < total_blocks_; ++i){
        void* ptr;
        CUDA_CHECK(cudaMallocHost(&ptr, block_size_));
        free_blocks_.push(ptr);
    }
}

HostMemoryPool::~HostMemoryPool(){
    while(!free_blocks_.empty()){
        void* ptr = free_blocks_.front();
        free_blocks_.pop();
        CUDA_CHECK(cudaFreeHost(ptr));
    }
}

void* HostMemoryPool::Allocate(){
    std::lock_guard<std::mutex> lock(mutex_);
    if(free_blocks_.empty()){
        // 动态分配更多固定内存块
        void* ptr;
        CUDA_CHECK(cudaMallocHost(&ptr, block_size_));
        return ptr;
    } else {
        void* ptr = free_blocks_.front();
        free_blocks_.pop();
        return ptr;
    }
}

void HostMemoryPool::Free(void* ptr){
    std::lock_guard<std::mutex> lock(mutex_);
    free_blocks_.push(ptr);
}

// MemoryManager 实现
MemoryManager::MemoryManager(size_t device_block_size, size_t device_num_blocks,
                             size_t host_block_size, size_t host_num_blocks,
                             size_t num_streams)
    : device_pool_(device_block_size, device_num_blocks),
      host_pool_(host_block_size, host_num_blocks),
      num_streams_(num_streams),
      current_stream_(0) {
    for(size_t i = 0; i < num_streams_; ++i){
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        streams_.push_back(stream);
    }
}

MemoryManager::~MemoryManager(){
    for(auto stream : streams_){
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
}

cudaStream_t MemoryManager::GetStream(){
    std::lock_guard<std::mutex> lock(stream_mutex_);
    cudaStream_t stream = streams_[current_stream_];
    current_stream_ = (current_stream_ + 1) % num_streams_;
    return stream;
}

void* MemoryManager::AllocateDeviceMemory(MemoryType type){
    std::lock_guard<std::mutex> lock(memory_mutex_);
    return device_pool_.Allocate();
}

void MemoryManager::FreeDeviceMemory(MemoryType type, void* ptr){
    std::lock_guard<std::mutex> lock(memory_mutex_);
    device_pool_.Free(ptr);
}

void* MemoryManager::AllocateHostMemory(MemoryType type){
    std::lock_guard<std::mutex> lock(memory_mutex_);
    return host_pool_.Allocate();
}

void MemoryManager::FreeHostMemory(MemoryType type, void* ptr){
    std::lock_guard<std::mutex> lock(memory_mutex_);
    host_pool_.Free(ptr);
}

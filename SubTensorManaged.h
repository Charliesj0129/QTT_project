// SubTensorManaged.h
#ifndef SUBTENSORMANAGED_H
#define SUBTENSORMANAGED_H

#include <vector>
#include <cuda_runtime.h>

// 用于管理子张量的结构
struct SubTensorManaged {
    float* d_data;  // GPU 内存指针
    float* h_data;  // Host (Pinned) 内存指针
    std::vector<size_t> start_indices; // 全局起始索引
    std::vector<size_t> sizes; // 子张量尺寸
    size_t total_size; // 总大小
    cudaStream_t stream; // 每个子张量专用的 CUDA 流

    // 构造函数
    SubTensorManaged(const std::vector<size_t>& starts, const std::vector<size_t>& szs)
        : start_indices(starts), sizes(szs), d_data(nullptr), h_data(nullptr), stream(0) {
        total_size = 1;
        for (size_t size : sizes) {
            total_size *= size;
        }
    }

    // 禁用默认构造函数
    SubTensorManaged() = delete;
};

#endif // SUBTENSORMANAGED_H

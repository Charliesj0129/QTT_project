// GPUContext.h
#ifndef GPUCONTEXT_H
#define GPUCONTEXT_H

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <mutex>
#include "ErrorChecking.h"

class GPUContext {
public:
    int gpu_id;
    cublasHandle_t cublas_handle;
    cusolverDnHandle_t cusolver_handle;
    std::mutex gpu_mutex; // 保护GPU上下文的互斥锁
    int desired_rank; // CI 的目标秩

    GPUContext(int id, int rank){
        gpu_id = id;
        desired_rank = rank;
        CUDA_CHECK(cudaSetDevice(gpu_id));
        CUBLAS_CHECK(cublasCreate(&cublas_handle));
        CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));
    }

    ~GPUContext(){
        CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
        CUBLAS_CHECK(cublasDestroy(cublas_handle));
    }
};

#endif // GPUCONTEXT_H

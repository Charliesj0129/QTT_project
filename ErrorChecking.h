// ErrorChecking.h
#ifndef ERRORCHECKING_H
#define ERRORCHECKING_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cstdio>
#include <cstdlib>

// CUDA 错误检查宏
#define CUDA_CHECK(err) { gpuAssert((err), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
    if (code != cudaSuccess){
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// cuBLAS 错误检查宏
#define CUBLAS_CHECK(err) { cublasAssert((err), __FILE__, __LINE__); }
inline void cublasAssert(cublasStatus_t code, const char *file, int line, bool abort=true){
    if (code != CUBLAS_STATUS_SUCCESS){
        fprintf(stderr,"CUBLASassert: %d %s %d\n", code, file, line);
        if (abort) exit(code);
    }
}

// cuSOLVER 错误检查宏
#define CUSOLVER_CHECK(err) { cusolverAssert((err), __FILE__, __LINE__); }
inline void cusolverAssert(cusolverStatus_t code, const char *file, int line, bool abort=true){
    if (code != CUSOLVER_STATUS_SUCCESS){
        fprintf(stderr,"CUSOLVERassert: %d %s %d\n", code, file, line);
        if (abort) exit(code);
    }
}

#endif // ERRORCHECKING_H

// MatrixCrossInterpolation.cu

#include "MatrixCrossInterpolation.h"
#include <iostream>
#include <cfloat>
#include "ErrorChecking.h"
#include <cusolverDn.h>

// CUDA 核函数实现：查找每行的最大绝对值元素
__global__ void MatrixCrossInterpolation::FindRowMaxKernel(const float* d_data, int rows, int cols, float* d_row_max_vals, int* d_row_max_cols){
    int row = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float sdata[];
    float max_val = -FLT_MAX;
    int max_col = -1;

    for(int col = tid; col < cols; col += blockDim.x){
        float val = abs(d_data[row * cols + col]);
        if(val > max_val){
            max_val = val;
            max_col = col;
        }
    }

    sdata[tid] = max_val;
    __syncthreads();

    // 归约查找块内的最大值
    for(int s = blockDim.x / 2; s > 0; s >>=1){
        if(tid < s){
            if(sdata[tid + s] > sdata[tid]){
                sdata[tid] = sdata[tid + s];
            }
        }
        __syncthreads();
    }

    // 第一个线程写入行的最大值和对应列
    if(tid == 0){
        d_row_max_vals[row] = sdata[0];
        d_row_max_cols[row] = max_col;
    }
}

// CUDA 核函数实现：清零选定的行和列
__global__ void MatrixCrossInterpolation::ZeroRowAndColumnKernel(float* d_data, int rows, int cols, int zero_row, int zero_col){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < rows * cols){
        int row = idx / cols;
        int col = idx % cols;
        if(row == zero_row || col == zero_col){
            d_data[idx] = 0.0f;
        }
    }
}

// 构造函数
MatrixCrossInterpolation::MatrixCrossInterpolation(float* d_M_input, int m, int n, int desired_rank, 
                                                   cublasHandle_t cublas, 
                                                   cusolverDnHandle_t cusolver,
                                                   cudaStream_t stream)
    : d_M(d_M_input), rows(m), cols(n), rank(desired_rank), 
      cublas_handle(cublas), cusolver_handle(cusolver), stream_(stream) {
    // 设置 cuBLAS 和 cuSOLVER 使用的流
    cublasSetStream(cublas_handle, stream_);
    cusolverDnSetStream(cusolver_handle, stream_);
    
    // 分配 P, C, R, P_inv, A_tilde 的GPU内存
    CUDA_CHECK(cudaMalloc(&d_P, rank * rank * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, rows * rank * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_R, rank * cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_P_inv, rank * rank * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_A_tilde, rows * cols * sizeof(float)));
}

// 实现 Rook Pivoting
void MatrixCrossInterpolation::find_pivots_rook(){
    for(int pivot_num = 0; pivot_num < rank; ++pivot_num){
        // Step 1: 查找每行的最大值
        float* d_row_max_vals;
        int* d_row_max_cols;
        CUDA_CHECK(cudaMalloc(&d_row_max_vals, rows * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_row_max_cols, rows * sizeof(int)));
        
        int threads = 256;
        int blocks = rows;
        size_t shared_mem = threads * sizeof(float);
        
        FindRowMaxKernel<<<blocks, threads, shared_mem, stream_>>>(d_M, rows, cols, d_row_max_vals, d_row_max_cols);
        CUDA_CHECK(cudaGetLastError());
        
        // 拷贝行最大值和列索引到主机
        std::vector<float> h_row_max_vals(rows);
        std::vector<int> h_row_max_cols(rows);
        CUDA_CHECK(cudaMemcpyAsync(h_row_max_vals.data(), d_row_max_vals, rows * sizeof(float), cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaMemcpyAsync(h_row_max_cols.data(), d_row_max_cols, rows * sizeof(int), cudaMemcpyDeviceToHost, stream_));
        
        // 同步流以确保数据已经被拷贝
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        
        // Step 2: 在主机上找到全局最大的值及其行和列
        float max_val = -FLT_MAX;
        int max_row = -1;
        int max_col = -1;
        for(int r = 0; r < rows; ++r){
            if(h_row_max_vals[r] > max_val){
                max_val = h_row_max_vals[r];
                max_row = r;
                max_col = h_row_max_cols[r];
            }
        }
        
        if(max_row == -1 || max_col == -1){
            std::cerr << "Failed to find a valid pivot at iteration " << pivot_num << std::endl;
            break;
        }
        
        T.push_back(max_row);
        J.push_back(max_col);
        
        // Step 3: 清零已选择的行和列，避免重复选择
        int total_elements = rows * cols;
        int block_size = 256;
        int grid_size = (total_elements + block_size - 1) / block_size;
        ZeroRowAndColumnKernel<<<grid_size, block_size, 0, stream_>>>(d_M, rows, cols, max_row, max_col);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // 清理临时设备内存
        CUDA_CHECK(cudaFree(d_row_max_vals));
        CUDA_CHECK(cudaFree(d_row_max_cols));
    }
}

void MatrixCrossInterpolation::find_pivots_full_search(){
    // 实现全局搜索的枢轴选择
    for(int pivot_num = 0; pivot_num < rank; ++pivot_num){
        // 在 GPU 上实现全局最大值搜索（可选，也可以在主机上实现）
        float max_val = -FLT_MAX;
        int max_row = -1;
        int max_col = -1;

        // 将矩阵拷贝到主机（效率较低，可优化）
        std::vector<float> h_M(rows * cols);
        CUDA_CHECK(cudaMemcpy(h_M.data(), d_M, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));

        for(int i = 0; i < rows; ++i){
            for(int j = 0; j < cols; ++j){
                float val = abs(h_M[i * cols + j]);
                if(val > max_val){
                    max_val = val;
                    max_row = i;
                    max_col = j;
                }
            }
        }

        if(max_row == -1 || max_col == -1){
            std::cerr << "Failed to find a valid pivot at iteration " << pivot_num << std::endl;
            break;
        }

        T.push_back(max_row);
        J.push_back(max_col);

        // 清零已选择的行和列，避免重复选择
        int total_elements = rows * cols;
        int block_size = 256;
        int grid_size = (total_elements + block_size - 1) / block_size;
        ZeroRowAndColumnKernel<<<grid_size, block_size, 0, stream_>>>(d_M, rows, cols, max_row, max_col);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

void MatrixCrossInterpolation::construct_interpolation(){
// Step 1: 从 d_M 中提取子矩阵 P, C, R
// Step 2: Compute inverse of P
int* d_info;
CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
int* d_ipiv;
CUDA_CHECK(cudaMalloc(&d_ipiv, rank * sizeof(int)));
float* d_work;
int lwork;
cusolverDnSgetrf_bufferSize(cusolver_handle, rank, rank, d_P, rank, &lwork);
CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(float)));

cusolverDnSgetrf(cusolver_handle, rank, rank, d_P, rank, d_work, d_ipiv, d_info);
CUDA_CHECK(cudaDeviceSynchronize());

// Check for successful LU factorization
int h_info = 0;
CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
if(h_info != 0){
    std::cerr << "LU factorization failed with info = " << h_info << std::endl;
    // Handle error appropriately
}

cusolverDnSgetri(cusolver_handle, rank, d_P, rank, d_ipiv, d_work, lwork, d_info);
CUDA_CHECK(cudaDeviceSynchronize());

// Check for successful inversion
CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
if(h_info != 0){
    std::cerr << "Matrix inversion failed with info = " << h_info << std::endl;
    // Handle error appropriately
}

// Step 3: Compute A_tilde = C * P_inv * R
float alpha = 1.0f;
float beta = 0.0f;

float* d_temp;
CUDA_CHECK(cudaMalloc(&d_temp, rows * rank * sizeof(float)));

// Compute C * P_inv
cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
            rows, rank, rank,
            &alpha,
            d_C, rows,
            d_P, rank,
            &beta,
            d_temp, rows);

// Compute (C * P_inv) * R
cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
            rows, cols, rank,
            &alpha,
            d_temp, rows,
            d_R, rank,
            &beta,
            d_A_tilde, rows);

// Free temporary memory
CUDA_CHECK(cudaFree(d_temp));
CUDA_CHECK(cudaFree(d_work));
CUDA_CHECK(cudaFree(d_info));
CUDA_CHECK(cudaFree(d_ipiv));
for(int i = 0; i < rank; ++i){
    // 提取 P
    CUDA_CHECK(cudaMemcpyAsync(&d_P[i * rank], &d_M[T[i] * cols + J[i]], rank * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
    
    // 提取 C
    for(int j = 0; j < rows; ++j){
        CUDA_CHECK(cudaMemcpyAsync(&d_C[j * rank + i], &d_M[j * cols + J[i]], sizeof(float), cudaMemcpyDeviceToDevice, stream_));
    }
    
    // 提取 R
    CUDA_CHECK(cudaMemcpyAsync(&d_R[i * cols], &d_M[T[i] * cols], cols * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
}

// Step 2: 计算 P 的逆矩阵 d_P_inv
int* d_info;
CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
int* d_ipiv;
CUDA_CHECK(cudaMalloc(&d_ipiv, rank * sizeof(int)));
float* d_work;
int lwork;
cusolverDnSgetrf_bufferSize(cusolver_handle, rank, rank, d_P, rank, &lwork);
CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(float)));
}

Eigen::MatrixXf MatrixCrossInterpolation::get_interpolated_matrix() const{
    // 将 d_A_tilde 从 GPU 拷贝到主机并返回
    Eigen::MatrixXf A_tilde_host(rows, cols);
    CUDA_CHECK(cudaMemcpy(A_tilde_host.data(), d_A_tilde, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));
    return A_tilde_host;
}

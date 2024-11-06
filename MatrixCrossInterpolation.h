// MatrixCrossInterpolation.h

#ifndef MATRIXCROSSINTERPOLATION_H
#define MATRIXCROSSINTERPOLATION_H

#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <Eigen/Dense>

class MatrixCrossInterpolation {
public:
    MatrixCrossInterpolation(float* d_M, int rows, int cols, int rank, 
                             cublasHandle_t cublas_handle, 
                             cusolverDnHandle_t cusolver_handle,
                             cudaStream_t stream);
    
    // 选择枢轴行和列
    void find_pivots_rook();
    void find_pivots_full_search();
    
    // 构建插值矩阵
    void construct_interpolation();
    
    // 获取插值矩阵
    Eigen::MatrixXf get_interpolated_matrix() const;

private:
    float* d_M; // GPU 上的输入矩阵指针
    int rows;
    int cols;
    int rank;
    std::vector<int> T; // Pivot row indices
    std::vector<int> J; // Pivot column indices
    float* d_P; // Pivot matrix在GPU上的指针
    float* d_C; // Column matrix
    float* d_R; // Row matrix
    float* d_P_inv; // Pseudo-inverse of P
    float* d_A_tilde; // Interpolated matrix
    cublasHandle_t cublas_handle;
    cusolverDnHandle_t cusolver_handle;
    cudaStream_t stream_; // CUDA 流

    // CUDA 核函数来查找每行的最大元素
    static __global__ void FindRowMaxKernel(const float* d_data, int rows, int cols, float* d_row_max_vals, int* d_row_max_cols);
    
    // CUDA 核函数来清零选定的行和列
    static __global__ void ZeroRowAndColumnKernel(float* d_data, int rows, int cols, int zero_row, int zero_col);
};

#endif // MATRIXCROSSINTERPOLATION_H

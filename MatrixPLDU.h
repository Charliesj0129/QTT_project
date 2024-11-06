#ifndef MATRIXPLDU_H
#define MATRIXPLDU_H

#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <cusolverDn.h>

class MatrixPLDU {
public:
    MatrixPLDU(float* d_M, int n, cusolverDnHandle_t cusolver_handle, cublasHandle_t cublas_handle);
    
    void compute();
    
    Eigen::MatrixXf getL() const;
    Eigen::MatrixXf getD() const;
    Eigen::MatrixXf getU() const;
    Eigen::MatrixXi getP() const; // Permutation indices

private:
    float* d_M; // GPU 上的輸入矩陣指針
    int n; // 矩陣大小 (n x n)
    float* d_L;
    float* d_D;
    float* d_U;
    int* d_pivots;
    cusolverDnHandle_t cusolver_handle;
    cublasHandle_t cublas_handle;

    Eigen::MatrixXf L;
    Eigen::MatrixXf D;
    Eigen::MatrixXf U;
    Eigen::MatrixXi P;
};

#endif // MATRIXPLDU_H

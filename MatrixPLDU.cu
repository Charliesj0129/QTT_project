#include "MatrixPLDU.h"
#include <iostream>
#include "ErrorChecking.h"

MatrixPLDU::MatrixPLDU(float* d_M_input, int size, cusolverDnHandle_t cusolver, cublasHandle_t cublas)
    : d_M(d_M_input), n(size), cusolver_handle(cusolver), cublas_handle(cublas) {
    // 分配 L, D, U 的 GPU 內存
    CUDA_CHECK(cudaMalloc(&d_L, n * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_D, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_U, n * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pivots, n * sizeof(int)));
}

void MatrixPLDU::compute(){
    // 使用 cuSOLVER 進行 LU 分解
    int lda = n;
    int lwork = 0;
    int info = 0;

    // 獲取 LU 分解的工作空間大小
    CUSOLVER_CHECK(cusolverDnSgetrf_bufferSize(cusolver_handle, n, n, d_M, lda, &lwork));

    // 分配工作空間
    float* d_work;
    CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(float)));

    // 執行 LU 分解
    CUSOLVER_CHECK(cusolverDnSgetrf(cusolver_handle, n, n, d_M, lda, d_work, d_pivots, &info));

    CUDA_CHECK(cudaDeviceSynchronize());

    // 檢查 LU 分解是否成功
    if(info != 0){
        std::cerr << "LU decomposition failed with info = " << info << std::endl;
        // 清理內存
        CUDA_CHECK(cudaFree(d_work));
        return;
    }

    // 提取 L 和 U 矩陣
    // L 是下三角，U 是上三角
    // L 的對角元素為 1，U 的對角元素為 LU 矩陣的對角元素

    // 使用 cuBLAS 將 L 和 U 從 LU 矩陣中提取出來
    // 先將 L 初始化為單位矩陣
    CUDA_CHECK(cudaMemset(d_L, 0, n * n * sizeof(float)));
    for(int i = 0; i < n; ++i){
        d_L[i + i * n] = 1.0f; // 對角線設置為1
    }

    // 提取 L
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < i; ++j){
            float value = 0.0f;
            CUBLAS_CHECK(cublasSdot(cublas_handle, 1, d_M + i + j * n, 1, d_M + i + j * n, 1, &value));
            L(i, j) = value;
            // 將值拷貝到 GPU 的 L 矩陣
            CUDA_CHECK(cudaMemcpy(d_L + i + j * n, &value, sizeof(float), cudaMemcpyHostToDevice));
        }
    }

    // 提取 U
    // U 的上三角部分直接從 LU 矩陣中提取
    CUDA_CHECK(cudaMemcpy(d_U, d_M, n * n * sizeof(float), cudaMemcpyDeviceToDevice));

    // 提取對角元素 D
    // D 是 U 的對角線元素
    CUDA_CHECK(cudaMemcpy(d_D, d_U, n * sizeof(float), cudaMemcpyDeviceToDevice)); // 只需拷貝對角線
    // 這裡僅簡化為將 D 設置為 U 的對角元素，實際應用中可能需要進一步處理

    // 構建 P 矩陣
    // P 是置換矩陣
    // 從 pivots 向量中構建 P
    Eigen::MatrixXi h_P(n, n);
    h_P.setIdentity();
    int* h_pivots = new int[n];
    CUDA_CHECK(cudaMemcpy(h_pivots, d_pivots, n * sizeof(int), cudaMemcpyDeviceToHost));

    for(int i = 0; i < n; ++i){
        if(h_pivots[i] != i){
            h_P.row(i).swap(h_P.row(h_pivots[i]));
        }
    }

    // 拷貝 P 到 GPU（可選）
    // 這裡將 P 拷貝到 GPU 的 d_P
    float* d_P_matrix;
    CUDA_CHECK(cudaMalloc(&d_P_matrix, n * n * sizeof(float)));
    Eigen::MatrixXf h_P_float = h_P.cast<float>();
    CUDA_CHECK(cudaMemcpy(d_P_matrix, h_P_float.data(), n * n * sizeof(float), cudaMemcpyHostToDevice));

    // 清理內存
    delete[] h_pivots;
    CUDA_CHECK(cudaFree(d_work));

    // 將 L, D, U 拷貝到 Eigen 矩陣
    // 這裡需要將 GPU 上的數據拷貝到主機
    Eigen::MatrixXf h_L(n, n);
    Eigen::VectorXf h_D(n);
    Eigen::MatrixXf h_U(n, n);

    CUDA_CHECK(cudaMemcpy(&h_L(0,0), d_L, n * n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_D(0), d_D, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_U(0,0), d_U, n * n * sizeof(float), cudaMemcpyDeviceToHost));

    L = h_L;
    D = h_D.asDiagonal();
    U = h_U;
    P = h_P;
}

Eigen::MatrixXf MatrixPLDU::getL() const{
    // 將 L 從 GPU 拷貝到主機
    Eigen::MatrixXf h_L(n, n);
    CUDA_CHECK(cudaMemcpy(&h_L(0,0), d_L, n * n * sizeof(float), cudaMemcpyDeviceToHost));
    return h_L;
}

Eigen::MatrixXf MatrixPLDU::getD() const{
    // 將 D 從 GPU 拷貝到主機
    Eigen::MatrixXf h_D(n, n);
    CUDA_CHECK(cudaMemcpy(&h_D(0,0), d_D, n * sizeof(float), cudaMemcpyDeviceToHost));
    return h_D;
}

Eigen::MatrixXf MatrixPLDU::getU() const{
    // 將 U 從 GPU 拷貝到主機
    Eigen::MatrixXf h_U(n, n);
    CUDA_CHECK(cudaMemcpy(&h_U(0,0), d_U, n * n * sizeof(float), cudaMemcpyDeviceToHost));
    return h_U;
}

Eigen::MatrixXi MatrixPLDU::getP() const{
    // P 已在 compute() 函數中構建並存儲在主機上
    return P;
}

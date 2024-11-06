// ComputationOrchestrationManager.cpp
#include "ComputationOrchestrationManager.h"
#include "MatrixCrossInterpolation.h"
#include "MatrixPLDU.h"
#include <iostream>

// 调度器构造函数
ComputationOrchestrationManager::ComputationOrchestrationManager(MemoryManager& mem_manager, int num_gpus, bool use_rook_pivoting)
    : mem_manager_(mem_manager), dependency_manager_(), use_rook_pivoting_(use_rook_pivoting) {
    for(int i = 0; i < num_gpus; ++i){
        // 假设每个 GPU 的 desired_rank 是固定的，例如 10
        gpu_contexts_.emplace_back(new GPUContext(i, 10));
    }
}

// 调度器析构函数
ComputationOrchestrationManager::~ComputationOrchestrationManager(){
    for(auto ctx : gpu_contexts_){
        delete ctx;
    }
}

// 添加任务到依赖管理器
void ComputationOrchestrationManager::AddTask(const Task& task){
    dependency_manager_.AddTask(task);
}

// 调度和执行任务
void ComputationOrchestrationManager::Run(){
    while(!dependency_manager_.AllTasksCompleted()){
        // 获取可执行的任务
        std::vector<Task> executable_tasks = dependency_manager_.GetExecutableTasks();

        for(const auto& task : executable_tasks){
            // 分配一个 GPU（简单轮询）
            GPUContext* ctx = gpu_contexts_[task.id % gpu_contexts_.size()];

            // 获取子张量的 CUDA 流
            cudaStream_t stream = task.sub_tensor.stream;

            // 根据任务类型执行相应的函数
            if(task.type == TaskType::CI){
                ExecuteCI(task, ctx, stream);
                // 任务完成后更新依赖管理器
                dependency_manager_.TaskCompleted(task.id);
            }
            else if(task.type == TaskType::PRRLU){
                ExecutePRRLU(task, ctx, stream);
                // 任务完成后更新依赖管理器
                dependency_manager_.TaskCompleted(task.id);
            }
        }
    }
}

void ComputationOrchestrationManager::ExecuteCI(const Task& task, GPUContext* ctx, cudaStream_t stream){
    // 使用子张量专用的流
    CUDA_CHECK(cudaSetDevice(ctx->gpu_id));
    CUDA_CHECK(cudaStreamSynchronize(stream)); // 确保数据已经拷贝完成

    // 使用 MatrixCrossInterpolation 进行插值
    MatrixCrossInterpolation mci(task.sub_tensor.d_data, task.sub_tensor.sizes[0], task.sub_tensor.sizes[1], ctx->desired_rank, ctx->cublas_handle, ctx->cusolver_handle, stream);
    
    if(use_rook_pivoting_){
        mci.find_pivots_rook();
    } else {
        mci.find_pivots_full_search();
    }

    mci.construct_interpolation();
    Eigen::MatrixXf A_tilde = mci.get_interpolated_matrix();

    // 将 A_tilde 拷贝回 GPU
    CUDA_CHECK(cudaMemcpyAsync(task.sub_tensor.d_data, A_tilde.data(), task.sub_tensor.total_size * sizeof(float), cudaMemcpyHostToDevice, stream));

    // 更新主机内存中的子张量数据
    Eigen::Map<Eigen::MatrixXf>(task.sub_tensor.h_data, task.sub_tensor.sizes[0], task.sub_tensor.sizes[1]) = A_tilde;

    Eigen::Map<Eigen::MatrixXf> h_data_matrix(task.sub_tensor.h_data, task.sub_tensor.sizes[0], task.sub_tensor.sizes[1]);
    float error = (h_data_matrix - A_tilde).norm();
    std::cout << "Task " << task.id << " CI 计算结果 - 重建误差: " << error << std::endl;
}

void ComputationOrchestrationManager::ExecutePRRLU(const Task& task, GPUContext* ctx, cudaStream_t stream){
    // 使用子张量专用的流
    CUDA_CHECK(cudaSetDevice(ctx->gpu_id));
    CUDA_CHECK(cudaStreamSynchronize(stream)); // 确保 CI 任务已完成

    // 使用 MatrixPLDU 进行 PLDU 分解
    MatrixPLDU pldu(task.sub_tensor.d_data, task.sub_tensor.sizes[0], ctx->cusolver_handle, ctx->cublas_handle);
    
    pldu.compute();
    Eigen::MatrixXf L = pldu.getL();
    Eigen::MatrixXf D = pldu.getD();
    Eigen::MatrixXf U = pldu.getU();
    Eigen::MatrixXi P = pldu.getP();

    // 计算重建误差（Frobenius 范数）
    Eigen::MatrixXf reconstructed = L * D * U;
    Eigen::Map<Eigen::MatrixXf> h_data_matrix(task.sub_tensor.h_data, task.sub_tensor.sizes[0], task.sub_tensor.sizes[1]);
    float error = (h_data_matrix - reconstructed).norm();
    std::cout << "Task " << task.id << " PRRLU 计算结果 - 重建误差: " << error << std::endl;

    // 可选：将 L, D, U 拷贝回 GPU 或进行其他处理
}

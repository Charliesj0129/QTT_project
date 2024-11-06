// main.cpp
#include "MemoryManager.h"
#include "ComputationOrchestrationManager.h"
#include "ResultAssembly.h"
#include "Task.h"
#include "SubTensorManaged.h"
#include "ErrorChecking.h" 
#include "TensorPartitioner.h"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

int main(){
    // 初始化 cuBLAS 和 cuSOLVER
    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    
    cusolverDnHandle_t cusolver_handle;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));

    // 初始化 MemoryManager
    size_t device_block_size = 1024 * sizeof(float); // 每个设备内存块大小（示例）
    size_t device_num_blocks = 100; // 设备内存池中块的数量

    size_t host_block_size = 1024 * sizeof(float); // 每个主机Pinned内存块大小（示例）
    size_t host_num_blocks = 100; // 主机内存池中块的数量

    size_t num_streams = 4; // CUDA流的数量

    MemoryManager mem_manager(device_block_size, device_num_blocks,
                              host_block_size, host_num_blocks,
                              num_streams);

    // 初始化 ComputationOrchestrationManager，假设有2个GPU，选择是否使用 Rook Pivoting
    int num_gpus = 2;
    bool use_rook_pivoting = true; // 根据需要选择
    ComputationOrchestrationManager orch_manager(mem_manager, num_gpus, use_rook_pivoting);

    // 定义全局张量尺寸
    std::vector<size_t> global_sizes = {4, 4}; // 示例全局矩阵大小

    // 定义每个维度的分割数量
    std::vector<size_t> partition_counts = {2, 2}; // 将4x4张量分割成2x2的子张量，每个子张量为2x2

    // 使用 TensorPartitioner 分割张量
    std::vector<SubTensorManaged> sub_tensors = TensorPartitioner::PartitionTensor(global_sizes, partition_counts);

    // 创建任务
    int task_id = 1;
    std::vector<Task> ci_tasks;
    std::vector<Task> prrlu_tasks;

    for(auto& sub_tensor : sub_tensors){
        // 为每个子张量分配一个独立的 CUDA 流
        sub_tensor.stream = mem_manager.GetStream();

        // 分配主机Pinned内存并初始化
        sub_tensor.h_data = static_cast<float*>(mem_manager.AllocateHostMemory(MemoryType::SubTensor));
        for(size_t j = 0; j < sub_tensor.total_size; ++j){
            sub_tensor.h_data[j] = static_cast<float>(rand()) / RAND_MAX; // 随机初始化
        }
        // 分配设备内存
        sub_tensor.d_data = static_cast<float*>(mem_manager.AllocateDeviceMemory(MemoryType::SubTensor));
        // 拷贝数据到设备
        CUDA_CHECK(cudaMemcpyAsync(sub_tensor.d_data, sub_tensor.h_data, sub_tensor.total_size * sizeof(float),
                                   cudaMemcpyHostToDevice, sub_tensor.stream));
        // 不需要同步流，利用异步传输

        // 创建CI任务
        Task ci_task(task_id, TaskType::CI, sub_tensor, {});
        ci_tasks.push_back(ci_task);
        orch_manager.AddTask(ci_task);

        // 创建PRRLU任务，依赖于对应的CI任务
        Task prrlu_task(task_id + 1, TaskType::PRRLU, sub_tensor, {task_id});
        prrlu_tasks.push_back(prrlu_task);
        orch_manager.AddTask(prrlu_task);

        task_id += 2;
    }

    // 运行调度和执行
    orch_manager.Run();

    // 汇总结果
    std::vector<size_t> global_dims = {4, 4}; // 示例全局矩阵大小
    ResultAssembly assembler(global_dims);

    // 添加映射关系（根据任务ID和起始索引）
    for(const auto& task : ci_tasks){
        assembler.AddMapping(task.id, task.sub_tensor.start_indices);
    }
    for(const auto& task : prrlu_tasks){
        assembler.AddMapping(task.id, task.sub_tensor.start_indices);
    }

    assembler.AssembleGlobalSolution(ci_tasks, prrlu_tasks);

    // 验证结果
    assembler.ValidateGlobalSolution();

    // 打印全局解决方案（调试用）
    assembler.PrintGlobalSolution();

    std::cout << "主程序执行完成。" << std::endl;

    // 清理 cuBLAS 和 cuSOLVER
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));

    // MemoryManager 的析构函数会自动释放内存

    return 0;
}

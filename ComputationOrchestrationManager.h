// ComputationOrchestrationManager.h
#ifndef COMPUTATIONORCHESTRATIONMANAGER_H
#define COMPUTATIONORCHESTRATIONMANAGER_H

#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include "MemoryManager.h"
#include "DependencyManager.h"
#include "GPUContext.h"
#include "Task.h"

class ComputationOrchestrationManager {
public:
    ComputationOrchestrationManager(MemoryManager& mem_manager, int num_gpus, bool use_rook_pivoting);
    ~ComputationOrchestrationManager();

    // 添加任务到依赖管理器
    void AddTask(const Task& task);

    // 运行调度和执行
    void Run();

    // 获取依赖管理器的引用
    const DependencyManager& GetDependencyManager() const;

private:
    MemoryManager& mem_manager_;
    DependencyManager dependency_manager_;
    std::vector<GPUContext*> gpu_contexts_;
    bool use_rook_pivoting_;

public:
    std::queue<Task> task_queue_;
    std::mutex queue_mutex_;

    // 执行CI任务
    void ExecuteCI(const Task& task, GPUContext* ctx, cudaStream_t stream);

    // 执行PRRLU任务
    void ExecutePRRLU(const Task& task, GPUContext* ctx, cudaStream_t stream);
};

#endif // COMPUTATIONORCHESTRATIONMANAGER_H

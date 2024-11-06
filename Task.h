// Task.h
#ifndef TASK_H
#define TASK_H

#include <vector>
#include "SubTensorManaged.h"

enum class TaskType {
    CI,
    PRRLU
};

// 任务结构体
struct Task {
    int id; // 任务ID
    TaskType type; // 任务类型
    SubTensorManaged sub_tensor; // 任务相关的子张量
    std::vector<int> dependencies; // 依赖的任务ID

    // 带参数的构造函数
    Task(int task_id, TaskType task_type, const SubTensorManaged& sub, const std::vector<int>& deps)
        : id(task_id), type(task_type), sub_tensor(sub), dependencies(deps) {}

    // 默认构造函数
    Task() = delete;
};

#endif // TASK_H

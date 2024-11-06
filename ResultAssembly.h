// ResultAssembly.h
#ifndef RESULTASSEMBLY_H
#define RESULTASSEMBLY_H

#include <vector>
#include <unordered_map>
#include <mutex>
#include <iostream>
#include "Task.h"
#include "SubTensorManaged.h"

struct GlobalSolution {
    std::vector<float> data; // 全局张量或矩阵数据
    std::vector<size_t> dimensions; // 全局数据的维度

    GlobalSolution(const std::vector<size_t>& dims);
    size_t GetGlobalIndex(const std::vector<size_t>& indices) const;
    void SetData(const std::vector<size_t>& indices, float value);
    float GetData(const std::vector<size_t>& indices) const;
};

struct MappingTable {
    // 子张量ID映射到全局起始索引
    std::unordered_map<int, std::vector<size_t>> sub_to_global_start;

    // 添加映射
    void AddMapping(int sub_id, const std::vector<size_t>& global_start);

    // 获取全局起始索引
    std::vector<size_t> GetGlobalStart(int sub_id) const;
};

// Result Assembly 类
class ResultAssembly {
public:
    ResultAssembly(const std::vector<size_t>& global_dims);

    // 添加映射关系
    void AddMapping(int sub_id, const std::vector<size_t>& global_start);

    // 汇总CI结果（例如，平均值）
    void AggregateCIResults(const std::vector<Task>& ci_tasks);

    // 汇总PRRLU结果
    void AggregatePRRLUResults(const std::vector<Task>& prrlu_tasks);

    // 处理重叠区域（示例：简单平均）
    void HandleOverlaps();

    // 汇总所有结果
    void AssembleGlobalSolution(const std::vector<Task>& ci_tasks, const std::vector<Task>& prrlu_tasks);

    // 验证全局解决方案
    void ValidateGlobalSolution();

    // 打印全局解决方案（调试用）
    void PrintGlobalSolution();

private:
    GlobalSolution global_solution_;
    MappingTable mapping_table_;
    std::mutex mutex_;
};

#endif // RESULTASSEMBLY_H

// DependencyManager.h
#ifndef DEPENDENCYMANAGER_H
#define DEPENDENCYMANAGER_H

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <mutex>
#include "Task.h"

class DependencyManager {
public:
    // 添加任务及其依赖
    void AddTask(const Task& task){
        std::lock_guard<std::mutex> lock(mutex_);
        if (tasks_.find(task.id) == tasks_.end()) {
            tasks_.emplace(task.id, task);  // 插入新的 Task
            in_degree_[task.id] = task.dependencies.size();
            if (task.dependencies.empty()) {
                ready_queue_.push(task.id);
            } else {
                for (auto dep : task.dependencies) {
                    adj_list_[dep].insert(task.id);
                }
            }
        }
    }

    // 获取当前可执行的任务（入度为0）
    std::vector<Task> GetExecutableTasks(){
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<Task> executable;
        while(!ready_queue_.empty()){
            int task_id = ready_queue_.front();
            ready_queue_.pop();
            executable.push_back(tasks_[task_id]);
            processed_tasks_.insert(task_id);
        }
        return executable;
    }

    // 任务完成后更新依赖关系
    void TaskCompleted(int task_id){
        std::lock_guard<std::mutex> lock(mutex_);
        if(adj_list_.find(task_id) != adj_list_.end()){
            for(auto dependent_id : adj_list_[task_id]){
                in_degree_[dependent_id]--;
                if(in_degree_[dependent_id] == 0 && processed_tasks_.find(dependent_id) == processed_tasks_.end()){
                    ready_queue_.push(dependent_id);
                }
            }
        }
    }

    // 检查是否所有任务都已处理
    bool AllTasksCompleted() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return processed_tasks_.size() == tasks_.size();
    }

private:
    std::unordered_map<int, Task> tasks_;
    std::unordered_map<int, int> in_degree_;
    std::unordered_map<int, std::unordered_set<int>> adj_list_;
    std::queue<int> ready_queue_;
    std::unordered_set<int> processed_tasks_;
    mutable std::mutex mutex_;
};

#endif // DEPENDENCYMANAGER_H

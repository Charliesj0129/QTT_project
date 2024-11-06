// TensorPartitioner.h
#ifndef TENSORPARTITIONER_H
#define TENSORPARTITIONER_H

#include <vector>
#include "SubTensorManaged.h"

class TensorPartitioner {
public:
    // 按照每个维度的分割数量分割张量
    static std::vector<SubTensorManaged> PartitionTensor(
        const std::vector<size_t>& global_sizes,
        const std::vector<size_t>& partition_counts
    ) {
        // 确保 partition_counts 与 global_sizes 的维度相同
        if (global_sizes.size() != partition_counts.size()) {
            throw std::invalid_argument("global_sizes 和 partition_counts 的维度必须相同");
        }

        std::vector<SubTensorManaged> sub_tensors;
        PartitionRecursive(global_sizes, partition_counts, 0, {}, sub_tensors);
        return sub_tensors;
    }

private:
    static void PartitionRecursive(
        const std::vector<size_t>& global_sizes,
        const std::vector<size_t>& partition_counts,
        size_t dim,
        std::vector<size_t> current_start,
        std::vector<SubTensorManaged>& sub_tensors
    ) {
        if (dim == global_sizes.size()) {
            // 计算每个维度的子张量尺寸
            std::vector<size_t> sizes(global_sizes.size());
            for (size_t i = 0; i < global_sizes.size(); ++i) {
                size_t partition_size = global_sizes[i] / partition_counts[i];
                size_t remainder = global_sizes[i] % partition_counts[i];
                sizes[i] = partition_size;
                if (current_start[i] / partition_size < remainder) {
                    sizes[i] += 1;
                }
            }
            sub_tensors.emplace_back(current_start, sizes);
            return;
        }

        size_t partition_size = global_sizes[dim] / partition_counts[dim];
        size_t remainder = global_sizes[dim] % partition_counts[dim];
        size_t num_partitions = partition_counts[dim];

        for (size_t i = 0; i < num_partitions; ++i) {
            size_t start = i * partition_size + std::min(i, remainder);
            std::vector<size_t> new_start = current_start;
            new_start.push_back(start);
            PartitionRecursive(global_sizes, partition_counts, dim + 1, new_start, sub_tensors);
        }
    }
};

#endif // TENSORPARTITIONER_H

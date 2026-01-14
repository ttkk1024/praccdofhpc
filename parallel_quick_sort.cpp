#include <iostream>
#include <vector>
#include <algorithm>
#include <thread>
#include <future>
#include <random>
#include <chrono>
#include <memory>
#include <functional>

class ParallelQuickSort {
private:
    static const int SEQUENTIAL_THRESHOLD = 1000; // 串行阈值
    static const int MAX_DEPTH = 4;              // 最大递归深度

    // 插入排序，用于小数组优化
    static void insertion_sort(std::vector<int>& arr, int left, int right) {
        for (int i = left + 1; i <= right; i++) {
            int key = arr[i];
            int j = i - 1;
            while (j >= left && arr[j] > key) {
                arr[j + 1] = arr[j];
                j--;
            }
            arr[j + 1] = key;
        }
    }

    // 三数取中选择pivot
    static int median_of_three(std::vector<int>& arr, int left, int right) {
        int mid = left + (right - left) / 2;

        if (arr[mid] < arr[left]) {
            std::swap(arr[left], arr[mid]);
        }
        if (arr[right] < arr[left]) {
            std::swap(arr[left], arr[right]);
        }
        if (arr[right] < arr[mid]) {
            std::swap(arr[mid], arr[right]);
        }

        return mid;
    }

    // Lomuto分区方案
    static int partition(std::vector<int>& arr, int left, int right) {
        int pivot_index = median_of_three(arr, left, right);
        std::swap(arr[pivot_index], arr[right]);
        int pivot = arr[right];

        int i = left - 1;
        for (int j = left; j < right; j++) {
            if (arr[j] <= pivot) {
                i++;
                std::swap(arr[i], arr[j]);
            }
        }
        std::swap(arr[i + 1], arr[right]);
        return i + 1;
    }

    // 串行快速排序
    static void sequential_quick_sort(std::vector<int>& arr, int left, int right) {
        if (left >= right) return;

        if (right - left + 1 <= SEQUENTIAL_THRESHOLD) {
            insertion_sort(arr, left, right);
            return;
        }

        int pivot = partition(arr, left, right);
        sequential_quick_sort(arr, left, pivot - 1);
        sequential_quick_sort(arr, pivot + 1, right);
    }

    // 递归并行快速排序
    static void parallel_quick_sort_recursive(std::vector<int>& arr, int left, int right, int depth) {
        if (left >= right || depth >= MAX_DEPTH) {
            sequential_quick_sort(arr, left, right);
            return;
        }

        if (right - left + 1 <= SEQUENTIAL_THRESHOLD) {
            sequential_quick_sort(arr, left, right);
            return;
        }

        int pivot = partition(arr, left, right);

        // 并行处理左右两部分
        auto future1 = std::async(std::launch::async, [&]() {
            parallel_quick_sort_recursive(arr, left, pivot - 1, depth + 1);
        });

        auto future2 = std::async(std::launch::async, [&]() {
            parallel_quick_sort_recursive(arr, pivot + 1, right, depth + 1);
        });

        // 等待两个子任务完成
        future1.wait();
        future2.wait();
    }

    // Map-Reduce风格的并行快速排序
    static std::vector<int> map_reduce_quick_sort(std::vector<int>& arr, int num_chunks = -1) {
        if (num_chunks == -1) {
            num_chunks = std::thread::hardware_concurrency();
        }

        if (arr.size() <= SEQUENTIAL_THRESHOLD) {
            std::sort(arr.begin(), arr.end());
            return arr;
        }

        // Map阶段：分割数组并排序每个块
        int chunk_size = arr.size() / num_chunks;
        std::vector<std::vector<int>> chunks;

        for (int i = 0; i < arr.size(); i += chunk_size) {
            int end = std::min(i + chunk_size, (int)arr.size());
            std::vector<int> chunk(arr.begin() + i, arr.begin() + end);
            chunks.push_back(chunk);
        }

        // 并行排序每个chunk
        std::vector<std::future<std::vector<int>>> futures;
        for (auto& chunk : chunks) {
            futures.push_back(std::async(std::launch::async, [&]() {
                sequential_quick_sort(chunk, 0, chunk.size() - 1);
                return chunk;
            }));
        }

        // 收集排序后的chunks
        std::vector<std::vector<int>> sorted_chunks;
        for (auto& future : futures) {
            sorted_chunks.push_back(future.get());
        }

        // Reduce阶段：合并已排序的chunks
        return merge_sorted_chunks(sorted_chunks);
    }

    // 合并多个已排序的数组
    static std::vector<int> merge_sorted_chunks(const std::vector<std::vector<int>>& chunks) {
        std::vector<int> result;
        std::vector<int> indices(chunks.size(), 0);

        while (true) {
            int min_val = INT_MAX;
            int min_chunk_idx = -1;

            // 找到所有chunk中最小的元素
            for (size_t i = 0; i < chunks.size(); i++) {
                if (indices[i] < chunks[i].size() && chunks[i][indices[i]] < min_val) {
                    min_val = chunks[i][indices[i]];
                    min_chunk_idx = i;
                }
            }

            if (min_chunk_idx == -1) break;

            result.push_back(min_val);
            indices[min_chunk_idx]++;
        }

        return result;
    }

public:
    // 并行快速排序主函数
    static void sort(std::vector<int>& arr) {
        if (arr.size() <= SEQUENTIAL_THRESHOLD) {
            std::sort(arr.begin(), arr.end());
            return;
        }
        parallel_quick_sort_recursive(arr, 0, arr.size() - 1, 0);
    }

    // Map-Reduce并行快速排序
    static void map_reduce_sort(std::vector<int>& arr) {
        auto sorted_arr = map_reduce_quick_sort(arr);
        arr = std::move(sorted_arr);
    }

    // 串行快速排序（用于性能对比）
    static void sequential_sort(std::vector<int>& arr) {
        sequential_quick_sort(arr, 0, arr.size() - 1);
    }
};

// 性能测试工具
class PerformanceTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time;

public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    double elapsed_seconds() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000000.0;
    }
};

// 生成随机数据
std::vector<int> generate_random_data(size_t size) {
    std::vector<int> data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 1000000);

    for (size_t i = 0; i < size; i++) {
        data[i] = dis(gen);
    }
    return data;
}

// 验证排序正确性
bool is_sorted(const std::vector<int>& arr) {
    for (size_t i = 1; i < arr.size(); i++) {
        if (arr[i] < arr[i-1]) return false;
    }
    return true;
}

// 性能基准测试
void benchmark_quick_sort_algorithms(size_t array_size = 100000, int num_tests = 3) {
    std::cout << "=== 快速排序性能测试 ===" << std::endl;
    std::cout << "数组大小: " << array_size << std::endl;
    std::cout << "CPU核心数: " << std::thread::hardware_concurrency() << std::endl;
    std::cout << std::endl;

    // 生成测试数据
    std::vector<int> test_data = generate_random_data(array_size);

    // 测试策略
    struct TestStrategy {
        std::string name;
        void (*sort_func)(std::vector<int>&);
    };

    std::vector<TestStrategy> strategies = {
        {"串行快速排序", ParallelQuickSort::sequential_sort},
        {"并行快速排序", ParallelQuickSort::sort},
        {"Map-Reduce并行快速排序", ParallelQuickSort::map_reduce_sort}
    };

    std::vector<double> results;

    for (const auto& strategy : strategies) {
        std::cout << "测试: " << strategy.name << std::endl;

        std::vector<double> times;
        for (int i = 0; i < num_tests; i++) {
            std::vector<int> data_copy = test_data;

            PerformanceTimer timer;
            timer.start();

            strategy.sort_func(data_copy);

            double execution_time = timer.elapsed_seconds();
            times.push_back(execution_time);

            // 验证排序正确性
            if (!is_sorted(data_copy)) {
                std::cout << "  ❌ 排序结果不正确!" << std::endl;
                return;
            }
        }

        double avg_time = 0.0;
        for (double time : times) avg_time += time;
        avg_time /= times.size();

        double min_time = *std::min_element(times.begin(), times.end());
        results.push_back(avg_time);

        std::cout << "  平均时间: " << avg_time << "s" << std::endl;
        std::cout << "  最佳时间: " << min_time << "s" << std::endl;
        std::cout << std::endl;
    }

    // 计算加速比
    std::cout << "=== 加速比分析 ===" << std::endl;
    double sequential_time = results[0];

    for (size_t i = 1; i < results.size(); i++) {
        double speedup = sequential_time / results[i];
        double efficiency = speedup / std::thread::hardware_concurrency();

        std::cout << strategies[i].name << ":" << std::endl;
        std::cout << "  加速比: " << speedup << "x" << std::endl;
        std::cout << "  效率: " << (efficiency * 100) << "%" << std::endl;
        std::cout << std::endl;
    }
}

// 演示函数
void demo_parallel_quick_sort() {
    std::cout << "=== 并行快速排序演示 ===" << std::endl;

    // 小规模数据演示
    std::vector<int> small_data = {64, 34, 25, 12, 22, 11, 90, 5, 77, 30};
    std::cout << "原始数据: ";
    for (int val : small_data) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    ParallelQuickSort::sort(small_data);

    std::cout << "排序结果: ";
    for (int val : small_data) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    bool is_correct = is_sorted(small_data);
    std::cout << "验证正确: " << (is_correct ? "✅" : "❌") << std::endl;
    std::cout << std::endl;

    // Map-Reduce演示
    std::cout << "=== Map-Reduce并行快速排序演示 ===" << std::endl;
    std::vector<int> small_data2 = {64, 34, 25, 12, 22, 11, 90, 5, 77, 30};
    std::cout << "原始数据: ";
    for (int val : small_data2) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    ParallelQuickSort::map_reduce_sort(small_data2);

    std::cout << "排序结果: ";
    for (int val : small_data2) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    is_correct = is_sorted(small_data2);
    std::cout << "验证正确: " << (is_correct ? "✅" : "❌") << std::endl;
    std::cout << std::endl;

    // 性能测试
    benchmark_quick_sort_algorithms();
}

int main() {
    demo_parallel_quick_sort();
    return 0;
}
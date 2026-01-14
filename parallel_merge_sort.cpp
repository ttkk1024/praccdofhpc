#include <iostream>
#include <vector>
#include <algorithm>
#include <thread>
#include <future>
#include <random>
#include <chrono>
#include <cmath>
#include <memory>

class ParallelMergeSort {
private:
    static const int SEQUENTIAL_THRESHOLD = 1000; // 串行阈值
    static const int MAX_DEPTH = 4;             // 最大递归深度

    // 串行归并排序
    static void sequential_merge_sort(std::vector<int>& arr, int left, int right) {
        if (left >= right) return;

        if (right - left + 1 <= SEQUENTIAL_THRESHOLD) {
            std::sort(arr.begin() + left, arr.begin() + right + 1);
            return;
        }

        int mid = left + (right - left) / 2;
        sequential_merge_sort(arr, left, mid);
        sequential_merge_sort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }

    // 归并操作
    static void merge(std::vector<int>& arr, int left, int mid, int right) {
        int n1 = mid - left + 1;
        int n2 = right - mid;

        std::vector<int> left_arr(n1), right_arr(n2);

        // 复制数据到临时数组
        for (int i = 0; i < n1; i++)
            left_arr[i] = arr[left + i];
        for (int j = 0; j < n2; j++)
            right_arr[j] = arr[mid + 1 + j];

        // 归并临时数组
        int i = 0, j = 0, k = left;
        while (i < n1 && j < n2) {
            if (left_arr[i] <= right_arr[j]) {
                arr[k] = left_arr[i];
                i++;
            } else {
                arr[k] = right_arr[j];
                j++;
            }
            k++;
        }

        // 复制剩余元素
        while (i < n1) {
            arr[k] = left_arr[i];
            i++;
            k++;
        }
        while (j < n2) {
            arr[k] = right_arr[j];
            j++;
            k++;
        }
    }

    // 递归并行归并排序
    static void parallel_merge_sort_recursive(std::vector<int>& arr, int left, int right, int depth) {
        if (left >= right || depth >= MAX_DEPTH) {
            sequential_merge_sort(arr, left, right);
            return;
        }

        int mid = left + (right - left) / 2;

        // 并行处理左右两部分
        auto future1 = std::async(std::launch::async, [&]() {
            parallel_merge_sort_recursive(arr, left, mid, depth + 1);
        });

        auto future2 = std::async(std::launch::async, [&]() {
            parallel_merge_sort_recursive(arr, mid + 1, right, depth + 1);
        });

        // 等待两个子任务完成
        future1.wait();
        future2.wait();

        // 合并结果
        merge(arr, left, mid, right);
    }

public:
    // 并行归并排序主函数
    static void sort(std::vector<int>& arr) {
        if (arr.size() <= SEQUENTIAL_THRESHOLD) {
            std::sort(arr.begin(), arr.end());
            return;
        }
        parallel_merge_sort_recursive(arr, 0, arr.size() - 1, 0);
    }

    // 串行归并排序（用于性能对比）
    static void sequential_sort(std::vector<int>& arr) {
        sequential_merge_sort(arr, 0, arr.size() - 1);
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
void benchmark_sorting_algorithms(size_t array_size = 100000, int num_tests = 3) {
    std::cout << "=== 并行归并排序性能测试 ===" << std::endl;
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
        {"串行归并排序", ParallelMergeSort::sequential_sort},
        {"并行归并排序", ParallelMergeSort::sort}
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
void demo_parallel_merge_sort() {
    std::cout << "=== 并行归并排序演示 ===" << std::endl;

    // 小规模数据演示
    std::vector<int> small_data = {64, 34, 25, 12, 22, 11, 90, 5, 77, 30};
    std::cout << "原始数据: ";
    for (int val : small_data) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    ParallelMergeSort::sort(small_data);

    std::cout << "排序结果: ";
    for (int val : small_data) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    bool is_correct = is_sorted(small_data);
    std::cout << "验证正确: " << (is_correct ? "✅" : "❌") << std::endl;
    std::cout << std::endl;

    // 性能测试
    benchmark_sorting_algorithms();
}

int main() {
    demo_parallel_merge_sort();
    return 0;
}
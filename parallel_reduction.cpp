#include <iostream>
#include <vector>
#include <thread>
#include <future>
#include <random>
#include <chrono>
#include <memory>
#include <functional>
#include <algorithm>
#include <numeric>
#include <climits>

class ParallelReduction {
private:
    int num_threads_;

    // 串行归约操作
    template<typename T>
    T sequential_reduction(const std::vector<T>& data, const std::string& operation = "sum") {
        if (data.empty()) {
            if (operation == "sum" || operation == "product") return T(0);
            if (operation == "min") return std::numeric_limits<T>::max();
            if (operation == "max") return std::numeric_limits<T>::min();
        }

        T result = data[0];
        for (size_t i = 1; i < data.size(); i++) {
            if (operation == "sum") result += data[i];
            else if (operation == "product") result *= data[i];
            else if (operation == "min") result = std::min(result, data[i]);
            else if (operation == "max") result = std::max(result, data[i]);
        }

        return result;
    }

    // 树形归约操作
    template<typename T>
    T tree_reduction(const std::vector<T>& data, const std::string& operation = "sum") {
        if (data.empty()) {
            if (operation == "sum" || operation == "product") return T(0);
            if (operation == "min") return std::numeric_limits<T>::max();
            if (operation == "max") return std::numeric_limits<T>::min();
        }

        std::vector<T> current_data = data;

        while (current_data.size() > 1) {
            std::vector<T> next_data;
            next_data.reserve((current_data.size() + 1) / 2);

            // 并行处理相邻元素对
            int chunk_size = std::max(1, static_cast<int>(current_data.size()) / num_threads_);
            std::vector<std::future<T>> futures;

            for (int i = 0; i < static_cast<int>(current_data.size()); i += 2) {
                auto future = std::async(std::launch::async, [&, i]() {
                    if (i + 1 < static_cast<int>(current_data.size())) {
                        if (operation == "sum") return current_data[i] + current_data[i + 1];
                        else if (operation == "product") return current_data[i] * current_data[i + 1];
                        else if (operation == "min") return std::min(current_data[i], current_data[i + 1]);
                        else if (operation == "max") return std::max(current_data[i], current_data[i + 1]);
                    } else {
                        return current_data[i];
                    }
                });
                futures.push_back(std::move(future));
            }

            // 收集结果
            for (auto& future : futures) {
                next_data.push_back(future.get());
            }

            current_data = std::move(next_data);
        }

        return current_data[0];
    }

    // 并行前缀和
    template<typename T>
    std::vector<T> parallel_prefix_sum(const std::vector<T>& data) {
        if (data.empty()) return {};

        std::vector<T> result = data;
        int n = data.size();

        // 并行计算前缀和
        auto process_segment = [&](int start, int end) {
            if (start > 0) {
                for (int i = start + 1; i < end; i++) {
                    result[i] += result[i - 1];
                }
            }
        };

        // 分段处理
        int chunk_size = std::max(1, n / num_threads_);
        std::vector<std::future<void>> futures;

        for (int i = 0; i < n; i += chunk_size) {
            int end = std::min(i + chunk_size, n);
            auto future = std::async(std::launch::async, process_segment, i, end);
            futures.push_back(std::move(future));
        }

        // 等待所有线程完成
        for (auto& future : futures) {
            future.wait();
        }

        // 处理段间依赖
        for (int i = chunk_size; i < n; i += chunk_size) {
            T carry = result[i - 1];
            int end = std::min(i + chunk_size, n);
            for (int j = i; j < end; j++) {
                result[j] += carry;
            }
        }

        return result;
    }

    // 分段归约
    template<typename T>
    T segmented_reduction(const std::vector<T>& data, int segment_size = -1) {
        if (data.empty()) return T(0);

        if (segment_size == -1) {
            segment_size = std::max(1, static_cast<int>(data.size()) / num_threads_);
        }

        // 分段
        std::vector<std::vector<T>> segments;
        for (size_t i = 0; i < data.size(); i += segment_size) {
            int end = std::min(i + segment_size, static_cast<int>(data.size()));
            segments.emplace_back(data.begin() + i, data.begin() + end);
        }

        // 并行处理每个段
        std::vector<std::future<T>> futures;
        for (auto& segment : segments) {
            auto future = std::async(std::launch::async, [segment]() {
                T sum = T(0);
                for (const auto& val : segment) {
                    sum += val;
                }
                return sum;
            });
            futures.push_back(std::move(future));
        }

        // 收集段结果
        std::vector<T> segment_results;
        for (auto& future : futures) {
            segment_results.push_back(future.get());
        }

        // 归约段结果
        return sequential_reduction(segment_results, "sum");
    }

    // 并行直方图
    template<typename T>
    std::vector<int> parallel_histogram(const std::vector<T>& data, int num_bins = 10) {
        if (data.empty()) return std::vector<int>(num_bins, 0);

        T min_val = *std::min_element(data.begin(), data.end());
        T max_val = *std::max_element(data.begin(), data.end());
        T bin_width = (max_val - min_val) / num_bins;
        if (bin_width == T(0)) bin_width = T(1);

        std::vector<int> histogram(num_bins, 0);

        // 分段处理
        int chunk_size = std::max(1, static_cast<int>(data.size()) / num_threads_);
        std::vector<std::future<std::vector<int>>> futures;

        for (int i = 0; i < static_cast<int>(data.size()); i += chunk_size) {
            int end = std::min(i + chunk_size, static_cast<int>(data.size()));
            auto future = std::async(std::launch::async, [&, i, end]() {
                std::vector<int> local_histogram(num_bins, 0);
                for (int j = i; j < end; j++) {
                    int bin_index = static_cast<int>((data[j] - min_val) / bin_width);
                    if (bin_index >= num_bins) bin_index = num_bins - 1;
                    local_histogram[bin_index]++;
                }
                return local_histogram;
            });
            futures.push_back(std::move(future));
        }

        // 合并直方图
        for (auto& future : futures) {
            auto local_hist = future.get();
            for (int i = 0; i < num_bins; i++) {
                histogram[i] += local_hist[i];
            }
        }

        return histogram;
    }

    // 并行查找最大值
    template<typename T>
    T parallel_search_max(const std::vector<T>& data) {
        if (data.empty()) return std::numeric_limits<T>::min();

        // 分段处理
        int chunk_size = std::max(1, static_cast<int>(data.size()) / num_threads_);
        std::vector<std::future<T>> futures;

        for (int i = 0; i < static_cast<int>(data.size()); i += chunk_size) {
            int end = std::min(i + chunk_size, static_cast<int>(data.size()));
            auto future = std::async(std::launch::async, [&, i, end]() {
                return *std::max_element(data.begin() + i, data.begin() + end);
            });
            futures.push_back(std::move(future));
        }

        // 收集结果
        std::vector<T> segment_maxs;
        for (auto& future : futures) {
            segment_maxs.push_back(future.get());
        }

        // 找出全局最大值
        return *std::max_element(segment_maxs.begin(), segment_maxs.end());
    }

    // 并行查找最小值
    template<typename T>
    T parallel_search_min(const std::vector<T>& data) {
        if (data.empty()) return std::numeric_limits<T>::max();

        // 分段处理
        int chunk_size = std::max(1, static_cast<int>(data.size()) / num_threads_);
        std::vector<std::future<T>> futures;

        for (int i = 0; i < static_cast<int>(data.size()); i += chunk_size) {
            int end = std::min(i + chunk_size, static_cast<int>(data.size()));
            auto future = std::async(std::launch::async, [&, i, end]() {
                return *std::min_element(data.begin() + i, data.begin() + end);
            });
            futures.push_back(std::move(future));
        }

        // 收集结果
        std::vector<T> segment_mins;
        for (auto& future : futures) {
            segment_mins.push_back(future.get());
        }

        // 找出全局最小值
        return *std::min_element(segment_mins.begin(), segment_mins.end());
    }

    // 并行向量点积
    template<typename T>
    T parallel_dot_product(const std::vector<T>& vector1, const std::vector<T>& vector2) {
        if (vector1.size() != vector2.size()) {
            throw std::invalid_argument("向量长度不匹配");
        }
        if (vector1.empty()) return T(0);

        // 分段处理
        int chunk_size = std::max(1, static_cast<int>(vector1.size()) / num_threads_);
        std::vector<std::future<T>> futures;

        for (int i = 0; i < static_cast<int>(vector1.size()); i += chunk_size) {
            int end = std::min(i + chunk_size, static_cast<int>(vector1.size()));
            auto future = std::async(std::launch::async, [&, i, end]() {
                T sum = T(0);
                for (int j = i; j < end; j++) {
                    sum += vector1[j] * vector2[j];
                }
                return sum;
            });
            futures.push_back(std::move(future));
        }

        // 收集部分和
        T total = T(0);
        for (auto& future : futures) {
            total += future.get();
        }

        return total;
    }

public:
    ParallelReduction(int num_threads = -1) {
        num_threads_ = (num_threads == -1) ? std::thread::hardware_concurrency() : num_threads;
    }

    // 主要的归约函数
    template<typename T>
    T reduce(const std::vector<T>& data, const std::string& method = "sequential", const std::string& operation = "sum") {
        if (method == "sequential") {
            return sequential_reduction(data, operation);
        } else if (method == "tree") {
            return tree_reduction(data, operation);
        } else if (method == "segmented") {
            return segmented_reduction(data);
        } else {
            throw std::invalid_argument("不支持的方法: " + method);
        }
    }

    // 并行前缀和
    template<typename T>
    std::vector<T> prefix_sum(const std::vector<T>& data) {
        return parallel_prefix_sum(data);
    }

    // 并行直方图
    template<typename T>
    std::vector<int> histogram(const std::vector<T>& data, int num_bins = 10) {
        return parallel_histogram(data, num_bins);
    }

    // 并行查找最值
    template<typename T>
    T find_max(const std::vector<T>& data) {
        return parallel_search_max(data);
    }

    template<typename T>
    T find_min(const std::vector<T>& data) {
        return parallel_search_min(data);
    }

    // 并行向量点积
    template<typename T>
    T dot_product(const std::vector<T>& vector1, const std::vector<T>& vector2) {
        return parallel_dot_product(vector1, vector2);
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

// 性能基准测试
void benchmark_reduction_operations(int data_size = 1000000, int num_tests = 3) {
    std::cout << "=== 归约操作性能测试 ===" << std::endl;
    std::cout << "数据大小: " << data_size << std::endl;
    std::cout << "CPU核心数: " << std::thread::hardware_concurrency() << std::endl;
    std::cout << std::endl;

    // 生成测试数据
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 1000);
    std::uniform_real_distribution<> dis_float(0.0, 1.0);

    std::vector<int> data(data_size);
    std::vector<double> vector1(data_size), vector2(data_size);

    for (int i = 0; i < data_size; i++) {
        data[i] = dis(gen);
        vector1[i] = dis_float(gen);
        vector2[i] = dis_float(gen);
    }

    struct TestStrategy {
        std::string name;
        std::function<double()> operation;
    };

    std::vector<TestStrategy> strategies = {
        {"串行求和", [&]() {
            ParallelReduction reducer;
            return reducer.reduce(data, "sequential", "sum");
        }},
        {"树形归约求和", [&]() {
            ParallelReduction reducer;
            return reducer.reduce(data, "tree", "sum");
        }},
        {"分段归约求和", [&]() {
            ParallelReduction reducer;
            return reducer.reduce(data, "segmented", "sum");
        }},
        {"并行前缀和", [&]() {
            ParallelReduction reducer;
            auto result = reducer.prefix_sum(data);
            return static_cast<double>(result.size());
        }},
        {"并行查找最大值", [&]() {
            ParallelReduction reducer;
            return static_cast<double>(reducer.find_max(data));
        }},
        {"并行查找最小值", [&]() {
            ParallelReduction reducer;
            return static_cast<double>(reducer.find_min(data));
        }},
        {"并行向量点积", [&]() {
            ParallelReduction reducer;
            return reducer.dot_product(vector1, vector2);
        }},
        {"并行直方图", [&]() {
            ParallelReduction reducer;
            auto result = reducer.histogram(data, 10);
            return static_cast<double>(result.size());
        }}
    };

    std::vector<double> results;

    for (const auto& strategy : strategies) {
        std::cout << "测试: " << strategy.name << std::endl;

        std::vector<double> times;
        for (int i = 0; i < num_tests; i++) {
            PerformanceTimer timer;
            timer.start();

            double result = strategy.operation();

            double execution_time = timer.elapsed_seconds();
            times.push_back(execution_time);
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
void demo_parallel_reduction() {
    std::cout << "=== 并行归约操作演示 ===" << std::endl;

    // 小规模数据演示
    std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::cout << "原始数据: ";
    for (int val : data) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    ParallelReduction reducer;

    // 串行归约
    int sequential_sum = reducer.reduce(data, "sequential", "sum");
    std::cout << "串行求和: " << sequential_sum << std::endl;

    // 树形归约
    int tree_sum = reducer.reduce(data, "tree", "sum");
    std::cout << "树形归约求和: " << tree_sum << std::endl;

    // 并行前缀和
    auto prefix_sum = reducer.prefix_sum(data);
    std::cout << "并行前缀和: ";
    for (int val : prefix_sum) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // 并行查找最值
    int max_val = reducer.find_max(data);
    int min_val = reducer.find_min(data);
    std::cout << "并行查找最大值: " << max_val << std::endl;
    std::cout << "并行查找最小值: " << min_val << std::endl;

    // 并行向量点积
    std::vector<double> vector1 = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> vector2 = {5.0, 6.0, 7.0, 8.0};
    double dot_product = reducer.dot_product(vector1, vector2);
    std::cout << "向量1: ";
    for (double val : vector1) std::cout << val << " ";
    std::cout << std::endl;
    std::cout << "向量2: ";
    for (double val : vector2) std::cout << val << " ";
    std::cout << std::endl;
    std::cout << "并行向量点积: " << dot_product << std::endl;

    // 并行直方图
    std::vector<int> histogram_data = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4};
    auto histogram = reducer.histogram(histogram_data, 4);
    std::cout << "直方图数据: ";
    for (int val : histogram_data) std::cout << val << " ";
    std::cout << std::endl;
    std::cout << "并行直方图: ";
    for (int count : histogram) std::cout << count << " ";
    std::cout << std::endl;

    std::cout << std::endl;

    // 性能测试
    benchmark_reduction_operations();
}

int main() {
    demo_parallel_reduction();
    return 0;
}
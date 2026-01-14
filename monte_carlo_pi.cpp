#include <iostream>
#include <vector>
#include <random>
#include <thread>
#include <future>
#include <chrono>
#include <memory>
#include <functional>
#include <cmath>
#include <algorithm>
#include <numeric>

class MonteCarloPi {
private:
    int num_threads_;

    // 串行蒙特卡洛π值计算
    double sequential_monte_carlo_pi(long long num_samples) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(-1.0, 1.0);

        long long inside_circle = 0;

        for (long long i = 0; i < num_samples; i++) {
            double x = dis(gen);
            double y = dis(gen);
            if (x*x + y*y <= 1.0) {
                inside_circle++;
            }
        }

        return 4.0 * inside_circle / num_samples;
    }

    // 并行蒙特卡洛π值计算
    double parallel_monte_carlo_pi(long long num_samples) {
        long long samples_per_thread = num_samples / num_threads_;
        long long remainder = num_samples % num_threads_;

        std::vector<std::future<long long>> futures;

        for (int t = 0; t < num_threads_; t++) {
            long long samples = samples_per_thread;
            if (t == num_threads_ - 1) {
                samples += remainder; // 最后一个线程处理余数
            }

            auto future = std::async(std::launch::async, [samples, t]() {
                // 为每个线程设置不同的随机种子
                std::random_device rd;
                std::mt19937 gen(rd() + t * 1000);
                std::uniform_real_distribution<double> dis(-1.0, 1.0);

                long long inside = 0;
                for (long long i = 0; i < samples; i++) {
                    double x = dis(gen);
                    double y = dis(gen);
                    if (x*x + y*y <= 1.0) {
                        inside++;
                    }
                }
                return inside;
            });

            futures.push_back(std::move(future));
        }

        // 收集结果
        long long total_inside = 0;
        for (auto& future : futures) {
            total_inside += future.get();
        }

        return 4.0 * total_inside / num_samples;
    }

    // 使用SIMD的向量化蒙特卡洛π值计算
    double vectorized_monte_carlo_pi(long long num_samples) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(-1.0, 1.0);

        long long inside_circle = 0;
        const int vector_size = 1000; // 批量处理大小

        for (long long i = 0; i < num_samples; i += vector_size) {
            int current_batch = std::min(static_cast<long long>(vector_size), num_samples - i);
            std::vector<double> x(current_batch);
            std::vector<double> y(current_batch);

            // 生成批量随机数
            for (int j = 0; j < current_batch; j++) {
                x[j] = dis(gen);
                y[j] = dis(gen);
            }

            // 向量化计算
            for (int j = 0; j < current_batch; j++) {
                if (x[j]*x[j] + y[j]*y[j] <= 1.0) {
                    inside_circle++;
                }
            }
        }

        return 4.0 * inside_circle / num_samples;
    }

    // 并行向量化蒙特卡洛π值计算
    double parallel_vectorized_monte_carlo_pi(long long num_samples) {
        long long samples_per_thread = num_samples / num_threads_;
        long long remainder = num_samples % num_threads_;

        std::vector<std::future<long long>> futures;

        for (int t = 0; t < num_threads_; t++) {
            long long samples = samples_per_thread;
            if (t == num_threads_ - 1) {
                samples += remainder;
            }

            auto future = std::async(std::launch::async, [samples, t]() {
                std::random_device rd;
                std::mt19937 gen(rd() + t * 1000);
                std::uniform_real_distribution<double> dis(-1.0, 1.0);

                long long inside = 0;
                const int vector_size = 1000;

                for (long long i = 0; i < samples; i += vector_size) {
                    int current_batch = std::min(static_cast<long long>(vector_size), samples - i);
                    std::vector<double> x(current_batch);
                    std::vector<double> y(current_batch);

                    // 生成批量随机数
                    for (int j = 0; j < current_batch; j++) {
                        x[j] = dis(gen);
                        y[j] = dis(gen);
                    }

                    // 向量化计算
                    for (int j = 0; j < current_batch; j++) {
                        if (x[j]*x[j] + y[j]*y[j] <= 1.0) {
                            inside++;
                        }
                    }
                }
                return inside;
            });

            futures.push_back(std::move(future));
        }

        // 收集结果
        long long total_inside = 0;
        for (auto& future : futures) {
            total_inside += future.get();
        }

        return 4.0 * total_inside / num_samples;
    }

    // 带置信区间的π值估计
    std::pair<double, double> estimate_pi_with_confidence(long long num_samples, double confidence_level = 0.95) {
        double pi_estimate = parallel_monte_carlo_pi(num_samples);

        // 计算标准误差
        double p = pi_estimate / 4.0; // 圆内点的比例
        double variance = p * (1 - p);
        double standard_error = std::sqrt(variance / num_samples);

        // 计算置信区间
        double z_score = 1.96; // 95%置信水平
        if (confidence_level == 0.99) {
            z_score = 2.576;
        } else if (confidence_level == 0.90) {
            z_score = 1.645;
        }

        double margin_of_error = z_score * standard_error * 4; // 乘以4转换为π的误差

        return {pi_estimate, margin_of_error};
    }

    // 自适应蒙特卡洛π值计算
    std::tuple<double, double, long long> adaptive_monte_carlo_pi(double target_error = 0.001, long long max_samples = 10000000) {
        long long samples = 1000;
        double pi_estimate = 0;
        double error = std::numeric_limits<double>::infinity();

        while (error > target_error && samples < max_samples) {
            // 增加样本数量
            long long additional_samples = std::min(10000LL, max_samples - samples);
            samples += additional_samples;

            // 计算新的π值
            double new_pi = parallel_monte_carlo_pi(additional_samples);
            pi_estimate = (pi_estimate * (samples - additional_samples) + new_pi * additional_samples) / samples;

            // 估算误差
            double p = pi_estimate / 4.0;
            double variance = p * (1 - p);
            double standard_error = std::sqrt(variance / samples);
            error = 1.96 * standard_error * 4;
        }

        return {pi_estimate, error, samples};
    }

public:
    MonteCarloPi(int num_threads = -1) {
        num_threads_ = (num_threads == -1) ? std::thread::hardware_concurrency() : num_threads;
    }

    // 主要的π值计算函数
    double calculate_pi(long long num_samples, const std::string& method = "sequential") {
        if (method == "sequential") {
            return sequential_monte_carlo_pi(num_samples);
        } else if (method == "parallel") {
            return parallel_monte_carlo_pi(num_samples);
        } else if (method == "vectorized") {
            return vectorized_monte_carlo_pi(num_samples);
        } else if (method == "parallel_vectorized") {
            return parallel_vectorized_monte_carlo_pi(num_samples);
        } else {
            throw std::invalid_argument("不支持的方法: " + method);
        }
    }

    // 获取置信区间
    std::pair<double, double> get_confidence_interval(long long num_samples, double confidence_level = 0.95) {
        return estimate_pi_with_confidence(num_samples, confidence_level);
    }

    // 自适应计算
    std::tuple<double, double, long long> adaptive_calculate_pi(double target_error = 0.001, long long max_samples = 10000000) {
        return adaptive_monte_carlo_pi(target_error, max_samples);
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
void benchmark_monte_carlo_pi(long long num_samples = 10000000, int num_tests = 3) {
    std::cout << "=== 蒙特卡洛π值计算性能测试 ===" << std::endl;
    std::cout << "样本数量: " << num_samples << std::endl;
    std::cout << "CPU核心数: " << std::thread::hardware_concurrency() << std::endl;
    std::cout << std::endl;

    std::vector<std::pair<std::string, std::string>> strategies = {
        {"串行蒙特卡洛", "sequential"},
        {"并行蒙特卡洛", "parallel"},
        {"向量化蒙特卡洛", "vectorized"},
        {"并行向量化蒙特卡洛", "parallel_vectorized"}
    };

    std::vector<double> results;

    for (const auto& strategy : strategies) {
        std::cout << "测试: " << strategy.first << std::endl;

        std::vector<double> times;
        std::vector<double> pi_values;

        for (int i = 0; i < num_tests; i++) {
            PerformanceTimer timer;
            timer.start();

            MonteCarloPi pi_calculator;
            double pi_value = pi_calculator.calculate_pi(num_samples, strategy.second);

            double execution_time = timer.elapsed_seconds();
            times.push_back(execution_time);
            pi_values.push_back(pi_value);
        }

        double avg_time = 0.0;
        for (double time : times) avg_time += time;
        avg_time /= times.size();

        double min_time = *std::min_element(times.begin(), times.end());
        double avg_pi = std::accumulate(pi_values.begin(), pi_values.end(), 0.0) / pi_values.size();
        double error = std::abs(avg_pi - M_PI);

        results.push_back(avg_time);

        std::cout << "  平均时间: " << avg_time << "s" << std::endl;
        std::cout << "  最佳时间: " << min_time << "s" << std::endl;
        std::cout << "  π估计值: " << avg_pi << std::endl;
        std::cout << "  误差: " << error << std::endl;
        std::cout << std::endl;
    }

    // 计算加速比
    std::cout << "=== 加速比分析 ===" << std::endl;
    double sequential_time = results[0];

    for (size_t i = 1; i < results.size(); i++) {
        double speedup = sequential_time / results[i];
        double efficiency = speedup / std::thread::hardware_concurrency();

        std::cout << strategies[i].first << ":" << std::endl;
        std::cout << "  加速比: " << speedup << "x" << std::endl;
        std::cout << "  效率: " << (efficiency * 100) << "%" << std::endl;
        std::cout << std::endl;
    }
}

// 演示函数
void demo_monte_carlo_pi() {
    std::cout << "=== 蒙特卡洛π值计算演示 ===" << std::endl;

    // 小规模演示
    long long num_samples = 1000000;
    std::cout << "样本数量: " << num_samples << std::endl;

    // 串行计算
    MonteCarloPi pi_calculator;
    double pi_sequential = pi_calculator.calculate_pi(num_samples, "sequential");
    std::cout << "串行蒙特卡洛π值: " << pi_sequential << std::endl;

    // 并行计算
    double pi_parallel = pi_calculator.calculate_pi(num_samples, "parallel");
    std::cout << "并行蒙特卡洛π值: " << pi_parallel << std::endl;

    // 向量化计算
    double pi_vectorized = pi_calculator.calculate_pi(num_samples, "vectorized");
    std::cout << "向量化蒙特卡洛π值: " << pi_vectorized << std::endl;

    // 真实π值
    std::cout << "真实π值: " << M_PI << std::endl;

    // 误差比较
    std::cout << std::endl << "误差比较:" << std::endl;
    std::cout << "  串行误差: " << std::abs(pi_sequential - M_PI) << std::endl;
    std::cout << "  并行误差: " << std::abs(pi_parallel - M_PI) << std::endl;
    std::cout << "  向量化误差: " << std::abs(pi_vectorized - M_PI) << std::endl;
    std::cout << std::endl;

    // 置信区间演示
    std::cout << "=== 置信区间演示 ===" << std::endl;
    auto [pi_estimate, margin_of_error] = pi_calculator.get_confidence_interval(1000000, 0.95);
    std::cout << "π估计值: " << pi_estimate << std::endl;
    std::cout << "95%置信区间: [" << (pi_estimate - margin_of_error) << ", " << (pi_estimate + margin_of_error) << "]" << std::endl;
    std::cout << std::endl;

    // 自适应计算演示
    std::cout << "=== 自适应蒙特卡洛演示 ===" << std::endl;
    double target_error = 0.001;
    auto [pi_adaptive, actual_error, samples_used] = pi_calculator.adaptive_calculate_pi(target_error);
    std::cout << "目标误差: " << target_error << std::endl;
    std::cout << "实际误差: " << actual_error << std::endl;
    std::cout << "使用样本数: " << samples_used << std::endl;
    std::cout << "π估计值: " << pi_adaptive << std::endl;
    std::cout << std::endl;

    // 性能测试
    benchmark_monte_carlo_pi();
}

int main() {
    demo_monte_carlo_pi();
    return 0;
}
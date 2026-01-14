#include <iostream>
#include <vector>
#include <algorithm>
#include <thread>
#include <future>
#include <random>
#include <chrono>
#include <memory>
#include <functional>
#include <cmath>

class ParallelMatrixMultiplication {
private:
    static const int SEQUENTIAL_THRESHOLD = 64; // 串行阈值
    static const int BLOCK_SIZE = 64;          // 分块大小

    // 串行矩阵乘法
    static std::vector<std::vector<int>> sequential_multiply(
        const std::vector<std::vector<int>>& A,
        const std::vector<std::vector<int>>& B) {

        int rows_A = A.size();
        int cols_A = A[0].size();
        int rows_B = B.size();
        int cols_B = B[0].size();

        if (cols_A != rows_B) {
            throw std::invalid_argument("矩阵维度不匹配");
        }

        std::vector<std::vector<int>> C(rows_A, std::vector<int>(cols_B, 0));

        for (int i = 0; i < rows_A; i++) {
            for (int j = 0; j < cols_B; j++) {
                for (int k = 0; k < cols_A; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }

        return C;
    }

    // 分块矩阵乘法
    static std::vector<std::vector<int>> block_multiply(
        const std::vector<std::vector<int>>& A,
        const std::vector<std::vector<int>>& B,
        int block_size = BLOCK_SIZE) {

        int rows_A = A.size();
        int cols_A = A[0].size();
        int rows_B = B.size();
        int cols_B = B[0].size();

        if (cols_A != rows_B) {
            throw std::invalid_argument("矩阵维度不匹配");
        }

        std::vector<std::vector<int>> C(rows_A, std::vector<int>(cols_B, 0));

        // 分块计算
        for (int i = 0; i < rows_A; i += block_size) {
            for (int j = 0; j < cols_B; j += block_size) {
                for (int k = 0; k < cols_A; k += block_size) {
                    // 计算当前块
                    int i_end = std::min(i + block_size, rows_A);
                    int j_end = std::min(j + block_size, cols_B);
                    int k_end = std::min(k + block_size, cols_A);

                    for (int ii = i; ii < i_end; ii++) {
                        for (int jj = j; jj < j_end; jj++) {
                            for (int kk = k; kk < k_end; kk++) {
                                C[ii][jj] += A[ii][kk] * B[kk][jj];
                            }
                        }
                    }
                }
            }
        }

        return C;
    }

    // 矩阵加法
    static std::vector<std::vector<int>> add_matrix(
        const std::vector<std::vector<int>>& A,
        const std::vector<std::vector<int>>& B) {

        int n = A.size();
        std::vector<std::vector<int>> result(n, std::vector<int>(n));

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                result[i][j] = A[i][j] + B[i][j];
            }
        }

        return result;
    }

    // 矩阵减法
    static std::vector<std::vector<int>> subtract_matrix(
        const std::vector<std::vector<int>>& A,
        const std::vector<std::vector<int>>& B) {

        int n = A.size();
        std::vector<std::vector<int>> result(n, std::vector<int>(n));

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                result[i][j] = A[i][j] - B[i][j];
            }
        }

        return result;
    }

    // 分割矩阵为四个子矩阵
    static void split_matrix(const std::vector<std::vector<int>>& matrix,
                           std::vector<std::vector<int>>& A11,
                           std::vector<std::vector<int>>& A12,
                           std::vector<std::vector<int>>& A21,
                           std::vector<std::vector<int>>& A22,
                           int mid) {

        int n = matrix.size();
        A11.resize(mid, std::vector<int>(mid));
        A12.resize(mid, std::vector<int>(n - mid));
        A21.resize(n - mid, std::vector<int>(mid));
        A22.resize(n - mid, std::vector<int>(n - mid));

        for (int i = 0; i < mid; i++) {
            for (int j = 0; j < mid; j++) {
                A11[i][j] = matrix[i][j];
            }
            for (int j = mid; j < n; j++) {
                A12[i][j - mid] = matrix[i][j];
            }
        }

        for (int i = mid; i < n; i++) {
            for (int j = 0; j < mid; j++) {
                A21[i - mid][j] = matrix[i][j];
            }
            for (int j = mid; j < n; j++) {
                A22[i - mid][j - mid] = matrix[i][j];
            }
        }
    }

    // 合并四个子矩阵
    static std::vector<std::vector<int>> combine_matrices(
        const std::vector<std::vector<int>>& C11,
        const std::vector<std::vector<int>>& C12,
        const std::vector<std::vector<int>>& C21,
        const std::vector<std::vector<int>>& C22) {

        int mid = C11.size();
        int n = mid * 2;
        std::vector<std::vector<int>> result(n, std::vector<int>(n));

        for (int i = 0; i < mid; i++) {
            for (int j = 0; j < mid; j++) {
                result[i][j] = C11[i][j];
                result[i][j + mid] = C12[i][j];
                result[i + mid][j] = C21[i][j];
                result[i + mid][j + mid] = C22[i][j];
            }
        }

        return result;
    }

    // 并行Strassen算法
    static std::vector<std::vector<int>> strassen_multiply(
        const std::vector<std::vector<int>>& A,
        const std::vector<std::vector<int>>& B) {

        int n = A.size();

        // 基础情况：小矩阵使用串行乘法
        if (n <= SEQUENTIAL_THRESHOLD) {
            return sequential_multiply(A, B);
        }

        // 确保矩阵大小是2的幂（填充零）
        int new_n = 1;
        while (new_n < n) {
            new_n <<= 1;
        }

        if (new_n != n) {
            auto A_padded = pad_matrix(A, new_n);
            auto B_padded = pad_matrix(B, new_n);
            auto result = strassen_multiply(A_padded, B_padded);
            return unpad_matrix(result, n);
        }

        // 分割矩阵
        int mid = n / 2;
        std::vector<std::vector<int>> A11, A12, A21, A22;
        std::vector<std::vector<int>> B11, B12, B21, B22;

        split_matrix(A, A11, A12, A21, A22, mid);
        split_matrix(B, B11, B12, B21, B22, mid);

        // 并行计算7个乘积
        auto P1_future = std::async(std::launch::async, [&]() {
            auto temp = subtract_matrix(B12, B22);
            return strassen_multiply(A11, temp);
        });

        auto P2_future = std::async(std::launch::async, [&]() {
            auto temp = add_matrix(A11, A12);
            return strassen_multiply(temp, B22);
        });

        auto P3_future = std::async(std::launch::async, [&]() {
            auto temp = add_matrix(A21, A22);
            return strassen_multiply(temp, B11);
        });

        auto P4_future = std::async(std::launch::async, [&]() {
            auto temp = subtract_matrix(B21, B11);
            return strassen_multiply(A22, temp);
        });

        auto P5_future = std::async(std::launch::async, [&]() {
            auto temp1 = add_matrix(A11, A22);
            auto temp2 = add_matrix(B11, B22);
            return strassen_multiply(temp1, temp2);
        });

        auto P6_future = std::async(std::launch::async, [&]() {
            auto temp1 = subtract_matrix(A12, A22);
            auto temp2 = add_matrix(B21, B22);
            return strassen_multiply(temp1, temp2);
        });

        auto P7_future = std::async(std::launch::async, [&]() {
            auto temp1 = subtract_matrix(A11, A21);
            auto temp2 = add_matrix(B11, B12);
            return strassen_multiply(temp1, temp2);
        });

        auto P1 = P1_future.get();
        auto P2 = P2_future.get();
        auto P3 = P3_future.get();
        auto P4 = P4_future.get();
        auto P5 = P5_future.get();
        auto P6 = P6_future.get();
        auto P7 = P7_future.get();

        // 计算结果矩阵的四个部分
        auto C11 = add_matrix(subtract_matrix(add_matrix(P5, P4), P2), P6);
        auto C12 = add_matrix(P1, P2);
        auto C21 = add_matrix(P3, P4);
        auto C22 = subtract_matrix(subtract_matrix(add_matrix(P5, P1), P3), P7);

        // 合并结果
        return combine_matrices(C11, C12, C21, C22);
    }

    // 填充矩阵到指定大小
    static std::vector<std::vector<int>> pad_matrix(const std::vector<std::vector<int>>& matrix, int new_n) {
        int n = matrix.size();
        std::vector<std::vector<int>> padded(new_n, std::vector<int>(new_n, 0));

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                padded[i][j] = matrix[i][j];
            }
        }

        return padded;
    }

    // 从填充矩阵中提取原始部分
    static std::vector<std::vector<int>> unpad_matrix(const std::vector<std::vector<int>>& matrix, int original_n) {
        std::vector<std::vector<int>> result(original_n, std::vector<int>(original_n));

        for (int i = 0; i < original_n; i++) {
            for (int j = 0; j < original_n; j++) {
                result[i][j] = matrix[i][j];
            }
        }

        return result;
    }

public:
    // 并行分块矩阵乘法
    static std::vector<std::vector<int>> parallel_block_multiply(
        const std::vector<std::vector<int>>& A,
        const std::vector<std::vector<int>>& B) {

        int rows_A = A.size();
        int cols_A = A[0].size();
        int rows_B = B.size();
        int cols_B = B[0].size();

        if (cols_A != rows_B) {
            throw std::invalid_argument("矩阵维度不匹配");
        }

        std::vector<std::vector<int>> C(rows_A, std::vector<int>(cols_B, 0));

        // 将结果矩阵的行分配给不同线程
        auto compute_rows = [&](int start_row, int end_row) {
            std::vector<std::vector<int>> local_C(end_row - start_row, std::vector<int>(cols_B, 0));
            for (int i = start_row; i < end_row; i++) {
                for (int j = 0; j < cols_B; j++) {
                    for (int k = 0; k < cols_A; k++) {
                        local_C[i - start_row][j] += A[i][k] * B[k][j];
                    }
                }
            }
            return std::make_pair(start_row, local_C);
        };

        // 使用线程池并行计算
        int num_threads = std::thread::hardware_concurrency();
        int rows_per_thread = rows_A / num_threads;

        std::vector<std::future<std::pair<int, std::vector<std::vector<int>>>>> futures;

        for (int t = 0; t < num_threads; t++) {
            int start_row = t * rows_per_thread;
            int end_row = (t == num_threads - 1) ? rows_A : start_row + rows_per_thread;

            auto future = std::async(std::launch::async, compute_rows, start_row, end_row);
            futures.push_back(std::move(future));
        }

        // 收集结果
        for (auto& future : futures) {
            auto result = future.get();
            int start_row = result.first;
            auto& local_C = result.second;

            for (size_t i = 0; i < local_C.size(); i++) {
                C[start_row + i] = std::move(local_C[i]);
            }
        }

        return C;
    }

    // 主要的矩阵乘法函数
    static std::vector<std::vector<int>> multiply(
        const std::vector<std::vector<int>>& A,
        const std::vector<std::vector<int>>& B,
        const std::string& method = "sequential") {

        if (method == "sequential") {
            return sequential_multiply(A, B);
        } else if (method == "block") {
            return block_multiply(A, B);
        } else if (method == "parallel_block") {
            return parallel_block_multiply(A, B);
        } else if (method == "strassen") {
            return strassen_multiply(A, B);
        } else {
            throw std::invalid_argument("不支持的方法: " + method);
        }
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

// 生成随机矩阵
std::vector<std::vector<int>> generate_random_matrix(int n, int max_val = 100) {
    std::vector<std::vector<int>> matrix(n, std::vector<int>(n));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, max_val);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i][j] = dis(gen);
        }
    }
    return matrix;
}

// 验证矩阵乘法正确性
bool verify_matrix_multiply(const std::vector<std::vector<int>>& A,
                          const std::vector<std::vector<int>>& B,
                          const std::vector<std::vector<int>>& C) {
    int rows_A = A.size();
    int cols_A = A[0].size();
    int cols_B = B[0].size();

    std::vector<std::vector<int>> expected(rows_A, std::vector<int>(cols_B, 0));

    for (int i = 0; i < rows_A; i++) {
        for (int j = 0; j < cols_B; j++) {
            for (int k = 0; k < cols_A; k++) {
                expected[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    for (int i = 0; i < rows_A; i++) {
        for (int j = 0; j < cols_B; j++) {
            if (C[i][j] != expected[i][j]) {
                return false;
            }
        }
    }
    return true;
}

// 性能基准测试
void benchmark_matrix_multiplication(int n = 256, int num_tests = 3) {
    std::cout << "=== 矩阵乘法性能测试 ===" << std::endl;
    std::cout << "矩阵大小: " << n << "x" << n << std::endl;
    std::cout << "CPU核心数: " << std::thread::hardware_concurrency() << std::endl;
    std::cout << std::endl;

    // 生成测试数据
    auto A = generate_random_matrix(n);
    auto B = generate_random_matrix(n);

    std::vector<std::pair<std::string, std::string>> strategies = {
        {"串行矩阵乘法", "sequential"},
        {"分块矩阵乘法", "block"},
        {"并行分块矩阵乘法", "parallel_block"}
    };

    // 仅对较小矩阵测试Strassen算法
    if (n <= 128) {
        strategies.push_back({"并行Strassen算法", "strassen"});
    }

    std::vector<double> results;

    for (const auto& strategy : strategies) {
        std::cout << "测试: " << strategy.first << std::endl;

        std::vector<double> times;
        for (int i = 0; i < num_tests; i++) {
            PerformanceTimer timer;
            timer.start();

            auto result = ParallelMatrixMultiplication::multiply(A, B, strategy.second);

            double execution_time = timer.elapsed_seconds();
            times.push_back(execution_time);

            // 验证正确性（小矩阵）
            if (n <= 64 && !verify_matrix_multiply(A, B, result)) {
                std::cout << "  ❌ 计算结果不正确!" << std::endl;
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

        std::cout << strategies[i].first << ":" << std::endl;
        std::cout << "  加速比: " << speedup << "x" << std::endl;
        std::cout << "  效率: " << (efficiency * 100) << "%" << std::endl;
        std::cout << std::endl;
    }
}

// 演示函数
void demo_parallel_matrix_multiplication() {
    std::cout << "=== 并行矩阵乘法演示 ===" << std::endl;

    // 小规模数据演示
    int n = 4;
    std::vector<std::vector<int>> A = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 16}
    };

    std::vector<std::vector<int>> B = {
        {1, 0, 1, 0},
        {0, 1, 0, 1},
        {1, 1, 0, 0},
        {0, 0, 1, 1}
    };

    std::cout << "矩阵A:" << std::endl;
    for (const auto& row : A) {
        for (int val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\n矩阵B:" << std::endl;
    for (const auto& row : B) {
        for (int val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    // 串行计算
    auto result = ParallelMatrixMultiplication::multiply(A, B, "sequential");

    std::cout << "\n串行计算结果:" << std::endl;
    for (const auto& row : result) {
        for (int val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    // 验证正确性
    if (verify_matrix_multiply(A, B, result)) {
        std::cout << "✅ 计算结果正确" << std::endl;
    } else {
        std::cout << "❌ 计算结果错误" << std::endl;
    }

    std::cout << std::endl;

    // 性能测试
    benchmark_matrix_multiplication();
}

int main() {
    demo_parallel_matrix_multiplication();
    return 0;
}
#!/usr/bin/env python3
"""
并行矩阵乘法算法实现
基于第三章的数据分解和并行数值计算
"""

import time
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

class ParallelMatrixMultiplication:
    """并行矩阵乘法类"""

    def __init__(self, strategy='threadpool', block_size=64):
        self.strategy = strategy
        self.block_size = block_size

    def sequential_matrix_multiply(self, A, B):
        """串行矩阵乘法实现"""
        rows_A, cols_A = len(A), len(A[0])
        rows_B, cols_B = len(B), len(B[0])

        if cols_A != rows_B:
            raise ValueError("矩阵维度不匹配")

        # 结果矩阵初始化
        C = [[0 for _ in range(cols_B)] for _ in range(rows_A)]

        # 标准三重循环
        for i in range(rows_A):
            for j in range(cols_B):
                for k in range(cols_A):
                    C[i][j] += A[i][k] * B[k][j]

        return C

    def block_matrix_multiply(self, A, B, block_size=None):
        """分块矩阵乘法实现"""
        if block_size is None:
            block_size = self.block_size

        rows_A, cols_A = len(A), len(A[0])
        rows_B, cols_B = len(B), len(B[0])

        if cols_A != rows_B:
            raise ValueError("矩阵维度不匹配")

        # 结果矩阵初始化
        C = [[0 for _ in range(cols_B)] for _ in range(rows_A)]

        # 分块计算
        for i in range(0, rows_A, block_size):
            for j in range(0, cols_B, block_size):
                for k in range(0, cols_A, block_size):
                    # 计算当前块
                    for ii in range(i, min(i + block_size, rows_A)):
                        for jj in range(j, min(j + block_size, cols_B)):
                            for kk in range(k, min(k + block_size, cols_A)):
                                C[ii][jj] += A[ii][kk] * B[kk][jj]

        return C

    def parallel_block_multiply(self, A, B):
        """并行分块矩阵乘法"""
        rows_A, cols_A = len(A), len(A[0])
        rows_B, cols_B = len(B), len(B[0])

        if cols_A != rows_B:
            raise ValueError("矩阵维度不匹配")

        # 结果矩阵初始化
        C = [[0 for _ in range(cols_B)] for _ in range(rows_A)]

        # 将结果矩阵的行分配给不同线程
        def compute_rows(start_row, end_row):
            local_C = [[0 for _ in range(cols_B)] for _ in range(end_row - start_row)]
            for i in range(start_row, end_row):
                for j in range(cols_B):
                    for k in range(cols_A):
                        local_C[i - start_row][j] += A[i][k] * B[k][j]
            return start_row, end_row, local_C

        # 使用线程池并行计算
        num_threads = mp.cpu_count()
        rows_per_thread = rows_A // num_threads

        futures = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for t in range(num_threads):
                start_row = t * rows_per_thread
                end_row = start_row + rows_per_thread if t < num_threads - 1 else rows_A

                future = executor.submit(compute_rows, start_row, end_row)
                futures.append(future)

            # 收集结果
            for future in futures:
                start_row, end_row, local_C = future.result()
                for i in range(start_row, end_row):
                    C[i] = local_C[i - start_row]

        return C

    def parallel_strassen_multiply(self, A, B):
        """并行Strassen算法矩阵乘法"""
        n = len(A)

        # 基础情况：小矩阵使用串行乘法
        if n <= 128:
            return self.sequential_matrix_multiply(A, B)

        # 确保矩阵大小是2的幂（填充零）
        if n & (n - 1) != 0:  # 不是2的幂
            new_n = 1
            while new_n < n:
                new_n <<= 1
            A_padded = self._pad_matrix(A, new_n)
            B_padded = self._pad_matrix(B, new_n)
            result = self.parallel_strassen_multiply(A_padded, B_padded)
            return self._unpad_matrix(result, n)

        # 分割矩阵
        mid = n // 2
        A11, A12, A21, A22 = self._split_matrix(A, mid)
        B11, B12, B21, B22 = self._split_matrix(B, mid)

        # 并行计算7个乘积
        def compute_p1():
            return self.parallel_strassen_multiply(A11, self._subtract_matrix(B12, B22))

        def compute_p2():
            return self.parallel_strassen_multiply(self._add_matrix(A11, A12), B22)

        def compute_p3():
            return self.parallel_strassen_multiply(self._add_matrix(A21, A22), B11)

        def compute_p4():
            return self.parallel_strassen_multiply(A22, self._subtract_matrix(B21, B11))

        def compute_p5():
            return self.parallel_strassen_multiply(self._add_matrix(A11, A22), self._add_matrix(B11, B22))

        def compute_p6():
            return self.parallel_strassen_multiply(self._subtract_matrix(A12, A22), self._add_matrix(B21, B22))

        def compute_p7():
            return self.parallel_strassen_multiply(self._subtract_matrix(A11, A21), self._add_matrix(B11, B12))

        with ThreadPoolExecutor(max_workers=7) as executor:
            p1_future = executor.submit(compute_p1)
            p2_future = executor.submit(compute_p2)
            p3_future = executor.submit(compute_p3)
            p4_future = executor.submit(compute_p4)
            p5_future = executor.submit(compute_p5)
            p6_future = executor.submit(compute_p6)
            p7_future = executor.submit(compute_p7)

            P1 = p1_future.result()
            P2 = p2_future.result()
            P3 = p3_future.result()
            P4 = p4_future.result()
            P5 = p5_future.result()
            P6 = p6_future.result()
            P7 = p7_future.result()

        # 计算结果矩阵的四个部分
        C11 = self._add_matrix(self._subtract_matrix(self._add_matrix(P5, P4), P2), P6)
        C12 = self._add_matrix(P1, P2)
        C21 = self._add_matrix(P3, P4)
        C22 = self._subtract_matrix(self._subtract_matrix(self._add_matrix(P5, P1), P3), P7)

        # 合并结果
        return self._combine_matrices(C11, C12, C21, C22)

    def _split_matrix(self, matrix, mid):
        """分割矩阵为四个子矩阵"""
        n = len(matrix)
        A11 = [row[:mid] for row in matrix[:mid]]
        A12 = [row[mid:] for row in matrix[:mid]]
        A21 = [row[:mid] for row in matrix[mid:]]
        A22 = [row[mid:] for row in matrix[mid:]]
        return A11, A12, A21, A22

    def _combine_matrices(self, A11, A12, A21, A22):
        """合并四个子矩阵"""
        n = len(A11) * 2
        result = [[0 for _ in range(n)] for _ in range(n)]

        for i in range(len(A11)):
            for j in range(len(A11)):
                result[i][j] = A11[i][j]
                result[i][j + len(A11)] = A12[i][j]
                result[i + len(A11)][j] = A21[i][j]
                result[i + len(A11)][j + len(A11)] = A22[i][j]

        return result

    def _add_matrix(self, A, B):
        """矩阵加法"""
        n = len(A)
        return [[A[i][j] + B[i][j] for j in range(n)] for i in range(n)]

    def _subtract_matrix(self, A, B):
        """矩阵减法"""
        n = len(A)
        return [[A[i][j] - B[i][j] for j in range(n)] for i in range(n)]

    def _pad_matrix(self, matrix, new_n):
        """将矩阵填充到指定大小"""
        n = len(matrix)
        padded = [[0 for _ in range(new_n)] for _ in range(new_n)]
        for i in range(n):
            for j in range(n):
                padded[i][j] = matrix[i][j]
        return padded

    def _unpad_matrix(self, matrix, original_n):
        """从填充矩阵中提取原始部分"""
        return [row[:original_n] for row in matrix[:original_n]]

    def sort(self, A, B):
        """根据策略选择矩阵乘法方法"""
        if self.strategy == 'sequential':
            return self.sequential_matrix_multiply(A, B)
        elif self.strategy == 'block':
            return self.block_matrix_multiply(A, B)
        elif self.strategy == 'parallel_block':
            return self.parallel_block_multiply(A, B)
        elif self.strategy == 'strassen':
            return self.parallel_strassen_multiply(A, B)
        else:
            raise ValueError(f"Unsupported strategy: {self.strategy}")

def generate_random_matrix(n, max_val=100):
    """生成随机矩阵"""
    return [[random.randint(1, max_val) for _ in range(n)] for _ in range(n)]

def compare_matrix_multiplication_methods(n=512, num_tests=3):
    """比较不同矩阵乘法方法的性能"""
    print(f"=== 矩阵乘法性能测试 ===")
    print(f"矩阵大小: {n}x{n}")
    print(f"CPU核心数: {mp.cpu_count()}")
    print()

    # 生成测试数据
    A = generate_random_matrix(n)
    B = generate_random_matrix(n)

    strategies = [
        ('串行矩阵乘法', 'sequential'),
        ('分块矩阵乘法', 'block'),
        ('并行分块矩阵乘法', 'parallel_block'),
    ]

    # 仅对较小矩阵测试Strassen算法
    if n <= 256:
        strategies.append(('并行Strassen算法', 'strassen'))

    results = {}

    for name, strategy in strategies:
        print(f"测试: {name}")
        times = []

        for i in range(num_tests):
            start_time = time.time()
            sorter = ParallelMatrixMultiplication(strategy=strategy)
            result = sorter.sort(A, B)
            end_time = time.time()

            execution_time = end_time - start_time
            times.append(execution_time)

            # 验证正确性（小矩阵）
            if n <= 128:
                expected = np.dot(np.array(A), np.array(B))
                if not np.allclose(np.array(result), expected):
                    print(f"  ❌ 计算结果不正确!")
                    return

        avg_time = sum(times) / len(times)
        min_time = min(times)
        results[strategy] = avg_time

        print(f"  平均时间: {avg_time:.4f}s")
        print(f"  最佳时间: {min_time:.4f}s")
        print()

    # 计算加速比
    print("=== 加速比分析 ===")
    sequential_time = results['sequential']

    for name, strategy in strategies:
        if strategy != 'sequential':
            speedup = sequential_time / results[strategy]
            efficiency = speedup / mp.cpu_count()
            print(f"{name}:")
            print(f"  加速比: {speedup:.2f}x")
            print(f"  效率: {efficiency:.2%}")
            print()

def demo_parallel_matrix_multiplication():
    """演示并行矩阵乘法"""
    print("=== 并行矩阵乘法演示 ===")

    # 小规模数据演示
    n = 4
    A = [[1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 10, 11, 12],
         [13, 14, 15, 16]]

    B = [[1, 0, 1, 0],
         [0, 1, 0, 1],
         [1, 1, 0, 0],
         [0, 0, 1, 1]]

    print("矩阵A:")
    for row in A:
        print(row)

    print("\n矩阵B:")
    for row in B:
        print(row)

    # 串行计算
    sorter = ParallelMatrixMultiplication(strategy='sequential')
    result = sorter.sort(A, B)

    print("\n串行计算结果:")
    for row in result:
        print(row)

    # 验证正确性
    expected = np.dot(np.array(A), np.array(B))
    if np.allclose(np.array(result), expected):
        print("✅ 计算结果正确")
    else:
        print("❌ 计算结果错误")

    print()

    # 性能测试
    compare_matrix_multiplication_methods()

if __name__ == "__main__":
    demo_parallel_matrix_multiplication()
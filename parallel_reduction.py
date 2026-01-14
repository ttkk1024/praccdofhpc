#!/usr/bin/env python3
"""
并行归约操作实现
基于第三章的并行算法设计原则和常见并行算法模式
"""

import time
import random
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import numpy as np

class ParallelReduction:
    """并行归约操作类"""

    def __init__(self, strategy='tree', num_threads=None):
        self.strategy = strategy
        self.num_threads = num_threads or mp.cpu_count()

    def sequential_reduction(self, data, operation='sum'):
        """串行归约操作"""
        if not data:
            return 0 if operation in ['sum', 'product'] else float('inf') if operation == 'min' else float('-inf')

        result = data[0]
        for i in range(1, len(data)):
            if operation == 'sum':
                result += data[i]
            elif operation == 'product':
                result *= data[i]
            elif operation == 'min':
                result = min(result, data[i])
            elif operation == 'max':
                result = max(result, data[i])

        return result

    def tree_reduction(self, data, operation='sum'):
        """树形归约操作"""
        if not data:
            return 0 if operation in ['sum', 'product'] else float('inf') if operation == 'min' else float('-inf')

        # 复制数据以避免修改原始数据
        current_data = data.copy()

        while len(current_data) > 1:
            next_data = []
            # 并行处理相邻元素对
            def process_pair(i):
                if i + 1 < len(current_data):
                    if operation == 'sum':
                        return current_data[i] + current_data[i + 1]
                    elif operation == 'product':
                        return current_data[i] * current_data[i + 1]
                    elif operation == 'min':
                        return min(current_data[i], current_data[i + 1])
                    elif operation == 'max':
                        return max(current_data[i], current_data[i + 1])
                else:
                    return current_data[i]

            # 使用线程池并行处理
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = []
                for i in range(0, len(current_data), 2):
                    future = executor.submit(process_pair, i)
                    futures.append(future)

                next_data = [future.result() for future in futures]

            current_data = next_data

        return current_data[0] if current_data else 0

    def parallel_prefix_sum(self, data):
        """并行前缀和（扫描操作）"""
        if not data:
            return []

        n = len(data)
        result = data.copy()

        # 并行计算前缀和
        def process_segment(start, end):
            if start == 0:
                return
            # 计算前缀和
            for i in range(start + 1, end):
                result[i] += result[i - 1]

        # 分段处理
        chunk_size = max(1, n // self.num_threads)
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            for i in range(0, n, chunk_size):
                end = min(i + chunk_size, n)
                future = executor.submit(process_segment, i, end)
                futures.append(future)

            for future in futures:
                future.result()

        # 处理段间依赖
        for i in range(chunk_size, n, chunk_size):
            carry = result[i - 1]
            for j in range(i, min(i + chunk_size, n)):
                result[j] += carry

        return result

    def segmented_reduction(self, data, segment_size=None):
        """分段归约操作"""
        if segment_size is None:
            segment_size = max(1, len(data) // self.num_threads)

        # 计算每个段的归约结果
        def reduce_segment(segment):
            if not segment:
                return 0
            result = segment[0]
            for i in range(1, len(segment)):
                result += segment[i]
            return result

        # 分段
        segments = [data[i:i + segment_size] for i in range(0, len(data), segment_size)]

        # 并行处理每个段
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            segment_results = list(executor.map(reduce_segment, segments))

        # 归约段结果
        return self.sequential_reduction(segment_results, 'sum')

    def parallel_histogram(self, data, num_bins=10):
        """并行直方图计算"""
        if not data:
            return [0] * num_bins

        min_val = min(data)
        max_val = max(data)
        bin_width = (max_val - min_val) / num_bins if max_val != min_val else 1

        # 初始化直方图
        histogram = [0] * num_bins

        # 分段处理数据
        chunk_size = max(1, len(data) // self.num_threads)
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

        # 并行计算每个段的直方图
        def process_chunk(chunk):
            local_histogram = [0] * num_bins
            for value in chunk:
                bin_index = int((value - min_val) / bin_width)
                if bin_index >= num_bins:
                    bin_index = num_bins - 1
                local_histogram[bin_index] += 1
            return local_histogram

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            chunk_histograms = list(executor.map(process_chunk, chunks))

        # 合并直方图
        for i in range(num_bins):
            histogram[i] = sum(h[i] for h in chunk_histograms)

        return histogram

    def parallel_search_max(self, data):
        """并行查找最大值"""
        if not data:
            return float('-inf')

        # 分段处理
        chunk_size = max(1, len(data) // self.num_threads)
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

        # 并行查找每个段的最大值
        def find_max(chunk):
            return max(chunk) if chunk else float('-inf')

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            segment_maxs = list(executor.map(find_max, chunks))

        # 找出全局最大值
        return max(segment_maxs)

    def parallel_search_min(self, data):
        """并行查找最小值"""
        if not data:
            return float('inf')

        # 分段处理
        chunk_size = max(1, len(data) // self.num_threads)
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

        # 并行查找每个段的最小值
        def find_min(chunk):
            return min(chunk) if chunk else float('inf')

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            segment_mins = list(executor.map(find_min, chunks))

        # 找出全局最小值
        return min(segment_mins)

    def parallel_dot_product(self, vector1, vector2):
        """并行向量点积"""
        if len(vector1) != len(vector2):
            raise ValueError("向量长度不匹配")

        if not vector1:
            return 0

        # 分段处理
        chunk_size = max(1, len(vector1) // self.num_threads)
        chunks = [(vector1[i:i + chunk_size], vector2[i:i + chunk_size])
                 for i in range(0, len(vector1), chunk_size)]

        # 并行计算每个段的点积
        def compute_dot_product(chunk):
            v1, v2 = chunk
            return sum(a * b for a, b in zip(v1, v2))

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            partial_sums = list(executor.map(compute_dot_product, chunks))

        # 归约部分和
        return sum(partial_sums)

def benchmark_reduction_operations(data_size=1000000, num_tests=3):
    """归约操作性能基准测试"""
    print(f"=== 归约操作性能测试 ===")
    print(f"数据大小: {data_size:,}")
    print(f"CPU核心数: {mp.cpu_count()}")
    print()

    # 生成测试数据
    data = [random.randint(1, 1000) for _ in range(data_size)]
    vector1 = [random.random() for _ in range(data_size)]
    vector2 = [random.random() for _ in range(data_size)]

    strategies = [
        ('串行求和', lambda: ParallelReduction().sequential_reduction(data, 'sum')),
        ('树形归约求和', lambda: ParallelReduction().tree_reduction(data, 'sum')),
        ('分段归约求和', lambda: ParallelReduction().segmented_reduction(data)),
        ('并行前缀和', lambda: ParallelReduction().parallel_prefix_sum(data)),
        ('并行查找最大值', lambda: ParallelReduction().parallel_search_max(data)),
        ('并行查找最小值', lambda: ParallelReduction().parallel_search_min(data)),
        ('并行向量点积', lambda: ParallelReduction().parallel_dot_product(vector1, vector2)),
        ('并行直方图', lambda: ParallelReduction().parallel_histogram(data, 10)),
    ]

    results = {}

    for name, operation in strategies:
        print(f"测试: {name}")
        times = []

        for i in range(num_tests):
            start_time = time.time()
            result = operation()
            end_time = time.time()

            execution_time = end_time - start_time
            times.append(execution_time)

        avg_time = sum(times) / len(times)
        min_time = min(times)
        results[name] = {
            'time': avg_time,
            'result': result if name != '并行前缀和' else len(result)
        }

        print(f"  平均时间: {avg_time:.4f}s")
        print(f"  最佳时间: {min_time:.4f}s")
        if name != '并行前缀和':
            print(f"  结果: {result}")
        else:
            print(f"  结果长度: {len(result)}")
        print()

    # 计算加速比（基于串行求和）
    print("=== 加速比分析 ===")
    sequential_time = results['串行求和']['time']

    for name, data in results.items():
        if name != '串行求和':
            speedup = sequential_time / data['time']
            efficiency = speedup / mp.cpu_count()
            print(f"{name}:")
            print(f"  加速比: {speedup:.2f}x")
            print(f"  效率: {efficiency:.2%}")
            print()

def demo_parallel_reduction():
    """演示并行归约操作"""
    print("=== 并行归约操作演示 ===")

    # 小规模数据演示
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    print(f"原始数据: {data}")

    reducer = ParallelReduction()

    # 串行归约
    sequential_sum = reducer.sequential_reduction(data, 'sum')
    print(f"串行求和: {sequential_sum}")

    # 树形归约
    tree_sum = reducer.tree_reduction(data, 'sum')
    print(f"树形归约求和: {tree_sum}")

    # 并行前缀和
    prefix_sum = reducer.parallel_prefix_sum(data)
    print(f"并行前缀和: {prefix_sum}")

    # 并行查找最值
    max_val = reducer.parallel_search_max(data)
    min_val = reducer.parallel_search_min(data)
    print(f"并行查找最大值: {max_val}")
    print(f"并行查找最小值: {min_val}")

    # 并行向量点积
    vector1 = [1, 2, 3, 4]
    vector2 = [5, 6, 7, 8]
    dot_product = reducer.parallel_dot_product(vector1, vector2)
    print(f"向量1: {vector1}")
    print(f"向量2: {vector2}")
    print(f"并行向量点积: {dot_product}")

    # 并行直方图
    histogram_data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
    histogram = reducer.parallel_histogram(histogram_data, 4)
    print(f"直方图数据: {histogram_data}")
    print(f"并行直方图: {histogram}")

    print()

    # 性能测试
    benchmark_reduction_operations()

if __name__ == "__main__":
    demo_parallel_reduction()
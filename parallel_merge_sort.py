#!/usr/bin/env python3
"""
并行归并排序算法实现
基于第三章的分治法并行算法设计
"""

import time
import random
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

def sequential_merge_sort(arr):
    """串行归并排序实现"""
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = sequential_merge_sort(arr[:mid])
    right = sequential_merge_sort(arr[mid:])

    return merge(left, right)

def merge(left, right):
    """归并两个已排序的数组"""
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result

def parallel_merge_sort_threadpool(arr, max_workers=None, threshold=1000):
    """使用线程池的并行归并排序"""
    if len(arr) <= threshold:
        return sequential_merge_sort(arr)

    if max_workers is None:
        max_workers = mp.cpu_count()

    def _parallel_merge_sort(arr, depth=0, max_depth=4):
        if len(arr) <= threshold or depth >= max_depth:
            return sequential_merge_sort(arr)

        mid = len(arr) // 2
        left_part = arr[:mid]
        right_part = arr[mid:]

        with ThreadPoolExecutor(max_workers=2) as executor:
            left_future = executor.submit(_parallel_merge_sort, left_part, depth + 1, max_depth)
            right_future = executor.submit(_parallel_merge_sort, right_part, depth + 1, max_depth)

            left_sorted = left_future.result()
            right_sorted = right_future.result()

        return merge(left_sorted, right_sorted)

    return _parallel_merge_sort(arr)

def parallel_merge_sort_processpool(arr, threshold=1000):
    """使用进程池的并行归并排序"""
    if len(arr) <= threshold:
        return sequential_merge_sort(arr)

    def _merge_sort_worker(arr_chunk):
        """进程池工作函数"""
        return sequential_merge_sort(arr_chunk)

    def _parallel_merge_sort(arr, depth=0, max_depth=3):
        if len(arr) <= threshold or depth >= max_depth:
            return sequential_merge_sort(arr)

        mid = len(arr) // 2
        left_part = arr[:mid]
        right_part = arr[mid:]

        with ProcessPoolExecutor(max_workers=2) as executor:
            left_future = executor.submit(_parallel_merge_sort, left_part, depth + 1, max_depth)
            right_future = executor.submit(_parallel_merge_sort, right_part, depth + 1, max_depth)

            left_sorted = left_future.result()
            right_sorted = right_future.result()

        return merge(left_sorted, right_sorted)

    return _parallel_merge_sort(arr)

class ParallelMergeSort:
    """并行归并排序类，支持多种并行策略"""

    def __init__(self, strategy='threadpool', threshold=1000):
        self.strategy = strategy
        self.threshold = threshold
        self.cpu_count = mp.cpu_count()

    def sort(self, arr):
        """根据策略选择排序方法"""
        if self.strategy == 'sequential':
            return sequential_merge_sort(arr)
        elif self.strategy == 'threadpool':
            return parallel_merge_sort_threadpool(arr, threshold=self.threshold)
        elif self.strategy == 'processpool':
            return parallel_merge_sort_processpool(arr, threshold=self.threshold)
        else:
            raise ValueError(f"Unsupported strategy: {self.strategy}")

def benchmark_sorting_algorithms(arr_size=100000, num_tests=3):
    """性能基准测试"""
    print(f"=== 并行归并排序性能测试 ===")
    print(f"数组大小: {arr_size:,}")
    print(f"CPU核心数: {mp.cpu_count()}")
    print()

    # 生成测试数据
    test_data = [random.randint(1, 1000000) for _ in range(arr_size)]

    strategies = [
        ('串行归并排序', 'sequential', None),
        ('线程池并行排序', 'threadpool', None),
        ('进程池并行排序', 'processpool', None),
    ]

    results = {}

    for name, strategy, workers in strategies:
        print(f"测试: {name}")
        times = []

        for i in range(num_tests):
            data_copy = test_data.copy()

            start_time = time.time()
            sorter = ParallelMergeSort(strategy=strategy)
            sorted_data = sorter.sort(data_copy)
            end_time = time.time()

            execution_time = end_time - start_time
            times.append(execution_time)

            # 验证排序正确性
            is_sorted = all(sorted_data[i] <= sorted_data[i+1] for i in range(len(sorted_data)-1))
            if not is_sorted:
                print(f"  ❌ 排序结果不正确!")
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

    for name, strategy, _ in strategies:
        if strategy != 'sequential':
            speedup = sequential_time / results[strategy]
            efficiency = speedup / mp.cpu_count()
            print(f"{name}:")
            print(f"  加速比: {speedup:.2f}x")
            print(f"  效率: {efficiency:.2%}")
            print()

def demo_parallel_merge_sort():
    """演示并行归并排序"""
    print("=== 并行归并排序演示 ===")

    # 小规模数据演示
    small_data = [64, 34, 25, 12, 22, 11, 90, 5, 77, 30]
    print(f"原始数据: {small_data}")

    sorter = ParallelMergeSort(strategy='threadpool')
    sorted_data = sorter.sort(small_data.copy())

    print(f"排序结果: {sorted_data}")
    print(f"验证正确: {'✅' if sorted(small_data) == sorted_data else '❌'}")
    print()

    # 性能测试
    benchmark_sorting_algorithms()

if __name__ == "__main__":
    demo_parallel_merge_sort()
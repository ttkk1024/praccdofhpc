#!/usr/bin/env python3
"""
并行快速排序算法实现
基于第三章的分治法并行算法设计
"""

import time
import random
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import threading

class ParallelQuickSort:
    """并行快速排序类"""

    def __init__(self, strategy='threadpool', threshold=1000, max_workers=None):
        self.strategy = strategy
        self.threshold = threshold
        self.max_workers = max_workers or mp.cpu_count()
        self.lock = threading.Lock()

    def sequential_quick_sort(self, arr, low, high):
        """串行快速排序实现"""
        if low < high:
            if high - low + 1 <= self.threshold:
                # 小数组使用插入排序
                self._insertion_sort(arr, low, high)
                return

            # 三数取中选择pivot
            pivot_index = self._median_of_three(arr, low, high)
            arr[pivot_index], arr[high] = arr[high], arr[pivot_index]

            pivot = self._partition(arr, low, high)
            self.sequential_quick_sort(arr, low, pivot - 1)
            self.sequential_quick_sort(arr, pivot + 1, high)

    def _insertion_sort(self, arr, low, high):
        """插入排序，用于小数组优化"""
        for i in range(low + 1, high + 1):
            key = arr[i]
            j = i - 1
            while j >= low and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key

    def _median_of_three(self, arr, low, high):
        """三数取中选择pivot"""
        mid = (low + high) // 2
        if arr[mid] < arr[low]:
            arr[low], arr[mid] = arr[mid], arr[low]
        if arr[high] < arr[low]:
            arr[low], arr[high] = arr[high], arr[low]
        if arr[high] < arr[mid]:
            arr[mid], arr[high] = arr[high], arr[mid]
        return mid

    def _partition(self, arr, low, high):
        """Lomuto分区方案"""
        pivot = arr[high]
        i = low - 1

        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]

        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1

    def parallel_quick_sort_threadpool(self, arr, low, high, depth=0, max_depth=4):
        """使用线程池的并行快速排序"""
        if low >= high or depth >= max_depth:
            self.sequential_quick_sort(arr, low, high)
            return

        if high - low + 1 <= self.threshold:
            self.sequential_quick_sort(arr, low, high)
            return

        # 三数取中选择pivot
        pivot_index = self._median_of_three(arr, low, high)
        arr[pivot_index], arr[high] = arr[high], arr[pivot_index]

        pivot = self._partition(arr, low, high)

        # 并行处理左右子数组
        with ThreadPoolExecutor(max_workers=2) as executor:
            left_future = executor.submit(
                self.parallel_quick_sort_threadpool,
                arr, low, pivot - 1, depth + 1, max_depth
            )
            right_future = executor.submit(
                self.parallel_quick_sort_threadpool,
                arr, pivot + 1, high, depth + 1, max_depth
            )

            left_future.result()
            right_future.result()

    def parallel_quick_sort_processpool(self, arr):
        """使用进程池的并行快速排序"""
        if len(arr) <= self.threshold:
            self.sequential_quick_sort(arr, 0, len(arr) - 1)
            return arr

        def _quick_sort_worker(data_chunk):
            """进程池工作函数"""
            if len(data_chunk) <= self.threshold:
                data_chunk.sort()
                return data_chunk

            pivot = data_chunk[len(data_chunk) // 2]
            less = [x for x in data_chunk if x < pivot]
            equal = [x for x in data_chunk if x == pivot]
            greater = [x for x in data_chunk if x > pivot]

            with ProcessPoolExecutor(max_workers=2) as executor:
                less_future = executor.submit(_quick_sort_worker, less)
                greater_future = executor.submit(_quick_sort_worker, greater)

                less_sorted = less_future.result()
                greater_sorted = greater_future.result()

            return less_sorted + equal + greater_sorted

        return _quick_sort_worker(arr)

    def sort(self, arr):
        """根据策略选择排序方法"""
        if self.strategy == 'sequential':
            self.sequential_quick_sort(arr, 0, len(arr) - 1)
        elif self.strategy == 'threadpool':
            self.parallel_quick_sort_threadpool(arr, 0, len(arr) - 1)
        elif self.strategy == 'processpool':
            sorted_arr = self.parallel_quick_sort_processpool(arr.copy())
            arr[:] = sorted_arr
        else:
            raise ValueError(f"Unsupported strategy: {self.strategy}")

def parallel_quicksort_task(arr_chunk):
    """并行任务函数，用于进程池"""
    if len(arr_chunk) <= 1000:
        arr_chunk.sort()
        return arr_chunk

    pivot = arr_chunk[len(arr_chunk) // 2]
    less = [x for x in arr_chunk if x < pivot]
    equal = [x for x in arr_chunk if x == pivot]
    greater = [x for x in arr_chunk if x > pivot]

    less.sort()
    greater.sort()

    return less + equal + greater

def parallel_quicksort_map_reduce(arr, num_chunks=None):
    """基于Map-Reduce的并行快速排序"""
    if num_chunks is None:
        num_chunks = mp.cpu_count()

    if len(arr) <= 1000:
        arr.sort()
        return arr

    # Map阶段：将数组分割并排序
    chunk_size = len(arr) // num_chunks
    chunks = []

    for i in range(0, len(arr), chunk_size):
        end = min(i + chunk_size, len(arr))
        chunks.append(arr[i:end])

    # 使用进程池并行排序每个chunk
    with ProcessPoolExecutor(max_workers=num_chunks) as executor:
        sorted_chunks = list(executor.map(parallel_quicksort_task, chunks))

    # Reduce阶段：合并已排序的chunks
    result = []
    while any(chunk for chunk in sorted_chunks):
        min_val = float('inf')
        min_chunk_idx = -1

        for i, chunk in enumerate(sorted_chunks):
            if chunk and chunk[0] < min_val:
                min_val = chunk[0]
                min_chunk_idx = i

        if min_chunk_idx != -1:
            result.append(sorted_chunks[min_chunk_idx].pop(0))

    return result

def benchmark_quick_sort_algorithms(arr_size=100000, num_tests=3):
    """快速排序性能基准测试"""
    print(f"=== 快速排序性能测试 ===")
    print(f"数组大小: {arr_size:,}")
    print(f"CPU核心数: {mp.cpu_count()}")
    print()

    # 生成测试数据
    test_data = [random.randint(1, 1000000) for _ in range(arr_size)]

    strategies = [
        ('串行快速排序', 'sequential', None),
        ('线程池并行快速排序', 'threadpool', None),
        ('进程池并行快速排序', 'processpool', None),
        ('Map-Reduce并行快速排序', 'mapreduce', parallel_quicksort_map_reduce),
    ]

    results = {}

    for name, strategy, custom_func in strategies:
        print(f"测试: {name}")
        times = []

        for i in range(num_tests):
            if strategy == 'mapreduce':
                data_copy = test_data.copy()
                start_time = time.time()
                sorted_data = custom_func(data_copy)
                end_time = time.time()
            else:
                data_copy = test_data.copy()
                sorter = ParallelQuickSort(strategy=strategy)
                start_time = time.time()
                sorter.sort(data_copy)
                end_time = time.time()
                sorted_data = data_copy

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

def demo_parallel_quick_sort():
    """演示并行快速排序"""
    print("=== 并行快速排序演示 ===")

    # 小规模数据演示
    small_data = [64, 34, 25, 12, 22, 11, 90, 5, 77, 30]
    print(f"原始数据: {small_data}")

    sorter = ParallelQuickSort(strategy='threadpool')
    sorter.sort(small_data)

    print(f"排序结果: {small_data}")
    print(f"验证正确: {'✅' if sorted(small_data) == small_data else '❌'}")
    print()

    # Map-Reduce演示
    print("=== Map-Reduce并行快速排序演示 ===")
    small_data2 = [64, 34, 25, 12, 22, 11, 90, 5, 77, 30]
    print(f"原始数据: {small_data2}")
    sorted_data = parallel_quicksort_map_reduce(small_data2)
    print(f"排序结果: {sorted_data}")
    print(f"验证正确: {'✅' if sorted(small_data2) == sorted_data else '❌'}")
    print()

    # 性能测试
    benchmark_quick_sort_algorithms()

if __name__ == "__main__":
    demo_parallel_quick_sort()
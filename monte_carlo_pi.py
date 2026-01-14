#!/usr/bin/env python3
"""
蒙特卡洛π值计算实现
基于第三章的并行数值计算和蒙特卡洛模拟
"""

import time
import random
import math
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import numpy as np

class MonteCarloPi:
    """蒙特卡洛π值计算类"""

    def __init__(self, strategy='sequential', num_threads=None):
        self.strategy = strategy
        self.num_threads = num_threads or mp.cpu_count()

    def sequential_monte_carlo_pi(self, num_samples):
        """串行蒙特卡洛π值计算"""
        inside_circle = 0

        for _ in range(num_samples):
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            if x*x + y*y <= 1:
                inside_circle += 1

        return 4.0 * inside_circle / num_samples

    def parallel_monte_carlo_pi_threadpool(self, num_samples):
        """使用线程池的并行蒙特卡洛π值计算"""
        num_threads = self.num_threads
        samples_per_thread = num_samples // num_threads

        def worker(samples):
            inside = 0
            for _ in range(samples):
                x = random.uniform(-1, 1)
                y = random.uniform(-1, 1)
                if x*x + y*y <= 1:
                    inside += 1
            return inside

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, samples_per_thread) for _ in range(num_threads)]

            total_inside = sum(future.result() for future in futures)

        return 4.0 * total_inside / num_samples

    def parallel_monte_carlo_pi_processpool(self, num_samples):
        """使用进程池的并行蒙特卡洛π值计算"""
        num_processes = self.num_threads
        samples_per_process = num_samples // num_processes

        def worker(samples):
            inside = 0
            # 为每个进程设置不同的随机种子
            random.seed(time.time() + id(worker))
            for _ in range(samples):
                x = random.uniform(-1, 1)
                y = random.uniform(-1, 1)
                if x*x + y*y <= 1:
                    inside += 1
            return inside

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [executor.submit(worker, samples_per_process) for _ in range(num_processes)]

            total_inside = sum(future.result() for future in futures)

        return 4.0 * total_inside / num_samples

    def parallel_monte_carlo_pi_numpy(self, num_samples):
        """使用NumPy的向量化蒙特卡洛π值计算"""
        # 生成随机点
        x = np.random.uniform(-1, 1, num_samples)
        y = np.random.uniform(-1, 1, num_samples)

        # 计算在圆内的点
        inside_circle = np.sum(x*x + y*y <= 1)

        return 4.0 * inside_circle / num_samples

    def parallel_monte_carlo_pi_numpy_parallel(self, num_samples):
        """使用NumPy和并行处理的蒙特卡洛π值计算"""
        num_threads = self.num_threads
        samples_per_thread = num_samples // num_threads

        def worker(samples):
            # 为每个线程设置不同的随机种子
            np.random.seed(int(time.time() * 1000000) % 2**32 + id(worker))
            x = np.random.uniform(-1, 1, samples)
            y = np.random.uniform(-1, 1, samples)
            return np.sum(x*x + y*y <= 1)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, samples_per_thread) for _ in range(num_threads)]

            total_inside = sum(future.result() for future in futures)

        return 4.0 * total_inside / num_samples

    def estimate_pi_with_confidence(self, num_samples, confidence_level=0.95):
        """带置信区间的π值估计"""
        # 使用并行计算
        pi_estimate = self.parallel_monte_carlo_pi_threadpool(num_samples)

        # 计算标准误差
        # 对于伯努利试验，方差 = p(1-p)
        p = pi_estimate / 4.0  # 圆内点的比例
        variance = p * (1 - p)
        standard_error = math.sqrt(variance / num_samples)

        # 计算置信区间
        z_score = 1.96  # 95%置信水平
        if confidence_level == 0.99:
            z_score = 2.576
        elif confidence_level == 0.90:
            z_score = 1.645

        margin_of_error = z_score * standard_error * 4  # 乘以4转换为π的误差

        return pi_estimate, margin_of_error

    def adaptive_monte_carlo_pi(self, target_error=0.001, max_samples=10000000):
        """自适应蒙特卡洛π值计算"""
        samples = 1000
        pi_estimate = 0
        error = float('inf')

        while error > target_error and samples < max_samples:
            # 增加样本数量
            additional_samples = min(10000, max_samples - samples)
            samples += additional_samples

            # 计算新的π值
            new_pi = self.parallel_monte_carlo_pi_threadpool(additional_samples)
            pi_estimate = (pi_estimate * (samples - additional_samples) + new_pi * additional_samples) / samples

            # 估算误差
            p = pi_estimate / 4.0
            variance = p * (1 - p)
            standard_error = math.sqrt(variance / samples)
            error = 1.96 * standard_error * 4

        return pi_estimate, error, samples

def benchmark_monte_carlo_pi(num_samples=10000000, num_tests=3):
    """蒙特卡洛π值计算性能基准测试"""
    print(f"=== 蒙特卡洛π值计算性能测试 ===")
    print(f"样本数量: {num_samples:,}")
    print(f"CPU核心数: {mp.cpu_count()}")
    print()

    strategies = [
        ('串行蒙特卡洛', 'sequential'),
        ('线程池并行蒙特卡洛', 'threadpool'),
        ('进程池并行蒙特卡洛', 'processpool'),
        ('NumPy向量化', 'numpy'),
        ('NumPy并行', 'numpy_parallel'),
    ]

    results = {}

    for name, strategy in strategies:
        print(f"测试: {name}")
        times = []
        pi_values = []

        for i in range(num_tests):
            start_time = time.time()

            if strategy == 'sequential':
                pi_value = MonteCarloPi().sequential_monte_carlo_pi(num_samples)
            elif strategy == 'threadpool':
                pi_value = MonteCarloPi().parallel_monte_carlo_pi_threadpool(num_samples)
            elif strategy == 'processpool':
                pi_value = MonteCarloPi().parallel_monte_carlo_pi_processpool(num_samples)
            elif strategy == 'numpy':
                pi_value = MonteCarloPi().parallel_monte_carlo_pi_numpy(num_samples)
            elif strategy == 'numpy_parallel':
                pi_value = MonteCarloPi().parallel_monte_carlo_pi_numpy_parallel(num_samples)

            end_time = time.time()

            execution_time = end_time - start_time
            times.append(execution_time)
            pi_values.append(pi_value)

        avg_time = sum(times) / len(times)
        min_time = min(times)
        avg_pi = sum(pi_values) / len(pi_values)

        results[strategy] = {
            'time': avg_time,
            'pi': avg_pi,
            'error': abs(avg_pi - math.pi)
        }

        print(f"  平均时间: {avg_time:.4f}s")
        print(f"  最佳时间: {min_time:.4f}s")
        print(f"  π估计值: {avg_pi:.6f}")
        print(f"  误差: {abs(avg_pi - math.pi):.6f}")
        print()

    # 计算加速比
    print("=== 加速比分析 ===")
    sequential_time = results['sequential']['time']

    for name, strategy in strategies:
        if strategy != 'sequential':
            speedup = sequential_time / results[strategy]['time']
            efficiency = speedup / mp.cpu_count()
            print(f"{name}:")
            print(f"  加速比: {speedup:.2f}x")
            print(f"  效率: {efficiency:.2%}")
            print()

def demo_monte_carlo_pi():
    """演示蒙特卡洛π值计算"""
    print("=== 蒙特卡洛π值计算演示 ===")

    # 小规模演示
    num_samples = 100000
    print(f"样本数量: {num_samples:,}")

    # 串行计算
    pi_sequential = MonteCarloPi().sequential_monte_carlo_pi(num_samples)
    print(f"串行蒙特卡洛π值: {pi_sequential:.6f}")

    # 并行计算
    pi_parallel = MonteCarloPi().parallel_monte_carlo_pi_threadpool(num_samples)
    print(f"并行蒙特卡洛π值: {pi_parallel:.6f}")

    # NumPy向量化
    pi_numpy = MonteCarloPi().parallel_monte_carlo_pi_numpy(num_samples)
    print(f"NumPy向量化π值: {pi_numpy:.6f}")

    # 真实π值
    print(f"真实π值: {math.pi:.6f}")

    # 误差比较
    print(f"\n误差比较:")
    print(f"  串行误差: {abs(pi_sequential - math.pi):.6f}")
    print(f"  并行误差: {abs(pi_parallel - math.pi):.6f}")
    print(f"  NumPy误差: {abs(pi_numpy - math.pi):.6f}")
    print()

    # 置信区间演示
    print("=== 置信区间演示 ===")
    pi_estimate, margin_of_error = MonteCarloPi().estimate_pi_with_confidence(1000000, 0.95)
    print(f"π估计值: {pi_estimate:.6f}")
    print(f"95%置信区间: [{pi_estimate - margin_of_error:.6f}, {pi_estimate + margin_of_error:.6f}]")
    print()

    # 自适应计算演示
    print("=== 自适应蒙特卡洛演示 ===")
    target_error = 0.001
    pi_adaptive, actual_error, samples_used = MonteCarloPi().adaptive_monte_carlo_pi(target_error)
    print(f"目标误差: {target_error}")
    print(f"实际误差: {actual_error:.6f}")
    print(f"使用样本数: {samples_used:,}")
    print(f"π估计值: {pi_adaptive:.6f}")
    print()

    # 性能测试
    benchmark_monte_carlo_pi()

if __name__ == "__main__":
    demo_monte_carlo_pi()
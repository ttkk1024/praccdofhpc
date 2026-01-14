#!/usr/bin/env python3
"""
并行算法性能分析和比较脚本
综合比较第三章中所有并行算法的性能
"""

import time
import random
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Callable
import sys
import os

# 导入各个算法模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from parallel_merge_sort import ParallelMergeSort
    from parallel_quick_sort import ParallelQuickSort
    from parallel_matrix_multiplication import ParallelMatrixMultiplication
    from parallel_bfs import ParallelBFS
    from parallel_dijkstra import ParallelDijkstra
    from monte_carlo_pi import MonteCarloPi
    from parallel_reduction import ParallelReduction
except ImportError as e:
    print(f"警告: 无法导入算法模块: {e}")
    print("请确保所有算法文件在同一目录下")

class PerformanceAnalyzer:
    """性能分析器类"""

    def __init__(self):
        self.results = {}
        self.cpu_count = mp.cpu_count()

    def run_performance_tests(self, data_sizes=None, num_tests=3):
        """运行所有算法的性能测试"""
        print("=== 并行算法性能综合分析 ===")
        print(f"CPU核心数: {self.cpu_count}")
        print()

        if data_sizes is None:
            data_sizes = [1000, 10000, 100000]

        # 1. 排序算法测试
        self.test_sorting_algorithms(data_sizes, num_tests)

        # 2. 矩阵乘法测试
        self.test_matrix_multiplication(data_sizes[:3], num_tests)

        # 3. 图算法测试
        self.test_graph_algorithms(data_sizes[:3], num_tests)

        # 4. 数值计算测试
        self.test_numerical_algorithms(data_sizes, num_tests)

        # 5. 归约操作测试
        self.test_reduction_operations(data_sizes, num_tests)

        # 生成报告
        self.generate_performance_report()

    def test_sorting_algorithms(self, data_sizes: List[int], num_tests: int):
        """测试排序算法"""
        print("=== 排序算法性能测试 ===")

        sorting_strategies = [
            ('串行归并排序', lambda data: ParallelMergeSort().sort(data.copy())),
            ('线程池归并排序', lambda data: ParallelMergeSort(strategy='threadpool').sort(data.copy())),
            ('串行快速排序', lambda data: ParallelQuickSort(strategy='sequential').sort(data.copy())),
            ('线程池快速排序', lambda data: ParallelQuickSort(strategy='threadpool').sort(data.copy())),
        ]

        for size in data_sizes:
            print(f"\n数据大小: {size:,}")
            data = [random.randint(1, 1000000) for _ in range(size)]

            for name, algorithm in sorting_strategies:
                times = []
                for _ in range(num_tests):
                    test_data = data.copy()
                    start_time = time.time()
                    algorithm(test_data)
                    end_time = time.time()
                    times.append(end_time - start_time)

                avg_time = sum(times) / len(times)
                self.results[f'{name}_{size}'] = {
                    'time': avg_time,
                    'category': '排序',
                    'data_size': size,
                    'algorithm': name
                }
                print(f"  {name}: {avg_time:.4f}s")

    def test_matrix_multiplication(self, matrix_sizes: List[int], num_tests: int):
        """测试矩阵乘法"""
        print("\n=== 矩阵乘法性能测试 ===")

        matrix_strategies = [
            ('串行矩阵乘法', lambda A, B: ParallelMatrixMultiplication(strategy='sequential').sort(A, B)),
            ('分块矩阵乘法', lambda A, B: ParallelMatrixMultiplication(strategy='block').sort(A, B)),
            ('并行分块矩阵乘法', lambda A, B: ParallelMatrixMultiplication(strategy='parallel_block').sort(A, B)),
        ]

        for size in matrix_sizes:
            print(f"\n矩阵大小: {size}x{size}")
            A = [[random.randint(1, 100) for _ in range(size)] for _ in range(size)]
            B = [[random.randint(1, 100) for _ in range(size)] for _ in range(size)]

            for name, algorithm in matrix_strategies:
                times = []
                for _ in range(num_tests):
                    try:
                        start_time = time.time()
                        algorithm(A, B)
                        end_time = time.time()
                        times.append(end_time - start_time)
                    except Exception as e:
                        print(f"    {name}: 跳过 (错误: {e})")
                        continue

                if times:
                    avg_time = sum(times) / len(times)
                    self.results[f'{name}_{size}'] = {
                        'time': avg_time,
                        'category': '矩阵乘法',
                        'data_size': size,
                        'algorithm': name
                    }
                    print(f"    {name}: {avg_time:.4f}s")

    def test_graph_algorithms(self, graph_sizes: List[int], num_tests: int):
        """测试图算法"""
        print("\n=== 图算法性能测试 ===")

        graph_strategies = [
            ('串行BFS', lambda graph, start: ParallelBFS(strategy='sequential').sequential_bfs(graph, start)),
            ('层级同步BFS', lambda graph, start: ParallelBFS(strategy='level_synchronous').level_synchronous_bfs(graph, start)),
            ('串行Dijkstra', lambda graph, start: ParallelDijkstra().dijkstra(graph, start, 'sequential')),
            ('并行Dijkstra', lambda graph, start: ParallelDijkstra().dijkstra(graph, start, 'parallel')),
        ]

        for size in graph_sizes:
            print(f"\n图大小: {size} 个节点")
            # 生成随机图
            graph = {}
            for i in range(size):
                neighbors = []
                for j in range(size):
                    if i != j and random.random() < 0.1:
                        neighbors.append(j)
                graph[i] = neighbors

            for name, algorithm in graph_strategies:
                times = []
                for _ in range(num_tests):
                    try:
                        start_time = time.time()
                        algorithm(graph, 0)
                        end_time = time.time()
                        times.append(end_time - start_time)
                    except Exception as e:
                        print(f"    {name}: 跳过 (错误: {e})")
                        continue

                if times:
                    avg_time = sum(times) / len(times)
                    self.results[f'{name}_{size}'] = {
                        'time': avg_time,
                        'category': '图算法',
                        'data_size': size,
                        'algorithm': name
                    }
                    print(f"    {name}: {avg_time:.4f}s")

    def test_numerical_algorithms(self, data_sizes: List[int], num_tests: int):
        """测试数值计算算法"""
        print("\n=== 数值计算性能测试 ===")

        numerical_strategies = [
            ('串行蒙特卡洛π', lambda samples: MonteCarloPi().sequential_monte_carlo_pi(samples)),
            ('并行蒙特卡洛π', lambda samples: MonteCarloPi().parallel_monte_carlo_pi_threadpool(samples)),
            ('NumPy蒙特卡洛π', lambda samples: MonteCarloPi().parallel_monte_carlo_pi_numpy(samples)),
        ]

        for size in data_sizes:
            print(f"\n样本数量: {size:,}")

            for name, algorithm in numerical_strategies:
                times = []
                for _ in range(num_tests):
                    try:
                        start_time = time.time()
                        algorithm(size)
                        end_time = time.time()
                        times.append(end_time - start_time)
                    except Exception as e:
                        print(f"    {name}: 跳过 (错误: {e})")
                        continue

                if times:
                    avg_time = sum(times) / len(times)
                    self.results[f'{name}_{size}'] = {
                        'time': avg_time,
                        'category': '数值计算',
                        'data_size': size,
                        'algorithm': name
                    }
                    print(f"    {name}: {avg_time:.4f}s")

    def test_reduction_operations(self, data_sizes: List[int], num_tests: int):
        """测试归约操作"""
        print("\n=== 归约操作性能测试 ===")

        reduction_strategies = [
            ('串行求和', lambda data: ParallelReduction().sequential_reduction(data, 'sum')),
            ('树形归约求和', lambda data: ParallelReduction().tree_reduction(data, 'sum')),
            ('分段归约求和', lambda data: ParallelReduction().segmented_reduction(data)),
            ('并行前缀和', lambda data: ParallelReduction().parallel_prefix_sum(data)),
        ]

        for size in data_sizes:
            print(f"\n数据大小: {size:,}")
            data = [random.randint(1, 1000) for _ in range(size)]

            for name, algorithm in reduction_strategies:
                times = []
                for _ in range(num_tests):
                    test_data = data.copy()
                    start_time = time.time()
                    algorithm(test_data)
                    end_time = time.time()
                    times.append(end_time - start_time)

                avg_time = sum(times) / len(times)
                self.results[f'{name}_{size}'] = {
                    'time': avg_time,
                    'category': '归约操作',
                    'data_size': size,
                    'algorithm': name
                }
                print(f"    {name}: {avg_time:.4f}s")

    def calculate_speedup_and_efficiency(self):
        """计算加速比和效率"""
        categories = ['排序', '矩阵乘法', '图算法', '数值计算', '归约操作']

        for category in categories:
            category_results = {k: v for k, v in self.results.items() if v['category'] == category}

            # 按数据大小分组
            data_sizes = sorted(list(set(v['data_size'] for v in category_results.values())))

            for size in data_sizes:
                size_results = {k: v for k, v in category_results.items() if v['data_size'] == size}

                # 找到串行版本作为基准
                serial_result = None
                for k, v in size_results.items():
                    if '串行' in v['algorithm']:
                        serial_result = v
                        break

                if serial_result:
                    for k, v in size_results.items():
                        if v['algorithm'] != serial_result['algorithm']:
                            speedup = serial_result['time'] / v['time']
                            efficiency = speedup / self.cpu_count
                            v['speedup'] = speedup
                            v['efficiency'] = efficiency

    def generate_performance_report(self):
        """生成性能报告"""
        self.calculate_speedup_and_efficiency()

        # 转换为DataFrame
        df = pd.DataFrame([
            {
                'Algorithm': v['algorithm'],
                'Category': v['category'],
                'Data_Size': v['data_size'],
                'Time': v['time'],
                'Speedup': v.get('speedup', 1.0),
                'Efficiency': v.get('efficiency', 1.0)
            }
            for k, v in self.results.items()
        ])

        print("\n=== 性能分析报告 ===")
        print("\n1. 总体性能比较:")
        print(df.groupby(['Category', 'Algorithm']).agg({
            'Time': ['min', 'mean'],
            'Speedup': 'mean',
            'Efficiency': 'mean'
        }).round(3))

        print("\n2. 最佳性能算法:")
        best_performance = df.loc[df.groupby('Category')['Speedup'].idxmax()]
        print(best_performance[['Category', 'Algorithm', 'Data_Size', 'Speedup', 'Efficiency']].round(3))

        print("\n3. 并行效率分析:")
        efficiency_analysis = df.groupby('Category').agg({
            'Speedup': ['mean', 'max'],
            'Efficiency': ['mean', 'max']
        }).round(3)
        print(efficiency_analysis)

        # 保存结果到CSV
        df.to_csv('performance_results.csv', index=False)
        print(f"\n详细结果已保存到: performance_results.csv")

        # 生成图表
        self.plot_performance_analysis(df)

    def plot_performance_analysis(self, df):
        """绘制性能分析图表"""
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # 1. 不同算法的执行时间对比
            df_pivot = df.pivot_table(values='Time', index='Algorithm', columns='Category', aggfunc='mean')
            df_pivot.plot(kind='bar', ax=axes[0, 0], title='算法执行时间对比')
            axes[0, 0].set_ylabel('时间 (秒)')
            axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            # 2. 加速比对比
            speedup_pivot = df.pivot_table(values='Speedup', index='Algorithm', columns='Category', aggfunc='mean')
            speedup_pivot.plot(kind='bar', ax=axes[0, 1], title='算法加速比对比')
            axes[0, 1].set_ylabel('加速比')
            axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            # 3. 效率对比
            efficiency_pivot = df.pivot_table(values='Efficiency', index='Algorithm', columns='Category', aggfunc='mean')
            efficiency_pivot.plot(kind='bar', ax=axes[1, 0], title='算法效率对比')
            axes[1, 0].set_ylabel('效率')
            axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            # 4. 数据规模对性能的影响
            data_size_analysis = df.groupby(['Category', 'Data_Size']).agg({'Speedup': 'mean'}).reset_index()
            for category in data_size_analysis['Category'].unique():
                cat_data = data_size_analysis[data_size_analysis['Category'] == category]
                axes[1, 1].plot(cat_data['Data_Size'], cat_data['Speedup'], marker='o', label=category)

            axes[1, 1].set_title('数据规模对加速比的影响')
            axes[1, 1].set_xlabel('数据大小')
            axes[1, 1].set_ylabel('平均加速比')
            axes[1, 1].legend()
            axes[1, 1].grid(True)

            plt.tight_layout()
            plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
            print("性能分析图表已保存到: performance_analysis.png")

        except ImportError:
            print("警告: matplotlib未安装，跳过图表生成")

def main():
    """主函数"""
    analyzer = PerformanceAnalyzer()

    # 自定义测试参数
    data_sizes = [1000, 5000, 10000]  # 可以根据需要调整
    num_tests = 3

    print(f"开始性能测试，数据规模: {data_sizes}")
    print(f"每个测试重复次数: {num_tests}")
    print()

    try:
        analyzer.run_performance_tests(data_sizes, num_tests)
        print("\n=== 测试完成 ===")

    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
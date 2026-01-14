#!/usr/bin/env python3
"""
并行Dijkstra最短路径算法实现
基于第三章的并行数值计算和并行图算法
"""

import time
import random
import heapq
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

class ParallelDijkstra:
    """并行Dijkstra最短路径算法类"""

    def __init__(self, strategy='sequential', num_threads=None):
        self.strategy = strategy
        self.num_threads = num_threads or mp.cpu_count()

    def sequential_dijkstra(self, graph, start_node):
        """串行Dijkstra算法实现"""
        # 初始化距离和前驱节点
        distances = {node: float('inf') for node in graph}
        distances[start_node] = 0
        previous = {node: None for node in graph}
        visited = set()

        # 优先队列：(distance, node)
        priority_queue = [(0, start_node)]

        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)

            # 如果已经访问过，跳过
            if current_node in visited:
                continue

            visited.add(current_node)

            # 更新邻居节点的距离
            for neighbor, weight in graph[current_node].items():
                if neighbor not in visited:
                    new_distance = current_distance + weight

                    if new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        previous[neighbor] = current_node
                        heapq.heappush(priority_queue, (new_distance, neighbor))

        return distances, previous

    def parallel_dijkstra(self, graph, start_node):
        """并行Dijkstra算法实现"""
        # 初始化
        distances = {node: float('inf') for node in graph}
        distances[start_node] = 0
        previous = {node: None for node in graph}
        visited = set()

        current_frontier = {start_node}

        while current_frontier:
            # 并行处理当前前沿的所有节点
            next_frontier = set()

            # 将前沿节点分组
            frontier_list = list(current_frontier)
            chunk_size = max(1, len(frontier_list) // self.num_threads)
            chunks = [frontier_list[i:i + chunk_size] for i in range(0, len(frontier_list), chunk_size)]

            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = []
                for chunk in chunks:
                    future = executor.submit(self._process_frontier_chunk, chunk, graph, distances, previous, visited)
                    futures.append(future)

                # 收集结果
                for future in futures:
                    chunk_next_frontier = future.result()
                    next_frontier.update(chunk_next_frontier)

            current_frontier = next_frontier

        return distances, previous

    def _process_frontier_chunk(self, chunk, graph, distances, previous, visited):
        """处理前沿块"""
        next_frontier = set()

        for current_node in chunk:
            if current_node in visited:
                continue

            visited.add(current_node)
            current_distance = distances[current_node]

            # 更新邻居节点的距离
            for neighbor, weight in graph[current_node].items():
                if neighbor not in visited:
                    new_distance = current_distance + weight

                    # 原子操作：使用锁保护距离更新
                    if new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        previous[neighbor] = current_node
                        next_frontier.add(neighbor)

        return next_frontier

    def bidirectional_dijkstra(self, graph, start, end):
        """双向Dijkstra算法"""
        if start == end:
            return 0, [start]

        # 正向搜索
        forward_distances = {node: float('inf') for node in graph}
        forward_distances[start] = 0
        forward_previous = {node: None for node in graph}
        forward_visited = set()
        forward_frontier = {start}

        # 反向搜索
        backward_distances = {node: float('inf') for node in graph}
        backward_distances[end] = 0
        backward_previous = {node: None for node in graph}
        backward_visited = set()
        backward_frontier = {end}

        meeting_node = None
        min_distance = float('inf')

        while forward_frontier and backward_frontier:
            # 并行处理正向前沿
            forward_next_frontier = set()
            forward_list = list(forward_frontier)
            chunk_size = max(1, len(forward_list) // self.num_threads)

            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = []
                for i in range(0, len(forward_list), chunk_size):
                    chunk = forward_list[i:i + chunk_size]
                    future = executor.submit(self._process_frontier_chunk_bidirectional,
                                           chunk, graph, forward_distances, forward_previous, forward_visited)
                    futures.append(future)

                for future in futures:
                    forward_next_frontier.update(future.result())

            forward_frontier = forward_next_frontier

            # 检查相遇点
            for node in forward_visited:
                if node in backward_visited:
                    total_distance = forward_distances[node] + backward_distances[node]
                    if total_distance < min_distance:
                        min_distance = total_distance
                        meeting_node = node

            if meeting_node:
                break

            # 并行处理反向前沿
            backward_next_frontier = set()
            backward_list = list(backward_frontier)
            chunk_size = max(1, len(backward_list) // self.num_threads)

            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = []
                for i in range(0, len(backward_list), chunk_size):
                    chunk = backward_list[i:i + chunk_size]
                    future = executor.submit(self._process_frontier_chunk_bidirectional_reverse,
                                           chunk, graph, backward_distances, backward_previous, backward_visited)
                    futures.append(future)

                for future in futures:
                    backward_next_frontier.update(future.result())

            backward_frontier = backward_next_frontier

            # 检查相遇点
            for node in backward_visited:
                if node in forward_visited:
                    total_distance = forward_distances[node] + backward_distances[node]
                    if total_distance < min_distance:
                        min_distance = total_distance
                        meeting_node = node

            if meeting_node:
                break

        if meeting_node is None:
            return float('inf'), []

        # 重构路径
        path = []
        current = meeting_node

        # 从meeting_node到start
        while current is not None:
            path.append(current)
            current = forward_previous[current]

        path.reverse()

        # 从meeting_node到end
        current = backward_previous[meeting_node]
        while current is not None:
            path.append(current)
            current = backward_previous[current]

        return min_distance, path

    def _process_frontier_chunk_bidirectional(self, chunk, graph, distances, previous, visited):
        """双向Dijkstra的正向处理"""
        next_frontier = set()

        for current_node in chunk:
            if current_node in visited:
                continue

            visited.add(current_node)
            current_distance = distances[current_node]

            for neighbor, weight in graph[current_node].items():
                if neighbor not in visited:
                    new_distance = current_distance + weight
                    if new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        previous[neighbor] = current_node
                        next_frontier.add(neighbor)

        return next_frontier

    def _process_frontier_chunk_bidirectional_reverse(self, chunk, graph, distances, previous, visited):
        """双向Dijkstra的反向处理"""
        next_frontier = set()

        for current_node in chunk:
            if current_node in visited:
                continue

            visited.add(current_node)
            current_distance = distances[current_node]

            for neighbor, weight in graph[current_node].items():
                if neighbor not in visited:
                    new_distance = current_distance + weight
                    if new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        previous[neighbor] = current_node
                        next_frontier.add(neighbor)

        return next_frontier

    def parallel_a_star(self, graph, start, end, heuristic_func):
        """并行A*算法"""
        # 初始化
        distances = {node: float('inf') for node in graph}
        distances[start] = 0
        previous = {node: None for node in graph}
        visited = set()

        # 优先队列：(f_score, g_score, node)
        priority_queue = [(heuristic_func(start, end), 0, start)]

        while priority_queue:
            f_score, current_distance, current_node = heapq.heappop(priority_queue)

            if current_node in visited:
                continue

            if current_node == end:
                break

            visited.add(current_node)

            # 并行处理邻居节点
            neighbors = list(graph[current_node].items())
            chunk_size = max(1, len(neighbors) // self.num_threads)
            chunks = [neighbors[i:i + chunk_size] for i in range(0, len(neighbors), chunk_size)]

            next_frontier = []

            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = []
                for chunk in chunks:
                    future = executor.submit(self._process_neighbors_chunk,
                                           chunk, current_node, current_distance, distances, previous, visited, heuristic_func, end)
                    futures.append(future)

                for future in futures:
                    next_frontier.extend(future.result())

            # 将新的前沿节点加入优先队列
            for neighbor, new_distance in next_frontier:
                f_score = new_distance + heuristic_func(neighbor, end)
                heapq.heappush(priority_queue, (f_score, new_distance, neighbor))

        return distances, previous

    def _process_neighbors_chunk(self, chunk, current_node, current_distance, distances, previous, visited, heuristic_func, end):
        """处理邻居节点块"""
        next_frontier = []

        for neighbor, weight in chunk:
            if neighbor not in visited:
                new_distance = current_distance + weight

                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous[neighbor] = current_node
                    next_frontier.append((neighbor, new_distance))

        return next_frontier

    def get_shortest_path(self, graph, start, end, method='sequential'):
        """获取最短路径"""
        if method == 'sequential':
            distances, previous = self.sequential_dijkstra(graph, start)
        elif method == 'parallel':
            distances, previous = self.parallel_dijkstra(graph, start)
        elif method == 'bidirectional':
            return self.bidirectional_dijkstra(graph, start, end)
        else:
            raise ValueError(f"Unsupported method: {method}")

        if distances[end] == float('inf'):
            return float('inf'), []

        # 重构路径
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = previous[current]

        path.reverse()
        return distances[end], path

def generate_random_weighted_graph(n, edge_probability=0.1, max_weight=10):
    """生成随机带权图"""
    graph = defaultdict(dict)

    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < edge_probability:
                weight = random.randint(1, max_weight)
                graph[i][j] = weight
                graph[j][i] = weight

    return graph

def manhattan_distance(node1, node2):
    """曼哈顿距离启发函数"""
    # 假设节点是坐标形式 (x, y)
    if isinstance(node1, int) and isinstance(node2, int):
        # 如果节点是整数，使用简单的差值
        return abs(node1 - node2)
    return abs(node1[0] - node2[0]) + abs(node1[1] - node2[1])

def benchmark_dijkstra_algorithms(graph_size=1000, num_tests=3):
    """Dijkstra算法性能基准测试"""
    print(f"=== Dijkstra算法性能测试 ===")
    print(f"图大小: {graph_size} 个节点")
    print(f"CPU核心数: {mp.cpu_count()}")
    print()

    # 生成测试图
    graph = generate_random_weighted_graph(graph_size, 0.05, 10)
    start_node = 0
    end_node = graph_size - 1

    strategies = [
        ('串行Dijkstra', 'sequential'),
        ('并行Dijkstra', 'parallel'),
        ('双向Dijkstra', 'bidirectional'),
    ]

    results = {}

    for name, method in strategies:
        print(f"测试: {name}")
        times = []

        for i in range(num_tests):
            start_time = time.time()
            dijkstra = ParallelDijkstra()
            distance, path = dijkstra.get_shortest_path(graph, start_node, end_node, method)
            end_time = time.time()

            execution_time = end_time - start_time
            times.append(execution_time)

        avg_time = sum(times) / len(times)
        min_time = min(times)
        results[method] = avg_time

        print(f"  平均时间: {avg_time:.4f}s")
        print(f"  最佳时间: {min_time:.4f}s")
        print(f"  最短距离: {distance}")
        print()

    # 计算加速比
    print("=== 加速比分析 ===")
    sequential_time = results['sequential']

    for name, method in strategies:
        if method != 'sequential':
            speedup = sequential_time / results[method]
            efficiency = speedup / mp.cpu_count()
            print(f"{name}:")
            print(f"  加速比: {speedup:.2f}x")
            print(f"  效率: {efficiency:.2%}")
            print()

def demo_parallel_dijkstra():
    """演示并行Dijkstra算法"""
    print("=== 并行Dijkstra算法演示 ===")

    # 小规模图演示
    graph = {
        0: {1: 4, 2: 2},
        1: {0: 4, 2: 1, 3: 5},
        2: {0: 2, 1: 1, 3: 8, 4: 10},
        3: {1: 5, 2: 8, 4: 2},
        4: {2: 10, 3: 2}
    }

    print("图结构:")
    for node, neighbors in graph.items():
        print(f"  {node}: {neighbors}")

    start_node = 0
    end_node = 4

    # 串行Dijkstra
    dijkstra = ParallelDijkstra(strategy='sequential')
    distance, path = dijkstra.get_shortest_path(graph, start_node, end_node, 'sequential')

    print(f"\n从节点 {start_node} 到 {end_node} 的最短路径:")
    print(f"  距离: {distance}")
    print(f"  路径: {' -> '.join(map(str, path))}")

    # 并行Dijkstra
    distance_parallel, path_parallel = dijkstra.get_shortest_path(graph, start_node, end_node, 'parallel')
    print(f"\n并行Dijkstra结果:")
    print(f"  距离: {distance_parallel}")
    print(f"  路径: {' -> '.join(map(str, path_parallel))}")

    # 双向Dijkstra
    distance_bidirectional, path_bidirectional = dijkstra.bidirectional_dijkstra(graph, start_node, end_node)
    print(f"\n双向Dijkstra结果:")
    print(f"  距离: {distance_bidirectional}")
    print(f"  路径: {' -> '.join(map(str, path_bidirectional))}")

    print()

    # 性能测试
    benchmark_dijkstra_algorithms()

if __name__ == "__main__":
    demo_parallel_dijkstra()
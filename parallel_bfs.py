#!/usr/bin/env python3
"""
并行BFS图搜索算法实现
基于第三章的并行图算法和并行搜索算法
"""

import time
import random
import heapq
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

class ParallelBFS:
    """并行BFS图搜索类"""

    def __init__(self, strategy='level_synchronous', num_threads=None):
        self.strategy = strategy
        self.num_threads = num_threads or mp.cpu_count()

    def sequential_bfs(self, graph, start_node):
        """串行BFS实现"""
        visited = set()
        distance = {}
        parent = {}

        queue = deque([start_node])
        visited.add(start_node)
        distance[start_node] = 0
        parent[start_node] = None

        while queue:
            current = queue.popleft()

            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distance[neighbor] = distance[current] + 1
                    parent[neighbor] = current
                    queue.append(neighbor)

        return distance, parent

    def level_synchronous_bfs(self, graph, start_node):
        """层级同步并行BFS实现"""
        visited = set()
        distance = {}
        parent = {}

        current_level = {start_node}
        visited.add(start_node)
        distance[start_node] = 0
        parent[start_node] = None
        level = 0

        while current_level:
            next_level = set()

            # 并行处理当前层级的所有节点
            def process_node(node):
                neighbors_to_add = []
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        neighbors_to_add.append(neighbor)
                        visited.add(neighbor)
                        distance[neighbor] = level + 1
                        parent[neighbor] = node
                return neighbors_to_add

            # 分配任务
            nodes_list = list(current_level)
            chunk_size = max(1, len(nodes_list) // self.num_threads)
            chunks = [nodes_list[i:i + chunk_size] for i in range(0, len(nodes_list), chunk_size)]

            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = []
                for chunk in chunks:
                    for node in chunk:
                        future = executor.submit(process_node, node)
                        futures.append(future)

                # 收集结果
                for future in futures:
                    neighbors = future.result()
                    next_level.update(neighbors)

            current_level = next_level
            level += 1

        return distance, parent

    def work_stealing_bfs(self, graph, start_node):
        """工作窃取并行BFS实现"""
        visited = set()
        distance = {}
        parent = {}

        # 每个线程维护自己的队列
        thread_queues = [deque() for _ in range(self.num_threads)]
        visited.add(start_node)
        distance[start_node] = 0
        parent[start_node] = None

        # 初始节点分配给第一个队列
        thread_queues[0].append(start_node)

        # 线程状态
        active_threads = self.num_threads
        completed = [False] * self.num_threads

        def worker(thread_id):
            local_queue = thread_queues[thread_id]

            while True:
                # 尝试从本地队列取任务
                if local_queue:
                    current = local_queue.popleft()
                else:
                    # 工作窃取：从其他队列末尾窃取
                    stolen = False
                    for i in range(self.num_threads):
                        if i != thread_id and thread_queues[i]:
                            # 从队列末尾窃取（减少竞争）
                            current = thread_queues[i].pop()
                            stolen = True
                            break

                    if not stolen:
                        # 检查是否所有线程都完成
                        if all(completed[j] or not thread_queues[j] for j in range(self.num_threads)):
                            completed[thread_id] = True
                            break
                        continue

                # 处理当前节点
                for neighbor in graph[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distance[neighbor] = distance[current] + 1
                        parent[neighbor] = current
                        # 随机分配到某个队列
                        target_queue = random.randint(0, self.num_threads - 1)
                        thread_queues[target_queue].append(neighbor)

        # 启动工作线程
        threads = []
        for i in range(self.num_threads):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        return distance, parent

    def frontier_based_bfs(self, graph, start_node):
        """基于前沿的并行BFS实现"""
        visited = set()
        distance = {}
        parent = {}

        frontier = {start_node}
        visited.add(start_node)
        distance[start_node] = 0
        parent[start_node] = None

        while frontier:
            next_frontier = set()

            # 并行处理前沿节点
            nodes_list = list(frontier)
            chunk_size = max(1, len(nodes_list) // self.num_threads)
            chunks = [nodes_list[i:i + chunk_size] for i in range(0, len(nodes_list), chunk_size)]

            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = []
                for chunk in chunks:
                    future = executor.submit(self._process_frontier_chunk, chunk, graph, visited, distance, parent)
                    futures.append(future)

                # 收集下一层前沿
                for future in futures:
                    next_frontier.update(future.result())

            frontier = next_frontier

        return distance, parent

    def _process_frontier_chunk(self, chunk, graph, visited, distance, parent):
        """处理前沿块"""
        next_frontier = set()
        current_distance = distance[chunk[0]] if chunk else 0

        for node in chunk:
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distance[neighbor] = current_distance + 1
                    parent[neighbor] = node
                    next_frontier.add(neighbor)

        return next_frontier

    def shortest_path(self, graph, start, end):
        """寻找最短路径"""
        distance, parent = self.sequential_bfs(graph, start)

        if end not in distance:
            return None, float('inf')

        # 重构路径
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = parent[current]

        path.reverse()
        return path, distance[end]

    def connected_components(self, graph):
        """寻找连通分量"""
        visited = set()
        components = []

        for node in graph:
            if node not in visited:
                component = self._bfs_component(graph, node, visited)
                components.append(component)

        return components

    def _bfs_component(self, graph, start_node, visited):
        """BFS查找连通分量"""
        component = []
        queue = deque([start_node])
        visited.add(start_node)

        while queue:
            current = queue.popleft()
            component.append(current)

            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return component

def generate_random_graph(n, edge_probability=0.1):
    """生成随机图"""
    graph = defaultdict(list)

    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < edge_probability:
                graph[i].append(j)
                graph[j].append(i)

    return graph

def benchmark_bfs_algorithms(graph_size=10000, num_tests=3):
    """BFS算法性能基准测试"""
    print(f"=== BFS算法性能测试 ===")
    print(f"图大小: {graph_size} 个节点")
    print(f"CPU核心数: {mp.cpu_count()}")
    print()

    # 生成测试图
    graph = generate_random_graph(graph_size, 0.05)
    start_node = 0

    strategies = [
        ('串行BFS', 'sequential'),
        ('层级同步BFS', 'level_synchronous'),
        ('工作窃取BFS', 'work_stealing'),
        ('前沿BFS', 'frontier_based'),
    ]

    results = {}

    for name, strategy in strategies:
        print(f"测试: {name}")
        times = []

        for i in range(num_tests):
            start_time = time.time()
            bfs = ParallelBFS(strategy=strategy)
            distance, parent = bfs.sequential_bfs(graph, start_node)
            end_time = time.time()

            execution_time = end_time - start_time
            times.append(execution_time)

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

def demo_parallel_bfs():
    """演示并行BFS"""
    print("=== 并行BFS演示 ===")

    # 小规模图演示
    graph = {
        0: [1, 2],
        1: [0, 3, 4],
        2: [0, 5],
        3: [1],
        4: [1, 5],
        5: [2, 4]
    }

    print("图结构:")
    for node, neighbors in graph.items():
        print(f"  {node}: {neighbors}")

    start_node = 0

    # 串行BFS
    bfs = ParallelBFS(strategy='sequential')
    distance, parent = bfs.sequential_bfs(graph, start_node)

    print(f"\n从节点 {start_node} 开始的BFS结果:")
    for node in sorted(distance.keys()):
        print(f"  节点 {node}: 距离={distance[node]}, 父节点={parent[node]}")

    # 最短路径演示
    end_node = 5
    path, dist = bfs.shortest_path(graph, start_node, end_node)
    if path:
        print(f"\n从 {start_node} 到 {end_node} 的最短路径: {' -> '.join(map(str, path))}")
        print(f"路径长度: {dist}")
    else:
        print(f"\n从 {start_node} 到 {end_node} 不可达")

    # 连通分量演示
    components = bfs.connected_components(graph)
    print(f"\n连通分量:")
    for i, component in enumerate(components):
        print(f"  分量 {i+1}: {component}")

    print()

    # 性能测试
    benchmark_bfs_algorithms()

if __name__ == "__main__":
    import threading
    demo_parallel_bfs()
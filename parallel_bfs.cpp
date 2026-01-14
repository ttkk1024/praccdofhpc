#include <iostream>
#include <vector>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <thread>
#include <future>
#include <random>
#include <chrono>
#include <memory>
#include <functional>
#include <algorithm>

class ParallelBFS {
private:
    int num_threads_;

    // 串行BFS实现
    std::pair<std::unordered_map<int, int>, std::unordered_map<int, int>>
    sequential_bfs(const std::unordered_map<int, std::vector<int>>& graph, int start_node) {
        std::unordered_set<int> visited;
        std::unordered_map<int, int> distance;
        std::unordered_map<int, int> parent;

        std::queue<int> queue;
        queue.push(start_node);
        visited.insert(start_node);
        distance[start_node] = 0;
        parent[start_node] = -1;

        while (!queue.empty()) {
            int current = queue.front();
            queue.pop();

            for (int neighbor : graph.at(current)) {
                if (visited.find(neighbor) == visited.end()) {
                    visited.insert(neighbor);
                    distance[neighbor] = distance[current] + 1;
                    parent[neighbor] = current;
                    queue.push(neighbor);
                }
            }
        }

        return {distance, parent};
    }

    // 层级同步并行BFS实现
    std::pair<std::unordered_map<int, int>, std::unordered_map<int, int>>
    level_synchronous_bfs(const std::unordered_map<int, std::vector<int>>& graph, int start_node) {
        std::unordered_set<int> visited;
        std::unordered_map<int, int> distance;
        std::unordered_map<int, int> parent;

        std::unordered_set<int> current_level;
        current_level.insert(start_node);
        visited.insert(start_node);
        distance[start_node] = 0;
        parent[start_node] = -1;
        int level = 0;

        while (!current_level.empty()) {
            std::unordered_set<int> next_level;

            // 并行处理当前层级的所有节点
            std::vector<int> nodes_list(current_level.begin(), current_level.end());
            int chunk_size = std::max(1, static_cast<int>(nodes_list.size()) / num_threads_);
            std::vector<std::future<std::vector<int>>> futures;

            for (int i = 0; i < nodes_list.size(); i += chunk_size) {
                int end = std::min(i + chunk_size, static_cast<int>(nodes_list.size()));
                std::vector<int> chunk(nodes_list.begin() + i, nodes_list.begin() + end);

                auto future = std::async(std::launch::async, [&, chunk]() {
                    std::vector<int> neighbors_to_add;
                    for (int node : chunk) {
                        for (int neighbor : graph.at(node)) {
                            if (visited.find(neighbor) == visited.end()) {
                                std::lock_guard<std::mutex> lock(mutex_);
                                if (visited.find(neighbor) == visited.end()) {
                                    visited.insert(neighbor);
                                    distance[neighbor] = level + 1;
                                    parent[neighbor] = node;
                                    neighbors_to_add.push_back(neighbor);
                                }
                            }
                        }
                    }
                    return neighbors_to_add;
                });

                futures.push_back(std::move(future));
            }

            // 收集结果
            for (auto& future : futures) {
                auto neighbors = future.get();
                next_level.insert(neighbors.begin(), neighbors.end());
            }

            current_level = std::move(next_level);
            level++;
        }

        return {distance, parent};
    }

    // 前沿并行BFS实现
    std::pair<std::unordered_map<int, int>, std::unordered_map<int, int>>
    frontier_based_bfs(const std::unordered_map<int, std::vector<int>>& graph, int start_node) {
        std::unordered_set<int> visited;
        std::unordered_map<int, int> distance;
        std::unordered_map<int, int> parent;

        std::unordered_set<int> frontier;
        frontier.insert(start_node);
        visited.insert(start_node);
        distance[start_node] = 0;
        parent[start_node] = -1;

        while (!frontier.empty()) {
            std::unordered_set<int> next_frontier;

            // 并行处理前沿节点
            std::vector<int> nodes_list(frontier.begin(), frontier.end());
            int chunk_size = std::max(1, static_cast<int>(nodes_list.size()) / num_threads_);
            std::vector<std::future<std::vector<int>>> futures;

            for (int i = 0; i < nodes_list.size(); i += chunk_size) {
                int end = std::min(i + chunk_size, static_cast<int>(nodes_list.size()));
                std::vector<int> chunk(nodes_list.begin() + i, nodes_list.begin() + end);

                auto future = std::async(std::launch::async, [&, chunk]() {
                    std::vector<int> next_frontier_chunk;
                    int current_distance = (chunk.empty()) ? 0 : distance[chunk[0]];

                    for (int node : chunk) {
                        for (int neighbor : graph.at(node)) {
                            if (visited.find(neighbor) == visited.end()) {
                                std::lock_guard<std::mutex> lock(mutex_);
                                if (visited.find(neighbor) == visited.end()) {
                                    visited.insert(neighbor);
                                    distance[neighbor] = current_distance + 1;
                                    parent[neighbor] = node;
                                    next_frontier_chunk.push_back(neighbor);
                                }
                            }
                        }
                    }
                    return next_frontier_chunk;
                });

                futures.push_back(std::move(future));
            }

            // 收集下一层前沿
            for (auto& future : futures) {
                auto next_frontier_chunk = future.get();
                next_frontier.insert(next_frontier_chunk.begin(), next_frontier_chunk.end());
            }

            frontier = std::move(next_frontier);
        }

        return {distance, parent};
    }

    std::mutex mutex_;

public:
    ParallelBFS(int num_threads = -1) {
        num_threads_ = (num_threads == -1) ? std::thread::hardware_concurrency() : num_threads;
    }

    // 主要的BFS函数
    std::pair<std::unordered_map<int, int>, std::unordered_map<int, int>>
    bfs(const std::unordered_map<int, std::vector<int>>& graph, int start_node, const std::string& method = "sequential") {
        if (method == "sequential") {
            return sequential_bfs(graph, start_node);
        } else if (method == "level_synchronous") {
            return level_synchronous_bfs(graph, start_node);
        } else if (method == "frontier_based") {
            return frontier_based_bfs(graph, start_node);
        } else {
            throw std::invalid_argument("不支持的方法: " + method);
        }
    }

    // 寻找最短路径
    std::pair<std::vector<int>, int>
    shortest_path(const std::unordered_map<int, std::vector<int>>& graph, int start, int end) {
        auto [distance, parent] = sequential_bfs(graph, start);

        if (distance.find(end) == distance.end()) {
            return {{}, INT_MAX};
        }

        // 重构路径
        std::vector<int> path;
        int current = end;
        while (current != -1) {
            path.push_back(current);
            current = parent[current];
        }

        std::reverse(path.begin(), path.end());
        return {path, distance[end]};
    }

    // 寻找连通分量
    std::vector<std::vector<int>>
    connected_components(const std::unordered_map<int, std::vector<int>>& graph) {
        std::unordered_set<int> visited;
        std::vector<std::vector<int>> components;

        for (const auto& [node, _] : graph) {
            if (visited.find(node) == visited.end()) {
                auto component = bfs_component(graph, node, visited);
                components.push_back(component);
            }
        }

        return components;
    }

private:
    std::vector<int>
    bfs_component(const std::unordered_map<int, std::vector<int>>& graph, int start_node, std::unordered_set<int>& visited) {
        std::vector<int> component;
        std::queue<int> queue;

        queue.push(start_node);
        visited.insert(start_node);

        while (!queue.empty()) {
            int current = queue.front();
            queue.pop();
            component.push_back(current);

            for (int neighbor : graph.at(current)) {
                if (visited.find(neighbor) == visited.end()) {
                    visited.insert(neighbor);
                    queue.push(neighbor);
                }
            }
        }

        return component;
    }
};

// 生成随机图
std::unordered_map<int, std::vector<int>>
generate_random_graph(int n, double edge_probability = 0.05) {
    std::unordered_map<int, std::vector<int>> graph;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < n; i++) {
        graph[i] = std::vector<int>();
    }

    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (dis(gen) < edge_probability) {
                graph[i].push_back(j);
                graph[j].push_back(i);
            }
        }
    }

    return graph;
}

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
void benchmark_bfs_algorithms(int graph_size = 10000, int num_tests = 3) {
    std::cout << "=== BFS算法性能测试 ===" << std::endl;
    std::cout << "图大小: " << graph_size << " 个节点" << std::endl;
    std::cout << "CPU核心数: " << std::thread::hardware_concurrency() << std::endl;
    std::cout << std::endl;

    // 生成测试图
    auto graph = generate_random_graph(graph_size, 0.05);
    int start_node = 0;

    std::vector<std::pair<std::string, std::string>> strategies = {
        {"串行BFS", "sequential"},
        {"层级同步BFS", "level_synchronous"},
        {"前沿BFS", "frontier_based"}
    };

    std::vector<double> results;

    for (const auto& strategy : strategies) {
        std::cout << "测试: " << strategy.first << std::endl;

        std::vector<double> times;
        for (int i = 0; i < num_tests; i++) {
            PerformanceTimer timer;
            timer.start();

            ParallelBFS bfs(num_threads_);
            auto [distance, parent] = bfs.bfs(graph, start_node, strategy.second);

            double execution_time = timer.elapsed_seconds();
            times.push_back(execution_time);
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
void demo_parallel_bfs() {
    std::cout << "=== 并行BFS演示 ===" << std::endl;

    // 小规模图演示
    std::unordered_map<int, std::vector<int>> graph = {
        {0, {1, 2}},
        {1, {0, 3, 4}},
        {2, {0, 5}},
        {3, {1}},
        {4, {1, 5}},
        {5, {2, 4}}
    };

    std::cout << "图结构:" << std::endl;
    for (const auto& [node, neighbors] : graph) {
        std::cout << "  " << node << ": ";
        for (int neighbor : neighbors) {
            std::cout << neighbor << " ";
        }
        std::cout << std::endl;
    }

    int start_node = 0;

    // 串行BFS
    ParallelBFS bfs;
    auto [distance, parent] = bfs.bfs(graph, start_node, "sequential");

    std::cout << "\n从节点 " << start_node << " 开始的BFS结果:" << std::endl;
    for (const auto& [node, dist] : distance) {
        std::cout << "  节点 " << node << ": 距离=" << dist << ", 父节点=" << parent[node] << std::endl;
    }

    // 最短路径演示
    int end_node = 5;
    auto [path, dist] = bfs.shortest_path(graph, start_node, end_node);
    if (!path.empty()) {
        std::cout << "\n从 " << start_node << " 到 " << end_node << " 的最短路径: ";
        for (size_t i = 0; i < path.size(); i++) {
            std::cout << path[i];
            if (i < path.size() - 1) std::cout << " -> ";
        }
        std::cout << std::endl;
        std::cout << "路径长度: " << dist << std::endl;
    } else {
        std::cout << "\n从 " << start_node << " 到 " << end_node << " 不可达" << std::endl;
    }

    // 连通分量演示
    auto components = bfs.connected_components(graph);
    std::cout << "\n连通分量:" << std::endl;
    for (size_t i = 0; i < components.size(); i++) {
        std::cout << "  分量 " << (i + 1) << ": ";
        for (int node : components[i]) {
            std::cout << node << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;

    // 性能测试
    benchmark_bfs_algorithms();
}

int main() {
    demo_parallel_bfs();
    return 0;
}
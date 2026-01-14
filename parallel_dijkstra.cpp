#include <iostream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <thread>
#include <future>
#include <random>
#include <chrono>
#include <memory>
#include <functional>
#include <algorithm>
#include <climits>

class ParallelDijkstra {
private:
    int num_threads_;

    // 串行Dijkstra算法实现
    std::pair<std::unordered_map<int, int>, std::unordered_map<int, int>>
    sequential_dijkstra(const std::unordered_map<int, std::unordered_map<int, int>>& graph, int start_node) {
        std::unordered_map<int, int> distances;
        std::unordered_map<int, int> previous;
        std::unordered_set<int> visited;

        // 初始化
        for (const auto& [node, _] : graph) {
            distances[node] = INT_MAX;
            previous[node] = -1;
        }
        distances[start_node] = 0;

        // 优先队列：(distance, node)
        std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<std::pair<int, int>>> pq;
        pq.push({0, start_node});

        while (!pq.empty()) {
            auto [current_distance, current_node] = pq.top();
            pq.pop();

            // 如果已经访问过，跳过
            if (visited.find(current_node) != visited.end()) {
                continue;
            }

            visited.insert(current_node);

            // 更新邻居节点的距离
            for (const auto& [neighbor, weight] : graph.at(current_node)) {
                if (visited.find(neighbor) == visited.end()) {
                    int new_distance = current_distance + weight;

                    if (new_distance < distances[neighbor]) {
                        distances[neighbor] = new_distance;
                        previous[neighbor] = current_node;
                        pq.push({new_distance, neighbor});
                    }
                }
            }
        }

        return {distances, previous};
    }

    // 并行Dijkstra算法实现
    std::pair<std::unordered_map<int, int>, std::unordered_map<int, int>>
    parallel_dijkstra(const std::unordered_map<int, std::unordered_map<int, int>>& graph, int start_node) {
        std::unordered_map<int, int> distances;
        std::unordered_map<int, int> previous;
        std::unordered_set<int> visited;

        // 初始化
        for (const auto& [node, _] : graph) {
            distances[node] = INT_MAX;
            previous[node] = -1;
        }
        distances[start_node] = 0;

        std::unordered_set<int> current_frontier;
        current_frontier.insert(start_node);

        while (!current_frontier.empty()) {
            std::unordered_set<int> next_frontier;

            // 将前沿节点分组
            std::vector<int> frontier_list(current_frontier.begin(), current_frontier.end());
            int chunk_size = std::max(1, static_cast<int>(frontier_list.size()) / num_threads_);

            std::vector<std::future<std::vector<int>>> futures;

            for (int i = 0; i < frontier_list.size(); i += chunk_size) {
                int end = std::min(i + chunk_size, static_cast<int>(frontier_list.size()));
                std::vector<int> chunk(frontier_list.begin() + i, frontier_list.begin() + end);

                auto future = std::async(std::launch::async, [&, chunk]() {
                    std::vector<int> chunk_next_frontier;
                    std::mutex mutex;

                    for (int current_node : chunk) {
                        if (visited.find(current_node) != visited.end()) {
                            continue;
                        }

                        std::lock_guard<std::mutex> lock(mutex);
                        if (visited.find(current_node) != visited.end()) {
                            continue;
                        }

                        visited.insert(current_node);
                        int current_distance = distances[current_node];

                        // 更新邻居节点的距离
                        for (const auto& [neighbor, weight] : graph.at(current_node)) {
                            if (visited.find(neighbor) == visited.end()) {
                                int new_distance = current_distance + weight;

                                if (new_distance < distances[neighbor]) {
                                    distances[neighbor] = new_distance;
                                    previous[neighbor] = current_node;
                                    chunk_next_frontier.push_back(neighbor);
                                }
                            }
                        }
                    }

                    return chunk_next_frontier;
                });

                futures.push_back(std::move(future));
            }

            // 收集结果
            for (auto& future : futures) {
                auto chunk_next_frontier = future.get();
                next_frontier.insert(chunk_next_frontier.begin(), chunk_next_frontier.end());
            }

            current_frontier = std::move(next_frontier);
        }

        return {distances, previous};
    }

    // 双向Dijkstra算法
    std::pair<int, std::vector<int>>
    bidirectional_dijkstra(const std::unordered_map<int, std::unordered_map<int, int>>& graph, int start, int end) {
        if (start == end) {
            return {0, {start}};
        }

        // 正向搜索
        std::unordered_map<int, int> forward_distances;
        std::unordered_map<int, int> forward_previous;
        std::unordered_set<int> forward_visited;
        std::unordered_set<int> forward_frontier;

        // 反向搜索
        std::unordered_map<int, int> backward_distances;
        std::unordered_map<int, int> backward_previous;
        std::unordered_set<int> backward_visited;
        std::unordered_set<int> backward_frontier;

        // 初始化
        for (const auto& [node, _] : graph) {
            forward_distances[node] = INT_MAX;
            forward_previous[node] = -1;
            backward_distances[node] = INT_MAX;
            backward_previous[node] = -1;
        }

        forward_distances[start] = 0;
        backward_distances[end] = 0;
        forward_frontier.insert(start);
        backward_frontier.insert(end);

        int meeting_node = -1;
        int min_distance = INT_MAX;

        while (!forward_frontier.empty() && !backward_frontier.empty()) {
            // 并行处理正向前沿
            std::unordered_set<int> forward_next_frontier;
            std::vector<int> forward_list(forward_frontier.begin(), forward_frontier.end());
            int chunk_size = std::max(1, static_cast<int>(forward_list.size()) / num_threads_);

            std::vector<std::future<std::vector<int>>> forward_futures;

            for (int i = 0; i < forward_list.size(); i += chunk_size) {
                int end_idx = std::min(i + chunk_size, static_cast<int>(forward_list.size()));
                std::vector<int> chunk(forward_list.begin() + i, forward_list.begin() + end_idx);

                auto future = std::async(std::launch::async, [&, chunk]() {
                    std::vector<int> chunk_next_frontier;
                    std::mutex mutex;

                    for (int current_node : chunk) {
                        if (forward_visited.find(current_node) != forward_visited.end()) {
                            continue;
                        }

                        std::lock_guard<std::mutex> lock(mutex);
                        if (forward_visited.find(current_node) != forward_visited.end()) {
                            continue;
                        }

                        forward_visited.insert(current_node);
                        int current_distance = forward_distances[current_node];

                        for (const auto& [neighbor, weight] : graph.at(current_node)) {
                            if (forward_visited.find(neighbor) == forward_visited.end()) {
                                int new_distance = current_distance + weight;
                                if (new_distance < forward_distances[neighbor]) {
                                    forward_distances[neighbor] = new_distance;
                                    forward_previous[neighbor] = current_node;
                                    chunk_next_frontier.push_back(neighbor);
                                }
                            }
                        }
                    }

                    return chunk_next_frontier;
                });

                forward_futures.push_back(std::move(future));
            }

            for (auto& future : forward_futures) {
                auto chunk_next_frontier = future.get();
                forward_next_frontier.insert(chunk_next_frontier.begin(), chunk_next_frontier.end());
            }

            forward_frontier = std::move(forward_next_frontier);

            // 检查相遇点
            for (int node : forward_visited) {
                if (backward_visited.find(node) != backward_visited.end()) {
                    int total_distance = forward_distances[node] + backward_distances[node];
                    if (total_distance < min_distance) {
                        min_distance = total_distance;
                        meeting_node = node;
                    }
                }
            }

            if (meeting_node != -1) {
                break;
            }

            // 并行处理反向前沿
            std::unordered_set<int> backward_next_frontier;
            std::vector<int> backward_list(backward_frontier.begin(), backward_frontier.end());
            chunk_size = std::max(1, static_cast<int>(backward_list.size()) / num_threads_);

            std::vector<std::future<std::vector<int>>> backward_futures;

            for (int i = 0; i < backward_list.size(); i += chunk_size) {
                int end_idx = std::min(i + chunk_size, static_cast<int>(backward_list.size()));
                std::vector<int> chunk(backward_list.begin() + i, backward_list.begin() + end_idx);

                auto future = std::async(std::launch::async, [&, chunk]() {
                    std::vector<int> chunk_next_frontier;
                    std::mutex mutex;

                    for (int current_node : chunk) {
                        if (backward_visited.find(current_node) != backward_visited.end()) {
                            continue;
                        }

                        std::lock_guard<std::mutex> lock(mutex);
                        if (backward_visited.find(current_node) != backward_visited.end()) {
                            continue;
                        }

                        backward_visited.insert(current_node);
                        int current_distance = backward_distances[current_node];

                        for (const auto& [neighbor, weight] : graph.at(current_node)) {
                            if (backward_visited.find(neighbor) == backward_visited.end()) {
                                int new_distance = current_distance + weight;
                                if (new_distance < backward_distances[neighbor]) {
                                    backward_distances[neighbor] = new_distance;
                                    backward_previous[neighbor] = current_node;
                                    chunk_next_frontier.push_back(neighbor);
                                }
                            }
                        }
                    }

                    return chunk_next_frontier;
                });

                backward_futures.push_back(std::move(future));
            }

            for (auto& future : backward_futures) {
                auto chunk_next_frontier = future.get();
                backward_next_frontier.insert(chunk_next_frontier.begin(), chunk_next_frontier.end());
            }

            backward_frontier = std::move(backward_next_frontier);

            // 检查相遇点
            for (int node : backward_visited) {
                if (forward_visited.find(node) != forward_visited.end()) {
                    int total_distance = forward_distances[node] + backward_distances[node];
                    if (total_distance < min_distance) {
                        min_distance = total_distance;
                        meeting_node = node;
                    }
                }
            }

            if (meeting_node != -1) {
                break;
            }
        }

        if (meeting_node == -1) {
            return {INT_MAX, {}};
        }

        // 重构路径
        std::vector<int> path;
        int current = meeting_node;

        // 从meeting_node到start
        while (current != -1) {
            path.push_back(current);
            current = forward_previous[current];
        }

        std::reverse(path.begin(), path.end());

        // 从meeting_node到end
        current = backward_previous[meeting_node];
        while (current != -1) {
            path.push_back(current);
            current = backward_previous[current];
        }

        return {min_distance, path};
    }

public:
    ParallelDijkstra(int num_threads = -1) {
        num_threads_ = (num_threads == -1) ? std::thread::hardware_concurrency() : num_threads;
    }

    // 主要的Dijkstra函数
    std::pair<std::unordered_map<int, int>, std::unordered_map<int, int>>
    dijkstra(const std::unordered_map<int, std::unordered_map<int, int>>& graph, int start_node, const std::string& method = "sequential") {
        if (method == "sequential") {
            return sequential_dijkstra(graph, start_node);
        } else if (method == "parallel") {
            return parallel_dijkstra(graph, start_node);
        } else {
            throw std::invalid_argument("不支持的方法: " + method);
        }
    }

    // 获取最短路径
    std::pair<int, std::vector<int>>
    get_shortest_path(const std::unordered_map<int, std::unordered_map<int, int>>& graph, int start, int end, const std::string& method = "sequential") {
        if (method == "sequential") {
            auto [distances, previous] = sequential_dijkstra(graph, start);
            if (distances[end] == INT_MAX) {
                return {INT_MAX, {}};
            }

            std::vector<int> path;
            int current = end;
            while (current != -1) {
                path.push_back(current);
                current = previous[current];
            }

            std::reverse(path.begin(), path.end());
            return {distances[end], path};

        } else if (method == "parallel") {
            auto [distances, previous] = parallel_dijkstra(graph, start);
            if (distances[end] == INT_MAX) {
                return {INT_MAX, {}};
            }

            std::vector<int> path;
            int current = end;
            while (current != -1) {
                path.push_back(current);
                current = previous[current];
            }

            std::reverse(path.begin(), path.end());
            return {distances[end], path};

        } else if (method == "bidirectional") {
            return bidirectional_dijkstra(graph, start, end);

        } else {
            throw std::invalid_argument("不支持的方法: " + method);
        }
    }
};

// 生成随机带权图
std::unordered_map<int, std::unordered_map<int, int>>
generate_random_weighted_graph(int n, double edge_probability = 0.05, int max_weight = 10) {
    std::unordered_map<int, std::unordered_map<int, int>> graph;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::uniform_int_distribution<> weight_dis(1, max_weight);

    for (int i = 0; i < n; i++) {
        graph[i] = std::unordered_map<int, int>();
    }

    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (dis(gen) < edge_probability) {
                int weight = weight_dis(gen);
                graph[i][j] = weight;
                graph[j][i] = weight;
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
void benchmark_dijkstra_algorithms(int graph_size = 1000, int num_tests = 3) {
    std::cout << "=== Dijkstra算法性能测试 ===" << std::endl;
    std::cout << "图大小: " << graph_size << " 个节点" << std::endl;
    std::cout << "CPU核心数: " << std::thread::hardware_concurrency() << std::endl;
    std::cout << std::endl;

    // 生成测试图
    auto graph = generate_random_weighted_graph(graph_size, 0.05, 10);
    int start_node = 0;
    int end_node = graph_size - 1;

    std::vector<std::pair<std::string, std::string>> strategies = {
        {"串行Dijkstra", "sequential"},
        {"并行Dijkstra", "parallel"},
        {"双向Dijkstra", "bidirectional"}
    };

    std::vector<double> results;

    for (const auto& strategy : strategies) {
        std::cout << "测试: " << strategy.first << std::endl;

        std::vector<double> times;
        for (int i = 0; i < num_tests; i++) {
            PerformanceTimer timer;
            timer.start();

            ParallelDijkstra dijkstra;
            auto [distance, path] = dijkstra.get_shortest_path(graph, start_node, end_node, strategy.second);

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
void demo_parallel_dijkstra() {
    std::cout << "=== 并行Dijkstra算法演示 ===" << std::endl;

    // 小规模图演示
    std::unordered_map<int, std::unordered_map<int, int>> graph = {
        {0, {{1, 4}, {2, 2}}},
        {1, {{0, 4}, {2, 1}, {3, 5}}},
        {2, {{0, 2}, {1, 1}, {3, 8}, {4, 10}}},
        {3, {{1, 5}, {2, 8}, {4, 2}}},
        {4, {{2, 10}, {3, 2}}}
    };

    std::cout << "图结构:" << std::endl;
    for (const auto& [node, neighbors] : graph) {
        std::cout << "  " << node << ": ";
        for (const auto& [neighbor, weight] : neighbors) {
            std::cout << neighbor << "(" << weight << ") ";
        }
        std::cout << std::endl;
    }

    int start_node = 0;
    int end_node = 4;

    // 串行Dijkstra
    ParallelDijkstra dijkstra;
    auto [distance, path] = dijkstra.get_shortest_path(graph, start_node, end_node, "sequential");

    std::cout << "\n从节点 " << start_node << " 到 " << end_node << " 的最短路径:" << std::endl;
    std::cout << "  距离: " << distance << std::endl;
    std::cout << "  路径: ";
    for (size_t i = 0; i < path.size(); i++) {
        std::cout << path[i];
        if (i < path.size() - 1) std::cout << " -> ";
    }
    std::cout << std::endl;

    // 并行Dijkstra
    auto [distance_parallel, path_parallel] = dijkstra.get_shortest_path(graph, start_node, end_node, "parallel");
    std::cout << "\n并行Dijkstra结果:" << std::endl;
    std::cout << "  距离: " << distance_parallel << std::endl;
    std::cout << "  路径: ";
    for (size_t i = 0; i < path_parallel.size(); i++) {
        std::cout << path_parallel[i];
        if (i < path_parallel.size() - 1) std::cout << " -> ";
    }
    std::cout << std::endl;

    // 双向Dijkstra
    auto [distance_bidirectional, path_bidirectional] = dijkstra.get_shortest_path(graph, start_node, end_node, "bidirectional");
    std::cout << "\n双向Dijkstra结果:" << std::endl;
    std::cout << "  距离: " << distance_bidirectional << std::endl;
    std::cout << "  路径: ";
    for (size_t i = 0; i < path_bidirectional.size(); i++) {
        std::cout << path_bidirectional[i];
        if (i < path_bidirectional.size() - 1) std::cout << " -> ";
    }
    std::cout << std::endl;

    std::cout << std::endl;

    // 性能测试
    benchmark_dijkstra_algorithms();
}

int main() {
    demo_parallel_dijkstra();
    return 0;
}
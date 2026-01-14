use rayon::prelude::*;
use std::collections::{HashMap, HashSet, BinaryHeap};
use std::sync::{Arc, Mutex};
use std::cmp::Ordering;
use priority_queue::PriorityQueue;

// 加权图的表示：邻接表
pub type WeightedGraph = HashMap<usize, HashMap<usize, f64>>;

#[derive(Clone, Debug)]
struct Node {
    id: usize,
    distance: f64,
}

impl Eq for Node {}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        other.distance.partial_cmp(&self.distance).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.distance.partial_cmp(&self.distance)
    }
}

pub struct ParallelDijkstra;

impl ParallelDijkstra {
    /// 串行Dijkstra算法实现
    pub fn sequential_dijkstra(graph: &WeightedGraph, start_node: usize) -> (HashMap<usize, f64>, HashMap<usize, Option<usize>>) {
        let mut distances = HashMap::new();
        let mut parent = HashMap::new();
        let mut visited = HashSet::new();

        // 初始化距离
        for &node in graph.keys() {
            distances.insert(node, f64::INFINITY);
            parent.insert(node, None);
        }

        distances.insert(start_node, 0.0);

        // 优先队列
        let mut pq = BinaryHeap::new();
        pq.push(Node {
            id: start_node,
            distance: 0.0,
        });

        while let Some(Node { id: current, distance: current_distance }) = pq.pop() {
            if visited.contains(&current) {
                continue;
            }

            visited.insert(current);

            if let Some(neighbors) = graph.get(&current) {
                for (&neighbor, &weight) in neighbors {
                    if !visited.contains(&neighbor) {
                        let new_distance = current_distance + weight;

                        if new_distance < distances[&neighbor] {
                            distances.insert(neighbor, new_distance);
                            parent.insert(neighbor, Some(current));
                            pq.push(Node {
                                id: neighbor,
                                distance: new_distance,
                            });
                        }
                    }
                }
            }
        }

        (distances, parent)
    }

    /// 并行Dijkstra算法实现
    pub fn parallel_dijkstra(graph: &WeightedGraph, start_node: usize) -> (HashMap<usize, f64>, HashMap<usize, Option<usize>>) {
        let mut distances = HashMap::new();
        let mut parent = HashMap::new();
        let mut visited = HashSet::new();

        // 初始化距离
        for &node in graph.keys() {
            distances.insert(node, f64::INFINITY);
            parent.insert(node, None);
        }

        distances.insert(start_node, 0.0);

        let mut current_frontier = HashSet::new();
        current_frontier.insert(start_node);

        while !current_frontier.is_empty() {
            let next_frontier = Arc::new(Mutex::new(HashSet::new()));

            let frontier_list: Vec<_> = current_frontier.iter().cloned().collect();
            let chunk_size = std::cmp::max(1, frontier_list.len() / rayon::current_num_threads());

            // 并行处理当前前沿
            frontier_list
                .par_chunks(chunk_size)
                .for_each(|chunk| {
                    let mut local_next_frontier = HashSet::new();
                    let mut local_distances = HashMap::new();
                    let mut local_parent = HashMap::new();

                    for &current in chunk {
                        if visited.contains(&current) {
                            continue;
                        }

                        visited.insert(current);
                        let current_distance = *distances.get(&current).unwrap_or(&f64::INFINITY);

                        if let Some(neighbors) = graph.get(&current) {
                            for (&neighbor, &weight) in neighbors {
                                if !visited.contains(&neighbor) {
                                    let new_distance = current_distance + weight;

                                    if new_distance < distances.get(&neighbor).copied().unwrap_or(f64::INFINITY) {
                                        // 原子操作更新距离
                                        distances.insert(neighbor, new_distance);
                                        parent.insert(neighbor, Some(current));
                                        local_next_frontier.insert(neighbor);
                                    }
                                }
                            }
                        }
                    }

                    // 合并下一层前沿
                    let mut global_next_frontier = next_frontier.lock().unwrap();
                    global_next_frontier.extend(local_next_frontier);
                });

            current_frontier = Arc::try_unwrap(next_frontier).unwrap().into_inner().unwrap();
        }

        (distances, parent)
    }

    /// 双向Dijkstra算法
    pub fn bidirectional_dijkstra(
        graph: &WeightedGraph,
        start: usize,
        end: usize,
    ) -> (f64, Vec<usize>) {
        if start == end {
            return (0.0, vec![start]);
        }

        // 正向搜索
        let mut forward_distances = HashMap::new();
        let mut forward_parent = HashMap::new();
        let mut forward_visited = HashSet::new();
        let mut forward_frontier = HashSet::new();

        // 反向搜索
        let mut backward_distances = HashMap::new();
        let mut backward_parent = HashMap::new();
        let mut backward_visited = HashSet::new();
        let mut backward_frontier = HashSet::new();

        // 初始化
        for &node in graph.keys() {
            forward_distances.insert(node, f64::INFINITY);
            forward_parent.insert(node, None);
            backward_distances.insert(node, f64::INFINITY);
            backward_parent.insert(node, None);
        }

        forward_distances.insert(start, 0.0);
        backward_distances.insert(end, 0.0);
        forward_frontier.insert(start);
        backward_frontier.insert(end);

        let mut meeting_node = None;
        let mut min_distance = f64::INFINITY;

        while !forward_frontier.is_empty() && !backward_frontier.is_empty() {
            // 并行处理正向前沿
            let forward_next_frontier = Self::process_frontier(
                &graph,
                &forward_frontier,
                &mut forward_distances,
                &mut forward_parent,
                &mut forward_visited,
            );

            forward_frontier = forward_next_frontier;

            // 检查相遇点
            for &node in &forward_visited {
                if backward_visited.contains(&node) {
                    let total_distance = forward_distances[&node] + backward_distances[&node];
                    if total_distance < min_distance {
                        min_distance = total_distance;
                        meeting_node = Some(node);
                    }
                }
            }

            if meeting_node.is_some() {
                break;
            }

            // 并行处理反向前沿
            let backward_next_frontier = Self::process_frontier(
                &graph,
                &backward_frontier,
                &mut backward_distances,
                &mut backward_parent,
                &mut backward_visited,
            );

            backward_frontier = backward_next_frontier;

            // 检查相遇点
            for &node in &backward_visited {
                if forward_visited.contains(&node) {
                    let total_distance = forward_distances[&node] + backward_distances[&node];
                    if total_distance < min_distance {
                        min_distance = total_distance;
                        meeting_node = Some(node);
                    }
                }
            }

            if meeting_node.is_some() {
                break;
            }
        }

        if let Some(meeting) = meeting_node {
            // 重构路径
            let mut path = Vec::new();
            let mut current = meeting;

            // 从meeting_node到start
            while let Some(Some(parent_node)) = forward_parent.get(&current) {
                path.push(current);
                current = *parent_node;
            }
            path.push(start);
            path.reverse();

            // 从meeting_node到end
            current = meeting;
            while let Some(Some(parent_node)) = backward_parent.get(&current) {
                current = *parent_node;
                path.push(current);
            }

            (min_distance, path)
        } else {
            (f64::INFINITY, Vec::new())
        }
    }

    /// 处理前沿节点
    fn process_frontier(
        graph: &WeightedGraph,
        frontier: &HashSet<usize>,
        distances: &mut HashMap<usize, f64>,
        parent: &mut HashMap<usize, Option<usize>>,
        visited: &mut HashSet<usize>,
    ) -> HashSet<usize> {
        let next_frontier = Arc::new(Mutex::new(HashSet::new()));

        let frontier_list: Vec<_> = frontier.iter().cloned().collect();
        let chunk_size = std::cmp::max(1, frontier_list.len() / rayon::current_num_threads());

        frontier_list
            .par_chunks(chunk_size)
            .for_each(|chunk| {
                let mut local_next_frontier = HashSet::new();

                for &current in chunk {
                    if visited.contains(&current) {
                        continue;
                    }

                    visited.insert(current);
                    let current_distance = *distances.get(&current).unwrap_or(&f64::INFINITY);

                    if let Some(neighbors) = graph.get(&current) {
                        for (&neighbor, &weight) in neighbors {
                            if !visited.contains(&neighbor) {
                                let new_distance = current_distance + weight;

                                if new_distance < distances.get(&neighbor).copied().unwrap_or(f64::INFINITY) {
                                    distances.insert(neighbor, new_distance);
                                    parent.insert(neighbor, Some(current));
                                    local_next_frontier.insert(neighbor);
                                }
                            }
                        }
                    }
                }

                let mut global_next_frontier = next_frontier.lock().unwrap();
                global_next_frontier.extend(local_next_frontier);
            });

        Arc::try_unwrap(next_frontier).unwrap().into_inner().unwrap()
    }

    /// A*算法实现
    pub fn a_star_search(
        graph: &WeightedGraph,
        start: usize,
        end: usize,
        heuristic: impl Fn(usize, usize) -> f64 + Sync,
    ) -> (f64, Vec<usize>) {
        let mut distances = HashMap::new();
        let mut parent = HashMap::new();
        let mut visited = HashSet::new();

        // 初始化距离
        for &node in graph.keys() {
            distances.insert(node, f64::INFINITY);
            parent.insert(node, None);
        }

        distances.insert(start, 0.0);

        // 优先队列：(f_score, g_score, node)
        let mut pq = PriorityQueue::new();
        let h_score = heuristic(start, end);
        pq.push(start, -(0.0 + h_score)); // 使用负值实现最小堆

        while let Some((current, _)) = pq.pop() {
            if current == end {
                break;
            }

            if visited.contains(&current) {
                continue;
            }

            visited.insert(current);

            if let Some(neighbors) = graph.get(&current) {
                for (&neighbor, &weight) in neighbors {
                    if !visited.contains(&neighbor) {
                        let new_distance = distances[&current] + weight;

                        if new_distance < distances[&neighbor] {
                            distances.insert(neighbor, new_distance);
                            parent.insert(neighbor, Some(current));

                            let f_score = new_distance + heuristic(neighbor, end);
                            pq.push(neighbor, -f_score);
                        }
                    }
                }
            }
        }

        let distance = *distances.get(&end).unwrap_or(&f64::INFINITY);
        let path = Self::reconstruct_path(&parent, start, end);

        (distance, path)
    }

    /// 并行A*算法
    pub fn parallel_a_star_search(
        graph: &WeightedGraph,
        start: usize,
        end: usize,
        heuristic: impl Fn(usize, usize) -> f64 + Sync + Send,
    ) -> (f64, Vec<usize>) {
        let mut distances = HashMap::new();
        let mut parent = HashMap::new();
        let mut visited = HashSet::new();

        // 初始化距离
        for &node in graph.keys() {
            distances.insert(node, f64::INFINITY);
            parent.insert(node, None);
        }

        distances.insert(start, 0.0);

        let mut current_frontier = HashSet::new();
        current_frontier.insert(start);

        while !current_frontier.is_empty() {
            let next_frontier = Arc::new(Mutex::new(HashSet::new()));

            let frontier_list: Vec<_> = current_frontier.iter().cloned().collect();
            let chunk_size = std::cmp::max(1, frontier_list.len() / rayon::current_num_threads());

            frontier_list
                .par_chunks(chunk_size)
                .for_each(|chunk| {
                    let mut local_next_frontier = HashSet::new();

                    for &current in chunk {
                        if visited.contains(&current) {
                            continue;
                        }

                        visited.insert(current);
                        let current_distance = *distances.get(&current).unwrap_or(&f64::INFINITY);

                        if let Some(neighbors) = graph.get(&current) {
                            for (&neighbor, &weight) in neighbors {
                                if !visited.contains(&neighbor) {
                                    let new_distance = current_distance + weight;

                                    if new_distance < distances.get(&neighbor).copied().unwrap_or(f64::INFINITY) {
                                        distances.insert(neighbor, new_distance);
                                        parent.insert(neighbor, Some(current));
                                        local_next_frontier.insert(neighbor);
                                    }
                                }
                            }
                        }
                    }

                    let mut global_next_frontier = next_frontier.lock().unwrap();
                    global_next_frontier.extend(local_next_frontier);
                });

            current_frontier = Arc::try_unwrap(next_frontier).unwrap().into_inner().unwrap();
        }

        let distance = *distances.get(&end).unwrap_or(&f64::INFINITY);
        let path = Self::reconstruct_path(&parent, start, end);

        (distance, path)
    }

    /// 重构路径
    fn reconstruct_path(parent: &HashMap<usize, Option<usize>>, start: usize, end: usize) -> Vec<usize> {
        let mut path = Vec::new();
        let mut current = Some(end);

        while let Some(node) = current {
            path.push(node);
            current = parent[&node];
        }

        path.reverse();

        if path[0] == start {
            path
        } else {
            Vec::new()
        }
    }

    /// 多源Dijkstra算法
    pub fn multi_source_dijkstra(
        graph: &WeightedGraph,
        sources: &[usize],
    ) -> (HashMap<usize, f64>, HashMap<usize, Option<usize>>) {
        let mut distances = HashMap::new();
        let mut parent = HashMap::new();
        let mut visited = HashSet::new();

        // 初始化距离
        for &node in graph.keys() {
            distances.insert(node, f64::INFINITY);
            parent.insert(node, None);
        }

        // 初始化多个源点
        let mut pq = BinaryHeap::new();
        for &source in sources {
            distances.insert(source, 0.0);
            pq.push(Node {
                id: source,
                distance: 0.0,
            });
        }

        while let Some(Node { id: current, distance: current_distance }) = pq.pop() {
            if visited.contains(&current) {
                continue;
            }

            visited.insert(current);

            if let Some(neighbors) = graph.get(&current) {
                for (&neighbor, &weight) in neighbors {
                    if !visited.contains(&neighbor) {
                        let new_distance = current_distance + weight;

                        if new_distance < distances[&neighbor] {
                            distances.insert(neighbor, new_distance);
                            parent.insert(neighbor, Some(current));
                            pq.push(Node {
                                id: neighbor,
                                distance: new_distance,
                            });
                        }
                    }
                }
            }
        }

        (distances, parent)
    }

    /// 并行多源Dijkstra算法
    pub fn parallel_multi_source_dijkstra(
        graph: &WeightedGraph,
        sources: &[usize],
    ) -> (HashMap<usize, f64>, HashMap<usize, Option<usize>>) {
        let mut distances = HashMap::new();
        let mut parent = HashMap::new();
        let mut visited = HashSet::new();

        // 初始化距离
        for &node in graph.keys() {
            distances.insert(node, f64::INFINITY);
            parent.insert(node, None);
        }

        // 初始化多个源点
        for &source in sources {
            distances.insert(source, 0.0);
        }

        let mut current_frontier: HashSet<usize> = sources.iter().cloned().collect();

        while !current_frontier.is_empty() {
            let next_frontier = Arc::new(Mutex::new(HashSet::new()));

            let frontier_list: Vec<_> = current_frontier.iter().cloned().collect();
            let chunk_size = std::cmp::max(1, frontier_list.len() / rayon::current_num_threads());

            frontier_list
                .par_chunks(chunk_size)
                .for_each(|chunk| {
                    let mut local_next_frontier = HashSet::new();

                    for &current in chunk {
                        if visited.contains(&current) {
                            continue;
                        }

                        visited.insert(current);
                        let current_distance = *distances.get(&current).unwrap_or(&f64::INFINITY);

                        if let Some(neighbors) = graph.get(&current) {
                            for (&neighbor, &weight) in neighbors {
                                if !visited.contains(&neighbor) {
                                    let new_distance = current_distance + weight;

                                    if new_distance < distances.get(&neighbor).copied().unwrap_or(f64::INFINITY) {
                                        distances.insert(neighbor, new_distance);
                                        parent.insert(neighbor, Some(current));
                                        local_next_frontier.insert(neighbor);
                                    }
                                }
                            }
                        }
                    }

                    let mut global_next_frontier = next_frontier.lock().unwrap();
                    global_next_frontier.extend(local_next_frontier);
                });

            current_frontier = Arc::try_unwrap(next_frontier).unwrap().into_inner().unwrap();
        }

        (distances, parent)
    }

    /// 限制距离的Dijkstra算法
    pub fn bounded_dijkstra(
        graph: &WeightedGraph,
        start_node: usize,
        max_distance: f64,
    ) -> (HashMap<usize, f64>, HashMap<usize, Option<usize>>) {
        let mut distances = HashMap::new();
        let mut parent = HashMap::new();
        let mut visited = HashSet::new();

        // 初始化距离
        for &node in graph.keys() {
            distances.insert(node, f64::INFINITY);
            parent.insert(node, None);
        }

        distances.insert(start_node, 0.0);

        // 优先队列
        let mut pq = BinaryHeap::new();
        pq.push(Node {
            id: start_node,
            distance: 0.0,
        });

        while let Some(Node { id: current, distance: current_distance }) = pq.pop() {
            if visited.contains(&current) || current_distance > max_distance {
                continue;
            }

            visited.insert(current);

            if let Some(neighbors) = graph.get(&current) {
                for (&neighbor, &weight) in neighbors {
                    if !visited.contains(&neighbor) {
                        let new_distance = current_distance + weight;

                        if new_distance <= max_distance && new_distance < distances[&neighbor] {
                            distances.insert(neighbor, new_distance);
                            parent.insert(neighbor, Some(current));
                            pq.push(Node {
                                id: neighbor,
                                distance: new_distance,
                            });
                        }
                    }
                }
            }
        }

        (distances, parent)
    }
}

/// 曼哈顿距离启发函数
pub fn manhattan_distance_2d(node1: (usize, usize), node2: (usize, usize)) -> f64 {
    (node1.0 as i32 - node2.0 as i32).abs() as f64 + (node1.1 as i32 - node2.1 as i32).abs() as f64
}

/// 欧几里得距离启发函数
pub fn euclidean_distance_2d(node1: (usize, usize), node2: (usize, usize)) -> f64 {
    let dx = node1.0 as f64 - node2.0 as f64;
    let dy = node1.1 as f64 - node2.1 as f64;
    (dx * dx + dy * dy).sqrt()
}

/// 创建测试图的辅助函数
pub fn create_weighted_test_graph() -> WeightedGraph {
    let mut graph = HashMap::new();

    graph.insert(0, [(1, 4.0), (2, 2.0)].iter().cloned().collect());
    graph.insert(1, [(0, 4.0), (2, 1.0), (3, 5.0)].iter().cloned().collect());
    graph.insert(2, [(0, 2.0), (1, 1.0), (3, 8.0), (4, 10.0)].iter().cloned().collect());
    graph.insert(3, [(1, 5.0), (2, 8.0), (4, 2.0)].iter().cloned().collect());
    graph.insert(4, [(2, 10.0), (3, 2.0)].iter().cloned().collect());

    // 添加缺失的节点
    for node in 0..5 {
        if !graph.contains_key(&node) {
            graph.insert(node, HashMap::new());
        }
    }

    graph
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_dijkstra() {
        let graph = create_weighted_test_graph();
        let (distances, _) = ParallelDijkstra::sequential_dijkstra(&graph, 0);

        assert_eq!(distances[&0], 0.0);
        assert_eq!(distances[&1], 3.0); // 0 -> 2 -> 1
        assert_eq!(distances[&2], 2.0);
        assert_eq!(distances[&3], 8.0); // 0 -> 2 -> 1 -> 3
        assert_eq!(distances[&4], 10.0); // 0 -> 2 -> 4
    }

    #[test]
    fn test_bidirectional_dijkstra() {
        let graph = create_weighted_test_graph();
        let (distance, path) = ParallelDijkstra::bidirectional_dijkstra(&graph, 0, 4);

        assert!(distance > 0.0);
        assert!(path.len() > 1);
        assert_eq!(path[0], 0);
        assert_eq!(path.last(), Some(&4));
    }

    #[test]
    fn test_a_star_search() {
        let graph = create_weighted_test_graph();
        let heuristic = |node: usize, _: usize| node as f64 * 0.1; // 简单启发函数

        let (distance, path) = ParallelDijkstra::a_star_search(&graph, 0, 4, heuristic);

        assert!(distance > 0.0);
        assert!(path.len() > 1);
        assert_eq!(path[0], 0);
        assert_eq!(path.last(), Some(&4));
    }

    #[test]
    fn test_multi_source_dijkstra() {
        let graph = create_weighted_test_graph();
        let (distances, _) = ParallelDijkstra::multi_source_dijkstra(&graph, &[0, 1]);

        assert_eq!(distances[&0], 0.0);
        assert_eq!(distances[&1], 0.0);
        assert!(distances[&2] <= 2.0); // 应该从0或1可达
    }
}
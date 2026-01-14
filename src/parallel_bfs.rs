use rayon::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};

pub struct ParallelBFS;

// 图的表示：邻接表
pub type Graph = HashMap<usize, Vec<usize>>;

impl ParallelBFS {
    /// 串行BFS实现
    pub fn sequential_bfs(graph: &Graph, start_node: usize) -> (HashMap<usize, usize>, HashMap<usize, Option<usize>>) {
        let mut distances = HashMap::new();
        let mut parent = HashMap::new();
        let mut visited = HashSet::new();

        let mut queue = VecDeque::new();
        queue.push_back(start_node);
        visited.insert(start_node);
        distances.insert(start_node, 0);
        parent.insert(start_node, None);

        while let Some(current) = queue.pop_front() {
            if let Some(neighbors) = graph.get(&current) {
                for &neighbor in neighbors {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        distances.insert(neighbor, distances[&current] + 1);
                        parent.insert(neighbor, Some(current));
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        (distances, parent)
    }

    /// 层级同步并行BFS实现
    pub fn level_synchronous_bfs(graph: &Graph, start_node: usize) -> (HashMap<usize, usize>, HashMap<usize, Option<usize>>) {
        let mut distances = HashMap::new();
        let mut parent = HashMap::new();
        let mut visited = HashSet::new();

        let mut current_level = HashSet::new();
        current_level.insert(start_node);
        visited.insert(start_node);
        distances.insert(start_node, 0);
        parent.insert(start_node, None);
        let mut level = 0;

        while !current_level.is_empty() {
            let next_level = Arc::new(Mutex::new(HashSet::new()));
            let visited_arc = Arc::new(Mutex::new(visited.clone()));
            let distances_arc = Arc::new(Mutex::new(distances.clone()));
            let parent_arc = Arc::new(Mutex::new(parent.clone()));

            let nodes_list: Vec<_> = current_level.iter().cloned().collect();
            let chunk_size = std::cmp::max(1, nodes_list.len() / rayon::current_num_threads());

            // 并行处理当前层级的所有节点
            nodes_list
                .par_chunks(chunk_size)
                .for_each(|chunk| {
                    let mut local_next_level = HashSet::new();
                    let mut local_visited = HashSet::new();
                    let mut local_distances = HashMap::new();
                    let mut local_parent = HashMap::new();

                    for &node in chunk {
                        if let Some(neighbors) = graph.get(&node) {
                            for &neighbor in neighbors {
                                if !visited_arc.lock().unwrap().contains(&neighbor) {
                                    local_visited.insert(neighbor);
                                    local_distances.insert(neighbor, level + 1);
                                    local_parent.insert(neighbor, Some(node));
                                    local_next_level.insert(neighbor);
                                }
                            }
                        }
                    }

                    // 合并结果
                    {
                        let mut global_visited = visited_arc.lock().unwrap();
                        let mut global_distances = distances_arc.lock().unwrap();
                        let mut global_parent = parent_arc.lock().unwrap();
                        let mut global_next_level = next_level.lock().unwrap();

                        for neighbor in local_next_level {
                            global_visited.insert(neighbor);
                            global_distances.insert(neighbor, level + 1);
                            global_parent.insert(neighbor, local_parent[&neighbor]);
                            global_next_level.insert(neighbor);
                        }
                    }
                });

            // 更新全局状态
            visited = Arc::try_unwrap(visited_arc).unwrap().into_inner().unwrap();
            distances = Arc::try_unwrap(distances_arc).unwrap().into_inner().unwrap();
            parent = Arc::try_unwrap(parent_arc).unwrap().into_inner().unwrap();
            current_level = Arc::try_unwrap(next_level).unwrap().into_inner().unwrap();

            level += 1;
        }

        (distances, parent)
    }

    /// 基于前沿的并行BFS实现
    pub fn frontier_based_bfs(graph: &Graph, start_node: usize) -> (HashMap<usize, usize>, HashMap<usize, Option<usize>>) {
        let mut distances = HashMap::new();
        let mut parent = HashMap::new();
        let visited = Arc::new(Mutex::new(HashSet::new()));

        let mut frontier = HashSet::new();
        frontier.insert(start_node);
        {
            let mut v = visited.lock().unwrap();
            v.insert(start_node);
        }
        distances.insert(start_node, 0);
        parent.insert(start_node, None);

        while !frontier.is_empty() {
            let next_frontier = Arc::new(Mutex::new(HashSet::new()));

            let nodes_list: Vec<_> = frontier.iter().cloned().collect();
            let chunk_size = std::cmp::max(1, nodes_list.len() / rayon::current_num_threads());

            // 并行处理前沿节点
            nodes_list
                .par_chunks(chunk_size)
                .for_each(|chunk| {
                    let mut local_next_frontier = HashSet::new();

                    for &node in chunk {
                        if let Some(neighbors) = graph.get(&node) {
                            for &neighbor in neighbors {
                                let mut global_visited = visited.lock().unwrap_or_else(|e| e.into_inner());
                                if !global_visited.contains(&neighbor) {
                                    global_visited.insert(neighbor);
                                    distances.insert(neighbor, distances[&node] + 1);
                                    parent.insert(neighbor, Some(node));
                                    local_next_frontier.insert(neighbor);
                                }
                            }
                        }
                    }

                    // 合并下一层前沿
                    let mut global_next_frontier = next_frontier.lock().unwrap();
                    global_next_frontier.extend(local_next_frontier);
                });

            frontier = Arc::try_unwrap(next_frontier).unwrap().into_inner().unwrap();
        }

        (distances, parent)
    }

    /// 工作窃取BFS实现
    pub fn work_stealing_bfs(graph: &Graph, start_node: usize) -> (HashMap<usize, usize>, HashMap<usize, Option<usize>>) {
        use crossbeam::queue::SegQueue;

        let distances = Arc::new(Mutex::new(HashMap::new()));
        let parent = Arc::new(Mutex::new(HashMap::new()));
        let visited = Arc::new(Mutex::new(HashSet::new()));

        // 每个线程维护自己的队列
        let num_threads = rayon::current_num_threads();
        let queues: Vec<Arc<SegQueue<usize>>> = (0..num_threads)
            .map(|_| Arc::new(SegQueue::new()))
            .collect();

        // 初始化
        {
            let mut dist = distances.lock().unwrap();
            let mut par = parent.lock().unwrap();
            let mut vis = visited.lock().unwrap();

            vis.insert(start_node);
            dist.insert(start_node, 0);
            par.insert(start_node, None);
        }

        // 将起始节点放入第一个队列
        queues[0].push(start_node);

        let completed = Arc::new(Mutex::new(vec![false; num_threads]));
        let active_threads = Arc::new(std::sync::atomic::AtomicUsize::new(num_threads));

        // 启动工作线程
        let handles: Vec<_> = queues
            .iter()
            .enumerate()
            .map(|(thread_id, queue)| {
                let distances_clone = Arc::clone(&distances);
                let parent_clone = Arc::clone(&parent);
                let visited_clone = Arc::clone(&visited);
                let queues_clone = queues.clone();
                let completed_clone = Arc::clone(&completed);
                let active_clone = Arc::clone(&active_threads);

                std::thread::spawn(move || {
                    while active_clone.load(Ordering::Relaxed) > 0 {
                        // 尝试从本地队列取任务
                        let current = queue.pop();

                        if let Ok(node) = current {
                            // 处理当前节点
                            if let Some(neighbors) = graph.get(&node) {
                                let mut dist = distances_clone.lock().unwrap();
                                let mut par = parent_clone.lock().unwrap();
                                let mut vis = visited_clone.lock().unwrap();

                                for &neighbor in neighbors {
                                    if !vis.contains(&neighbor) {
                                        vis.insert(neighbor);
                                        dist.insert(neighbor, dist[&node] + 1);
                                        par.insert(neighbor, Some(node));

                                        // 随机分配到某个队列
                                        let target_queue = rand::random::<usize>() % num_threads;
                                        queues_clone[target_queue].push(neighbor);
                                    }
                                }
                            }
                        } else {
                            // 工作窃取：从其他队列末尾窃取
                            let mut stolen = false;
                            for i in 0..num_threads {
                                if i != thread_id {
                                    if let Ok(other_node) = queues_clone[i].pop() {
                                        queue.push(other_node);
                                        stolen = true;
                                        break;
                                    }
                                }
                            }

                            if !stolen {
                                // 检查是否所有队列都为空
                                let all_empty = queues_clone.iter().all(|q| q.is_empty());
                                if all_empty {
                                    let mut completed_lock = completed_clone.lock().unwrap();
                                    completed_lock[thread_id] = true;

                                    let remaining = completed_lock.iter().filter(|&&c| !c).count();
                                    if remaining == 0 {
                                        active_clone.store(0, Ordering::Relaxed);
                                    }
                                    break;
                                }
                            }
                        }
                    }
                })
            })
            .collect();

        // 等待所有线程完成
        for handle in handles {
            handle.join().unwrap();
        }

        let distances_final = Arc::try_unwrap(distances).unwrap().into_inner().unwrap();
        let parent_final = Arc::try_unwrap(parent).unwrap().into_inner().unwrap();

        (distances_final, parent_final)
    }

    /// 并发安全的BFS实现
    pub fn concurrent_bfs(graph: &Graph, start_node: usize) -> (HashMap<usize, usize>, HashMap<usize, Option<usize>>) {
        use dashmap::DashMap;

        let distances = DashMap::new();
        let parent = DashMap::new();
        let visited = DashMap::new();

        let mut frontier = HashSet::new();
        frontier.insert(start_node);

        visited.insert(start_node, true);
        distances.insert(start_node, 0);
        parent.insert(start_node, None);

        let level = std::sync::atomic::AtomicUsize::new(0);

        while !frontier.is_empty() {
            let next_frontier = Arc::new(std::sync::Mutex::new(HashSet::new()));

            let nodes_list: Vec<_> = frontier.iter().cloned().collect();
            let chunk_size = std::cmp::max(1, nodes_list.len() / rayon::current_num_threads());

            // 并行处理前沿节点
            nodes_list
                .par_chunks(chunk_size)
                .for_each(|chunk| {
                    let mut local_next_frontier = HashSet::new();

                    for &node in chunk {
                        if let Some(neighbors) = graph.get(&node) {
                            for &neighbor in neighbors {
                                // 使用DashMap的并发安全操作
                                if !visited.contains_key(&neighbor) {
                                    visited.insert(neighbor, true);
                                    distances.insert(neighbor, distances.get(&node).unwrap().value() + 1);
                                    parent.insert(neighbor, Some(node));
                                    local_next_frontier.insert(neighbor);
                                }
                            }
                        }
                    }

                    // 合并下一层前沿
                    let mut global_next_frontier = next_frontier.lock().unwrap();
                    global_next_frontier.extend(local_next_frontier);
                });

            frontier = Arc::try_unwrap(next_frontier).unwrap().into_inner().unwrap();
            level.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        // 转换为标准HashMap
        let distances_final: HashMap<_, _> = distances.into_iter().collect();
        let parent_final: HashMap<_, _> = parent.into_iter().collect();

        (distances_final, parent_final)
    }

    /// 多源BFS实现
    pub fn multi_source_bfs(graph: &Graph, start_nodes: &[usize]) -> (HashMap<usize, usize>, HashMap<usize, Option<usize>>) {
        let mut distances = HashMap::new();
        let mut parent = HashMap::new();
        let mut visited = HashSet::new();

        let mut queue = VecDeque::new();

        // 初始化多个源点
        for &start_node in start_nodes {
            visited.insert(start_node);
            distances.insert(start_node, 0);
            parent.insert(start_node, None);
            queue.push_back(start_node);
        }

        while let Some(current) = queue.pop_front() {
            if let Some(neighbors) = graph.get(&current) {
                for &neighbor in neighbors {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        distances.insert(neighbor, distances[&current] + 1);
                        parent.insert(neighbor, Some(current));
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        (distances, parent)
    }

    /// 并行多源BFS实现
    pub fn parallel_multi_source_bfs(graph: &Graph, start_nodes: &[usize]) -> (HashMap<usize, usize>, HashMap<usize, Option<usize>>) {
        let mut distances = HashMap::new();
        let mut parent = HashMap::new();
        let mut visited = HashSet::new();

        // 使用Rayon并行初始化
        start_nodes.par_iter().for_each(|&start_node| {
            visited.insert(start_node);
            distances.insert(start_node, 0);
            parent.insert(start_node, None);
        });

        let mut current_level: HashSet<usize> = start_nodes.iter().cloned().collect();

        while !current_level.is_empty() {
            let next_level = Arc::new(Mutex::new(HashSet::new()));

            let nodes_list: Vec<_> = current_level.iter().cloned().collect();
            let chunk_size = std::cmp::max(1, nodes_list.len() / rayon::current_num_threads());

            nodes_list
                .par_chunks(chunk_size)
                .for_each(|chunk| {
                    let mut local_next_level = HashSet::new();

                    for &node in chunk {
                        if let Some(neighbors) = graph.get(&node) {
                            for &neighbor in neighbors {
                                if !visited.contains(&neighbor) {
                                    visited.insert(neighbor);
                                    distances.insert(neighbor, distances[&node] + 1);
                                    parent.insert(neighbor, Some(node));
                                    local_next_level.insert(neighbor);
                                }
                            }
                        }
                    }

                    let mut global_next_level = next_level.lock().unwrap();
                    global_next_level.extend(local_next_level);
                });

            current_level = Arc::try_unwrap(next_level).unwrap().into_inner().unwrap();
        }

        (distances, parent)
    }

    /// 限制深度的BFS实现
    pub fn bounded_bfs(graph: &Graph, start_node: usize, max_depth: usize) -> (HashMap<usize, usize>, HashMap<usize, Option<usize>>) {
        let mut distances = HashMap::new();
        let mut parent = HashMap::new();
        let mut visited = HashSet::new();

        let mut current_level = HashSet::new();
        current_level.insert(start_node);
        visited.insert(start_node);
        distances.insert(start_node, 0);
        parent.insert(start_node, None);

        let mut level = 0;

        while !current_level.is_empty() && level < max_depth {
            let next_level = Arc::new(Mutex::new(HashSet::new()));

            let nodes_list: Vec<_> = current_level.iter().cloned().collect();
            let chunk_size = std::cmp::max(1, nodes_list.len() / rayon::current_num_threads());

            nodes_list
                .par_chunks(chunk_size)
                .for_each(|chunk| {
                    let mut local_next_level = HashSet::new();

                    for &node in chunk {
                        if let Some(neighbors) = graph.get(&node) {
                            for &neighbor in neighbors {
                                if !visited.contains(&neighbor) {
                                    visited.insert(neighbor);
                                    distances.insert(neighbor, level + 1);
                                    parent.insert(neighbor, Some(node));
                                    local_next_level.insert(neighbor);
                                }
                            }
                        }
                    }

                    let mut global_next_level = next_level.lock().unwrap();
                    global_next_level.extend(local_next_level);
                });

            current_level = Arc::try_unwrap(next_level).unwrap().into_inner().unwrap();
            level += 1;
        }

        (distances, parent)
    }
}

/// 使用Petgraph库的BFS实现
pub fn petgraph_bfs(graph: &Graph, start_node: usize) -> (HashMap<usize, usize>, HashMap<usize, Option<usize>>) {
    let mut pg = Graph::new_undirected();

    // 转换为Petgraph格式
    let mut node_map = HashMap::new();
    let mut nodes = Vec::new();

    // 添加节点
    for &node in graph.keys() {
        let pg_node = pg.add_node(node);
        node_map.insert(node, pg_node);
        nodes.push(node);
    }

    // 添加边
    for (&node, neighbors) in graph {
        let pg_node = node_map[&node];
        for &neighbor in neighbors {
            let pg_neighbor = node_map[&neighbor];
            if pg_node != pg_neighbor {
                pg.add_edge(pg_node, pg_neighbor, ());
            }
        }
    }

    // 执行BFS
    let mut distances = HashMap::new();
    let mut parent = HashMap::new();
    let mut visited = HashSet::new();

    let start_pg_node = node_map[&start_node];
    let mut queue = VecDeque::new();
    queue.push_back(start_pg_node);
    visited.insert(start_node);
    distances.insert(start_node, 0);
    parent.insert(start_node, None);

    while let Some(current_pg_node) = queue.pop_front() {
        let current_node = pg[current_pg_node];

        for neighbor_pg_node in pg.neighbors(current_pg_node) {
            let neighbor_node = pg[neighbor_pg_node];

            if !visited.contains(&neighbor_node) {
                visited.insert(neighbor_node);
                distances.insert(neighbor_node, distances[&current_node] + 1);
                parent.insert(neighbor_node, Some(current_node));
                queue.push_back(neighbor_pg_node);
            }
        }
    }

    (distances, parent)
}

/// 找到最短路径
pub fn shortest_path(parent: &HashMap<usize, Option<usize>>, start: usize, end: usize) -> Option<Vec<usize>> {
    if !parent.contains_key(&end) {
        return None;
    }

    let mut path = Vec::new();
    let mut current = Some(end);

    while let Some(node) = current {
        path.push(node);
        current = parent[&node];
    }

    path.reverse();

    if path[0] == start {
        Some(path)
    } else {
        None
    }
}

/// 找到连通分量
pub fn connected_components(graph: &Graph) -> Vec<Vec<usize>> {
    let mut visited = HashSet::new();
    let mut components = Vec::new();

    for &node in graph.keys() {
        if !visited.contains(&node) {
            let component = ParallelBFS::bfs_component(graph, node, &mut visited);
            components.push(component);
        }
    }

    components
}

/// BFS查找连通分量
fn bfs_component(graph: &Graph, start_node: usize, visited: &mut HashSet<usize>) -> Vec<usize> {
    let mut component = Vec::new();
    let mut queue = VecDeque::new();

    queue.push_back(start_node);
    visited.insert(start_node);

    while let Some(current) = queue.pop_front() {
        component.push(current);

        if let Some(neighbors) = graph.get(&current) {
            for &neighbor in neighbors {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    queue.push_back(neighbor);
                }
            }
        }
    }

    component
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_graph() -> Graph {
        let mut graph = HashMap::new();
        graph.insert(0, vec![1, 2]);
        graph.insert(1, vec![0, 3, 4]);
        graph.insert(2, vec![0, 5]);
        graph.insert(3, vec![1]);
        graph.insert(4, vec![1, 5]);
        graph.insert(5, vec![2, 4]);
        graph
    }

    #[test]
    fn test_sequential_bfs() {
        let graph = create_test_graph();
        let (distances, parent) = ParallelBFS::sequential_bfs(&graph, 0);

        assert_eq!(distances[&0], 0);
        assert_eq!(distances[&1], 1);
        assert_eq!(distances[&2], 1);
        assert_eq!(distances[&3], 2);
        assert_eq!(distances[&4], 2);
        assert_eq!(distances[&5], 2);
    }

    #[test]
    fn test_parallel_bfs() {
        let graph = create_test_graph();
        let (distances, parent) = ParallelBFS::level_synchronous_bfs(&graph, 0);

        assert_eq!(distances[&0], 0);
        assert_eq!(distances[&1], 1);
        assert_eq!(distances[&2], 1);
        assert_eq!(distances[&3], 2);
        assert_eq!(distances[&4], 2);
        assert_eq!(distances[&5], 2);
    }

    #[test]
    fn test_shortest_path() {
        let graph = create_test_graph();
        let (_, parent) = ParallelBFS::sequential_bfs(&graph, 0);
        let path = shortest_path(&parent, 0, 5);

        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(path[0], 0);
        assert_eq!(path.last(), Some(&5));
    }

    #[test]
    fn test_connected_components() {
        let graph = create_test_graph();
        let components = connected_components(&graph);

        assert_eq!(components.len(), 1);
        assert_eq!(components[0].len(), 6);
    }
}
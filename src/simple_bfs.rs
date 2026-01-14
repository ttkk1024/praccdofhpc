use rayon::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};

pub struct SimpleParallelBFS;

pub type Graph = HashMap<usize, Vec<usize>>;

impl SimpleParallelBFS {
    /// 简化版并行BFS - 使用DashMap简化并发控制
    pub fn simple_parallel_bfs(graph: &Graph, start_node: usize) -> (HashMap<usize, usize>, HashMap<usize, Option<usize>>) {
        use dashmap::DashMap;

        let distances = DashMap::new();
        let parent = DashMap::new();
        let visited = DashMap::new();

        let mut current_level = HashSet::new();
        current_level.insert(start_node);

        distances.insert(start_node, 0);
        parent.insert(start_node, None);
        visited.insert(start_node, true);

        let mut level = 0;

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
                                if !visited.contains_key(&neighbor) {
                                    visited.insert(neighbor, true);
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

        let distances: HashMap<usize, usize> = distances.into_iter().collect();
        let parent: HashMap<usize, Option<usize>> = parent.into_iter().collect();

        (distances, parent)
    }

    /// 串行BFS作为基准
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
}
mod parallel_merge_sort;
mod parallel_quick_sort;
mod parallel_matrix_multiplication;
mod monte_carlo_pi;
mod simple_bfs;

use std::time::Instant;
use std::collections::HashMap;
use ndarray::Array2;
use rayon::prelude::*;
use rand::Rng;

pub struct SimplePerformanceTester;

#[derive(Debug, Clone)]
pub struct TestResult {
    pub algorithm: String,
    pub category: String,
    pub data_size: usize,
    pub time_ms: f64,
}

impl SimplePerformanceTester {
    pub fn new() -> Self {
        Self
    }

    pub fn run_basic_tests(&self) {
        println!("=== Rust并行算法基础性能测试 ===");
        println!("CPU核心数: {}", rayon::current_num_threads());
        println!();

        // 测试排序算法
        self.test_sorting();

        // 测试矩阵乘法
        self.test_matrix_multiplication();

        // 测试蒙特卡洛π计算
        self.test_monte_carlo_pi();

        // 测试BFS
        self.test_bfs();
    }

    fn test_sorting(&self) {
        println!("=== 排序算法测试 ===");

        let data_sizes = [10_000, 100_000];
        let num_tests = 1;

        for size in data_sizes {
            println!("\n数据大小: {}", size);

            // 生成测试数据
            let test_data = self.generate_random_data(size);

            let algorithms = [
                ("串行归并排序", Box::new(|data: &mut Vec<i32>| {
                    parallel_merge_sort::ParallelMergeSort::sequential_merge_sort(data);
                }) as Box<dyn Fn(&mut Vec<i32>)>),
                ("并行归并排序", Box::new(|data: &mut Vec<i32>| {
                    parallel_merge_sort::ParallelMergeSort::parallel_merge_sort(data);
                })),
                ("串行快速排序", Box::new(|data: &mut Vec<i32>| {
                    parallel_quick_sort::ParallelQuickSort::sequential_quick_sort(data);
                })),
                ("并行快速排序", Box::new(|data: &mut Vec<i32>| {
                    parallel_quick_sort::ParallelQuickSort::parallel_quick_sort(data);
                })),
            ];

            for (name, algorithm) in algorithms {
                let times = self.benchmark_algorithm(&test_data, algorithm, num_tests);
                let avg_time = times.iter().sum::<f64>() / times.len() as f64;

                println!("  {}: {:.2}ms", name, avg_time);
            }
        }
    }

    fn test_matrix_multiplication(&self) {
        println!("\n=== 矩阵乘法测试 ===");

        let matrix_sizes = [128, 256];
        let num_tests = 1;

        for size in matrix_sizes {
            println!("\n矩阵大小: {}x{}", size, size);

            // 生成测试矩阵
            let a = self.generate_random_matrix(size);
            let b = self.generate_random_matrix(size);

            let algorithms = [
                ("串行矩阵乘法", Box::new(|a: &Array2<f64>, b: &Array2<f64>| {
                    parallel_matrix_multiplication::ParallelMatrixMultiplication::sequential_multiply(a, b)
                }) as Box<dyn Fn(&Array2<f64>, &Array2<f64>) -> Array2<f64>>),
                ("并行矩阵乘法", Box::new(|a: &Array2<f64>, b: &Array2<f64>| {
                    parallel_matrix_multiplication::ParallelMatrixMultiplication::rayon_multiply(a, b)
                })),
            ];

            for (name, algorithm) in algorithms {
                let times = self.benchmark_matrix_algorithm(&a, &b, algorithm, num_tests);
                let avg_time = times.iter().sum::<f64>() / times.len() as f64;

                println!("  {}: {:.2}ms", name, avg_time);
            }
        }
    }

    fn test_monte_carlo_pi(&self) {
        println!("\n=== 蒙特卡洛π计算测试 ===");

        let sample_sizes = [1_000_000, 10_000_000];
        let num_tests = 1;

        for samples in sample_sizes {
            println!("\n样本数量: {}", samples);

            let algorithms = [
                ("串行蒙特卡洛π", Box::new(|n: u64| {
                    monte_carlo_pi::MonteCarloPi::sequential_monte_carlo_pi(n)
                }) as Box<dyn Fn(u64) -> f64>),
                ("并行蒙特卡洛π", Box::new(|n: u64| {
                    monte_carlo_pi::MonteCarloPi::parallel_monte_carlo_pi(n)
                })),
            ];

            for (name, algorithm) in algorithms {
                let times = self.benchmark_pi_algorithm(samples, algorithm, num_tests);
                let avg_time = times.iter().sum::<f64>() / times.len() as f64;

                let pi_estimate = algorithm(samples);
                let accuracy = (pi_estimate - std::f64::consts::PI).abs();

                println!("  {}: {:.2}ms (π误差: {:.6})", name, avg_time, accuracy);
            }
        }
    }

    fn test_bfs(&self) {
        println!("\n=== BFS图算法测试 ===");

        let graph_sizes = [10_000, 50_000];
        let num_tests = 1;

        for size in graph_sizes {
            println!("\n图大小: {} 个节点", size);

            // 生成随机图
            let graph = self.generate_random_graph(size);

            let algorithms = [
                ("串行BFS", Box::new(|g: &simple_bfs::Graph, start: usize| {
                    simple_bfs::SimpleParallelBFS::sequential_bfs(g, start)
                }) as Box<dyn Fn(&simple_bfs::Graph, usize) -> (HashMap<usize, usize>, HashMap<usize, Option<usize>>)>),
                ("并行BFS", Box::new(|g: &simple_bfs::Graph, start: usize| {
                    simple_bfs::SimpleParallelBFS::simple_parallel_bfs(g, start)
                })),
            ];

            for (name, algorithm) in algorithms {
                let times = self.benchmark_graph_algorithm(&graph, algorithm, num_tests);
                let avg_time = times.iter().sum::<f64>() / times.len() as f64;

                println!("  {}: {:.2}ms", name, avg_time);
            }
        }
    }

    fn benchmark_algorithm<F>(&self, data: &[i32], algorithm: F, num_tests: usize) -> Vec<f64>
    where
        F: Fn(&mut Vec<i32>) + Sync,
    {
        let mut times = Vec::new();

        for _ in 0..num_tests {
            let mut test_data = data.to_vec();
            let start = Instant::now();
            algorithm(&mut test_data);
            let duration = start.elapsed();
            times.push(duration.as_secs_f64() * 1000.0); // 转换为毫秒
        }

        times
    }

    fn benchmark_matrix_algorithm<F>(&self, a: &Array2<f64>, b: &Array2<f64>, algorithm: F, num_tests: usize) -> Vec<f64>
    where
        F: Fn(&Array2<f64>, &Array2<f64>) -> Array2<f64> + Sync,
    {
        let mut times = Vec::new();

        for _ in 0..num_tests {
            let start = Instant::now();
            let _result = algorithm(a, b);
            let duration = start.elapsed();
            times.push(duration.as_secs_f64() * 1000.0);
        }

        times
    }

    fn benchmark_pi_algorithm<F>(&self, samples: u64, algorithm: F, num_tests: usize) -> Vec<f64>
    where
        F: Fn(u64) -> f64 + Sync,
    {
        let mut times = Vec::new();

        for _ in 0..num_tests {
            let start = Instant::now();
            let _result = algorithm(samples);
            let duration = start.elapsed();
            times.push(duration.as_secs_f64() * 1000.0);
        }

        times
    }

    fn benchmark_graph_algorithm<F>(&self, graph: &simple_bfs::Graph, algorithm: F, num_tests: usize) -> Vec<f64>
    where
        F: Fn(&simple_bfs::Graph, usize) -> (HashMap<usize, usize>, HashMap<usize, Option<usize>>) + Sync,
    {
        let mut times = Vec::new();

        for _ in 0..num_tests {
            let start = Instant::now();
            let _result = algorithm(graph, 0);
            let duration = start.elapsed();
            times.push(duration.as_secs_f64() * 1000.0);
        }

        times
    }

    fn generate_random_data(&self, size: usize) -> Vec<i32> {
        let mut rng = rand::thread_rng();
        (0..size).map(|_| rng.gen_range(1..1000)).collect()
    }

    fn generate_random_matrix(&self, size: usize) -> Array2<f64> {
        Array2::from_shape_fn((size, size), |_| rand::random::<f64>())
    }

    fn generate_random_graph(&self, size: usize) -> simple_bfs::Graph {
        let mut graph = simple_bfs::Graph::new();
        let mut rng = rand::thread_rng();

        for i in 0..size {
            let neighbors: Vec<usize> = (0..size)
                .filter(|&j| i != j && rng.gen_bool(0.1))
                .collect();
            graph.insert(i, neighbors);
        }

        graph
    }
}

fn main() {
    println!("开始Rust并行算法基础性能测试...");

    let tester = SimplePerformanceTester::new();
    tester.run_basic_tests();

    println!("\n=== 测试完成 ===");
}
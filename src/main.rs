use std::time::Instant;
use rayon::prelude::*;
use rand::Rng;

fn main() {
    println!("=== Rust并行算法基础测试 ===");
    println!("CPU核心数: {}", rayon::current_num_threads());

    // 测试1: 并行归并排序
    println!("\n1. 并行归并排序测试:");
    let size = 100_000;
    let mut data = generate_random_data(size);
    let mut data_copy = data.clone();

    let start = Instant::now();
    data_copy.par_sort();
    let parallel_time = start.elapsed().as_millis();

    let start = Instant::now();
    data.sort();
    let sequential_time = start.elapsed().as_millis();

    let speedup = sequential_time as f64 / parallel_time as f64;
    println!("  数据大小: {}", size);
    println!("  并行时间: {}ms", parallel_time);
    println!("  串行时间: {}ms", sequential_time);
    println!("  加速比: {:.2}x", speedup);

    // 测试2: 并行矩阵乘法
    println!("\n2. 并行矩阵乘法测试:");
    let matrix_size = 200;
    let a = generate_random_matrix(matrix_size);
    let b = generate_random_matrix(matrix_size);

    let start = Instant::now();
    let parallel_result = parallel_matrix_multiply(&a, &b);
    let parallel_time = start.elapsed().as_millis();

    let start = Instant::now();
    let sequential_result = sequential_matrix_multiply(&a, &b);
    let sequential_time = start.elapsed().as_millis();

    let speedup = sequential_time as f64 / parallel_time as f64;
    println!("  矩阵大小: {}x{}", matrix_size, matrix_size);
    println!("  并行时间: {}ms", parallel_time);
    println!("  串行时间: {}ms", sequential_time);
    println!("  加速比: {:.2}x", speedup);

    // 测试3: 并行蒙特卡洛π计算
    println!("\n3. 并行蒙特卡洛π计算测试:");
    let samples = 10_000_000;

    let start = Instant::now();
    let parallel_pi = parallel_monte_carlo_pi(samples);
    let parallel_time = start.elapsed().as_millis();

    let start = Instant::now();
    let sequential_pi = sequential_monte_carlo_pi(samples);
    let sequential_time = start.elapsed().as_millis();

    let speedup = sequential_time as f64 / parallel_time as f64;
    let parallel_error = (parallel_pi - std::f64::consts::PI).abs();
    let sequential_error = (sequential_pi - std::f64::consts::PI).abs();

    println!("  样本数量: {}", samples);
    println!("  并行时间: {}ms, π值: {:.6}, 误差: {:.6}", parallel_time, parallel_pi, parallel_error);
    println!("  串行时间: {}ms, π值: {:.6}, 误差: {:.6}", sequential_time, sequential_pi, sequential_error);
    println!("  加速比: {:.2}x", speedup);

    println!("\n=== 测试完成 ===");
}

fn generate_random_data(size: usize) -> Vec<i32> {
    let mut rng = rand::thread_rng();
    (0..size).map(|_| rng.gen_range(1..1000)).collect()
}

fn generate_random_matrix(size: usize) -> Vec<Vec<f64>> {
    let mut rng = rand::thread_rng();
    (0..size)
        .map(|_| (0..size).map(|_| rng.gen_range(0.0..1.0)).collect())
        .collect()
}

fn parallel_matrix_multiply(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut result = vec![vec![0.0; n]; n];

    result.par_iter_mut().enumerate().for_each(|(i, row)| {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a[i][k] * b[k][j];
            }
            row[j] = sum;
        }
    });

    result
}

fn sequential_matrix_multiply(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut result = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a[i][k] * b[k][j];
            }
            result[i][j] = sum;
        }
    }

    result
}

fn parallel_monte_carlo_pi(samples: u64) -> f64 {
    let inside: u64 = (0..samples)
        .into_par_iter()
        .map(|_| {
            let x: f64 = rand::random::<f64>() * 2.0 - 1.0;
            let y: f64 = rand::random::<f64>() * 2.0 - 1.0;
            if x * x + y * y <= 1.0 { 1 } else { 0 }
        })
        .sum();

    4.0 * (inside as f64) / (samples as f64)
}

fn sequential_monte_carlo_pi(samples: u64) -> f64 {
    let mut inside = 0u64;

    for _ in 0..samples {
        let x: f64 = rand::random::<f64>() * 2.0 - 1.0;
        let y: f64 = rand::random::<f64>() * 2.0 - 1.0;
        if x * x + y * y <= 1.0 {
            inside += 1;
        }
    }

    4.0 * (inside as f64) / (samples as f64)
}
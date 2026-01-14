use rayon::prelude::*;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

pub struct MonteCarloPi;

impl MonteCarloPi {
    /// 串行蒙特卡洛π值计算
    pub fn sequential_monte_carlo_pi(num_samples: u64) -> f64 {
        let mut rng = rand::thread_rng();
        let mut inside_circle = 0u64;

        for _ in 0..num_samples {
            let x: f64 = rng.gen_range(-1.0..=1.0);
            let y: f64 = rng.gen_range(-1.0..=1.0);
            if x*x + y*y <= 1.0 {
                inside_circle += 1;
            }
        }

        4.0 * (inside_circle as f64) / (num_samples as f64)
    }

    /// 使用线程池的并行蒙特卡洛π值计算
    pub fn parallel_monte_carlo_pi(num_samples: u64) -> f64 {
        let num_threads = rayon::current_num_threads();
        let samples_per_thread = num_samples / num_threads as u64;
        let remainder = num_samples % num_threads as u64;

        let results: Vec<u64> = (0..num_threads)
            .into_par_iter()
            .map(|thread_id| {
                let mut local_samples = samples_per_thread;
                if thread_id == num_threads - 1 {
                    local_samples += remainder; // 最后一个线程处理余数
                }

                // 为每个线程设置不同的随机种子
                let seed = thread_id as u64 * 1000 + std::process::id() as u64;
                let mut rng = StdRng::seed_from_u64(seed);

                let mut inside = 0u64;
                for _ in 0..local_samples {
                    let x: f64 = rng.gen_range(-1.0..=1.0);
                    let y: f64 = rng.gen_range(-1.0..=1.0);
                    if x*x + y*y <= 1.0 {
                        inside += 1;
                    }
                }
                inside
            })
            .collect();

        let total_inside: u64 = results.iter().sum();
        4.0 * (total_inside as f64) / (num_samples as f64)
    }

    /// 使用原子操作的并行蒙特卡洛π值计算
    pub fn atomic_monte_carlo_pi(num_samples: u64) -> f64 {
        let inside_counter = Arc::new(AtomicU64::new(0));
        let num_threads = rayon::current_num_threads();
        let samples_per_thread = num_samples / num_threads as u64;
        let remainder = num_samples % num_threads as u64;

        rayon::scope(|s| {
            for thread_id in 0..num_threads {
                let counter = Arc::clone(&inside_counter);
                let mut local_samples = samples_per_thread;
                if thread_id == num_threads - 1 {
                    local_samples += remainder;
                }

                s.spawn(move |_| {
                    let seed = thread_id as u64 * 1000 + std::process::id() as u64;
                    let mut rng = StdRng::seed_from_u64(seed);

                    let mut local_inside = 0u64;
                    for _ in 0..local_samples {
                        let x: f64 = rng.gen_range(-1.0..=1.0);
                        let y: f64 = rng.gen_range(-1.0..=1.0);
                        if x*x + y*y <= 1.0 {
                            local_inside += 1;
                        }
                    }

                    // 原子操作更新计数器
                    counter.fetch_add(local_inside, Ordering::Relaxed);
                });
            }
        });

        let total_inside = inside_counter.load(Ordering::Relaxed);
        4.0 * (total_inside as f64) / (num_samples as f64)
    }

    /// 向量化蒙特卡洛π值计算
    pub fn vectorized_monte_carlo_pi(num_samples: u64) -> f64 {
        const VEC_SIZE: usize = 1024;
        let mut inside_total = 0u64;

        for chunk_start in (0..num_samples).step_by(VEC_SIZE as u64) {
            let chunk_size = std::cmp::min(VEC_SIZE as u64, num_samples - chunk_start);

            // 生成批量随机数
            let mut rng = rand::thread_rng();
            let mut points = Vec::with_capacity(chunk_size as usize);

            for _ in 0..chunk_size {
                let x: f64 = rng.gen_range(-1.0..=1.0);
                let y: f64 = rng.gen_range(-1.0..=1.0);
                points.push((x, y));
            }

            // 向量化计算
            let inside_chunk: u64 = points
                .par_iter()
                .map(|&(x, y)| if x*x + y*y <= 1.0 { 1 } else { 0 })
                .sum();

            inside_total += inside_chunk;
        }

        4.0 * (inside_total as f64) / (num_samples as f64)
    }

    /// 并行向量化蒙特卡洛π值计算
    pub fn parallel_vectorized_monte_carlo_pi(num_samples: u64) -> f64 {
        const VEC_SIZE: usize = 1024;
        const CHUNKS_PER_THREAD: usize = 16;

        let num_threads = rayon::current_num_threads();
        let total_chunks = (num_samples as usize + VEC_SIZE - 1) / VEC_SIZE;
        let chunks_per_thread = (total_chunks + num_threads - 1) / num_threads;

        let results: Vec<u64> = (0..num_threads)
            .into_par_iter()
            .map(|thread_id| {
                let start_chunk = thread_id * chunks_per_thread;
                let end_chunk = std::cmp::min(start_chunk + chunks_per_thread, total_chunks);

                let mut local_inside = 0u64;
                let seed = thread_id as u64 * 1000 + std::process::id() as u64;
                let mut rng = StdRng::seed_from_u64(seed);

                for chunk_idx in start_chunk..end_chunk {
                    let chunk_start = chunk_idx * VEC_SIZE;
                    let chunk_size = std::cmp::min(VEC_SIZE, num_samples as usize - chunk_start);

                    // 生成批量随机数
                    let points: Vec<(f64, f64)> = (0..chunk_size)
                        .map(|_| {
                            let x: f64 = rng.gen_range(-1.0..=1.0);
                            let y: f64 = rng.gen_range(-1.0..=1.0);
                            (x, y)
                        })
                        .collect();

                    // 向量化计算
                    let inside_chunk: u64 = points
                        .par_iter()
                        .map(|&(x, y)| if x*x + y*y <= 1.0 { 1 } else { 0 })
                        .sum();

                    local_inside += inside_chunk;
                }

                local_inside
            })
            .collect();

        let total_inside: u64 = results.iter().sum();
        4.0 * (total_inside as f64) / (num_samples as f64)
    }

    /// 带置信区间的π值估计
    pub fn estimate_pi_with_confidence(num_samples: u64, confidence_level: f64) -> (f64, f64) {
        let pi_estimate = Self::parallel_monte_carlo_pi(num_samples);

        // 计算标准误差
        let p = pi_estimate / 4.0; // 圆内点的比例
        let variance = p * (1.0 - p);
        let standard_error = (variance / num_samples as f64).sqrt();

        // 计算置信区间
        let z_score = match confidence_level {
            0.90 => 1.645,
            0.95 => 1.96,
            0.99 => 2.576,
            _ => 1.96, // 默认95%
        };

        let margin_of_error = z_score * standard_error * 4.0; // 乘以4转换为π的误差

        (pi_estimate, margin_of_error)
    }

    /// 自适应蒙特卡洛π值计算
    pub fn adaptive_monte_carlo_pi(target_error: f64, max_samples: u64) -> (f64, f64, u64) {
        let mut samples = 1000u64;
        let mut pi_estimate = 0.0;
        let mut error = f64::INFINITY;

        while error > target_error && samples < max_samples {
            // 增加样本数量
            let additional_samples = std::cmp::min(10000, max_samples - samples);
            samples += additional_samples;

            // 计算新的π值
            let new_pi = Self::parallel_monte_carlo_pi(additional_samples);
            pi_estimate = (pi_estimate * (samples - additional_samples) as f64 +
                          new_pi * additional_samples as f64) / samples as f64;

            // 估算误差
            let p = pi_estimate / 4.0;
            let variance = p * (1.0 - p);
            let standard_error = (variance / samples as f64).sqrt();
            error = 1.96 * standard_error * 4.0;
        }

        (pi_estimate, error, samples)
    }

    /// 分层抽样蒙特卡洛π值计算
    pub fn stratified_monte_carlo_pi(num_samples: u64, num_strata: u32) -> f64 {
        let samples_per_stratum = num_samples / num_strata as u64;
        let remainder = num_samples % num_strata as u64;

        let results: Vec<u64> = (0..num_strata)
            .into_par_iter()
            .map(|stratum_id| {
                let mut local_samples = samples_per_stratum;
                if stratum_id == num_strata - 1 {
                    local_samples += remainder;
                }

                let seed = stratum_id as u64 * 1000 + std::process::id() as u64;
                let mut rng = StdRng::seed_from_u64(seed);

                // 分层抽样：将[-1,1]区间分成strata个子区间
                let stratum_width = 2.0 / num_strata as f64;
                let stratum_start = -1.0 + stratum_id as f64 * stratum_width;
                let stratum_end = stratum_start + stratum_width;

                let mut inside = 0u64;
                for _ in 0..local_samples {
                    // 在当前分层内均匀采样
                    let x = rng.gen_range(stratum_start..=stratum_end);
                    let y = rng.gen_range(-1.0..=1.0);

                    if x*x + y*y <= 1.0 {
                        // 检查是否在圆内且在当前分层
                        inside += 1;
                    }
                }
                inside
            })
            .collect();

        let total_inside: u64 = results.iter().sum();
        4.0 * (total_inside as f64) / (num_samples as f64)
    }

    /// 重要性采样蒙特卡洛π值计算
    pub fn importance_sampling_monte_carlo_pi(num_samples: u64) -> f64 {
        let mut rng = rand::thread_rng();

        // 使用重要性采样：在圆内密集采样，在圆外稀疏采样
        let mut inside_total = 0u64;
        let mut total_samples = 0u64;

        while total_samples < num_samples {
            // 在[0,1]内采样，然后转换到[-1,1]
            let r: f64 = rng.gen_range(0.0..=1.0);
            let theta: f64 = rng.gen_range(0.0..=2.0 * std::f64::consts::PI);

            // 极坐标转换
            let x = r * theta.cos();
            let y = r * theta.sin();

            if x*x + y*y <= 1.0 {
                inside_total += 1;
            }
            total_samples += 1;
        }

        4.0 * (inside_total as f64) / (total_samples as f64)
    }

    /// 低差异序列蒙特卡洛π值计算（使用Sobol序列）
    pub fn sobol_monte_carlo_pi(num_samples: u64) -> f64 {
        let mut inside_total = 0u64;

        // 简化的Sobol序列生成器
        let mut x = 0.5;
        let mut y = 0.5;

        for i in 0..num_samples {
            // Sobol序列的简单实现
            let gray_code = i ^ (i >> 1);
            x = (gray_code as f64 / u64::MAX as f64) * 2.0 - 1.0;
            y = ((gray_code.rotate_left(16) as f64 / u64::MAX as f64)) * 2.0 - 1.0;

            if x*x + y*y <= 1.0 {
                inside_total += 1;
            }
        }

        4.0 * (inside_total as f64) / (num_samples as f64)
    }

    /// 多重蒙特卡洛估计（减少方差）
    pub fn multiple_monte_carlo_pi(num_samples: u64, num_estimates: u32) -> (f64, f64) {
        let estimates: Vec<f64> = (0..num_estimates)
            .into_par_iter()
            .map(|_| Self::parallel_monte_carlo_pi(num_samples / num_estimates as u64))
            .collect();

        let mean = estimates.iter().sum::<f64>() / num_estimates as f64;

        // 计算标准差
        let variance = estimates
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / (num_estimates - 1) as f64;

        let std_dev = variance.sqrt();
        let standard_error = std_dev / (num_estimates as f64).sqrt();

        (mean, standard_error)
    }

    /// GPU风格的并行蒙特卡洛（模拟CUDA/OpenCL风格）
    pub fn gpu_style_monte_carlo_pi(num_samples: u64) -> f64 {
        const WORK_GROUP_SIZE: usize = 256;
        const WORK_GROUPS: usize = 64;

        let total_threads = WORK_GROUP_SIZE * WORK_GROUPS;
        let samples_per_thread = num_samples / total_threads as u64;
        let remainder = num_samples % total_threads as u64;

        let results: Vec<u64> = (0..total_threads)
            .into_par_iter()
            .map(|thread_id| {
                let mut local_samples = samples_per_thread;
                if thread_id == total_threads - 1 {
                    local_samples += remainder;
                }

                // 模拟GPU线程ID计算
                let work_group_id = thread_id / WORK_GROUP_SIZE;
                let local_thread_id = thread_id % WORK_GROUP_SIZE;
                let seed = (work_group_id as u64) * 1000 + (local_thread_id as u64) * 10 + std::process::id() as u64;

                let mut rng = StdRng::seed_from_u64(seed);

                let mut inside = 0u64;
                for _ in 0..local_samples {
                    let x: f64 = rng.gen_range(-1.0..=1.0);
                    let y: f64 = rng.gen_range(-1.0..=1.0);
                    if x*x + y*y <= 1.0 {
                        inside += 1;
                    }
                }
                inside
            })
            .collect();

        let total_inside: u64 = results.iter().sum();
        4.0 * (total_inside as f64) / (num_samples as f64)
    }
}

/// 创建随机数生成器的辅助函数
fn create_rng_with_seed(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

/// 计算圆内点的辅助函数
fn is_inside_circle(x: f64, y: f64) -> bool {
    x*x + y*y <= 1.0
}

/// 批量计算圆内点的辅助函数
fn count_inside_circle_batch(points: &[(f64, f64)]) -> u64 {
    points
        .par_iter()
        .map(|&(x, y)| if is_inside_circle(x, y) { 1 } else { 0 })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_monte_carlo_pi() {
        let pi = MonteCarloPi::sequential_monte_carlo_pi(1000000);
        let error = (pi - std::f64::consts::PI).abs();
        assert!(error < 0.01, "π估计值误差过大: {}", error);
    }

    #[test]
    fn test_parallel_monte_carlo_pi() {
        let pi = MonteCarloPi::parallel_monte_carlo_pi(1000000);
        let error = (pi - std::f64::consts::PI).abs();
        assert!(error < 0.01, "π估计值误差过大: {}", error);
    }

    #[test]
    fn test_stratified_monte_carlo_pi() {
        let pi = MonteCarloPi::stratified_monte_carlo_pi(1000000, 10);
        let error = (pi - std::f64::consts::PI).abs();
        assert!(error < 0.01, "π估计值误差过大: {}", error);
    }

    #[test]
    fn test_adaptive_monte_carlo_pi() {
        let (pi, error, samples) = MonteCarloPi::adaptive_monte_carlo_pi(0.001, 10000000);
        assert!(error <= 0.001, "自适应算法未达到目标误差");
        assert!(samples <= 10000000, "样本数超过最大限制");
    }

    #[test]
    fn test_confidence_interval() {
        let (pi, margin) = MonteCarloPi::estimate_pi_with_confidence(1000000, 0.95);
        let error = (pi - std::f64::consts::PI).abs();
        assert!(error < margin, "真实误差超出置信区间");
    }
}
use rayon::prelude::*;
use ndarray::{Array2, ArrayView2, s};
use std::sync::Arc;

pub struct ParallelMatrixMultiplication;

impl ParallelMatrixMultiplication {
    /// 串行矩阵乘法实现
    pub fn sequential_multiply(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
        let (rows_a, cols_a) = a.dim();
        let (rows_b, cols_b) = b.dim();

        assert_eq!(cols_a, rows_b, "矩阵维度不匹配");

        let mut result = Array2::<f64>::zeros((rows_a, cols_b));

        for i in 0..rows_a {
            for j in 0..cols_b {
                let mut sum = 0.0;
                for k in 0..cols_a {
                    sum += a[[i, k]] * b[[k, j]];
                }
                result[[i, j]] = sum;
            }
        }

        result
    }

    /// 并行矩阵乘法 - 按行分块
    pub fn parallel_multiply_by_rows(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
        let (rows_a, cols_a) = a.dim();
        let (rows_b, cols_b) = b.dim();

        assert_eq!(cols_a, rows_b, "矩阵维度不匹配");

        let result = Array2::<f64>::from_shape_fn((rows_a, cols_b), |(i, j)| {
            let mut sum = 0.0;
            for k in 0..cols_a {
                sum += a[[i, k]] * b[[k, j]];
            }
            sum
        });

        result
    }

    /// 并行矩阵乘法 - 使用Rayon
    pub fn rayon_multiply(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
        let (rows_a, cols_a) = a.dim();
        let (rows_b, cols_b) = b.dim();

        assert_eq!(cols_a, rows_b, "矩阵维度不匹配");

        let result = Array2::<f64>::from_shape_fn((rows_a, cols_b), move |(i, j)| {
            let mut sum = 0.0;
            for k in 0..cols_a {
                sum += a[[i, k]] * b[[k, j]];
            }
            sum
        });

        result
    }

    /// 分块矩阵乘法
    pub fn block_multiply(a: &Array2<f64>, b: &Array2<f64>, block_size: usize) -> Array2<f64> {
        let (rows_a, cols_a) = a.dim();
        let (rows_b, cols_b) = b.dim();

        assert_eq!(cols_a, rows_b, "矩阵维度不匹配");

        let mut result = Array2::<f64>::zeros((rows_a, cols_b));

        // 分块计算
        for i in (0..rows_a).step_by(block_size) {
            let i_end = (i + block_size).min(rows_a);

            for j in (0..cols_b).step_by(block_size) {
                let j_end = (j + block_size).min(cols_b);

                for k in (0..cols_a).step_by(block_size) {
                    let k_end = (k + block_size).min(cols_a);

                    // 计算当前块
                    for ii in i..i_end {
                        for jj in j..j_end {
                            let mut sum = 0.0;
                            for kk in k..k_end {
                                sum += a[[ii, kk]] * b[[kk, jj]];
                            }
                            result[[ii, jj]] += sum;
                        }
                    }
                }
            }
        }

        result
    }

    /// 并行分块矩阵乘法
    pub fn parallel_block_multiply(a: &Array2<f64>, b: &Array2<f64>, block_size: usize) -> Array2<f64> {
        let (rows_a, cols_a) = a.dim();
        let (rows_b, cols_b) = b.dim();

        assert_eq!(cols_a, rows_b, "矩阵维度不匹配");

        let mut result = Array2::<f64>::zeros((rows_a, cols_b));

        // 将结果矩阵的行分配给不同线程
        let num_threads = rayon::current_num_threads();
        let rows_per_thread = rows_a / num_threads;

        (0..num_threads).into_par_iter().for_each(|t| {
            let start_row = t * rows_per_thread;
            let end_row = if t == num_threads - 1 {
                rows_a
            } else {
                start_row + rows_per_thread
            };

            // 计算当前线程负责的行
            for i in start_row..end_row {
                for j in 0..cols_b {
                    let mut sum = 0.0;
                    for k in 0..cols_a {
                        sum += a[[i, k]] * b[[k, j]];
                    }
                    // 原子操作更新结果
                    unsafe {
                        *result.uget_mut((i, j)) = sum;
                    }
                }
            }
        });

        result
    }

    /// Strassen算法矩阵乘法
    pub fn strassen_multiply(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
        let (rows_a, cols_a) = a.dim();
        let (rows_b, cols_b) = b.dim();

        assert_eq!(cols_a, rows_b, "矩阵维度不匹配");
        assert_eq!(rows_a, cols_a, "Strassen算法要求方阵");
        assert_eq!(rows_b, cols_b, "Strassen算法要求方阵");

        if rows_a <= 64 {
            return Self::sequential_multiply(a, b);
        }

        // 确保矩阵大小是2的幂
        let n = rows_a;
        let new_n = if n.is_power_of_two() {
            n
        } else {
            n.next_power_of_two()
        };

        if new_n != n {
            let a_padded = Self::pad_matrix(a, new_n);
            let b_padded = Self::pad_matrix(b, new_n);
            let result_padded = Self::strassen_multiply(&a_padded, &b_padded);
            return Self::unpad_matrix(&result_padded, n);
        }

        // 分割矩阵
        let mid = n / 2;

        let a11 = a.slice(s![0..mid, 0..mid]);
        let a12 = a.slice(s![0..mid, mid..]);
        let a21 = a.slice(s![mid.., 0..mid]);
        let a22 = a.slice(s![mid.., mid..]);

        let b11 = b.slice(s![0..mid, 0..mid]);
        let b12 = b.slice(s![0..mid, mid..]);
        let b21 = b.slice(s![mid.., 0..mid]);
        let b22 = b.slice(s![mid.., mid..]);

        // 并行计算7个乘积
        let p1 = rayon::spawn(move || {
            let temp = &b12 - &b22;
            Self::strassen_multiply(&a11, &temp)
        });

        let p2 = rayon::spawn(move || {
            let temp = &a11 + &a12;
            Self::strassen_multiply(&temp, &b22)
        });

        let p3 = rayon::spawn(move || {
            let temp = &a21 + &a22;
            Self::strassen_multiply(&temp, &b11)
        });

        let p4 = rayon::spawn(move || {
            let temp = &b21 - &b11;
            Self::strassen_multiply(&a22, &temp)
        });

        let p5 = rayon::spawn(move || {
            let temp1 = &a11 + &a22;
            let temp2 = &b11 + &b22;
            Self::strassen_multiply(&temp1, &temp2)
        });

        let p6 = rayon::spawn(move || {
            let temp1 = &a12 - &a22;
            let temp2 = &b21 + &b22;
            Self::strassen_multiply(&temp1, &temp2)
        });

        let p7 = rayon::spawn(move || {
            let temp1 = &a11 - &a21;
            let temp2 = &b11 + &b12;
            Self::strassen_multiply(&temp1, &temp2)
        });

        // 收集结果
        let p1 = p1.join();
        let p2 = p2.join();
        let p3 = p3.join();
        let p4 = p4.join();
        let p5 = p5.join();
        let p6 = p6.join();
        let p7 = p7.join();

        // 计算结果矩阵的四个部分
        let c11 = &p5 + &p4 - &p2 + &p6;
        let c12 = &p1 + &p2;
        let c21 = &p3 + &p4;
        let c22 = &p5 + &p1 - &p3 - &p7;

        // 合并结果
        Self::combine_matrices(&c11, &c12, &c21, &c22)
    }

    /// 填充矩阵到指定大小
    fn pad_matrix(matrix: &Array2<f64>, new_n: usize) -> Array2<f64> {
        let (n, _) = matrix.dim();
        let mut padded = Array2::<f64>::zeros((new_n, new_n));

        for i in 0..n {
            for j in 0..n {
                padded[[i, j]] = matrix[[i, j]];
            }
        }

        padded
    }

    /// 从填充矩阵中提取原始部分
    fn unpad_matrix(matrix: &Array2<f64>, original_n: usize) -> Array2<f64> {
        matrix.slice(s![0..original_n, 0..original_n]).to_owned()
    }

    /// 合并四个子矩阵
    fn combine_matrices(c11: &Array2<f64>, c12: &Array2<f64>, c21: &Array2<f64>, c22: &Array2<f64>) -> Array2<f64> {
        let mid = c11.dim().0;
        let n = mid * 2;

        let mut result = Array2::<f64>::zeros((n, n));

        for i in 0..mid {
            for j in 0..mid {
                result[[i, j]] = c11[[i, j]];
                result[[i, j + mid]] = c12[[i, j]];
                result[[i + mid, j]] = c21[[i, j]];
                result[[i + mid, j + mid]] = c22[[i, j]];
            }
        }

        result
    }

    /// Winograd算法矩阵乘法 - 优化版本
    pub fn winograd_multiply(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
        let (rows_a, cols_a) = a.dim();
        let (rows_b, cols_b) = b.dim();

        assert_eq!(cols_a, rows_b, "矩阵维度不匹配");

        let mut result = Array2::<f64>::zeros((rows_a, cols_b));

        // 预计算行和列的和
        let row_factors = Self::compute_row_factors(a);
        let col_factors = Self::compute_col_factors(b);

        // 主要计算
        for i in 0..rows_a {
            for j in 0..cols_b {
                let mut sum = -row_factors[i] - col_factors[j];
                for k in 0..(cols_a / 2) {
                    sum += (a[[i, 2 * k]] + b[[2 * k + 1, j]]) *
                           (a[[i, 2 * k + 1]] + b[[2 * k, j]]);
                }
                result[[i, j]] = sum;
            }
        }

        // 处理奇数列
        if cols_a % 2 == 1 {
            let k = cols_a - 1;
            for i in 0..rows_a {
                for j in 0..cols_b {
                    result[[i, j]] += a[[i, k]] * b[[k, j]];
                }
            }
        }

        result
    }

    /// 计算行因子
    fn compute_row_factors(a: &Array2<f64>) -> Vec<f64> {
        let (rows, cols) = a.dim();
        let mut factors = vec![0.0; rows];

        for i in 0..rows {
            for k in (0..cols).step_by(2) {
                if k + 1 < cols {
                    factors[i] += a[[i, k]] * a[[i, k + 1]];
                }
            }
        }

        factors
    }

    /// 计算列因子
    fn compute_col_factors(b: &Array2<f64>) -> Vec<f64> {
        let (rows, cols) = b.dim();
        let mut factors = vec![0.0; cols];

        for j in 0..cols {
            for k in (0..rows).step_by(2) {
                if k + 1 < rows {
                    factors[j] += b[[k, j]] * b[[k + 1, j]];
                }
            }
        }

        factors
    }

    /// 并行Winograd算法
    pub fn parallel_winograd_multiply(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
        let (rows_a, cols_a) = a.dim();
        let (rows_b, cols_b) = b.dim();

        assert_eq!(cols_a, rows_b, "矩阵维度不匹配");

        let row_factors = Self::compute_row_factors(a);
        let col_factors = Self::compute_col_factors(b);

        let result = Array2::<f64>::from_shape_fn((rows_a, cols_b), |(i, j)| {
            let mut sum = -row_factors[i] - col_factors[j];
            for k in 0..(cols_a / 2) {
                sum += (a[[i, 2 * k]] + b[[2 * k + 1, j]]) *
                       (a[[i, 2 * k + 1]] + b[[2 * k, j]]);
            }
            sum
        });

        // 处理奇数列
        if cols_a % 2 == 1 {
            let k = cols_a - 1;
            let mut result_mut = result.into_owned();
            for i in 0..rows_a {
                for j in 0..cols_b {
                    result_mut[[i, j]] += a[[i, k]] * b[[k, j]];
                }
            }
            return result_mut;
        }

        result
    }

    /// 自适应矩阵乘法 - 根据矩阵大小选择算法
    pub fn adaptive_multiply(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
        let (rows_a, cols_a) = a.dim();
        let (rows_b, cols_b) = b.dim();

        assert_eq!(cols_a, rows_b, "矩阵维度不匹配");

        let n = rows_a;

        if n <= 64 {
            Self::sequential_multiply(a, b)
        } else if n <= 512 {
            Self::parallel_block_multiply(a, b, 64)
        } else if n.is_power_of_two() && n <= 2048 {
            Self::strassen_multiply(a, b)
        } else {
            Self::parallel_winograd_multiply(a, b)
        }
    }
}

/// 使用ndarray-linalg的BLAS优化矩阵乘法
pub fn blas_multiply(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    a.dot(b)
}

/// 流水线矩阵乘法 - 适用于矩阵链乘法
pub fn pipelined_multiply(matrices: &[Array2<f64>]) -> Option<Array2<f64>> {
    if matrices.is_empty() {
        return None;
    }

    if matrices.len() == 1 {
        return Some(matrices[0].clone());
    }

    let mut result = matrices[0].clone();

    for matrix in &matrices[1..] {
        result = ParallelMatrixMultiplication::adaptive_multiply(&result, matrix);
    }

    Some(result)
}

/// 分布式矩阵乘法 - 模拟分布式计算
pub fn distributed_multiply(a: &Array2<f64>, b: &Array2<f64>, num_partitions: usize) -> Array2<f64> {
    let (rows_a, cols_a) = a.dim();
    let (rows_b, cols_b) = b.dim();

    assert_eq!(cols_a, rows_b, "矩阵维度不匹配");

    let mut result = Array2::<f64>::zeros((rows_a, cols_b));

    // 将结果矩阵分块
    let rows_per_partition = rows_a / num_partitions;

    (0..num_partitions).into_par_iter().for_each(|p| {
        let start_row = p * rows_per_partition;
        let end_row = if p == num_partitions - 1 {
            rows_a
        } else {
            start_row + rows_per_partition
        };

        let local_result = Array2::<f64>::from_shape_fn((end_row - start_row, cols_b), |(i, j)| {
            let global_i = start_row + i;
            let mut sum = 0.0;
            for k in 0..cols_a {
                sum += a[[global_i, k]] * b[[k, j]];
            }
            sum
        });

        // 合并结果
        for i in 0..(end_row - start_row) {
            for j in 0..cols_b {
                unsafe {
                    *result.uget_mut((start_row + i, j)) = local_result[[i, j]];
                }
            }
        }
    });

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_sequential_multiply() {
        let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let b = arr2(&[[5.0, 6.0], [7.0, 8.0]]);
        let expected = arr2(&[[19.0, 22.0], [43.0, 50.0]]);

        let result = ParallelMatrixMultiplication::sequential_multiply(&a, &b);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_strassen_multiply() {
        let a = arr2(&[[1.0, 2.0, 3.0, 4.0],
                       [5.0, 6.0, 7.0, 8.0],
                       [9.0, 10.0, 11.0, 12.0],
                       [13.0, 14.0, 15.0, 16.0]]);
        let b = arr2(&[[1.0, 0.0, 1.0, 0.0],
                       [0.0, 1.0, 0.0, 1.0],
                       [1.0, 1.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0, 1.0]]);

        let expected = ParallelMatrixMultiplication::sequential_multiply(&a, &b);
        let result = ParallelMatrixMultiplication::strassen_multiply(&a, &b);

        // 由于浮点精度，使用近似比较
        assert!(result.abs_diff_eq(&expected, 1e-10));
    }

    #[test]
    fn test_large_matrix() {
        let n = 128;
        let a = Array2::<f64>::from_shape_fn((n, n), |(_, _)| rand::random::<f64>());
        let b = Array2::<f64>::from_shape_fn((n, n), |(_, _)| rand::random::<f64>());

        let expected = ParallelMatrixMultiplication::sequential_multiply(&a, &b);
        let result = ParallelMatrixMultiplication::adaptive_multiply(&a, &b);

        assert!(result.abs_diff_eq(&expected, 1e-10));
    }
}
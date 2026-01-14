use rayon::prelude::*;
use std::sync::Arc;
use std::ops::{Add, Mul};
use std::iter::Sum;

pub struct ParallelReduction;

impl ParallelReduction {
    /// 串行归约操作
    pub fn sequential_reduction<T, F>(data: &[T], operation: F) -> Option<T>
    where
        T: Clone + Send + Sync,
        F: Fn(&T, &T) -> T,
    {
        if data.is_empty() {
            return None;
        }

        let mut result = data[0].clone();
        for i in 1..data.len() {
            result = operation(&result, &data[i]);
        }

        Some(result)
    }

    /// 树形归约操作
    pub fn tree_reduction<T, F>(data: &[T], operation: F) -> Option<T>
    where
        T: Clone + Send + Sync,
        F: Fn(&T, &T) -> T + Sync,
    {
        if data.is_empty() {
            return None;
        }

        let mut current_data: Vec<T> = data.to_vec();

        while current_data.len() > 1 {
            let next_data: Vec<T> = current_data
                .par_chunks(2)
                .map(|chunk| {
                    if chunk.len() == 1 {
                        chunk[0].clone()
                    } else {
                        operation(&chunk[0], &chunk[1])
                    }
                })
                .collect();

            current_data = next_data;
        }

        Some(current_data[0].clone())
    }

    /// 并行求和
    pub fn parallel_sum<T>(data: &[T]) -> Option<T>
    where
        T: Add<Output = T> + Clone + Send + Sync + Default,
    {
        if data.is_empty() {
            return None;
        }

        Some(data.par_iter().cloned().reduce(|| T::default(), |a, b| a + b))
    }

    /// 并行乘积
    pub fn parallel_product<T>(data: &[T]) -> Option<T>
    where
        T: Mul<Output = T> + Clone + Send + Sync,
        T: std::iter::Product<T>,
    {
        if data.is_empty() {
            return None;
        }

        Some(data.par_iter().product())
    }

    /// 并行查找最大值
    pub fn parallel_max<T>(data: &[T]) -> Option<T>
    where
        T: Ord + Clone + Send + Sync,
    {
        if data.is_empty() {
            return None;
        }

        data.par_iter().max().cloned()
    }

    /// 并行查找最小值
    pub fn parallel_min<T>(data: &[T]) -> Option<T>
    where
        T: Ord + Clone + Send + Sync,
    {
        if data.is_empty() {
            return None;
        }

        data.par_iter().min().cloned()
    }

    /// 并行前缀和
    pub fn parallel_prefix_sum<T>(data: &[T]) -> Vec<T>
    where
        T: Add<Output = T> + Clone + Send + Sync + Default,
    {
        if data.is_empty() {
            return Vec::new();
        }

        let n = data.len();
        let mut result = vec![T::default(); n];

        // 第一个元素
        result[0] = data[0].clone();

        // 并行计算前缀和
        let chunk_size = std::cmp::max(1, n / rayon::current_num_threads());

        for i in (1..n).step_by(chunk_size) {
            let end = std::cmp::min(i + chunk_size, n);

            // 先计算每个块内的前缀和
            for j in i..end {
                result[j] = result[j - 1].clone() + data[j].clone();
            }

            // 如果不是第一个块，需要加上前一个块的最后一个元素
            if i > 1 {
                let carry = result[i - 1].clone();
                for j in i..end {
                    result[j] = result[j].clone() + carry.clone();
                }
            }
        }

        result
    }

    /// 并行扫描操作（前缀操作的通用版本）
    pub fn parallel_scan<T, F>(data: &[T], operation: F, identity: T) -> Vec<T>
    where
        T: Clone + Send + Sync,
        F: Fn(&T, &T) -> T + Sync,
    {
        if data.is_empty() {
            return Vec::new();
        }

        let n = data.len();
        let mut result = vec![identity.clone(); n];

        // 第一个元素
        result[0] = operation(&identity, &data[0]);

        // 并行计算扫描
        let chunk_size = std::cmp::max(1, n / rayon::current_num_threads());

        for i in (1..n).step_by(chunk_size) {
            let end = std::cmp::min(i + chunk_size, n);

            // 计算每个块内的扫描
            for j in i..end {
                result[j] = operation(&result[j - 1], &data[j]);
            }

            // 如果不是第一个块，需要传播前一个块的影响
            if i > 1 {
                let carry = result[i - 1].clone();
                for j in i..end {
                    result[j] = operation(&carry, &result[j]);
                }
            }
        }

        result
    }

    /// 分段归约操作
    pub fn segmented_reduction<T, F>(data: &[T], segment_size: usize, operation: F) -> Vec<T>
    where
        T: Clone + Send + Sync,
        F: Fn(&T, &T) -> T + Sync,
    {
        if data.is_empty() || segment_size == 0 {
            return Vec::new();
        }

        data.par_chunks(segment_size)
            .map(|chunk| {
                if chunk.len() == 1 {
                    chunk[0].clone()
                } else {
                    chunk[1..].iter().fold(chunk[0].clone(), |acc, x| operation(&acc, x))
                }
            })
            .collect()
    }

    /// 并行直方图计算
    pub fn parallel_histogram<T>(data: &[T], num_bins: usize) -> Vec<usize>
    where
        T: Ord + Clone + Send + Sync + Into<f64>,
    {
        if data.is_empty() || num_bins == 0 {
            return Vec::new();
        }

        // 找到数据范围
        let min_val = data.par_iter().min().unwrap().clone().into();
        let max_val = data.par_iter().max().unwrap().clone().into();
        let range = max_val - min_val;

        if range == 0.0 {
            let mut histogram = vec![0; num_bins];
            histogram[0] = data.len();
            return histogram;
        }

        let bin_width = range / num_bins as f64;

        // 并行计算每个元素的桶索引
        let bin_indices: Vec<usize> = data
            .par_iter()
            .map(|x| {
                let val = x.clone().into();
                let index = ((val - min_val) / bin_width) as usize;
                std::cmp::min(index, num_bins - 1)
            })
            .collect();

        // 统计每个桶的频数
        let mut histogram = vec![0; num_bins];
        for &index in &bin_indices {
            histogram[index] += 1;
        }

        histogram
    }

    /// 并行向量点积
    pub fn parallel_dot_product<T>(vector1: &[T], vector2: &[T]) -> Option<T>
    where
        T: Add<Output = T> + Mul<Output = T> + Default + Clone + Send + Sync,
    {
        if vector1.len() != vector2.len() || vector1.is_empty() {
            return None;
        }

        let result = vector1
            .par_iter()
            .zip(vector2.par_iter())
            .map(|(a, b)| a.clone() * b.clone())
            .sum();

        Some(result)
    }

    /// 并行向量范数计算
    pub fn parallel_norm<T>(vector: &[T]) -> Option<f64>
    where
        T: Mul<Output = T> + Add<Output = T> + Into<f64> + Clone + Send + Sync,
    {
        if vector.is_empty() {
            return None;
        }

        let sum_of_squares = vector
            .par_iter()
            .map(|x| {
                let val = x.clone().into();
                val * val
            })
            .sum::<f64>();

        Some(sum_of_squares.sqrt())
    }

    /// 并行查找最大值及其索引
    pub fn parallel_max_with_index<T>(data: &[T]) -> Option<(usize, T)>
    where
        T: Ord + Clone + Send + Sync,
    {
        if data.is_empty() {
            return None;
        }

        data.par_iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.cmp(b))
            .map(|(idx, val)| (idx, val.clone()))
    }

    /// 并行查找最小值及其索引
    pub fn parallel_min_with_index<T>(data: &[T]) -> Option<(usize, T)>
    where
        T: Ord + Clone + Send + Sync,
    {
        if data.is_empty() {
            return None;
        }

        data.par_iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.cmp(b))
            .map(|(idx, val)| (idx, val.clone()))
    }

    /// 并行累积操作
    pub fn parallel_cumulative_op<T, F>(data: &[T], operation: F) -> Vec<T>
    where
        T: Clone + Send + Sync,
        F: Fn(&T, &T) -> T + Sync,
    {
        if data.is_empty() {
            return Vec::new();
        }

        let n = data.len();
        let mut result = vec![data[0].clone(); n];

        // 并行计算累积值
        let chunk_size = std::cmp::max(1, n / rayon::current_num_threads());

        for i in (1..n).step_by(chunk_size) {
            let end = std::cmp::min(i + chunk_size, n);

            // 计算每个块内的累积值
            for j in i..end {
                result[j] = operation(&result[j - 1], &data[j]);
            }

            // 如果不是第一个块，需要传播前一个块的影响
            if i > 1 {
                let carry = result[i - 1].clone();
                for j in i..end {
                    result[j] = operation(&carry, &result[j]);
                }
            }
        }

        result
    }

    /// 并行归约到标量（带初始值）
    pub fn parallel_reduce_with_init<T, F>(data: &[T], init: T, operation: F) -> T
    where
        T: Clone + Send + Sync,
        F: Fn(&T, &T) -> T + Sync,
    {
        if data.is_empty() {
            return init;
        }

        data.par_iter().cloned().fold(|| init.clone(), |acc, x| operation(&acc, &x))
    }

    /// 并行归约到标量（无初始值）
    pub fn parallel_reduce<T, F>(data: &[T], operation: F) -> Option<T>
    where
        T: Clone + Send + Sync,
        F: Fn(&T, &T) -> T + Sync,
    {
        if data.is_empty() {
            return None;
        }

        Some(data.par_iter().cloned().reduce_with(operation).unwrap())
    }

    /// 并行条件计数
    pub fn parallel_count_if<T, F>(data: &[T], predicate: F) -> usize
    where
        T: Send + Sync,
        F: Fn(&T) -> bool + Sync,
    {
        data.par_iter().filter(|x| predicate(x)).count()
    }

    /// 并行条件求和
    pub fn parallel_sum_if<T, F>(data: &[T], predicate: F) -> T
    where
        T: Add<Output = T> + Default + Clone + Send + Sync,
        F: Fn(&T) -> bool + Sync,
    {
        data.par_iter()
            .filter(|x| predicate(x))
            .cloned()
            .sum()
    }

    /// 并行分组归约
    pub fn parallel_group_reduce<K, V, F>(
        data: &[(K, V)],
        group_fn: impl Fn(&K) -> usize + Sync,
        reduce_fn: F,
        num_groups: usize,
    ) -> Vec<V>
    where
        K: Clone + Send + Sync,
        V: Clone + Send + Sync,
        F: Fn(&V, &V) -> V + Sync,
        V: Default,
    {
        if data.is_empty() || num_groups == 0 {
            return vec![V::default(); num_groups];
        }

        let mut groups: Vec<Vec<V>> = vec![Vec::new(); num_groups];

        // 并行分组
        for &(ref key, ref value) in data {
            let group_idx = group_fn(key) % num_groups;
            groups[group_idx].push(value.clone());
        }

        // 并行归约每个组
        groups
            .into_par_iter()
            .map(|group| {
                if group.is_empty() {
                    V::default()
                } else {
                    group[1..].iter().fold(group[0].clone(), |acc, x| reduce_fn(&acc, x))
                }
            })
            .collect()
    }
}

/// 数值特征计算
pub mod numerical_features {
    use super::*;

    /// 计算均值
    pub fn mean<T>(data: &[T]) -> Option<f64>
    where
        T: Into<f64> + Clone + Send + Sync,
    {
        if data.is_empty() {
            return None;
        }

        let sum: f64 = data.par_iter().map(|x| x.clone().into()).sum();
        Some(sum / data.len() as f64)
    }

    /// 计算方差
    pub fn variance<T>(data: &[T]) -> Option<f64>
    where
        T: Into<f64> + Clone + Send + Sync,
    {
        let mean_val = mean(data)?;
        let n = data.len() as f64;

        let sum_sq_diff: f64 = data
            .par_iter()
            .map(|x| {
                let val = x.clone().into();
                (val - mean_val).powi(2)
            })
            .sum();

        Some(sum_sq_diff / n)
    }

    /// 计算标准差
    pub fn standard_deviation<T>(data: &[T]) -> Option<f64>
    where
        T: Into<f64> + Clone + Send + Sync,
    {
        variance(data).map(|var| var.sqrt())
    }

    /// 计算中位数
    pub fn median<T>(data: &[T]) -> Option<T>
    where
        T: Ord + Clone + Send + Sync,
    {
        if data.is_empty() {
            return None;
        }

        let mut sorted_data = data.to_vec();
        sorted_data.par_sort();

        let mid = sorted_data.len() / 2;
        Some(sorted_data[mid].clone())
    }

    /// 计算四分位数
    pub fn quartiles<T>(data: &[T]) -> Option<(T, T, T)>
    where
        T: Ord + Clone + Send + Sync,
    {
        let median_val = median(data)?;

        let mut sorted_data = data.to_vec();
        sorted_data.par_sort();

        let n = sorted_data.len();
        let q1_idx = n / 4;
        let q3_idx = 3 * n / 4;

        Some((
            sorted_data[q1_idx].clone(),
            median_val,
            sorted_data[q3_idx].clone(),
        ))
    }
}

/// 字符串特征计算
pub mod string_features {
    use super::*;

    /// 计算字符串长度统计
    pub fn string_length_stats(data: &[String]) -> Option<(usize, usize, f64)>
    where
        String: Send + Sync,
    {
        if data.is_empty() {
            return None;
        }

        let lengths: Vec<usize> = data.par_iter().map(|s| s.len()).collect();
        let min_len = lengths.par_iter().min().copied()?;
        let max_len = lengths.par_iter().max().copied()?;
        let mean_len = lengths.par_iter().sum::<usize>() as f64 / data.len() as f64;

        Some((min_len, max_len, mean_len))
    }

    /// 计算字符频率
    pub fn character_frequency(data: &[String]) -> std::collections::HashMap<char, usize> {
        let mut freq_map = std::collections::HashMap::new();

        for string in data {
            for ch in string.chars() {
                *freq_map.entry(ch).or_insert(0) += 1;
            }
        }

        freq_map
    }

    /// 查找最长公共前缀
    pub fn longest_common_prefix(data: &[String]) -> String {
        if data.is_empty() {
            return String::new();
        }

        if data.len() == 1 {
            return data[0].clone();
        }

        let first = &data[0];
        let mut prefix_len = first.len();

        for string in &data[1..] {
            let mut i = 0;
            while i < prefix_len && i < string.len() && first.chars().nth(i) == string.chars().nth(i) {
                i += 1;
            }
            prefix_len = i;
        }

        first[..prefix_len].to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_sum() {
        let data = vec![1, 2, 3, 4, 5];
        let result = ParallelReduction::parallel_sum(&data);
        assert_eq!(result, Some(15));
    }

    #[test]
    fn test_parallel_max() {
        let data = vec![1, 5, 3, 9, 2];
        let result = ParallelReduction::parallel_max(&data);
        assert_eq!(result, Some(9));
    }

    #[test]
    fn test_parallel_prefix_sum() {
        let data = vec![1, 2, 3, 4, 5];
        let result = ParallelReduction::parallel_prefix_sum(&data);
        assert_eq!(result, vec![1, 3, 6, 10, 15]);
    }

    #[test]
    fn test_parallel_histogram() {
        let data = vec![1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0];
        let histogram = ParallelReduction::parallel_histogram(&data, 5);
        assert_eq!(histogram.len(), 5);
        assert_eq!(histogram.iter().sum::<usize>(), data.len());
    }

    #[test]
    fn test_numerical_features() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = numerical_features::mean(&data);
        assert_eq!(mean, Some(3.0));

        let variance = numerical_features::variance(&data);
        assert!(variance.unwrap() > 0.0);
    }
}
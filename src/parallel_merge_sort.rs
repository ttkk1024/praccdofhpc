use rayon::prelude::*;
use std::cmp::Ordering;

pub struct ParallelMergeSort;

impl ParallelMergeSort {
    /// 串行归并排序实现
    pub fn sequential_merge_sort<T: Ord + Clone>(arr: &mut [T]) {
        if arr.len() <= 1 {
            return;
        }

        let mid = arr.len() / 2;
        Self::sequential_merge_sort(&mut arr[..mid]);
        Self::sequential_merge_sort(&mut arr[mid..]);
        Self::merge(arr, mid);
    }

    /// 并行归并排序实现
    pub fn parallel_merge_sort<T: Ord + Clone + Send>(arr: &mut [T]) {
        Self::parallel_merge_sort_recursive(arr, 0);
    }

    /// 并行归并排序递归实现
    fn parallel_merge_sort_recursive<T: Ord + Clone + Send>(arr: &mut [T], depth: usize) {
        const SEQUENTIAL_THRESHOLD: usize = 1000;
        const MAX_DEPTH: usize = 4;

        if arr.len() <= SEQUENTIAL_THRESHOLD || depth >= MAX_DEPTH {
            Self::sequential_merge_sort(arr);
            return;
        }

        let mid = arr.len() / 2;

        // 并行处理左右两部分
        let (left, right) = arr.split_at_mut(mid);

        rayon::join(
            || Self::parallel_merge_sort_recursive(left, depth + 1),
            || Self::parallel_merge_sort_recursive(right, depth + 1),
        );

        Self::merge(arr, mid);
    }

    /// 就地归并操作
    fn merge<T: Ord + Clone>(arr: &mut [T], mid: usize) {
        let mut left = arr[..mid].to_vec();
        let mut right = arr[mid..].to_vec();

        let mut i = 0;
        let mut j = 0;
        let mut k = 0;

        while i < left.len() && j < right.len() {
            if left[i] <= right[j] {
                arr[k] = left[i].clone();
                i += 1;
            } else {
                arr[k] = right[j].clone();
                j += 1;
            }
            k += 1;
        }

        // 复制剩余元素
        while i < left.len() {
            arr[k] = left[i].clone();
            i += 1;
            k += 1;
        }

        while j < right.len() {
            arr[k] = right[j].clone();
            j += 1;
            k += 1;
        }
    }

    /// 使用Rayon的并行归并排序
    pub fn rayon_merge_sort<T: Ord + Clone + Send>(arr: &mut [T]) {
        if arr.len() <= 1000 {
            Self::sequential_merge_sort(arr);
            return;
        }

        // 使用Rayon的排序算法（通常是并行快速排序，但我们可以实现归并）
        arr.par_sort();
    }

    /// 自适应归并排序 - 根据数组大小和系统特性调整策略
    pub fn adaptive_merge_sort<T: Ord + Clone + Send>(arr: &mut [T]) {
        let cpu_count = num_cpus::get();
        let threshold = std::cmp::max(1000, arr.len() / (cpu_count * 4));

        if arr.len() <= threshold {
            Self::sequential_merge_sort(arr);
        } else {
            Self::parallel_merge_sort(arr);
        }
    }
}

/// 使用Rayon的高效并行归并排序实现
pub fn rayon_efficient_merge_sort<T: Ord + Clone + Send + Sync>(arr: &mut [T]) {
    if arr.len() <= 1 {
        return;
    }

    // 使用Rayon的分治策略
    let mid = arr.len() / 2;

    let (left, right) = arr.split_at_mut(mid);

    rayon::join(
        || rayon_efficient_merge_sort(left),
        || rayon_efficient_merge_sort(right),
    );

    // 使用高效归并
    merge_slices(left, right, arr);
}

fn merge_slices<T: Ord + Clone + Send + Sync + std::marker::Copy>(left: &[T], right: &[T], result: &mut [T]) {
    let mut i = 0;
    let mut j = 0;
    let mut k = 0;

    while i < left.len() && j < right.len() {
        if left[i] <= right[j] {
            result[k] = left[i].clone();
            i += 1;
        } else {
            result[k] = right[j].clone();
            j += 1;
        }
        k += 1;
    }

    // 复制剩余元素
    result[k..].copy_from_slice(&left[i..]);
    result[k + left[i..].len()..].copy_from_slice(&right[j..]);
}

/// 流水线归并排序 - 适用于大规模数据
pub fn pipelined_merge_sort<T: Ord + Clone + Send + Sync>(arr: &mut [T]) {
    if arr.len() <= 1 {
        return;
    }

    // 分块处理
    let chunk_size = std::cmp::max(1000, arr.len() / num_cpus::get());
    let chunks: Vec<_> = arr.chunks_mut(chunk_size).collect();

    // 第一阶段：并行排序各个块
    chunks.par_iter_mut().for_each(|chunk| {
        chunk.sort();
    });

    // 第二阶段：归并所有已排序的块
    merge_sorted_chunks(&mut chunks);
}

fn merge_sorted_chunks<T: Ord + Clone + Send + Sync>(chunks: &mut [&mut [T]]) {
    if chunks.len() <= 1 {
        return;
    }

    let mut temp_chunks = Vec::new();
    let mut i = 0;

    while i < chunks.len() {
        if i + 1 < chunks.len() {
            // 归并两个相邻的块
            let left = chunks[i];
            let right = chunks[i + 1];

            let merged_len = left.len() + right.len();
            let mut merged = vec![left[0].clone(); merged_len];

            merge_slices(left, right, &mut merged);

            temp_chunks.push(merged);
        } else {
            // 最后一个单独的块
            temp_chunks.push(chunks[i].to_vec());
        }
        i += 2;
    }

    // 递归归并
    if temp_chunks.len() > 1 {
        let mut new_chunks: Vec<_> = temp_chunks.iter_mut().map(|v| v.as_mut_slice()).collect();
        merge_sorted_chunks(&mut new_chunks);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn test_sequential_merge_sort() {
        let mut arr = vec![5, 2, 8, 1, 9];
        ParallelMergeSort::sequential_merge_sort(&mut arr);
        assert_eq!(arr, vec![1, 2, 5, 8, 9]);
    }

    #[test]
    fn test_parallel_merge_sort() {
        let mut arr = vec![5, 2, 8, 1, 9, 3, 7, 4, 6];
        ParallelMergeSort::parallel_merge_sort(&mut arr);
        assert_eq!(arr, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_large_array() {
        let size = 10000;
        let mut arr: Vec<i32> = (0..size).collect();
        arr.shuffle(&mut rand::thread_rng());

        let mut sorted = arr.clone();
        sorted.sort();

        ParallelMergeSort::parallel_merge_sort(&mut arr);
        assert_eq!(arr, sorted);
    }

    #[test]
    fn test_empty_and_single_element() {
        let mut arr: Vec<i32> = vec![];
        ParallelMergeSort::parallel_merge_sort(&mut arr);
        assert!(arr.is_empty());

        let mut arr = vec![42];
        ParallelMergeSort::parallel_merge_sort(&mut arr);
        assert_eq!(arr, vec![42]);
    }
}
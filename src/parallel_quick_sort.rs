use rayon::prelude::*;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::sync::Arc;
use std::ops::Add;
use std::iter::Sum;

pub struct ParallelQuickSort;

impl ParallelQuickSort {
    /// 串行快速排序实现
    pub fn sequential_quick_sort<T: Ord + Clone>(arr: &mut [T]) {
        if arr.len() <= 1 {
            return;
        }

        let pivot_index = Self::partition(arr);
        let (left, right) = arr.split_at_mut(pivot_index);
        Self::sequential_quick_sort(left);
        Self::sequential_quick_sort(&mut right[1..]);
    }

    /// 并行快速排序实现
    pub fn parallel_quick_sort<T: Ord + Clone + Send>(arr: &mut [T]) {
        Self::parallel_quick_sort_recursive(arr, 0);
    }

    /// 并行快速排序递归实现
    fn parallel_quick_sort_recursive<T: Ord + Clone + Send>(arr: &mut [T], depth: usize) {
        const SEQUENTIAL_THRESHOLD: usize = 1000;
        const MAX_DEPTH: usize = 4;

        if arr.len() <= SEQUENTIAL_THRESHOLD || depth >= MAX_DEPTH {
            Self::sequential_quick_sort(arr);
            return;
        }

        let pivot_index = Self::partition(arr);
        let (left, right) = arr.split_at_mut(pivot_index);

        // 并行处理左右两部分
        rayon::join(
            || Self::parallel_quick_sort_recursive(left, depth + 1),
            || Self::parallel_quick_sort_recursive(&mut right[1..], depth + 1),
        );
    }

    /// 三数取中选择pivot
    fn median_of_three<T: Ord>(arr: &mut [T]) -> usize {
        if arr.len() < 3 {
            return 0;
        }

        let last = arr.len() - 1;
        let mid = arr.len() / 2;

        // 确保 arr[0] <= arr[mid] <= arr[last]
        if arr[mid] < arr[0] {
            arr.swap(0, mid);
        }
        if arr[last] < arr[0] {
            arr.swap(0, last);
        }
        if arr[last] < arr[mid] {
            arr.swap(mid, last);
        }

        mid
    }

    /// Lomuto分区方案
    fn partition<T: Ord>(arr: &mut [T]) -> usize {
        if arr.len() <= 1 {
            return 0;
        }

        // 三数取中
        let pivot_index = Self::median_of_three(arr);
        arr.swap(pivot_index, arr.len() - 1);

        let pivot = arr.len() - 1;
        let mut i = 0;

        for j in 0..pivot {
            if arr[j] <= arr[pivot] {
                arr.swap(i, j);
                i += 1;
            }
        }

        arr.swap(i, pivot);
        i
    }

    /// 使用随机pivot的快速排序
    pub fn randomized_quick_sort<T: Ord + Clone + Send>(arr: &mut [T]) {
        Self::randomized_quick_sort_recursive(arr, 0);
    }

    fn randomized_quick_sort_recursive<T: Ord + Clone + Send>(arr: &mut [T], depth: usize) {
        const SEQUENTIAL_THRESHOLD: usize = 500;
        const MAX_DEPTH: usize = 5;

        if arr.len() <= SEQUENTIAL_THRESHOLD || depth >= MAX_DEPTH {
            Self::sequential_quick_sort(arr);
            return;
        }

        // 随机选择pivot
        let mut rng = rand::thread_rng();
        let pivot_index = rng.gen_range(0..arr.len());
        arr.swap(pivot_index, arr.len() - 1);

        let pivot_index = Self::partition(arr);
        let (left, right) = arr.split_at_mut(pivot_index);

        rayon::join(
            || Self::randomized_quick_sort_recursive(left, depth + 1),
            || Self::randomized_quick_sort_recursive(&mut right[1..], depth + 1),
        );
    }

    /// 双轴快速排序 - 处理重复元素的优化版本
    pub fn dual_pivot_quick_sort<T: Ord + Clone + Send>(arr: &mut [T]) {
        Self::dual_pivot_quick_sort_recursive(arr, 0);
    }

    fn dual_pivot_quick_sort_recursive<T: Ord + Clone + Send>(arr: &mut [T], depth: usize) {
        const SEQUENTIAL_THRESHOLD: usize = 1000;
        const MAX_DEPTH: usize = 4;

        if arr.len() <= SEQUENTIAL_THRESHOLD || depth >= MAX_DEPTH {
            Self::sequential_quick_sort(arr);
            return;
        }

        if arr.len() < 2 {
            return;
        }

        let (p1, p2) = Self::dual_partition(arr);

        let (left, middle_right) = arr.split_at_mut(p1);
        let (middle, right) = middle_right.split_at_mut(p2 - p1);

        rayon::join(
            || Self::dual_pivot_quick_sort_recursive(left, depth + 1),
            || {
                rayon::join(
                    || Self::dual_pivot_quick_sort_recursive(middle, depth + 1),
                    || Self::dual_pivot_quick_sort_recursive(right, depth + 1),
                );
            },
        );
    }

    /// 双轴分区
    fn dual_partition<T: Ord>(arr: &mut [T]) -> (usize, usize) {
        if arr.len() < 2 {
            return (0, arr.len());
        }

        // 确保 arr[0] < arr[last]
        if arr[0] > arr[arr.len() - 1] {
            arr.swap(0, arr.len() - 1);
        }

        let pivot1 = 0;
        let pivot2 = arr.len() - 1;

        let mut i = 1;
        let mut lt = 1; // 小于pivot1的元素
        let mut gt = arr.len() - 2; // 大于pivot2的元素

        while i <= gt {
            if arr[i] < arr[pivot1] {
                arr.swap(i, lt);
                lt += 1;
                i += 1;
            } else if arr[i] > arr[pivot2] {
                arr.swap(i, gt);
                gt -= 1;
            } else {
                i += 1;
            }
        }

        // 将pivot放到正确位置
        arr.swap(pivot1, lt - 1);
        arr.swap(pivot2, gt + 1);

        (lt - 1, gt + 1)
    }

    /// 基于Rayon的并行快速排序
    pub fn rayon_quick_sort<T: Ord + Clone + Send>(arr: &mut [T]) {
        arr.par_sort();
    }

    /// 自适应快速排序 - 根据数据特性选择策略
    pub fn adaptive_quick_sort<T: Ord + Clone + Send>(arr: &mut [T]) {
        // 检查是否已经排序
        if arr.windows(2).all(|w| w[0] <= w[1]) {
            return;
        }

        // 检查是否逆序
        if arr.windows(2).all(|w| w[0] >= w[1]) {
            arr.reverse();
            return;
        }

        // 对于小数组使用插入排序
        if arr.len() <= 50 {
            Self::insertion_sort(arr);
            return;
        }

        // 使用并行快速排序
        Self::parallel_quick_sort(arr);
    }

    /// 插入排序 - 用于小数组优化
    fn insertion_sort<T: Ord>(arr: &mut [T]) {
        for i in 1..arr.len() {
            let mut j = i;
            while j > 0 && arr[j - 1] > arr[j] {
                arr.swap(j - 1, j);
                j -= 1;
            }
        }
    }

    /// 就地归并 - 用于优化快速排序
    pub fn quick_merge_sort<T: Ord + Clone + Send>(arr: &mut [T]) {
        const MERGE_THRESHOLD: usize = 64;

        if arr.len() <= MERGE_THRESHOLD {
            Self::insertion_sort(arr);
            return;
        }

        let pivot_index = Self::partition(arr);
        let (left, right) = arr.split_at_mut(pivot_index);

        rayon::join(
            || Self::quick_merge_sort(left),
            || Self::quick_merge_sort(&mut right[1..]),
        );

        // 对于已排序的小段使用归并
        if left.len() <= MERGE_THRESHOLD && right.len() <= MERGE_THRESHOLD {
            merge_slices(left, &right[1..], arr);
        }
    }
}

/// 使用Rayon的高效并行快速排序
pub fn rayon_efficient_quick_sort<T: Ord + Clone + Send + Sync>(arr: &mut [T]) {
    if arr.len() <= 1000 {
        arr.sort();
        return;
    }

    // 使用Rayon的内置并行排序
    arr.par_sort();
}

/// 流水线快速排序 - 适用于大规模数据流
pub fn pipelined_quick_sort<T: Ord + Clone + Send + Sync + 'static>(
    chunks: &mut [Vec<T>],
    num_threads: usize,
) {
    // 并行排序各个块
    chunks.par_iter_mut().for_each(|chunk| {
        chunk.par_sort();
    });

    // 归并已排序的块
    merge_sorted_chunks(chunks);
}

fn merge_sorted_chunks<T: Ord + Clone + Send + Sync + 'static>(chunks: &mut [Vec<T>]) {
    if chunks.len() <= 1 {
        return;
    }

    let mut result = Vec::new();
    let mut heap = std::collections::BinaryHeap::new();

    // 初始化堆
    for (chunk_idx, chunk) in chunks.iter().enumerate() {
        if !chunk.is_empty() {
            heap.push(std::cmp::Reverse((chunk[0].clone(), chunk_idx, 0)));
        }
    }

    while let Some(std::cmp::Reverse((value, chunk_idx, pos))) = heap.pop() {
        result.push(value);

        if pos + 1 < chunks[chunk_idx].len() {
            heap.push(std::cmp::Reverse((
                chunks[chunk_idx][pos + 1].clone(),
                chunk_idx,
                pos + 1,
            )));
        }
    }

    // 将结果复制回第一个块
    chunks[0] = result;
    for chunk in &mut chunks[1..] {
        chunk.clear();
    }
}

/// 合并两个已排序的切片
fn merge_slices<T: Ord + Clone + Send + Sync>(left: &[T], right: &[T], result: &mut [T]) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn test_sequential_quick_sort() {
        let mut arr = vec![5, 2, 8, 1, 9];
        ParallelQuickSort::sequential_quick_sort(&mut arr);
        assert_eq!(arr, vec![1, 2, 5, 8, 9]);
    }

    #[test]
    fn test_parallel_quick_sort() {
        let mut arr = vec![5, 2, 8, 1, 9, 3, 7, 4, 6];
        ParallelQuickSort::parallel_quick_sort(&mut arr);
        assert_eq!(arr, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_dual_pivot_quick_sort() {
        let mut arr = vec![5, 2, 8, 1, 9, 3, 7, 4, 6, 5, 5];
        ParallelQuickSort::dual_pivot_quick_sort(&mut arr);
        assert_eq!(arr, vec![1, 2, 3, 4, 5, 5, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_large_array() {
        let size = 10000;
        let mut arr: Vec<i32> = (0..size).collect();
        arr.shuffle(&mut rand::thread_rng());

        let mut sorted = arr.clone();
        sorted.sort();

        ParallelQuickSort::parallel_quick_sort(&mut arr);
        assert_eq!(arr, sorted);
    }

    #[test]
    fn test_empty_and_single_element() {
        let mut arr: Vec<i32> = vec![];
        ParallelQuickSort::parallel_quick_sort(&mut arr);
        assert!(arr.is_empty());

        let mut arr = vec![42];
        ParallelQuickSort::parallel_quick_sort(&mut arr);
        assert_eq!(arr, vec![42]);
    }
}
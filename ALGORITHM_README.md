# 第三章：并行算法实现总结

本目录包含了第三章中所有常见并行算法的Python和C++实现。这些实现基于第三章介绍的并行算法设计原则和模式，包括分治法、流水线、主从模式、工作窃取等。

## 文件结构

### 排序算法
- `parallel_merge_sort.py` - 并行归并排序 (Python)
- `parallel_merge_sort.cpp` - 并行归行排序 (C++)
- `parallel_quick_sort.py` - 并行快速排序 (Python)
- `parallel_quick_sort.cpp` - 并行快速排序 (C++)

### 数值计算
- `parallel_matrix_multiplication.py` - 并行矩阵乘法 (Python)
- `parallel_matrix_multiplication.cpp` - 并行矩阵乘法 (C++)
- `monte_carlo_pi.py` - 蒙特卡洛π值计算 (Python)
- `monte_carlo_pi.cpp` - 蒙特卡洛π值计算 (C++)

### 图算法
- `parallel_bfs.py` - 并行BFS图搜索 (Python)
- `parallel_bfs.cpp` - 并行BFS图搜索 (C++)
- `parallel_dijkstra.py` - 并行Dijkstra最短路径 (Python)
- `parallel_dijkstra.cpp` - 并行Dijkstra最短路径 (C++)

### 归约操作
- `parallel_reduction.py` - 并行归约操作 (Python)
- `parallel_reduction.cpp` - 并行归约操作 (C++)

### 性能分析
- `performance_analyzer.py` - 综合性能分析和比较脚本
- `performance_results.csv` - 性能测试结果数据
- `performance_analysis.png` - 性能分析图表

## 算法特性

### 并行归并排序
- **策略**: 分治法 + 线程池/进程池
- **特点**: 递归分解，多线程并行处理子问题
- **适用**: 大规模数据排序

### 并行快速排序
- **策略**: 分治法 + Map-Reduce
- **特点**: 三数取中优化，工作窃取负载均衡
- **适用**: 通用排序场景

### 并行矩阵乘法
- **策略**: 数据分解 + 分块算法
- **特点**: Strassen算法，缓存友好的分块计算
- **适用**: 大型矩阵运算

### 蒙特卡洛π值计算
- **策略**: Embarrassingly Parallel
- **特点**: 自适应采样，置信区间估计
- **适用**: 随机模拟和数值积分

### 并行BFS
- **策略**: 层级同步 + 工作窃取
- **特点**: 前沿并行处理，负载均衡
- **适用**: 图遍历和最短路径

### 并行Dijkstra
- **策略**: 双向搜索 + 并行前沿
- **特点**: A*启发式，优先队列优化
- **适用**: 加权图最短路径

### 并行归约
- **策略**: 树形归约 + 分段处理
- **特点**: 前缀和，直方图计算
- **适用**: 数据聚合和统计

## 编译和运行

### Python版本
```bash
# 运行单个算法
python parallel_merge_sort.py
python parallel_quick_sort.py
python parallel_matrix_multiplication.py
python monte_carlo_pi.py
python parallel_bfs.py
python parallel_dijkstra.py
python parallel_reduction.py

# 运行性能分析
python performance_analyzer.py
```

### C++版本
```bash
# 编译单个算法
g++ -std=c++17 -O3 -pthread parallel_merge_sort.cpp -o parallel_merge_sort
g++ -std=c++17 -O3 -pthread parallel_quick_sort.cpp -o parallel_quick_sort
g++ -std=c++17 -O3 -pthread parallel_matrix_multiplication.cpp -o parallel_matrix_multiplication
g++ -std=c++17 -O3 -pthread monte_carlo_pi.cpp -o monte_carlo_pi
g++ -std=c++17 -O3 -pthread parallel_bfs.cpp -o parallel_bfs
g++ -std=c++17 -O3 -pthread parallel_dijkstra.cpp -o parallel_dijkstra
g++ -std=c++17 -O3 -pthread parallel_reduction.cpp -o parallel_reduction

# 运行编译后的程序
./parallel_merge_sort
./parallel_quick_sort
./parallel_matrix_multiplication
./monte_carlo_pi
./parallel_bfs
./parallel_dijkstra
./parallel_reduction
```

## 性能优化特性

### 1. 负载均衡
- 工作窃取算法
- 动态任务分配
- 分块大小自适应

### 2. 内存优化
- 缓存友好的数据访问
- 减少内存分配
- 数据局部性优化

### 3. 通信优化
- 批量数据传输
- 异步通信
- 减少同步开销

### 4. 算法级优化
- 串行阈值设置
- 启发式算法
- 数学优化技巧

## 性能分析

运行 `performance_analyzer.py` 可以获得：

1. **执行时间对比**: 不同算法在不同数据规模下的执行时间
2. **加速比分析**: 并行算法相对于串行算法的加速比
3. **效率分析**: 并行效率和资源利用率
4. **可扩展性**: 算法随数据规模增长的表现

## 使用建议

1. **选择合适的并行策略**:
   - 数据密集型: 使用数据分解
   - 计算密集型: 使用函数分解
   - 递归问题: 使用分治法

2. **优化并行粒度**:
   - 过细: 通信开销大
   - 过粗: 负载不均衡
   - 建议: 根据问题规模调整

3. **注意同步开销**:
   - 减少锁竞争
   - 使用无锁数据结构
   - 避免频繁同步

4. **内存访问模式**:
   - 利用缓存局部性
   - 减少内存分配
   - 使用对齐内存

## 参考资料

- 《高性能计算导论》第三章
- MPI/OpenMP并行编程指南
- 并行算法设计模式
- 现代处理器架构优化
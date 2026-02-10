---
aliases:
- 复合泊松过程
- Compound Poisson
- Compound Poisson Process
tags:
- concept
- stochastic processes
---
# 复合泊松过程

## 定义

$设 {N(t), t ≥ 0} 是参数为 λ 的泊松过程，{Y_k, k = 1, 2, ...} 是一列独立同分布的随机变量，且与 {N(t)} 相互独立。$

定义过程：

$ X(t) = \sum_{k=1}^{N(t)} Y_k, \quad t \ge 0 $

则称 {X(t), t ≥ 0} 为复合泊松过程。

### 直观理解

- N(t) 记录在 [0, t] 内发生的"事件"次数
- $Y_k$ 表示第 k 个事件带来的"增量"或"损失/收益"
- X(t) 是所有事件增量的总和

## 性质

### 1. 独立增量

复合泊松过程继承了泊松过程的独立增量性质：不相交的时间区间上的增量相互独立。

### 2. 均值

若 $E[Y_1^2] < \infty$，则：

$ E[X(t)] = \lambda t \cdot E[Y_1] $

### 3. 方差

$ D[X(t)] = Var[X(t)] = \lambda t \cdot E[Y_1^2] $

### 4. 特征函数

略（考试不要求）

## 特例

### 1. 泊松过程本身

若 $Y_k = 1$（常数），则 $X(t) = N(t)$ 退化为普通泊松过程。

### 2. 复合泊松分布

在任意固定时刻 t，X(t) 的分布是复合泊松分布。

## 应用场景

1. **保险理赔**：
   - N(t)：时间段 [0, t] 内的理赔次数
   - $Y_k$：第 k 次理赔的金额
   - X(t)：[0, t] 内的总理赔额

2. **库存系统**：
   - N(t)：到货批次数
   - $Y_k$：每批的到货数量
   - X(t)：总到货量

3. **网络流量**：
   - N(t)：数据包到达次数
   - $Y_k$：每个数据包的大小
   - X(t)：总流量

4. **投资组合**：
   - N(t)：交易次数
   - $Y_k$：每次交易的收益
   - X(t)：总收益

## 相关概念

- [[Poisson Process|泊松过程]]
- [[Renewal Process|更新过程]]

## 计算示例

假设：
- 事件到达率 λ = 2（每小时 2 次）
- $每次事件的增量 Y_k ~ Exp(5)，即 E[Y_1] = 1/5 = 0.2, E[Y_1^2] = 2/25 = 0.08$

则对于 t = 10 小时：

$ E[X(10)] = 2 \times 10 \times 0.2 = 4 $
$ Var[X(10)] = 2 \times 10 \times 0.08 = 1.6 $

即 10 小时期望总增量为 4，方差为 1.6。

## 重要性质总结

- 保留了泊松过程的独立增量性质
- 均值和方差与时间 λt 成正比
- 可以建模"事件数量 + 每次影响"的累积过程

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM ""
WHERE (
  contains(file.path, "01_Math/") OR
  contains(file.path, "02_Economy/") OR
  contains(file.path, "03_Computer_Science/")
) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```

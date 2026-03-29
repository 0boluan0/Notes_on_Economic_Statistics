---
aliases:
- 严平稳过程
- Strictly Stationary Process
- Strict
- Strict-Sense Stationary Process
tags:
- concept
- stochastic processes
---
# 严平稳过程

>[!note] 定义
>
> 严平稳过程（Strict-Sense Stationary Process）是指随机过程的有限维分布不随时间的推移而改变。
>
> 对于任意时间 $t_1$, $t_2$, ..., $t_n$ 和任意时间差 τ：
>
> 随机变量 (X($t_1$), X($t_2$), ..., X($t_n$)) 和 (X($t_1$+τ), X($t_2$+τ), ..., X($t_n$+τ)) 的联合分布相同。
>
## 直观理解

- 无论何时观察过程，其统计行为都是相同的
- 过程的统计特性具有时间平移不变性

## 性质

### 1. 均值常数

如果均值存在，则：

$ E[X(t)] = \mu \quad (\text{常数}) $

### 2. 方差常数

如果方差存在，则：

$ Var[X(t)] = \sigma^2 \quad (\text{常数}) $

### 3. 相关函数仅与时间差有关

$ R_X(t_1, t_2) = R_X(t_2 - t_1) $

### 4. 任意阶矩（如果存在）不随时间变化

## 与宽平稳过程的关系

### 宽平稳过程

二阶矩过程均值常数，相关函数仅与时间差有关。

### 两者关系

1. **严平稳 ⇒ 宽平稳**（如果二阶矩存在）
   - 严平稳条件更强
   - 要求所有阶的分布都平移不变

2. **宽平稳 ⇏/ ⇒ 严平稳**
   - 宽平稳只要求二阶矩性质
   - 不能保证更高阶矩的性质

### 特例：正态过程

对于正态随机过程：

$ \text{宽平稳} \iff \text{严平稳} $

这是因为正态分布完全由一、二阶矩决定。

## 各态历经性

>[!note] 定义
>
> 对于严平稳过程，如果统计特性可以用一个样本函数的时间平均来代替，则称其具有各态历经性（Ergodicity）。
>
### 均值各态历经性

$ \lim_{T \to \infty} \frac{1}{2T} \int_{-T}^T X(t) dt = E[X(t)] $

### 相关函数各态历经性

$ \lim_{T \to \infty} \frac{1}{2T} \int_{-T}^T X(t)X(t+\tau) dt = R_X(\tau) $

## 应用

1. **热力学**：平衡态系统的随机涨落
2. **通信工程**：平稳噪声信号
3. **经济学**：具有长期稳定性的时间序列
4. **物理学**：布朗运动（增量严平稳）

## 相关概念

- [[Wide-Sense Stationary Process|宽平稳过程]]
- [[Ergodicity|各态历经性]]
| 特征 | 严平稳 | 宽平稳 |
|------|---------|---------|
| 要求 | 所有阶的分布平移不变 | 仅二阶矩平移不变 |
| 条件 | 更强 | 较弱 |
| 均值 | 常数 | 常数 |
| 方差 | 常数 | 常数 |
| 相关函数 | 仅与时间差有关 | 仅与时间差有关 |
| 适用性 | 理论分析 | 实际应用（更常见）|

## 重要性

- 提供了随机过程"稳定性"的最强定义
- 是许多理论分析的基础
- 在正态情况下与宽平稳等价，简化分析

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

---
aliases:
- 泊松过程
- Poisson Counting Process
- Poisson Process
tags:
- concept
- stochastic processes
---
# 泊松过程

## 定义

泊松过程 {N(t), t ≥ 0} 是一个计数过程，表示在时间区间 [0, t] 内事件 A 发生的总次数。

### 等价定义 1（泊松分布）

满足以下四个条件的过程是参数为 λ > 0 的泊松过程：

1. **初值条件**：$N(0) = 0$
2. **独立增量**：对于任意互不重叠的时间区间，其增量相互独立
3. **平稳增量**：对于任意 s < t，增量 N(t) - N(s) 的分布只与区间长度 t-s 有关
4. **泊松分布**：在任意长度为 t 的时间区间内，事件发生次数服从参数为 λt 的泊松分布

$ P\{N(t) = n\} = e^{-\lambda t}\frac{(\lambda t)^n}{n!}, \quad n = 0, 1, 2, \dots $

### 等价定义 2（指数间隔）

相邻事件到达间隔 $X_i$ 互相独立，且每个 $X_i$ 服从指数分布 Exp(λ)。

## 数字特征

对于齐次泊松过程（λ 不随时间变化）：

### 1. 均值函数

$ E[N(t)] = \lambda t $

### 2. 方差函数

$ Var[N(t)] = \lambda t $

### 3. 协方差函数

对于 0 ≤ s < t：

$ Cov(N(s), N(t)) = \lambda \min(s, t) $

## 无记忆性

由于相邻事件到达间隔服从指数分布，具有无记忆性：

$ P\{X > s + t \mid X > s\} = P\{X > t\} $

这也是泊松过程"没有后效"或"平稳"的重要根源。

## 相关分布

### 1. 时间间隔分布

相邻事件的时间间隔 $T_n$ 相互独立，且均服从指数分布 Exp(λ)：

$ P\{T_n \le t\} = 1 - e^{-\lambda t}, \quad t \ge 0 $

$ f_{T_n}(t) = \lambda e^{-\lambda t}, \quad t \ge 0 $

### 2. 等待时间分布

第 n 次事件的等待时间 $W_n$ = $T_1$ + $T_2$ + ... + $T_n$ 服从 Gamma(n, λ) 分布：
$$

$ f_{W_n}(t) = \frac{\lambda e^{-\lambda t}(\lambda t)^{n-1}}{(n-1)!}, \quad t \ge 0 $

- 均值：$E[W_n] = \frac{n}{\lambda}$
- 方差：$Var[W_n] = \frac{n}{\lambda^2}$

### 3. 到达时间的条件分布

若已知 [0, t] 内发生了 n 次事件，则这 n 个到达时刻在 [0, t] 上等价于 n 个 U(0, t) 变量的顺序统计量。

联合密度：

$ f(t_1, t_2, \dots, t_n | N(t) = n) = \frac{n!}{t^n}, \quad 0 < t_1 < t_2 < \dots < t_n < t $

## 非齐次泊松过程

### 定义

强度函数 λ(t) 随时间变化的泊松过程。

### 均值函数

$ m_X(t) = \int_0^t \lambda(s)ds $

### 分布

$ P\{N(t) = n\} = \frac{[m_X(t)]^n}{n!} e^{-m_X(t)}, \quad n \ge 0 $

## 应用场景

1. **电话呼叫**：单位时间内接收到的呼叫次数
2. **顾客到达**：商店、银行等服务系统
3. **故障发生**：设备故障次数
4. **交通事故**：特定路段的事故数量
5. **放射性衰变**：粒子发射数

## 相关概念

- [[Compound Poisson Process|复合泊松过程]]
- [[Renewal Process|更新过程]]
- [[Exponential Distribution|指数分布]]

## 性质总结

- 独立增量 + 平稳增量 + 泊松分布
- 时间间隔独立同指数分布
- 等待时间服从 Gamma 分布
- 均值 = 方差 = λt

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

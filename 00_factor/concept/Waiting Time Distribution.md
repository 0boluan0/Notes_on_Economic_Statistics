---
aliases:
- Waiting Time Distribution
- Waiting Time
- 等待时间分布
tags:
- concept
- stochastic-processes
---
# Waiting Time Distribution

## 一句话记忆

泊松过程里，第 $n$ 次事件的等待时间是前 $n$ 个指数间隔的和。

## 它是什么

在强度为 $\lambda$ 的 [[Poisson Process]] 中，相邻事件间隔 $T_1,T_2,\dots$ 独立同分布且 $T_i\sim \mathrm{Exp}(\lambda)$。第 $n$ 次事件发生时刻为：

$$
W_n=T_1+\cdots+T_n
$$

因此：

$$
W_n\sim \mathrm{Gamma}(n,\lambda)
$$

这里采用 rate 参数 $\lambda$。

## 解决什么判断

- 从 0 开始等到第 $n$ 次事件需要多久。
- 泊松计数 $N(t)$ 和到达时间 $W_n$ 如何互相表达。
- 为什么泊松过程到达时间会接 Gamma/Erlang 分布。

## 最小例子

若顾客到达服从强度 $\lambda$ 的泊松过程，则等到第 3 个顾客的时间：

$$
W_3=T_1+T_2+T_3\sim \mathrm{Gamma}(3,\lambda)
$$

## 易混点

- 相邻事件间隔 $T_i$ 是指数分布；第 $n$ 次事件等待时间 $W_n$ 是 Gamma 分布。
- $W_n\le t$ 等价于 $N(t)\ge n$。
- Gamma 的参数可能用 rate 或 scale，读题时要确认。

## 来自课程位置

- [[03_泊松过程#2.2. 等待时间分布]]

## 关联卡片

- [[Poisson Process]]
- [[Nonhomogeneous Poisson Process]]
- [[Renewal Process]]
- [[Gamma]]
- [[Exponential Distribution]]

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

---
aliases:
- Nonhomogeneous Poisson Process
- Non-homogeneous Poisson Process
- NHPP
- 非齐次泊松过程
tags:
- concept
- stochastic-processes
---
# Nonhomogeneous Poisson Process

## 一句话记忆

非齐次泊松过程是到达强度随时间变化的泊松计数过程。

## 它是什么

Nonhomogeneous Poisson Process 是强度函数 $\lambda(t)$ 随时间变化的计数过程。它仍有独立增量，但增量分布取决于积分强度：

$$
m(t)=\int_0^t\lambda(s)\,ds
$$

因此：

$$
N(t)\sim \mathrm{Poisson}(m(t))
$$

## 解决什么判断

- 到达率不是常数时，如何计算一段时间内事件次数。
- 为什么不同时间段的平均到达数要用积分强度。
- 齐次泊松过程和非齐次泊松过程的区别。

## 最小例子

若 $\lambda(t)=2t$，则 $[0,T]$ 内的平均到达数是：

$$
m(T)=\int_0^T2t\,dt=T^2
$$

所以 $N(T)\sim \mathrm{Poisson}(T^2)$。

## 易混点

- 非齐次泊松过程不再有平稳增量。
- 强度函数 $\lambda(t)$ 不是均值函数，积分后的 $m(t)$ 才是均值函数。
- 若 $\lambda(t)=\lambda$ 为常数，就退化为 [[Poisson Process]]。

## 来自课程位置

- [[03_泊松过程#3.非齐次泊松过程]]
- [[00_随机考点]]

## 关联卡片

- [[Poisson Process]]
- [[Waiting Time Distribution]]
- [[Compound Poisson Process]]
- [[Renewal Process]]

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

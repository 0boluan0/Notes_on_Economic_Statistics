---
aliases:
- Renewal Process
- 更新过程
tags:
- concept
- stochastic-processes
---
# Renewal Process

## 一句话记忆

更新过程是由一串独立同分布的等待时间累加出来的计数过程。

## 它是什么

设 $X_1,X_2,\dots$ 是非负 IID 随机变量，表示相邻更新之间的时间间隔。令：

$$
S_n=X_1+\cdots+X_n
$$

并定义：

$$
N(t)=\max\{n:S_n\le t\}
$$

则 $\{N(t),t\ge0\}$ 是 renewal process。

## 解决什么判断

- 设备更换、故障到达、重复事件能否看作“更新”。
- 第 $n$ 次事件时间 $S_n$ 和时间 $t$ 前事件次数 $N(t)$ 如何互相转换。
- 泊松过程为什么是更新过程的特殊情形。

## 最小例子

每台设备寿命独立同分布，坏了立即换新。到时间 $t$ 为止更换的设备数就是一个更新过程。

## 易混点

- Renewal process 的间隔只要求 IID，不一定是指数分布。
- 当间隔服从指数分布时，更新过程才是 [[Poisson Process]]。
- $S_n\le t$ 等价于 $N(t)\ge n$。

## 来自课程位置

- [[03_泊松过程#6. 更新过程]]

## 关联卡片

- [[Poisson Process]]
- [[Waiting Time Distribution]]
- [[IID]]
- [[Exponential Distribution]]
- [[Compound Poisson Process]]

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

---
aliases: [Riemann Sum, 黎曼和, 积分和]
tags: [concept, calculus]
---
# Riemann Sum

把区间分成小段，在第 $i$ 段选择样本点 $x_i^*$，形成

$$
\sum_{i=1}^n f(x_i^*)\Delta x_i.
$$

它用有限个矩形近似累计量；适当极限给出定积分。

## 最小例子

在 $[0,1]$ 上用 $n$ 个等宽右端点矩形近似 $\int_0^1x\,dx$。

## 关联卡片

- [[Definite Integral]]
- [[Numerical Integration Methods]]

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM "" WHERE (contains(file.path, "01_Math/") OR contains(file.path, "02_Economy/") OR contains(file.path, "03_Computer_Science/")) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```

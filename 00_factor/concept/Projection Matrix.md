---
aliases:
- Projection Matrix
- 投影矩阵
tags:
- concept
- 线性代数
---
# Projection Matrix

## 它是什么
- 「Projection Matrix」是指实现投影线性变换的矩阵。

## 最小可检索信息
- 定义：满足 $Pb$ 等于向量 $b$ 在某个子空间上的投影。
- 符号/公式：若 $A$ 的列张成投影子空间且列独立，则 $P=A(A^TA)^{-1}A^T$。
- 最小例子：投影到 $\operatorname{span}(a)$ 时，$P=\frac{aa^T}{a^Ta}$。

## 关键性质
- 投影矩阵满足 $P^2=P$。
- 正交投影矩阵还是对称矩阵：$P^T=P$。
- 特征值只能是 0 或 1。

## 关联卡片
- [[Orthogonal Projection]]
- [[Least Squares]]
- [[Orthogonal Matrix]]

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM ""
WHERE (
  contains(file.path, "01_Math/") OR
  contains(file.path, "02_Economy/") OR
  contains(file.outlinks, this.file.link)
)
SORT file.mtime DESC
```

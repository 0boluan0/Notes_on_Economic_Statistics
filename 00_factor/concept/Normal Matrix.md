---
aliases:
  - Normal Matrix
  - normal matrices
  - 正规矩阵
tags:
  - 线性代数
  - concept
---

# Normal Matrix

## 它是什么

复方阵 $A$ 若与自己的共轭转置可交换，

$$
A^*A=AA^*,
$$

则称为 normal matrix。它恰好是可以被 unitary 矩阵对角化的复方阵：

$$
A=Q\Lambda Q^*,\qquad Q^*Q=I.
$$

Hermitian、skew-Hermitian、unitary 矩阵都属于 normal 矩阵。

## 最小例子

实旋转矩阵

$$
\begin{bmatrix}0&-1\\1&0\end{bmatrix}
$$

不是实对称矩阵，却是 normal 矩阵，并可在 $\mathbb C$ 上 unitary 对角化。

## 边界

一般可对角化矩阵未必 normal；相似对角化中的特征向量不必正交。

## 关联卡片

- [[Hermitian Matrix]]
- [[Unitary Matrix]]
- [[Diagonalization]]
- [[Spectral Decomposition]]

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

---
aliases:
  - Hermitian Matrix
  - self-adjoint matrix
  - Hermitian 矩阵
  - 厄米矩阵
  - 共轭对称矩阵
tags:
  - 线性代数
  - concept
---

# Hermitian Matrix

## 它是什么

复方阵 $A$ 若满足

$$
A^*=\overline A^{\,T}=A,
$$

则称为 Hermitian 矩阵。它是实对称矩阵在复数域中的对应物。

Hermitian 矩阵的特征值全为实数，并存在标准正交的复特征向量基：

$$
A=Q\Lambda Q^*,\qquad Q^*Q=I,
$$

其中 $\Lambda$ 为实对角矩阵。

## 最小例子

$$
\begin{bmatrix}2&i\\-i&3\end{bmatrix}
$$

是 Hermitian 矩阵，因为非对角元互为共轭。

## 边界

- Hermitian 条件是 $A^*=A$，不是 $A^T=A$。
- 对角元必须为实数。

## 关联卡片

- [[Symmetric Matrix]]
- [[Unitary Matrix]]
- [[Spectral Decomposition]]
- [[Positive Definite Matrix]]

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

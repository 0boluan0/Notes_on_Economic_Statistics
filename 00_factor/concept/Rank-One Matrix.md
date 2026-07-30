---
aliases:
  - Rank-One Matrix
  - rank-one matrices
  - 秩一矩阵
tags:
  - 线性代数
  - concept
---

# Rank-One Matrix

## 它是什么

非零矩阵 $A$ 若 $\operatorname{rank}(A)=1$，称为秩一矩阵。它恰好可以写成两个非零向量的外积：

$$
A=uv^*.
$$

实数情形为 $A=uv^T$。对任意 $x$，

$$
Ax=u(v^*x),
$$

所以所有输出都位于同一条直线 $\operatorname{span}(u)$ 上。

## 最小例子

$$
\begin{bmatrix}2&-4\\1&-2\end{bmatrix}
=\begin{bmatrix}2\\1\end{bmatrix}
\begin{bmatrix}1&-2\end{bmatrix}.
$$

## 边界

- 零矩阵的秩是 $0$，不是秩一。
- 外积表示不唯一：$(cu)(v^*/c)$ 给出同一矩阵。

## 关联卡片

- [[Matrix Rank]]
- [[Column Space]]
- [[Singular Value Decomposition]]
- [[Low-Rank Approximation]]

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

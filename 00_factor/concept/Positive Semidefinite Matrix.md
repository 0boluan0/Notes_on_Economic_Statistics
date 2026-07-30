---
aliases:
  - Positive Semidefinite Matrix
  - PSD matrix
  - positive semidefinite
  - 正半定矩阵
  - 半正定矩阵
tags:
  - 线性代数
  - concept
---

# Positive Semidefinite Matrix

## 它是什么

实对称矩阵 $A$ 若对所有 $x$ 都满足

$$
x^TAx\ge0,
$$

则称为正半定矩阵。复数情形要求 $A=A^*$ 且 $x^*Ax\ge0$。

对实对称或 Hermitian 矩阵，下列描述等价：

- 所有特征值非负；
- 存在矩阵 $B$ 使 $A=B^*B$；
- 二次型在任何方向都不向下。

## 最小例子

$$
\begin{bmatrix}1&-1\\-1&1\end{bmatrix}
$$

的特征值为 $0,2$，因此正半定但不正定。

## 边界

半正定允许非零 $x$ 使 $x^TAx=0$，所以矩阵可以奇异，也不保证严格极小值。

## 关联卡片

- [[Positive Definite Matrix]]
- [[Quadratic Form]]
- [[Graph Laplacian]]
- [[Singular Value]]

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

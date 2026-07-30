---
aliases:
  - Algebraic and Geometric Multiplicity
  - eigenvalue multiplicity
  - 代数重数与几何重数
  - 特征值重数
tags:
  - 线性代数
  - concept
---

# Algebraic and Geometric Multiplicity

## 它是什么

对方阵 $A$ 的特征值 $\lambda$：

- **代数重数** $a_\lambda$ 是 $\lambda$ 作为特征多项式根的重数；
- **几何重数** $g_\lambda$ 是特征空间 $N(A-\lambda I)$ 的维数。

总有

$$
1\le g_\lambda\le a_\lambda.
$$

矩阵可对角化，当且仅当每个特征值都有 $g_\lambda=a_\lambda$，等价于所有特征空间的维数之和为 $n$。

## 最小例子

$$
J=\begin{bmatrix}2&1\\0&2\end{bmatrix}
$$

中，$\lambda=2$ 的代数重数为 $2$，但 $N(J-2I)$ 只有一维，所以几何重数为 $1$，矩阵不可对角化。

## 关联卡片

- [[Characteristic Polynomial]]
- [[Diagonalization]]
- [[Jordan Form]]
- [[Eigenvectors]]

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

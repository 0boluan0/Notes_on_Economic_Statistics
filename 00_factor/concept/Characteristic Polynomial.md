---
aliases:
  - Characteristic Polynomial
  - characteristic equation
  - 特征多项式
  - 特征方程
tags:
  - 线性代数
  - concept
---

# Characteristic Polynomial

## 它是什么

方阵 $A\in\mathbb F^{n\times n}$ 的特征多项式定义为

$$
p_A(\lambda)=\det(\lambda I-A).
$$

它的根就是 $A$ 的特征值，因为

$$
p_A(\lambda)=0
\iff \lambda I-A\text{ 奇异}
\iff N(A-\lambda I)\ne\{0\}.
$$

不同教材也会使用 $\det(A-\lambda I)$；两者只差整体因子 $(-1)^n$，根完全相同。

## 最小例子

$$
A=\begin{bmatrix}2&1\\0&3\end{bmatrix}
\quad\Longrightarrow\quad
p_A(\lambda)=(\lambda-2)(\lambda-3).
$$

## 边界

特征多项式只对方阵定义；矩形矩阵应使用奇异值而不是特征值来描述伸缩结构。

## 关联卡片

- [[Eigenvalues]]
- [[Eigenvectors]]
- [[Algebraic and Geometric Multiplicity]]
- [[Determinant]]

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

---
aliases:
- Matrix Inverse
- Inverse Matrix
- inverse
- 矩阵的逆
- 逆矩阵
tags:
  - concept
  - 线性代数
---
# Matrix Inverse

## 先记一句话

矩阵的逆就是：**把一个可逆线性变换完整 undo 掉的矩阵**。

如果
$$
AA^{-1}=A^{-1}A=I,
$$
那么 $A^{-1}$ 是 $A$ 的逆矩阵。

## 它是什么

逆矩阵只对方阵的可逆情形存在。

对 $Ax=b$ 来说，如果 $A$ 可逆，那么解唯一，并且
$$
x=A^{-1}b.
$$

这意味着 $A$ 没有把任何非零方向压成 0，也没有丢失信息。

## 一个最小例子

对
$$
A=
\begin{bmatrix}
a&b\\
c&d
\end{bmatrix},
$$
若
$$
ad-bc\neq0,
$$
则
$$
A^{-1}
=
\frac{1}{ad-bc}
\begin{bmatrix}
d&-b\\
-c&a
\end{bmatrix}.
$$

分母就是 [[Determinant]]。如果 determinant 为 0，逆矩阵不存在。

## 它在题里负责什么

- 判断 $Ax=b$ 是否对每个 $b$ 都有唯一解。
- 把“可逆”连接到 rank、determinant、nullspace、pivot。
- 解释 left inverse / right inverse / pseudoinverse 的边界。

## 和 rank 的关系

对 $n\times n$ 方阵，下列说法等价：

- $A$ 可逆；
- $\operatorname{rank}(A)=n$；
- $N(A)=\{0\}$；
- $C(A)=\mathbb{R}^n$；
- $\det(A)\neq0$；
- 每一列都有 pivot。

这些等价关系集中在 [[Invertible Matrix Equivalence Chain]]。

## 常见误区

- 不要在计算题里先求逆再解方程；消元通常更稳、更快。
- 非方阵没有普通逆，但可能有 left inverse、right inverse 或 [[Pseudoinverse]]。
- $\det(A)$ 很小和 $\det(A)=0$ 是不同问题；前者常涉及数值稳定性，后者是结构不可逆。

## 来自课程位置

- [[01_Ax = b and the Four Subspaces#Session 1.4 Multiplication and inverse matrices|Session 1.4]]：inverse 与 elimination。
- [[02_Least Squares, Determinants and Eigenvalues#Session 2.7 Cramer's rule, inverse matrix and volume|Session 2.7]]：inverse formula 与 determinant。
- [[03_Positive Definite Matrices and Applications#Session 3.8 Left and right inverses and pseudoinverse|Session 3.8]]：普通逆、左右逆与伪逆。

## 关联卡片

- [[Determinant]]
- [[Cramer's Rule]]
- [[Matrix Rank]]
- [[Invertible Matrix Equivalence Chain]]
- [[Pseudoinverse]]


## 最小例子

把 **Matrix Inverse** 放在最简单的可计算情形中：先给定定义所需的最小输入，再按定义计算输出；若关键关系不成立，就不能把该对象归入本概念。这个检查也能帮助区分相近概念。
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

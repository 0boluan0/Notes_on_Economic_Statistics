---
aliases:
- Determinant
- determinant
- det
- 行列式
tags:
- concept
- 线性代数
---
# Determinant

## 先记一句话

行列式就是：**方阵作为线性变换时，对有向体积的缩放倍数**。

它同时告诉你两件事：

- 体积被放大、缩小或翻转多少；
- 方阵是否把空间压扁到低维。

## 它是什么

对 $2\times2$ 矩阵
$$
A=
\begin{bmatrix}
a&b\\
c&d
\end{bmatrix},
$$
行列式是
$$
\det(A)=ad-bc.
$$

几何上，$|\det(A)|$ 是单位正方形被 $A$ 变换后的面积。

如果 $\det(A)=0$，说明面积或体积被压成 0，矩阵不可逆。

## 一个最小例子

令
$$
A=
\begin{bmatrix}
2&0\\
0&3
\end{bmatrix}.
$$

它把 $x$ 方向放大 2 倍，把 $y$ 方向放大 3 倍，所以面积放大
$$
2\cdot3=6
$$
倍。

因此
$$
\det(A)=6.
$$

## 它在题里负责什么

- 判断方阵是否可逆：$\det(A)\neq0$。
- 判断列向量是否线性无关：$\det(A)\neq0$。
- 计算体积缩放：体积乘以 $|\det(A)|$。
- 连接特征值：若 $A$ 有特征值 $\lambda_i$，则
  $$
  \det(A)=\prod_i\lambda_i.
  $$

## 常见误区

- determinant 只对方阵定义。
- $\det(A)=0$ 不是“小”，而是结构性地压扁了某个方向。
- 行列式适合解释可逆性和体积，不适合直接作为大型线性方程组的数值求解方法。

## 来自课程位置

- [[02_Least Squares, Determinants and Eigenvalues#Session 2.5 Properties of determinants|Session 2.5]]：行列式性质、奇异性与体积。
- [[02_Least Squares, Determinants and Eigenvalues#Session 2.6 Determinant formulas and cofactors|Session 2.6]]：cofactor 展开与公式。
- [[02_Least Squares, Determinants and Eigenvalues#Session 2.7 Cramer's rule, inverse matrix, and volume|Session 2.7]]：Cramer's rule、逆矩阵与体积解释。

## 关联卡片

- [[Matrix Inverse]]
- [[Cramer's Rule]]
- [[Eigenvalues]]
- [[Invertible Matrix Equivalence Chain]]

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

---
aliases:
- Cramer's Rule
- Cramers Rule
- 克拉默法则
tags:
- concept
- 线性代数
---

# Cramer's Rule

## 先记一句话

Cramer's Rule 是：**用行列式比值显式写出可逆方阵方程 $Ax=b$ 的解**。

它不是日常计算的首选方法，而是用来说明解和 determinant 之间的结构关系。

## 它是什么

若 $A$ 是 $n\times n$ 方阵且
$$
\det(A)\neq0,
$$
则 $Ax=b$ 的第 $i$ 个分量是
$$
x_i=\frac{\det(A_i)}{\det(A)}.
$$

这里 $A_i$ 表示把 $A$ 的第 $i$ 列替换成 $b$ 后得到的矩阵。

## 一个最小例子

二维系统
$$
\begin{bmatrix}
a&b\\
c&d
\end{bmatrix}
\begin{bmatrix}
x_1\\x_2
\end{bmatrix}
=
\begin{bmatrix}
e\\f
\end{bmatrix}
$$
在 $\det(A)=ad-bc\neq0$ 时，
$$
x_1=
\frac{
\det\begin{bmatrix}e&b\\f&d\end{bmatrix}
}{\det(A)},
\qquad
x_2=
\frac{
\det\begin{bmatrix}a&e\\c&f\end{bmatrix}
}{\det(A)}.
$$

## 它在题里负责什么

- 证明解对 $b$ 的线性依赖。
- 连接 inverse formula 和 determinant。
- 在小维度符号题中快速写解。

## 常见误区

- 只适用于可逆方阵；非方阵和 $\det(A)=0$ 的情况不能用。
- 它理论上漂亮，但数值计算中通常不如消元稳定。
- 不要把 Cramer's Rule 当成所有线性系统的通用解法。

## 来自课程位置

- [[02_Least Squares, Determinants and Eigenvalues#Session 2.7 Cramer's rule, inverse matrix and volume|Session 2.7]]：Cramer's rule 与 inverse formula。

## 关联卡片

- [[Determinant]]
- [[Matrix Inverse]]
- [[Linear system solution structure]]

## 边界与相关概念

本卡片只保留该概念的定义、记号与最小直觉；具体估计步骤、证明或完整应用应放在对应的 `procedure`、`proof` 或 `framework` 卡片中。

相关卡片：[[02_Least Squares, Determinants and Eigenvalues]]、[[Determinant]]、[[Matrix Inverse]]、[[Linear system solution structure]]。

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

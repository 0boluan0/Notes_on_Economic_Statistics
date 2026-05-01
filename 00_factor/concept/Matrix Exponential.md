---
aliases:
- Matrix Exponential
- matrix exponential
- eAt
- 矩阵指数
tags:
- concept
- 线性代数
---
# Matrix Exponential

## 先记一句话

矩阵指数就是：**线性微分方程 $u'(t)=Au(t)$ 的演化算子**。

标量方程
$$
u'(t)=au(t)
$$
的解是 $u(t)=e^{at}u(0)$。

矩阵版本就是
$$
u(t)=e^{At}u(0).
$$

## 它是什么

矩阵指数定义为
$$
e^A=I+A+\frac{A^2}{2!}+\frac{A^3}{3!}+\cdots.
$$

所以
$$
e^{At}=I+At+\frac{(At)^2}{2!}+\cdots.
$$

它描述的是矩阵 $A$ 持续作用时，状态如何随时间变化。

## 一个最小例子

若
$$
A=
\begin{bmatrix}
2&0\\
0&3
\end{bmatrix},
$$
则
$$
e^{At}=
\begin{bmatrix}
e^{2t}&0\\
0&e^{3t}
\end{bmatrix}.
$$

两个坐标方向完全解耦，各自按自己的特征值增长。

## 和对角化的关系

如果
$$
A=S\Lambda S^{-1},
$$
那么
$$
e^{At}=Se^{\Lambda t}S^{-1}.
$$

而
$$
e^{\Lambda t}
$$
只需要对每个对角元取指数。

所以对角化把系统拆成独立的一维指数增长或衰减。

## 它在题里负责什么

- 解线性常微分方程组 $u'=Au$。
- 分析稳定性：看特征值实部。
- 解释动力系统的增长、衰减与振荡。
- 在 diagonalization 失败时，Jordan block 会带来额外多项式因子。

## 常见误区

- 一般不能把 $e^{A+B}$ 写成 $e^Ae^B$，除非 $A$ 和 $B$ 可交换。
- 矩阵指数不是对矩阵每个元素分别取指数。
- 对角化是计算矩阵指数的常用入口，但不是定义本身。

## 来自课程位置

- [[02_Least Squares, Determinants and Eigenvalues#Session 2.10 Differential equations and $e^{At}$|Session 2.10]]：矩阵指数与微分方程。
- [[03_Positive Definite Matrices and Applications#Session 3.4 Similar matrices and Jordan form|Session 3.4]]：Jordan 情形下的矩阵指数。

## 关联卡片

- [[Diagonalization]]
- [[Eigenvalues]]
- [[Eigenvectors]]
- [[Jordan Form]]
- [[Markov Matrix]]

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

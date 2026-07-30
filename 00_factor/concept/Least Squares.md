---
aliases:
- Least Squares
- least-squares
- 最小二乘
- 最小二乘法
tags:
  - concept
  - 线性代数
---
# Least Squares

## 先记一句话

最小二乘就是：**当 $Ax=b$ 没有精确解时，找一个让误差最小的 $x$**。

它不是在强行让 $Ax=b$ 成立，而是在求
$$
\min_x \|Ax-b\|^2.
$$

## 它解决什么判断

如果 $b\notin C(A)$，那么原方程 $Ax=b$ 无解。

但我们仍然可以找 $C(A)$ 里离 $b$ 最近的向量：
$$
p=A\hat{x}.
$$

于是 least squares 的目标变成：

> 在所有可能的 $Ax$ 里，找最接近 $b$ 的那个。

## 几何图像

最优点满足
$$
b=p+e,
$$
其中
$$
p\in C(A),\qquad e\perp C(A).
$$

所以 least squares 的核心不是某个代数公式，而是：

> 残差必须与列空间正交。

由此得到
$$
A^T(b-A\hat{x})=0,
$$
也就是正规方程
$$
A^TA\hat{x}=A^Tb.
$$

## 一个最小例子

用直线拟合散点时，通常所有点不会刚好落在一条直线上。

如果把直线参数写成 $x$，把观测数据写成 $b$，那么 $Ax=b$ 通常无解。

least squares 做的事是选择 $\hat{x}$，让所有残差平方和最小：
$$
\|A\hat{x}-b\|^2.
$$

几何上，这就是把 $b$ 投影到 $A$ 的列空间上。

## 它在题里负责什么

- 过定方程组没有精确解时，给出最佳近似解。
- 解释回归 fitted values：$\hat{b}=A\hat{x}=Pb$。
- 解释 residual：$e=b-\hat{b}$，且 $A^Te=0$。
- 连接线性代数和统计里的 OLS。

## 常见误区

- least squares 解 $\hat{x}$ 不是原方程的精确解；它是误差最小的参数。
- $\hat{x}$、投影点 $p=A\hat{x}$、残差 $e=b-p$ 三个对象要分清。
- 正规方程不是凭空来的，它来自残差正交。
- 如果 $A^TA$ 不可逆，不要硬写 $(A^TA)^{-1}$；改用 [[Pseudoinverse]] 或 QR/SVD。

## 来自课程位置

- [[02_Least Squares, Determinants and Eigenvalues#Session 2.2 Projections onto subspaces|Session 2.2]]：投影给出 closest point。
- [[02_Least Squares, Determinants and Eigenvalues#Session 2.3 Projection matrices and least squares|Session 2.3]]：least squares 与正规方程。

## 关联卡片

- [[Orthogonal Projection]]
- [[Projection Matrix]]
- [[Least Squares via Normal Equations]]
- [[QR Decomposition]]
- [[Pseudoinverse]]


## 最小例子

把 **Least Squares** 放在最简单的可计算情形中：先给定定义所需的最小输入，再按定义计算输出；若关键关系不成立，就不能把该对象归入本概念。这个检查也能帮助区分相近概念。
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

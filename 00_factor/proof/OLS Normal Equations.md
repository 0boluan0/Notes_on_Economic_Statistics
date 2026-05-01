---
aliases:
- OLS Normal Equations
- OLS正规方程
- 最小二乘正规方程
tags:
- proof
- econometrics
---
# OLS Normal Equations

## 假设

线性模型写作：

$$
y=X\beta+u
$$

目标函数为残差平方和：

$$
RSS(\beta)=(y-X\beta)'(y-X\beta)
$$

假设 $X$ 满列秩，使 $X'X$ 可逆。

## 推导链

展开目标函数：

$$
RSS(\beta)=y'y-2\beta'X'y+\beta'X'X\beta
$$

对 $\beta$ 求一阶条件：

$$
\frac{\partial RSS}{\partial \beta}=-2X'y+2X'X\beta=0
$$

得到正规方程：

$$
X'X\hat\beta=X'y
$$

若 $X'X$ 可逆：

$$
\hat\beta=(X'X)^{-1}X'y
$$

残差为 $\hat u=y-X\hat\beta$，所以正规方程也等价于：

$$
X'\hat u=0
$$

## 结论

OLS 的一阶条件要求残差与每一个解释变量正交。矩阵公式 $\hat\beta=(X'X)^{-1}X'y$ 来自这个正交条件。

## 关联卡片

- [[OLS Basics]]
- [[OLS Estimator]]
- [[Residual]]
- [[Linear Regression Model]]

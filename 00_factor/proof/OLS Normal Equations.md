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

## 适用边界

- 一阶条件在二次目标函数下也是充分条件，因为 $X'X$ 半正定；严格凸性和唯一解还需要 $X$ 满列秩。
- 若含截距，正交条件包括残差和常数列正交，即残差和为零；无截距模型不能机械套用均值分解。
- $X'X$ 病态时不应显式求逆，数值实现优先使用 QR 或 SVD，并报告条件数。

## 复现规范

保存设计矩阵列顺序、是否含截距、缺失值处理和求解器；复核 $\|X'\hat u\|$ 是否接近零，并用 QR/SVD 结果与正规方程结果交叉检查。

## 关联卡片

- [[OLS Basics]]
- [[OLS Estimator]]
- [[Residual]]
- [[Linear Regression Model]]

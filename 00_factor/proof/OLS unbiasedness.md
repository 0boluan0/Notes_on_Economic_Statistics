---
aliases:
- OLS unbiasedness
- OLS 无偏性
- OLS估计量的无偏性
tags:
- proof
- econometrics
---
# OLS unbiasedness

## 假设

线性模型：

$$
y=X\beta+u
$$

OLS 估计量：

$$
\hat\beta=(X'X)^{-1}X'y
$$

核心条件：

- $X$ 满列秩。
- 条件外生性：$E[u\mid X]=0$。

## 推导链

代入模型：

$$
\hat\beta
=(X'X)^{-1}X'(X\beta+u)
=\beta+(X'X)^{-1}X'u
$$

在给定 $X$ 条件下取期望：

$$
E[\hat\beta\mid X]
=\beta+(X'X)^{-1}X'E[u\mid X]
$$

由 $E[u\mid X]=0$：

$$
E[\hat\beta\mid X]=\beta
$$

因此：

$$
E[\hat\beta]=\beta
$$

## 结论

在满秩和条件外生性下，OLS 估计量有限样本无偏。

## 边界

- 若解释变量与误差项相关，通常产生偏误。
- 无偏性是有限样本性质；大样本是否收敛看 [[OLS consistency]]。

## 关联卡片

- [[OLS Estimator]]
- [[Endogeneity]]
- [[Gauss-Markov theorem]]

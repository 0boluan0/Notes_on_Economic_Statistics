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

## 适用边界

- 条件外生性 $E[u\mid X]=0$ 比 $E[X'u]=0$ 更强；后者只足以支持某些无条件矩推导，不能保证给定样本设计下无偏。
- 随机设计下需要区分条件无偏 $E(\hat\beta\mid X)=\beta$ 与无条件无偏；固定设计下则把 $X$ 视为已知常数。
- 异方差不必然破坏无偏性，但会改变方差和经典推断；内生性则直接破坏关键期望为零的步骤。

## 复现规范

明确 $X$ 是固定还是随机、是否含截距、缺失值规则和标准误类型；在结果表中同时标注无偏性假设与稳健推断方式。

## 关联卡片

- [[OLS Estimator]]
- [[Endogeneity]]
- [[Gauss-Markov theorem]]

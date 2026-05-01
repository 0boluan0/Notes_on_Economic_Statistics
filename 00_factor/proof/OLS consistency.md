---
aliases:
- OLS consistency
- OLS 一致性
- OLS估计量的一致性
tags:
- proof
- econometrics
---
# OLS consistency

## 假设

线性模型：

$$
y_i=x_i'\beta+u_i
$$

核心大样本条件：

- $E[x_i u_i]=0$。
- $\frac{1}{n}X'X\xrightarrow{p}Q_{xx}$，且 $Q_{xx}$ 满秩可逆。
- 合适的大数定律适用。

## 推导链

从 OLS 分解出发：

$$
\hat\beta-\beta=(X'X)^{-1}X'u
$$

同时乘除 $n$：

$$
\hat\beta-\beta
=\left(\frac{1}{n}X'X\right)^{-1}
\left(\frac{1}{n}X'u\right)
$$

由大数定律：

$$
\frac{1}{n}X'X\xrightarrow{p}Q_{xx}
$$

以及外生性：

$$
\frac{1}{n}X'u\xrightarrow{p}E[x_i u_i]=0
$$

由连续映射定理：

$$
\left(\frac{1}{n}X'X\right)^{-1}\xrightarrow{p}Q_{xx}^{-1}
$$

所以：

$$
\hat\beta-\beta\xrightarrow{p}Q_{xx}^{-1}\cdot 0=0
$$

## 结论

$$
\hat\beta\xrightarrow{p}\beta
$$

OLS 在上述条件下一致。

## 边界

- 一致性不要求有限样本完全无偏，但要求样本扩大后偏差消失。
- 内生性会导致 $\frac{1}{n}X'u$ 不收敛到 0，从而破坏一致性。

## 关联卡片

- [[OLS Estimator]]
- [[OLS unbiasedness]]
- [[Law of Large Numbers]]
- [[Continuous Mapping Theorem]]
- [[Endogeneity]]

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

## 适用边界

- 这里需要样本矩阵的极限 $Q_{xx}$ 非奇异；仅有 $E[x_i u_i]=0$ 但没有大数定律或稳定抽样结构，并不足以推出一致性。
- 时间序列、面板或聚类数据需要相应的依赖结构大数定律，不能默认 i.i.d. 证明原样成立。
- 一致性不保证有限样本无偏，也不保证渐近正态；后者还需要中心极限定理和有限矩条件。

## 复现规范

报告样本独立性/依赖性假设、矩条件、固定效应或趋势处理、样本量，并配套给出渐近标准误和收敛诊断；不要用单次大样本结果替代识别假设。

## 关联卡片

- [[OLS Estimator]]
- [[OLS unbiasedness]]
- [[Law of Large Numbers]]
- [[Continuous Mapping Theorem]]
- [[Endogeneity]]

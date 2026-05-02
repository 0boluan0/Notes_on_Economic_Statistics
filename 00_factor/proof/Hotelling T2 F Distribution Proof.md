---
aliases:
- Hotelling T2 F Distribution Proof
- Hotelling T² F 分布推导
tags:
- proof
- multivariate statistics
---
# Hotelling T2 F Distribution Proof

## 假设

$$
X_1,\ldots,X_n\sim N_p(\mu,\Sigma),
\qquad H_0:\mu=\mu_0.
$$

定义
$$
T^2=n(\bar X-\mu_0)'S^{-1}(\bar X-\mu_0).
$$

## 推导链

在 $H_0$ 下，
$$
\sqrt n(\bar X-\mu_0)\sim N_p(0,\Sigma).
$$

正态总体下，样本均值与样本协方差矩阵独立，且
$$
(n-1)S\sim W_p(n-1,\Sigma).
$$

令
$$
Z=\sqrt n\,\Sigma^{-1/2}(\bar X-\mu_0),
$$
则
$$
Z\sim N_p(0,I_p).
$$

同时，Wishart 部分给出样本协方差对 $\Sigma$ 的随机估计。把正态二次型和独立 Wishart 矩阵组合后，可得到 Hotelling 统计量与 $F$ 分布的关系：
$$
\frac{n-p}{p(n-1)}T^2\sim F_{p,n-p}.
$$

## 结论

等价地，
$$
T^2\sim \frac{p(n-1)}{n-p}F_{p,n-p}.
$$

这就是单样本均值向量检验的拒绝域来源。

## 来自课程位置

- [[05_ 总体平均向量的推论#1.2. 均值向量的假设检验|第5章 2 Hotelling T² 检验]]

## 关联卡片

- [[Hotelling T2 Test]]
- [[Wishart Distribution]]
- [[Multivariate Normal Distribution]]

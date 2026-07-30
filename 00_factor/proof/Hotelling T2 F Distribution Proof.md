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

## 适用边界

- 需要 $n>p$ 以保证 $S^{-1}$ 存在；小样本下还依赖多元正态假设，重尾或异常值会显著改变有限样本分布。
- 这里的 $S$ 是以 $n-1$ 为分母的无偏样本协方差矩阵；若软件使用其他归一化，必须同步调整统计量。
- 多个均值向量比较、协方差不齐或高维 $p\ge n$ 时，不能直接使用单样本 Hotelling $T^2$ 的 F 转换。

## 复现规范

记录 $n,p$、协方差矩阵的定义、$\mu_0$、显著性水平和正态性/异常值检查；报告 $T^2$、转换后的 F 值、自由度与 p 值，而不是只报告是否显著。

## 来自课程位置

- [[05_ 总体平均向量的推论#1.2. 均值向量的假设检验|第5章 2 Hotelling T² 检验]]

## 关联卡片

- [[Hotelling T2 Test]]
- [[Wishart Distribution]]
- [[Multivariate Normal Distribution]]

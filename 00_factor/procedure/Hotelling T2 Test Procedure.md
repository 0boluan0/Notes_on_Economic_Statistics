---
aliases:
- Hotelling T2 Test Procedure
- Hotelling T² 检验步骤
tags:
- procedure
- multivariate statistics
type: procedure
---
# Hotelling T2 Test Procedure

## 输入

- 样本 $X_1,\ldots,X_n$，每个为 $p$ 维向量。
- 目标均值向量 $\mu_0$。
- 显著性水平 $\alpha$。

## 输出

- $T^2$ 统计量。
- $F$ 转换后的拒绝判断。
- 对 $H_0:\mu=\mu_0$ 的结论。

## Step 1. 计算样本矩

$$
\bar X=\frac1n\sum_{j=1}^nX_j,\qquad
S=\frac1{n-1}\sum_{j=1}^n(X_j-\bar X)(X_j-\bar X)'.
$$

## Step 2. 计算统计量

$$
T^2=n(\bar X-\mu_0)'S^{-1}(\bar X-\mu_0).
$$

## Step 3. 转成 F 检验

$$
F=\frac{n-p}{p(n-1)}T^2\sim F_{p,n-p}.
$$

拒绝域：
$$
F>F_{p,n-p}(1-\alpha).
$$

## 检查点

- $n>p$，否则 $S$ 可能不可逆。
- 样本来自多元正态或样本量足够大。
- $\mu_0$ 维度必须和 $\bar X$ 一致。

## 常见错误

- 把 $S/n$ 和 $S$ 的逆误用，漏掉前面的 $n$。
- 直接查 $T^2$ 的临界值而忘记转换为 $F$。
- 逐个做 $t$ 检验后当作联合检验。

## 来自课程位置

- [[05_ 总体平均向量的推论#1.2. 均值向量的假设检验|第5章 2 Hotelling T² 检验]]

## 关联卡片

- [[Hotelling T2 Test]]
- [[Confidence Ellipsoid]]
- [[Multivariate Mean Inference Map]]

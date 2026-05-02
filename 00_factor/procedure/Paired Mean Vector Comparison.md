---
aliases:
- Paired Mean Vector Comparison
- Paired Hotelling T2 Test
- 配对均值向量比较
- 配对 Hotelling T2 检验
tags:
- procedure
- multivariate statistics
---
# Paired Mean Vector Comparison

## 输入

- 同一对象在两种处理下的多变量观测 $X_{1j},X_{2j}$。
- 样本量 $n$、维度 $p$、显著性水平 $\alpha$。

## 输出

- 对 $H_0:\delta=0$ 的 Hotelling $T^2$ 检验结论。
- 必要时给出差值分量的同时置信区间。

## Step 1. 构造差值向量

$$
D_j=X_{1j}-X_{2j}.
$$

把问题转成单样本均值向量检验：
$$
H_0:E(D_j)=\delta=0.
$$

## Step 2. 计算差值样本矩

$$
\bar D=\frac1n\sum_{j=1}^nD_j,
\qquad
S_d=\frac1{n-1}\sum_{j=1}^n(D_j-\bar D)(D_j-\bar D)'.
$$

## Step 3. 计算统计量

$$
T^2=n\bar D'S_d^{-1}\bar D.
$$

并转换为
$$
\frac{n-p}{p(n-1)}T^2\sim F_{p,n-p}.
$$

## 检查点

- 必须先做配对差值，再对差值做单样本多元检验。
- 配对样本不是两独立样本，不能直接 pooled covariance。

## 常见错误

- 把 $X_1$ 和 $X_2$ 当成独立两样本。
- 差值方向前后不一致，导致均值差解释反了。

## 来自课程位置

- [[06_比较多个均值向量comparisons of multivariate mean vectors#1.2. 配对样本均值向量比较|第6章 2 配对样本均值向量比较]]

## 关联卡片

- [[Hotelling T2 Test]]
- [[Simultaneous Confidence Intervals]]

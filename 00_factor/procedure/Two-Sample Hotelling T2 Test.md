---
aliases:
- Two-Sample Hotelling T2 Test
- Two Sample Mean Vector Comparison
- 两独立样本 Hotelling T2 检验
- 两总体均值向量比较
tags:
- procedure
- multivariate statistics
type: procedure
---
# Two-Sample Hotelling T2 Test

## 输入

- 两个独立样本，样本量 $n_1,n_2$。
- 维度 $p$，显著性水平 $\alpha$。
- 协方差矩阵相等假设是否可接受。

## 输出

- 对 $H_0:\mu_1-\mu_2=0$ 的检验结论。

## Step 1. 判断协方差结构

若假设
$$
\Sigma_1=\Sigma_2=\Sigma,
$$
使用 pooled covariance。

若不相等，不能直接使用标准 pooled 两样本 $T^2$。

## Step 2. 计算 pooled covariance

$$
S_p=\frac{(n_1-1)S_1+(n_2-1)S_2}{n_1+n_2-2}.
$$

## Step 3. 计算统计量

$$
T^2=\frac{n_1n_2}{n_1+n_2}
(\bar X_1-\bar X_2)'S_p^{-1}(\bar X_1-\bar X_2).
$$

## Step 4. 转换为 F

$$
\frac{n_1+n_2-p-1}{p(n_1+n_2-2)}T^2
\sim F_{p,n_1+n_2-p-1}.
$$

## 检查点

- 两样本必须独立。
- $S_p$ 必须可逆。
- 协方差相等假设是这套公式的关键边界。

## 常见错误

- 协方差不等时仍强行 pooled。
- 把 paired comparison 和 two-sample comparison 混用。

## 来自课程位置

- [[06_比较多个均值向量comparisons of multivariate mean vectors#1.4. 两个独立总体均值向量比较|第6章 3 两个总体均值向量比较]]

## 关联卡片

- [[Hotelling T2 Test]]
- [[Multivariate Mean Inference Map]]

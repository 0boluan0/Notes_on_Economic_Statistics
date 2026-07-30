---
aliases:
- Factor Analysis PC Method
- Principal Component Method for Factor Analysis
- 因子分析主成分法
tags:
- procedure
- multivariate statistics
type: procedure
---
# Factor Analysis PC Method

## 输入

- 样本协方差矩阵 $S$ 或相关矩阵 $R$。
- 希望提取的因子数 $m$。

## 输出

- 因子载荷矩阵 $\hat L$。
- 特殊方差估计 $\hat\Psi$。
- 公共度估计 $h_i^2$。

## Step 1. 做特征值分解

$$
S=\sum_{j=1}^p\lambda_j e_je_j'.
$$

按特征值从大到小排序。

## Step 2. 保留前 m 个因子

构造载荷矩阵
$$
\hat L=
\left[
\sqrt{\lambda_1}e_1,\ldots,\sqrt{\lambda_m}e_m
\right].
$$

## Step 3. 计算公共度

对第 $i$ 个变量：
$$
h_i^2=\sum_{j=1}^m \hat l_{ij}^2.
$$

## Step 4. 计算特殊方差

$$
\hat\psi_i=s_{ii}-h_i^2.
$$

## 检查点

- $\hat\psi_i$ 不应为明显负数。
- 因子数 $m$ 过少会导致共同结构拟合差。
- 载荷矩阵需要结合变量含义解释，必要时旋转。

## 常见错误

- 把 PCA 的主成分直接当成因子解释。
- 忘记区分公共度和特殊方差。
- 把 $L$ 的行列方向写反。

## 来自课程位置

- [[09_因子分析Factor Analysis and Inference for Structured#1.5.1. 主成分法（Principal Component Method）|第9章 3.1 主成分法]]

## 关联卡片

- [[Factor Analysis]]
- [[Factor Loadings]]
- [[Communality]]
- [[Specific Variance]]

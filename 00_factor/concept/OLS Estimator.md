---
aliases:
- OLS Estimator
- Ordinary Least Squares Estimator
- OLS估计量
tags:
- concept
- econometrics
---
# OLS Estimator

## 先记一句话

OLS Estimator 是最小化残差平方和得到的系数估计量。

## 它是什么

在 $X'X$ 可逆时：

$$
\hat\beta_{OLS}=(X'X)^{-1}X'y
$$

代入 $y=X\beta+u$：

$$
\hat\beta_{OLS}=\beta+(X'X)^{-1}X'u
$$

这条分解是理解偏误、一致性和方差的核心。

## 解决什么判断

它回答：“样本数据会生成什么随机估计量，以及误差项如何进入估计误差？”

## 最小例子

简单回归中，斜率估计量 $\hat\beta_1$ 是样本协方差除以样本方差；矩阵公式是多元版本。

## 易混点

- Estimator 是随机规则；estimate 是某个样本算出的具体数值。
- $X'X$ 不可逆时，OLS 系数不能唯一估计，通常是完全多重共线性。
- $\hat\beta$ 的性质要看假设：无偏见 [[OLS unbiasedness]]，一致见 [[OLS consistency]]。

## 来自课程位置

- [[05_多元回归模型的矩阵表达]]
- [[09_联立方程模型(内生性)]]

## 关联卡片

- [[OLS Basics]]
- [[OLS Normal Equations]]
- [[Gauss-Markov theorem]]
- [[Multicollinearity]]

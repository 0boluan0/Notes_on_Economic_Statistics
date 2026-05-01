---
aliases:
- OLS Basics
- Ordinary Least Squares
- OLS基础
- 最小二乘法基础
- OLS
tags:
- concept
- econometrics
---
# OLS Basics

## 先记一句话

OLS 是通过最小化残差平方和来估计线性回归参数的方法。

## 它是什么

给定线性模型：

$$
y=X\beta+u
$$

OLS 选择 $\hat\beta$ 最小化：

$$
RSS(\beta)=(y-X\beta)'(y-X\beta)
$$

当 $X'X$ 可逆时：

$$
\hat\beta=(X'X)^{-1}X'y
$$

## 解决什么判断

它回答：“在给定线性回归模型和样本数据时，怎样选择一组系数让残差平方和最小？”

## 最小例子

估计教育年限对工资的关系时，OLS 选择截距和斜率，让样本中预测工资与实际工资的平方误差之和最小。

## 易混点

- OLS 是估计方法，不等于“因果识别已经成立”。
- OLS 系数能否无偏、一致，取决于外生性等假设，见 [[OLS unbiasedness]] 和 [[OLS consistency]]。
- OLS 的最优性是条件性的，见 [[Gauss-Markov theorem]]。

## 来自课程位置

- [[02_一元线性回归]]
- [[03_多元线性回归]]
- [[05_多元回归模型的矩阵表达]]

## 关联卡片

- [[Linear Regression Model]]
- [[OLS Estimator]]
- [[OLS Estimation Steps]]
- [[OLS Normal Equations]]
- [[Residual]]

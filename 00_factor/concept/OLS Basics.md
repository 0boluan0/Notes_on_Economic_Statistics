---
aliases:
- OLS Basics
- Ordinary Least Squares
- OLS基础
- 最小二乘法基础
- OLS
tags:
- concept
---
# OLS Basics

OLS（Ordinary Least Squares）是在线性回归中通过最小化残差平方和来估计参数的方法。

## 定义

给定模型 $y = X\beta + u$，OLS 选择 $\hat{\beta}$ 使得：
$$
\hat{\beta} = \arg\min_{\beta} \sum_{i=1}^n (y_i - x_i' \beta)^2
$$

## 估计量形式

当 X'X 可逆时：
$$
\hat{\beta} = (X'X)^{-1}X'y
$$

## 最小例子

简单回归 $y_i = \beta_0 + \beta_1 x_i + u_i$，OLS 选择 $\beta_0, \beta_1$ 使残差平方和最小。

## 关键含义

- 误差平方惩罚更重，强调“大误差”被更强地压制。
- 结果等价于使残差与解释变量正交（$X'\hat{u} = 0$）。

## 相关链接

- [[OLS Estimator|OLS估计量]]
- [[OLS Estimation Steps|OLS估计步骤]]
- [[OLS unbiasedness|OLS估计量的无偏性]]
- [[OLS consistency|OLS估计量的一致性]]

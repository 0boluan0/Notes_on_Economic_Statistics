---
aliases:
- Multivariate Linear Regression
- 多元线性回归
tags:
- concept
- multivariate statistics
- econometrics
---
# Multivariate Linear Regression

>[!note] 一句话记忆
> 多元线性回归用多个解释变量线性解释一个响应变量，是矩阵形式最自然的回归模型。

## 它是什么

$$
Y=X\beta+\epsilon.
$$

OLS 估计量为
$$
\hat\beta=(X'X)^{-1}X'Y.
$$

## 解决什么判断

- 多个自变量如何共同解释一个因变量。
- 回归系数如何用矩阵公式估计。
- 拟合值、残差、$R^2$ 和显著性检验如何组织。

## 最小例子

用收入、年龄、教育年限解释消费支出。

## 易混点

- “多元线性回归”在中文里常指多个解释变量、一个因变量；英文 multivariate regression 有时指多个因变量，要看课程语境。
- OLS 的矩阵公式需要 $X'X$ 可逆。

## 来自课程位置

- [[07_多元线性回归multivariate linear regression#1. 第7章：多元线性回归（Multivariate Linear Regression）|第7章 多元线性回归]]

## 关联卡片

- [[OLS Basics]]
- [[OLS Estimator]]
- [[Residual]]
- [[F-test]]

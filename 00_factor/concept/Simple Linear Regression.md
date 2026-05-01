---
aliases:
- Simple Linear Regression
- SLR
- 一元线性回归
tags:
- concept
- econometrics
---
# Simple Linear Regression

## 先记一句话

一元线性回归用一个解释变量描述因变量的条件均值。

## 它是什么

总体模型：

$$
Y_i=\beta_0+\beta_1X_i+u_i
$$

条件均值：

$$
E(Y_i\mid X_i)=\beta_0+\beta_1X_i
$$

## 解决什么判断

它回答：“一个解释变量变化时，被解释变量的平均水平如何变化？”

## 最小例子

消费函数：

$$
C_i=\beta_0+\beta_1Y_i+u_i
$$

其中 $\beta_1$ 可解释为 [[Marginal Propensity to Consume]]。

## 易混点

- 一元回归只有一个解释变量，不等于因果关系自动成立。
- $\beta_1$ 是条件均值斜率，不是每个个体的实际变化。
- 多个解释变量时转入 [[Multiple Linear Regression]]。

## 来自课程位置

- [[02_一元线性回归]]

## 关联卡片

- [[Linear Regression Model]]
- [[OLS Basics]]
- [[Residual]]
- [[t Test]]

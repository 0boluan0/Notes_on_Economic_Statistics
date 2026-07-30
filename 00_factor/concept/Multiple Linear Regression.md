---
aliases:
- Multiple Linear Regression
- MLR
- 多元线性回归
tags:
  - concept
  - econometrics
---
# Multiple Linear Regression

## 先记一句话

多元线性回归用多个解释变量估计“控制其他变量后”的偏效应。

## 它是什么

$$
Y_i=\beta_0+\beta_1X_{1i}+\cdots+\beta_kX_{ki}+u_i
$$

或矩阵形式：

$$
Y=X\beta+u
$$

## 解决什么判断

它回答：“在其他变量不变的条件下，某个解释变量对 $Y$ 的条件均值有什么边际影响？”

## 最小例子

工资方程：

$$
\log wage_i=\beta_0+\beta_1education_i+\beta_2experience_i+u_i
$$

$\beta_1$ 是控制经验后教育的偏效应。

## 易混点

- 多元回归的系数是偏效应，不是简单相关。
- 多放变量不一定更好，可能引入 [[Multicollinearity]] 或过度拟合。
- 估计和推断仍依赖 [[OLS Basics]]、外生性和误差结构假设。

## 来自课程位置

- [[03_多元线性回归]]
- [[05_多元回归模型的矩阵表达]]

## 关联卡片

- [[Linear Regression Model]]
- [[OLS Estimation Steps]]
- [[F-test]]
- [[R-squared]]

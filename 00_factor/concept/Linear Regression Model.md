---
aliases:
- Linear Regression Model
- 线性回归模型
- linear model
tags:
- concept
- econometrics
---
# Linear Regression Model

## 先记一句话

线性回归模型用参数线性的方式描述因变量与解释变量之间的条件均值关系。

## 它是什么

矩阵形式：

$$
y=X\beta+u
$$

其中 $y$ 是因变量向量，$X$ 是解释变量矩阵，$\beta$ 是待估参数，$u$ 是误差项。

## 解决什么判断

它回答：“能否把研究问题写成一个关于参数线性的回归方程，并用样本估计系数？”

## 最小例子

工资方程：

$$
\log(wage_i)=\beta_0+\beta_1 education_i+u_i
$$

## 易混点

- “线性”指参数线性，不要求变量本身只能一次项；$\log(x)$、$x^2$ 也可以作为解释变量进入线性模型。
- 模型定义不等于估计方法；最常见估计方法是 [[OLS Basics|OLS]]。
- 因果解释需要外生性，不能只靠回归形式。

## 来自课程位置

- [[02_一元线性回归]]
- [[03_多元线性回归]]
- [[05_多元回归模型的矩阵表达]]

## 关联卡片

- [[OLS Basics]]
- [[OLS Estimator]]
- [[Gauss-Markov theorem]]
- [[Endogeneity]]

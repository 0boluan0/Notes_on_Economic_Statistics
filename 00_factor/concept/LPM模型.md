---
aliases:
  - Linear Probability Model
  - 线性概率模型
tags:
  - 计量经济学
  - 离散选择模型
---

LPM模型（Linear Probability Model，线性概率模型）是用OLS估计二元被解释变量（取值为0或1）的模型。

## 模型形式

$$y_i = \beta_0 + \beta_1 x_{i1} + \cdots + \beta_k x_{ik} + \varepsilon_i$$

其中y_i ∈ {0, 1}是二元变量。

## 解释

在给定x_i条件下，y_i的条件期望为：

$$E(y_i | x_i) = \beta_0 + \beta_1 x_{i1} + \cdots + \beta_k x_{ik}$$

这是事件发生的概率：
$$P(y_i = 1 | x_i) = \beta_0 + \beta_1 x_{i1} + \cdots + \beta_k x_{ik}$$

## 优点

1. **简单直观**：OLS估计，易于理解和计算
2. **边际效应直接**：系数直接表示x_j变化1个单位对概率的影响

## 缺点

### 1. 预测概率可能超出[0,1]范围

虽然理论上概率应在[0,1]之间，但LPM的预测值可能小于0或大于1。

### 2. 误差项非正态

y_i只能取0或1，因此ε_i只能取两个值：
- 当y_i = 1时，ε_i = 1 - X_iβ
- 当y_i = 0时，ε_i = -X_iβ

不服从正态分布。

### 3. 异方差

$$\textVar}(\varepsilon_i | x_i) = P(y_i=1)P(y_i=0) = P(y_i=1)[1-P(y_i=1)] = X_i\beta[1-X_i\beta]$$

方差依赖于x_i，存在异方差。

### 4. 不可能存在完全预测

如果存在某些x使得P(y=1|x) = 0或1，对应的x值不能有异方差，违反假设。

## 改进方法

### 1. 异方差稳健标准误

使用怀特标准误（White Standard Errors）修正t检验。

### 2. 加权最小二乘（WLS）

使用1/[Xβ(1-Xβ)]作为权重。

### 3. 检验能力度量：R_p²

$$R_p^2 = \frac{1}{\sum y_i} \sum (y_i - \hat{y}_i)^2 + \frac{1}{\sum (1-y_i)} \sum (y_i - \hat{y}_i)^2$$

## 模型选择

LPM适用于：
- 仅关注边际效应
- 样本中预测值不太接近0或1的边界
- 作为其他模型的初步分析

通常使用logit或probit模型替代LPM。

相关链接: [[logit模型]], [[probit模型]]

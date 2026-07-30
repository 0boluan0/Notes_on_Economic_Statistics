---
aliases:
- Endogeneity
- 内生性
- 内生性问题
tags:
- concept
- econometrics
---

# Endogeneity

## 先记一句话

内生性是解释变量和误差项相关，导致 OLS 估计因果参数有偏且不一致。

## 它是什么

核心问题是：

$$
E[u\mid X]\ne 0
$$

或至少：

$$
Cov(X,u)\ne 0
$$

这会让 OLS 分解式中的 $(X'X)^{-1}X'u$ 不再消失。

## 解决什么判断

它回答：“回归系数能不能解释成因果效应，还是只是在混合相关性和遗漏因素？”

## 最小例子

估计教育对工资的影响时，若能力影响教育也影响工资，但模型遗漏能力，教育变量就和误差项相关，OLS 教育回报率有偏。

## 易混点

- 内生性会破坏一致性；[[Heteroskedasticity]] 和 [[Autocorrelation]] 在外生性仍成立时通常主要破坏标准误和效率。
- 加控制变量能缓解遗漏变量，但不能保证所有内生性消失。
- 常见处理是 [[Instrumental Variable]]、[[2SLS]]、固定效应、DID、RDD 或自然实验。

## 来自课程位置

- [[09_联立方程模型(内生性)]]
- [[04_模型设定]]

## 关联卡片

- [[Endogeneity Diagnosis]]
- [[Instrumental Variable]]
- [[2SLS]]
- [[Hausman Test]]
- [[Simultaneity Bias]]

## 边界与相关概念

本卡片只保留该概念的定义、记号与最小直觉；具体估计步骤、证明或完整应用应放在对应的 `procedure`、`proof` 或 `framework` 卡片中。

相关卡片：[[Heteroskedasticity]]、[[Autocorrelation]]、[[Instrumental Variable]]、[[2SLS]]、[[09_联立方程模型(内生性)]]、[[04_模型设定]]、[[Endogeneity Diagnosis]]、[[Hausman Test]]。

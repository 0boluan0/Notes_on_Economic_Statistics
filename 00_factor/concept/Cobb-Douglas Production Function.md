---
aliases:
- Cobb-Douglas Production Function
- Cobb-Douglas
- CD Production Function
- CD生产函数
tags:
- concept
- economics
- econometrics
---

# Cobb-Douglas Production Function

## 先记一句话

Cobb-Douglas 生产函数用资本和劳动的幂函数形式描述产出。

## 它是什么

$$
Y=AK^\alpha L^\beta
$$

取对数后可线性化：

$$
\ln Y=\ln A+\alpha\ln K+\beta\ln L+u
$$

## 解决什么判断

它回答：“资本和劳动分别对产出有多大弹性？”

## 最小例子

若 $\alpha=0.3$，资本增加 1%，产出约增加 0.3%。

## 易混点

- $\alpha+\beta=1$ 表示规模报酬不变。
- 取对数后可用 [[Multiple Linear Regression]] 估计。
- 生产函数参数能否因果解释，仍取决于识别假设。

## 来自课程位置

- [[03_多元线性回归]]

## 关联卡片

- [[Multiple Linear Regression]]
- [[F-test]]
- [[Omitted Variable Bias]]

## 边界与相关概念

本卡片只保留该概念的定义、记号与最小直觉；具体估计步骤、证明或完整应用应放在对应的 `procedure`、`proof` 或 `framework` 卡片中。

相关卡片：[[Multiple Linear Regression]]、[[03_多元线性回归]]、[[F-test]]、[[Omitted Variable Bias]]。

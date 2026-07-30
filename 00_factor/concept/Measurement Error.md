---
aliases:
- Measurement Error
- Errors-in-Variables
- 测量误差
- 变量测量误差
tags:
  - concept
  - econometrics
---
# Measurement Error

## 先记一句话

测量误差指观测变量不等于真实变量；解释变量测量误差通常比被解释变量测量误差更危险。

## 它是什么

若真实解释变量为 $X_i$，但观测到：

$$
X_i^*=X_i+v_i
$$

用 $X_i^*$ 回归时，误差项会包含 $v_i$，导致解释变量和误差项相关。

## 解决什么判断

它回答：“数据记录不准会不会破坏系数估计？”

## 最小例子

用自报收入解释消费。若收入自报误差很大，消费函数斜率可能被压低。

## 易混点

- 解释变量测量误差常导致系数有偏且不一致。
- 被解释变量测量误差若与解释变量独立，通常主要增大误差方差。
- 测量误差不是样本量变大就自动消失的问题。

## 来自课程位置

- [[04_模型设定]]

## 关联卡片

- [[Endogeneity]]
- [[Omitted Variable Bias]]
- [[Model Misspecification]]

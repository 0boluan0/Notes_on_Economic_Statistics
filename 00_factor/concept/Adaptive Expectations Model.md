---
aliases:
- Adaptive Expectations Model
- Adaptive Expectations
- 适应性预期模型
tags:
- concept
- econometrics
- time-series
---
# Adaptive Expectations Model

## 先记一句话

适应性预期模型假设主体根据上一期预测误差逐步修正本期预期。

## 它是什么

$$
P_t^e=P_{t-1}^e+\lambda(P_{t-1}-P_{t-1}^e),
\qquad 0<\lambda\le1
$$

$\lambda$ 是调整速度。

## 解决什么判断

它回答：“预期变量不可观测时，能否用过去预测误差刻画预期调整？”

## 最小例子

若上一期实际通胀高于预期通胀，本期预期通胀会上调一部分。

## 易混点

- 适应性预期不是理性预期；它只用过去误差修正。
- 引入适应性预期后，模型常可转化为 [[ADL]] 形式。

## 来自课程位置

- [[11_平稳时间序列模型]]

## 关联卡片

- [[ADL]]
- [[Distributed Lag Model]]
- [[Partial Adjustment Model]]

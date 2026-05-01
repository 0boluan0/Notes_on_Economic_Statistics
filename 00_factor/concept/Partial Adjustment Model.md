---
aliases:
- Partial Adjustment Model
- 部分调整模型
tags:
- concept
- econometrics
- time-series
---
# Partial Adjustment Model

## 先记一句话

部分调整模型假设经济变量不会立刻达到目标值，而是每期只完成一部分调整。

## 它是什么

$$
Y_t-Y_{t-1}=\lambda(Y_t^*-Y_{t-1}),
\qquad 0<\lambda\le1
$$

等价于：

$$
Y_t=(1-\lambda)Y_{t-1}+\lambda Y_t^*
$$

## 解决什么判断

它回答：“实际变量如何逐步靠近目标水平？”

## 最小例子

企业目标资本存量上升后，实际投资不会一期内全部完成，而是逐步调整。

## 易混点

- $\lambda$ 是调整速度，不是长期效应本身。
- 模型会自然产生滞后因变量，因此估计时要关注自相关和内生性。

## 来自课程位置

- [[11_平稳时间序列模型]]

## 关联卡片

- [[ADL]]
- [[Adaptive Expectations Model]]
- [[Durbin h Test]]

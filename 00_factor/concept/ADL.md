---
aliases:
- ADL
- Autoregressive Distributed Lag Model
- 自回归分布滞后模型
tags:
- concept
- econometrics
- time-series
---
# ADL

## 先记一句话

ADL 模型同时放入被解释变量滞后项和解释变量当前及滞后项，用来描述动态调整关系。

## 它是什么

$$
y_t=\alpha+\sum_{i=1}^{p}\phi_i y_{t-i}
+\sum_{j=0}^{q}\beta_j x_{t-j}+\varepsilon_t
$$

其中 $p$ 是 $y$ 的滞后阶数，$q$ 是 $x$ 的滞后阶数。

## 解决什么判断

它回答：“$x$ 的影响是否分期传导，且 $y$ 是否有自身惯性？”

## 最小例子

消费不仅受本期收入影响，也受上期消费惯性影响：

$$
C_t=\alpha+\phi C_{t-1}+\beta_0Y_t+\beta_1Y_{t-1}+\varepsilon_t
$$

## 易混点

- [[Distributed Lag Model]] 只放解释变量滞后项；ADL 还放被解释变量滞后项。
- 含滞后被解释变量后，若误差项自相关，OLS 会有偏且不一致。
- 动态模型中的自相关检验不能机械使用 DW，应看 [[Durbin h Test]] 或 [[Breusch-Godfrey Test]]。

## 来自课程位置

- [[11_平稳时间序列模型]]

## 关联卡片

- [[Distributed Lag Model]]
- [[Adaptive Expectations Model]]
- [[Partial Adjustment Model]]
- [[Durbin h Test]]

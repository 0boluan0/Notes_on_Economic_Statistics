---
aliases:
- ADL
- Autoregressive Distributed Lag Model
- ADL Model
- 自回归分布滞后模型
tags:
- concept
- 时间序列
- 计量经济学
---
# ADL

## 先记一句话

ADL 是：**同时用被解释变量自己的滞后项和解释变量的当期/滞后项来描述动态调整**。

## 它是什么

常见写法：
$$
y_t=\alpha+\sum_{i=1}^{p}\phi_i y_{t-i}+\sum_{j=0}^{q}\beta_jx_{t-j}+\varepsilon_t.
$$

其中 $y$ 的滞后项刻画自身惯性，$x$ 的当期和滞后项刻画外部输入的传导。

## 它解决什么判断

ADL 回答：

- $x$ 的影响是否分期传导；
- $y$ 是否有自身惯性；
- 短期效应和长期效应如何区分；
- 哪些滞后项应进入模型。

## 最小例子

消费模型：
$$
C_t=\alpha+\phi C_{t-1}+\beta_0Y_t+\beta_1Y_{t-1}+\varepsilon_t.
$$

这表示消费既有自身惯性，也受本期和上期收入影响。

## 易混点

- [[Distributed Lag Model]] 只放解释变量滞后项；ADL 还放被解释变量滞后项。
- ADL 与 [[Transfer Function Model]] 很接近，但传递函数更强调输入到输出的滞后结构。
- 含滞后被解释变量后，自相关检验不能机械依赖 DW，应看 [[Durbin h Test]] 或 [[Breusch-Godfrey Test]]。

## 来自课程位置

- [[05_多方程模型Multi-equation Time Series Models#3.1. ADL的定义|时间序列 05：ADL 定义]]

## 关联卡片

- [[Distributed Lag Model]]
- [[Transfer Function Model]]
- [[Cross-Correlation Function]]
- [[Leading Indicator]]
- [[Durbin h Test]]

---
aliases:
- Granger Causality Test
- Granger Causality
- 格兰杰因果检验
- 格兰杰因果关系检验
tags:
- concept
- 时间序列
- 计量经济学
---
# Granger Causality Test

## 先记一句话

格兰杰因果检验就是：**看一个变量的过去信息是否能显著改善另一个变量的预测**。

## 它是什么

若加入 $x$ 的滞后项后能更好预测 $y$，则称 $x$ Granger-cause $y$。

在 VAR 中，检验“$x$ 是否格兰杰导致 $y$”通常就是检验 $y_t$ 方程里 $x$ 的所有滞后系数是否同时为 0。

## 它解决什么判断

- 某变量是否提供预测信息。
- VAR 中哪些滞后项应保留。
- 动态系统中信息传导方向。

## 最小例子

非受限模型：
$$
y_t=\alpha+\sum_i a_iy_{t-i}+\sum_i b_ix_{t-i}+u_t.
$$

原假设：
$$
H_0:b_1=b_2=\cdots=b_p=0.
$$

若拒绝，说明 $x$ 的滞后项对预测 $y$ 有增量信息。

## 易混点

- Granger causality 不是哲学或结构因果，只是预测因果。
- 结果对滞后阶数、平稳性和变量集合敏感。
- 协整系统中还要考虑误差修正项，不能只看差分滞后项。

## 来自课程位置

- [[05_多方程模型Multi-equation Time Series Models#4.5. 格兰杰因果检验|时间序列 05：VAR 中的格兰杰因果]]
- [[07_协整和误差修正模型#2.3. 协整与误差修正模型|时间序列 07：协整系统中的因果判断]]

## 关联卡片

- [[VAR Model]]
- [[Error Correction Model]]
- [[Leading Indicator]]
- [[Forecast Evaluation]]

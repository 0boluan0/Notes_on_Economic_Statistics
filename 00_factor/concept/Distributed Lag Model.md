---
aliases:
- Distributed Lag Model
- DLM
- Distributed Lag
- 分布滞后模型
tags:
- concept
- 时间序列
- 计量经济学
---

# Distributed Lag Model

## 先记一句话

分布滞后模型就是：**让解释变量的影响分散在多个时期逐步体现**。

## 它是什么

基本形式：
$$
y_t=\alpha+\beta_0x_t+\beta_1x_{t-1}+\cdots+\beta_qx_{t-q}+u_t.
$$

每个 $\beta_j$ 表示 $x$ 在滞后 $j$ 期对当前 $y$ 的影响。

## 它解决什么判断

- 政策或冲击是否有滞后效应。
- 当期效应、短期效应和累计效应分别多大。
- 输入变量是否可能是 [[Leading Indicator|领先指标]]。

## 最小例子

货币政策对产出的影响可能不是当期完成，而是在未来若干期逐步显现。

## 易混点

- 分布滞后模型不一定包含 $y_{t-1}$。一旦加入 $y$ 的滞后，就进入 [[ADL]]。
- 滞后项太多容易多重共线，需要结合信息准则和理论约束。
- 解释累计效应时，要把相关滞后系数相加。

## 来自课程位置

- [[05_多方程模型Multi-equation Time Series Models#3.1. ADL的定义|时间序列 05：分布滞后与 ADL]]

## 关联卡片

- [[ADL]]
- [[Transfer Function Model]]
- [[Cross-Correlation Function]]
- [[Leading Indicator]]

## 边界与相关概念

本卡片只保留该概念的定义、记号与最小直觉；具体估计步骤、证明或完整应用应放在对应的 `procedure`、`proof` 或 `framework` 卡片中。

相关卡片：[[Leading Indicator]]、[[ADL]]、[[05_多方程模型Multi-equation Time Series Models]]、[[Transfer Function Model]]、[[Cross-Correlation Function]]。

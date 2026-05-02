---
aliases:
- VAR Model
- VAR
- Vector Autoregression
- Vector Autoregressive Model
- 向量自回归模型
tags:
- concept
- 时间序列
---
# VAR Model

## 先记一句话

VAR 是：**把多个变量都当作内生变量，用它们共同的滞后值解释当前值的多方程时间序列模型**。

## 它是什么

VAR$(p)$ 写作：
$$
x_t=A_0+A_1x_{t-1}+\cdots+A_px_{t-p}+e_t.
$$

$x_t$ 是变量向量，$A_i$ 是系数矩阵，$e_t$ 是简约形式扰动。

## 它解决什么判断

VAR 用来处理多变量动态系统：

- 一个变量的滞后是否预测另一个变量；
- 冲击如何在系统内传导；
- 多变量预测；
- [[Granger Causality Test|格兰杰因果检验]]；
- [[Impulse Response Function|脉冲响应函数]] 和 [[Variance Decomposition|方差分解]]。

## 最小例子

二维 VAR(1)：
$$
\begin{bmatrix}y_t\\z_t\end{bmatrix}
=A_0+A_1\begin{bmatrix}y_{t-1}\\z_{t-1}\end{bmatrix}+e_t.
$$

这表示 $y_t$ 和 $z_t$ 都可以由二者上一期共同解释。

## 稳定性

VAR 稳定要求 companion matrix 的特征值在单位圆内。

在稳定时，VAR 可以写成 [[VMA]] 表示，冲击影响会逐渐衰减。

## 易混点

- [[Reduced Form VAR]] 能估计预测关系，但不能直接解释结构冲击。
- [[Structural VAR]] 需要额外识别约束，才有经济结构含义。
- VAR 不自动解决所有内生性问题，它是把变量系统化地共同建模。

## 来自课程位置

- [[05_多方程模型Multi-equation Time Series Models#4.1. VAR定义|时间序列 05：VAR 定义]]
- [[05_多方程模型Multi-equation Time Series Models#4.2. VAR稳定性和平稳性|时间序列 05：VAR 稳定性]]

## 关联卡片

- [[Reduced Form VAR]]
- [[Structural VAR]]
- [[VMA]]
- [[Impulse Response Function]]
- [[Variance Decomposition]]
- [[Granger Causality Test]]

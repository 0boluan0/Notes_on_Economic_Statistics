---
aliases:
- Reduced Form VAR
- Reduced-form VAR
- Reduced Form Vector Autoregression
- 简约VAR
- 简约向量自回归
tags:
- concept
- 时间序列
---
# Reduced Form VAR

## 先记一句话

简约 VAR 就是：**把同期关系吸收到误差项里，只用滞后变量解释当前变量的 VAR**。

## 它是什么

从结构形式
$$
B x_t=\Gamma_0+\Gamma_1x_{t-1}+\varepsilon_t
$$
左乘 $B^{-1}$ 得到：
$$
x_t=A_0+A_1x_{t-1}+e_t.
$$

其中
$$
e_t=B^{-1}\varepsilon_t.
$$

## 它解决什么判断

- VAR 能否逐方程用 OLS 估计。
- 残差之间为什么可能同期相关。
- 为什么简约形式不能直接给出结构冲击解释。

## 最小例子

二维简约 VAR：
$$
\begin{bmatrix}y_t\\z_t\end{bmatrix}
=A_0+A_1
\begin{bmatrix}y_{t-1}\\z_{t-1}\end{bmatrix}
 + e_t.
$$

每个方程右边都是滞后变量，没有当期另一变量。

## 易混点

- 简约误差 $e_t$ 是预测误差，不一定是经济结构冲击。
- 简约 VAR 可以估计动态预测关系，但要解释冲击来源还要回到 [[Structural VAR]]。
- 残差同期相关不妨碍逐方程 OLS 估计均值方程，但会影响结构解释。

## 来自课程位置

- [[05_多方程模型Multi-equation Time Series Models#4.1. VAR定义|时间序列 05：简约 VAR]]

## 关联卡片

- [[VAR Model]]
- [[Structural VAR]]
- [[VMA]]
- [[Impulse Response Function]]


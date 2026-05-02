---
aliases:
- Structural VAR
- SVAR
- Structural Vector Autoregression
- 结构VAR
- 结构向量自回归
tags:
- concept
- 时间序列
---
# Structural VAR

## 先记一句话

结构 VAR 就是：**把同期变量相互影响和结构冲击显式写出来的 VAR 系统**。

## 它是什么

二维例子：
$$
B x_t=\Gamma_0+\Gamma_1x_{t-1}+\varepsilon_t.
$$

$B$ 描述同期关系，$\varepsilon_t$ 是结构冲击。

## 它解决什么判断

- 哪些变量在同一期直接影响彼此。
- VAR 中的冲击是否能被解释为经济结构冲击。
- 是否可以做有结构含义的 [[Impulse Response Function|脉冲响应分析]]。

## 最小例子

若假定变量 1 当期影响变量 2，但变量 2 当期不反向影响变量 1，就等于给 $B$ 加了一个递归识别约束。

## 易混点

- 简约 VAR 可以用 OLS 估计，但结构 VAR 需要额外识别约束。
- 没有识别约束时，不能把简约残差直接解释为结构冲击。
- Cholesky 识别的结果依赖变量排序。

## 来自课程位置

- [[05_多方程模型Multi-equation Time Series Models#4.1. VAR定义|时间序列 05：结构 VAR]]
- [[05_多方程模型Multi-equation Time Series Models#4.3. VAR 的估计|时间序列 05：Cholesky 识别]]

## 关联卡片

- [[VAR Model]]
- [[Reduced Form VAR]]
- [[VMA]]
- [[Impulse Response Function]]


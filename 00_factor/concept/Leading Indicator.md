---
aliases:
- Leading Indicator
- Leading Variable
- 领先指标
- 先行指标
tags:
- concept
- 时间序列
---
# Leading Indicator

## 先记一句话

领先指标就是：**先于目标变量变化、能帮助预测目标变量未来变化的变量**。

## 它是什么

在传递函数
$$
C(L)=c_0+c_1L+c_2L^2+\cdots
$$
中，如果 $c_0=0$ 但某个滞后系数 $c_j\neq0$，就说明 $z_t$ 不影响当期 $y_t$，但会影响未来的 $y$。

这时 $z_t$ 可以被看作 $y_t$ 的领先指标。

## 它解决什么判断

- 某变量是否对另一个变量有预测价值。
- 影响是否先发生在输入变量，再传导到输出变量。
- 干预、政策或宏观指标是否具有提前信号。

## 最小例子

如果订单量变化领先产出两期，那么订单量可能是产出的领先指标。

## 易混点

- 领先指标不等于因果变量。它可能只是提前反映共同信息。
- 领先关系要和 [[Granger Causality Test]]、经济机制、样本外预测一起判断。
- 当期系数为 0 不代表没有关系，可能只是关系滞后出现。

## 来自课程位置

- [[05_多方程模型Multi-equation Time Series Models#3.1. ADL的定义|时间序列 05：leading indicator]]

## 关联卡片

- [[Transfer Function Model]]
- [[Cross-Correlation Function]]
- [[Granger Causality Test]]


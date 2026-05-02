---
aliases:
- Transfer Function Model
- Transfer Function
- 传递函数模型
- 传递函数
tags:
- concept
- 时间序列
---
# Transfer Function Model

## 先记一句话

传递函数模型就是：**描述外生输入 $z_t$ 的变化如何经过若干滞后传导到输出 $y_t$**。

## 它是什么

常见形式：
$$
y_t=a_0+A(L)y_{t-1}+C(L)z_t+\varepsilon_t.
$$

$C(L)$ 是传递函数：
$$
C(L)=c_0+c_1L+c_2L^2+\cdots.
$$

它说明 $z_t$ 的当期和滞后值分别怎样影响 $y_t$。

## 它解决什么判断

- 输入变量是否有当期效应。
- 输入变量的影响从第几期开始出现。
- 影响是短期尖刺，还是通过 AR 动态逐步衰减。

## 最小例子

若
$$
C(L)=c_2L^2,
$$
说明 $z_t$ 对 $y_t$ 没有当期和一期影响，主要在两期后体现。

## 易混点

- 传递函数不只是分布滞后，还常和 $y_t$ 自身动态一起出现。
- 如果 $z_t$ 本身有自相关，直接看 CCF 可能误判，需要预白化或联合建模。
- $C(L)$ 的识别依赖模型结构，不是纯粹从相关图机械读出来。

## 来自课程位置

- [[05_多方程模型Multi-equation Time Series Models#3.1. ADL的定义|时间序列 05：ADL 与传递函数]]
- [[05_多方程模型Multi-equation Time Series Models#3.4. 识别与估计|时间序列 05：传递函数识别]]

## 关联卡片

- [[ADL]]
- [[Distributed Lag Model]]
- [[Cross-Correlation Function]]
- [[Leading Indicator]]


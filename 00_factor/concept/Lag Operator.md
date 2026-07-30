---
aliases:
- Lag Operator
- Backshift Operator
- L operator
- 滞后算子
- 后移算子
tags:
  - concept
  - 时间序列
---
# Lag Operator

## 先记一句话

滞后算子就是：**把一个时间序列往过去推若干期**。

## 它是什么

定义：
$$
L^i y_t=y_{t-i}.
$$

所以：
$$
Ly_t=y_{t-1},\qquad L^2y_t=y_{t-2}.
$$

算子乘法像指数：
$$
L^iL^j=L^{i+j}.
$$

## 它解决什么判断

- 把差分写成 $\Delta=(1-L)$。
- 把 ARMA/ARIMA 模型写成多项式形式。
- 判断特征根和单位根时，快速从递推式切换到算子多项式。

## 最小例子

AR(1)：
$$
y_t=a_1y_{t-1}+\varepsilon_t
$$
可以写成：
$$
(1-a_1L)y_t=\varepsilon_t.
$$

## 易混点

- $L$ 不是参数，是作用在时间下标上的算子。
- $L^{-1}$ 表示前移，但课程里通常少用，避免把预测和代数展开混在一起。
- 特征根判断时要区分 $a_1$ 和算子多项式根。AR(1) 平稳要求 $|a_1|<1$，等价于 $1-a_1L=0$ 的根在单位圆外。

## 来自课程位置

- [[02_差分方程Difference Equation#1.3. 术语：滞后算子（Lag operator）|时间序列 02：滞后算子]]

## 关联卡片

- [[Difference Operator]]
- [[Difference Equation]]
- [[ARMA]]
- [[AR(1) stationarity]]

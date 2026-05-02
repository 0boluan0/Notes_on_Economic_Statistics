---
aliases:
- Difference Equation
- Difference Equations
- Linear Difference Equation
- 差分方程
- 线性差分方程
tags:
- concept
- 时间序列
---
# Difference Equation

## 先记一句话

差分方程就是：**用一个变量的过去值、外生项和时间来决定它当前取值的离散时间递推关系**。

## 它是什么

时间序列课中常见的线性差分方程写作：
$$
y_t=a_0+\sum_{i=1}^{n}a_iy_{t-i}+x_t.
$$

其中：

- $y_{t-i}$ 是自身滞后项；
- $x_t$ 是外生过程或扰动项；
- $a_i$ 是常系数。

## 它解决什么判断

差分方程负责判断动态系统长期会怎样：

- 是否收敛到稳定特解；
- 是否爆炸；
- 是否出现单位根；
- AR/ARMA 的冲击影响是否会衰减。

## 最小例子

AR(1) 可以看成一阶随机差分方程：
$$
y_t=a_0+a_1y_{t-1}+\varepsilon_t.
$$

若 $|a_1|<1$，过去冲击逐渐衰减；若 $|a_1|\ge 1$，过程不再稳定。

## 易混点

- 差分方程不是差分算子。差分方程是递推关系，[[Difference Operator]] 是运算符。
- 求解差分方程时，要区分齐次解、特解和初始条件。
- 特征根的“在单位圆内/外”要看你使用的是系数形式还是滞后算子多项式。

## 来自课程位置

- [[02_差分方程Difference Equation#2.1. 线性差分方程|时间序列 02：线性差分方程]]
- [[02_差分方程Difference Equation#2.3. 求解差分方程|时间序列 02：求解差分方程]]

## 关联卡片

- [[Difference Operator]]
- [[Lag Operator]]
- [[AR(1) stationarity]]
- [[Autoregressive Model]]
- [[Random Walk]]

---
aliases:
- GPD
- Generalized Pareto Distribution
- 广义帕累托分布
tags:
- concept
- risk-management
- statistics
---
# GPD

## 先记一句话

GPD 是 EVT/POT 方法中用来拟合“超过阈值以后还会超多少”的尾部分布。

## 它是什么

给定高阈值 $u$，若 $Y=L-u\mid L>u$ 是超额损失，GPD 近似为：

$$
P(Y>y)\approx \left(1+\xi\frac{y}{\beta}\right)^{-1/\xi}
$$

其中 $\xi$ 是形状参数，$\beta$ 是尺度参数。

## 解决什么判断

它回答：“已经进入损失尾部以后，继续出现更大损失的概率衰减有多慢？”

## 最小例子

若 500 个损失样本中有 25 个超过阈值 $u$，就用这 25 个超额损失拟合 GPD，再计算 99.9% [[VaR]]。

## 易混点

- $u$ 是阈值；$y=L-u$ 是超额损失，不要把二者混在一起。
- $\xi>0$ 表示厚尾；$\xi$ 越大，极端风险越重。
- GPD 只拟合阈值以上的尾部，不解释整个损失分布。

## 来自课程位置

- [[14_VaR参数法和模拟法]]

## 关联卡片

- [[EVT]]
- [[EVT VaR Calculation]]
- [[VaR]]
- [[ES]]

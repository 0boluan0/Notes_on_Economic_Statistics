---
aliases:
- Autocorrelation
- Serial Correlation
- 自相关
- 序列相关
tags:
- concept
- econometrics
- time-series
---
# Autocorrelation

## 先记一句话

Autocorrelation 是同一个变量或误差项与自己过去值相关。

## 它是什么

滞后 $k$ 阶自相关：

$$
\rho_k=\frac{Cov(x_t,x_{t-k})}{Var(x_t)}
$$

在回归诊断中，更常关心误差项是否满足：

$$
Cov(u_t,u_{t-k}\mid X)=0
$$

## 解决什么判断

它回答：“当前误差或变量是否带着过去的信息，导致观测之间不是独立的？”

## 最小例子

若本期残差为正后，下期残差也更可能为正，说明残差可能存在正自相关。

## 易混点

- 时间序列本身有自相关不一定是问题；回归误差有自相关才会破坏经典标准误。
- 自相关诊断见 [[Autocorrelation Diagnosis]]。
- 同时存在异方差和自相关时，常用 [[Newey-West]] 修正标准误。

## 来自课程位置

- [[08_自相关]]
- [[11_平稳时间序列模型]]

## 关联卡片

- [[Durbin-Watson Statistic]]
- [[Breusch-Godfrey Test]]
- [[Newey-West]]
- [[Cochrane-Orcutt]]

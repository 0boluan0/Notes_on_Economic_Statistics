---
aliases:
- Durbin h Test
- Durbin h检验
- 德宾 h 检验
tags:
- system
- econometrics
- time-series
---
# Durbin h Test

## 诊断目标

在含滞后被解释变量的动态回归中检验一阶自相关。

## 为什么不用 DW

若模型包含 $Y_{t-1}$，[[Durbin-Watson Statistic]] 不再适合直接判断误差自相关，此时可用 Durbin h。

## 统计量

$$
h=\left(1-\frac{d}{2}\right)
\sqrt{\frac{n}{1-n\operatorname{Var}(\hat\beta_{lag})}}
$$

其中 $d$ 是 DW 统计量，$\hat\beta_{lag}$ 是滞后被解释变量的系数估计。

## 检查点

- 根号内必须为正，否则 h 检验无效。
- $|h|>1.96$ 时，常在 5% 水平下拒绝无一阶自相关。
- 可替代方案是 [[Breusch-Godfrey Test]]。

## 来自课程位置

- [[11_平稳时间序列模型]]

## 关联卡片

- [[ADL]]
- [[Autocorrelation Diagnosis]]
- [[Breusch-Godfrey Test]]

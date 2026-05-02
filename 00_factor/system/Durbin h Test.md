---
aliases:
- Durbin h Test
- Durbin h检验
- 德宾h检验
tags:
- system
- 时间序列
- 计量经济学
---
# Durbin h Test

## 诊断目标

Durbin h 检验用于：**在含滞后被解释变量的动态回归中检查一阶自相关**。

## 为什么不用 DW

如果模型中含有 $y_{t-1}$，普通 [[Durbin-Watson Statistic]] 不再适合直接判断误差自相关。

此时可以使用 Durbin h，或改用 [[Breusch-Godfrey Test]]。

## 统计量

$$
h=\left(1-\frac{d}{2}\right)
\sqrt{\frac{n}{1-n\operatorname{Var}(\hat\beta_{lag})}}.
$$

其中 $d$ 是 DW 统计量，$\hat\beta_{lag}$ 是滞后被解释变量的系数估计。

## 检查点

- 根号内必须为正，否则 h 检验无效。
- $|h|>1.96$ 时，常在 5% 水平下拒绝无一阶自相关。
- 若不确定，优先报告 BG 检验作为稳健替代。

## 风险点

- 把含滞后因变量的模型仍按普通 DW 读结论。
- 只看统计量，不检查根号内是否有效。
- 把误差自相关和变量滞后动态混为一谈。

## 来自课程位置

- [[05_多方程模型Multi-equation Time Series Models#3. 自回归分布滞后ADL  Autoregressive Distributed Lag|时间序列 05：动态模型背景]]
- [[08_自相关#4.2. DW 检验|计量经济学 08：自相关诊断]]

## 关联卡片

- [[ADL]]
- [[Autocorrelation Diagnosis]]
- [[Breusch-Godfrey Test]]
- [[Durbin-Watson Statistic]]

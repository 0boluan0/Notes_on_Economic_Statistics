---
aliases:
- Common Stochastic Trend
- Common Trend
- 共同随机趋势
- 共同趋势
tags:
- concept
- 时间序列
---

# Common Stochastic Trend

## 先记一句话

共同随机趋势就是：**多个非平稳变量共享同一类随机游走成分，所以某些线性组合能把非平稳部分抵消掉**。

## 它是什么

若 $y_t$ 和 $z_t$ 都是 $I(1)$，但存在 $\beta$ 使得
$$
y_t-\beta z_t
$$
是 $I(0)$，则它们的随机趋势不是完全独立的，而是有共同部分。

这就是 [[Cointegration|协整]] 的直觉来源。

## 它解决什么判断

- 为什么两个各自非平稳的变量可能存在长期均衡。
- 为什么协整变量不能只做差分后丢掉水平关系。
- 为什么 $n$ 个变量、$r$ 个协整关系时，还有 $n-r$ 个共同随机趋势。

## 最小例子

长期利率和短期利率可能各自有随机趋势，但利差可能平稳。此时两者共享某个利率水平的共同随机趋势。

## 易混点

- 共同趋势不是普通趋势线，而是随机趋势成分。
- 变量一起上升不等于共享随机趋势，还要看线性组合是否平稳。
- 共同随机趋势越少，协整关系越多。

## 来自课程位置

- [[07_协整和误差修正模型#2.2. 协整与共同随机趋势|时间序列 07：共同随机趋势]]

## 关联卡片

- [[Cointegration]]
- [[Error Correction Model]]
- [[Johansen Cointegration Test]]
- [[Cointegration theorem]]

## 边界与相关概念

本卡片只保留该概念的定义、记号与最小直觉；具体估计步骤、证明或完整应用应放在对应的 `procedure`、`proof` 或 `framework` 卡片中。

相关卡片：[[Cointegration]]、[[07_协整和误差修正模型]]、[[Error Correction Model]]、[[Johansen Cointegration Test]]、[[Cointegration theorem]]。

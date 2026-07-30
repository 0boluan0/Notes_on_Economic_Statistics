---
aliases:
- Cornish-Fisher Expansion
- Cornish-Fisher展开
- Cornish Fisher expansion
tags:
- concept
- statistics
- risk-management
---

# Cornish-Fisher Expansion

## 先记一句话

Cornish-Fisher Expansion 用偏度和峰度修正正态分位数，常用于非正态 P&L 的 VaR 近似。

## 它是什么

若 $z_\alpha$ 是标准正态分位数，偏度为 $S$，超额峰度为 $K$，修正分位数可近似写为：

$$
z_{CF}=z_\alpha+\frac{1}{6}(z_\alpha^2-1)S
+\frac{1}{24}(z_\alpha^3-3z_\alpha)K
-\frac{1}{36}(2z_\alpha^3-5z_\alpha)S^2
$$

## 解决什么判断

它回答：“如果收益或 P&L 分布偏斜、厚尾，正态 VaR 分位数该往哪里修？”

## 最小例子

期权组合的 P&L 由 Delta-Gamma 近似得到，分布可能偏斜。这时可以用 Cornish-Fisher 把正态分位数调整成考虑偏度和峰度的分位数。

## 易混点

- 它修正分位数，不是重新拟合完整分布。
- 偏度、峰度估计不稳时，修正结果也不稳。
- 它常和 [[Delta-Gamma Approximation]]、[[VaR]] 联系，而不是基本 Greeks 定义。

## 来自课程位置

- [[14_VaR参数法和模拟法]]

## 关联卡片

- [[Delta-Gamma Approximation]]
- [[VaR]]
- [[Nonlinear Products]]

## 边界与相关概念

本卡片只保留该概念的定义、记号与最小直觉；具体估计步骤、证明或完整应用应放在对应的 `procedure`、`proof` 或 `framework` 卡片中。

相关卡片：[[Delta-Gamma Approximation]]、[[VaR]]、[[14_VaR参数法和模拟法]]、[[Nonlinear Products]]。

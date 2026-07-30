---
aliases:
- Delta
- 期权Delta
- Δ
tags:
- concept
- derivatives
- risk-management
---

# Delta

## 先记一句话

Delta 是产品价值对标的价格变化的一阶敏感度。

## 它是什么

若产品价值为 $V$，标的价格为 $S$：

$$
\Delta=\frac{\partial V}{\partial S}
$$

它表示标的价格变动一个小单位时，产品价值大约变动多少。

## 解决什么判断

它回答：“这个组合现在相当于持有多少单位标的资产？”

## 最小例子

看涨期权 Delta 为 0.6，标的价格上升 1 元，期权价格一阶近似上升 0.6 元。

## 易混点

- Delta 是局部线性近似，标的价格变大后 Delta 自身会变，这由 [[Gamma]] 衡量。
- Delta 中性不等于无风险；仍可能有 Gamma、Vega、Theta、Rho 风险。
- [[Linear Products]] 的 Delta 通常稳定，[[Nonlinear Products]] 的 Delta 会随市场状态变化。

## 来自课程位置

- [[08_操作员如何管理风险暴露]]

## 关联卡片

- [[Gamma]]
- [[Delta Approximation]]
- [[Delta Hedging]]
- [[Greeks Hedging Map]]

## 边界与相关概念

本卡片只保留该概念的定义、记号与最小直觉；具体估计步骤、证明或完整应用应放在对应的 `procedure`、`proof` 或 `framework` 卡片中。

相关卡片：[[Gamma]]、[[Linear Products]]、[[Nonlinear Products]]、[[08_操作员如何管理风险暴露]]、[[Delta Approximation]]、[[Delta Hedging]]、[[Greeks Hedging Map]]。

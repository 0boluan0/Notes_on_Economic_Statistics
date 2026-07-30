---
aliases:
- Gamma
- 期权Gamma
- Γ
tags:
  - concept
  - derivatives
  - risk-management
---
# Gamma

## 先记一句话

Gamma 是 Delta 对标的价格变化的敏感度，也就是标的价格风险的二阶项。

## 它是什么

$$
\Gamma=\frac{\partial^2 V}{\partial S^2}
=\frac{\partial \Delta}{\partial S}
$$

Gamma 越大，标的价格稍微变化后 Delta 调整得越快。

## 解决什么判断

它回答：“Delta 对冲需要多频繁调整，以及只做 Delta 近似会有多大弯曲误差？”

## 最小例子

当前 Delta = 0.5，Gamma = 0.1。若标的上升 1 元，新的 Delta 近似为 0.6。

## 易混点

- Gamma 不是价格方向暴露，而是 Delta 变化速度。
- Delta 中性组合如果 Gamma 很大，遇到大幅价格变化仍可能亏损。
- Gamma 常和 [[Delta-Gamma Approximation]] 一起用于非线性产品风险近似。

## 来自课程位置

- [[08_操作员如何管理风险暴露]]

## 关联卡片

- [[Delta]]
- [[Delta-Gamma Approximation]]
- [[Nonlinear Products]]
- [[Greeks Hedging Map]]

---
aliases:
- Rho
- 期权Rho
- 利率敏感度
- ρ
tags:
- concept
- derivatives
- risk-management
---
# Rho

## 先记一句话

Rho 是产品价值对无风险利率变化的敏感度。

## 它是什么

$$
\rho=\frac{\partial V}{\partial r}
$$

其中 $r$ 是无风险利率或对应货币的折现利率。

## 解决什么判断

它回答：“利率变化会通过折现和远期价格影响我的衍生品价值多少？”

## 最小例子

普通股票看涨期权通常 Rho 为正：无风险利率上升时，行权价现值下降，看涨期权价值上升。

## 易混点

- 短期期权里 Rho 常小于 Delta/Gamma/Vega 的重要性，但长期期权和外汇期权不能忽略。
- 外汇期权可能有本币利率 Rho 和外币利率 Rho。
- 利率产品的风险更常用 [[duration|Duration]]、[[Basis Point Value (BPV)]] 等固定收益口径。

## 来自课程位置

- [[08_操作员如何管理风险暴露]]

## 关联卡片

- [[Delta]]
- [[Vega]]
- [[duration|Duration]]
- [[Basis Point Value (BPV)]]

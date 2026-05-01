---
aliases:
- Vega
- 期权Vega
- 波动率敏感度
tags:
- concept
- derivatives
- volatility
---
# Vega

## 先记一句话

Vega 是产品价值对隐含波动率变化的敏感度。

## 它是什么

$$
Vega=\frac{\partial V}{\partial \sigma}
$$

其中 $\sigma$ 通常指 [[Implied Volatility|隐含波动率]]。

## 解决什么判断

它回答：“波动率报价上升或下降时，我的组合价值会怎么变？”

## 最小例子

Vega = 2000 表示隐含波动率上升 1 个波动率点时，组合价值约上升 2000。

## 易混点

- 线性产品通常 Vega 接近 0；期权和结构化产品通常 Vega 显著。
- 期权多头常见 Vega 为正，但组合 Vega 取决于多空方向。
- Gamma 中性不自动意味着 Vega 中性。

## 来自课程位置

- [[08_操作员如何管理风险暴露]]
- [[10_波动率]]

## 关联卡片

- [[Implied Volatility]]
- [[Historical Volatility]]
- [[Realized Volatility]]
- [[Greeks Hedging Map]]

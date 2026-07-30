---
aliases:
- Linear Products
- 线性产品
- linear derivatives
tags:
  - concept
  - derivatives
---
# Linear Products

## 先记一句话

Linear Products 的价值对标的风险因子近似是一条直线，所以一阶敏感度通常足够描述小幅变化。

## 它是什么

线性产品可近似写成：

$$
V=aS+b
$$

常见例子包括现货、远期、期货、部分互换现金流。

## 解决什么判断

它回答：“我是否可以主要用 Delta/DV01 这类一阶风险管理工具？”

## 最小例子

远期合约价值随标的价格上升近似线性增加，因此 Delta 近似比期权更稳定。

## 易混点

- 线性产品不等于没有风险，只是二阶价格风险较弱。
- 利率互换对整个曲线未必简单线性，仍可能需要 [[Key Rate Duration]]。
- 与 [[Nonlinear Products]] 的边界取决于定价函数是否有显著弯曲。

## 来自课程位置

- [[08_操作员如何管理风险暴露]]

## 关联卡片

- [[Delta]]
- [[Delta Approximation]]
- [[Nonlinear Products]]

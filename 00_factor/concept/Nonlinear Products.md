---
aliases:
- Nonlinear Products
- 非线性产品
- nonlinear derivatives
tags:
- concept
- derivatives
---
# Nonlinear Products

## 先记一句话

Nonlinear Products 的价值对标的变量是弯的，所以只看一阶 Delta 容易漏掉二阶和波动率风险。

## 它是什么

非线性产品的价值函数不是简单 $V=aS+b$。期权、含权债券、结构化产品通常都属于非线性产品。

## 解决什么判断

它回答：“我是否必须看 Gamma、Vega、Theta 甚至重新定价，而不能只看 Delta？”

## 最小例子

看涨期权价值随标的价格上升而上升，但斜率 Delta 会变化；这种弯曲由 [[Gamma]] 衡量。

## 易混点

- 非线性产品不是不能近似，而是近似要加入 [[Delta-Gamma Approximation]] 或直接重估。
- 非线性风险常在市场大幅波动时暴露。
- 对冲非线性产品通常需要其他期权，而不只是现货。

## 来自课程位置

- [[08_操作员如何管理风险暴露]]

## 关联卡片

- [[Gamma]]
- [[Vega]]
- [[Delta-Gamma Approximation]]
- [[Linear Products]]

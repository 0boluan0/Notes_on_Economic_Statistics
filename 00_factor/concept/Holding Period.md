---
aliases:
- Holding Period
- 持有期
- VaR horizon
- 风险持有期
tags:
- concept
- risk-management
---
# Holding Period

## 先记一句话

Holding Period 是 VaR 问“未来多长时间内可能亏多少”的那个时间长度。

## 它是什么

在 VaR 报告中，持有期 $h$ 指从当前时点到风险衡量终点的时间跨度，例如 1 日、10 日或 1 个月。

## 解决什么判断

它回答：“这个风险数字覆盖的是明天，还是覆盖未来十个交易日？”

## 最小例子

1 日 99% VaR = 100 万，与 10 日 99% VaR = 316 万不是同一个问题；后者可能在独立同分布近似下用平方根时间规则从前者缩放。

## 易混点

- 持有期不是历史样本窗口；历史样本窗口见 [[Observation Window]]。
- 平方根时间缩放依赖独立同分布和波动率稳定等近似，遇到 [[Volatility Clustering]] 时要谨慎。
- 流动性差的资产应使用更长持有期。

## 来自课程位置

- [[12_VAR风险]]

## 关联卡片

- [[VaR]]
- [[Confidence Level]]
- [[Observation Window]]
- [[Market Risk]]

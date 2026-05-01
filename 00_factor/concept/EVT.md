---
aliases:
- EVT
- Extreme Value Theory
- 极值理论
- 极端值理论
tags:
- concept
- risk-management
- statistics
---
# EVT

## 先记一句话

EVT 是专门研究分布尾部极端事件的统计工具，用来在普通样本分位数不够时估计极端 VaR 或 ES。

## 它是什么

Extreme Value Theory 不试图拟合整个损失分布，而是聚焦最大损失或超过高阈值的尾部损失。金融风险中常用 POT 方法，把超过阈值的超额损失近似为 [[GPD]]。

## 解决什么判断

它回答：“历史样本里极端损失太少时，能否用尾部分布外推高置信度风险？”

## 最小例子

用 500 个日损失观测估计 99.9% VaR 时，直接历史分位数几乎没有足够尾部样本；EVT 会先拟合超过 95% 或 97.5% 阈值的损失，再外推。

## 易混点

- EVT 不是让极端风险“更准确”的魔法；阈值、样本量和尾部稳定性决定可靠性。
- EVT 适合尾部，不适合普通均值方差问题。
- POT/GPD 是课程里最常和 VaR 计算相连的 EVT 用法。

## 来自课程位置

- [[14_VaR参数法和模拟法]]

## 关联卡片

- [[GPD]]
- [[EVT VaR Calculation]]
- [[VaR]]
- [[ES]]
- [[Operational Risk]]

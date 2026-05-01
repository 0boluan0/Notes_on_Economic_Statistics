---
aliases:
- Basis Point Value
- Basis Point Value (BPV)
- BPV
- DV01
- PV01
- 基点价值
tags:
- concept
- fixed-income
- risk-management
---
# Basis Point Value (BPV)

## 先记一句话

BPV/DV01 是利率变动 1 个基点时，头寸价值变化多少金额。

## 它是什么

若价格或市值为 $P$，修正久期为 $D_{mod}$，则：

$$
BPV\approx D_{mod}P\times 0.0001
$$

符号方向取决于头寸；很多题目把 BPV 报告为损失或收益的绝对金额。

## 解决什么判断

它回答：“利率上升或下降 1bp，我的头寸会变动多少钱？”

## 最小例子

市值 1000 万、修正久期 5，则 BPV 约为：

$$
5\times 1000万\times 0.0001=5000
$$

## 易混点

- BPV 是金额，不是百分比。
- [[Dollar Duration]] 和 BPV 只差利率变动单位；BPV 固定是 1bp。
- 对冲题要匹配方向相反的 BPV，见 [[DV01 Hedge Calculation]]。

## 来自课程位置

- [[09_利率风险]]

## 关联卡片

- [[Dollar Duration]]
- [[Modified Duration]]
- [[DV01 Hedge Calculation]]
- [[Key Rate Duration]]

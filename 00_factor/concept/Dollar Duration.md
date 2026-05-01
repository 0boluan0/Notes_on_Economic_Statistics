---
aliases:
- Dollar Duration
- 绝对额久期
- 美元久期
tags:
- concept
- fixed-income
- risk-management
---
# Dollar Duration

## 先记一句话

Dollar Duration 把价格百分比敏感度换成金额敏感度。

## 它是什么

若债券价格或组合市值为 $P$，修正久期为 $D_{mod}$，则对收益率变化 $\Delta y$ 的金额近似为：

$$
\Delta P\approx -D_{mod}P\Delta y
$$

有时也把 $D_{mod}P$ 称为 dollar duration，再乘以具体的利率变化幅度得到金额变化。

## 解决什么判断

它回答：“利率动一下，我这个头寸大约亏或赚多少钱？”

## 最小例子

市值 1000 万、修正久期 5，若收益率上升 1%，价格约变动：

$$
\Delta P\approx -5\times 1000万\times 0.01=-50万
$$

## 易混点

- Dollar Duration 本身常对应 100% 的收益率单位变化；1bp 口径更常写成 [[Basis Point Value (BPV)]] 或 DV01。
- 符号要看头寸方向和利率变化方向；课程做题常只报告损失绝对值。
- 金额敏感度可以跨工具加总，是对冲题的核心。

## 来自课程位置

- [[09_利率风险]]

## 关联卡片

- [[Modified Duration]]
- [[Basis Point Value (BPV)]]
- [[DV01 Hedge Calculation]]
- [[Convexity]]

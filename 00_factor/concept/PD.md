---
aliases:
- PD
- Probability of Default
- 违约概率
tags:
- concept
- credit-risk
---
# PD

## 先记一句话

PD 是借款人或交易对手在给定期限内发生违约的概率。

## 它是什么

$$
PD=P(\text{default within }T)
$$

期限必须明确，例如 1 年 PD、5 年累计 PD 或某一区间边际 PD。

## 解决什么判断

它回答：“违约这件事发生的可能性有多大？”

## 最小例子

一年期 PD = 2%，表示模型估计未来一年违约概率为 2%。

## 易混点

- PD 不等于损失金额；损失还要乘 [[LGD]] 和 [[EAD]]。
- 点时违约概率、边际违约概率、累计违约概率要分清。
- 压力情景下 PD 可能上升，不能只用平稳时期平均值。

## 来自课程位置

- [[16_巴塞尔协议]]
- [[21_信用在险价值]]

## 关联卡片

- [[Credit Risk]]
- [[Default Risk]]
- [[LGD]]
- [[EAD]]
- [[Risk-Weighted Assets]]

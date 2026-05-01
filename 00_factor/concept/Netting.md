---
aliases:
- Netting
- Net Settlement
- 净额结算
- 净额抵销
tags:
- concept
- credit-risk
- derivatives
---
# Netting

## 先记一句话

Netting 是把同一交易对手下可抵销的正负头寸合并，只对净敞口计风险。

## 它是什么

在法律协议允许时，多笔衍生品交易可以在违约或结算时抵销：

$$
Net\ Exposure=\max\left(\sum_i V_i,0\right)
$$

而不是逐笔正价值相加。

## 解决什么判断

它回答：“交易对手违约时，我真正暴露在风险中的金额是多少？”

## 最小例子

对同一交易对手，一笔互换对我正价值 100，另一笔负价值 70。若可净额结算，当前净敞口是 30，而不是 100。

## 易混点

- Netting 必须有可执行法律协议支持，不是账面上想抵就抵。
- Netting 降低 [[EAD]]，从而影响 [[CVA]] 和 [[Risk-Weighted Assets]]。
- 净额结算不同于抵押品；抵押品还要看 CSA 和折扣率。

## 来自课程位置

- [[15_《巴塞尔协议I II》和 偿付能力法案II]]
- [[17_OTC衍生产品市场的监管]]

## 关联卡片

- [[CVA]]
- [[EAD]]
- [[Credit Risk]]
- [[Position]]

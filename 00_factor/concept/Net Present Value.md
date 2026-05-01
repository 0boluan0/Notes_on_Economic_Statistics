---
aliases:
- Net Present Value
- NPV
- 净现值
- 净现值法
tags:
- concept
- finance
---
# Net Present Value

## 先记一句话

NPV 是把项目所有未来增量现金流折现到今天后求和，判断项目是否创造价值。

## 它是什么

$$
NPV=\sum_{t=0}^{n}\frac{CF_t}{(1+r)^t}
$$

其中 $CF_0$ 通常是初始投资，为负数；$r$ 是与项目风险匹配的折现率。

## 解决什么判断

它回答：“这个项目按资本成本折现后，是否增加股东财富？”

## 最小例子

初始投资 -100，下一年现金流 110，折现率 5%：

$$
NPV=-100+\frac{110}{1.05}=4.76
$$

NPV 为正，项目创造价值。

## 易混点

- NPV 是金额指标，不能直接忽略项目规模差异。
- NPV 依赖折现率，折现率必须与现金流风险和币种匹配。
- [[Internal Rate of Return|IRR]] 可能有多重根或规模误导，互斥项目优先看 NPV。

## 来自课程位置

- [[05_投资项目资本预算]]

## 关联卡片

- [[NPV Calculation]]
- [[Internal Rate of Return]]
- [[Profitability Index]]
- [[Time Value of Money]]

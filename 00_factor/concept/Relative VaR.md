---
aliases:
- Relative VaR
- 相对VaR
- 相对风险价值
tags:
- concept
- risk-management
---
# Relative VaR

## 先记一句话

Relative VaR 度量的是组合相对某个 benchmark 或目标收益的下行偏离，而不是组合本身的绝对损失。

## 它是什么

令 $\Delta V_p$ 是组合价值变化，$\Delta V_b$ 是 benchmark 价值变化，相对损益可写为：

$$
\Delta V^{rel}=\Delta V_p-\Delta V_b
$$

Relative VaR 是相对损益分布的尾部分位数：

$$
\operatorname{VaR}^{rel}_{\alpha,h}=-q_{1-\alpha}(\Delta V^{rel}_h)
$$

## 解决什么判断

它回答：“组合相对基准输掉很多的风险有多大？”

## 最小例子

某基金对沪深 300 的 1 日 95% Relative VaR = 80 万，表示未来 1 日相对基准跑输超过 80 万的概率约为 5%。

## 易混点

- 组合本身盈利，也可能相对 benchmark 亏损；Relative VaR 不等于绝对亏损。
- 风险管理报告必须说明 benchmark，否则相对 VaR 没有参照物。
- 监管资本通常更关心 [[Absolute VaR]]，投资绩效风险更常用相对口径。

## 来自课程位置

- [[12_VAR风险]]

## 关联卡片

- [[VaR]]
- [[Absolute VaR]]
- [[Component VaR]]

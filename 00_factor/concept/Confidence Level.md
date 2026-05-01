---
aliases:
- Confidence Level
- 置信水平
- VaR confidence level
tags:
- concept
- risk-management
- statistics
---
# Confidence Level

## 先记一句话

在 VaR 语境里，置信水平 $\alpha$ 决定你看损失分布的多深尾部。

## 它是什么

VaR 的 $\alpha$ 置信水平对应尾部概率 $1-\alpha$：

$$
P(L>\operatorname{VaR}_\alpha)\approx 1-\alpha
$$

99% VaR 意味着模型允许约 1% 的例外概率。

## 解决什么判断

它回答：“这张风险报告关注普通坏情况，还是更极端的坏情况？”

## 最小例子

同一组合下，99% VaR 通常高于 95% VaR，因为它取更深的损失尾部。

## 易混点

- 99% VaR 不是“有 99% 概率损失这么多”，而是“约 1% 概率损失超过它”。
- 置信水平越高，尾部样本越少，[[VaR Standard Error]] 往往越大。
- 不同置信水平的 VaR 不能直接比较监管含义。

## 来自课程位置

- [[12_VAR风险]]

## 关联卡片

- [[VaR]]
- [[Holding Period]]
- [[Observation Window]]
- [[Kupiec Test]]

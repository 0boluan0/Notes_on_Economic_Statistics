---
aliases:
- VaR Standard Error
- VaR标准误
- VaR估计标准误
tags:
- concept
- risk-management
---
# VaR Standard Error

## 先记一句话

VaR Standard Error 衡量的是“样本分位数这个 VaR 估计本身有多不稳”。

## 它是什么

若 VaR 被看作损失分布的分位数估计，近似标准误为：

$$
\operatorname{SE}(\widehat{\operatorname{VaR}}_\alpha)
\approx
\frac{1}{f(q_\alpha)}
\sqrt{\frac{\alpha(1-\alpha)}{n}}
$$

其中 $f(q_\alpha)$ 是 VaR 分位点处的密度，$n$ 是样本量。

## 解决什么判断

它回答：“当前样本长度下，这个 VaR 数字是不是可能因为抽样误差而大幅摆动？”

## 最小例子

历史模拟只有 250 个交易日，99% VaR 实际上由最差的 2 到 3 个观测决定；这时 VaR 标准误通常很大。

## 易混点

- 标准误不是 VaR 本身，而是 VaR 估计的不确定性。
- $f(q_\alpha)$ 越小，尾部越稀疏，标准误越大。
- 增加样本量能降低抽样误差，但不能修复结构性模型错误。

## 来自课程位置

- [[12_VAR风险]]
- [[14_VaR参数法和模拟法]]

## 关联卡片

- [[VaR]]
- [[Historical Simulation VaR]]
- [[Confidence Level]]
- [[Observation Window]]

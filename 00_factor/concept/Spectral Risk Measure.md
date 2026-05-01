---
aliases:
- Spectral Risk Measure
- 光谱风险度量
- 谱风险度量
tags:
- concept
- risk-management
---
# Spectral Risk Measure

## 先记一句话

Spectral Risk Measure 用一组风险厌恶权重对不同分位损失加权平均，越尾部的损失通常权重越高。

## 它是什么

令 $q_p(L)$ 是损失 $L$ 的 $p$ 分位数，谱风险度量可写为：

$$
\rho(L)=\int_0^1 \phi(p)q_p(L)\,dp
$$

其中 $\phi(p)$ 是非负权重函数，通常要求越靠近尾部权重越高。

## 解决什么判断

它回答：“如果不只看单一 VaR 分位点，而是按风险厌恶程度重视整个尾部，风险是多少？”

## 最小例子

[[ES]] 可以看作对最坏尾部区间平均加权的一类谱风险度量。

## 易混点

- [[VaR]] 只看一个分位点；Spectral Risk Measure 看一段分位损失的加权。
- 权重函数必须体现风险厌恶，否则只是普通加权平均。
- 它更像风险度量框架，不是考试里最常用的计算步骤。

## 来自课程位置

- [[12_VAR风险]]

## 关联卡片

- [[Coherent Risk Measure]]
- [[ES]]
- [[VaR]]

---
aliases:
- Parallel Trends
- 平行趋势
- parallel trend assumption
- DID平行趋势
tags:
- concept
- econometrics
- causal-inference
---
# Parallel Trends

## 先记一句话

平行趋势是假设如果没有处理，处理组和对照组的结果变量会沿着相同趋势变化。

## 它是什么

潜在结果表示下：

$$
E[Y_{post}(0)-Y_{pre}(0)\mid G=1]
=E[Y_{post}(0)-Y_{pre}(0)\mid G=0]
$$

这里 $Y(0)$ 是未处理状态下的潜在结果。

## 解决什么判断

它回答：“对照组能不能代表处理组在没有政策时会发生的变化？”

## 最小例子

政策前几年，处理组和对照组就业率走势基本同步。政策后处理组就业率相对上升，才更有可能解释为政策效应。

## 易混点

- 处理前趋势相似只是支持证据，不是平行趋势的证明。
- 控制变量不能机械修复不可信的对照组。
- 处理前出现显著预趋势，DID 识别通常很危险。

## 来自课程位置

- [[13_面板数据模型]]

## 关联卡片

- [[DID]]
- [[DID Framework]]
- [[DID Diagnostics]]
- [[DID Identification Proof]]

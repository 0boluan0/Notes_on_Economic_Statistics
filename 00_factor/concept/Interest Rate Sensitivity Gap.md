---
aliases:
- Interest Rate Sensitivity Gap
- Repricing Gap
- 利率敏感性缺口
- 重新定价缺口
tags:
- concept
- risk-management
- banking
---
# Interest Rate Sensitivity Gap

## 先记一句话

利率敏感性缺口看的是某个时间桶内，利率敏感资产和利率敏感负债谁先重新定价、金额差多少。

## 它是什么

在给定期限桶内：

$$
GAP=RSA-RSL
$$

其中 $RSA$ 是利率敏感资产，$RSL$ 是利率敏感负债。

## 解决什么判断

它回答：“短期利率变化会先影响资产收益，还是先影响负债成本？”

## 最小例子

一年内 $RSA=120$ 亿、$RSL=100$ 亿，则 GAP 为正。若利率上升，净利息收入通常先受益。

## 易混点

- 这是净利息收入视角；经济价值视角看 [[Duration Gap]]。
- GAP 的符号解释依赖利率上升还是下降。
- 不同时间桶不能随意合并，否则会掩盖期限错配。

## 来自课程位置

- [[09_利率风险]]

## 关联卡片

- [[Repricing Risk]]
- [[Duration Gap]]
- [[Basis Risk]]

---
aliases:
- Tier 1 Capital Ratio
- Tier 1 Capital
- CET1
- AT1
- 一级资本比率
- 核心一级资本比率
tags:
- system
- banking
- regulation
---
# Tier 1 Capital Ratio

## 诊断目标

判断银行最能吸收损失的高质量资本是否充足。

## 资本构成

- CET1：普通股、留存收益等最高质量资本。
- AT1：符合条件的附加一级资本工具。

## 公式

$$
Tier\ 1\ Ratio=\frac{CET1+AT1}{RWA}
$$

$$
CET1\ Ratio=\frac{CET1}{RWA}
$$

## 快速阈值

- CET1 最低 4.5% RWA。
- Tier 1 最低 6.0% RWA。
- 加资本留存缓冲后，CET1 常用目标为 7.0%。

## 诊断流程

1. 先看 CET1 是否达标。
2. 再看 AT1 是否过度支撑 Tier 1。
3. 与总资本充足率一起看资本质量。

## 常见风险点

- 分红和回购过多侵蚀 CET1。
- RWA 快速扩张导致比率下降。
- AT1 占比高，真实吸收损失能力弱于表面 Tier 1。

## 来自课程位置

- [[16_巴塞尔协议]]

## 关联卡片

## 复现规范

记录输入数据与样本区间、模型/检验设定、阈值或显著性水平、软件版本和处理决策；保留诊断图表与原始输出，使“发现—判断—修正”链条可复核。

- [[Basel Capital Adequacy Ratio]]
- [[Tier 2 Capital Ratio]]
- [[Risk-Weighted Assets]]

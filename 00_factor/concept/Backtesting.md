---
aliases:
- Backtesting
- Back-testing
- 回测
- 回溯检验
tags:
- concept
- risk-management
---
# Backtesting

## 先记一句话

Backtesting 是把模型事前预测和事后实际结果对账，检查模型有没有系统性失效。

## 它是什么

在 VaR 语境中，回测把每日预测的 VaR 与实际损益比较，统计实际损失超过 VaR 的例外次数，并检查例外是否随机分布。

## 解决什么判断

它回答：“这个风险模型在真实历史中是否按它承诺的概率失效？”

## 最小例子

99% 1 日 VaR 回测 250 天，理论上大约会出现 2 到 3 次例外；如果出现 12 次，模型大概率低估风险。

## 易混点

- 回测是模型验证，不是 VaR 计算方法。
- 例外次数看 [[Kupiec Test]]；例外是否聚集看 [[Clustering Test]]。
- 系统化诊断和整改流程见 system 卡：[[Backtesting Diagnostics|VaR 回测诊断]]。

## 来自课程位置

- [[12_VAR风险]]

## 关联卡片

- [[VaR]]
- [[Kupiec Test]]
- [[Christoffersen Test]]
- [[VaR Standard Error]]

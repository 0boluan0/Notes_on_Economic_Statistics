---
aliases:
- Clustering Test
- Clustered Violations Test
- VaR violation clustering test
- 聚束效应检验
- 例外聚集检验
tags:
- concept
- risk-management
---
# Clustering Test

## 先记一句话

Clustering Test 检验 VaR 例外是否成串出现；成串出现通常说明模型没有捕捉波动聚集或机制切换。

## 它是什么

在 VaR 回测中，定义例外指示变量：

$$
I_t=\mathbf{1}\{L_t>\operatorname{VaR}_t\}
$$

Clustering Test 检查 $I_t$ 是否近似独立，尤其关注 $I_t=1$ 后下一期继续出现例外的概率是否异常高。

## 解决什么判断

它回答：“VaR 模型虽然平均例外次数可能对，但是否在危机期连续失效？”

## 最小例子

250 天中出现 5 次例外，数量可能不算太多；但如果 5 次集中在同一周，模型对风险状态变化的反应就有问题。

## 易混点

- 例外次数问题看 [[Kupiec Test]]；例外时间分布问题看 Clustering Test。
- 聚束不是公共经济学里的 [[Bunching]]，这里指风险模型例外在时间上聚集。
- 常见正式实现是 [[Christoffersen Test|Christoffersen 独立性或条件覆盖率检验]]。

## 来自课程位置

- [[12_VAR风险]]

## 关联卡片

- [[Backtesting]]
- [[Volatility Clustering]]
- [[GARCH]]
- [[EWMA]]

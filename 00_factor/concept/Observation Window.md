---
aliases:
- Observation Window
- Lookback Window
- 观察期
- 历史观察窗口
tags:
- concept
- risk-management
---
# Observation Window

## 先记一句话

Observation Window 是用来估计 VaR 或波动率的历史数据长度。

## 它是什么

观察窗口是从历史数据中截取多少个观测用于估计风险参数或经验分布，例如最近 250 个交易日、500 个交易日或 3 年日数据。

## 解决什么判断

它回答：“这个风险模型是用最近市场状态估计，还是用更长历史平滑估计？”

## 最小例子

历史模拟 VaR 用最近 250 日收益率，观察窗口就是 250 个交易日。

## 易混点

- 观察窗口越长，抽样误差通常越小，但可能混入过时市场状态。
- 观察窗口越短，模型反应更快，但 [[VaR Standard Error]] 更大。
- 观察窗口不是 [[Holding Period|持有期]]。

## 来自课程位置

- [[14_VaR参数法和模拟法]]

## 关联卡片

- [[Historical Simulation VaR]]
- [[EWMA]]
- [[GARCH]]
- [[Backtesting]]

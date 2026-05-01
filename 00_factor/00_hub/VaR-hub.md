---
aliases:
- VaR hub
- Value at Risk hub
- 风险价值 hub
- VaR知识地图
tags:
- hub
- risk-management
---
# VaR-hub

## 这组卡解决什么

VaR 这一簇只回答一个主线问题：在给定时间 horizon 和置信水平下，如何把组合未来损失压缩成一个可报告、可回测、可监管比较的风险数字。

学习路线不要从公式开始。先确认 [[VaR]] 的定义口径，再选计算方法，最后做 [[Backtesting|回测诊断]]。

## 先学核心口径

- [[VaR]]：风险价值的基本定义，必须同时带上置信水平、持有期和组合口径。
- [[Absolute VaR]] 与 [[Relative VaR]]：绝对损失口径 vs. 相对基准口径。
- [[Confidence Level]]、[[Holding Period]]、[[Observation Window]]：VaR 报告中最容易被省略但最影响数字含义的三个参数。
- [[ES]]：VaR 之外进一步问“超过 VaR 以后平均会亏多少”。

## 再学三类计算方法

- [[VaR Method Selection]]：先判断题目应当用哪种 VaR 方法。
- [[Variance-Covariance Method]] 与 [[VaR Parametric Method]]：线性组合、近似正态、要快速闭式估计时使用。
- [[Historical Simulation Method]] 与 [[Historical Simulation VaR]]：不想假设分布、直接用历史情景排序时使用。
- [[Monte Carlo Simulation Method]] 与 [[Monte Carlo Simulation VaR]]：复杂组合、非线性产品、路径依赖产品需要重新定价时使用。
- [[EVT]]、[[GPD]] 与 [[EVT VaR Calculation]]：高置信度尾部外推，重点是阈值和尾部样本稳定性。

## 组合贡献与资本分摊

- [[Marginal VaR]]：某一风险因子或头寸微小变化对 VaR 的边际影响。
- [[Incremental VaR]]：加入或删除某个头寸后，总 VaR 改变量。
- [[Component VaR]]：把总 VaR 分摊到各头寸，常用于风险预算。

## 回测与模型诊断

- [[Backtesting]]：回测是什么。
- [[Backtesting Diagnostics|VaR 回测诊断]]：系统卡，记录例外次数、聚束、监管信号灯和整改方向。
- [[Kupiec Test]]：例外次数是否符合名义置信水平。
- [[Clustering Test]] 与 [[Christoffersen Test]]：例外是否在时间上聚集。
- [[VaR Standard Error]]：样本分位数估计有多不稳定。

## 风险度量边界

- [[Coherent Risk Measure]]：判断一个风险度量是否满足一致性四公理。
- [[Spectral Risk Measure]]：用风险厌恶权重整合分位损失的风险度量。
- [[ES]] 通常比 [[VaR]] 更符合尾部风险管理；[[VaR]] 的短板是看不到阈值之后的损失严重程度，且一般不满足次可加性。

## 来自课程位置

- [[12_VAR风险]]：VaR 定义、置信水平、持有期、回测与监管口径。
- [[14_VaR参数法和模拟法]]：参数法、历史模拟法、蒙特卡罗模拟法和 EVT。
- [[22_情景分析和压力测试]]：VaR 与压力测试、情景分析的边界。

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM ""
WHERE (
  contains(file.path, "01_Math/") OR
  contains(file.path, "02_Economy/") OR
  contains(file.path, "03_Computer_Science/")
) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```

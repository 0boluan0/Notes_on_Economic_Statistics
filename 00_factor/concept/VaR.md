---
aliases:
- VaR
tags:
- concept
---
VaR（Value at Risk，风险价值）表示在正常市场条件下、给定置信水平（如95%或99%）和持有期内，金融资产或投资组合可能遭受的最大潜在损失。换言之，VaR回答的是："在未来特定时期内，有一定概率（如99%）损失不超过多少。"

估计VaR标准误:[[VaR Standard Error|VaR标准误]]

## 相关链接

- 风险度量相关：[[ES]], [[Coherent Risk Measure|一致性风险度量]]
- VaR分解：[[Marginal VaR|边际VaR]], [[Incremental VaR|递增VaR]], [[Component VaR|成分VaR]]
- 估计方法：[[Historical Simulation Method|历史模拟法]], [[Variance-Covariance Method|方差-协方差法]], [[Monte Carlo Simulation Method|蒙特卡罗模拟方法]]
- 检验：[[00_factor/concept/Backtesting|回溯检验]], [[VaR Standard Error|VaR标准误]], [[Stressed VaR|压力VaR]]

## $source_notes$

- [[12_VAR风险]]
- [[14_VaR参数法和模拟法]]

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

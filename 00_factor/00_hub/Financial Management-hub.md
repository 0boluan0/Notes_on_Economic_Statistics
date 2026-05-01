---
aliases:
- Financial Management-hub
- Financial Management
- 财务管理
- 财务管理知识地图
tags:
- hub
- finance
---
# Financial Management-hub

## 这组卡解决什么

财务管理这组卡围绕三个问题：不同时点的钱如何比较，项目是否创造价值，企业财务状况是否健康。

## 学习路线

1. 先学现金流折现：[[Time Value of Money]]、[[Present Value]]、[[Future Value]]、[[Annuity]]。
2. 再做投资决策：[[Capital Budgeting Decision Map]]、[[Net Present Value]]、[[NPV Calculation]]、[[Internal Rate of Return]]、[[IRR Calculation]]、[[Profitability Index]]。
3. 再看证券估值：[[Bond and Stock Valuation]]、[[Bond Valuation Model]]、[[Stock Valuation Model]]、[[Yield to Maturity]]、[[CAPM]]。
4. 债券风险接到利率风险簇：[[duration|Duration]]、[[Modified Duration]]、[[Convexity]]、[[Basis Point Value (BPV)]]。
5. 最后做企业诊断：[[Financial Ratio Analysis]]、[[DuPont Analysis]]、[[Sustainable Growth Rate]]、[[Internal Growth Rate]]。

## 货币时间价值

- [[Time Value of Money]]：为什么现金流必须放到同一时点比较。
- [[Present Value]]：未来现金流折现到今天。
- [[Future Value]]：今天的钱复利到未来。
- [[Annuity]]：等额现金流的现值和终值。

## 投资决策

- [[Net Present Value]]：项目是否创造价值的核心概念。
- [[NPV Calculation]]：可执行计算步骤。
- [[Internal Rate of Return]] 与 [[IRR Calculation]]：使 NPV 为 0 的折现率。
- [[Profitability Index]]：资金受限时的相对价值指标。
- [[Capital Budgeting Decision Map]]：判断 NPV、IRR、PI 什么时候优先。
- [[Investment Decisions]]：把 NPV、IRR、PI 和情景分析合成决策流程。

## 证券估值

- [[Bond and Stock Valuation]]：把债券和股票估值落到步骤。
- [[Bond Valuation Model]]：债券价值等于票息和本金的现值。
- [[Stock Valuation Model]]：股票价值等于未来股利或自由现金流的现值。
- [[Yield to Maturity]]：反推债券现金流的隐含收益率。
- [[CAPM]] 与 [[CAPM Estimation]]：估计权益资本成本和系统性风险。

## 财务比率分析

- 流动性：[[Current Ratio]]、[[Quick Ratio]]、[[Cash Ratio]]。
- 盈利能力：[[Return on Equity]]、[[Return on Assets]]、[[Net Profit Margin]]。
- 杠杆和偿债：[[Debt-to-Asset Ratio]]、[[Debt-to-Equity Ratio]]、[[Times Interest Earned Ratio]]。
- 营运效率：[[Inventory Turnover Ratio]]、[[Accounts Receivable Turnover Ratio]]。
- 分解框架：[[DuPont Analysis]]、[[Equity Multiplier]]。

## 课程笔记入口

- [[03_价值评估基准#一.货币时间价值概念(Concept of Time Value of Money)|01_货币时间价值]]
- [[05_投资项目资本预算]]
- [[06_债券和股票估价|03_债券与股票估值]]
- [[02_财务报表分析和财务预测#第二节 财务比率分析|04_财务比率分析]]

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

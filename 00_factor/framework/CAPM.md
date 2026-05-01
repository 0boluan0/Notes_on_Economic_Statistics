---
aliases:
- CAPM
- Capital Asset Pricing Model
- 资本资产定价模型
tags:
- framework
- finance
---
# CAPM

## 什么时候用

当题目要估计权益资本成本、解释资产预期收益和系统性风险的关系，或为股票估值选择折现率时，用 CAPM。

## 为什么这样看

CAPM 把资产风险拆成两类：可以通过分散化消掉的非系统性风险，以及无法消掉、需要被补偿的系统性风险。市场只补偿后者，所以核心变量是 $\beta$。

$$
E(R_i)=R_f+\beta_i(E(R_m)-R_f)
$$

## 题型识别

- “求必要报酬率 / 股权资本成本”：直接套 CAPM。
- “解释 beta”：说明资产对市场组合的敏感度。
- “股票估值缺折现率”：CAPM 给 [[Stock Valuation Model]] 的 $r$。
- “组合 beta”：用权重加权各资产 beta。

## 边界条件

- 需要可代表市场组合的基准指数。
- Beta 估计依赖样本期和收益率频率。
- CAPM 假设单一系统性风险因子，现实中行业、规模、价值等风险可能未被捕捉。

## 失败模式

- 把历史平均收益率直接当资本成本。
- 用不匹配的市场指数估计 beta。
- 忽略杠杆变化导致 beta 变化。
- 用 CAPM 精确预测短期收益。

## 来自课程位置

- [[03_资产定价理论及其发展#2.1.3.1.夏普CAPM|03_债券与股票估值]]

## 关联卡片

- [[CAPM Estimation]]
- [[Stock Valuation Model]]
- [[Bond and Stock Valuation]]
- [[Sharpe Ratio]]

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

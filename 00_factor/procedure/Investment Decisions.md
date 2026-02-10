---
aliases:
- Investment Decisions
- 投资决策
tags:
- procedure
---
# Investment Decisions

## 适用场景

用于资本预算或项目筛选，判断是否投资、何时投资、规模多大。

## 所需数据/条件

- 项目现金流（初始投入、运营现金流、终值）
- 折现率（资本成本、项目风险溢价）
- 约束条件（资金上限、政策或产能约束）

## 决策步骤

### 步骤 1：界定项目与现金流边界

- 明确项目周期与里程碑。
- 只保留“增量现金流”。

### 步骤 2：确定折现率与风险调整

- 以 WACC 或可比项目回报为基准。
- 对高风险项目加入风险溢价。

### 步骤 3：计算核心指标

- 净现值（NPV）
- 内部收益率（IRR）
- 现值指数（PI）
- 回收期（必要时）

### 步骤 4：做敏感性与情景分析

- 关键变量（销量、价格、成本、折现率）上下浮动。
- 输出 NPV/IRR 的区间。

### 步骤 5：形成决策建议

- NPV > 0 且满足约束 → 建议投资。
- 多项目比较时优先 NPV 最大或 PI 最高。

## 输出物

- 投资决策备忘录（假设、现金流、指标、风险结论）。

## 常见误区

- 将沉没成本计入现金流。
- 仅看 IRR 忽略规模与再投资假设。

## 相关链接

- [[Net Present Value (2)|净现值]]
- [[Internal Rate of Return|内部收益率]]
- [[Profitability Index|现值指数]]

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

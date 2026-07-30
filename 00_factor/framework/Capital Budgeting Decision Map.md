---
aliases:
- Capital Budgeting Decision Map
- 投资项目资本预算判断框架
- 资本预算决策框架
tags:
- framework
- finance
type: framework
---
# Capital Budgeting Decision Map

## 什么时候用

当题目不是单纯让你算 NPV 或 IRR，而是要判断“项目该不该投、多个项目怎么选、为什么指标冲突”时，用这张框架卡。

## 为什么这样看

资本预算的核心不是找一个最好看的指标，而是回答项目是否创造价值、约束条件是什么、指标之间为什么可能冲突。

## 题型识别

- “是否接受该项目”：先看 [[Net Present Value]]。
- “互斥项目怎么选”：以 NPV 为主，IRR/PI 只作辅助解释。
- “资金有限怎么组合”：先看可分割性，再用 [[Profitability Index]] 做排序参考。
- “IRR 与 NPV 结论不同”：检查项目规模、现金流时点和非常规现金流。
- “项目风险不同”：折现率必须按风险调整，不能用同一个资本成本硬套。

## 边界条件

- NPV 需要可信现金流和折现率。
- IRR 需要现金流符号结构支持唯一解。
- PI 适合资本受限场景，不适合直接替代 NPV。
- 回收期类指标重视流动性，不完整衡量价值创造。

## 失败模式

- 把会计利润当现金流。
- 用同一折现率比较风险不同的项目。
- 互斥项目只看 IRR 高低。
- 只算指标，不解释现金流假设。
- 忽略通胀、税、营运资本和残值。

## 来自课程位置

- [[05_投资项目资本预算]]

## 关联卡片

- [[Net Present Value]]
- [[NPV Calculation]]
- [[Internal Rate of Return]]
- [[IRR Calculation]]
- [[Profitability Index]]
- [[Investment Decisions]]

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

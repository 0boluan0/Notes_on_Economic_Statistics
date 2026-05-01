---
aliases:
- Profitability Index
- PI
- 现值指数
- 获利指数
tags:
- concept
- finance
---
# Profitability Index

## 一句话记忆

PI 是每投入 1 元现值成本能带来多少现值流入。

## 它是什么

Profitability Index 是未来现金流入现值与投资流出现值的比率：

$$
PI=\frac{PV(\text{cash inflows})}{|PV(\text{cash outflows})|}
$$

若只有初始投资 $CF_0<0$，也可写成：

$$
PI=1+\frac{NPV}{|CF_0|}
$$

## 解决什么判断

- 资本预算有资金约束时，哪些项目单位投资效率更高。
- 项目是否至少覆盖投资成本。
- NPV 为正的项目中，哪个更适合放进有限预算组合。

## 最小例子

一个项目现金流入现值为 120，初始投资现值为 100：

$$
PI=120/100=1.2
$$

说明每投入 1 元现值成本，带来 1.2 元现值流入。

## 易混点

- PI 是相对效率，不是绝对价值。
- 互斥项目规模差异很大时，PI 高的项目不一定创造最多价值。
- 没有资金约束时，核心判断仍是 [[Net Present Value]]。

## 来自课程位置

- [[05_投资项目资本预算]]

## 关联卡片

- [[Net Present Value]]
- [[NPV Calculation]]
- [[Internal Rate of Return]]
- [[Capital Budgeting Decision Map]]

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

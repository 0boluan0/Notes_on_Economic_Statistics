---
aliases:
- Future Value
- FV
- 终值
tags:
- concept
- finance
---
# Future Value

## 一句话记忆

终值是把今天的钱按复利推到未来后的价值。

## 它是什么

Future Value 是当前金额 $PV$ 经过 $t$ 期复利后在未来时点的金额：

$$
FV=PV(1+r)^t
$$

## 解决什么判断

- 今天投入一笔钱，未来会变成多少。
- 存款、债券、年金、项目现金流是否达到目标金额。
- 同一个现金流序列需要统一到未来时点时如何处理。

## 最小例子

今天存入 100 元，年利率 5%，一年后的终值是：

$$
FV=100(1.05)=105
$$

## 易混点

- Future Value 是复利后的未来金额；[[Present Value]] 是折现后的今天金额。
- 复利次数、计息周期、现金流周期必须统一。
- 终值变大不等于项目创造价值；投资判断仍要回到 [[Net Present Value]]。

## 来自课程位置

- [[03_价值评估基准#一.货币时间价值概念(Concept of Time Value of Money)|01_货币时间价值]]

## 关联卡片

- [[Time Value of Money]]
- [[Present Value]]
- [[Annuity]]
- [[NPV Calculation]]

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

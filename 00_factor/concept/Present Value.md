---
aliases:
- Present Value
- PV
- 现值
tags:
- concept
- finance
---
# Present Value

## 一句话记忆

现值是把未来现金流搬回今天后的价值；比较现金流前，先把它们放到同一个时间点。

## 它是什么

Present Value 是未来某一期现金流 $CF_t$ 按折现率 $r$ 折回 $t=0$ 的金额：

$$
PV=\frac{CF_t}{(1+r)^t}
$$

## 解决什么判断

- 未来收到的钱今天值多少钱。
- 不同时间发生的现金流能不能直接比较。
- 投资项目、债券、股票估值时每期现金流该怎样进入总价值。

## 最小例子

一年后收到 100 元，折现率为 5%：

$$
PV=\frac{100}{1.05}=95.24
$$

这表示一年后的 100 元，在今天约等于 95.24 元。

## 易混点

- Present Value 不是现金流本身，而是折现后的现金流。
- 折现率越高，同一笔未来现金流的现值越低。
- 现金流周期和折现率周期必须一致，月现金流不能直接套年折现率。

## 来自课程位置

- [[03_价值评估基准#一.货币时间价值概念(Concept of Time Value of Money)|01_货币时间价值]]
- [[06_债券和股票估价|03_债券与股票估值]]

## 关联卡片

- [[Time Value of Money]]
- [[Future Value]]
- [[Annuity]]
- [[Net Present Value]]
- [[Bond Valuation Model]]

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

---
aliases:
- Annuity
- 年金
- 普通年金
- 预付年金
- 递延年金
- 永续年金
tags:
- concept
- finance
---
# Annuity

## 一句话记忆

年金是一串等额、等间隔发生的现金流。

## 它是什么

Annuity 指每期金额相同、时间间隔相同的一组现金流。财务管理中常见的房贷还款、租金、养老金、固定票息都可以用年金思路处理。

## 解决什么判断

- 一串等额现金流是否可以用简化公式计算。
- 现金流发生在期末、期初、延迟开始，还是永续持续。
- 估值时应该用普通年金、预付年金、递延年金还是永续年金。

## 最小例子

每年年末收到 100 元，持续 3 年，折现率 5%。这是一笔普通年金，其现值为：

$$
PVA=100\cdot\frac{1-(1.05)^{-3}}{0.05}
$$

## 常见类型

- 普通年金：每期期末支付，默认情形通常是它。
- 预付年金：每期期初支付，比普通年金多滚动一期利息。
- 递延年金：现金流不是马上开始，而是延迟若干期后开始。
- 永续年金：等额现金流无限期持续，现值为 $A/r$。

## 易混点

- 年金公式成立的关键不是“很多期”，而是金额相等且间隔固定。
- 期初支付和期末支付差一个 $(1+r)$，不能混用。
- 永续年金要求 $r>0$，且隐含现金流长期稳定。

## 来自课程位置

- [[03_价值评估基准#一.货币时间价值概念(Concept of Time Value of Money)|01_货币时间价值]]
- [[06_债券和股票估价|03_债券与股票估值]]

## 关联卡片

- [[Time Value of Money]]
- [[Present Value]]
- [[Future Value]]
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

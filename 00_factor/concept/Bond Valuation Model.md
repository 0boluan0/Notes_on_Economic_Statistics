---
aliases:
- Bond Valuation Model
- Bond Pricing Model
- 债券估价模型
- 债券定价模型
tags:
- concept
- finance
---

# Bond Valuation Model

## 一句话记忆

债券价值等于未来票息和到期本金折现到今天的总和。

## 它是什么

Bond Valuation Model 用折现现金流给债券估值：

$$
P=\sum_{t=1}^{n}\frac{C_t}{(1+r)^t}+\frac{F}{(1+r)^n}
$$

其中 $C_t$ 是第 $t$ 期票息，$F$ 是面值，$r$ 是与风险和期限匹配的折现率。

## 解决什么判断

- 债券当前价格是否高于或低于内在价值。
- 票面利率、市场利率、期限如何影响债券价格。
- 债券价格变化为什么会接到 [[duration|Duration]] 和 [[Convexity]]。

## 最小例子

一张面值 100、每年付息 5、两年到期的债券，折现率 4%：

$$
P=\frac{5}{1.04}+\frac{105}{1.04^2}
$$

## 易混点

- 票面利率决定票息，市场折现率决定价格。
- 市场利率上升时，固定票息债券价格下降。
- 到期收益率是给定价格后反推的折现率，不是票面利率。
- 久期和凸性不是估值公式本身，而是价格对利率变化的敏感度。

## 来自课程位置

- [[06_债券和股票估价|03_债券与股票估值]]
- [[06_债券和股票估价#第一节 债券估价(Bond Valuation)|02_债券定价]]

## 关联卡片

- [[Bond and Stock Valuation]]
- [[Yield to Maturity]]
- [[Present Value]]
- [[duration|Duration]]
- [[Modified Duration]]
- [[Convexity]]

## 边界与相关概念

本卡片只保留该概念的定义、记号与最小直觉；具体估计步骤、证明或完整应用应放在对应的 `procedure`、`proof` 或 `framework` 卡片中。

相关卡片：[[duration]]、[[Convexity]]、[[06_债券和股票估价]]、[[Bond and Stock Valuation]]、[[Yield to Maturity]]、[[Present Value]]、[[Modified Duration]]。

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

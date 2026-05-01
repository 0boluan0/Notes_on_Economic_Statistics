---
aliases:
- Internal Rate of Return
- IRR
- 内部收益率
- 内含报酬率
tags:
- concept
- finance
---
# Internal Rate of Return

## 一句话记忆

IRR 是让项目 NPV 正好等于 0 的折现率。

## 它是什么

Internal Rate of Return 是满足下式的 $r$：

$$
\sum_{t=0}^{n}\frac{CF_t}{(1+r)^t}=0
$$

它把项目现金流压缩成一个“收益率”指标。

## 解决什么判断

- 项目自身隐含收益率是多少。
- 独立项目能否超过要求收益率。
- 现金流是否存在多重收益率或无解问题。

## 最小例子

今天投资 100，明年收到 110：

$$
-100+\frac{110}{1+IRR}=0
$$

所以 $IRR=10\%$。

## 易混点

- IRR 是相对收益率，[[Net Present Value]] 是绝对价值创造。
- 常规现金流通常只有一个 IRR；非常规现金流可能有多个 IRR。
- 互斥项目出现 NPV 与 IRR 排序冲突时，以 NPV 为主。
- IRR 默认中间现金流可以按 IRR 再投资，这常常不现实。

## 来自课程位置

- [[05_投资项目资本预算]]

## 关联卡片

- [[IRR Calculation]]
- [[Net Present Value]]
- [[NPV Calculation]]
- [[Profitability Index]]
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

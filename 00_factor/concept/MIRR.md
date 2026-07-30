---
aliases:
- Modified Internal Rate of Return
- MIRR
- 修正内部收益率
tags:
  - concept
---
# Modified Internal Rate of Return (MIRR)

>[!note] 它是什么
> - 「MIRR」是指将正现金流以再投资率复利、负现金流以融资率折现后求的内部收益率。
>
>[!note] 最小可检索信息
> - 定义：将正现金流以再投资率复利、负现金流以融资率折现后求的内部收益率。
> - 符号/公式：$MIRR=\left(\frac{FV_{pos}}{PV_{neg}}\right)^{1/n}-1。$
> - 最小例子：再投资率10%的项目MIRR=12%。
>
## 关联卡片
- [[IRR Calculation]]


## 最小例子

把 **MIRR** 放在最简单的可计算情形中：先给定定义所需的最小输入，再按定义计算输出；若关键关系不成立，就不能把该对象归入本概念。这个检查也能帮助区分相近概念。
## 概念边界

本卡片只回答“它是什么”，集中在定义、核心关系与最小例子；具体估计步骤、证明和诊断流程应分别放在 procedure、proof 或 system 卡片中。
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

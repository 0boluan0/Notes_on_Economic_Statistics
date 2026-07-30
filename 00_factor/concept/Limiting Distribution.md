---
aliases:
- Limiting Distribution
- 极限分布
tags:
  - concept
---
# Limiting Distribution

>[!note] 它是什么
> - 「Limiting Distribution」是指统计量在样本量趋于无穷时的极限分布。
>
>[!note] 最小可检索信息
> - 定义：统计量在样本量趋于无穷时的极限分布。
> - 符号/公式：如 $\sqrt{n}(\bar X-\mu)\Rightarrow N(0,\sigma^2)$。
> - 最小例子：样本均值的渐近正态。
>
## 关联卡片
- [[Stationary Distribution]]


## 最小例子

把 **Limiting Distribution** 放在最简单的可计算情形中：先给定定义所需的最小输入，再按定义计算输出；若关键关系不成立，就不能把该对象归入本概念。这个检查也能帮助区分相近概念。
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

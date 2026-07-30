---
aliases:
- Merton Model
- Merton structural model
- 默顿模型
- Merton模型
tags:
  - concept
---
# Merton Model

>[!note] 它是什么
> - 「Merton Model」是指结构性信用风险模型，将股权视为资产对债务的看涨期权。
>
>[!note] 最小可检索信息
> - 定义：结构性信用风险模型，将股权视为资产对债务的看涨期权。
> - 符号/公式：$PD=N(-d_2)，d_{1,2}=\frac{\ln(V/D)+(\mu\pm\tfrac12\sigma^2)T}{\sigma\sqrt{T}}。$
> - 最小例子：资产价值低于债务面值时公司违约。
>
## 关联卡片
- [[Default Risk]]


## 最小例子

把 **Merton Model** 放在最简单的可计算情形中：先给定定义所需的最小输入，再按定义计算输出；若关键关系不成立，就不能把该对象归入本概念。这个检查也能帮助区分相近概念。
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

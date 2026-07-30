---
aliases:
- Likelihood Ratio Test
- LR Test
- 似然比检验
tags:
  - concept
---
# Likelihood Ratio Test

>[!note] 它是什么
> - 「Likelihood Ratio Test」是指比较嵌套模型拟合优度的检验。
>
>[!note] 最小可检索信息
> - 定义：比较嵌套模型拟合优度的检验。
> - 符号/公式：$LR=-2\ln(L_0/L_1)。$
> - 最小例子：检验是否需要加入解释变量。
>
## 关联卡片
- [[MLE Estimation Steps]]


## 最小例子

把 **Likelihood Ratio Test** 放在最简单的可计算情形中：先给定定义所需的最小输入，再按定义计算输出；若关键关系不成立，就不能把该对象归入本概念。这个检查也能帮助区分相近概念。
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

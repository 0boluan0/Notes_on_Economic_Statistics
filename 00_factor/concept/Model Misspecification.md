---
aliases:
- Model Misspecification
- Model Specification Error
- 模型设定偏误
tags:
  - concept
---
# Model Misspecification

>[!note] 它是什么
> - 「Model Misspecification」是指模型形式与真实数据生成过程不一致。
>
>[!note] 最小可检索信息
> - 定义：模型形式与真实数据生成过程不一致。
> - 符号/公式：无固定符号。
> - 最小例子：遗漏关键变量导致系数偏误。
>
## 关联卡片
- [[Econometrics-hub]]


## 最小例子

把 **Model Misspecification** 放在最简单的可计算情形中：先给定定义所需的最小输入，再按定义计算输出；若关键关系不成立，就不能把该对象归入本概念。这个检查也能帮助区分相近概念。
## 符号表达

将本概念记为 $C_{ModelMisspec}$；使用时先明确对象、条件与输出，再判断 $C$ 是否满足定义。
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

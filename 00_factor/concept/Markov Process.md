---
aliases:
- Markov Process
- 马尔可夫过程
tags:
  - concept
---
# Markov Process

>[!note] 它是什么
> - 「Markov Process」是指满足无记忆性（马尔可夫性）的随机过程。
>
>[!note] 最小可检索信息
> - 定义：满足无记忆性（马尔可夫性）的随机过程。
> - 符号/公式：$P(X_{t+h}|\mathcal{F}_t)=P(X_{t+h}|X_t)。$
> - 最小例子：离散马尔可夫链或连续时间马尔可夫过程。
>
## 关联卡片
- [[Chapman-Kolmogorov equation]]


## 最小例子

把 **Markov Process** 放在最简单的可计算情形中：先给定定义所需的最小输入，再按定义计算输出；若关键关系不成立，就不能把该对象归入本概念。这个检查也能帮助区分相近概念。
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

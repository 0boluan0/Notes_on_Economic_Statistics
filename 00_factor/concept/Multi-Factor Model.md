---
aliases:
- 多因子模型
- Multi-Factor Model
- Multi
tags:
  - 经济
  - concept
---
# Multi-Factor Model

**多因子模型：**可以推广到 $M$ 个因子。假设有因子 $F_1, \ldots, F_M$ 彼此独立且均为 $N(0,1)$，每个变量 $U_i$ 有对应的加载向量 $(a_{i1}, a_{i2}, \dots, a_{iM})$，则：
$$
U_i = a_{i1}F_1 + a_{i2}F_2 + \cdots + a_{iM}F_M \;+\; \sqrt{\,1 - \sum_{m=1}^M a_{im}^2\,}\;Z_i \,.
$$ 
在保证 $1 - \sum_{m}a_{im}^2 \ge 0$ 的前提下，每个 $U_i$ 方差仍为1。任意两变量的相关系数是各自对公共因子加载的**逐因子乘积之和**：
$$
Corr(U_i, U_j) = \sum_{m=1}^M a_{im}\,a_{jm} \,.
$$ 
例如，在两因子模型下 $Corr(U_i, U_j) = a_{i1}a_{j1} + a_{i2}a_{j2}$。单因子模型是 $M=1$ 的特例。


## 最小例子

把 **Multi-Factor Model** 放在最简单的可计算情形中：先给定定义所需的最小输入，再按定义计算输出；若关键关系不成立，就不能把该对象归入本概念。这个检查也能帮助区分相近概念。
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

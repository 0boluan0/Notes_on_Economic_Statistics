---
aliases:
- LU分解
- LU Factorization
- LU
- LU Decomposition
tags:
  - 线性代数
  - 矩阵分解
  - concept
---
# LU Decomposition

>[!note] 它是什么
> - 「LU Decomposition」是指把矩阵分解为下三角矩阵 L 与上三角矩阵 U 的乘积。
>
>[!note] 最小可检索信息
> - 定义：若 A 可写成 $A=LU$，其中 L 是单位下三角矩阵、U 是上三角矩阵，则称 A 有 LU 分解。
> - 符号/公式：若需要换行，通常写成 $PA=LU$。
> - 最小例子：高斯消元中的消元乘子记录在 L 中，消元结果记录在 U 中。
>
## 关键性质
- LU 分解本质上是把高斯消元矩阵化。
- 同一个 A 面对多个不同右端项时，LU 分解可以显著减少重复计算。
- 若有主元为零，通常要引入 [[Permutation Matrix]]。

## 关联卡片
- [[Permutation Matrix]]
- [[Matrix Inverse]]
- [[Matrix Rank]]


## 最小例子

把 **LU Decomposition** 放在最简单的可计算情形中：先给定定义所需的最小输入，再按定义计算输出；若关键关系不成立，就不能把该对象归入本概念。这个检查也能帮助区分相近概念。
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

---
aliases:
- Linear Systems
- 线性方程组
tags:
  - concept
---
# Linear Systems

线性方程组是由若干线性方程构成的方程集合，可写为矩阵形式 $Ax=b$。

## 矩阵表示

- $A$：系数矩阵
- $x$：未知向量
- $b$：常数项向量

## 解的类型

- **唯一解**：方程组相容且主元变量数等于未知数个数。
- **无穷多解**：方程组相容且存在自由变量。
- **无解**：增广矩阵出现矛盾行。

## 关键判别（秩条件）

- 有解当且仅当 $\mathrm{rank}(A)=\mathrm{rank}(A|b)$。
- 唯一解当且仅当 $\mathrm{rank}(A)=n$（满列秩）。

>[!example] 最小例子
>
> $$
> \begin{cases}
> 2x+y=1\\
> x-y=0
> \end{cases}
> $$
> $\Rightarrow$
> $$
> A =
> \begin{bmatrix}
> 2&1\\1&-1
> \end{bmatrix}
> $$
> \quad
> $$
> b =
> \begin{bmatrix}
> 1\\0
> \end{bmatrix}
> $$
>
## 相关链接

- [[Linear system solution structure|线性方程组解的结构]]
- [[Matrix Inverse|逆矩阵]]
- [[Matrix Rank|矩阵的秩]]


## 最小例子

把 **Linear Systems** 放在最简单的可计算情形中：先给定定义所需的最小输入，再按定义计算输出；若关键关系不成立，就不能把该对象归入本概念。这个检查也能帮助区分相近概念。
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

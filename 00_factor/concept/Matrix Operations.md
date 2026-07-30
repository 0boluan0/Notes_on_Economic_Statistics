---
aliases:
- Matrix Operations
- 矩阵运算
tags:
  - concept
---
# Matrix Operations

矩阵运算是线性代数中用于组合与变换线性映射的基本规则集合。

## 基本运算

- **加法/减法**：同维矩阵按元素相加。
- **数乘**：矩阵所有元素乘以同一标量。
- **矩阵乘法**：$A_{m\times n}B_{n\times k}$ 生成 $m\times k$ 矩阵。
- **转置**：$A'$(或$A^T$) 将行列互换。
- **逆矩阵**：若 $A^{-1}A=I$，则 $A$ 可逆。

## 乘法直觉

矩阵乘法等价于线性变换的复合：先做 $B$ 的变换，再做 $A$ 的变换。

>[!example] 最小例子
>
> 若
> $$
> A=
> \begin{bmatrix}
> 1&2\\3&4
> \end{bmatrix},\quad
> B=
> \begin{bmatrix}
> 2&0\\1&2
> \end{bmatrix}
> $$
> 则
> $$
> AB=
> \begin{bmatrix}
> 1\cdot2+2\cdot1 & 1\cdot0+2\cdot2\\3\cdot2+4\cdot1 & 3\cdot0+4\cdot2
> \end{bmatrix}
> =
> \begin{bmatrix}
> 4&4\\10&8
> \end{bmatrix}
> $$
>
## 常见性质

- 一般不满足交换律：$AB \neq BA$。
- 满足结合律：$(AB)C = A(BC)$。
- 分配律：$A(B+C)=AB+AC$。

## 相关链接

- [[Matrix Inverse|逆矩阵]]
- [[Matrix Rank|矩阵的秩]]
- [[Linear system solution structure|线性方程组解的结构]]


## 最小例子

把 **Matrix Operations** 放在最简单的可计算情形中：先给定定义所需的最小输入，再按定义计算输出；若关键关系不成立，就不能把该对象归入本概念。这个检查也能帮助区分相近概念。
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

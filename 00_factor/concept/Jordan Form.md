---
aliases:
- Jordan Form
- Jordan Canonical Form
- Jordan normal form
- 若尔当标准形
- Jordan 标准形
tags:
  - concept
  - 线性代数
---
# Jordan Form

## 先记一句话

Jordan Form 是：**当对角化失败时，用最接近对角矩阵的形式描述方阵**。

它回答的是：

> 特征向量不够时，矩阵到底差在哪里？

## 它是什么

若 $A$ 不能找到足够多线性无关特征向量，就不能写成
$$
A=S\Lambda S^{-1}.
$$

在复数域中，仍然可以写成
$$
A=PJP^{-1},
$$
其中 $J$ 由若干 Jordan blocks 组成。

一个 Jordan block 长这样：
$$
J_\lambda=
\begin{bmatrix}
\lambda&1&0\\
0&\lambda&1\\
0&0&\lambda
\end{bmatrix}.
$$

对角线是特征值，上超对角的 1 表示缺失的特征向量要用 generalized eigenvectors 补。

## 一个最小例子

矩阵
$$
A=
\begin{bmatrix}
1&1\\
0&1
\end{bmatrix}
$$
只有一个特征值 $\lambda=1$。

但它只有一个独立特征向量，所以不能对角化。

它本身就是一个 Jordan block。

## 它在题里负责什么

- 解释为什么 repeated eigenvalue 不一定能 diagonalize。
- 处理矩阵幂和矩阵指数中的非对角化情形。
- 说明长期行为中为什么除了 $\lambda^k$ 或 $e^{\lambda t}$，还会出现多项式因子。

## 常见误区

- Jordan form 不是优先使用的计算工具；它是理解对角化失败的结构工具。
- 重复特征值不必然导致 Jordan block；关键是特征向量数量是否足够。
- SVD 不是 Jordan form 的替代版本。Jordan 处理方阵相似结构，SVD 处理任意矩阵的几何结构。

## 来自课程位置

- [[03_Positive Definite Matrices and Applications#Session 3.4 Similar matrices and Jordan form|Session 3.4]]：similarity、diagonalization failure 与 Jordan form。

## 关联卡片

- [[Diagonalization]]
- [[Eigenvalues]]
- [[Eigenvectors]]
- [[Matrix Exponential]]
- [[Choosing Matrix Decompositions]]


## 最小例子

把 **Jordan Form** 放在最简单的可计算情形中：先给定定义所需的最小输入，再按定义计算输出；若关键关系不成立，就不能把该对象归入本概念。这个检查也能帮助区分相近概念。
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

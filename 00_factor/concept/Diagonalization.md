---
aliases:
- Diagonalization
- diagonalize
- 对角化
tags:
- concept
- 线性代数
---
# Diagonalization

## 先记一句话

对角化就是：**找到一组特征向量基，让矩阵作用变成各方向各管各的缩放**。

如果
$$
A=S\Lambda S^{-1},
$$
其中 $S$ 的列是特征向量，$\Lambda$ 的对角元是特征值，那么 $A$ 可对角化。

## 它解决什么判断

对角化回答：

> 有没有一组坐标，使得 $A$ 的作用完全解耦？

如果能对角化，那么矩阵幂、差分方程、微分方程都会变简单：
$$
A^k=S\Lambda^kS^{-1}.
$$

每个特征方向只需要看自己的 $\lambda^k$。

## 一个最小例子

若
$$
A=
\begin{bmatrix}
2&0\\
0&3
\end{bmatrix},
$$
它已经在标准基下对角化了。

沿 $e_1$ 方向放大 2 倍，沿 $e_2$ 方向放大 3 倍。

如果某个矩阵不是对角矩阵，但有两个线性无关特征向量，也可以换到这两个特征向量组成的基底中，让它变成对角矩阵。

## 什么时候成功

对 $n\times n$ 矩阵，若有 $n$ 个线性无关特征向量，则可对角化。

常见充分条件：

- 有 $n$ 个互异特征值；
- 实对称矩阵一定可正交对角化。

## 什么时候失败

失败通常不是因为特征值算错，而是因为特征向量不够。

这时要考虑 [[Jordan Form]]。

## 常见误区

- 有特征值不等于可对角化；必须有足够多线性无关特征向量。
- 重复特征值不一定失败，但必须检查 eigenspace 维数。
- 对角化只适用于方阵；非方阵要看 [[Singular Value Decomposition]]。

## 来自课程位置

- [[02_Least Squares, Determinants and Eigenvalues#Session 2.9 Diagonalization and powers of A|Session 2.9]]：对角化与矩阵幂。
- [[03_Positive Definite Matrices and Applications#Session 3.4 Similar matrices and Jordan form|Session 3.4]]：对角化失败与 Jordan form。

## 关联卡片

- [[Eigenvalues]]
- [[Eigenvectors]]
- [[Matrix Exponential]]
- [[Jordan Form]]
- [[Choosing Matrix Decompositions]]

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

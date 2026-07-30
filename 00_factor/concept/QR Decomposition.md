---
aliases:
- QR Decomposition
- QR factorization
- QR 分解
tags:
- concept
- 线性代数
---
# QR Decomposition

## 先记一句话

QR 分解就是：**把矩阵的列方向改写成一组标准正交方向，再记录原列如何由这些方向拼出**。

常见形式是
$$
A=QR.
$$

其中：

- $Q$ 的列是标准正交向量；
- $R$ 是上三角矩阵。

## 它解决什么判断

QR 主要解决两个问题：

- 原来的列向量不正交，算投影和 least squares 很麻烦；
- 换成正交基后，内积、长度、误差都更容易处理。

所以 QR 是 [[Gram-Schmidt Orthogonalization]] 的矩阵版本。

## 一个最小图像

如果 $A$ 的列是
$$
a_1,a_2,\dots,a_n,
$$
Gram-Schmidt 把它们变成
$$
q_1,q_2,\dots,q_n.
$$

然后每个原列 $a_j$ 都可以写成前面若干个 $q_i$ 的线性组合：
$$
a_j=r_{1j}q_1+\cdots+r_{jj}q_j.
$$

把这些系数收集起来，就是上三角矩阵 $R$。

## 它在题里负责什么

- 用更稳定的方式做 least squares。
- 把 $A^TA\hat{x}=A^Tb$ 改写成更好解的三角系统。
- 解释为什么正交列让计算变简单：$Q^TQ=I$。

## 和 [[Least Squares]] 的关系

如果
$$
A=QR
$$
且 $Q$ 列正交，那么 least squares 可写为
$$
R\hat{x}=Q^Tb.
$$

这比直接形成 $A^TA$ 更干净，也更接近数值计算里的实际做法。

## 常见误区

- QR 不是只为了“分解矩阵”，它的实际作用是把列空间换成正交坐标。
- $Q$ 不一定是方阵；在 thin QR 里，$Q$ 只保留 $A$ 的列空间需要的方向。
- $R$ 上三角不是偶然，它记录的是 Gram-Schmidt 中“新列只依赖前面正交方向”的顺序。

## 来自课程位置

- [[02_Least Squares, Determinants and Eigenvalues#Session 2.4 Orthogonal matrices and Gram–Schmidt|Session 2.4]]：从 Gram-Schmidt 到 QR 分解。

## 关联卡片

- [[Gram-Schmidt Orthogonalization]]
- [[Orthogonal Matrix]]
- [[Least Squares]]
- [[Projection Matrix]]

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

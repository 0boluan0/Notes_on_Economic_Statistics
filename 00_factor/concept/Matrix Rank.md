---
aliases:
- Matrix Rank
- rank
- 秩
- 矩阵的秩
tags:
- 线性代数
- concept
---
# Matrix Rank

## 先记一句话

秩就是：**矩阵里真正独立的方向有多少个**。

一个矩阵看起来可能有很多行、很多列，但其中有些信息是重复的。[[Matrix Rank]] 就是在问：

> 这个矩阵到底提供了几个独立方向？

## 从 `Ax=b` 怎么理解

在 $Ax=b$ 里，$A$ 的列向量负责拼出右端 $b$。

- 如果列方向很多且独立，$A$ 能打出的输出范围大。
- 如果列方向重复，$A$ 能打出的输出范围小。

所以 rank 也可以理解成：

> [[Column Space]] 的维数，也就是 $A$ 能输出的有效维度。

## 一个最小例子

令
$$
A=\begin{bmatrix}
1&2&3\\
2&4&6
\end{bmatrix}.
$$

表面上它有 3 列，但三列都在同一条方向上：
$$
\begin{bmatrix}2\\4\end{bmatrix}=2\begin{bmatrix}1\\2\end{bmatrix},
\qquad
\begin{bmatrix}3\\6\end{bmatrix}=3\begin{bmatrix}1\\2\end{bmatrix}.
$$

所以这个矩阵真正只有 1 个独立列方向：
$$
\operatorname{rank}(A)=1.
$$

这会同时带来两个后果：

- [[Column Space]] 只有一维，很多 $b$ 不可达。
- [[Null Space]] 维数是 $3-1=2$，所以有两个自由方向。

## 怎么从消元里读 rank

做 row reduction 后，数 pivot 的个数：

> pivot 个数 = rank。

直觉是：每出现一个 pivot，就说明发现了一个新的独立方向；没有 pivot 的列会变成自由变量。

## 它在题里负责什么

rank 是 Unit I 的“维数账本”：

- $\dim C(A)=r$：列空间维数是 rank。
- $\dim C(A^T)=r$：行空间维数也是 rank。
- $\dim N(A)=n-r$：零空间维数是列数减 rank。
- $\dim N(A^T)=m-r$：左零空间维数是行数减 rank。

如果 $A$ 是 $m\times n$ 矩阵，记 $r=\operatorname{rank}(A)$，那么 rank 把四个基本子空间的维数全部定住。

## 和可逆性的关系

对 $n\times n$ 方阵：

- $\operatorname{rank}(A)=n$：满秩，$A$ 可逆。
- $\operatorname{rank}(A)<n$：秩亏，$A$ 奇异，不可逆。

但对非方阵，不能直接说“可逆”。这时要改说满列秩、满行秩、left inverse 或 right inverse。

## 必要公式

$$
0\leq \operatorname{rank}(A)\leq \min(m,n).
$$

$$
\operatorname{rank}(A)=\dim C(A)=\dim C(A^T).
$$

$$
\operatorname{rank}(A)+\dim N(A)=n.
$$

## 常见误区

- 行数或列数多，不代表 rank 高；重复方向不增加 rank。
- row reduction 会改变具体列向量，所以找 [[Column Space]] 的基要回到原矩阵取 pivot columns。
- rank 不是单纯的计算结果，它同时控制可达性、自由变量、维数和可逆性。

## 来自课程位置

- [[01_Ax = b and the Four Subspaces#Session 1.3 Elimination with matrices|Session 1.3]]：pivot 个数就是 rank。
- [[01_Ax = b and the Four Subspaces#Session 1.7 Column space and nullspace|Session 1.7]]：rank 是列空间维数。
- [[01_Ax = b and the Four Subspaces#Session 1.8 Solving Ax = 0: pivot variables, special solutions|Session 1.8]]：自由变量个数是 $n-r$。
- [[01_Ax = b and the Four Subspaces#Session 1.11 The four fundamental subspaces|Session 1.11]]：rank 决定四个基本子空间的维数。

## 关联卡片

- [[Column Space]]
- [[Null Space]]
- [[Row Space]]
- [[Left Nullspace]]
- [[Reduced Row Echelon Form]]
- [[Invertible Matrix Equivalence Chain]]

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

---
aliases:
- Column Space
- column space
- 列空间
- Col(A)
- C(A)
tags:
- 线性代数
- concept
---

# Column Space

## 它是什么

列空间就是：矩阵 $A$ 能做出来的所有输出。

## 先记一句话

列空间就是：**矩阵 $A$ 能做出来的所有输出**。

学 $Ax=b$ 时，可以把 $A$ 想成一台机器：

- 输入：$x$
- 输出：$Ax$
- 目标：让输出等于 $b$

所以问 $Ax=b$ 有没有解，本质是在问：

> 目标 $b$ 是不是这台机器可能输出的东西？

如果是，$Ax=b$ 有解。
如果不是，$Ax=b$ 无解。

## 怎么读 $Ax=b$

假设 $A$ 有两列：
$$
A=\begin{bmatrix}a_1&a_2\end{bmatrix}.
$$

那么
$$
Ax=x_1a_1+x_2a_2.
$$

这句话的意思是：$Ax$ 只能由 $A$ 的列向量拼出来。你可以调 $x_1,x_2$，也就是调每一列取多少倍，但你不能凭空造出列向量方向之外的东西。

所以：

> 列空间 = 所有“能由 $A$ 的列向量拼出来”的向量。

记号写作：
$$
C(A)=\operatorname{span}\{A\text{ 的所有列}\}.
$$

## 一个最小例子

令
$$
A=\begin{bmatrix}
1&2\\
2&4
\end{bmatrix}.
$$

它的两列是
$$
\begin{bmatrix}1\\2\end{bmatrix},
\qquad
\begin{bmatrix}2\\4\end{bmatrix}.
$$

第二列只是第一列的 2 倍，所以这两列其实只给了一个方向：
$$
C(A)=\operatorname{span}\left\{
\begin{bmatrix}1\\2\end{bmatrix}
\right\}.
$$

也就是说，$A$ 只能输出落在这条直线上的向量。

现在看两个右端：

$$
b_1=\begin{bmatrix}3\\6\end{bmatrix},
\qquad
b_2=\begin{bmatrix}3\\5\end{bmatrix}.
$$

- $b_1$ 在同一条直线上，因为 $6=2\cdot 3$，所以 $Ax=b_1$ 有解。
- $b_2$ 不在那条直线上，因为 $5\neq 2\cdot 3$，所以 $Ax=b_2$ 无解。

这里不需要先算完整答案。你先判断：**这个 $b$ 是否落在 $A$ 的列空间里**。

## 它在题里负责什么

- $b\in C(A)$：系统相容，至少有一个解。
- $b\notin C(A)$：系统不相容，无解。
- 列空间越小，不可达的 $b$ 越多。
- 列空间如果铺满整个输出空间，那么每个 $b$ 都可达。

## 和 [[Null Space]] 的分工

这两个概念一定要分开：

| 问题 | 看什么 |
| --- | --- |
| $Ax=b$ 有没有解？ | 看 $b$ 是否在 [[Column Space]] |
| 有解以后是否唯一？ | 看 [[Null Space]] 有没有非零向量 |

一句话：

- [[Column Space]] 管 **存在性**。
- [[Null Space]] 管 **唯一性**。

## 必要公式

如果
$$
A=\begin{bmatrix}a_1&a_2&\cdots&a_n\end{bmatrix}\in\mathbb{R}^{m\times n},
$$
那么
$$
C(A)=\{x_1a_1+\cdots+x_na_n:x_i\in\mathbb{R}\}.
$$

列空间活在 $\mathbb{R}^m$，因为 $Ax$ 的输出有 $m$ 个分量。

并且：
$$
\dim C(A)=\operatorname{rank}(A).
$$

也就是说，列空间的维数就是 [[Matrix Rank]]。

## 常见误区

- 列空间的基要从**原矩阵 A 的 pivot columns** 中取，不要直接拿 rref 的 pivot columns 当基。
- 对方阵来说，$C(A)=\mathbb{R}^n$ 才表示每个 $b$ 都可达；但“某一个 $b$ 可达”不能说明 $A$ 可逆。
- 不要把列空间和 [[Null Space]] 放在同一个空间里比较。列空间在输出端，零空间在输入端。

## 来自课程位置

- [[01_Ax = b and the Four Subspaces#Session 1.1 The geometry of linear equations|Session 1.1]]：column picture 把 $Ax=b$ 读成列向量线性组合。
- [[01_Ax = b and the Four Subspaces#Session 1.7 Column space and nullspace|Session 1.7]]：列空间控制哪些 $b$ 可达。
- [[01_Ax = b and the Four Subspaces#Session 1.9 Solving Ax = b: row reduced form R|Session 1.9]]：一般系统先判断 $b$ 是否落在列空间中。
- [[01_Ax = b and the Four Subspaces#Session 1.11 The four fundamental subspaces|Session 1.11]]：列空间是四个基本子空间之一，与 [[Left Nullspace]] 正交互补。

## 关联卡片
- [[Null Space]]
- [[Subspace]]
- [[Matrix Rank]]
- [[Linear system solution structure]]
- [[Reading the Four Fundamental Subspaces from RREF]]
- [[Left Nullspace]]

## 边界与相关概念

本卡片只保留该概念的定义、记号与最小直觉；具体估计步骤、证明或完整应用应放在对应的 `procedure`、`proof` 或 `framework` 卡片中。

相关卡片：[[Null Space]]、[[Matrix Rank]]、[[01_Ax = b and the Four Subspaces]]、[[Left Nullspace]]、[[Subspace]]、[[Linear system solution structure]]、[[Reading the Four Fundamental Subspaces from RREF]]。

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

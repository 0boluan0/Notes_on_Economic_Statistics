---
aliases:
- Null Space
- nullspace
- Kernel
- 核
- 零空间
tags:
- 线性代数
- concept
---
# Null Space

## 先记一句话

零空间就是：**矩阵 $A$ 会压成 0 的所有输入方向**。

学 $Ax=0$ 时，可以把 $A$ 想成一台机器：

- 输入：$x$
- 输出：$Ax$
- 目标：输出变成 0

所以 [[Null Space]] 问的是：

> 哪些非零输入会被 $A$ 完全吃掉？

如果只有 $x=0$ 会被吃掉，说明 $A$ 没有多余输入方向。
如果存在非零 $x$ 也被吃掉，说明 $A$ 把某些方向压扁了。

## 怎么读 $Ax=0$

假设
$$
A=\begin{bmatrix}a_1&a_2&\cdots&a_n\end{bmatrix}.
$$

那么
$$
Ax=x_1a_1+x_2a_2+\cdots+x_na_n.
$$

所以 $Ax=0$ 的意思是：

> 能不能用 $A$ 的列向量拼出 0，而且不是所有系数都为 0？

如果可以，说明这些列向量之间有冗余；也就是说，有些列可以被其他列解释掉。

## 一个最小例子

令
$$
A=\begin{bmatrix}
1&2\\
2&4
\end{bmatrix}.
$$

要求 $Ax=0$，就是
$$
x_1\begin{bmatrix}1\\2\end{bmatrix}
+x_2\begin{bmatrix}2\\4\end{bmatrix}
=0.
$$

因为第二列是第一列的 2 倍，只要
$$
x_1=-2x_2
$$
就会抵消掉。令 $x_2=t$，得到
$$
x=t\begin{bmatrix}-2\\1\end{bmatrix}.
$$

所以
$$
N(A)=\operatorname{span}\left\{
\begin{bmatrix}-2\\1\end{bmatrix}
\right\}.
$$

这说明 $A$ 会把方向 $\begin{bmatrix}-2\\1\end{bmatrix}$ 压成 0。

## 它在题里负责什么

零空间负责判断：**有解以后是否唯一**。

如果 $x_p$ 是 $Ax=b$ 的一个解，且 $x_n\in N(A)$，那么
$$
A(x_p+x_n)=b.
$$

也就是说，一旦你有一个解，就可以沿着零空间方向移动，仍然是解。

所以：

- $N(A)=\{0\}$：没有多余方向，解如果存在就是唯一的。
- $N(A)$ 有非零向量：解如果存在，就一定不唯一。

## 和 [[Column Space]] 的分工

| 问题 | 看什么 |
| --- | --- |
| $Ax=b$ 有没有解？ | 看 $b$ 是否在 [[Column Space]] |
| 有解以后是否唯一？ | 看 [[Null Space]] 是否只有零向量 |

一句话：

- [[Column Space]] 管 **存在性**。
- [[Null Space]] 管 **唯一性**。

## 必要公式

对 $m\times n$ 矩阵 $A$，
$$
N(A)=\{x\in\mathbb{R}^n:Ax=0\}.
$$

零空间活在 $\mathbb{R}^n$，因为它描述的是输入 $x$ 的自由方向。

如果 $\operatorname{rank}(A)=r$，那么
$$
\dim N(A)=n-r.
$$

这就是 [[Matrix Rank]] 与自由变量数量的关系。

## 常见误区

- $Ax=0$ 永远有零解；真正要问的是有没有**非零解**。
- 零空间里的向量长度等于 $A$ 的列数 $n$，不是行数 $m$。
- 零空间不是某一个解，而是一整个子空间。
- 不要把“无解”和“零空间只有零向量”混在一起。$Ax=0$ 不会无解；一般的 $Ax=b$ 才可能无解。

## 来自课程位置

- [[01_Ax = b and the Four Subspaces#Session 1.7 Column space and nullspace|Session 1.7]]：零空间控制为什么解不唯一。
- [[01_Ax = b and the Four Subspaces#Session 1.8 Solving Ax = 0: pivot variables, special solutions|Session 1.8]]：用 free variables 和 special solutions 写出零空间基。
- [[01_Ax = b and the Four Subspaces#Session 1.9 Solving Ax = b: row reduced form R|Session 1.9]]：一般解写成 particular solution + nullspace。
- [[01_Ax = b and the Four Subspaces#Session 1.11 The four fundamental subspaces|Session 1.11]]：零空间是四个基本子空间之一，与 [[Row Space]] 正交互补。

## 关联卡片

- [[Column Space]]
- [[Linear system solution structure]]
- [[Matrix Rank]]
- [[Reduced Row Echelon Form]]
- [[Reading the Four Fundamental Subspaces from RREF]]
- [[Row Space]]
- [[Subspace]]

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

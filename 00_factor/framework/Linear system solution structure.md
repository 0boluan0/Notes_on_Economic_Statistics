---
aliases:
- Linear system solution structure
- 线性方程组解的结构
- particular solution plus nullspace
- 特解加零空间
tags:
- framework
- 线性代数
---
# Linear system solution structure

## 这张卡回答什么问题

给定 $Ax=b$，你不要一上来就只想“怎么算”。先问三个问题：

1. 有没有解？
2. 如果有解，是唯一还是无穷多？
3. 如果无穷多，全部解长什么样？

这张卡的核心框架是：

> 先用 [[Column Space]] 判断有没有解，再用 [[Null Space]] 描述所有解。

## 一句话框架

如果 $Ax=b$ 有一个特解 $x_p$，那么全部解都是
$$
x=x_p+x_n,\qquad x_n\in N(A).
$$

也就是：

> 全部解 = 一个特解 + 零空间里的所有自由方向。

## 为什么是这样

假设 $x_p$ 已经满足
$$
Ax_p=b.
$$

如果 $x_n\in N(A)$，那么
$$
Ax_n=0.
$$

所以
$$
A(x_p+x_n)=Ax_p+Ax_n=b+0=b.
$$

这说明：从一个解出发，只要沿着零空间方向走，仍然是解。

反过来，如果 $x_1,x_2$ 都是 $Ax=b$ 的解，那么
$$
A(x_1-x_2)=Ax_1-Ax_2=b-b=0.
$$

所以两个解之间的差一定在 [[Null Space]] 里。

## 三种结局

| 条件 | 解的情况 | 直觉 |
| --- | --- | --- |
| $b\notin C(A)$ | 无解 | 目标 $b$ 不可达 |
| $b\in C(A)$ 且 $N(A)=\{0\}$ | 唯一解 | 可达，而且没有自由方向 |
| $b\in C(A)$ 且 $N(A)$ 有非零向量 | 无穷多解 | 可达，但可以沿零空间移动 |

这就是 Unit I 里最重要的解结构。

## 一个最小例子

令
$$
A=\begin{bmatrix}
1&2\\
2&4
\end{bmatrix}.
$$

### 情况 1：无解

若
$$
b=\begin{bmatrix}3\\5\end{bmatrix},
$$
它不在 [[Column Space]] 中，所以 $Ax=b$ 无解。

### 情况 2：无穷多解

若
$$
b=\begin{bmatrix}3\\6\end{bmatrix},
$$
它在 [[Column Space]] 中，所以有解。

一个特解是
$$
x_p=\begin{bmatrix}3\\0\end{bmatrix}.
$$

而
$$
N(A)=\operatorname{span}\left\{
\begin{bmatrix}-2\\1\end{bmatrix}
\right\}.
$$

所以全部解是
$$
x=\begin{bmatrix}3\\0\end{bmatrix}
+t\begin{bmatrix}-2\\1\end{bmatrix}.
$$

这就是“特解 + 零空间”。

## 做题时怎么用

遇到 $Ax=b$：

1. 先判断 $b$ 是否在 [[Column Space]] 中。
2. 如果不在，停止：无解。
3. 如果在，找一个 particular solution $x_p$。
4. 再求 [[Null Space]] $N(A)$。
5. 把全部解写成 $x=x_p+x_n$。

## 常见误区

- particular solution 不唯一；换一个特解，最后描述的是同一个解集。
- 零空间只由 $A$ 决定，和 $b$ 无关。
- 无解时不能写“特解 + 零空间”，因为根本没有特解。
- 有自由变量不代表无解；它表示有解时不唯一。

## 来自课程位置

- [[01_Ax = b and the Four Subspaces#Session 1.1 The geometry of linear equations|Session 1.1]]：唯一解、无解、无穷多解的几何图像。
- [[01_Ax = b and the Four Subspaces#Session 1.7 Column space and nullspace|Session 1.7]]：列空间控制存在性，零空间控制唯一性。
- [[01_Ax = b and the Four Subspaces#Session 1.9 Solving Ax = b: row reduced form R|Session 1.9]]：全部解写成 particular solution + nullspace。

## 关联卡片

- [[Column Space]]
- [[Null Space]]
- [[Matrix Rank]]
- [[Reduced Row Echelon Form]]
- [[Reading the Four Fundamental Subspaces from RREF]]

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

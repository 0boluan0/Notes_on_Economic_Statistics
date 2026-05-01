---
aliases:
- Reduced Row Echelon Form
- Row Reduced Form
- RREF
- rref
- 行最简形
- 简化行阶梯形
tags:
- 线性代数
- concept
---
# Reduced Row Echelon Form

## 先记一句话

RREF 是：**把方程组消元到最容易读结构的形状**。

它不是为了把矩阵变漂亮，而是为了让你一眼读出：

- pivot columns
- free variables
- rank
- nullspace
- general solution

## 它长什么样

一个矩阵处于 reduced row echelon form 时，通常有这些特征：

- 每个非零行都有一个 leading 1；
- leading 1 所在列的其他位置都是 0；
- leading 1 从上到下逐渐向右移动；
- 零行放在最下面。

你不需要把这些特征死背成清单。更重要的是知道：

> RREF 把主变量和自由变量的关系完全暴露出来。

## 一个最小例子

若增广矩阵的 rref 是
$$
\left[
\begin{array}{ccc|c}
1&0&2&1\\
0&1&-1&3
\end{array}
\right],
$$
那么 $x_3$ 是自由变量。令 $x_3=t$：
$$
x_1=1-2t,
\qquad
x_2=3+t.
$$

所以全部解是
$$
x=
\begin{bmatrix}1\\3\\0\end{bmatrix}
+t\begin{bmatrix}-2\\1\\1\end{bmatrix}.
$$

这里 RREF 直接暴露了：

- 一个 particular solution；
- 一个 nullspace 方向；
- rank 是 2；
- free variable 有 1 个。

## 它在题里负责什么

- 用 pivot 个数读 [[Matrix Rank]]。
- 用 free variables 写 [[Null Space]]。
- 用矛盾行判断 $Ax=b$ 是否无解。
- 用 pivot columns 位置回到原矩阵找 [[Column Space]] 的 basis。
- 用非零行找 [[Row Space]] 的 basis。

## 常见误区

- rref 的 pivot columns 位置很重要，但列空间基要回到原矩阵取对应列。
- row reduction 保持解集，但不保持原列向量的具体几何位置。
- RREF 不是最终答案；它是读 rank、basis、nullspace、general solution 的中间结构。

## 来自课程位置

- [[01_Ax = b and the Four Subspaces#Session 1.3 Elimination with matrices|Session 1.3]]：消元让 pivot / rank / free variables 可读。
- [[01_Ax = b and the Four Subspaces#Session 1.8 Solving Ax = 0: pivot variables, special solutions|Session 1.8]]：从 rref 读 nullspace。
- [[01_Ax = b and the Four Subspaces#Session 1.9 Solving Ax = b: row reduced form R|Session 1.9]]：从 rref 读一般解和相容条件。
- [[01_Ax = b and the Four Subspaces#Session 1.11 The four fundamental subspaces|Session 1.11]]：从 rref 读四个基本子空间。

## 关联卡片

- [[Matrix Rank]]
- [[Null Space]]
- [[Column Space]]
- [[Row Space]]
- [[Linear system solution structure]]
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

---
aliases:
- Subspace
- 线性子空间
- 子空间
tags:
- 线性代数
- concept
---
# Subspace

## 先记一句话

子空间就是：**大空间里一个仍然像空间一样运作的小空间**。

如果一个集合在 $\mathbb{R}^n$ 里，并且你在里面做加法、数乘都不会跑出去，它就是 subspace。

## 最快判断法

判断一个集合是不是 subspace，先问三件事：

1. 有没有零向量？
2. 两个里面的向量相加，还在里面吗？
3. 里面的向量乘任意数，还在里面吗？

更快的版本：

> 任意线性组合还在里面吗？

也就是：
$$
u,v\in S,\ c,d\in\mathbb{R}
\quad\Longrightarrow\quad
cu+dv\in S.
$$

## 一个最小例子

在 $\mathbb{R}^3$ 中，
$$
S=\{(x,y,z):x+y+z=0\}
$$
是 subspace。

因为它是过原点的平面。

但
$$
T=\{(x,y,z):x+y+z=1\}
$$
不是 subspace。

因为它不经过原点。你把其中一个向量乘以 0，会得到 0，但 0 不满足 $x+y+z=1$。

## Unit I 里最重要的子空间

对 $m\times n$ 矩阵 $A$：

- [[Column Space]] 是 $\mathbb{R}^m$ 的子空间。
- [[Null Space]] 是 $\mathbb{R}^n$ 的子空间。
- [[Row Space]] 是 $\mathbb{R}^n$ 的子空间。
- [[Left Nullspace]] 是 $\mathbb{R}^m$ 的子空间。

这四个就是 Unit I 的核心对象。

## 和齐次 / 非齐次方程的关系

齐次方程
$$
Ax=0
$$
的解集是 subspace，因为它经过原点，并且对线性组合封闭。

非齐次方程
$$
Ax=b,\qquad b\neq 0
$$
的解集通常不是 subspace，而是一个平移后的 subspace。它可能是一条不经过原点的直线，或一个不经过原点的平面。

## 常见误区

- 看起来像直线/平面，不代表是 subspace；必须经过原点。
- 只检查加法不够，还要检查数乘。
- $Ax=b$ 的解集如果 $b\neq0$，即使有无穷多解，也通常不是 subspace。
- subspace 不是“空间里任意一块区域”，它必须保留线性结构。

## 来自课程位置

- [[01_Ax = b and the Four Subspaces#Session 1.6 Transposes, permutations, vector spaces|Session 1.6]]：用 closure 判断子空间。
- [[01_Ax = b and the Four Subspaces#Session 1.7 Column space and nullspace|Session 1.7]]：column space 和 nullspace 是矩阵天然带出的子空间。

## 关联卡片

- [[Vector Space]]
- [[Column Space]]
- [[Null Space]]
- [[Row Space]]
- [[Left Nullspace]]
- [[Linear system solution structure]]

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

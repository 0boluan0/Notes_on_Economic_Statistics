---
aliases:
- Dimension
- dimension
- 维数
tags:
- 线性代数
- concept
---
# Dimension

## 先记一句话

维数就是：**描述一个空间最少需要几个独立方向**。

更正式地说：

> dimension 是任意一组 [[Basis]] 中向量的个数。

## 为什么 basis 不唯一，但 dimension 唯一

同一个空间可以有很多不同的 basis。

比如 $\mathbb{R}^2$ 中，标准基是一组 basis：
$$
\begin{bmatrix}1\\0\end{bmatrix},
\begin{bmatrix}0\\1\end{bmatrix}.
$$

但
$$
\begin{bmatrix}1\\1\end{bmatrix},
\begin{bmatrix}1\\-1\end{bmatrix}
$$
也是一组 basis。

它们长得不同，但都有 2 个向量。所以 $\mathbb{R}^2$ 的维数是 2。

## 一个最小例子

平面
$$
S=\{(x,y,z):x+y+z=0\}\subseteq\mathbb{R}^3
$$
是二维的。

虽然它活在三维空间里，但描述它只需要两个独立方向，例如：
$$
\begin{bmatrix}1\\-1\\0\end{bmatrix},
\qquad
\begin{bmatrix}1\\0\\-1\end{bmatrix}.
$$

所以
$$
\dim S=2.
$$

## Unit I 里的维数账本

对 $m\times n$ 矩阵 $A$，若 $\operatorname{rank}(A)=r$：

$$
\dim C(A)=r,
\qquad
\dim C(A^T)=r,
$$
$$
\dim N(A)=n-r,
\qquad
\dim N(A^T)=m-r.
$$

这四个数字是四个基本子空间的维数闭环。

## 常见误区

- 空间活在 $\mathbb{R}^3$ 里，不代表它一定是 3 维。
- basis 可以换，但 dimension 不会变。
- rank 不是“矩阵大小”，而是 column space / row space 的 dimension。

## 来自课程位置

- [[01_Ax = b and the Four Subspaces#Session 1.10 Independence, basis, and dimension|Session 1.10]]：dimension 是 basis 的向量个数。
- [[01_Ax = b and the Four Subspaces#Session 1.11 The four fundamental subspaces|Session 1.11]]：四个基本子空间的维数由 rank 决定。

## 关联卡片

- [[Basis]]
- [[Linear Independence]]
- [[Matrix Rank]]
- [[Column Space]]
- [[Null Space]]
- [[Row Space]]
- [[Left Nullspace]]

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

---
aliases:
- Basis
- basis
- 基
- 基底
tags:
- 线性代数
- concept
---

# Basis

## 它是什么

基就是：刚好够用、没有冗余的一组向量。

## 先记一句话

基就是：**刚好够用、没有冗余的一组向量**。

它同时满足两件事：

1. 能张成整个空间；
2. 线性无关。

所以 basis 是一个空间的“最小完整坐标框架”。

## 为什么要有 basis

如果一组向量能张成空间，但里面有冗余，你可以删掉一些。

如果一组向量线性无关，但还不能张成整个空间，你还需要补方向。

basis 正好卡在中间：

> 不多，也不少。

## 一个最小例子

在 $\mathbb{R}^2$ 中，
$$
e_1=\begin{bmatrix}1\\0\end{bmatrix},
\qquad
e_2=\begin{bmatrix}0\\1\end{bmatrix}
$$
是一组 basis。

因为任意
$$
\begin{bmatrix}x\\y\end{bmatrix}
$$
都能唯一写成
$$
x e_1+y e_2.
$$

这里“唯一”很重要：它来自线性无关。

## 矩阵里怎么找 basis

- [[Column Space]] 的 basis：取原矩阵的 pivot columns。
- [[Row Space]] 的 basis：取 rref 的非零行。
- [[Null Space]] 的 basis：取 special solutions。
- [[Left Nullspace]] 的 basis：解 $A^Ty=0$ 得到 special solutions。

## 和 [[Dimension]] 的关系

一个空间可以有很多不同的 basis，但所有 basis 的向量个数都一样。

这个固定个数就是 [[Dimension]]。

## 常见误区

- basis 不唯一；唯一的是 basis 的向量个数。
- “张成”不够，还必须线性无关。
- “线性无关”不够，还必须张成整个目标空间。
- 找 column space 的 basis 时，要回到原矩阵取 pivot columns，不要从 rref 直接抄列。

## 来自课程位置

- [[01_Ax = b and the Four Subspaces#Session 1.10 Independence, basis, and dimension|Session 1.10]]：basis 是 independence 和 span 的结合。
- [[01_Ax = b and the Four Subspaces#Session 1.11 The four fundamental subspaces|Session 1.11]]：四个基本子空间都要会找 basis。

## 关联卡片

- [[Linear Independence]]
- [[Dimension]]
- [[Matrix Rank]]
- [[Column Space]]
- [[Null Space]]
- [[Row Space]]
- [[Reading the Four Fundamental Subspaces from RREF]]

## 边界与相关概念

本卡片只保留该概念的定义、记号与最小直觉；具体估计步骤、证明或完整应用应放在对应的 `procedure`、`proof` 或 `framework` 卡片中。

相关卡片：[[Column Space]]、[[Row Space]]、[[Null Space]]、[[Left Nullspace]]、[[Dimension]]、[[01_Ax = b and the Four Subspaces]]、[[Linear Independence]]、[[Matrix Rank]]。

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

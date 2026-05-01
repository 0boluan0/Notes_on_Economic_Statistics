---
aliases:
- Fourier Series
- Fourier basis
- 傅里叶级数
- 傅里叶基
tags:
- concept
- 线性代数
---
# Fourier Series

## 先记一句话

Fourier Series 就是：**把函数拆到一组正交的正弦、余弦基底上**。

它在线代课里的意义是：

> 投影和正交基不只存在于有限维向量，也可以用于函数空间。

## 它是什么

周期函数可以写成
$$
f(x)\sim a_0+\sum_{k=1}^{\infty}(a_k\cos kx+b_k\sin kx).
$$

这些系数本质上来自“把函数投影到正交基上”。

如果某个方向和其他方向正交，系数就能单独用内积取出来。

## 一个最小图像

在 $\mathbb{R}^n$ 中，用标准正交基展开向量：
$$
x=(q_1^Tx)q_1+\cdots+(q_n^Tx)q_n.
$$

Fourier series 做的是同一件事，只是基底变成函数：
$$
1,\cos x,\sin x,\cos2x,\sin2x,\dots
$$

## 它在题里负责什么

- 作为“正交投影思想”的函数空间版本。
- 说明为什么正交基让系数计算简单。
- 连接 [[Orthogonality]]、[[Orthogonal Projection]] 和 FFT。

## 常见误区

- Fourier series 不是单纯背三角公式；它的结构是正交投影。
- 系数公式来自内积，不是随便匹配。
- 离散 Fourier transform / FFT 是相关但更偏计算的版本。

## 来自课程位置

- [[02_Least Squares, Determinants and Eigenvalues#Session 2.11 Markov matrices; Fourier series|Session 2.11]]：Fourier basis 作为正交基和投影思想的延伸。
- [[03_Positive Definite Matrices and Applications#Session 3.2 Complex matrices; fast Fourier transform|Session 3.2]]：复数、Fourier matrix 与 FFT。

## 关联卡片

- [[Orthogonality]]
- [[Orthogonal Projection]]
- [[Orthogonal Matrix]]
- [[Change of Basis]]

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

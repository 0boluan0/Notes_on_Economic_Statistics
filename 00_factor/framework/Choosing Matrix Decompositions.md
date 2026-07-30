---
aliases:
- Choosing Matrix Decompositions
- How to Choose Matrix Decompositions
- Matrix decomposition choice
- 如何选择矩阵分解
tags:
- framework
- 线性代数
type: framework
---
# Choosing Matrix Decompositions

## 这张卡回答什么问题

看到矩阵分解题时，不要先背公式。先判断：

> 题目想利用矩阵的哪种结构？

这张卡只负责分流：什么时候想 diagonalization，什么时候想 Jordan，什么时候想 SVD，什么时候想 QR。

## 一句话框架

- 反复作用、长期行为、矩阵幂：先想 [[Diagonalization]]。
- 对角化失败：转向 [[Jordan Form]]。
- 任意矩阵、秩、压缩、伪逆：直接想 [[Singular Value Decomposition]]。
- 最小二乘、正交列、数值求解：想 [[QR Decomposition]]。

## 如何识别题型

| 题目信号 | 优先工具 | 原因 |
| --- | --- | --- |
| $A^k$、$e^{At}$、长期行为 | [[Diagonalization]] | 特征方向解耦 |
| repeated eigenvalue 且特征向量不够 | [[Jordan Form]] | 解释对角化失败 |
| rectangular、rank-deficient、low-rank approximation | [[Singular Value Decomposition]] | 任意矩阵都适用 |
| least squares、orthonormal basis、stable solve | [[QR Decomposition]] | 避免直接形成 $A^TA$ |
| symmetric / positive definite | [[Spectral Decomposition]] | 正交特征分解最强 |

## 边界和失败模式

- 非方阵不能 diagonalize。
- 有重复特征值时，不能默认 diagonalize。
- 一般矩阵不能默认有 $Q\Lambda Q^T$；那是对称矩阵的谱分解。
- SVD 很通用，但若题目明确问动力系统长期行为，eigen/diagonalization 更直接。
- QR 适合 least squares 和正交化，但它本身不直接告诉你长期行为。

## 一个最小判断例子

题目问：

> Compute $A^{100}$.

先想 [[Diagonalization]]，因为矩阵幂在特征向量基底里最简单。

题目问：

> Find the best rank-1 approximation of A.

直接想 [[Singular Value Decomposition]]，因为低秩近似由最大 singular values 控制。

题目问：

> Solve an overdetermined least squares problem stably.

优先想 [[QR Decomposition]]，而不是先把 $A^TA$ 算出来。

## 来自课程位置

- [[02_Least Squares, Determinants and Eigenvalues#Session 2.9 Diagonalization and powers of A|Session 2.9]]：diagonalization 与 powers。
- [[03_Positive Definite Matrices and Applications#Session 3.4 Similar matrices and Jordan form|Session 3.4]]：Jordan form。
- [[03_Positive Definite Matrices and Applications#Session 3.5 Singular value decomposition|Session 3.5]]：SVD。
- [[02_Least Squares, Determinants and Eigenvalues#Session 2.4 Orthogonal matrices and Gram–Schmidt|Session 2.4]]：QR。

## 关联卡片

- [[Diagonalization]]
- [[Jordan Form]]
- [[Singular Value Decomposition]]
- [[QR Decomposition]]
- [[Spectral Decomposition]]
- [[Pseudoinverse]]

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

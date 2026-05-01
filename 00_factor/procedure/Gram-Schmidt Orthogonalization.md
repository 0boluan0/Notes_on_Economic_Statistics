---
aliases:
- Gram-Schmidt Orthogonalization
- Gram-Schmidt
- Gram-Schmidt process
- 格拉姆-施密特正交化
tags:
- procedure
- 线性代数
---
# Gram-Schmidt Orthogonalization

## 这张卡什么时候用

当题目给你一组线性无关向量
$$
a_1,\dots,a_n
$$
并要求构造正交组、标准正交组或 QR 分解时，用 Gram-Schmidt。

## 输入

- 一组线性无关向量 $a_1,\dots,a_n$。
- 它们张成的子空间不变；算法只是在同一个子空间里换成更好用的正交方向。

## 输出

- 正交向量 $u_1,\dots,u_n$；
- 标准正交向量 $q_1,\dots,q_n$；
- 若把 $q_i$ 按列放入 $Q$，可得到 [[QR Decomposition]]。

## Step 1. 保留第一个方向

令
$$
u_1=a_1.
$$

如果要标准正交向量，再单位化：
$$
q_1=\frac{u_1}{\|u_1\|}.
$$

## Step 2. 每来一个新向量，减掉旧方向上的投影

对第 $k$ 个向量：
$$
u_k=a_k-\sum_{j=1}^{k-1}\operatorname{proj}_{u_j}(a_k).
$$

其中
$$
\operatorname{proj}_{u_j}(a_k)=u_j\frac{u_j^Ta_k}{u_j^Tu_j}.
$$

这一步的意思是：把 $a_k$ 中已经被旧正交方向解释掉的部分全部拿掉，只保留新的正交方向。

## Step 3. 单位化

如果需要标准正交组，令
$$
q_k=\frac{u_k}{\|u_k\|}.
$$

所有 $q_k$ 都满足
$$
q_i^Tq_j=0\ (i\neq j),\qquad \|q_i\|=1.
$$

## Step 4. 检查输出

- 每个 $u_k$ 都应该与前面的 $u_j$ 正交。
- 每个 $q_k$ 的长度应该是 1。
- 新向量组张成的空间应该和原向量组相同。

## 常见错误

- 只减去最近一个方向，而不是减去所有已经生成的方向。
- 忘记单位化，导致得到的是正交组而不是标准正交组。
- 输入向量线性相关时，某一步会得到零向量；这不是小数误差，而是说明没有新方向。

## 来自课程位置

- [[02_Least Squares, Determinants and Eigenvalues#Session 2.4 Orthogonal matrices and Gram-Schmidt|Session 2.4]]：Gram-Schmidt、orthogonal matrix 与 QR。

## 关联卡片

- [[Orthogonality]]
- [[Orthogonal Matrix]]
- [[QR Decomposition]]
- [[Projection Matrix]]

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

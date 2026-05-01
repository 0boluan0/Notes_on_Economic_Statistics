---
aliases:
- Reading the Four Fundamental Subspaces from RREF
- Read Four Fundamental Subspaces from RREF
- 从 rref 读取四个基本子空间
tags:
- procedure
- 线性代数
---
# Reading the Four Fundamental Subspaces from RREF

## 这个流程做什么

给你一个矩阵 $A$，这个流程用 row reduction 的结果读出四个基本子空间：

- [[Column Space]] $C(A)$
- [[Null Space]] $N(A)$
- [[Row Space]] $C(A^T)$
- [[Left Nullspace]] $N(A^T)$

它的目标不是再解释定义，而是告诉你：**拿到 rref 后，每个空间的基从哪里读**。

## 输入

- 原矩阵 $A\in\mathbb{R}^{m\times n}$。
- $A$ 的 rref，或至少 pivot columns / free columns 信息。

## 输出

- 四个基本子空间各自的一组基。
- 四个基本子空间各自的维数。
- 每个空间活在哪个 ambient space。

## Step 1. 先做 rref，标出 pivot/free columns

对 $A$ 做 row reduction，得到 rref。

记：

- pivot 个数为 $r$
- $A$ 有 $m$ 行、$n$ 列

那么
$$
\operatorname{rank}(A)=r.
$$

## Step 2. 读 [[Column Space]]

列空间基要从**原矩阵 $A$** 取：

- 找 rref 里的 pivot columns 位置；
- 回到原矩阵 $A$；
- 取原矩阵中这些位置的列。

这些原矩阵 pivot columns 构成 $C(A)$ 的一组基。

维数：
$$
\dim C(A)=r.
$$

所在空间：
$$
C(A)\subseteq\mathbb{R}^m.
$$

## Step 3. 读 [[Row Space]]

行空间基直接从 rref 取：

- 取 rref 的所有非零行；
- 这些非零行构成 $C(A^T)$ 的一组基。

维数：
$$
\dim C(A^T)=r.
$$

所在空间：
$$
C(A^T)\subseteq\mathbb{R}^n.
$$

## Step 4. 读 [[Null Space]]

零空间来自解
$$
Ax=0.
$$

做法：

- 把 free variables 设成参数；
- 每次让一个 free variable 为 1，其余 free variables 为 0；
- 解出 pivot variables；
- 得到的向量就是 special solutions。

所有 special solutions 构成 $N(A)$ 的一组基。

维数：
$$
\dim N(A)=n-r.
$$

所在空间：
$$
N(A)\subseteq\mathbb{R}^n.
$$

## Step 5. 读 [[Left Nullspace]]

左零空间来自解
$$
A^Ty=0.
$$

最稳妥做法：

- 对 $A^T$ 重复 Step 1-4；
- 解出 $A^Ty=0$ 的 special solutions；
- 它们构成 $N(A^T)$ 的一组基。

维数：
$$
\dim N(A^T)=m-r.
$$

所在空间：
$$
N(A^T)\subseteq\mathbb{R}^m.
$$

## 最后检查

| 空间 | 从哪里读基 | 维数 | 所在空间 |
| --- | --- | --- | --- |
| $C(A)$ | 原矩阵 pivot columns | $r$ | $\mathbb{R}^m$ |
| $N(A)$ | special solutions of $Ax=0$ | $n-r$ | $\mathbb{R}^n$ |
| $C(A^T)$ | rref 的非零行 | $r$ | $\mathbb{R}^n$ |
| $N(A^T)$ | special solutions of $A^Ty=0$ | $m-r$ | $\mathbb{R}^m$ |

再检查两组正交关系：
$$
C(A^T)\perp N(A),
\qquad
C(A)\perp N(A^T).
$$

## 常见错误

- 用 rref 的 pivot columns 当 [[Column Space]] 的基。错：列空间基必须回到原矩阵取。
- 忘记四个空间活在两个不同空间里：$C(A)$、$N(A^T)$ 在 $\mathbb{R}^m$；$C(A^T)$、$N(A)$ 在 $\mathbb{R}^n$。
- 只写维数，不写基。题目要求“find the subspace”时，通常需要一组基。
- 把 left nullspace 忘掉。它不是附属品，它负责输出端的相容条件。

## 来自课程位置

- [[01_Ax = b and the Four Subspaces#Session 1.8 Solving Ax = 0: pivot variables, special solutions|Session 1.8]]：special solutions 生成 nullspace。
- [[01_Ax = b and the Four Subspaces#Session 1.10 Independence, basis, and dimension|Session 1.10]]：basis 和 dimension 的语言。
- [[01_Ax = b and the Four Subspaces#Session 1.11 The four fundamental subspaces|Session 1.11]]：四个基本子空间的总收束。

## 关联卡片

- [[Column Space]]
- [[Row Space]]
- [[Null Space]]
- [[Left Nullspace]]
- [[Matrix Rank]]
- [[Reduced Row Echelon Form]]
- [[Basis]]
- [[Dimension]]

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

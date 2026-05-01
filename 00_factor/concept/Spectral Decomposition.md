---
aliases:
- Spectral Decomposition
- Spectral Theorem
- Eigenvalue Decomposition
- 谱分解
- 谱定理
tags:
- concept
- 线性代数
---
# Spectral Decomposition

## 先记一句话

谱分解就是：**把实对称矩阵拆成一组互相正交的特征方向及其伸缩倍数**。

对实对称矩阵 $A$，
$$
A=Q\Lambda Q^T.
$$

其中 $Q$ 的列是标准正交特征向量，$\Lambda$ 的对角元是特征值。

## 它是什么

谱分解是 [[Symmetric Matrix]] 的最好结构。

它说明 $A$ 的作用可以被拆成：

- 先转到正交特征向量坐标；
- 每个方向乘以自己的特征值；
- 再转回原坐标。

也可以写成
$$
A=\lambda_1q_1q_1^T+\cdots+\lambda_nq_nq_n^T.
$$

这里 $q_iq_i^T$ 是投影到第 $i$ 个特征方向的投影矩阵。

## 一个最小例子

如果
$$
A=Q
\begin{bmatrix}
5&0\\
0&2
\end{bmatrix}
Q^T,
$$
那么 $A$ 在 $q_1$ 方向放大 5 倍，在 $q_2$ 方向放大 2 倍。

因为 $Q$ 是正交矩阵，这两个方向互不干扰。

## 它在题里负责什么

- 快速计算矩阵幂：
  $$
  A^k=Q\Lambda^kQ^T.
  $$
- 判断正定性：特征值全正。
- 理解二次型：
  $$
  x^TAx=\sum_i\lambda_i z_i^2
  $$
  其中 $z=Q^Tx$。
- 解释 PCA 中主成分方向和方差大小。

## 常见误区

- 谱分解的标准版本要求实对称矩阵；一般矩阵不能直接写成 $Q\Lambda Q^T$。
- 它不是普通 diagonalization 的同义词；普通对角化是 $S\Lambda S^{-1}$，不一定有正交 $Q$。
- 特征向量要单位化后才能放进 $Q$。

## 来自课程位置

- [[03_Positive Definite Matrices and Applications#Session 3.1 Symmetric matrices and positive definiteness|Session 3.1]]：实对称矩阵的正交对角化。

## 关联卡片

- [[Symmetric Matrix]]
- [[Positive Definite Matrix]]
- [[Orthogonal Matrix]]
- [[Eigenvalues]]
- [[Eigenvectors]]

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

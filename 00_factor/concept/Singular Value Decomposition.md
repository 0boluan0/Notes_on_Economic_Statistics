---
aliases:
- Singular Value Decomposition
- SVD
- singular value decomposition
- 奇异值分解
tags:
- concept
- 线性代数
---
# Singular Value Decomposition

## 先记一句话

SVD 就是：**任意矩阵都能被拆成输入正交方向、缩放大小、输出正交方向**。

对任意 $m\times n$ 矩阵 $A$，
$$
A=U\Sigma V^T.
$$

这比对角化更通用：不要求方阵，不要求对称，也不要求有特征向量基。

## 它是什么

在
$$
A=U\Sigma V^T
$$
中：

- $V$ 的列是输入空间里的正交方向；
- $\Sigma$ 的非负对角元是 singular values；
- $U$ 的列是输出空间里的正交方向。

几何上，$A$ 做了三件事：

1. 先在输入空间换到 $V$ 这组好坐标；
2. 沿每个方向按 $\sigma_i$ 缩放；
3. 再转到输出空间的 $U$ 方向。

## 一个最小图像

如果一个矩阵 rank 为 1，那么它只有一个非零 singular value。

这说明它只保留一个有效输入方向，其他方向都被压到 0 或无贡献。

所以 SVD 会直接暴露 rank：
$$
\operatorname{rank}(A)=\text{非零 singular values 的个数}.
$$

## 它在题里负责什么

- 统一处理任意矩阵的结构。
- 读 rank、range、nullspace 和四个基本子空间。
- 构造 [[Pseudoinverse]]。
- 做低秩近似和压缩。
- 解释 ill-conditioning：小 singular values 会放大反解中的误差。

## 和 eigen 的关系

Singular values 来自
$$
A^TA
$$
的特征值：
$$
\sigma_i=\sqrt{\lambda_i(A^TA)}.
$$

所以 SVD 可以看成把非对称、非方阵的问题转回对称正定/半正定结构。

## 常见误区

- SVD 不是只给方阵用；它正是为任意矩阵准备的。
- singular value 一定非负，和 eigenvalue 可能为负或复数不同。
- $U$ 和 $V$ 分别活在输出空间和输入空间，不要混成一组方向。

## 来自课程位置

- [[03_Positive Definite Matrices and Applications#Session 3.5 Singular value decomposition|Session 3.5]]：SVD、四个基本子空间与伪逆。
- [[03_Positive Definite Matrices and Applications#Session 3.8 Left and right inverses; pseudoinverse|Session 3.8]]：SVD 支撑 Moore-Penrose pseudoinverse。

## 关联卡片

- [[Pseudoinverse]]
- [[Orthogonal Matrix]]
- [[Spectral Decomposition]]
- [[Choosing Matrix Decompositions]]
- [[Matrix Rank]]

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

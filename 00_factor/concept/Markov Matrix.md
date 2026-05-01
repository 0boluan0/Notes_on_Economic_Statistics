---
aliases:
- Markov Matrix
- stochastic matrix
- transition matrix
- 马尔可夫矩阵
- 转移矩阵
tags:
- concept
- 线性代数
---
# Markov Matrix

## 先记一句话

Markov matrix 就是：**描述概率或人口如何在状态之间转移的矩阵**。

课程里常用列随机矩阵：每列元素非负，且每列和为 1。

## 它是什么

若状态向量 $u_k$ 表示第 $k$ 期各状态的比例或数量，转移规则写成
$$
u_{k+1}=Au_k.
$$

如果 $A$ 的列和为 1 且元素非负，那么总量被保留：
$$
\mathbf{1}^Tu_{k+1}=\mathbf{1}^Tu_k.
$$

## 一个最小例子

两状态转移：
$$
A=
\begin{bmatrix}
0.8&0.3\\
0.2&0.7
\end{bmatrix}.
$$

每列和都是 1：
$$
0.8+0.2=1,\qquad 0.3+0.7=1.
$$

如果
$$
u_0=\begin{bmatrix}1\\0\end{bmatrix},
$$
那么 $u_1=Au_0$ 给出一期后在两个状态中的分布。

## 稳态怎么看

稳态向量满足
$$
Au=u.
$$

这等价于
$$
(A-I)u=0.
$$

所以稳态是特征值 $1$ 对应的 [[Eigenvectors]]。

## 它在题里负责什么

- 分析反复转移 $u_k=A^ku_0$。
- 找长期稳定分布。
- 用 eigenvalues 判断收敛速度。
- 连接矩阵幂、对角化和动力系统。

## 常见误区

- 课程里可能用“列和为 1”的约定；有些教材用“行和为 1”。先看状态向量写在左边还是右边。
- 稳态不是 $Au=0$，而是 $Au=u$。
- 特征值 1 很重要，但是否唯一、是否收敛还要看其他特征值和链的结构。

## 来自课程位置

- [[02_Least Squares, Determinants and Eigenvalues#Session 2.11 Markov matrices; Fourier series|Session 2.11]]：Markov matrix、稳态和特征值 1。

## 关联卡片

- [[Eigenvalues]]
- [[Eigenvectors]]
- [[Diagonalization]]
- [[Matrix Exponential]]
- [[Stationary Distribution]]

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

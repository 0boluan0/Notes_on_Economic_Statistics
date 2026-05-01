---
aliases:
- Orthogonality
- orthogonal
- perpendicular
- 正交
- 正交性
tags:
- concept
- 线性代数
---
# Orthogonality

## 先记一句话

正交就是：**两个方向在内积意义下互不贡献、互不解释**。

在二维图里它像“垂直”。在线性代数里更重要的是：

> 正交让一个向量可以被拆成互不干扰的几部分。

所以 Unit II 里很多东西都从正交开始：投影、最小二乘、Gram-Schmidt、Fourier、对称矩阵的正交特征向量。

## 它是什么

两个向量 $x,y$ 正交，当且仅当
$$
x^Ty=0.
$$

如果 $S,T$ 是两个子空间，说 $S\perp T$，意思是任意
$$
s\in S,\qquad t\in T
$$
都满足
$$
s^Tt=0.
$$

这不是说两个空间“看起来分开”，而是说它们里面任意方向都互相没有投影分量。

## 一个最小例子

在 $\mathbb{R}^2$ 中，
$$
e_1=\begin{bmatrix}1\\0\end{bmatrix},
\qquad
e_2=\begin{bmatrix}0\\1\end{bmatrix}
$$
满足
$$
e_1^Te_2=0.
$$

所以它们正交。

如果一个向量写成
$$
x=3e_1+2e_2,
$$
那么 $3e_1$ 和 $2e_2$ 两部分的长度可以分开算：
$$
\|x\|^2=3^2+2^2.
$$

这就是后面投影和最小二乘一直在用的几何账本。

## 它在题里负责什么

- 判断误差是否已经最小：看残差是否与目标子空间正交。
- 建立投影条件：$b-p\perp S$。
- 建立正规方程：$A^T(b-A\hat{x})=0$。
- 解释四个基本子空间：
  $$
  C(A^T)\perp N(A),\qquad C(A)\perp N(A^T).
  $$
- 解释为什么正交基好用：坐标互不干扰，长度和误差可以分开看。

## 和 [[Orthogonal Complement]] 的关系

正交性说的是“两个对象互相垂直”。

[[Orthogonal Complement]] 更进一步：给定一个子空间 $S$，把所有与 $S$ 正交的向量收集成一个新空间 $S^\perp$。

所以：

- 正交是关系；
- 正交补是由这个关系生成出来的子空间。

## 常见误区

- 正交不是“不相关”的随口说法；在线性代数里必须落到内积为 0。
- 子空间正交要求任意向量都正交，不是只找一两个代表向量。
- 不要把 $C(A)$ 和 $N(A)$ 直接说成正交。真正配对的是 $C(A^T)$ 与 $N(A)$，以及 $C(A)$ 与 $N(A^T)$。

## 来自课程位置

- [[02_Least Squares, Determinants and Eigenvalues#Session 2.1 Orthogonal vectors and subspaces|Session 2.1]]：正交、正交补、四个基本子空间的正交关系。
- [[02_Least Squares, Determinants and Eigenvalues#Session 2.2 Projections onto subspaces|Session 2.2]]：投影由误差正交决定。
- [[03_Positive Definite Matrices and Applications#Session 3.1 Symmetric matrices and positive definiteness|Session 3.1]]：对称矩阵的特征向量可以选成正交基。

## 关联卡片

- [[Orthogonal Complement]]
- [[Orthogonal Projection]]
- [[Projection Matrix]]
- [[Least Squares]]
- [[Orthogonal Matrix]]
- [[Gram-Schmidt Orthogonalization]]

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

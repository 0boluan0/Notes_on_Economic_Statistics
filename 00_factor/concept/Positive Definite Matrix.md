---
aliases:
- Positive Definite Matrix
- positive definite
- PD matrix
- 正定矩阵
tags:
- concept
- 线性代数
---
# Positive Definite Matrix

## 先记一句话

正定矩阵就是：**二次型 $x^TAx$ 在每个非零方向上都严格为正的对称矩阵**。

它表达的是：

> 没有平坦方向，也没有向下方向。

所以它常常对应严格极小值、稳定的最小二乘和椭圆形等高线。

## 它是什么

在课程语境里，通常先假设 $A$ 是实对称矩阵。

若对所有非零向量 $x$，
$$
x^TAx>0,
$$
则 $A$ 是 positive definite。

如果只满足
$$
x^TAx\geq0,
$$
则是 positive semidefinite，不是 positive definite。

## 一个最小例子

矩阵
$$
A=
\begin{bmatrix}
2&0\\
0&3
\end{bmatrix}
$$
是正定的，因为
$$
x^TAx=2x_1^2+3x_2^2>0
$$
对任意非零 $x$ 都成立。

## 常用判据

对实对称矩阵，以下条件等价：

- $x^TAx>0$ 对所有非零 $x$ 成立；
- 所有特征值都大于 0；
- 无行交换、保持对称结构消元时，所有 pivot 都大于 0；
- 所有 leading principal minors 都大于 0。

具体执行流程见 [[Testing Positive Definiteness]]。

## 它在题里负责什么

- 判断二次型是否有严格极小值。
- 判断 Hessian 是否给出 local minimum。
- 解释 least squares 中 $A^TA$ 何时可逆。
- 连接 [[Symmetric Matrix]]、[[Spectral Decomposition]] 和优化。

## 常见误区

- 正定判据大多默认矩阵对称；不检查对称性就套判据容易错。
- 对角线元素为正不够。
- determinant 为正也不够；二维以上可能仍不是正定。
- semidefinite 允许某些非零方向取到 0，不能说 strict minimum。

## 来自课程位置

- [[03_Positive Definite Matrices and Applications#Session 3.1 Symmetric matrices and positive definiteness|Session 3.1]]：正定矩阵的等价判据。
- [[03_Positive Definite Matrices and Applications#Session 3.3 Positive definite matrices and minima|Session 3.3]]：正定矩阵与极小值。

## 关联卡片

- [[Symmetric Matrix]]
- [[Spectral Decomposition]]
- [[Testing Positive Definiteness]]
- [[Least Squares]]
- [[Orthogonal Matrix]]

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

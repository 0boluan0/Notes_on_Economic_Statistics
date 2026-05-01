---
aliases:
- Orthogonal Projection
- projection onto a subspace
- 正交投影
- 投影
tags:
- concept
- 线性代数
---
# Orthogonal Projection

## 先记一句话

正交投影就是：**在一个子空间里找离 $b$ 最近的点**。

它的判别条件不是“看起来最近”，而是：
$$
b-p\perp S.
$$

这里 $p$ 是投影点，$S$ 是目标子空间，$b-p$ 是误差。

## 它解决什么判断

遇到 “closest point”“best approximation”“least squares” 时，先想到投影。

投影把问题拆成两部分：
$$
b=p+e,
$$
其中
$$
p\in S,\qquad e\in S^\perp.
$$

也就是说：$b$ 被拆成“子空间能解释的部分”和“子空间解释不了的正交误差”。

## 一个最小例子

把
$$
b=\begin{bmatrix}3\\1\end{bmatrix}
$$
投到
$$
S=\operatorname{span}\left\{
a=\begin{bmatrix}1\\2\end{bmatrix}
\right\}
$$
上。

投影点一定长成 $p=\hat{x}a$。误差正交给出
$$
a^T(b-\hat{x}a)=0.
$$

所以
$$
\hat{x}=\frac{a^Tb}{a^Ta}=\frac{5}{5}=1,
$$
从而
$$
p=a=\begin{bmatrix}1\\2\end{bmatrix}.
$$

检查误差：
$$
e=b-p=\begin{bmatrix}2\\-1\end{bmatrix},
\qquad
a^Te=0.
$$

## 为什么它就是最近点

对任意 $s\in S$，
$$
b-s=(p-s)+e.
$$

因为 $p-s\in S$，而 $e\perp S$，所以
$$
\|b-s\|^2=\|p-s\|^2+\|e\|^2\geq \|e\|^2.
$$

等号只在 $s=p$ 时成立。于是 $p$ 就是离 $b$ 最近的点。

## 高维时怎么读

如果 $S=C(A)$，也就是由 $A$ 的列张成，那么投影点写成
$$
p=A\hat{x}.
$$

误差正交变成
$$
A^T(b-A\hat{x})=0.
$$

这一步就是 [[Least Squares]] 和 [[Least Squares via Normal Equations]] 的来源。

## 常见误区

- 正交的是误差 $b-p$，不是 $b$ 本身。
- 一维投影公式只有在投到一条线时才直接用；投到多维子空间时要写成正规方程。
- 投影点 $p$ 和坐标 $\hat{x}$ 不是同一个对象。$p$ 在原空间里，$\hat{x}$ 是用 $A$ 的列表示 $p$ 的系数。

## 来自课程位置

- [[02_Least Squares, Determinants and Eigenvalues#Session 2.2 Projections onto subspaces|Session 2.2]]：投影点由误差正交刻画。
- [[02_Least Squares, Determinants and Eigenvalues#Session 2.3 Projection matrices and least squares|Session 2.3]]：投影进入矩阵形式和最小二乘。

## 关联卡片

- [[Orthogonality]]
- [[Orthogonal Complement]]
- [[Projection Matrix]]
- [[Least Squares]]
- [[Least Squares via Normal Equations]]

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

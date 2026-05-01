---
aliases:
- Invertible Matrix Equivalence Chain
- Invertible Matrix Equivalences
- 可逆矩阵等价链
tags:
- framework
- 线性代数
---
# Invertible Matrix Equivalence Chain

## 这张卡回答什么问题

对方阵 $A$，很多说法其实都在讲同一件事：

> $A$ 没有丢方向，所以可以被反过来。

这张卡用来把“可逆、唯一解、满秩、零空间、列空间、主元、行列式、特征值”串成一条链。

## 先记直觉

可逆矩阵像一台没有压扁空间的机器：

- 不会把非零输入压成 0；
- 每个输出 $b$ 都能追溯到唯一输入 $x$；
- 没有方向丢失；
- 所以可以反向恢复输入。

一旦某个方向被压扁，就不可能可逆。

## 等价链

对 $n\times n$ 方阵 $A$，下面这些说法等价：

| 说法 | 直觉 |
| --- | --- |
| $A$ 可逆 | 可以反向恢复输入 |
| $Ax=b$ 对每个 $b$ 都有唯一解 | 每个目标都可达且只有一个来源 |
| $Ax=0$ 只有零解 | 没有非零方向被压成 0 |
| $N(A)=\{0\}$ | [[Null Space]] 没有自由方向 |
| $C(A)=\mathbb{R}^n$ | [[Column Space]] 铺满整个输出空间 |
| $\operatorname{rank}(A)=n$ | 有 $n$ 个独立方向 |
| 每一列都有 pivot | 没有自由变量 |
| 列向量线性无关且张成 $\mathbb{R}^n$ | 既不冗余，也不缺方向 |
| $\det(A)\neq 0$ | 空间体积没有被压成 0 |
| 0 不是 $A$ 的特征值 | 没有方向被映到 0 |

做题时，看到其中任意一个条件，都可以切换到整条链上的其他说法。

## 一个最小例子

$$
A=\begin{bmatrix}
1&2\\
2&4
\end{bmatrix}
$$

不可逆，因为第二列是第一列的 2 倍。

同一件事可以换成很多说法：

- rank 只有 1，不是 2；
- 列空间只是一条线，不是整个 $\mathbb{R}^2$；
- 零空间有非零向量 $\begin{bmatrix}-2\\1\end{bmatrix}$；
- 有些 $b$ 不可达，比如 $\begin{bmatrix}3\\5\end{bmatrix}$；
- 有些可达的 $b$ 会有无穷多解，比如 $\begin{bmatrix}3\\6\end{bmatrix}$。

这些不是五个独立事实，而是“方向丢失”这件事的五种表述。

## 什么时候调用这张卡

- 题目问 inverse 是否存在。
- 题目给了 rank、pivot、nullspace、column space、determinant、eigenvalue 中的一种信息，要你推出其他信息。
- 你想判断 $Ax=b$ 是对所有 $b$ 都唯一可解，还是只对某些 $b$ 可解。

## 最容易混的点

- “对某个 $b$ 有唯一解”不等于“对每个 $b$ 都有唯一解”。
- 这条完整等价链只适用于方阵。
- 非方阵不能说普通 inverse，只能改说满列秩、满行秩、left inverse、right inverse 或 pseudoinverse。
- $C(A)=\mathbb{R}^n$ 这句话只在 $n\times n$ 方阵语境下直接进入可逆链；一般 $m\times n$ 矩阵要先看所在空间。

## 来自课程位置

- [[01_Ax = b and the Four Subspaces#Session 1.3 Elimination with matrices|Session 1.3]]：pivot 和 rank 暴露独立方向。
- [[01_Ax = b and the Four Subspaces#Session 1.4 Multiplication and inverse matrices|Session 1.4]]：inverse 意味着每个 $b$ 唯一可解。
- [[01_Ax = b and the Four Subspaces#Session 1.7 Column space and nullspace|Session 1.7]]：column space / nullspace 给出存在性和唯一性解释。

## 关联卡片

- [[Matrix Inverse]]
- [[Matrix Rank]]
- [[Column Space]]
- [[Null Space]]
- [[Linear Independence]]
- [[Singular Matrix]]
- [[Determinant]]
- [[Eigenvalues]]

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

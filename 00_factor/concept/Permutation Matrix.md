---
aliases:
- 排列矩阵
- 置换矩阵
- Permutation Matrix
tags:
- 线性代数
- 矩阵
- 数学
- concept
---

# Permutation Matrix

## 它是什么

置换矩阵（permutation matrix）是把单位矩阵的行重新排列后得到的方阵；等价地，它的每一行和每一列都恰好有一个 $1$，其余元素都是 $0$。

若排列 $\pi$ 满足

$$
Pe_j=e_{\pi(j)},
$$

则 $P$ 只改变标准基向量的顺序，不改变它们的长度或夹角。

## 核心性质

- $P^{-1}=P^T$，所以置换矩阵是正交矩阵；
- $\det P=\operatorname{sgn}(\pi)\in\{1,-1\}$；
- $\|Px\|=\|x\|$；
- 左乘 $PA$ 重新排列 $A$ 的行，右乘 $AP$ 重新排列 $A$ 的列。

## 最小例子

交换第一、第二个坐标的置换矩阵是

$$
P=
\begin{bmatrix}
0&1&0\\
1&0&0\\
0&0&1
\end{bmatrix},
\qquad
P(x_1,x_2,x_3)^T=(x_2,x_1,x_3)^T.
$$

在带换行的消元中，$P$ 用来记录行交换，因此常见分解写作 $PA=LU$。怎样选择主元和执行消元属于方法卡 [[Gaussian Elimination|高斯消元]]，而不是本概念的定义。

## 关联卡片

- [[Elementary Matrix]]
- [[Orthogonal Matrix]]
- [[LU Decomposition]]
- [[Gaussian Elimination]]

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

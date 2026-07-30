---
aliases:
- 不可逆矩阵
- 退化矩阵
- 奇异矩阵
- Singular Matrix
tags:
- 线性代数
- 矩阵
- 数学
- concept
---

# Singular Matrix

## 它是什么

奇异矩阵（singular matrix）是不可逆的方阵。对 $A\in\mathbb F^{n\times n}$，下列陈述彼此等价：

$$
\begin{aligned}
A\text{ 奇异}
&\iff A^{-1}\text{ 不存在}\\
&\iff \det A=0\\
&\iff \operatorname{rank}(A)<n\\
&\iff N(A)\ne\{0\}\\
&\iff 0\text{ 是 }A\text{ 的特征值}.
\end{aligned}
$$

因此它的行向量与列向量都线性相关。

## 几何直觉

线性映射 $x\mapsto Ax$ 把至少一个非零方向压到零，所以像空间的维数小于 $n$：二维空间可能被压成直线或原点，三维空间可能被压成平面、直线或原点。

这也解释了为什么 $Ax=b$ 不可能对每个 $b$ 都有唯一解：若 $b\notin C(A)$，则无解；若 $b\in C(A)$，任一特解都可以加上非零零空间向量，因此解不唯一。

## 最小例子

$$
A=
\begin{bmatrix}
1&2\\
2&4
\end{bmatrix}
$$

的第二列是第一列的两倍，所以 $\operatorname{rank}(A)=1<2$、$\det A=0$，并且 $(-2,1)^T\in N(A)$；故 $A$ 奇异。

## 概念边界

“奇异”是精确的代数性质；“接近奇异”则是数值稳定性问题，应通过 [[Condition Number|条件数]] 与 [[Singular Value|奇异值]] 描述。秩亏系统的求解选择见 [[Pseudoinverse|伪逆]] 与 [[Choosing Matrix Decompositions|矩阵分解选择框架]]。

## 关联卡片

- [[Matrix Rank]]
- [[Null Space]]
- [[Matrix Inverse]]
- [[Determinant]]

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

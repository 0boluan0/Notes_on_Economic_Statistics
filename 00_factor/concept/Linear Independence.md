---
aliases:
- Linear Independence
- linearly independent
- 线性无关
tags:
  - 线性代数
  - concept
---
# Linear Independence

## 先记一句话

线性无关就是：**每个向量都真的带来了新方向，没有谁是多余的**。

如果一组向量里有某个向量能由其他向量拼出来，那么它就是冗余的，这组向量线性相关。

## 怎么用 0 来判断

给定向量 $v_1,\dots,v_k$。

如果
$$
c_1v_1+\cdots+c_kv_k=0
$$
只有一种办法成立：
$$
c_1=\cdots=c_k=0,
$$
那么这组向量线性无关。

如果存在一组不全为 0 的系数也能拼出 0，那么它们线性相关。

直觉是：非零系数能拼出 0，就说明这些向量之间发生了抵消，至少有一个方向是冗余的。

## 一个最小例子

在 $\mathbb{R}^2$ 里，
$$
v_1=\begin{bmatrix}1\\2\end{bmatrix},
\qquad
v_2=\begin{bmatrix}2\\4\end{bmatrix}.
$$

因为
$$
v_2=2v_1,
$$
所以这两个向量线性相关。

你也可以写成
$$
-2v_1+v_2=0.
$$

这里系数 $-2,1$ 不全为 0，却能拼出 0，所以不是线性无关。

## 它在矩阵里负责什么

矩阵的列向量线性无关，意思是：

- 没有冗余列；
- [[Null Space]] 只有零向量；
- 每一列都是 pivot column；
- $\operatorname{rank}(A)$ 等于列数。

如果列向量线性相关，就会出现自由变量和非零零空间方向。

## 和 [[Basis]] 的关系

[[Basis]] 要同时满足两件事：

1. 能张成整个空间；
2. 内部线性无关。

线性无关只管“没有冗余”，还不保证“够不够覆盖整个空间”。

## 常见误区

- “向量个数多”不代表张成空间，也可能只是冗余很多。
- “能张成”不代表线性无关；一个生成集可以很冗余。
- 在 $\mathbb{R}^n$ 里，超过 $n$ 个向量一定线性相关。
- $n$ 个线性无关向量在 $\mathbb{R}^n$ 中会自动构成一组 basis。

## 来自课程位置

- [[01_Ax = b and the Four Subspaces#Session 1.10 Independence, basis, and dimension|Session 1.10]]：independence、basis、dimension 的统一入口。
- [[01_Ax = b and the Four Subspaces#Session 1.3 Elimination with matrices|Session 1.3]]：pivot columns 暴露独立方向。

## 关联卡片

- [[Basis]]
- [[Dimension]]
- [[Matrix Rank]]
- [[Null Space]]
- [[Column Space]]
- [[Invertible Matrix Equivalence Chain]]


## 最小例子

把 **Linear Independence** 放在最简单的可计算情形中：先给定定义所需的最小输入，再按定义计算输出；若关键关系不成立，就不能把该对象归入本概念。这个检查也能帮助区分相近概念。
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

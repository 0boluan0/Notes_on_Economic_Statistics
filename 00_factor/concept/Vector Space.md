---
aliases:
- Vector Space
- 线性空间
- 向量空间
tags:
- 线性代数
- concept
---
# Vector Space

## 先记一句话

向量空间就是：**一类对象的集合，你可以在里面做加法和数乘，结果还留在这个集合里**。

它不一定只装普通箭头。向量空间里的“向量”可以是：

- $\mathbb{R}^n$ 里的数列；
- 多项式；
- 函数；
- 矩阵；
- 线性方程的解。

只要加法和数乘规则稳定，它就可以是 vector space。

## 为什么线代要讲“空间”

线代不是只研究一个向量，而是研究一整类向量能不能被组合、压缩、分解。

比如：

- [[Column Space]] 是所有可能输出；
- [[Null Space]] 是所有会被压成 0 的输入；
- [[Row Space]] 是所有行向量能组合出来的方向。

这些都要先有“空间”的语言才能说清。

## 判断一个集合是不是向量空间

最核心检查是：

1. 有零向量；
2. 两个元素相加后还在集合里；
3. 一个元素乘任意标量后还在集合里。

更短地说：

> 对线性组合封闭。

也就是如果 $u,v$ 在集合里，那么
$$
cu+dv
$$
也必须还在集合里。

## 一个最小例子

集合
$$
S=\{(x,y,z)\in\mathbb{R}^3:x+y+z=0\}
$$
是一个向量空间。

因为两个满足 $x+y+z=0$ 的向量相加，仍然满足这个条件；乘一个数后，也仍然满足这个条件。

但
$$
T=\{(x,y,z)\in\mathbb{R}^3:x+y+z=1\}
$$
不是向量空间，因为它不包含零向量，也不对数乘封闭。

## 和 [[Subspace]] 的关系

如果一个集合本来就待在某个更大的 vector space 里面，而且自己也对线性组合封闭，它就是 subspace。

所以在 Unit I 里，你大多数时候真正要判断的是：

> 这个集合是不是 $\mathbb{R}^n$ 的子空间？

## 必要公式

向量空间最重要的封闭性可以压成一句：
$$
u,v\in V,\ c,d\in\mathbb{R}
\quad\Longrightarrow\quad
cu+dv\in V.
$$

## 常见误区

- “元素长得像向量”不够，关键是加法和数乘是否封闭。
- 非齐次方程 $Ax=b$ 的解集通常不是 vector space，因为它不经过 0。
- 齐次方程 $Ax=0$ 的解集是 vector space，也就是 [[Null Space]]。

## 来自课程位置

- [[01_Ax = b and the Four Subspaces#Session 1.6 Transposes, permutations, vector spaces|Session 1.6]]：从集合升级为空间，重点是 closure。
- [[01_Ax = b and the Four Subspaces#Session 1.7 Column space and nullspace|Session 1.7]]：column space 和 nullspace 是最先出现的矩阵相关子空间。

## 关联卡片

- [[Subspace]]
- [[Column Space]]
- [[Null Space]]
- [[Row Space]]
- [[Left Nullspace]]
- [[Linear Independence]]
- [[Basis]]
- [[Dimension]]

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

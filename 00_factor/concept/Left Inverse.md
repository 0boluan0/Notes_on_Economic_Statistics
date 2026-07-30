---
aliases:
- Left Inverse
- left inverse
- 左逆
tags:
  - concept
  - 线性代数
---
# Left Inverse

## 先记一句话

左逆就是：**从左边乘上去能把 $A$ 抵消掉的矩阵**。

如果存在 $B$ 使
$$
BA=I,
$$
那么 $B$ 是 $A$ 的 left inverse。

## 它是什么

左逆对应的是“输入没有被混淆”。

若 $A$ 是 $m\times n$ 矩阵，并且列满秩：
$$
\operatorname{rank}(A)=n,
$$
则 $A$ 可能有左逆。

典型形式是
$$
(A^TA)^{-1}A^T.
$$

## 一个最小图像

高而瘦的矩阵如果列独立，就没有把两个不同输入压到同一个输出。

所以可以从输出中恢复输入：
$$
BAx=x.
$$

这就是 $BA=I$ 的含义。

## 它在题里负责什么

- 判断 tall matrix 是否能恢复输入。
- 连接 full column rank 与 least squares。
- 解释 [[Pseudoinverse]] 在满列秩时为什么是 $(A^TA)^{-1}A^T$。

## 常见误区

- 左逆不要求 $AB=I$。
- 非方阵的左逆一般不唯一。
- 有左逆意味着列独立，不意味着每个输出 $b$ 都可达。

## 来自课程位置

- [[03_Positive Definite Matrices and Applications#Session 3.8 Left and right inverses and pseudoinverse|Session 3.8]]：left inverse、right inverse 与 pseudoinverse。

## 关联卡片

- [[Right Inverse]]
- [[Pseudoinverse]]
- [[Matrix Inverse]]
- [[Matrix Rank]]
- [[Least Squares]]


## 最小例子

把 **Left Inverse** 放在最简单的可计算情形中：先给定定义所需的最小输入，再按定义计算输出；若关键关系不成立，就不能把该对象归入本概念。这个检查也能帮助区分相近概念。
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

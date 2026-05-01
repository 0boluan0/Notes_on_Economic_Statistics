---
aliases:
- Right Inverse
- right inverse
- 右逆
tags:
- concept
- 线性代数
---
# Right Inverse

## 先记一句话

右逆就是：**从右边乘上去能让 $A$ 覆盖整个输出空间的矩阵**。

如果存在 $B$ 使
$$
AB=I,
$$
那么 $B$ 是 $A$ 的 right inverse。

## 它是什么

右逆对应的是“每个输出都能被打出来”。

若 $A$ 是 $m\times n$ 矩阵，并且行满秩：
$$
\operatorname{rank}(A)=m,
$$
则 $A$ 可能有右逆。

典型形式是
$$
A^T(AA^T)^{-1}.
$$

## 一个最小图像

矮而宽的矩阵如果行独立，输出空间里的每个 $b$ 都可达。

所以可以为每个目标 $b$ 找到一个输入：
$$
A(Bb)=b.
$$

这就是 $AB=I$ 的含义。

## 它在题里负责什么

- 判断 wide matrix 是否能覆盖整个输出空间。
- 连接 full row rank 与可达性。
- 解释 [[Pseudoinverse]] 在满行秩时为什么是 $A^T(AA^T)^{-1}$。

## 常见误区

- 右逆不要求 $BA=I$。
- 有右逆意味着每个 $b$ 可达，不意味着解唯一。
- 非方阵的右逆一般不唯一。

## 来自课程位置

- [[03_Positive Definite Matrices and Applications#Session 3.8 Left and right inverses; pseudoinverse|Session 3.8]]：left inverse、right inverse 与 pseudoinverse。

## 关联卡片

- [[Left Inverse]]
- [[Pseudoinverse]]
- [[Matrix Inverse]]
- [[Column Space]]
- [[Matrix Rank]]

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

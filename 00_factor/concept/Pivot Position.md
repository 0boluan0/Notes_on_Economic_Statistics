---
aliases:
  - Pivot Position
  - Pivot
  - pivot position
  - 主元位置
  - 主元
tags:
  - 线性代数
  - concept
---

# Pivot Position

## 它是什么

主元位置是矩阵化为行阶梯形后，每个非零行最左侧领先元素所在的位置。对应的列称为主元列，对应的未知量称为主元变量。

若 $A$ 的阶梯形中有 $r$ 个主元，则

$$
r=\operatorname{rank}(A).
$$

主元位置由 $A$ 决定；不同合法行消元路径可能产生不同数值中间步骤，但最终主元列编号相同。

## 最小例子

$$
R=\begin{bmatrix}1&2&0&3\\0&0&1&-1\\0&0&0&0\end{bmatrix}
$$

的主元位于第 1、3 列，因此 $x_1,x_3$ 是主元变量，$x_2,x_4$ 是自由变量，秩为 $2$。

## 边界

- 主元不必等于 $1$；只有简化行阶梯形要求主元为 $1$。
- 找 $C(A)$ 的基时，应取原矩阵的主元列，而不是阶梯形矩阵的对应列。

## 关联卡片

- [[Gaussian Elimination]]
- [[Reduced Row Echelon Form]]
- [[Matrix Rank]]
- [[Rank-Nullity Theorem]]

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

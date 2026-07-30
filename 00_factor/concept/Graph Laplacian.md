---
aliases:
  - Graph Laplacian
  - Laplacian matrix
  - 图拉普拉斯矩阵
  - 图拉普拉斯
tags:
  - 线性代数
  - graph-theory
  - concept
---

# Graph Laplacian

## 它是什么

给无向图选定任意边方向，并令关联矩阵 $B$ 的每行对应一条边、每列对应一个节点，则图拉普拉斯矩阵为

$$
L=B^TB=D-W,
$$

其中 $D$ 是度数对角阵，$W$ 是邻接矩阵。若采用转置的关联矩阵约定，则写成 $L=BB^T$；关键是 $L$ 必须作用在节点空间。

## 核心关系

$$
x^TLx=\|Bx\|^2=\sum_{(i,j)\in E}(x_i-x_j)^2\ge0.
$$

因此 $L$ 对称半正定，且 $L\mathbf1=0$。零特征值的重数等于图的连通分量个数。

## 最小例子

两个节点由一条边相连时，

$$
L=\begin{bmatrix}1&-1\\-1&1\end{bmatrix}.
$$

## 关联卡片

- [[Incidence Matrix]]
- [[Positive Semidefinite Matrix]]
- [[Null Space]]
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

---
aliases:
  - Incidence Matrix
  - oriented incidence matrix
  - 关联矩阵
  - 发生矩阵
tags:
  - 线性代数
  - graph-theory
  - concept
---

# Incidence Matrix

## 它是什么

关联矩阵用 $-1,0,1$ 编码图的节点—边关系。MIT 18.06SC 采用“边乘节点”约定：若有 $m$ 条边、$n$ 个节点，则

$$
A\in\mathbb R^{m\times n},
$$

每行对应一条人为定向的边；边从节点 $i$ 指向节点 $j$ 时，该行第 $i$ 项为 $-1$、第 $j$ 项为 $+1$。

有些教材使用转置约定（节点乘边）。两种约定等价，但同一道推导中不能混用。

## 最小例子

边 $1\to2$ 对应

$$
A=\begin{bmatrix}-1&1\end{bmatrix}.
$$

节点势 $x=(x_1,x_2)^T$ 被映为边势差 $Ax=x_2-x_1$。

## 核心关系

- $A\mathbf1=0$：整体平移所有节点势不改变边差；
- $N(A)$ 记录每个连通分量上的常势；
- $N(A^T)$ 记录环流；
- 对连通图，$\operatorname{rank}(A)=n-1$；
- $A^TA$ 是作用在节点空间的图 Laplacian。

反转一条边只把对应行乘 $-1$，因此不改变秩、零空间或 $A^TA$。

## 关联卡片

- [[Graph Laplacian]]
- [[Column Space]]
- [[Null Space]]
- [[Left Nullspace]]

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

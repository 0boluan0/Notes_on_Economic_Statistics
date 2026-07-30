---
aliases:
  - Row Equivalence
  - row-equivalent matrices
  - 行等价
tags:
  - 线性代数
  - concept
---

# Row Equivalence

## 它是什么

若矩阵 $B$ 可由矩阵 $A$ 经过有限次初等行变换得到，则称 $A$ 与 $B$ 行等价。等价地，存在可逆矩阵 $E$ 使

$$
B=EA.
$$

行等价保持行空间、秩和齐次方程的零空间；对增广矩阵执行相同行变换，还保持线性方程组的解集。

## 不保持什么

行等价通常不保持列空间中的具体向量：$C(EA)=E(C(A))$，它与 $C(A)$ 维数相同，却一般不是同一个子空间。

## 最小例子

$$
\begin{bmatrix}1&2\\2&4\end{bmatrix}
\sim
\begin{bmatrix}1&2\\0&0\end{bmatrix}.
$$

两者秩均为 $1$、零空间相同，但列空间分别是 $\operatorname{span}(1,2)^T$ 与 $\operatorname{span}(1,0)^T$。

## 关联卡片

- [[Elementary Matrix]]
- [[Gaussian Elimination]]
- [[Row Space]]
- [[Null Space]]

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

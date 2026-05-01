---
aliases:
- Similar Matrix
- Similar Matrices
- similarity
- 相似矩阵
tags:
- concept
- 线性代数
---
# Similar Matrix

## 先记一句话

相似矩阵就是：**同一个线性变换在不同基底下的两个矩阵表示**。

如果
$$
B=M^{-1}AM,
$$
则 $A$ 与 $B$ 相似。

## 它是什么

矩阵本身依赖坐标系。

换一组基以后，同一个线性变换会有不同的矩阵表示。相似关系记录的就是这种“表示变了，本体没变”。

所以相似矩阵会保留线性变换的核心结构。

## 一个最小例子

[[Diagonalization]] 就是相似变换的特殊目标：
$$
\Lambda=S^{-1}AS.
$$

这里 $S$ 的列是特征向量。换到特征向量基底后，$A$ 的表示变成对角矩阵 $\Lambda$。

## 它在题里负责什么

- 解释为什么换基后矩阵形式会改变。
- 判断哪些量不随基改变：特征值、trace、determinant。
- 连接 diagonalization 与 [[Jordan Form]]。

## 常见误区

- 相似不是“长得像”，而是必须存在可逆矩阵 $M$ 使 $B=M^{-1}AM$。
- 相似矩阵元素可以完全不同，但特征值相同。
- 相似关系只用于方阵，因为它描述同一空间到自身的线性变换。

## 来自课程位置

- [[03_Positive Definite Matrices and Applications#Session 3.4 Similar matrices and Jordan form|Session 3.4]]：similarity、diagonalization 与 Jordan form。
- [[03_Positive Definite Matrices and Applications#Session 3.7 Change of basis; image compression|Session 3.7]]：换基如何改变矩阵表示。

## 关联卡片

- [[Change of Basis]]
- [[Linear Transformation]]
- [[Diagonalization]]
- [[Jordan Form]]

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

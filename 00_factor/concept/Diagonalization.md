---
aliases:
- Diagonalization
- 对角化
tags:
- concept
- 线性代数
---
# Diagonalization

## 它是什么
- 「Diagonalization」是指把矩阵表示成相似于对角矩阵的形式。

## 最小可检索信息
- 定义：若 A 有 n 个线性无关特征向量，则可写成 $A=S\Lambda S^{-1}$。
- 符号/公式：$S^{-1}AS=\Lambda$。
- 最小例子：具有互异特征值的矩阵一定可对角化。

## 关键性质
- 对角化把矩阵幂、矩阵指数、差分方程和微分方程计算大幅简化。
- 失败的原因通常是特征向量不够多，而不是算术不熟。

## 关联卡片
- [[Eigenvalues]]
- [[Eigenvectors]]
- [[Similar Matrix]]

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM ""
WHERE (
  contains(file.path, "01_Math/") OR
  contains(file.path, "02_Economy/") OR
  contains(file.outlinks, this.file.link)
)
SORT file.mtime DESC
```

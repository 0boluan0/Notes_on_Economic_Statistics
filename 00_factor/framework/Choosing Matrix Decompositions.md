---
aliases:
- Choosing Matrix Decompositions
- How to Choose Matrix Decompositions
- 如何选择矩阵分解
tags:
- framework
- 线性代数
---
# Choosing Matrix Decompositions

## 核心问题
- 这张卡只回答：什么时候应该想到 diagonalization、Jordan form，什么时候应该直接转去 SVD。

## 一句话框架
- 先问矩阵是不是方阵；再问它有没有足够多特征向量；再问你需要的是“动力系统解释”还是“任意矩阵的稳定结构”。

## 先用 diagonalization 的场景
- 矩阵是方阵。
- 你关心 `A^k`、$e^{At}$、长期行为、稳态、反复作用。
- 矩阵有足够多线性无关特征向量。

## 必须看 Jordan form 的场景
- 矩阵是方阵。
- 你知道或怀疑它不可对角化。
- 题目在问“缺了多少特征向量”“为什么不能 diagonalize”“标准形长什么样”。

## 直接转去 SVD 的场景
- 矩阵不一定是方阵。
- 你关心 rank、四个基本子空间、compression、best low-rank approximation、pseudoinverse。
- 题目不是在问“长期迭代”，而是在问“任意矩阵的几何结构”。

## 快速判断规则
- powers of A / matrix exponential -> 先想 diagonalization。
- repeated eigenvalue but vectors not够 -> 想 Jordan。
- rectangular / rank-deficient / best approximation -> 想 SVD。

## 为什么三者不要混
- diagonalization 是“找到不变方向后完全解耦”。
- Jordan 是“解耦失败后最接近对角化的标准形”。
- SVD 不依赖特征向量，而是寻找输入方向与输出方向的最佳正交坐标系。

## 常见误判
- 只因为题目出现特征值，就默认一定能 diagonalize。
- 对非方阵还想硬套 diagonalization。
- 把 SVD 当成“特征分解的改名版”。

## 关联卡片
- [[Diagonalization]]
- [[Jordan Form]]
- [[Singular Value Decomposition]]
- [[Pseudoinverse]]
- [[Matrix Exponential]]

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

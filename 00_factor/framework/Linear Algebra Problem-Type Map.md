---
aliases:
- Linear Algebra Problem-Type Map
- 线性代数题型地图
tags:
- framework
- 线性代数
---
# Linear Algebra Problem-Type Map

## 核心问题
- 这张卡只回答：看到一道线代题时，第一反应应该走哪条主线。

## 题面信号 -> 首选路线

### 如果题目在问“有没有解 / 为什么不唯一 / 从 rref 读结构”
- 先走 `Ax=b` 与四个基本子空间路线。
- 关键词：pivot、rank、free variables、column space、nullspace、basis。

### 如果题目在问“closest / best fit / minimum error / orthogonal”
- 先走 projection / least squares 路线。
- 关键词：orthogonal error、normal equations、projection matrix、QR。

### 如果题目在问“可逆 / 体积 / 行操作对某个量的影响”
- 先走 determinant / inverse 路线。
- 关键词：determinant、cofactor、adjugate、volume、singular。

### 如果题目在问“长期行为 / powers of A / differential equations / steady state”
- 先走 eigen / diagonalization / matrix exponential 路线。
- 关键词：eigenvalue、eigenvector、diagonalization、Markov、$e^{At}$。

### 如果题目在问“正定 / 二次型 / 极小值 / Hessian”
- 先走 symmetric / positive definite 路线。
- 关键词：quadratic form、positive definite、minimum、symmetric。

### 如果题目在问“任意矩阵的标准形 / compression / best low-rank / pseudoinverse”
- 先走 SVD 路线。
- 关键词：singular value、low-rank approximation、pseudoinverse、rectangular matrix。

## 为什么这张地图有用
- 它逼你先做“题型识别”，再做公式调用。
- 线代最容易卡住的地方不是不会算，而是走错路线。

## 最容易混淆的岔路
- `Ax=b` 无解时，不要继续硬求 inverse，转 least squares。
- 有 repeated eigenvalue 时，不要默认一定能 diagonalize，先检查特征向量够不够。
- 非方阵或秩亏矩阵，不要把普通 inverse 当默认工具，转 SVD / pseudoinverse。

## 关联卡片
- [[Reading the Four Fundamental Subspaces from RREF]]
- [[Least Squares]]
- [[Determinant]]
- [[Eigenvalues]]
- [[Positive Definite Matrix]]
- [[Choosing Matrix Decompositions]]

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

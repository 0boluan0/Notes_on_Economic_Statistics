---
aliases:
- 线性代数
- 线性代数-hub
- Linear Algebra
- Linear Algebra-hub
tags:
- hub
- 数学
---
# 线性代数 Hub

## 核心概念

[[Vector Space|向量空间]]
[[Matrix Rank|矩阵的秩]]
[[Orthogonality|正交性]]
[[Linear Transformation|线性变换]]

## 向量空间

[[Vector Space|向量空间]]
[[Subspace|子空间]]
[[Linear Independence|线性无关]]
[[Row Space|行空间]]
[[Left Nullspace|左零空间]]

## 矩阵运算与分解

### 矩阵的秩
[[Matrix Rank|矩阵的秩]]
[[Matrix rank properties|矩阵秩的性质]]
[[Null Space|零空间]]
[[Column Space|列空间]]
[[Row Space|行空间]]

### 矩阵分解
[[LU Decomposition|LU分解]]
[[Permutation Matrix|置换矩阵]]
[[Spectral Decomposition|谱分解]]
[[Diagonalization|对角化]]
[[Singular Value Decomposition|奇异值分解]]

### 特殊矩阵
[[Symmetric Matrix|对称矩阵]]
[[Singular Matrix|奇异矩阵]]
[[Wishart Distribution|Wishart分布]]
[[Orthogonal Matrix|正交矩阵]]
[[Projection Matrix|投影矩阵]]
[[Markov Matrix|马尔可夫矩阵]]

## 线性方程组

[[Linear system solution structure|线性方程组解的结构]]
[[Determinant|行列式]]
[[Matrix Inverse|逆矩阵]]
[[Least Squares|最小二乘]]
[[Pseudoinverse|伪逆]]
[[Cramer's Rule|克拉默法则]]

## 多变量统计

### 随机向量
[[Linear Combination of Two Random Variables|两个随机变量线性组合]]
[[Multivariate Normal Distribution|多元正态分布]]

### 距离度量
[[Mahalanobis Distance|马哈拉诺比斯距离]]

### 因子分析
[[Factor Analysis|因子分析]]
[[Single-Factor Model|单因子模型]]
[[Multi-Factor Model|多因子模型]]
[[PCA|主成分分析]]

## 课程笔记

- [[00_MIT OCW 18.06SC course map|MIT OCW 18.06SC course map]]
- [[01_Ax = b and the Four Subspaces|Ax = b and the Four Subspaces]]
- [[02_Least Squares, Determinants and Eigenvalues|Least Squares, Determinants and Eigenvalues]]
- [[03_Positive Definite Matrices and Applications|Positive Definite Matrices and Applications]]
- [[04_Review and exam roadmap|Review and exam roadmap]]

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

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

## 先从哪条主线进

### 主线 1：`Ax=b`、消元与四个基本子空间

- 课程入口：[[01_Ax = b and the Four Subspaces]]
- 先读这些卡：
  - [[Linear system solution structure]]
  - [[Column Space]]
  - [[Null Space]]
  - [[Row Space]]
  - [[Left Nullspace]]
  - [[Matrix Rank]]
  - [[Reading the Four Fundamental Subspaces from RREF]]

### 主线 2：正交、投影、最佳逼近

- 课程入口：[[02_Least Squares, Determinants and Eigenvalues#Session 2.1 Orthogonal vectors and subspaces|Unit II 正交主线]]
- 先读这些卡：
  - [[Orthogonality]]
  - [[Orthogonal Complement]]
  - [[Orthogonal Projection]]
  - [[Projection Matrix]]
  - [[Least Squares]]
  - [[Least Squares via Normal Equations]]
  - [[Gram-Schmidt Orthogonalization]]
  - [[QR Decomposition]]
  - [[Orthogonal Matrix]]

### 主线 3：特征结构、标准形与矩阵分解

- 课程入口：[[02_Least Squares, Determinants and Eigenvalues#Session 2.8 Eigenvalues and eigenvectors|Unit II eigen 主线]]、[[03_Positive Definite Matrices and Applications]]
- 先读这些卡：
  - [[Determinant]]
  - [[Eigenvalues]]
  - [[Eigenvectors]]
  - [[Diagonalization]]
  - [[Positive Definite Matrix]]
  - [[Testing Positive Definiteness]]
  - [[Jordan Form]]
  - [[Singular Value Decomposition]]
  - [[Pseudoinverse]]
  - [[Choosing Matrix Decompositions]]

## 按题型找入口

### 题型：判断 `Ax=b` 有没有解、解是否唯一

- [[01_Ax = b and the Four Subspaces]]
- [[Linear system solution structure]]
- [[Matrix Rank]]

### 题型：从 rref 读 basis、dimension、four fundamental subspaces

- [[01_Ax = b and the Four Subspaces#Session 1.11 The four fundamental subspaces|Unit I 核心 session]]
- [[Reading the Four Fundamental Subspaces from RREF]]
- [[Column Space]]
- [[Null Space]]

### 题型：做投影、least squares、QR

- [[02_Least Squares, Determinants and Eigenvalues#Session 2.2 Projections onto subspaces|projection]]
- [[02_Least Squares, Determinants and Eigenvalues#Session 2.3 Projection matrices and least squares|least squares]]
- [[02_Least Squares, Determinants and Eigenvalues#Session 2.4 Orthogonal matrices and Gram-Schmidt|QR]]
- [[Least Squares via Normal Equations]]

### 题型：求 determinant、cofactor、inverse、volume

- [[02_Least Squares, Determinants and Eigenvalues#Session 2.5 Properties of determinants|determinant 主线]]
- [[Determinant]]
- [[Cramer's Rule]]
- [[Matrix Inverse]]

### 题型：分析矩阵幂、稳态、动力系统

- [[02_Least Squares, Determinants and Eigenvalues#Session 2.8 Eigenvalues and eigenvectors|eigen]]
- [[02_Least Squares, Determinants and Eigenvalues#Session 2.9 Diagonalization and powers of A|diagonalization]]
- [[02_Least Squares, Determinants and Eigenvalues#Session 2.10 Differential equations and $e^{At}$|matrix exponential]]
- [[Markov Matrix]]

### 题型：判断正定、说明为什么该用 Jordan / SVD / pseudoinverse

- [[03_Positive Definite Matrices and Applications#Session 3.1 Symmetric matrices and positive definiteness|positive definite]]
- [[03_Positive Definite Matrices and Applications#Session 3.4 Similar matrices and Jordan form|Jordan]]
- [[03_Positive Definite Matrices and Applications#Session 3.5 Singular value decomposition|SVD]]
- [[03_Positive Definite Matrices and Applications#Session 3.8 Left and right inverses; pseudoinverse|pseudoinverse]]
- [[Testing Positive Definiteness]]

## 课程导航

- 总览：[[00_MIT OCW 18.06SC course map]]
- Unit I：[[01_Ax = b and the Four Subspaces]]
- Unit II：[[02_Least Squares, Determinants and Eigenvalues]]
- Unit III：[[03_Positive Definite Matrices and Applications]]
- 考前入口：[[04_Review and exam roadmap]]

## 核心概念

### 子空间与解结构
- [[Vector Space]]
- [[Subspace]]
- [[Linear Independence]]
- [[Basis]]
- [[Dimension]]
- [[Matrix Rank]]
- [[Reduced Row Echelon Form]]
- [[Column Space]]
- [[Row Space]]
- [[Null Space]]
- [[Left Nullspace]]

### 正交与逼近
- [[Orthogonality]]
- [[Orthogonal Complement]]
- [[Orthogonal Projection]]
- [[Projection Matrix]]
- [[Least Squares]]
- [[Orthogonal Matrix]]
- [[QR Decomposition]]

### determinant / eigen / 分解
- [[Determinant]]
- [[Matrix Inverse]]
- [[Eigenvalues]]
- [[Eigenvectors]]
- [[Diagonalization]]
- [[Matrix Exponential]]
- [[Symmetric Matrix]]
- [[Positive Definite Matrix]]
- [[Spectral Decomposition]]
- [[Jordan Form]]
- [[Singular Value Decomposition]]
- [[Pseudoinverse]]

### 变换与表示
- [[Linear Transformation]]
- [[Change of Basis]]
- [[Permutation Matrix]]
- [[Incidence Matrix]]

## 方法卡

- [[Linear system solution structure]]
- [[Reading the Four Fundamental Subspaces from RREF]]
- [[Gram-Schmidt Orthogonalization]]
- [[Least Squares via Normal Equations]]
- [[Testing Positive Definiteness]]
- [[Choosing Matrix Decompositions]]
- [[Linear Algebra Problem-Type Map]]

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

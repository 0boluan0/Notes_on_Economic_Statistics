---
aliases:
- 线性子空间
- 子空间
- Subspace
tags:
- 线性代数
- 线性空间
- concept
---
子空间（Subspace）是向量空间V的一个子集U，其本身也是一个向量空间，即对V中的加法和数乘运算封闭。

## 定义

U是向量空间V的子空间，当且仅当U满足以下三个条件：

1. **非空性**：U ≠ ∅（等价于0∈U）
2. **加法封闭**：对任意u, v∈U，有u + v∈U
3. **数乘封闭**：对任意u∈U和标量α，有αu∈U

## R²的子空间

在二维平面中，子空间包括：

1. **原点**：{0}
2. **过原点的直线**：所有形如t·v（t∈R）的向量
3. **整个平面**：R²

注意：不经过原点的直线不是子空间。

## R³的子空间

在三维空间中，子空间包括：

1. **原点**：{0}
2. **过原点的直线**：一维子空间
3. **过原点的平面**：二维子空间
4. **整个空间**：R³

## 矩阵相关的子空间

### 列空间（Column Space）C(A)

矩阵A的所有列向量的线性组合构成的子空间。

对于m×n矩阵A，列空间是Rᵐ的子空间。

方程Ax = b有解的条件是b∈C(A)。

### 行空间（Row Space）

矩阵A的所有行向量的线性组合构成的子空间。

### 零空间（Null Space）N(A)

所有满足Ax = 0的向量x构成的子空间。

对于m×n矩阵A，零空间是Rⁿ的子空间。

## 子空间的运算

### 交

两个子空间U₁和U₂的交U₁ ∩ U₂仍然是子空间。

### 和

U₁ + U₂ = {u₁ + u₂ : u₁∈U₁, u₂∈U₂}也是子空间。

### 直和

如果每个向量在U₁ + U₂中的表示唯一，则称为直和，记作U₁ ⊕ U₂。

## 维数定理

如果U₁和U₂是有限维子空间，则：
dim(U₁ ∩ U₂) + dim(U₁ + U₂) = dim(U₁) + dim(U₂)

相关链接: [[Vector Space|向量空间]], [[Null Space|零空间]], [[Column Space|列空间]]

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

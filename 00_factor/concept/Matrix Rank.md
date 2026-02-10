---
aliases:
- 秩
- 矩阵的秩
- Matrix Rank
tags:
- 线性代数
- 矩阵
- 数学
- concept
---
矩阵的秩（Rank）是矩阵中线性无关的行向量或列向量的最大个数，反映了矩阵包含的独立信息的数量。

## 定义

### 列秩（Column Rank）

矩阵列空间（所有列向量的线性组合构成的子空间）的维数。

### 行秩（Row Rank）

矩阵行空间（所有行向量的线性组合构成的子空间）的维数。

### 基本定理

**列秩 = 行秩**，统称为矩阵的秩，记作rank(A)或r(A)。

## 计算方法

### 高斯消元法

1. 将矩阵通过行变换化为行阶梯形式（REF）
2. 统计主元（非零行第一个非零元素）的个数
3. 主元的个数就是矩阵的秩

### 其他方法

- 计算非零子式的最高阶数
- 使用奇异值分解
- 计算线性无关列向量的个数

## 秩的性质

### 秩的范围

对于m×n矩阵A：
0 ≤ rank(A) ≤ min(m, n)

### 矩阵运算的秩

1. **转置**：rank(A) = rank(Aᵀ)
2. **乘积**：rank(AB) ≤ min(rank(A), rank(B))
3. **和**：rank(A + B) ≤ rank(A) + rank(B)

### 满秩矩阵

- **列满秩**：rank(A) = n（n为列数）
- **行满秩**：rank(A) = m（m为行数）
- **满秩**：rank(A) = min(m, n)

### 可逆矩阵

方阵A可逆当且仅当rank(A) = n（n为阶数）。

## 秩与零空间

### 秩-零化度定理

对于m×n矩阵A：
rank(A) + nullity(A) = n

其中nullity(A)是零空间的维数。

### 应用

给定矩阵的秩，可以确定：
- 列空间的维数 = rank(A)
- 行空间的维数 = rank(A)
- 零空间的维数 = n - rank(A)

## 秩在方程组中的应用

### 齐次方程组Ax = 0

- 若rank(A) = n，只有零解
- 若rank(A) < n，有无穷多解，自由变量个数 = n - rank(A)

### 非齐次方程组Ax = b

- 若rank(A) = rank([A|b]) = n，有唯一解
- 若rank(A) = rank([A|b]) < n，有无穷多解
- 若rank(A) < rank([A|b])，无解

## 重要不等式

### 西尔维斯特不等式

对于可相乘的矩阵A和B：
rank(A) + rank(B) - n ≤ rank(AB) ≤ min(rank(A), rank(B))

其中n是A的列数（也是B的行数）。

### 弗罗贝尼乌斯不等式

rank(AB) ≥ rank(A) + rank(B) - n

相关链接: [[Null Space|零空间]], [[Subspace|子空间]], [[Singular Matrix|奇异矩阵]]

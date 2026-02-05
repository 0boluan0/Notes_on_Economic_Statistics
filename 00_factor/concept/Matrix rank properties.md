---
aliases:
- 矩阵秩定理
- 矩阵秩的性质
- Rank properties
- Matrix
- Matrix rank properties
tags:
- proof
- 02_linear algebra
- 数学
- concept
---
# 矩阵秩的性质

## 定理内容

设 $A$ 为 $m \times n$ 矩阵，则矩阵的秩 $\text{rank}(A)$ 定义为：

**列秩**：$A$ 的列向量组的极大线性无关组所含向量的个数
**行秩**：$A$ 的行向量组的极大线性无关组所含向量的个数

**秩-零度定理（Sylvester 定理）**：
$\text{rank}(A) + \text{nullity}(A) = n$

$其中 \text{nullity}(A) = \dim(N(A)) 是零空间的维数。$

**秩的不等式**：
1. $0 \leq \text{rank}(A) \leq \min(m, n)$
2. $\text{rank}(AB) \leq \min(\text{rank}(A), \text{rank}(B))$
3. $\text{rank}(A+B) \leq \text{rank}(A) + \text{rank}(B)$

## 证明思路

通过行化简（高斯消元）将矩阵化为阶梯形，证明主元数量等于列秩和行秩。利用线性方程组解的结构证明秩-零度定理。

## 证明过程

### 步骤 1：行秩等于列秩

设矩阵 $A$ 经过行化简得到阶梯形矩阵 $R$。行变换不改变：
- 行向量组生成的行空间
- 列向量之间的线性关系

因此：
- $\text{row rank}(A) = \text{row rank}(R)$
- $\text{col rank}(A) = \text{col rank}(R)$

对于阶梯形矩阵 $R$：
- 每一行的第一个非零元素（主元）所在列线性无关
- 主元所在的列构成列向量组的极大线性无关组
- 非零行构成行向量组的极大线性无关组

因此：
$\text{row rank}(R) = \text{col rank}(R) = \text{主元个数} = \text{rank}(R)$

结论：
$\text{rank}(A) = \text{row rank}(A) = \text{col rank}(A)$

### 步骤 2：证明秩-零度定理

考虑线性方程组 $Ax = 0$，其中 $x \in \mathbb{R}^n$。

设 $\text{rank}(A) = r$。通过行化简将 $A$ 化为简化阶梯形 $R$，设 $R$ 有 $r$ 个主元，对应主元列和自由变量列。

设主元变量为 $x_1, \ldots, x_r$，自由变量为 $x_{r+1}, \ldots, x_n$。

**构造零空间的基**：

给每个自由变量赋值 1，其余自由变量为 0，解出主元变量：

例如，令 $x_{r+1} = 1$，$x_{r+2} = \cdots = x_n = 0$：
- 从 $R$ 的最后一行开始，用回代法求出 $x_r, x_{r-1}, \ldots, x_1$
- 得到一个解向量 $v_1$

类似地，令 $x_{r+2} = 1$，其余自由变量为 0，得到解向量 $v_2$，依此类推。

共得到 $n - r$ 个线性无关的解向量 $\{v_1, v_2, \ldots, v_{n-r}\}$，构成零空间 $N(A)$ 的基。

因此：
$\text{nullity}(A) = n - r$

即：
$\text{rank}(A) + \text{nullity}(A) = r + (n - r) = n$

### 步骤 3：证明秩的上下界

设 $A$ 为 $m \times n$ 矩阵。

- 列秩不超过列数：$\text{rank}(A) \leq n$
- 行秩不超过行数：$\text{rank}(A) \leq m$

因此：
$\text{rank}(A) \leq \min(m, n)$

### 步骤 4：证明矩阵乘积的秩不等式

设 $A$ 为 $m \times n$ 矩阵，$B$ 为 $n \times p$ 矩阵。

**列空间角度**：
$C(AB) = \{ABx \mid x \in \mathbb{R}^p\} = \{A(Bx) \mid x \in \mathbb{R}^p\} \subseteq \{Ay \mid y \in \mathbb{R}^n\} = C(A)$

AB 的列空间是 A 的列空间的子空间，因此：
$\text{rank}(AB) = \dim(C(AB)) \leq \dim(C(A)) = \text{rank}(A)$

**行空间角度**（利用 (AB)^T = B^T A^T）：
$\text{rank}(AB) = \text{rank}((AB)^T) = \text{rank}(B^T A^T) \leq \text{rank}(B^T) = \text{rank}(B)$

综合：
$\text{rank}(AB) \leq \min(\text{rank}(A), \text{rank}(B))$

### 步骤 5：证明矩阵和的秩不等式

$C(A+B) \subseteq C(A) + C(B)$

其中 C(A) + C(B) = \{y + z \mid y \in C(A), z \in C(B)\} 是列空间的和。

因此：
$\dim(C(A+B)) \leq \dim(C(A) + C(B)) \leq \dim(C(A)) + \dim(C(B))$

即：
$\text{rank}(A+B) \leq \text{rank}(A) + \text{rank}(B)$

## 结论

秩是矩阵最基本的不变量之一，其几何和代数含义：

1. **列秩** = 列空间的维数 = 线性无关列向量的最大个数
2. **行秩** = 行空间的维数 = 线性无关行向量的最大个数
3. **秩-零度定理**：矩阵的秩加上零空间的维数等于列数

**重要应用**：

- **线性方程组解的判定**：
  - $Ax = b$ 有解当且仅当 $\text{rank}(A) = \text{rank}(A|b)$
  - 唯一解当且仅当 $\text{rank}(A) = n$（满列秩）
  - 无穷多解当且仅当 $\text{rank}(A) < n$

- **矩阵的可逆性**：
  - $n \times n$ 方阵可逆当且仅当 $\text{rank}(A) = n$（满秩）

- **协整检验**（Johansen 方法）：
  - 通过 $\pi$ 矩阵的秩判断协整关系数量

## 相关概念
[[Linear system solution structure|线性方程组解的结构]]
[[Matrix Inverse|矩阵的逆]]

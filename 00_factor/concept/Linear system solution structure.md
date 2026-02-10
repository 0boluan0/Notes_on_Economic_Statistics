---
aliases:
- 齐次与非齐次线性方程组解的关系
- 线性方程组解的结构
- Linear system solution structure
tags:
- proof
- 02_linear algebra
- 数学
- concept
- 计算机
---
# 线性方程组解的结构

## 定理内容

$考虑线性方程组 Ax = b，其中 A 为 m \times n 矩阵，x \in \mathbb{R}^n，b \in \mathbb{R}^m。$

**齐次线性方程组** $Ax = 0$：
- 解集合构成零空间 $N(A)$
- $设 \text{rank}(A) = r，则 \dim(N(A)) = n - r$

**非齐次线性方程组** $Ax = b$：
- 解的结构：$x = x_p + x_h，其中 x_p 为特解，x_h \in N(A) 为齐次方程的通解$

**解的存在唯一性判别**：

1. **有解条件**：$\text{rank}(A) = \text{rank}(A|b)（增广矩阵秩等于系数矩阵秩）$
2. **唯一解**：$\text{rank}(A) = n（满列秩）$
3. **无穷多解**：$\text{rank}(A) < n$ 且方程组相容

## 证明思路

利用矩阵的秩-零度定理，证明齐次方程解空间的维数。通过非齐次方程与齐次方程的关系，导出解的结构。

## 证明过程

### 步骤 1：齐次方程解的性质

考虑 $Ax = 0$。

**线性性**：
- 若 $x_1$, $x_2$ 是解，则 $c_1$ $x_1$ + $c_2$ $x_2$ 也是解
- 证明：$A(c_1 x_1 + c_2 x_2) = c_1 Ax_1 + c_2 Ax_2 = 0$

因此，解集合 $N(A)$ 构成向量空间（子空间）。

### 步骤 2：齐次方程解空间的维数

$设 \text{rank}(A) = r，将 A 化为简化阶梯形 R。$

- $R$ 有 $r$ 个主元，对应 $r$ 个主元变量
- 剩余 $n - r$ 个变量为自由变量

每个自由变量可以任意取值，主元变量由自由变量唯一确定。因此解空间的维数等于自由变量的个数：

$\dim(N(A)) = n - r$

这验证了秩-零度定理：
$\text{rank}(A) + \dim(N(A)) = r + (n - r) = n$

### 步骤 3：非齐次方程的通解结构

**引理**：$设 x_p 是 Ax = b 的一个特解，则所有解可表示为：$
$x = x_p + x_h$

$其中 x_h 满足 Ax_h = 0（即 x_h \in N(A)）。$

**证明**：

(1) **充分性**：$若 x = x_p + x_h，则：$
$Ax = A(x_p + x_h) = Ax_p + Ax_h = b + 0 = b$
因此 $x$ 是解。

(2) **必要性**：$若 x 是 Ax = b 的任意解，令 x_h = x - x_p，则：$
$Ax_h = A(x - x_p) = Ax - Ax_p = b - b = 0$
因此 x_h \in N(A)，且 x = x_p + x_h。

### 步骤 4：解的存在条件（Rouché-Capelli 定理）

方程组 Ax = b 有解当且仅当向量 b 位于 A 的列空间中。

构造增广矩阵 [A|b]，其列空间 C(A|b) 由 A 的列和 b 生成。

**必要性**：若有解 x，则 Ax = b = \sum_{i=1}^n x_i a_i（a_i 为 A 的第 i 列）
因此 b \in C(A)。

**充分性**：若 b \in C(A)，则存在 x 使得 Ax = b。

用秩表示：
$\text{rank}(A) = \dim(C(A)) = \dim(C(A|b)) = \text{rank}(A|b)$

即系数矩阵的秩等于增广矩阵的秩。

### 步骤 5：解的唯一性

**唯一解**：若 Ax = b 有两个解 x_1, x_2，则：
$A(x_1 - x_2) = Ax_1 - Ax_2 = b - b = 0$

因此 x_1 - x_2 \in N(A)。

- 若 $\dim(N(A)) = 0$（即 $\text{rank}(A) = n$），则 $N(A) = \{0\}$，所以 $x_1 - x_2 = 0$，即 $x_1 = x_2$
- 若 \dim(N(A)) > 0，则存在非零的 x_h \in N(A)，若 $x_p$ 是解，则 $x_p$ + $x_h$ 也是解，解不唯一

### 步骤 6：矩阵形式总结

设 $\text{rank}(A) = r$：

| 条件 | 解的情况 |
|------|----------|
| $\text{rank}(A) \neq \text{rank}(A|b)$ | 无解 |
| $\text{rank}(A) = \text{rank}(A|b) = n$ | 唯一解 |
| $\text{rank}(A) = \text{rank}(A|b) < n$ | 无穷多解，解空间维数为 $n - r$ |

## 结论

线性方程组解的结构揭示了**特解与通解**的关系：

1. **齐次方程**：解空间是零空间 $N(A)$，维数为 $n - \text{rank}(A)$
2. **非齐次方程**：解 = 特解 + 齐次通解，解集合是零空间的平移
3. **几何意义**：非齐次方程的解集是通过零空间原点的平行超平面

**参数化通解的步骤**：

1. 求增广矩阵 $[A|b]$ 的简化阶梯形
2. 判断方程组是否有解（比较秩）
3. 若有解，令自由变量为参数（如 $t_1, t_2, \ldots$）
4. 用回代法将主元变量表示为自由变量的线性组合
5. 将解写为向量形式：$x = x_p + t_1 v_1 + t_2 v_2 + \cdots$

**应用**：
- 求逆矩阵的算法（解 AX = I）
- 最小二乘法（Ax = b 无精确解时求近似解）
- 差分方程的迭代求解

## 相关概念
[[Matrix rank properties|矩阵秩的性质]]
[[Matrix Inverse|矩阵的逆]]

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

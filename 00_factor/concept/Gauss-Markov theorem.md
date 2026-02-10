---
aliases:
- Gauss-Markov 定理
- OLS 最优性
- 高斯-马尔可夫定理
- Gauss
- Gauss-Markov theorem
tags:
- proof
- 01_Econometrics
- concept
---
# 高斯-马尔可夫定理

## 定理内容

**经典线性回归模型（CLRM）**：
$y = X\beta + \varepsilon$

其中：
- $y$ 为 $n \times 1$ 观测向量
- $X$ 为 $n \times k$ 设计矩阵（秩为 $k$）
- $\beta$ 为 $k \times 1$ 未知参数向量
- $\varepsilon$ 为 $n \times 1$ 误差项

**高斯-马尔可夫假设**：

1. **线性性**：$y = X\beta + \varepsilon（参数线性）$
2. **严格外生性**：$E[\varepsilon | X] = 0$
3. **同方差性**：$\text{Var}(\varepsilon | X) = \sigma^2 I_n$
4. **无自相关**：$\text{Cov}(\varepsilon_i, \varepsilon_j) = 0 当 i \neq j$
5. **非随机回归元**（或条件于 $X$）：$X$ 可以视为固定

**高斯-马尔可夫定理**：

在高斯-马尔可夫假设下，**普通最小二乘法（OLS）估计量** $\hat{\beta}_{OLS}$ 是 $\beta$ 的**最佳线性无偏估计量（BLUE）**：

1. **线性**：$\hat{\beta}_{OLS}$ 是 $y$ 的线性组合
2. **无偏**：$E[\hat{\beta}_{OLS}] = \beta$
3. **有效**：在所有线性无偏估计量中，OLS 具有最小方差

## 证明思路

首先推导 OLS 估计量的公式，证明其线性和无偏性。然后考虑任意线性无偏估计量，通过方差矩阵的比较证明 OLS 的方差最小。

## 证明过程

### 步骤 1：推导 OLS 估计量

最小化残差平方和：
$S(\beta) = (y - X\beta)'(y - X\beta)$

展开并对 $\beta$ 求导：
$$
\begin{aligned}
S(\beta) &= y'y - 2\beta'X'y + \beta'X'X\beta \\
\frac{\partial S}{\partial \beta} &= -2X'y + 2X'X\beta
\end{aligned}
$$

令导数为 0，得到**正规方程组**：
$X'X\hat{\beta} = X'y$

解得 OLS 估计量：
$\hat{\beta}_{OLS} = (X'X)^{-1}X'y$

### 步骤 2：证明 OLS 是线性估计量

$\hat{\beta}_{OLS}$ 可以表示为：
$\hat{\beta}_{OLS} = (X'X)^{-1}X'y = A y$

$其中 A = (X'X)^{-1}X' 是 k \times n 矩阵，仅依赖于 X。$

因此 $\hat{\beta}_{OLS}$ 是观测向量 $y$ 的**线性组合**。

### 步骤 3：证明 OLS 是无偏的

取期望（条件于 $X$）：
$$
\begin{aligned}
E[\hat{\beta}_{OLS} | X] &= E[(X'X)^{-1}X'y | X] \\
&= (X'X)^{-1}X'E[y | X] \\
&= (X'X)^{-1}X'E[X\beta + \varepsilon | X] \\
&= (X'X)^{-1}X'(X\beta + E[\varepsilon | X]) \\
&= (X'X)^{-1}X'X\beta + (X'X)^{-1}X' \cdot 0 \\
&= \beta
\end{aligned}
$$

$因此 E[\hat{\beta}_{OLS}] = \beta，OLS 估计量是**无偏的**。$

### 步骤 4：计算 OLS 的方差

$$
\begin{aligned}
\text{Var}(\hat{\beta}_{OLS} | X)
&= \text{Var}((X'X)^{-1}X'y | X) \\
&= (X'X)^{-1}X' \text{Var}(y | X) X (X'X)^{-1} \\
&= (X'X)^{-1}X' \text{Var}(X\beta + \varepsilon | X) X (X'X)^{-1} \\
&= (X'X)^{-1}X' (\sigma^2 I) X (X'X)^{-1} \\
&= \sigma^2 (X'X)^{-1}X'X (X'X)^{-1} \\
&= \sigma^2 (X'X)^{-1}
\end{aligned}
$$

### 步骤 5：考虑任意线性无偏估计量

$设 \tilde{\beta} = Cy 为任意线性估计量，其中 C 为 k \times n 矩阵。$

**无偏性条件**：
$E[\tilde{\beta} | X] = CE[y | X] = CX\beta = \beta$

这要求 $CX = I$（单位矩阵），即 $C$ 必须满足 $CX = I$。

### 步骤 6：比较 OLS 与任意线性无偏估计量的方差

计算 $\tilde{\beta}$ 的方差：
$\text{Var}(\tilde{\beta} | X) = \text{Var}(Cy | X) = C \text{Var}(y | X) C' = \sigma^2 C C'$

$令 C = (X'X)^{-1}X' + D，其中 D 是任意 k \times n 矩阵。$

由无偏性条件 $CX = I$：
$[(X'X)^{-1}X' + D]X = I \Rightarrow (X'X)^{-1}X'X + DX = I \Rightarrow I + DX = I \Rightarrow DX = 0$

计算 $\tilde{\beta}$ 的方差：
$$
\begin{aligned}
\text{Var}(\tilde{\beta} | X)
&= \sigma^2 [(X'X)^{-1}X' + D][(X'X)^{-1}X' + D]' \\
&= \sigma^2 [(X'X)^{-1}X' + D][X (X'X)^{-1} + D'] \\
&= \sigma^2 [(X'X)^{-1}X'X (X'X)^{-1} + (X'X)^{-1}X'D' + D X (X'X)^{-1} + D D'] \\
&= \sigma^2 [(X'X)^{-1} + (X'X)^{-1}X'D' + D X (X'X)^{-1} + D D']
\end{aligned}
$$

利用 $DX = 0$，则 $(DX)' = X'D' = 0$，因此：
$\text{Var}(\tilde{\beta} | X) = \sigma^2 [(X'X)^{-1} + D D']$

由于 $D D'$ 是半正定矩阵（对角线元素非负），有：
$\text{Var}(\tilde{\beta} | X) = \text{Var}(\hat{\beta}_{OLS} | X) + \sigma^2 D D' \geq \text{Var}(\hat{\beta}_{OLS} | X)$

即：
$\text{Var}(\tilde{\beta}) - \text{Var}(\hat{\beta}_{OLS}) = \sigma^2 D D' \text{ 是半正定的}$

### 步骤 7：证明最小方差性

对于任意线性无偏估计量 $\tilde{\beta}$，有：

$$
\begin{aligned}
\text{Var}(\tilde{\beta}) &= \text{Var}(\hat{\beta}_{OLS}) + \sigma^2 D D' \\
\text{diag}[\text{Var}(\tilde{\beta})] &\geq \text{diag}[\text{Var}(\hat{\beta}_{OLS})]
\end{aligned}
$$

这意味着 OLS 估计量的每个元素的方差都不大于任意其他线性无偏估计量对应元素的方差。

**特殊情况**：$当且仅当 D = 0（即 C = (X'X)^{-1}X'）时，\tilde{\beta} = \hat{\beta}_{OLS}。$

## 结论

高斯-马尔可夫定理保证了在经典线性回归模型假设下，OLS 估计量具有 BLUE 性质：

1. **线性**：$\hat{\beta}_{OLS} = (X'X)^{-1}X'y$
2. **无偏**：$E[\hat{\beta}_{OLS}] = \beta$
3. **有效**：$\text{Var}(\hat{\beta}_{OLS}) = \sigma^2 (X'X)^{-1} 达到最小$

**OLS 的方差公式**：
$\text{Var}(\hat{\beta}_{OLS}) = \sigma^2 (X'X)^{-1}$

**假设的必要性**：

- **外生性**：$E[\varepsilon | X] = 0 是无偏性的必要条件$
- **同方差和无自相关**：$\text{Var}(\varepsilon | X) = \sigma^2 I 是有效性的必要条件$
- 若违反同方差（异方差），应使用 GLS（广义最小二乘法）
- 若违反无自相关（序列相关），应使用 Newey-West 标准误

## 相关概念
[[OLS Estimator|OLS估计量]]
[[Linear Regression Model|线性回归模型]]
[[OLS Estimation Steps|OLS估计步骤]]
[[Heteroskedasticity|异方差]]

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

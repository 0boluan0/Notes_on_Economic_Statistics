---
aliases:
- OLS 无偏性
- OLS估计量的无偏性
- OLS
- OLS unbiasedness
tags:
- proof
- 01_Econometrics
- concept
---
# OLS 估计量的无偏性

## 定理内容

考虑线性回归模型：
$$y = X\beta + \varepsilon$$

其中：
- $y$ 为 $n \times 1$ 观测向量
- $X$ 为 $n \times k$ 设计矩阵（秩为 $k$）
- $\beta$ 为 $k \times 1$ 未知参数向量
- $\varepsilon$ 为 $n \times 1$ 误差项

**假设**：
1. **外生性**：$E[\varepsilon | X] = 0$（或 $E[\varepsilon] = 0$ 且 $X$ 与 $\varepsilon$ 不相关）
2. **设计矩阵满秩**：$\text{rank}(X) = k$（无完全多重共线性）

**OLS 估计量**：
$$\hat{\beta}_{OLS} = (X'X)^{-1}X'y$$

**无偏性定理**：
$$E[\hat{\beta}_{OLS}] = \beta$$

## 证明思路

将 OLS 估计量表示为 $y$ 的线性函数，代入模型 $y = X\beta + \varepsilon$，利用误差项的零期望和外生性证明期望等于真实参数。

## 证明过程

### 步骤 1：写出 OLS 估计量

从正规方程组 $X'X\hat{\beta} = X'y$ 解得：
$$\hat{\beta}_{OLS} = (X'X)^{-1}X'y$$

### 步骤 2：将 $y$ 用模型表示

代入 $y = X\beta + \varepsilon$：
$$
\begin{aligned}
\hat{\beta}_{OLS}
&= (X'X)^{-1}X'(X\beta + \varepsilon) \\
&= (X'X)^{-1}X'X\beta + (X'X)^{-1}X'\varepsilon \\
&= \beta + (X'X)^{-1}X'\varepsilon
\end{aligned}
$$

### 步骤 3：取期望（无条件期望）

$$
\begin{aligned}
E[\hat{\beta}_{OLS}]
&= E[\beta + (X'X)^{-1}X'\varepsilon] \\
&= E[\beta] + E[(X'X)^{-1}X'\varepsilon]
\end{aligned}
$$

由于 $\beta$ 是非随机参数，$E[\beta] = \beta$。

### 步骤 4：分析误差项的期望

这里有两种情况：

**情形 1：条件期望（严格外生性）**

假设 $E[\varepsilon | X] = 0$，使用期望的迭代律：
$$
\begin{aligned}
E[(X'X)^{-1}X'\varepsilon]
&= E[E[(X'X)^{-1}X'\varepsilon | X]] \\
&= E[(X'X)^{-1}X'E[\varepsilon | X]] \\
&= E[(X'X)^{-1}X' \cdot 0] \\
&= 0
\end{aligned}
$$

**情形 2：无条件期望（弱外生性）**

假设 $E[\varepsilon] = 0$ 且 $X$ 与 $\varepsilon$ 不相关（或 $X$ 非随机）：
$$E[(X'X)^{-1}X'\varepsilon] = (X'X)^{-1}X'E[\varepsilon] = (X'X)^{-1}X' \cdot 0 = 0$$

### 步骤 5：结论

$$
\begin{aligned}
E[\hat{\beta}_{OLS}]
&= \beta + 0 \\
&= \beta
\end{aligned}
$$

## 结论

OLS 估计量是无偏的，即：
$$E[\hat{\beta}_{OLS}] = \beta$$

**外生性条件的必要性**：

无偏性的关键假设是**外生性**（exogeneity）：

1. **严格外生性**：$E[\varepsilon | X] = 0$
   - 这是最强的假设，意味着误差项与过去、现在、未来的回归元都不相关
   - 例如：在时间序列中，$\varepsilon_t$ 与所有 $X_s$（$s \geq 1$）都不相关

2. **弱外生性/同期外生性**：$E[\varepsilon_t | X_t] = 0$
   - 较弱的假设，只要求误差项与同期回归元不相关
   - 例如：$\varepsilon_t$ 与 $X_t$ 不相关

3. **前定性（Predeterminedness）**：$E[\varepsilon_t | X_1, \ldots, X_t] = 0$
   - 误差项只依赖过去的回归元，不依赖未来

**违反外生性的后果**：

若外生性不成立（如遗漏变量偏差、测量误差、联立性），则：
$$E[\hat{\beta}_{OLS}] \neq \beta$$

估计量是**有偏的**（biased）。

**常见的外生性违背情形**：

1. **遗漏变量偏差（OVB）**：
   - 真实模型：$y = X_1 \beta_1 + X_2 \beta_2 + \varepsilon$
   - 估计模型：$y = X_1 \hat{\beta}_1 + \tilde{\varepsilon}$
   - 若 $\text{Cov}(X_1, X_2) \neq 0$，则 $\hat{\beta}_1$ 有偏

2. **测量误差**：
   - 若 $X$ 有测量误差，则 OLS 估计量向零偏移（衰减偏移）

3. **联立性（Simultaneity）**：
   - 内生解释变量与误差项相关

## 相关概念
[[00_factor/concept/Gauss-Markov theorem|高斯-马尔可夫定理]]
[[Linear Regression Model|线性回归模型]]

---
aliases:
- OLS 一致性
- OLS估计量的一致性
- OLS
tags:
- proof
- 01_Econometrics
- concept
---
# OLS 估计量的一致性

## 定理内容

考虑线性回归模型：
$$y = X\beta + \varepsilon$$

**OLS 估计量**：
$$\hat{\beta}_{OLS} = (X'X)^{-1}X'y$$

**一致性定义**：
$$\hat{\beta}_{OLS} \xrightarrow{p} \beta$$

即当样本容量 $n \to \infty$ 时，估计量依概率收敛到真实参数。

**一致性定理**（在适当假设下）：
$$\lim_{n \to \infty} \hat{\beta}_{OLS} = \beta \quad \text{（以概率为 1）}$$

## 证明思路

将 OLS 估计量展开为真实参数与误差项的和，利用大数定律和依概率收敛的性质证明当样本量增大时，误差项的影响趋近于零。

## 证明过程

### 步骤 1：展开 OLS 估计量

$$
\begin{aligned}
\hat{\beta}_{OLS}
&= (X'X)^{-1}X'y \\
&= (X'X)^{-1}X'(X\beta + \varepsilon) \\
&= (X'X)^{-1}X'X\beta + (X'X)^{-1}X'\varepsilon \\
&= \beta + (X'X)^{-1}X'\varepsilon
\end{aligned}
$$

### 步骤 2：重写误差项

将 $(X'X)^{-1}X'$ 重新写为：
$$(X'X)^{-1}X' = \left(\frac{1}{n}X'X\right)^{-1} \frac{1}{n}X'$$

因此：
$$\hat{\beta}_{OLS} - \beta = \left(\frac{1}{n}X'X\right)^{-1} \frac{1}{n}X'\varepsilon$$

### 步骤 3：应用大数定律

假设以下条件成立：

1. **回归元的外生性**：$E[X_i'\varepsilon_i] = 0$
2. **回归元的遍历性**：$\frac{1}{n}X'X \xrightarrow{p} E[X_i'X_i] = Q_{XX}$（满秩有限矩阵）
3. **$Q_{XX}$ 可逆**（无完全多重共线性）

由**大数定律（LLN）**：
$$\frac{1}{n}X'X \xrightarrow{p} E[X_i'X_i] = Q_{XX}$$
$$\frac{1}{n}X'\varepsilon \xrightarrow{p} E[X_i'\varepsilon_i] = 0$$

### 步骤 4：应用连续映射定理

由 $\frac{1}{n}X'X \xrightarrow{p} Q_{XX}$ 和 $Q_{XX}$ 可逆，根据连续映射定理：
$$\left(\frac{1}{n}X'X\right)^{-1} \xrightarrow{p} Q_{XX}^{-1}$$

### 步骤 5：综合收敛结果

$$
\begin{aligned}
\hat{\beta}_{OLS} - \beta
&= \left(\frac{1}{n}X'X\right)^{-1} \frac{1}{n}X'\varepsilon \\
&\xrightarrow{p} Q_{XX}^{-1} \cdot 0 \\
&= 0
\end{aligned}
$$

因此：
$$\hat{\beta}_{OLS} \xrightarrow{p} \beta$$

### 步骤 6：渐近正态性（额外）

由**中心极限定理（CLT）**：
$$\sqrt{n}\left(\frac{1}{n}X'\varepsilon\right) \xrightarrow{d} N(0, \Sigma)$$

其中 $\Sigma = E[X_i'\varepsilon_i \varepsilon_i' X_i]$。

因此：
$$\sqrt{n}(\hat{\beta}_{OLS} - \beta) \xrightarrow{d} N(0, Q_{XX}^{-1} \Sigma Q_{XX}^{-1})$$

在同方差假设下：
$$\sqrt{n}(\hat{\beta}_{OLS} - \beta) \xrightarrow{d} N(0, \sigma^2 Q_{XX}^{-1})$$

## 结论

OLS 估计量是一致的，只要满足以下条件：

1. **外生性**：$E[X_i'\varepsilon_i] = 0$
2. **遍历性**：$\frac{1}{n}X'X$ 收敛到满秩有限矩阵
3. **无完全多重共线性**：$Q_{XX}$ 可逆

**一致性与无偏性的区别**：

| 性质 | 定义 | 条件 |
|------|------|------|
| **无偏性** | $E[\hat{\beta}] = \beta$ | 有限样本性质，需要严格外生性 |
| **一致性** | $\hat{\beta} \xrightarrow{p} \beta$ | 大样本性质，要求更弱的条件 |

**一致但可能有偏的例子**：

某些估计方法（如岭回归、LASSO）在小样本下有偏，但随着样本增大，偏移趋近于零，因此是一致的。

**非一致的例子**：

1. **存在固定效应而使用混合 OLS**
2. **内生回归元**
3. **动态面板数据的短期系数估计**

## 相关概念
[[00_factor/concept/OLS unbiasedness|OLS估计量的无偏性]]
[[00_factor/concept/Gauss-Markov theorem|高斯-马尔可夫定理]]
[[00_factor/concept/Central Limit Theorem|中心极限定理]]

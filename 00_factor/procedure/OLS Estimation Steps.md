---
aliases:
- 最小二乘估计步骤
- OLS估计步骤
- OLS
tags:
- procedure
- 01_Econometrics
---
# OLS估计步骤

## 适用场景

估计线性回归模型 $y = X\beta + \epsilon$ 的参数，适用于满足经典假设的截面数据或时间序列数据。

## 所需数据/条件

- 因变量向量 $y$（$n \times 1$）
- 自变量矩阵 $X$（$n \times k$），包含常数项列
- 满足高斯-马尔可夫假设：
  1. 线性性：$E[y|X] = X\beta$
  2. 外生性：$E[\epsilon|X] = 0$
  3. 同方差：$\text{Var}[\epsilon|X] = \sigma^2 I$
  4. 无自相关：$\text{Cov}[\epsilon_i, \epsilon_j|X] = 0, i \neq j$
  5. 无完全多重共线性：$X$ 满秩为 $k$

## 计算步骤

### 步骤 1：准备数据矩阵

构造包含常数项的自变量矩阵：
$$ X = \begin{pmatrix} 1 & x_{11} & \cdots & x_{1,k-1} \\ 1 & x_{21} & \cdots & x_{2,k-1} \\ \vdots & \vdots & \ddots & \vdots \\ 1 & x_{n1} & \cdots & x_{n,k-1} \end{pmatrix} $$

**注意点**：检查 $X$ 是否满秩，避免完全多重共线性导致不可估计。

### 步骤 2：计算X'X和X'y

$$ X'X = \sum_{i=1}^{n} x_i x_i' $$
$$ X'y = \sum_{i=1}^{n} x_i y_i $$

**注意点**：这些是 $k \times k$ 和 $k \times 1$ 矩阵。

### 步骤 3：求解参数估计量

$$ \hat{\beta} = (X'X)^{-1}X'y $$

**注意点**：需要矩阵求逆，若 $X'X$ 接近奇异，考虑岭回归或删除相关变量。

### 步骤 4：计算拟合值和残差

$$ \hat{y} = X\hat{\beta} $$
$$ \hat{\epsilon} = y - \hat{y} $$

### 步骤 5：估计误差方差

$$ \hat{\sigma}^2 = \frac{\hat{\epsilon}'\hat{\epsilon}}{n-k} $$

**注意点**：自由度为 $n-k$（样本量减参数数）。

### 步骤 6：计算参数估计量的协方差矩阵

$$ \text{Var}(\hat{\beta}) = \hat{\sigma}^2 (X'X)^{-1} $$

**注意点**：对角线元素为各参数估计量的方差，用于标准误计算。

### 步骤 7：计算拟合优度

**总平方和（TSS）**：
$$ \text{TSS} = (y - \bar{y}1)'(y - \bar{y}1) $$

**回归平方和（ESS）**：
$$ \text{ESS} = (\hat{y} - \bar{y}1)'(\hat{y} - \bar{y}1) $$

**残差平方和（RSS）**：
$$ \text{RSS} = \hat{\epsilon}'\hat{\epsilon} $$

**判定系数 $R^2$**：
$$ R^2 = \frac{\text{ESS}}{\text{TSS}} = 1 - \frac{\text{RSS}}{\text{TSS}} $$

**调整 $R^2$**：
$$ \bar{R}^2 = 1 - \frac{\text{RSS}/(n-k)}{\text{TSS}/(n-1)} $$

**注意点**：$R^2$ 随自变量增加而单调递增，$\bar{R}^2$ 考虑了自由度惩罚。

## 关键公式

**正规方程组**：
$$ X'X\hat{\beta} = X'y $$

**参数估计量**：
$$ \hat{\beta} = (X'X)^{-1}X'y $$

**BLUE性质**：
在满足高斯-马尔可夫假设下，$\hat{\beta}$ 是最佳线性无偏估计量（BLUE）。

## 常见问题

1. **多重共线性**：$X'X$ 接近奇异，参数估计不稳定，标准误很大。
2. **异方差**：$\hat{\sigma}^2$ 不再一致，标准误有偏。
3. **自相关**：时间序列数据不满足独立性假设。
4. **内生性**：自变量与误差项相关，参数估计有偏且不一致。

## 相关概念
[[多重共线性]]
[[异方差]]
[[自相关]]
[[内生性]]
[[MLE估计步骤]]

## 相关链接

- OLS性质：[[高斯-马尔可夫定理]]
- 违反假设的后果：[[异方差]], [[多重共线性]], [[自相关]]

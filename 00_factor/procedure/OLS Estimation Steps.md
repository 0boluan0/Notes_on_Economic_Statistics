---
aliases:
- OLS Estimation Steps
- OLS估计步骤
- 最小二乘估计步骤
tags:
- procedure
- econometrics
type: procedure
---
# OLS Estimation Steps

## 这张卡什么时候用

需要从数据矩阵出发，完整计算线性回归的 OLS 系数、残差、标准误和拟合优度时使用。

## 输入

- 因变量向量 $y$。
- 解释变量矩阵 $X$，通常包含常数项。
- 样本量 $n$ 和参数个数 $k$。

## 输出

- $\hat\beta$。
- 拟合值 $\hat y$ 和残差 $\hat u$。
- 残差方差、标准误、$t/F$ 检验、$R^2$。

## Step 1：构造数据矩阵

确认 $X$ 包含需要的常数项、虚拟变量和控制变量，并检查是否满列秩。

## Step 2：计算 OLS 系数

$$
\hat\beta=(X'X)^{-1}X'y
$$

如果 $X'X$ 不可逆，先检查完全多重共线性。

## Step 3：计算拟合值和残差

$$
\hat y=X\hat\beta,\qquad \hat u=y-\hat y
$$

并检查：

$$
X'\hat u=0
$$

## Step 4：估计残差方差

$$
\hat\sigma^2=\frac{\hat u'\hat u}{n-k}
$$

## Step 5：计算标准误

经典同方差标准误：

$$
Var(\hat\beta\mid X)=\hat\sigma^2(X'X)^{-1}
$$

若存在 [[Heteroskedasticity]] 或 [[Autocorrelation]]，不要使用这个经典标准误，改用 [[White Robust Standard Errors]] 或 [[Newey-West]]。

## Step 6：做推断与诊断

- 单个系数：[[t Test]]。
- 多个限制：[[F-test]]。
- 拟合优度：[[R-squared]]。
- 残差诊断：[[Heteroscedasticity Diagnosis]]、[[Autocorrelation Diagnosis]]、[[Multicollinearity]]、[[Endogeneity Diagnosis]]。

## 常见错误

- 忘记常数项。
- 把 OLS 系数显著当成因果成立。
- 有异方差/自相关时仍使用经典标准误。
- 完全共线变量同时放入模型。

## 来自课程位置

- [[02_一元线性回归]]
- [[03_多元线性回归]]
- [[05_多元回归模型的矩阵表达]]

## 关联卡片

- [[OLS Basics]]
- [[OLS Estimator]]
- [[Residual]]
- [[Gauss-Markov theorem]]

# 1. 第7章：多元线性回归（Multivariate Linear Regression）

>[!summary] 本章主线
> 本章把线性回归写成矩阵形式。重点不是重新学习计量经济学，而是用矩阵表达 OLS、残差、平方和分解和显著性检验。

## 1.1. 模型形式

单方程多元线性回归写作
$$
Y=\beta_0+\beta_1x_1+\cdots+\beta_kx_k+\epsilon.
$$

矩阵形式为
$$
\mathbf Y=\mathbf X\boldsymbol\beta+\boldsymbol\epsilon.
$$

其中：

- $\mathbf Y$ 是 $n\times1$ 响应变量向量；
- $\mathbf X$ 是 $n\times(k+1)$ 设计矩阵，通常含截距列；
- $\boldsymbol\beta$ 是 $(k+1)\times1$ 系数向量；
- $\boldsymbol\epsilon$ 是 $n\times1$ 误差向量。

## 1.2. 模型假设

1. 线性关系成立。
2. $E(\boldsymbol\epsilon)=0$。
3. $\operatorname{Var}(\boldsymbol\epsilon)=\sigma^2I_n$。
4. $\mathbf X$ 无完全多重共线性，即 $X'X$ 可逆。

>[!warning] 边界
> 这里的“多元线性回归”按本课语境主要是多个解释变量、一个响应变量。英文里 multivariate regression 有时指多个响应变量，阅读时要看上下文。

## 1.3. 最小二乘估计

OLS 估计量为
$$
\hat{\boldsymbol\beta}=(X'X)^{-1}X'Y.
$$

拟合值与残差为
$$
\hat Y=X\hat\beta,\qquad \hat\epsilon=Y-\hat Y.
$$

在经典假设下：
$$
E(\hat\beta)=\beta,\qquad
\operatorname{Var}(\hat\beta)=\sigma^2(X'X)^{-1}.
$$

## 1.4. 平方和分解

总平方和：
$$
TSS=Y'Y-n\bar Y^2.
$$

回归平方和：
$$
RSS=\hat Y'\hat Y-n\bar Y^2.
$$

残差平方和：
$$
ESS=\hat\epsilon'\hat\epsilon.
$$

分解为
$$
TSS=RSS+ESS.
$$

## 1.5. 决定系数

$$
R^2=\frac{RSS}{TSS}=1-\frac{ESS}{TSS}.
$$

>[!note] 解读
> $R^2$ 衡量样本内拟合比例，取值在 $[0,1]$。它不是因果强度，也不能单独证明模型正确。

## 1.6. 假设检验

### 1.6.1. 总体显著性 F 检验

检验
$$
H_0:\beta_1=\cdots=\beta_k=0.
$$

统计量为
$$
F=\frac{RSS/k}{ESS/(n-k-1)}
\sim F_{k,n-k-1}.
$$

### 1.6.2. 单个变量 t 检验

检验
$$
H_0:\beta_j=0.
$$

统计量为
$$
t=\frac{\hat\beta_j}{\sqrt{\hat\sigma^2[(X'X)^{-1}]_{jj}}}.
$$

## 1.7. 大样本性质

在常规条件下，
$$
\hat\beta\approx N\left(\beta,\sigma^2(X'X)^{-1}\right).
$$

## 1.8. 关联卡片

- [[Multivariate Linear Regression]]
- [[OLS Basics]]
- [[OLS Estimator]]
- [[Residual]]
- [[F-test]]
- [[t Test]]

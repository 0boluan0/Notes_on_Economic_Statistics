# 1. 第7章：多元线性回归（Multivariate Linear Regression）
<!-- bilingual-en:start -->
*1. Chapter 7: Multivariate Linear Regression*
<!-- bilingual-en:end -->

>[!note] 本章主线
> 本章把线性回归写成矩阵形式。重点不是重新学习计量经济学，而是用矩阵表达 OLS、残差、平方和分解和显著性检验。
> <!-- bilingual-en:start -->
> This chapter expresses linear regression in matrix form. The aim is not to relearn econometrics, but to represent OLS, residuals, the decomposition of sums of squares, and significance tests with matrices.
> <!-- bilingual-en:end -->

## 1.1. 模型形式
<!-- bilingual-en:start -->
*1.1. Model Form*
<!-- bilingual-en:end -->

单方程多元线性回归写作
<!-- bilingual-en:start -->
A single-equation multiple linear regression is written as
<!-- bilingual-en:end -->
$$
Y=\beta_0+\beta_1x_1+\cdots+\beta_kx_k+\epsilon.
$$

矩阵形式为
<!-- bilingual-en:start -->
Its matrix form is
<!-- bilingual-en:end -->
$$
\mathbf Y=\mathbf X\boldsymbol\beta+\boldsymbol\epsilon.
$$

其中：
<!-- bilingual-en:start -->
where:
<!-- bilingual-en:end -->

- $\mathbf Y$ 是 $n\times1$ 响应变量向量；
- $\mathbf X$ 是 $n\times(k+1)$ 设计矩阵，通常含截距列；
- $\boldsymbol\beta$ 是 $(k+1)\times1$ 系数向量；
- $\boldsymbol\epsilon$ 是 $n\times1$ 误差向量。
<!-- bilingual-en:start -->
- $\mathbf Y$ is the $n\times1$ vector of responses;
- $\mathbf X$ is the $n\times(k+1)$ design matrix, usually including an intercept column;
- $\boldsymbol\beta$ is the $(k+1)\times1$ coefficient vector;
- $\boldsymbol\epsilon$ is the $n\times1$ error vector.
<!-- bilingual-en:end -->

## 1.2. 模型假设
<!-- bilingual-en:start -->
*1.2. Model Assumptions*
<!-- bilingual-en:end -->

1. 线性关系成立。
2. $E(\boldsymbol\epsilon)=0$。
3. $\operatorname{Var}(\boldsymbol\epsilon)=\sigma^2I_n$。
4. $\mathbf X$ 无完全多重共线性，即 $X'X$ 可逆。
<!-- bilingual-en:start -->
1. The relationship is linear.
2. $E(\boldsymbol\epsilon)=0$.
3. $\operatorname{Var}(\boldsymbol\epsilon)=\sigma^2I_n$.
4. $\mathbf X$ has no perfect multicollinearity, so $X'X$ is invertible.
<!-- bilingual-en:end -->

>[!attention] 边界
> 这里的“多元线性回归”按本课语境主要是多个解释变量、一个响应变量。英文里 multivariate regression 有时指多个响应变量，阅读时要看上下文。
> <!-- bilingual-en:start -->
> In this course, the term used in the Chinese title mainly means several explanatory variables with one response. In English, *multivariate regression* can instead refer to several response variables, so the intended meaning must be inferred from context.
> <!-- bilingual-en:end -->

## 1.3. 最小二乘估计
<!-- bilingual-en:start -->
*1.3. Least-Squares Estimation*
<!-- bilingual-en:end -->

OLS 估计量为
<!-- bilingual-en:start -->
The OLS estimator is
<!-- bilingual-en:end -->
$$
\hat{\boldsymbol\beta}=(X'X)^{-1}X'Y.
$$

拟合值与残差为
<!-- bilingual-en:start -->
The fitted values and residuals are
<!-- bilingual-en:end -->
$$
\hat Y=X\hat\beta,\qquad \hat\epsilon=Y-\hat Y.
$$

在经典假设下：
<!-- bilingual-en:start -->
Under the classical assumptions:
<!-- bilingual-en:end -->
$$
E(\hat\beta)=\beta,\qquad
\operatorname{Var}(\hat\beta)=\sigma^2(X'X)^{-1}.
$$

## 1.4. 平方和分解
<!-- bilingual-en:start -->
*1.4. Decomposition of Sums of Squares*
<!-- bilingual-en:end -->

总平方和：
<!-- bilingual-en:start -->
Total sum of squares:
<!-- bilingual-en:end -->
$$
TSS=Y'Y-n\bar Y^2.
$$

回归平方和：
<!-- bilingual-en:start -->
Regression sum of squares:
<!-- bilingual-en:end -->
$$
RSS=\hat Y'\hat Y-n\bar Y^2.
$$

残差平方和：
<!-- bilingual-en:start -->
Error sum of squares:
<!-- bilingual-en:end -->
$$
ESS=\hat\epsilon'\hat\epsilon.
$$

分解为
<!-- bilingual-en:start -->
The decomposition is
<!-- bilingual-en:end -->
$$
TSS=RSS+ESS.
$$

## 1.5. 决定系数
<!-- bilingual-en:start -->
*1.5. Coefficient of Determination*
<!-- bilingual-en:end -->

$$
R^2=\frac{RSS}{TSS}=1-\frac{ESS}{TSS}.
$$

>[!note] 解读
> $R^2$ 衡量样本内拟合比例，取值在 $[0,1]$。它不是因果强度，也不能单独证明模型正确。
> <!-- bilingual-en:start -->
> $R^2$ measures the share of in-sample variation fitted by the model and lies in $[0,1]$. It is not a measure of causal strength and cannot, by itself, establish that the model is correct.
> <!-- bilingual-en:end -->

## 1.6. 假设检验
<!-- bilingual-en:start -->
*1.6. Hypothesis Tests*
<!-- bilingual-en:end -->

### 1.6.1. 总体显著性 F 检验
<!-- bilingual-en:start -->
*1.6.1. Overall-Significance F-test*
<!-- bilingual-en:end -->

检验
<!-- bilingual-en:start -->
Test
<!-- bilingual-en:end -->
$$
H_0:\beta_1=\cdots=\beta_k=0.
$$

统计量为
<!-- bilingual-en:start -->
using the statistic
<!-- bilingual-en:end -->
$$
F=\frac{RSS/k}{ESS/(n-k-1)}
\sim F_{k,n-k-1}.
$$

### 1.6.2. 单个变量 t 检验
<!-- bilingual-en:start -->
*1.6.2. t-test for an Individual Variable*
<!-- bilingual-en:end -->

检验
<!-- bilingual-en:start -->
Test
<!-- bilingual-en:end -->
$$
H_0:\beta_j=0.
$$

统计量为
<!-- bilingual-en:start -->
using the statistic
<!-- bilingual-en:end -->
$$
t=\frac{\hat\beta_j}{\sqrt{\hat\sigma^2[(X'X)^{-1}]_{jj}}}.
$$

## 1.7. 大样本性质
<!-- bilingual-en:start -->
*1.7. Large-Sample Properties*
<!-- bilingual-en:end -->

在常规条件下，
<!-- bilingual-en:start -->
Under standard conditions,
<!-- bilingual-en:end -->
$$
\hat\beta\approx N\left(\beta,\sigma^2(X'X)^{-1}\right).
$$

## 1.8. 关联卡片
<!-- bilingual-en:start -->
*1.8. Related Cards*
<!-- bilingual-en:end -->

- [[多元线性回归#模型与维度|Multivariate Linear Regression]]
- [[OLS 线性回归#模型、条件均值与线性投影|OLS Basics]]
- [[多元线性回归#最小二乘估计|OLS Estimator]]
- [[多元线性回归#诊断与边界|Residual]]
- [[多元线性回归#线性假设与解释|F-test]]
- [[多元线性回归#线性假设与解释|t Test]]

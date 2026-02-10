---
aliases:
- 极大似然估计
- MLE
- Maximum Likelihood Estimation
tags:
- 统计学
- 估计方法
- concept
---
极大似然估计（Maximum Likelihood Estimation, MLE）是一种参数估计方法，通过最大化样本观测值的似然函数来估计未知参数。

## 似然函数

给定独立同分布样本x₁, ..., $x_n$，来自分布f(x; θ)，似然函数为：

$L(\theta) = \prod_{i=1}^{n} f(x_i; \theta)$

其中θ是待估参数向量。

## 对数似然函数

为便于计算，常使用对数似然函数：

$\ell(\theta) = \ln L(\theta) = \sum_{i=1}^{n} \ln f(x_i; \theta)$

对数似然函数和似然函数在同一点达到最大值。

## MLE估计量

$\hat{\theta}_{MLE} = \arg\max_{\theta} L(\theta) = \arg\max_{\theta} \ell(\theta)$

## 一阶条件

$\frac{\partial \ell(\theta)}{\partial \theta} = 0$

## 示例：正态分布的MLE

给定x₁, ..., $x_n$ ~ N(μ, σ²)，对数似然函数：

$\ell(\mu, \sigma^2) = -\frac{n}{2}\ln(2\pi) - \frac{n}{2}\ln(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i - \mu)^2$

### 估计μ

$\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^{n}(x_i - \mu) = 0$

$\hat{\mu} = \bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$

### 估计σ²

$\frac{\partial \ell}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}\sum_{i=1}^{n}(x_i - \bar{x})^2 = 0$

$\hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2$

## 性质

### 1. 不变性（Invariance）

如果$\hat{\theta}$是θ的MLE，则g($\hat{\theta}$)是g(θ)的MLE。

### 2. 一致性（Consistency）

在大样本下，$\hat{\theta}_{MLE} \xrightarrow{p} \theta$。

### 3. 渐近正态性（Asymptotic Normality）

$\sqrt{n}(\hat{\theta}_{MLE} - \theta) \xrightarrow{d} N(0, I(\theta)^{-1})$

其中I(θ)是Fisher信息矩阵。

### 4. 渐近有效性（Asymptotic与其他）

估计）

在所有相合估计量中，MLE具有最小的渐近方差。

## Fisher信息矩阵

$I(\theta) = -E\left[\frac{\partial^2 \ell(\theta)}{\partial \theta \partial \theta'}\right]$

或：

$I(\theta) = E\left[\left(\frac{\partial \ell(\theta)}{\partial \theta}\right)\left(\frac{\partial \ell(\theta)}{\partial \theta}\right)'\right]$

## 渐近方差

$\text{Avar}(\hat{\theta}_{MLE}) = I(\theta)^{-1}$

## 似然比检验

### 1. 似然比统计量

$LR = -2 \ln\left(\frac{L(\theta_0)}{L(\theta)}\right) = 2(\ell(\theta) - \ell(\theta_0))$

### 2. Wald检验

$W = (\hat{\theta} - \theta_0)' \text{Avar}(\hat{\theta})^{-1} (\hat{\theta} - \theta_0)$

### 3. 拉格朗日乘数检验

$LM = \left[\frac{\partial \ell(\theta_0)}{\partial \theta}\right]' I(\theta_0)^{-1} \left[\frac{\partial \ell(\theta_0)}{\partial \theta}\right]$

在大样本下，LR ≈ W ≈ LM ~ χ²(r)。

## 与OLS的关系

在古典线性回归假定下，MLE和OLS估计量等价。

## 与GMM的关系

MLE是GMM的特例，使用期望和样本矩的差作为矩条件。

## 应用

1. **参数估计**：广泛用于各种分布的参数估计
2. **模型估计**：logit、probit、GARCH等模型的估计
3. **假设检验**：似然比检验、Wald检验、LM检验

相关链接: [[00_factor/concept/F-test|F检验]], [[Logit Model|logit模型]], [[Probit Model|probit模型]]

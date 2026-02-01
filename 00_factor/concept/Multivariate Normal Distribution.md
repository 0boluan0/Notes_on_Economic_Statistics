---
aliases:
- 多元正态分布
- MVN
tags:
- concept
- multivariate statistics
---
# 多元正态分布

## 定义

多元正态分布是单变量正态分布的高维扩展，描述 p 维随机向量的联合分布。

$$ X \sim N_p(\mu, \Sigma) $$

其中：
- $\mu$：p × 1 均值向量（Mean Vector）
- $\Sigma$：p × p 协方差矩阵（Covariance Matrix），必须是正定矩阵

## 概率密度函数

$$ f(X) = \frac{1}{(2\pi)^{p/2} |\Sigma|^{1/2}} e^{-\frac{1}{2} (X-\mu)'\Sigma^{-1}(X-\mu)} $$

### 关键组成部分

1. **常数项**：$(2\pi)^{p/2} |\Sigma|^{1/2}$，确保积分为 1
2. **指数项**：$(X-\mu)'\Sigma^{-1}(X-\mu)$，马哈拉诺比斯距离的平方

## 二元正态分布特例（p = 2）

$$ f(X_1, X_2) = \frac{1}{2\pi|\Sigma|^{1/2}} e^{-\frac{1}{2} \begin{pmatrix} X_1 - \mu_1 \\ X_2 - \mu_2 \end{pmatrix}' \Sigma^{-1} \begin{pmatrix} X_1 - \mu_1 \\(X_2 - \mu_2 \end{pmatrix}} $$

协方差矩阵：
$$ \Sigma = \begin{pmatrix} \sigma_1^2 & \rho\sigma_1\sigma_2 \\ \rho\sigma_1\sigma_2 & \sigma_2^2 \end{pmatrix} $$

## 核心性质

### 1. 线性组合
如果 $X \sim N_p(\mu, \Sigma)$ 且 a 是 p × 1 向量：
$$ a'X \sim N(a'\mu, a'\Sigma a) $$

**反向推导**：如果对于任意向量 a，$a'X \sim N(a'\mu, a'\Sigma a)$ 都成立，则 $X \sim N_p(\mu, \Sigma)$。

### 2. 仿射变换
如果 A 是 q × p 矩阵：
$$ AX \sim N_q(A\mu, A\Sigma A') $$

### 3. 平移
如果 d 是 p × 1 常数向量：
$$ X + d \sim N_p(\mu + d, \Sigma) $$

### 4. 条件分布（不考）
设 $X = \begin{pmatrix} X_1 \\ X_2 \end{pmatrix}$，则：
$$ X_1 | X_2 = x_2 \sim N(\mu_1 + \Sigma_{12}\Sigma_{22}^{-1}(x_2 - \mu_2), \Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}) $$

## 独立性

在多元正态分布中：
- **零协方差 ⇔ 独立**
- $\Sigma_{12} = 0$ 当且仅当 $X_1$ 与 $X_2$ 独立

## 等概率密度曲线

满足 $(X - \mu)'\Sigma^{-1}(X - \mu) = c^2$ 的点构成的椭圆（或椭球）：

- **中心**：位于均值向量 $\mu$
- **主轴方向**：对应 $\Sigma$ 的特征向量
- **主轴长度**：与特征值的平方根成正比

## 相关概念

- [[Wishart 分布]]
- [[马哈拉诺比斯距离]]
- [[Hotelling T² 检验]]

## 应用

- 主成分分析（PCA）
- 线性判别分析（LDA）
- 多元回归分析
- 协方差矩阵建模

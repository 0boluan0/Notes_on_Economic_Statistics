# 1. 第9章：因子分析（Factor Analysis）
<!-- bilingual-en:start -->
*1. Chapter 9: Factor Analysis*
<!-- bilingual-en:end -->

>[!note] 本章主线
> 因子分析可以看成 PCA 之后更进一步的问题：不仅想降维，还想用少数不可观测的公共因子解释变量之间的协方差结构。
> <!-- bilingual-en:start -->
> Factor analysis goes beyond PCA: it seeks not only to reduce dimension, but also to explain the covariance structure among variables with a small number of unobserved common factors.
> <!-- bilingual-en:end -->

## 1.1. 引言
<!-- bilingual-en:start -->
*1.1. Introduction*
<!-- bilingual-en:end -->

因子分析的目的：用少量随机因子描述多个变量之间的协方差结构。
<!-- bilingual-en:start -->
The purpose of factor analysis is to describe the covariance structure among many variables using a small number of random factors.
<!-- bilingual-en:end -->

核心思想：
<!-- bilingual-en:start -->
The central idea is that:
<!-- bilingual-en:end -->

- 如果变量可以按相关性分组，组内变量高度相关，组间变量相关较低；
- 每个组可由一个公共因子代表；
- 每个变量还保留自己独有的特殊部分。
<!-- bilingual-en:start -->
- Variables can be grouped by correlation, with high within-group and lower between-group correlations.
- Each group can be represented by a common factor.
- Each variable also retains an idiosyncratic component of its own.
<!-- bilingual-en:end -->

## 1.2. 正交因子模型（Orthogonal Factor Model）
<!-- bilingual-en:start -->
*1.2. The Orthogonal Factor Model*
<!-- bilingual-en:end -->

### 1.2.1. 模型设定
<!-- bilingual-en:start -->
*1.2.1. Model Specification*
<!-- bilingual-en:end -->

给定 $p\times1$ 随机向量 $X$，均值为 $\mu$，协方差矩阵为 $\Sigma$。
<!-- bilingual-en:start -->
Let the $p\times1$ random vector $X$ have mean $\mu$ and covariance matrix $\Sigma$.
<!-- bilingual-en:end -->

模型写作
<!-- bilingual-en:start -->
The model is
<!-- bilingual-en:end -->
$$
X-\mu=LF+\epsilon.
$$

其中：
<!-- bilingual-en:start -->
where:
<!-- bilingual-en:end -->

- $L$ 是 $p\times m$ 因子载荷矩阵；
- $F$ 是 $m\times1$ 公共因子向量，不可观测；
- $\epsilon$ 是 $p\times1$ 特殊因子向量。
<!-- bilingual-en:start -->
- $L$ is the $p\times m$ matrix of factor loadings;
- $F$ is the unobserved $m\times1$ vector of common factors;
- $\epsilon$ is the $p\times1$ vector of specific factors.
<!-- bilingual-en:end -->

### 1.2.2. 假设条件
<!-- bilingual-en:start -->
*1.2.2. Assumptions*
<!-- bilingual-en:end -->

$$
E(F)=0,\qquad \operatorname{Cov}(F)=I_m.
$$

$$
E(\epsilon)=0,\qquad \operatorname{Cov}(\epsilon)=\Psi.
$$

其中 $\Psi$ 是对角矩阵。它为对角矩阵的原因是：变量之间的共同相关性已经由公共因子 $F$ 解释，剩下的是每个变量独有的特殊部分。
<!-- bilingual-en:start -->
Here $\Psi$ is diagonal because the common factors $F$ have already explained covariance shared across variables, leaving only the variable-specific components.
<!-- bilingual-en:end -->

此外：
<!-- bilingual-en:start -->
In addition:
<!-- bilingual-en:end -->
$$
\operatorname{Cov}(\epsilon,F)=0.
$$

### 1.2.3. 协方差分解
<!-- bilingual-en:start -->
*1.2.3. Covariance Decomposition*
<!-- bilingual-en:end -->

由模型可得
<!-- bilingual-en:start -->
The model implies
<!-- bilingual-en:end -->
$$
\Sigma=LL'+\Psi.
$$

其中：
<!-- bilingual-en:start -->
where:
<!-- bilingual-en:end -->

- $LL'$ 是公共因子贡献的协方差；
- $\Psi$ 是特殊因子的协方差。
<!-- bilingual-en:start -->
- $LL'$ is the covariance contributed by the common factors;
- $\Psi$ is the covariance of the specific factors.
<!-- bilingual-en:end -->

>[!note] 复习核心
> 因子分析最重要的公式就是 $\Sigma=LL'+\Psi$。PCA 没有这个“公共部分 + 特殊部分”的模型分解。
> <!-- bilingual-en:start -->
> The central formula in factor analysis is $\Sigma=LL'+\Psi$. PCA does not contain this model-based decomposition into common and specific components.
> <!-- bilingual-en:end -->

## 1.3. 公共度与特殊方差
<!-- bilingual-en:start -->
*1.3. Communality and Specific Variance*
<!-- bilingual-en:end -->

第 $i$ 个变量的公共度为
<!-- bilingual-en:start -->
The communality of variable $i$ is
<!-- bilingual-en:end -->
$$
h_i^2=\sum_{j=1}^m l_{ij}^2.
$$

特殊方差为
<!-- bilingual-en:start -->
Its specific variance is
<!-- bilingual-en:end -->
$$
\psi_i=\sigma_{ii}-h_i^2.
$$

所以
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->
$$
\sigma_{ii}=h_i^2+\psi_i.
$$

>[!attention] Heywood case
> 如果估计出 $\hat\psi_i<0$，通常说明模型或因子数设定有问题，不应机械接受。
> <!-- bilingual-en:start -->
> An estimate with $\hat\psi_i<0$ usually indicates a problem with the model or the chosen number of factors and should not be accepted mechanically.
> <!-- bilingual-en:end -->

## 1.4. 因子载荷的非唯一性
<!-- bilingual-en:start -->
*1.4. Non-Uniqueness of Factor Loadings*
<!-- bilingual-en:end -->

因子载荷矩阵 $L$ 不是唯一的。
<!-- bilingual-en:start -->
The loading matrix $L$ is not unique.
<!-- bilingual-en:end -->

如果 $T$ 是正交矩阵，则
<!-- bilingual-en:start -->
If $T$ is orthogonal, then
<!-- bilingual-en:end -->
$$
L^*=LT
$$
也满足同样的协方差结构，因为
<!-- bilingual-en:start -->
has the same covariance structure because
<!-- bilingual-en:end -->
$$
(LT)(LT)'=LTT'L'=LL'.
$$

因此因子分析常配合旋转，让载荷矩阵更容易解释。
<!-- bilingual-en:start -->
Factor analysis therefore commonly uses rotations to make the loading matrix easier to interpret.
<!-- bilingual-en:end -->

## 1.5. 参数估计方法
<!-- bilingual-en:start -->
*1.5. Parameter-Estimation Methods*
<!-- bilingual-en:end -->

### 1.5.1. 主成分法（Principal Component Method）
<!-- bilingual-en:start -->
*1.5.1. Principal-Component Method*
<!-- bilingual-en:end -->

给定样本协方差矩阵 $S$，先做特征值分解：
<!-- bilingual-en:start -->
Given the sample covariance matrix $S$, first compute its eigendecomposition:
<!-- bilingual-en:end -->
$$
S=\sum_{j=1}^p\lambda_j e_je_j'.
$$

保留最大的 $m$ 个特征值，近似为
<!-- bilingual-en:start -->
Retain the largest $m$ eigenvalues and use the approximation
<!-- bilingual-en:end -->
$$
\Sigma\approx LL'+\Psi.
$$

因子载荷矩阵估计为
<!-- bilingual-en:start -->
The estimated loading matrix is
<!-- bilingual-en:end -->
$$
\hat L=
\left[
\sqrt{\lambda_1}e_1,\sqrt{\lambda_2}e_2,\ldots,\sqrt{\lambda_m}e_m
\right].
$$

特殊方差估计为
<!-- bilingual-en:start -->
The estimated specific variances are
<!-- bilingual-en:end -->
$$
\hat\psi_i=s_{ii}-\sum_{j=1}^m\hat l_{ij}^2.
$$

>[!example] 做题顺序
> 先求特征值和特征向量，再取前 $m$ 个构造 $\hat L$，最后逐个变量算公共度和特殊方差。
> <!-- bilingual-en:start -->
> First compute the eigenvalues and eigenvectors, use the leading $m$ to construct $\hat L$, and then calculate each variable's communality and specific variance.
> <!-- bilingual-en:end -->

### 1.5.2. 极大似然法（Maximum Likelihood Method）
<!-- bilingual-en:start -->
*1.5.2. Maximum-Likelihood Method*
<!-- bilingual-en:end -->

若假设
<!-- bilingual-en:start -->
If
<!-- bilingual-en:end -->
$$
X\sim N_p(\mu,\Sigma),
$$
可在约束 $L'\Psi^{-1}L$ 为对角矩阵下估计 $L$ 和 $\Psi$。
<!-- bilingual-en:start -->
$L$ and $\Psi$ can be estimated subject to the constraint that $L'\Psi^{-1}L$ is diagonal.
<!-- bilingual-en:end -->

>[!note] 课堂提示
> 旧笔记标注“考试不会考”。本轮整理只保留识别信息，不展开推导。
> <!-- bilingual-en:start -->
> The old note says that this material will not be examined. This revision retains only enough information to recognise the method and does not develop the derivation.
> <!-- bilingual-en:end -->

## 1.6. 因子数量选择
<!-- bilingual-en:start -->
*1.6. Choosing the Number of Factors*
<!-- bilingual-en:end -->

常见依据：
<!-- bilingual-en:start -->
Common criteria include:
<!-- bilingual-en:end -->

1. 碎石图。
2. 累计方差解释率。
3. 残差矩阵 $S-(LL'+\Psi)$。
4. 信息准则，如 AIC 和 BIC。
5. 似然比检验。
<!-- bilingual-en:start -->

&nbsp;
**1.** A scree plot.<br>
**2.** Cumulative explained variance.<br>
**3.** The residual matrix $S-(LL'+\Psi)$.<br>
**4.** Information criteria such as AIC and BIC.<br>
**5.** A likelihood-ratio test.<br>
<!-- bilingual-en:end -->

>[!attention] 解释优先
> 因子数量不是越多越好；因子数量过多会失去“用少数潜在维度解释结构”的意义。
> <!-- bilingual-en:start -->
> More factors are not automatically better. Too many factors undermine the purpose of explaining structure with a small number of latent dimensions.
> <!-- bilingual-en:end -->

## 1.7. 关联卡片
<!-- bilingual-en:start -->
*1.7. Related Cards*
<!-- bilingual-en:end -->

- [[因子分析#因子模型|Factor Analysis]]
- [[因子分析#因子模型|Factor Analysis PC Method]]
- [[因子分析#因子模型|Factor Loadings]]
- [[因子分析#因子模型|Communality]]
- [[因子分析#因子模型|Specific Variance]]
- [[主成分分析 PCA#PCA 与因子分析的选择|PCA vs Factor Analysis]]

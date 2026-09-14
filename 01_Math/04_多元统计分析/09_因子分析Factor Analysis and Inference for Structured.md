# 1. 第9章：因子分析（Factor Analysis）
<!-- bilingual-en:start -->
*1. Chapter 9: Factor Analysis*
<!-- bilingual-en:end -->

>[!note] 本章主线
> PCA 用少数线性组合压缩总变异；因子分析则假定少数不可观测的公共因子与变量特有误差共同生成协方差。二者都处理低维结构，但回答的问题不同。
> <!-- bilingual-en:start -->
> PCA compresses total variation into a few linear combinations. Factor analysis instead models covariance as arising from a few unobserved common factors plus variable-specific errors. Both use low-dimensional structure, but answer different questions.
> <!-- bilingual-en:end -->

![[因子分析.canvas]]

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

本节使用 $\operatorname{Cov}(F)=I_m$ 的正交标准化特例；允许因子相关的一般形式与协方差分解见[[公共因子模型]]。
<!-- bilingual-en:start -->
This section uses the orthogonal standardised special case $\operatorname{Cov}(F)=I_m$. See [[公共因子模型|the common-factor model]] for the correlated-factor form and its covariance decomposition.
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

以下平方载荷求和公式依赖本章的正交标准化设定；斜交因子下的通式见[[公共度与特殊方差]]。
<!-- bilingual-en:start -->
The sum-of-squared-loadings formula below relies on this chapter's orthogonal standardisation. See [[公共度与特殊方差|communality and unique variance]] for the oblique-factor formula.
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
> 如果估计出 $\hat\psi_i<0$，就出现[[Heywood解|Heywood 不当解]]。它可能来自模型错设、因子数、样本不稳定或边界结构，不能机械接受，也不能未经诊断就归因于某一个原因。
> <!-- bilingual-en:start -->
> An estimate with $\hat\psi_i<0$ is an improper [[Heywood解|Heywood solution]]. It can arise from misspecification, factor retention, sample instability, or a boundary structure; it should neither be accepted mechanically nor assigned a single cause without diagnosis.
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

这里的等价变换与“拟合不变”边界见[[因子旋转]]。选择[[正交与斜交旋转|正交或斜交旋转]]后，斜交输出还必须区分[[模式矩阵与结构矩阵]]。
<!-- bilingual-en:start -->
See [[因子旋转|factor rotation]] for the equivalent transformation and fit-invariance boundary. After choosing an [[正交与斜交旋转|orthogonal or oblique rotation]], an oblique solution must also distinguish the [[模式矩阵与结构矩阵|pattern and structure matrices]].
<!-- bilingual-en:end -->

## 1.5. 参数估计方法
<!-- bilingual-en:start -->
*1.5. Parameter-Estimation Methods*
<!-- bilingual-en:end -->

[[因子提取]]先定义给定因子数后的共同输入与输出；三种算法怎样采用不同估计准则并获得不同推断资格，见[[因子提取方法辨析]]。本节只保留课程要求的计算口径。
<!-- bilingual-en:start -->
See [[因子提取|factor extraction]] for the common input and output, and [[因子提取方法辨析|the comparison of extraction methods]] for their distinct estimation criteria and inferential guarantees. This section retains only the computational form required by the course.
<!-- bilingual-en:end -->

### 1.5.1. 主成分因子法（Principal-Component Factor Method）
<!-- bilingual-en:start -->
*1.5.1. Principal-Component Factor Method*
<!-- bilingual-en:end -->

本章沿用课程名称“主成分因子法”；它从完整 $S$ 的谱分解构造与 PCA 相同的未旋转载荷，准确边界见[[主成分载荷近似]]。
<!-- bilingual-en:start -->
This chapter retains the course label “principal-component factor method.” Its unrotated loadings come from the full covariance spectrum and coincide with PCA loadings; see [[主成分载荷近似|principal-component loading approximation]].
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

### 1.5.2. 主轴因子法（Principal-Axis Factoring）
<!-- bilingual-en:start -->
*1.5.2. Principal-Axis Factoring*
<!-- bilingual-en:end -->

主轴因子法先用公共度估计替换相关或协方差矩阵的对角线，再提取前 $m$ 个共同方向并迭代更新公共度。它与从完整 $S$ 出发的主成分载荷近似不是同一算法，详见[[主轴因子法]]。
<!-- bilingual-en:start -->
Principal-axis factoring replaces the diagonal of the correlation or covariance matrix with communality estimates, extracts the leading common-factor directions, and iterates the communalities. See [[主轴因子法|principal-axis factoring]].
<!-- bilingual-en:end -->

### 1.5.3. 极大似然法（Maximum Likelihood Method）
<!-- bilingual-en:start -->
*1.5.3. Maximum-Likelihood Method*
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

这一步拟合的是 Gaussian 概率模型；似然目标、识别与经典检验的适用条件见[[极大似然因子法]]。
<!-- bilingual-en:start -->
This step fits a Gaussian probability model. See [[极大似然因子法|maximum-likelihood factor analysis]] for the likelihood target, identification, and inferential conditions.
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

常见依据包括：
<!-- bilingual-en:start -->
Common criteria include:
<!-- bilingual-en:end -->

1. 碎石图与平行分析。
2. 残差矩阵 $S-(LL'+\Psi)$ 及局部残差相关。
3. 极大似然模型中的似然比检验或 AIC、BIC，但只在其分布与正则条件可信时解释。
4. 每个因子的指标支撑、理论含义与合理旋转下的稳定性。
5. 新样本或留出样本中的结构复现。
<!-- bilingual-en:start -->

&nbsp;
**1.** A scree plot and parallel analysis.<br>
**2.** The residual matrix $S-(LL'+\Psi)$ and local residual correlations.<br>
**3.** Likelihood-ratio tests or AIC/BIC for maximum-likelihood models, interpreted only when their distributional and regularity conditions are credible.<br>
**4.** Indicator support, theoretical meaning, and stability under reasonable rotations.<br>
**5.** Replication in a new or held-out sample.<br>
<!-- bilingual-en:end -->

>[!attention] 解释优先
> 因子数量不是越多越好；过多因子会追逐抽样噪声，过少因子则会把系统结构留在残差中。特征值大于 1 或固定累计解释率都只能作为线索，不能单独决定因子数。完整边界见[[因子数选择]]。
> <!-- bilingual-en:start -->
> More factors are not automatically better: too many can follow sampling noise, while too few leave systematic structure in the residuals. Eigenvalue-greater-than-one and fixed cumulative-variance rules are clues, not standalone decisions; see [[因子数选择|choosing the number of factors]].
> <!-- bilingual-en:end -->

因子数选择与样本是否足以稳定恢复结构相互关联，但不是同一个问题；固定“每变量若干样本”的规则为何不足，见[[因子分析样本量边界]]。
<!-- bilingual-en:start -->
Factor retention and sample adequacy are related but distinct questions. See [[因子分析样本量边界|the factor-analysis sample-size boundary]] for why a fixed observations-per-variable rule is insufficient.
<!-- bilingual-en:end -->

## 1.7. 关联卡片
<!-- bilingual-en:start -->
*1.7. Related Cards*
<!-- bilingual-en:end -->

- [[公共因子模型|Factor Analysis]]
- [[因子载荷|Factor Loadings]]
- [[公共度与特殊方差|Communality and Specific Variance]]
- [[因子提取|Factor Extraction]]
- [[主成分载荷近似|Principal-Component Loading Approximation]]
- [[主轴因子法|Principal-Axis Factoring]]
- [[极大似然因子法|Maximum-Likelihood Factor Analysis]]
- [[因子提取方法辨析|Comparing Extraction Methods]]
- [[因子数选择|Choosing the Number of Factors]]
- [[因子分析样本量边界|Sample-Size Boundary]]
- [[因子旋转|Factor Rotation]]
- [[正交与斜交旋转|Orthogonal and Oblique Rotation]]
- [[模式矩阵与结构矩阵|Pattern and Structure Matrices]]
- [[Heywood解|Heywood Solutions]]
- [[因子结构验证|Factor-Structure Validation]]
- [[因子解释边界|Interpretive Limits]]
- [[因子因果边界|Causal Boundary]]
- [[PCA与因子分析|PCA vs Factor Analysis]]

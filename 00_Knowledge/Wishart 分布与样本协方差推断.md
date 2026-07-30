---
aliases:
  - "Wishart Distribution"
  - "Wishart 分布"
  - "Sample Covariance Distribution"
status: source-checked
---

# Wishart 分布与样本协方差推断
<!-- bilingual-en:start -->
*The Wishart distribution and inference for sample covariance*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 描述多元正态样本的协方差矩阵怎样随机波动，为多元均值检验、协方差检验和判别分析提供抽样分布。
> **具体锚点：** 一元正态样本的样本方差经缩放服从 $\chi^2$；多元情形把平方和升级为外积和，得到 Wishart 分布。
> **核心难点：** 不同教材对尺度与逆尺度参数化不同；自由度不足时矩阵必奇异，不能套用需要求逆的公式。
> **为什么重要：** Hotelling $T^2$、MANOVA、Gaussian covariance estimation 和 Bayesian inverse-Wishart 模型都以此为基础。
> **继续：** 先掌握 [[多元正态分布]] 和 [[多元数据、随机向量与样本协方差]]，再核对参数化与秩条件。
<!-- bilingual-en:start -->
> [!summary] Quick recovery
> **What it solves:** It describes random variation in the covariance matrix of a multivariate normal sample and supplies sampling distributions for multivariate mean tests, covariance tests, and discriminant analysis.
> **Concrete anchor:** A scaled sample variance from a univariate normal sample has a $\chi^2$ distribution. In several dimensions, squared terms become sums of outer products, producing the Wishart distribution.
> **Central difficulty:** Textbooks use different scale and inverse-scale parameterisations. With insufficient degrees of freedom, the matrix is necessarily singular and formulas requiring inversion do not apply.
> **Why it matters:** Hotelling's $T^2$, MANOVA, Gaussian covariance estimation, and Bayesian inverse-Wishart models all rely on it.
> **Continue with:** First master [[多元正态分布|the multivariate normal distribution]] and [[多元数据、随机向量与样本协方差|multivariate data and sample covariance]], then check parameterisation and rank conditions.
<!-- bilingual-en:end -->

> [!source] 本节依据
> - [Penn State STAT 505, Lessons 4–7](https://online.stat.psu.edu/stat505/)：核验正态抽样下样本均值、样本协方差与多元推断。
> - Johnson & Wichern, *Applied Multivariate Statistical Analysis*, 6th ed.：核验 Wishart 参数化、矩与秩条件。
<!-- bilingual-en:start -->
> [!source] Sources for this section
> - [Penn State STAT 505, Lessons 4–7](https://online.stat.psu.edu/stat505/) was used to verify sample means, sample covariance, and multivariate inference under normal sampling.
> - Johnson and Wichern, *Applied Multivariate Statistical Analysis*, 6th ed., was used to verify Wishart parameterisation, moments, and rank conditions.
<!-- bilingual-en:end -->

## Wishart 与样本协方差
<!-- bilingual-en:start -->
*Wishart distributions and sample covariance*
<!-- bilingual-en:end -->

正态随机样本下，$(n-1)S$ 服从以 $Sigma$ 为尺度、自由度 $n-1$ 的 Wishart 分布，且样本均值与 S 独立。Wishart 是多元版卡方；不同教材参数化可能不同，使用密度和期望公式前必须核对尺度约定。
<!-- bilingual-en:start -->
Under normal random sampling, $(n-1)S$ has a Wishart distribution with scale $Sigma$ and $n-1$ degrees of freedom, and the sample mean is independent of $S$. The Wishart distribution is the multivariate analogue of chi-square. Textbooks use different parameterisations, so scale conventions must be checked before using density or expectation formulas.
<!-- bilingual-en:end -->

若 $Z_1,\ldots,Z_\nu$ 独立服从 $N_p(0,\Sigma)$，则 $W=\sum_{i=1}^{\nu}Z_iZ_i^T\sim W_p(\Sigma,\nu)$，并有 $E[W]=\nu\Sigma$。中心化样本损失一个自由度，解释了为什么样本协方差对应 $n-1$ 而非 n。
<!-- bilingual-en:start -->
If $Z_1,\ldots,Z_\nu$ are independent $N_p(0,\Sigma)$ vectors, then $W=\sum_{i=1}^{\nu}Z_iZ_i^T\sim W_p(\Sigma,\nu)$ and $E[W]=\nu\Sigma$. Centring a sample uses one degree of freedom, explaining why sample covariance is associated with $n-1$ rather than $n$.
<!-- bilingual-en:end -->

## 秩、可逆性与高维边界
<!-- bilingual-en:start -->
*Rank, invertibility, and the high-dimensional boundary*
<!-- bilingual-en:end -->

外积和的秩至多为自由度，所以当 $\nu<p$ 时 W 必奇异；样本协方差在 $n-1<p$ 时不能求普通逆。即使刚好可逆，$p$ 接近 n 时最小特征值不稳，基于 $S^{-1}$ 的距离和检验也可能高度敏感。
<!-- bilingual-en:start -->
The rank of a sum of outer products cannot exceed its degrees of freedom, so $W$ is necessarily singular when $\nu<p$. Ordinary inversion of sample covariance is impossible when $n-1<p$. Even when barely invertible, the smallest eigenvalues are unstable as $p$ approaches $n$, making distances and tests based on $S^{-1}$ highly sensitive.
<!-- bilingual-en:end -->

## Worked example：从卡方到 Wishart
<!-- bilingual-en:start -->
*Worked example: from chi-square to Wishart*
<!-- bilingual-en:end -->

当 p=1 时，$W=\sum_{i=1}^{\nu}Z_i^2$，且 $Z_i/\sigma\sim N(0,1)$，所以 $W/\sigma^2\sim\chi_\nu^2$。Wishart 因而不是陌生的新对象，而是把一元平方和替换为向量外积和；对角元素是平方和，非对角元素同时记录交叉乘积。
<!-- bilingual-en:start -->
When $p=1$, $W=\sum_{i=1}^{\nu}Z_i^2$ and $Z_i/\sigma\sim N(0,1)$, so $W/\sigma^2\sim\chi_\nu^2$. The Wishart distribution is therefore not an unrelated new object: it replaces a univariate sum of squares with a sum of vector outer products. Diagonal elements contain sums of squares and off-diagonal elements contain cross-products.
<!-- bilingual-en:end -->

## 参数化与应用诊断
<!-- bilingual-en:start -->
*Parameterisation and application diagnostics*
<!-- bilingual-en:end -->

看到 $W_p(V,\nu)$ 时，先检查作者是否规定 $E[W]=\nu V$，还是把 V 作为 precision/逆尺度；再检查自由度和维数。软件函数的参数名 scale 不保证与教材一致。
<!-- bilingual-en:start -->
When encountering $W_p(V,\nu)$, first check whether the author defines $E[W]=\nu V$ or treats $V$ as a precision or inverse-scale parameter, then check degrees of freedom and dimension. A software argument named “scale” does not guarantee agreement with a textbook's convention.
<!-- bilingual-en:end -->

Wishart 结论依赖独立同分布多元正态样本。重尾、相关观察、异方差或结构变化都会改变样本协方差分布；大样本稳健性不能替代对有限样本逆矩阵和尾概率的检查。
<!-- bilingual-en:start -->
Wishart results rely on an iid multivariate normal sample. Heavy tails, correlated observations, heteroskedasticity, or structural change alter the distribution of sample covariance. Large-sample robustness does not replace checking finite-sample inverse matrices and tail probabilities.
<!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum self-check*
<!-- bilingual-en:end -->

### Wishart 分布在多元推断中扮演什么角色？
<!-- bilingual-en:start -->
*What role does the Wishart distribution play in multivariate inference?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 它描述正态样本协方差矩阵的抽样波动，类似卡方分布描述一元样本方差。
<!-- bilingual-en:start -->
> [!answer]- Answer
> It describes sampling variation in the covariance matrix of a normal sample, analogous to the way a chi-square distribution describes a univariate sample variance.
<!-- bilingual-en:end -->

### 为什么 $n\le p$ 时样本协方差必然奇异？
<!-- bilingual-en:start -->
*Why is sample covariance necessarily singular when $n\le p$?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 中心化后的 n 个向量最多张成 n−1 维空间，外积和的秩至多 n−1，小于 p。
<!-- bilingual-en:start -->
> [!answer]- Answer
> After centring, $n$ vectors span at most an $(n-1)$-dimensional space. Their sum of outer products therefore has rank at most $n-1<p$.
<!-- bilingual-en:end -->

### 使用 Wishart 软件函数前为何必须核对参数化？
<!-- bilingual-en:start -->
*Why must parameterisation be checked before using a Wishart software function?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 不同定义把参数写成尺度或逆尺度，期望与密度中的矩阵位置随之改变；混用会得到系统性错误。
<!-- bilingual-en:start -->
> [!answer]- Answer
> Different definitions use a scale or inverse scale, changing where the matrix appears in expectations and densities. Mixing conventions produces systematic errors.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [Penn State STAT 505, Lessons 4–7](https://online.stat.psu.edu/stat505/)：逐项核验正态抽样下样本均值与协方差、Hotelling 推断的分布基础。
<!-- bilingual-en:start -->
- [Penn State STAT 505, Lessons 4–7](https://online.stat.psu.edu/stat505/) was checked section by section for sample means and covariances under normal sampling and the distributional basis of Hotelling inference.
<!-- bilingual-en:end -->
- Johnson & Wichern, *Applied Multivariate Statistical Analysis*, 6th ed.：交叉核验 Wishart 定义、参数化、矩、独立性与秩条件。
<!-- bilingual-en:start -->
- Johnson and Wichern, *Applied Multivariate Statistical Analysis*, 6th ed., was used to cross-check the Wishart definition, parameterisations, moments, independence, and rank conditions.
<!-- bilingual-en:end -->

---
aliases:
  - Residual matrix versus error SSP
  - Multivariate residual matrix and SSCP
student_os: knowledge-atom
atom_id: STAT-MVR-005
atom_set: multivariate-linear-regression
atom_type: distinction
status: source-checked
part_of:
  - "[[多元线性回归.canvas|多元线性回归]]"
requires:
  - "[[共享设计的逐响应OLS]]"
leads_to:
  - "[[误差SSP抽样律]]"
  - "[[误差SSP秩边界]]"
related:
  - "[[多响应一般线性假设]]"
---

<!-- bilingual-en:start -->
# Multivariate residuals and error SSP
<!-- bilingual-en:end -->

*多响应残差与误差 SSP*

<!-- bilingual-en:start -->
After fitting several responses to the same observations, we can ask two different questions: **where did the fit miss each observation**, and **how much residual variation remains across the responses**? The residual matrix answers the first question. Its sums-of-squares-and-cross-products matrix, called error SSP or SSCP, summarizes the second. The summary is useful precisely because it compresses the data, but that compression loses the identities of individual observations.
<!-- bilingual-en:end -->

在同一批观测上拟合多个响应后，可以问两个不同的问题：**拟合在每条观测上偏离了多少**，以及**各响应还剩多少残差变动、这些变动怎样一起出现**？残差矩阵回答前者；误差平方和与交叉乘积矩阵，简称 error SSP 或 SSCP，汇总后者。这种压缩使汇总有用，也使它失去个别观测的身份信息。

<!-- bilingual-en:start -->
## From observed values to residuals
<!-- bilingual-en:end -->

*从观测值到残差*

<!-- bilingual-en:start -->
In [[多响应线性回归|multivariate linear regression]], let $Y$ be the real $n\times q$ response matrix: rows are observations and columns are responses. Here $q$ always counts **responses**. A shared $n\times k$ design matrix $X$ records the predictors, and the fitted $k\times q$ coefficient matrix $\hat B$ gives fitted values $\hat Y=X\hat B$. Subtract fitted from observed values entry by entry:
<!-- bilingual-en:end -->

在[[多响应线性回归]]中，实数矩阵 $Y$ 的尺寸为 $n\times q$，行是观测，列是响应。本卡的 $q$ 始终表示**响应数**。共享的 $n\times k$ 设计矩阵 $X$ 记录预测变量；拟合出的 $k\times q$ 系数矩阵 $\hat B$ 给出拟合值 $\hat Y=X\hat B$。逐项用观测值减去拟合值：

$$
R=Y-\hat Y=Y-X\hat B\in\mathbb R^{n\times q},
\qquad R_{ia}=Y_{ia}-\hat Y_{ia}.
$$

<!-- bilingual-en:start -->
Thus $R_{ia}$ says how far observation $i$ missed the fitted value for response $a$; a positive residual means the observed value was higher. Forming $R$ requires a specified fit, not a Gaussian assumption. For [[共享设计的逐响应OLS|ordinary least squares with a shared design]], each column is fitted by minimizing its sum of squared residuals. The degrees-of-freedom statements below refer to that unweighted fit.
<!-- bilingual-en:end -->

因此，$R_{ia}$ 表示第 $i$ 条观测在第 $a$ 个响应上偏离拟合值多少；残差为正意味着观测值更高。形成 $R$ 需要先指定拟合方式，不需要 Gaussian 假设。[[共享设计的逐响应OLS|共享设计下的普通最小二乘]]逐列最小化残差平方和；下文的自由度结论针对这种不加权拟合。

<!-- bilingual-en:start -->
## What multiplying by the transpose keeps
<!-- bilingual-en:end -->

*乘上转置后保留什么*

<!-- bilingual-en:start -->
The [[矩阵转置|transpose]] $R^T$ exchanges rows and columns. Multiplying it by $R$ takes the [[向量点积|dot product]] of every pair of residual columns:
<!-- bilingual-en:end -->

[[矩阵转置|转置]] $R^T$ 交换行列；再乘 $R$，就是计算每一对残差列的[[向量点积]]：

$$
E_{SSP}=R^TR\in\mathbb R^{q\times q},
\qquad (E_{SSP})_{ab}=\sum_{i=1}^{n}R_{ia}R_{ib}.
$$

<!-- bilingual-en:start -->
When $a=b$, this is response $a$'s residual **sum of squares**. When $a\ne b$, it is a **sum of cross-products**: observations with residuals of the same sign contribute positively, and those with opposite signs contribute negatively. These are sums, with no covariance-estimation denominator yet. The matrix is symmetric because exchanging $a$ and $b$ leaves every product unchanged.
<!-- bilingual-en:end -->

$a=b$ 时，这是响应 $a$ 的残差**平方和**；$a\ne b$ 时，这是残差**交叉乘积之和**：同一条观测的两个残差同号，就贡献正值；异号，就贡献负值。它们目前都是和，还没有除以估计协方差所需的分母。交换 $a,b$ 不改变任何乘积，所以这张矩阵对称。

> [!example] Three observations, two responses · 三条观测、两个响应
> <!-- bilingual-en:start -->
> Fit only an intercept to each column of
> $$Y=\begin{bmatrix}1&2\\2&4\\3&3\end{bmatrix}.$$
> The fitted values in every row are the column means $(2,3)$. Therefore
> $$R=\begin{bmatrix}-1&-1\\0&1\\1&0\end{bmatrix},\qquad
> E_{SSP}=\begin{bmatrix}2&1\\1&2\end{bmatrix}.$$
> For example, the off-diagonal entry is $(-1)(-1)+0(1)+1(0)=1$. The first row of $R$ identifies an observation below both fitted means. The $2\times2$ summary records the total cross-product, but does not identify that row.
> <!-- bilingual-en:end -->
>
> 对上述 $Y$ 的每列只拟合一个截距，每一行的拟合值都是列均值 $(2,3)$，由此得到所列 $R$ 和 $E_{SSP}$。例如非对角元为 $(-1)(-1)+0(1)+1(0)=1$。$R$ 的第一行指出：第一条观测在两个响应上都低于拟合均值。$2\times2$ 汇总保留交叉乘积总和，却不保留这条观测的身份。

<!-- bilingual-en:start -->
Reordering the rows of $R$ leaves $R^TR$ unchanged. Consequently, residual plots and case-level checks need $R$ together with the observation labels or predictors; $E_{SSP}$ alone cannot recover which case had a large residual. With $80$ observations and two responses, the same distinction is $R:80\times2$ versus $E_{SSP}:2\times2$.
<!-- bilingual-en:end -->

重新排列 $R$ 的行，不会改变 $R^TR$。因此，残差图和逐条观测的检查需要 $R$ 及观测标签或预测变量；仅凭 $E_{SSP}$ 无法恢复哪一条观测残差很大。若有 $80$ 条观测、两个响应，这个区别就是 $R$ 为 $80\times2$，$E_{SSP}$ 为 $2\times2$。

<!-- bilingual-en:start -->
For any response-combination coefficients $a\in\mathbb R^q$, $Ra$ is the residual vector for that combination. Its squared [[欧氏范数|Euclidean length]] is
<!-- bilingual-en:end -->

对任意响应组合系数 $a\in\mathbb R^q$，$Ra$ 是该组合的残差向量。它的[[欧氏范数|欧氏长度]]平方为

$$a^TE_{SSP}a=\|Ra\|_2^2\ge0.$$

<!-- bilingual-en:start -->
This makes $E_{SSP}$ a [[Gram矩阵正定判据|positive-semidefinite Gram matrix]]. It may still be singular: some nonzero combination can have zero residual throughout. The exact dimensional limit is the separate [[误差SSP秩边界|error-SSP rank bound]].
<!-- bilingual-en:end -->

所以 $E_{SSP}$ 是[[Gram矩阵正定判据|半正定的 Gram 矩阵]]，但仍可能奇异：某个非零响应组合的残差可能处处为零。精确的维度限制由[[误差SSP秩边界]]解释。

<!-- bilingual-en:start -->
## From a sum to a covariance estimate
<!-- bilingual-en:end -->

*从平方和到协方差估计*

<!-- bilingual-en:start -->
Let $r=\operatorname{rank}(X)$ be the number of independent fitted directions and let $\nu=n-r$ be the residual degrees of freedom. When $\nu>0$, $E_{SSP}/\nu$ is the usual residual covariance estimate. In the intercept-only example, $r=1$ and $\nu=2$, so this is exactly the [[样本协方差矩阵|sample covariance]] $\begin{bmatrix}1&1/2\\1/2&1\end{bmatrix}$. A general regression removes fitted predictor effects as well as any fitted intercept, so its residual covariance is not simply the covariance of the original response columns.
<!-- bilingual-en:end -->

令 $r=\operatorname{rank}(X)$ 表示独立拟合方向的个数，$\nu=n-r$ 是残差自由度。$\nu>0$ 时，$E_{SSP}/\nu$ 是通常的残差协方差估计。只含截距的例子中，$r=1,\nu=2$，所以它恰好是[[样本协方差矩阵]] $\begin{bmatrix}1&1/2\\1/2&1\end{bmatrix}$。一般回归还扣除了预测变量所拟合的作用，因此残差协方差不等于原始响应列的样本协方差。

<!-- bilingual-en:start -->
Unbiasedness needs a sampling model, but not necessarily normality. For fixed $X$, suppose the true model is $Y=XB+U$ and the error rows have mean zero, common finite [[协方差矩阵|covariance matrix]] $\Sigma$, and are independent across observations. Write $M=I_n-P_X$, where $P_X$ is the [[正交投影|orthogonal projection]] onto the columns of $X$ and $I_n$ leaves every $n$-vector unchanged. Then $R=MU$, $M^TM=M$, and
<!-- bilingual-en:end -->

无偏性需要抽样模型，但不一定需要正态性。固定 $X$，假设真实模型为 $Y=XB+U$，误差行均值为零、具有共同有限[[协方差矩阵]] $\Sigma$，并在观测之间独立。写 $M=I_n-P_X$，其中 $P_X$ 是到 $X$ 列空间的[[正交投影]]，$I_n$ 使每个 $n$ 维向量保持不变。于是 $R=MU$，$M^TM=M$，并且

$$
\mathbb E[E_{SSP}]
=\mathbb E[U^TMU]
=\sum_i M_{ii}\Sigma
=\nu\Sigma.
$$

<!-- bilingual-en:start -->
The cross-row terms vanish by independence and zero means; the diagonal sum of this residual projection equals its rank $\nu$ (the [[迹与特征值总和|trace]] counts its $\nu$ eigenvalues equal to one). This proves unbiasedness after division by $\nu>0$. Zero cross-row covariances would also suffice. Gaussian assumptions provide the stronger [[误差SSP抽样律|exact Wishart sampling law]] and, when the error SSP is positive definite, the distinct maximum-likelihood covariance estimate $E_{SSP}/n$. At $\nu=0$, residual division by $\nu$ is undefined.
<!-- bilingual-en:end -->

跨观测项因独立与零均值而消失；残差投影的对角和等于其秩 $\nu$（[[迹与特征值总和|迹]]计算其 $\nu$ 个等于一的特征值之和）。除以 $\nu>0$ 就证明了无偏性；跨观测协方差为零也已足够。Gaussian 假设进一步给出[[误差SSP抽样律|精确的 Wishart 抽样分布]]，以及误差 SSP 正定时采用另一分母的极大似然协方差估计 $E_{SSP}/n$。$\nu=0$ 时不能除以残差自由度。

> [!warning] Fitted residuals are not the model errors · 拟合残差不是真实模型误差
> <!-- bilingual-en:start -->
> $U=Y-XB$ uses the unknown true coefficients; $R=Y-X\hat B$ uses coefficients fitted from the same data. Even if the rows of $U$ are iid, the rows of $R$ generally are not independent. With an intercept, every residual column sums to zero: changing the residuals in some rows constrains the others. Independence assumptions for the model errors must not be transferred to the fitted residual rows.
> <!-- bilingual-en:end -->
>
> $U=Y-XB$ 使用未知的真实系数；$R=Y-X\hat B$ 使用从同一数据中拟合的系数。即使 $U$ 的行 iid，$R$ 的行通常也不独立。含截距时，每个残差列的总和为零，部分行的残差会约束其余行。不能把真实模型误差的独立假设直接搬给拟合残差行。

## 来源与核验

<!-- bilingual-en:start -->
- [Sawyer, Washington University, *Multivariate Linear Models*, pp. 1–3, especially (3.2b)](https://www.math.wustl.edu/~sawyer/handouts/manovas.pdf#page=3): observation-by-response layout, common-design fitting, and the response-by-response residual SSCP product. Here $q$ denotes the source's response dimension $d$.
- [R official `SSD` documentation](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/SSD.html): distinguishes residual sums of squares and products, residual degrees of freedom, and the estimated covariance matrix.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.2sum.pdf#page=2|MIT 18.06SC, Session 2.2, p. 2]]: OLS projection and orthogonal residuals. The entrywise expectation above verifies the finite-second-moment unbiasedness claim without a Gaussian assumption.
- [R official `summary.manova` documentation](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/summary.manova.html): SSP matrices and residual degrees of freedom appear separately in MANOVA output; the residual rank check also explains why matrix dimensions alone do not justify inversion.
<!-- bilingual-en:end -->

- Sawyer 讲义第 1–3 页，尤其式 (3.2b)：核对观测×响应排列、共享设计拟合和响应×响应残差 SSCP；本卡 $q$ 对应原文响应数 $d$。
- R 官方 `SSD` 文档：核对残差平方和与交叉乘积、残差自由度和协方差估计的区别。
- MIT 第 2.2 节摘要第 2 页：核对 OLS 投影与正交残差；正文逐项取期望，核验不要求 Gaussian 的有限二阶矩无偏结论。
- R 官方 `summary.manova` 文档：核对输出中的 SSP、自由度及残差秩检查。三行数据的残差、乘积和分母在正文直接计算。

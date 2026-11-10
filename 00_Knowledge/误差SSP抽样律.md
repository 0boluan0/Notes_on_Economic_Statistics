---
aliases:
  - Residual SSP sampling law
  - Regression residual Wishart law
  - 回归残差的 Wishart 分布
student_os: knowledge-atom
atom_id: STAT-MVR-011
atom_set: multivariate-linear-regression
atom_type: theorem-boundary
status: source-checked
part_of:
  - "[[多元线性回归.canvas|多元线性回归]]"
requires:
  - "[[多响应残差与误差SSP]]"
  - "[[矩阵正态误差结构]]"
  - "[[Wishart分布]]"
related:
  - "[[误差SSP秩边界]]"
  - "[[Wishart秩与可逆性]]"
---

<!-- bilingual-en:start -->
# The sampling law of the error SSP
<!-- bilingual-en:end -->

*误差 SSP 的抽样分布*

<!-- bilingual-en:start -->
After a regression has fitted the means of several responses, the residuals still fluctuate from sample to sample. Their [[多响应残差与误差SSP|error SSP]] is the matrix of residual sums of squares and cross-products. Its **sampling law** describes that fluctuation under a specified data-generating model. With independent Gaussian model errors, the law is Wishart; its degrees of freedom count the observation-space directions left after fitting.
<!-- bilingual-en:end -->

回归拟合多个响应的均值以后，剩余残差仍会随样本变化。[[多响应残差与误差SSP|误差 SSP]]是残差平方和与交叉乘积组成的矩阵；它的**抽样分布**描述在指定数据生成模型下，这张矩阵怎样变化。模型误差独立且 Gaussian 时，它服从 Wishart 分布，自由度计算拟合后剩余的观测空间方向数。

<!-- bilingual-en:start -->
## State the model before using the distribution
<!-- bilingual-en:end -->

*先说模型，再使用分布*

<!-- bilingual-en:start -->
Use the [[多响应线性回归|shared-design model]] $Y=XB+U$, with $n\ge1$ observations and $q\ge1$ responses. Thus $Y,U$ are real $n\times q$ matrices, $X$ is an $n\times k$ design matrix, and $B$ is a fixed $k\times q$ coefficient matrix. Hold $X$ fixed throughout. For a random design, the same statements are conditional on $X$, provided the following assumptions hold conditional on its realised value.
<!-- bilingual-en:end -->

使用[[多响应线性回归|共享设计模型]] $Y=XB+U$，设有 $n\ge1$ 个观测、$q\ge1$ 个响应。因此 $Y,U$ 是实数 $n\times q$ 矩阵，$X$ 是 $n\times k$ 设计矩阵，$B$ 是固定的 $k\times q$ 系数矩阵。先始终固定 $X$；若设计随机，只要下列条件在给定其实现值后成立，相同结论就理解为条件于 $X$ 的结论。

<!-- bilingual-en:start -->
Assume the error rows, written as $q$-dimensional column vectors $u_i$, are [[独立同分布|iid]] [[联合Gaussian|Gaussian vectors]] $N_q(0,\Sigma)$ with a real [[正定矩阵|positive-definite]] covariance $\Sigma$. The zero error means make $XB$ the correct mean table. In [[矩阵正态误差结构|matrix-normal notation]], this is $Y\mid X\sim MN_{n,q}(XB,I_n,\Sigma)$, where $I_n$ is the identity row factor. The response rows may have different means even though the error rows are iid.
<!-- bilingual-en:end -->

假设将误差行写成 $q$ 维列向量 $u_i$ 后，它们是 [[独立同分布|iid]] 的 [[联合Gaussian|Gaussian 向量]] $N_q(0,\Sigma)$，实协方差矩阵 $\Sigma$ [[正定矩阵|正定]]。误差均值为零，保证 $XB$ 是正确的均值表。用[[矩阵正态误差结构|矩阵正态记号]]，即 $Y\mid X\sim MN_{n,q}(XB,I_n,\Sigma)$，其中 $I_n$ 为单位行因子。即使误差行 iid，响应行仍可具有不同均值。

<!-- bilingual-en:start -->
Fit all response columns by [[共享设计的逐响应OLS|ordinary, unweighted least squares]] with freely chosen coefficient columns. Let $r=\operatorname{rank}(X)$, the number of independent design directions, and let $\nu=n-r$. If $P_X$ is the [[正交投影|orthogonal projector]] onto the column space of $X$, define
<!-- bilingual-en:end -->

对各响应列使用[[共享设计的逐响应OLS|普通、不加权最小二乘]]，各列系数可自由选择。令 $r=\operatorname{rank}(X)$ 为独立设计方向数，$\nu=n-r$。记 $P_X$ 为到 $X$ 列空间的[[正交投影|正交投影矩阵]]，定义

$$R=Y-X\hat B=(I_n-P_X)Y,
\qquad E_{SSP}=R^TR.$$

> [!important] Residual Wishart law · 残差 Wishart 律
> <!-- bilingual-en:start -->
> Under the stated model and fitting rule, if the integer $\nu=n-r\ge1$, then
> $$E_{SSP}\mid X\sim W_q(\Sigma,\nu).$$
> Here $W_q(\Sigma,\nu)$ uses the [[Wishart分布|outer-product convention]]: add $\nu$ independent outer products $z_\ell z_\ell^T$ with $z_\ell\sim N_q(0,\Sigma)$. There is no division by $\nu$ in this definition. Its mean is $\nu\Sigma$.
> <!-- bilingual-en:end -->
>
> 在上述模型与拟合规则下，若整数 $\nu=n-r\ge1$，则 $E_{SSP}\mid X\sim W_q(\Sigma,\nu)$。这里采用 [[Wishart分布|外积和约定]]：将 $\nu$ 个独立 $N_q(0,\Sigma)$ 向量的外积 $z_\ell z_\ell^T$ 相加，定义中不除以 $\nu$，其均值为 $\nu\Sigma$。

<!-- bilingual-en:start -->
The count is $n-r$, not automatically $n-k$: redundant design columns do not consume new directions. When $X$ has full column rank, $r=k$ and the two expressions agree. If $\nu=0$, $P_X=I_n$ and $R=E_{SSP}=0$; treat this as a deterministic zero matrix, separately from the positive-degree Wishart notation above. There is no residual covariance estimate obtained by dividing by zero.
<!-- bilingual-en:end -->

自由度是 $n-r$，不能机械写成 $n-k$，因为冗余设计列不会消耗新方向。$X$ 满列秩时才有 $r=k$。若 $\nu=0$，则 $P_X=I_n$，$R=E_{SSP}=0$；这里将它单独视为确定的零矩阵，不套用上面的正自由度 Wishart 记号，也不能除以零来估计残差协方差。

<!-- bilingual-en:start -->
## Find the independent Gaussian coordinates
<!-- bilingual-en:end -->

*找到真正独立的 Gaussian 坐标*

<!-- bilingual-en:start -->
The fitted residual rows are generally dependent, so we cannot justify the theorem by calling them iid Gaussian. Instead, choose an $n\times\nu$ matrix $Q$ whose columns form a [[标准正交组|unit-length, mutually perpendicular basis]] for the residual space. Then $Q^TQ=I_\nu$ and $I_n-P_X=QQ^T$. Since $(I_n-P_X)XB=0$, the residuals depend on the true errors through
<!-- bilingual-en:end -->

拟合残差的原始各行通常有依赖，不能把它们叫作 iid Gaussian 来证明定理。应选取 $n\times\nu$ 矩阵 $Q$，其列构成残差空间的一组[[标准正交组|长度为一、两两垂直的基]]。于是 $Q^TQ=I_\nu$、$I_n-P_X=QQ^T$。由于 $(I_n-P_X)XB=0$，残差通过下面的变换依赖真实误差：

$$Z=Q^TU\in\mathbb R^{\nu\times q},\qquad
R=QZ,\qquad E_{SSP}=Z^TZ=\sum_{\ell=1}^{\nu}z_\ell z_\ell^T.$$

<!-- bilingual-en:start -->
Here $z_\ell$ is row $\ell$ of $Z$, written as a column. It is a weighted sum of the error vectors. The [[Gaussian仿射闭包|Gaussian affine-closure theorem]] makes all these coordinates jointly Gaussian. For coordinates $\ell,m$,
<!-- bilingual-en:end -->

其中 $z_\ell$ 是 $Z$ 第 $\ell$ 行写成的列向量，是误差向量的加权和。[[Gaussian仿射闭包]]使这些坐标整体联合 Gaussian。对坐标 $\ell,m$，

$$\operatorname{Cov}(z_\ell,z_m\mid X)
=\left(\sum_{i=1}^n Q_{i\ell}Q_{im}\right)\Sigma
=\begin{cases}\Sigma,&\ell=m,\\0,&\ell\ne m.\end{cases}$$

<!-- bilingual-en:start -->
The last equality is exactly the orthonormal-column condition. The [[联合高斯独立判据|Gaussian block-independence criterion]] now makes the $z_\ell$ independent, each with mean zero and covariance $\Sigma$. Thus $Z^TZ$ has the defining Wishart construction. These are $\nu$ residual coordinates; multiplying back by $Q$ produces $n$ fitted residual rows with constraints among them.
<!-- bilingual-en:end -->

最后一个等号恰好使用了列标准正交条件。[[联合高斯独立判据|Gaussian 块独立判据]]因此使 $z_\ell$ 相互独立，每个均值为零、协方差为 $\Sigma$；$Z^TZ$ 就是 Wishart 的定义构造。这里有 $\nu$ 个残差坐标；再乘回 $Q$，才得到彼此带有约束的 $n$ 个拟合残差行。

> [!example] Two observations leave one contrast · 两个观测只剩一个差异方向
> <!-- bilingual-en:start -->
> Fit an intercept to two $q$-response observations. Then $r=1$, $\nu=1$, and a residual-space basis is $Q=(1,-1)^T/\sqrt2$. The one coordinate is $z=(u_1-u_2)/\sqrt2$, which has law $N_q(0,\Sigma)$. The fitted residual rows are $z^T/\sqrt2$ and $-z^T/\sqrt2$, so they are negatives of one another, while
> $$E_{SSP}=zz^T\sim W_q(\Sigma,1).$$
> For $q=2$, this matrix has rank one almost surely. The Wishart law remains meaningful even though its usual nonsingular density and its ordinary inverse do not exist here.
> <!-- bilingual-en:end -->
>
> 对两个、每个含 $q$ 个响应的观测拟合截距，有 $r=1,\nu=1$，可取残差空间基 $Q=(1,-1)^T/\sqrt2$。唯一坐标为 $z=(u_1-u_2)/\sqrt2\sim N_q(0,\Sigma)$；拟合残差两行分别为 $z^T/\sqrt2$ 和 $-z^T/\sqrt2$，恰好互为相反数，而 $E_{SSP}=zz^T\sim W_q(\Sigma,1)$。$q=2$ 时，它几乎必然秩一。Wishart 外积和分布仍有意义，只是这里没有通常的非奇异密度，也没有普通逆。

<!-- bilingual-en:start -->
More generally, the [[误差SSP秩边界|rank bound]] gives rank at most $\min(q,\nu)$ for every data set. Under the positive-definite Gaussian model, equality holds almost surely. Therefore $\nu\ge q$ makes $E_{SSP}$ positive definite almost surely, whereas merely satisfying that dimension inequality does not guarantee invertibility for every fixed numerical sample.
<!-- bilingual-en:end -->

一般地，[[误差SSP秩边界]]对每份数据给出秩至多为 $\min(q,\nu)$；在正定 Gaussian 模型下，等号几乎必然成立。因此 $\nu\ge q$ 使 $E_{SSP}$ 几乎必然正定，但只满足这个维度不等式，并不保证任意一份固定数值样本都可逆。

<!-- bilingual-en:start -->
## Why unbiasedness uses the residual degrees of freedom
<!-- bilingual-en:end -->

*无偏性为何使用残差自由度*

<!-- bilingual-en:start -->
For $\nu>0$, the Wishart mean immediately gives $\mathbb E[E_{SSP}/\nu\mid X]=\Sigma$. This means the estimator averages to the true covariance over repeated samples; it does not say the estimate equals $\Sigma$ in the observed sample. In fact, this unbiasedness conclusion needs less than the entire Wishart law.
<!-- bilingual-en:end -->

$\nu>0$ 时，由 Wishart 均值直接得到 $\mathbb E[E_{SSP}/\nu\mid X]=\Sigma$。这表示重复抽样后取平均会得到真实协方差，不是说本次估计一定等于 $\Sigma$。而且，这个无偏结论所需条件比整条 Wishart 分布律更弱。

<!-- bilingual-en:start -->
Keep the correct mean and fixed design, but assume only finite second moments, zero-mean error rows, common covariance $\Sigma$, and zero cross-row covariances. With $M=I_n-P_X$,
<!-- bilingual-en:end -->

保留正确均值与固定设计，仅假设误差有有限二阶矩、行均值为零、共同协方差为 $\Sigma$，且跨行协方差为零。令 $M=I_n-P_X$，则

$$\mathbb E[E_{SSP}\mid X]
=\mathbb E[U^TMU\mid X]
=\sum_i M_{ii}\Sigma
=\nu\Sigma.$$

<!-- bilingual-en:start -->
Cross-row terms vanish, and the diagonal sum is the [[迹与特征值总和|trace]] of a rank-$\nu$ projection. Thus the same divisor gives unbiasedness without normality or full independence. The identity is valid even for singular $\Sigma$. What is lost without Gaussian assumptions is the automatic **exact Wishart reference distribution**, not this moment calculation.
<!-- bilingual-en:end -->

跨行项消失，对角和是秩为 $\nu$ 的投影矩阵的[[迹与特征值总和|迹]]。因此，即使没有正态性或完整独立性，这个分母仍给出无偏估计；$\Sigma$ 奇异时，期望恒等式也成立。去掉 Gaussian 假设后失去的是自动成立的**精确 Wishart 参考分布**，不是这项矩计算。

<!-- bilingual-en:start -->
## Why Gaussian maximum likelihood uses a different divisor
<!-- bilingual-en:end -->

*Gaussian 极大似然为何采用另一个分母*

<!-- bilingual-en:start -->
Return to independent Gaussian error rows, with an unrestricted unknown positive-definite $\Sigma$ and unrestricted $B$. [[极大似然估计|Maximum likelihood]] chooses parameters that maximize the observed-data joint density. For each fixed $\Sigma$, the [[共享设计的逐响应OLS|shared-design weighted-loss argument]] gives the same OLS fitted means. After optimizing those means, the profile log-likelihood is
<!-- bilingual-en:end -->

回到独立 Gaussian 误差行模型，令未知 $\Sigma$ 遍历所有正定矩阵，$B$ 也不加约束。[[极大似然估计]]选择使已观测数据联合密度最大的参数。对每个固定 $\Sigma$，[[共享设计的逐响应OLS|共享设计的加权损失论证]]给出同一个 OLS 拟合均值；先优化均值后，轮廓对数似然为

$$\ell_{\rm prof}(\Sigma)
=C-\frac n2\log\det\Sigma
-\frac12\operatorname{tr}(\Sigma^{-1}E_{SSP}).$$

<!-- bilingual-en:start -->
Here $C$ does not depend on $\Sigma$, $\det$ is the [[行列式|determinant]], and $\operatorname{tr}$ is the sum of diagonal entries. The coefficient $n/2$ comes from multiplying $n$ Gaussian row densities. If the **observed** $E_{SSP}$ is positive definite, the unique covariance maximizer is $E_{SSP}/n$. The [[Gaussian协方差MLE边界|scatter-matrix likelihood argument]] proves global optimality; it applies to this profile likelihood with the fitted residual scatter in place of centered sample scatter.
<!-- bilingual-en:end -->

其中 $C$ 与 $\Sigma$ 无关，$\det$ 是[[行列式]]，$\operatorname{tr}$ 是对角元之和。系数 $n/2$ 来自 $n$ 个 Gaussian 行密度相乘。若**本次观测到的** $E_{SSP}$ 正定，唯一的协方差最大化值为 $E_{SSP}/n$。[[Gaussian协方差MLE边界|散布矩阵的似然论证]]证明全局最优；这里将其中的中心化样本散布换成拟合残差散布，轮廓似然具有完全相同的形式。

<!-- bilingual-en:start -->
If $E_{SSP}$ is singular, the likelihood is unbounded above: shrink a candidate covariance along a direction with no residual variation. Its determinant tends to zero while the residual penalty in that direction stays zero. Thus $E_{SSP}/n$ is then outside the positive-definite parameter space and is not an attained MLE. The [[误差SSP秩边界|relevant regression threshold]] is residual rank, not the unknown-mean iid rule copied without adjustment. With rank-deficient $X$, the fitted means and covariance MLE can still be unique while multiple coefficient matrices give those same means.
<!-- bilingual-en:end -->

若 $E_{SSP}$ 奇异，似然向上无界：沿一个没有残差变动的方向压缩候选协方差，行列式趋于零，而该方向的残差惩罚仍为零。因此，这时 $E_{SSP}/n$ 在正定参数空间之外，不是已经达到的 MLE。[[误差SSP秩边界|回归中应检查的门槛]]是残差秩，不能不作调整就搬用 iid 未知均值模型的样本量规则。若 $X$ 秩亏，拟合均值和协方差 MLE 仍可能唯一，但多组系数矩阵可以产生相同均值。

> [!example] Eighty observations and six fitted directions · 80 条观测、6 个拟合方向
> <!-- bilingual-en:start -->
> With $n=80$ and $r=6$, $\nu=74$. The unbiased residual covariance estimate is $E_{SSP}/74$; if $E_{SSP}\succ0$, the Gaussian covariance MLE is $E_{SSP}/80$. These are different objectives. Under the Gaussian sampling model with $q\le74$, the MLE exists almost surely and its expectation is $(74/80)\Sigma$. Dividing by $74$ corrects this mean shrinkage; it does not maximize the same likelihood. When $r=0$, the two divisors coincide.
> <!-- bilingual-en:end -->
>
> $n=80,r=6$ 时，$\nu=74$，无偏残差协方差估计为 $E_{SSP}/74$；若 $E_{SSP}\succ0$，Gaussian 协方差 MLE 为 $E_{SSP}/80$。两者回答不同优化问题。在 $q\le74$ 的 Gaussian 抽样模型下，MLE 几乎必然存在，期望为 $(74/80)\Sigma$；除以 $74$ 修正这种均值收缩，却不是极大化同一个似然。$r=0$ 时两个分母相同。

<!-- bilingual-en:start -->
The exact law also depends on the correct mean space and the fitting rule. If the true mean has a component outside the fitted column space, the residual coordinates are no longer centered. If observation rows are correlated, the unadjusted coordinates need not be independent with covariance $\Sigma$. Constraints, penalties or a different row metric likewise need their own derivation; none automatically preserves the displayed central Wishart law.
<!-- bilingual-en:end -->

精确分布还依赖正确的均值空间与拟合规则。若真实均值在拟合列空间之外仍有分量，残差坐标便不再均值为零；若观测行相关，未经调整的坐标也未必独立且同协方差为 $\Sigma$。约束、惩罚或不同的观测行度量同样需要重新推导，都不会自动保留这里的中心 Wishart 分布。

## 来源与核验

<!-- bilingual-en:start -->
- [Sawyer, Washington University, *Multivariate Linear Models*, §§1–3 and 6, pp.1–3,7](https://www.math.wustl.edu/~sawyer/handouts/manovas.pdf): Gaussian error model, OLS and residual Wishart law. The response-sized product is $R^TR$ as in (3.2b); the later reversed-product typo is not used.
- [MIT 6.436J, Lecture14, pp.3–6](https://ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/ceb423b6a703abd60429367b733f45bf_MIT6_436JF18_lec14.pdf#page=3): Gaussian affine closure, covariance and independence. The displayed orthogonal-coordinate calculation derives the Wishart law from these properties.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.2sum.pdf#page=2|MIT 18.06SC, Session2.2, p.2]]: projection and residual orthogonality. [Fan et al., *Statistical Foundations of Data Science*, printed pp.27–28 / PDF43–44](https://fan.princeton.edu/sites/g/files/toruqf5476/files/documents/chapters1-3.pdf#page=43) gives the scalar finite-moment residual-variance calculation and conditional-design interpretation; the entrywise argument above supplies the multivariate extension.
- [SciPy `matrix_normal`, Notes](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.matrix_normal.html): positive-definite Gaussian density and trace quadratic form. Its identity-row case gives the profile likelihood above; existence is checked separately by the scatter-rank argument.
- [R official `SSD` documentation](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/SSD.html): residual SSP, residual degrees of freedom and covariance-estimation objects. The two-observation contrast and $80/74$ calculations are explicit checks.
<!-- bilingual-en:end -->

- Sawyer 第1–3、7页核对 Gaussian 误差、OLS 与残差 Wishart 律；采用式 (3.2b) 的 $R^TR$，未沿用后文乘积转置笔误。
- MIT 概率论第14讲第3–6页核对 Gaussian 变换、协方差和独立性，正文据此推导残差坐标的分布。
- MIT 18.06SC 第2.2节第2页核对投影与正交残差；Fan 等印刷27–28页核对标量有限矩无偏公式与条件设计，正文逐元素推广。
- SciPy 官方密度提供轮廓似然的迹二次型，存在性另由散布秩论证；R 官方文档核对 SSP、自由度与协方差估计对象。两个观测与 $80/74$ 例子直接计算。

---
aliases:
  - Feasible generalized least squares
  - FGLS
student_os: knowledge-atom
atom_id: ECON-ERR-013
atom_set: regression-error-covariance
atom_type: definition
status: source-checked
requires:
  - "[[广义最小二乘]]"
related:
  - "[[异方差稳健协方差]]"
  - "[[HAC协方差]]"
leads_to:
  - "[[AR1准差分]]"
  - "[[误差协方差决策]]"
part_of:
  - "[[异方差与自相关.canvas|异方差与自相关]]"
---
<!-- bilingual-en:start -->
# Feasible generalized least squares
<!-- bilingual-en:end -->

*可行广义最小二乘（FGLS）*

<!-- bilingual-en:start -->
Feasible GLS uses an **estimated error-covariance model** to choose how observations should be weighted or combined. [[广义最小二乘|GLS]] would use the true covariance shape if it were known. FGLS makes that calculation possible by first estimating a structured approximation, then substituting it into the GLS formula. “Feasible” describes the computable procedure; it does not by itself certify its accuracy.
<!-- bilingual-en:end -->

可行 GLS 使用**估计出来的误差协方差模型**，决定如何给观测加权或将它们组合。[[广义最小二乘|GLS]]在协方差形状已知时直接利用它；FGLS 先估计一个有结构的近似，再代入 GLS 公式，使计算成为可能。“可行”说明这个程序能执行，并不自动保证它准确。

> [!example] Estimating two instruments' precision · 先估两类仪器的精度
> <!-- bilingual-en:start -->
> Four readings measure a common mean: instrument A gives $9,11$ and instrument B gives $10,18$. Suppose the working covariance model gives independent errors within and across instruments, with one unknown variance per instrument. The first OLS mean is $12$, so the residuals are $-3,-1,-2,6$. Average their squares within each group: $\hat v_A=(9+1)/2=5$ and $\hat v_B=(4+36)/2=20$.
> <!-- bilingual-en:end -->
>
> 四次读数测量共同均值：A 类仪器给出 $9,11$，B 类给出 $10,18$。工作协方差模型假设仪器内外误差独立，每类仪器各有一个未知方差。第一次 OLS 均值为 $12$，残差为 $-3,-1,-2,6$。按组平均残差平方，得到 $\hat v_A=(9+1)/2=5$、$\hat v_B=(4+36)/2=20$。
>
> <!-- bilingual-en:start -->
> The next fit uses inverse estimated variances, so each A reading receives four times each B reading's weight. The estimated mean becomes $(4\times9+4\times11+10+18)/10=10.8$. These four numbers demonstrate the calculation; two readings per group do not make the estimated variances reliable. The new fitted residuals also need not reproduce exactly the variances used to obtain the weights.
> <!-- bilingual-en:end -->
>
> 第二次拟合使用估计方差的倒数，因此每个 A 类读数的权重是每个 B 类读数的四倍，估计均值变成 $(4\times9+4\times11+10+18)/10=10.8$。这四个数字只演示计算；每组两个观测并不能让方差估计变得可靠。重新拟合后的残差，也不必恰好重现用于构造权重的方差。

<!-- bilingual-en:start -->
## What is estimated before the final regression?
<!-- bilingual-en:end -->

*最终回归之前估计什么*

<!-- bilingual-en:start -->
Write $y=X\beta+u$, where $y$ is the $n$-response vector, $X$ is the real $n\times k$ design matrix, $\beta$ contains $k$ unknown coefficients, and $u$ contains the model errors. Treat $X$ as fixed, or condition on its value. The intended [[条件期望|conditional-mean]] model is $E(u\mid X)=0$: errors average to zero given the design. Assume finite conditional second moments. A usual covariance model is $\operatorname{Cov}(u\mid X)=\sigma^2\Omega(\theta)$: a small parameter vector $\theta$ controls the pattern across all observations, while $\sigma^2>0$ is a common scale.
<!-- bilingual-en:end -->

写成 $y=X\beta+u$，其中 $y$ 为含 $n$ 个响应的向量，$X$ 为实数 $n\times k$ 设计矩阵，$\beta$ 包含 $k$ 个未知系数，$u$ 包含模型误差。将 $X$ 视为固定，或给定其实现值。所要描述的[[条件期望|条件均值]]模型为 $E(u\mid X)=0$，即给定设计后误差平均为零。再假设条件二阶矩有限。常用协方差模型为 $\operatorname{Cov}(u\mid X)=\sigma^2\Omega(\theta)$：用少量参数 $\theta$ 控制所有观测之间的波动结构，$\sigma^2>0$ 为共同尺度。

<!-- bilingual-en:start -->
A typical route is to fit the mean first, often by OLS when its [[OLS一致性条件|consistency conditions]] hold, and form residuals $\hat u_i=y_i-x_i^T\hat\beta_0$. Here $x_i^T$ is row $i$ of $X$, and $\hat\beta_0$ is the initial fit. Fit the covariance model to these residuals, obtaining $\hat\theta$ and $\hat\Omega$. A diagonal model estimates a positive variance function for [[加权最小二乘|WLS]]; an [[AR1准差分|AR(1) model]] estimates a parameter controlling lagged error correlation. One residual per observation is not enough to freely estimate all $n(n+1)/2$ entries of an unrestricted covariance matrix.
<!-- bilingual-en:end -->

典型路线是先拟合均值，例如在[[OLS一致性条件]]成立时使用 OLS，再构造残差 $\hat u_i=y_i-x_i^T\hat\beta_0$；其中 $x_i^T$ 是 $X$ 第 $i$ 行，$\hat\beta_0$ 是初始估计。用这些残差拟合协方差模型，得到 $\hat\theta$ 和 $\hat\Omega$。对角模型为[[加权最小二乘|WLS]]估计正的方差函数；[[AR1准差分|AR(1) 模型]]估计控制误差滞后相关的参数。每个观测只有一个残差，不足以自由估计无结构协方差矩阵的全部 $n(n+1)/2$ 个元素。

<!-- bilingual-en:start -->
If $X$ has [[线性无关|linearly independent]] columns and the fitted matrix $\hat\Omega$ is symmetric [[正定矩阵|positive definite]], the final calculation is
<!-- bilingual-en:end -->

若 $X$ 各列[[线性无关]]，且拟合矩阵 $\hat\Omega$ 对称[[正定矩阵|正定]]，最后计算

$$
\hat\beta_F=(X^T\hat\Omega^{-1}X)^{-1}X^T\hat\Omega^{-1}y.
$$

<!-- bilingual-en:start -->
A common positive scale in $\hat\Omega$ cancels. Nonpositive fitted variances make a diagonal inverse-variance fit invalid; nearly zero ones give enormous weights. A positive variance model or an explicit lower bound can address this computational problem, but the choice must be reported and checked for sensitivity. It does not validate the covariance model.
<!-- bilingual-en:end -->

$\hat\Omega$ 中共同的正常数尺度会抵消。非正的拟合方差不能用于对角逆方差拟合；接近零的方差则会产生极大权重。使用正方差模型或明确的下界，可以处理这个计算问题，但需要报告选择并检查敏感性，它不等于验证了协方差模型。

<!-- bilingual-en:start -->
## Estimating weights changes the statistical argument
<!-- bilingual-en:end -->

*估计权重后，统计论证也变了*

<!-- bilingual-en:start -->
Known GLS uses a matrix fixed given $X$, so the coefficient estimator is linear in $y$ and its mean follows directly from $E(u\mid X)=0$. FGLS usually estimates its weights from the same $y$. Its multiplying matrix then depends on the errors, and finite-sample unbiasedness or the exact Gaussian $t/F$ laws of known-covariance GLS do not automatically carry over. Even Gaussian original errors do not make the random estimated transformation a fixed Gaussian experiment.
<!-- bilingual-en:end -->

已知协方差的 GLS 使用给定 $X$ 后固定的矩阵，因此系数估计量对 $y$ 线性，均值可以直接由 $E(u\mid X)=0$ 推出。FGLS 通常从同一批 $y$ 估计权重，乘在结果前的矩阵因此依赖误差，已知协方差 GLS 的有限样本无偏性或精确高斯 $t/F$ 律不能自动沿用。即使原误差高斯，随机估计出来的变换也不是一个固定变换的高斯实验。

<!-- bilingual-en:start -->
For large-sample [[估计量一致性|consistency]], it is the weighted regression equations that must stabilise. Keep the number of coefficients $k$ fixed as $n$ grows, write $\hat W_n=\hat\Omega_n^{-1}$, and let $Q$ be a fixed positive definite $k\times k$ matrix. The following are explicit sufficient conditions, where $\xrightarrow{p}$ means [[依概率收敛|convergence in probability]]:
<!-- bilingual-en:end -->

大样本[[估计量一致性|一致性]]要求加权回归方程本身稳定。让样本量 $n$ 增长，同时保持系数个数 $k$ 固定；记 $\hat W_n=\hat\Omega_n^{-1}$，并令 $Q$ 为固定正定的 $k\times k$ 矩阵。下面是一组明确的充分条件；$\xrightarrow{p}$ 表示[[依概率收敛]]：

$$
\hat Q_n=\frac1nX_n^T\hat W_nX_n\xrightarrow{p}Q,\qquad
\hat s_n=\frac1nX_n^T\hat W_nu_n\xrightarrow{p}0.
$$

<!-- bilingual-en:start -->
The first keeps the weighted design informative; the second says weighted errors average away. Their role is visible in the exact identity $\hat\beta_F-\beta=\hat Q_n^{-1}\hat s_n$. Zero conditional mean alone does not ensure either limit for arbitrary estimated weights. Saying only that “the covariance estimate is consistent” is also incomplete when its dimension grows with $n$; one must control the weighted equations and the inverse.
<!-- bilingual-en:end -->

第一条使加权设计保有信息，第二条要求加权误差能平均消失。它们的作用直接体现在恒等式 $\hat\beta_F-\beta=\hat Q_n^{-1}\hat s_n$ 中。单有零条件均值，并不能保证任意估计权重满足这两个极限。随着 $n$ 增长，协方差矩阵的维度也增长，因此只说“协方差估计一致”仍不完整，还要控制加权方程和逆矩阵。

<!-- bilingual-en:start -->
In the opening two-instrument model, one sufficient route is [[独立同分布|iid errors]] within each group, independence between the two groups, a common true mean, fixed positive finite group variances, and group sample sizes increasing with nonzero limiting proportions. The [[弱大数定律|law of large numbers]] then makes the initial mean and the group residual-square averages consistent. The estimated weights approach the correct finite inverse variances. This is a justification for that growing-sample model, not evidence supplied by the four illustrative readings.
<!-- bilingual-en:end -->

对开头两类仪器的模型，一条充分路线是：每组内部误差[[独立同分布]]、两组相互独立、共同真实均值、各组方差固定且正而有限，两个组的样本量都增加，所占比例趋于非零。[[弱大数定律|大数定律]]使初始均值和各组残差平方平均一致，估计权重趋于正确且有限的逆方差。这是在为该增长样本模型提供理由，并不是前面四个示例读数已经提供了这些证据。

> [!note]- When estimating the covariance is negligible at first order · 什么时候估计协方差的一阶影响消失
> <!-- bilingual-en:start -->
> Let $W_n=\Omega_n^{-1}$ be the true shape precision and $\hat\beta_G$ the corresponding known-shape GLS estimator. Suppose $Q_n=X_n^TW_nX_n/n\xrightarrow{p}Q\succ0$ and $X_n^TW_nu_n/\sqrt n$ stays bounded in probability: for every small tail probability, a fixed bound eventually contains this random vector with at least that probability's complement. If additionally
> <!-- bilingual-en:end -->
>
> 令 $W_n=\Omega_n^{-1}$ 为真实形状矩阵的逆，$\hat\beta_G$ 为相应已知形状的 GLS。假设 $Q_n=X_n^TW_nX_n/n\xrightarrow{p}Q\succ0$，且 $X_n^TW_nu_n/\sqrt n$ 依概率有界，即对于任意小的尾概率，总能找到固定界限，使这个随机向量最终至少以相应补概率落在界内。若还满足
>
> $$
> \frac1nX_n^T(\hat W_n-W_n)X_n\xrightarrow{p}0,\qquad
> \frac1{\sqrt n}X_n^T(\hat W_n-W_n)u_n\xrightarrow{p}0,
> $$
>
> <!-- bilingual-en:start -->
> then $\sqrt n(\hat\beta_F-\hat\beta_G)\xrightarrow{p}0$. This follows by subtracting the two weighted-equation identities. Under these conditions, FGLS inherits GLS's limiting distribution, so estimating covariance need not add a separate first-order variance term. Consistent estimation of a correctly specified, suitably regular finite-parameter variance model can deliver these conditions; merely fitting some positive matrix does not.
> <!-- bilingual-en:end -->
>
> 那么 $\sqrt n(\hat\beta_F-\hat\beta_G)\xrightarrow{p}0$，可由两个加权方程恒等式相减得到。在这些条件下，FGLS 继承 GLS 的极限分布，因此估计协方差未必需要额外的一阶方差项。正确设定并满足适当正则条件的有限参数方差模型，其一致估计可以满足这些条件；仅仅拟合出某个正定矩阵不够。

<!-- bilingual-en:start -->
## What to conclude from an FGLS fit
<!-- bilingual-en:end -->

*如何理解 FGLS 的结果*

<!-- bilingual-en:start -->
A misspecified covariance model need not destroy consistency for a correctly specified conditional-mean coefficient if the weighted limits above still hold. It can lose the desired efficiency, and its model-based standard errors can be inconsistent. A robust covariance formula must match the fitted estimator and the actual dependence structure; one cannot simply attach an OLS covariance matrix to different FGLS coefficients. If the intended target is only a [[条件均值与线性投影|best linear projection]], reweighting can instead change the population target itself.
<!-- bilingual-en:end -->

若上述加权极限仍成立，协方差模型错设未必破坏正确条件均值系数的一致性；但可能失去期望的效率，基于该错误模型的标准误也可能不一致。稳健协方差公式必须匹配实际估计量及依赖结构，不能直接把 OLS 的协方差矩阵贴到不同的 FGLS 系数上。如果原目标只是[[条件均值与线性投影|最佳线性投影]]，重新加权还可能改变总体目标本身。

<!-- bilingual-en:start -->
Report the covariance form, how it was estimated, the range of weights, and sensitivity to plausible alternatives. Iterating until coefficients stop changing establishes an algorithmic stopping point, not the truth of the model. When the aim is inference about a stable OLS target, [[异方差稳健协方差|HC]] or [[HAC协方差|HAC]] may instead estimate its uncertainty under their own sampling conditions. The choice is a modelling decision, developed in [[误差协方差决策|the error-covariance decision rule]].
<!-- bilingual-en:end -->

报告协方差形式、估计方式、权重范围及对合理替代设定的敏感性。迭代直到系数不再明显变化，只建立了算法停止点，不证明模型真实。若目标是对稳定的 OLS 目标进行推断，可以在各自抽样条件下使用[[异方差稳健协方差|HC]]或[[HAC协方差|HAC]]估计其不确定性。如何选择属于建模判断，见[[误差协方差决策]]。

## 来源与核验

<!-- bilingual-en:start -->
- [Hansen, *Econometrics* (2010)](https://users.ssc.wisc.edu/~xshi28/Econometrics_Hansen.pdf), §7.1, printed pp. 104–107 (PDF pp. 111–114), especially Theorem 7.1.1: residual-based variance modelling, positivity, asymptotic GLS equivalence and covariance misspecification. The weighted-moment conditions above are stated explicitly and checked by the displayed estimator identities.
- [MIT 14.32, *More on FGLS and How to Test for Heteroskedasticity*](https://ocw.mit.edu/courses/14-32-econometrics-spring-2007/c68f54a5cac9d51f45f5916c123a223f_rec_4_20.pdf), PDF pp. 1–2: the residual–variance-model–weighted-fit sequence and the need for positive, credible variance models. Individual residuals are not treated here as consistent estimates of individual standard deviations.
- [Shalizi, *Weighted and Generalized Least Squares*](https://www.stat.cmu.edu/~cshalizi/mreg/15/lectures/24/lecture-24--25.pdf), §§4–7, PDF pp. 14, 19–27: estimating variance functions, iterative fitting and structured covariance models.
- [[02_Economy/01_Econometrics/07_异方差.md#4.2. 可行广义最小二乘 FGLS|Permanent heteroskedasticity notes, §4.2]] and [[02_Economy/01_Econometrics/08_自相关.md#5.2. 科克伦–奥克特 CO 迭代估计|permanent serial-correlation notes, §5.2]]: course sequence. The four-reading illustration is a direct calculation, not a simulation or a claim about finite-sample bias.
<!-- bilingual-en:end -->

- Hansen §7.1 支持残差方差建模、正性、渐近等价及协方差错设边界；本卡明确给出加权矩条件，并按估计量恒等式核验。
- MIT 第 1–2 页支持操作顺序和可信正方差模型要求；本卡没有将单个残差当成单个标准差的一致估计。
- Shalizi 上述章节支持方差函数、迭代拟合及结构化协方差估计；永久课程用于确定操作顺序。四读数例子是直接计算，不是模拟或有限样本偏差结论。

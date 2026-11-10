---
aliases:
  - Wald joint test for linear restrictions
  - 回归 Wald 检验
student_os: knowledge-atom
atom_id: ECON-OLS-021
atom_set: regression-inference
atom_type: procedure
status: source-checked
part_of:
  - "[[回归推断.canvas|回归推断]]"
  - "[[回归模型比较与选择.canvas|回归模型比较与选择]]"
requires:
  - "[[线性组合推断]]"
  - "[[标准误口径匹配]]"
leads_to:
  - "[[回归联合显著边界]]"
related:
  - "[[回归t检验]]"
  - "[[嵌套模型F检验]]"
  - "[[Chow结构稳定性检验]]"
---

<!-- bilingual-en:start -->
# Wald test for linear restrictions
<!-- bilingual-en:end -->

*线性限制的 Wald 联合检验*

<!-- bilingual-en:start -->
A **Wald test** asks whether an unrestricted estimate lies unusually far from a proposed parameter restriction, after accounting for its sampling uncertainty. “Unrestricted” means that we estimate the model before forcing the tested restrictions to hold. For several restrictions, distance depends on how their estimation errors move together: checking each coefficient separately can miss a well-estimated joint direction.

This card treats fixed linear restrictions. A [[线性组合推断|linear combination]] is a weighted sum of coefficients; stacking several such sums lets us ask one joint question, such as whether two slopes are both zero.
<!-- bilingual-en:end -->

**Wald 检验**问的是：考虑抽样不确定性后，无约束估计是否离提出的参数限制异常远。“无约束”表示先估计模型，不强制被检验的限制成立。多个限制共同出现时，距离取决于估计误差怎样一起波动；分别检查每个系数，可能漏掉一个估计得很精确的联合方向。

本卡讨论固定的线性限制。[[线性组合推断|线性组合]]是系数的加权和；把几项组合堆起来，就能提出一个联合问题，例如两个斜率是否同时为零。

> [!example] Opposite movements can be surprising · 反向波动可能更不寻常
> <!-- bilingual-en:start -->
> Suppose two coefficient estimates are $(1,-1)$, and the proposed null sets both coefficients to zero. Both estimated standard errors are $1$, but their estimated covariance is $0.8$:
> $$\widehat S=\begin{bmatrix}1&0.8\\0.8&1\end{bmatrix}.$$
> Positive covariance means the two estimation errors tend to move in the same direction. The departure $(1,-1)$ goes against that shared movement. Standardizing jointly gives a squared distance of $10$. A departure $(1,1)$, with the same two absolute individual t ratios, gives only $10/9$. We will calculate these values below.
>
> These are distances, not yet p-values. A sampling argument must supply the reference distribution before either distance can be called significant.
> <!-- bilingual-en:end -->
>
> 设两个系数估计为 $(1,-1)$，零假设要求两者都为零。两项估计标准误都是 $1$，估计协方差为 $0.8$，矩阵如上。正协方差表示估计误差倾向同向移动；偏离 $(1,-1)$ 却逆着共同方向走，联合标准化后的平方距离为 $10$。换成 $(1,1)$，虽然两个单项 t 比值的绝对值仍一样，平方距离却只有 $10/9$，下文会算出它们。
>
> 这些首先只是距离，不是 p 值。必须由抽样依据提供参考分布，才能进一步判断显著性。

<!-- bilingual-en:start -->
## Specify the restriction and its covariance
<!-- bilingual-en:end -->

*明确限制及其协方差*

<!-- bilingual-en:start -->
Let $\beta\in\mathbb R^k$ be a fixed coefficient vector and $\hat\beta$ its estimate. State the [[零假设与备择假设|null hypothesis]] as
$$H_0:R\beta=r.$$
Here $R$ is a fixed $q\times k$ matrix of weights with $1\le q\le k$, and $r\in\mathbb R^q$ lists fixed proposed values. Its $q$ rows must be [[线性无关|linearly independent]] for $q$ to count distinct restrictions. With $\beta=(\beta_0,\beta_1,\beta_2)^\top$, the rows $(0,1,0)$ and $(0,0,1)$ test $\beta_1=\beta_2=0$; the single row $(0,1,1)$ with $r=1$ instead tests $\beta_1+\beta_2=1$. The alternative to the joint null is that at least one restriction fails.

Write $d=R\hat\beta-r$ for the observed departure and $\widehat V$ for the complete estimated covariance of $\hat\beta$ on its **unscaled estimator** scale. By covariance propagation, the estimated covariance of $d$ is $\widehat S=R\widehat V R^\top$, where $\top$ means [[矩阵转置|transpose]]. Its diagonal entries are estimated variances of the restrictions and its off-diagonal entries their covariances.
<!-- bilingual-en:end -->

令 $\beta\in\mathbb R^k$ 是固定系数向量，$\hat\beta$ 是估计，[[零假设与备择假设|零假设]]写为 $H_0:R\beta=r$。其中 $R$ 是固定的 $q\times k$ 权重矩阵，$1\le q\le k$；$r\in\mathbb R^q$ 列出固定的假设值。$q$ 行须[[线性无关]]，才能让 $q$ 计数不同限制。若 $\beta=(\beta_0,\beta_1,\beta_2)^\top$，两行 $(0,1,0)$、$(0,0,1)$ 检验 $\beta_1=\beta_2=0$；一行 $(0,1,1)$ 配 $r=1$，则检验 $\beta_1+\beta_2=1$。联合零假设的备择是至少一条限制不成立。

用 $d=R\hat\beta-r$ 表示观察到的偏离，$\widehat V$ 表示 $\hat\beta$ **未放大尺度**上的完整估计协方差。传播后，$d$ 的估计协方差为 $\widehat S=R\widehat V R^\top$，$\top$ 表示[[矩阵转置|转置]]。对角元是各限制的估计方差，非对角元是它们之间的协方差。

<!-- bilingual-en:start -->
For the ordinary inverse-based statistic, require $\widehat S$ to be symmetric [[正定矩阵|positive definite]]: every nonzero combination of these restriction errors has positive estimated variance. It is then invertible, and
$$\boxed{W=d^\top\widehat S^{-1}d
=(R\hat\beta-r)^\top[R\widehat V R^\top]^{-1}(R\hat\beta-r).}$$
The inverse weights directions by their precision, so $W\ge0$. If the estimated restriction covariance were diagonal, $W$ would be the sum of their squared standardized departures. Estimated off-diagonal covariances change that calculation; the squared distance uses the whole matrix. Neither the actual finite-sample covariance identity nor the name “Wald” supplies a reference law on its own.
<!-- bilingual-en:end -->

普通求逆形式要求 $\widehat S$ 对称且[[正定矩阵|正定]]：这些限制误差的每个非零组合都有正的估计方差。它于是可逆，Wald 统计量定义如上。逆矩阵按精度给各方向加权，所以 $W\ge0$。若限制协方差估计为对角矩阵，$W$ 就是标准化偏离平方的和；存在非零的估计协方差时，整个矩阵都会参与计算。有限样本的协方差恒等式或“Wald”这个名字，都没有单独提供参考分布。

<!-- bilingual-en:start -->
In the example, $\widehat S^{-1}=\frac1{0.36}\begin{bmatrix}1&-0.8\\-0.8&1\end{bmatrix}$. Thus $d=(1,-1)^\top$ gives $W=(1+1+1.6)/0.36=10$, whereas $d=(1,1)^\top$ gives $(1+1-1.6)/0.36=10/9$. If the covariance were zero with the same unit variances, both distances would be $2$. The original question “can the joint result change while individual standard errors stay the same?” therefore has answer yes: [[回归联合显著边界|joint and separate tests answer different questions]].
<!-- bilingual-en:end -->

例子的逆矩阵为 $\widehat S^{-1}=\frac1{0.36}\begin{bmatrix}1&-0.8\\-0.8&1\end{bmatrix}$。因此 $d=(1,-1)^\top$ 得到 $W=(1+1+1.6)/0.36=10$；$d=(1,1)^\top$ 得到 $(1+1-1.6)/0.36=10/9$。若保持单位方差却把协方差改成零，两者距离都会变成 $2$。所以，即使各项标准误不变，联合结果也可以改变；[[回归联合显著边界|联合与逐项检验回答的是不同问题]]。

> [!warning] Independent restrictions do not guarantee an invertible covariance · 独立限制不保证协方差可逆
> <!-- bilingual-en:start -->
> Full row rank of $R$ alone is insufficient. For example, $R=I_2$ has two independent rows, but $\widehat V=\operatorname{diag}(1,0)$ gives a singular $\widehat S$. Here $I_2$ is the $2\times2$ identity matrix, and $\operatorname{diag}(1,0)$ places $1,0$ on its diagonal. Conversely, selecting only the first coordinate with $R=(1,0)$ gives the positive scalar covariance $1$, even though the full $\widehat V$ is singular.
>
> If a restriction is duplicated, such as rows $a^\top$ and $2a^\top$ with compatible right sides, remove the redundant copy before counting degrees of freedom. Incompatible right sides describe an empty null. Substituting a pseudoinverse into a singular case does not by itself justify a $\chi_q^2$ law; the estimable directions and their limiting distribution require a separate argument.
> <!-- bilingual-en:end -->
>
> 只要求 $R$ 满行秩还不够。例如 $R=I_2$ 有两行独立限制，但 $\widehat V=\operatorname{diag}(1,0)$ 使 $\widehat S$ 奇异。这里 $I_2$ 是二阶单位矩阵，$\operatorname{diag}(1,0)$ 表示对角元依次为 $1,0$。反过来，只用 $R=(1,0)$ 选择第一坐标，就得到正的标量协方差 $1$，尽管整张 $\widehat V$ 奇异。
>
> 若两行分别是 $a^\top$、$2a^\top$ 且右端相容，应先去掉冗余副本再数自由度；不相容的右端描述的是空零假设。遇到奇异情况直接塞入伪逆，不会自动获得 $\chi_q^2$ 分布；哪些方向可估、其极限分布是什么，仍需单独论证。

<!-- bilingual-en:start -->
## Where the large-sample chi-square law comes from
<!-- bilingual-en:end -->

*大样本卡方分布从何而来*

<!-- bilingual-en:start -->
One standard sufficient route uses fixed $k,q,R,r$ as sample size $n$ grows. Suppose the estimator and a symmetric covariance estimator satisfy
$$\sqrt n(\hat\beta_n-\beta)\xrightarrow{d}N_k(0,V_\infty),\qquad
n\widehat V_n\xrightarrow{p}V_\infty,\qquad
S_\infty=RV_\infty R^\top\succ0.$$
The first arrow is [[依分布收敛|convergence in distribution]] to a centered [[联合Gaussian|joint Gaussian vector]]; the second is [[依概率收敛|convergence in probability]] to its fixed covariance matrix. The symbol $\succ0$ means positive definite. The scale matters: $V_\infty$ belongs to the error multiplied by $\sqrt n$, whereas $\widehat V_n$ estimates the covariance on the original estimator scale, usually of order $1/n$.

Under $H_0$, $\sqrt n\,d_n$ therefore converges in distribution to $N_q(0,S_\infty)$, and
$$W_n=(\sqrt n\,d_n)^\top[n\widehat S_n]^{-1}(\sqrt n\,d_n)
\xrightarrow{d}\chi_q^2.$$
[[Slutsky定理|Slutsky's theorem]] permits the consistent matrix estimate to replace the limit. Standardizing a nonsingular Gaussian vector turns the limiting quadratic form into the sum of $q$ independent squared standard normals, the definition of [[卡方分布|chi-square]]. The positive-definite limit makes $\widehat S_n$ positive definite with probability tending to one. If a finite sample is singular or otherwise not positive definite, an explicit convention is needed to define a sequence for this theorem; a fixed value on these vanishing-probability events leaves the limit unchanged, but does not turn that exceptional sample into a valid ordinary Wald test.
<!-- bilingual-en:end -->

一条标准的充分路径是在样本量 $n$ 增大时，保持 $k,q,R,r$ 固定，并满足上面的估计量与对称协方差估计条件。第一个箭头表示[[依分布收敛]]到零均值[[联合Gaussian]]向量，第二个表示[[依概率收敛]]到其固定协方差；$\succ0$ 表示正定。尺度很重要：$V_\infty$ 属于乘了 $\sqrt n$ 的估计误差，$\widehat V_n$ 却估计原估计量尺度的协方差，通常是 $1/n$ 量级。

零假设下，$\sqrt n\,d_n$ 依分布收敛到 $N_q(0,S_\infty)$，因此上式给出 $W_n\xrightarrow{d}\chi_q^2$。[[Slutsky定理]]允许用一致矩阵估计代替极限；把非退化 Gaussian 向量标准化后，极限二次型成为 $q$ 个独立标准正态平方的和，即[[卡方分布]]。正定极限使 $\widehat S_n$ 正定的概率趋于一。若某个有限样本仍奇异或不正定，为应用定理须明确如何在该事件定义统计量序列；在这些概率趋零的事件上指定固定值，不改变极限，却不表示该异常样本已有有效的普通 Wald 检验。

<!-- bilingual-en:start -->
It is not enough to say that two covariance estimates both tend to zero. For instance, if the true covariance is $V_\infty/n$ but one uses $2V_\infty/n$, the absolute difference vanishes while the resulting Wald statistic is half the correctly scaled statistic. Likewise, if a source supplies an estimate $\widehat V_\infty$ of the $\sqrt n$-scale matrix, use $n\,d^\top[R\widehat V_\infty R^\top]^{-1}d$, not the unscaled formula without $n$.

With this asymptotic justification, reject for large $W$: compare with the upper-tail $\chi_q^2$ critical value, or report the corresponding approximate [[p值条件解释|p-value]]. In the two-dimensional example, a valid $\chi_2^2$ approximation would give tail probabilities $e^{-10/2}\approx0.0067$ and $e^{-(10/9)/2}\approx0.574$. These values rely on that approximation, not merely on observing the covariance matrix.
<!-- bilingual-en:end -->

不能只说两张协方差估计都趋于零。例如真实协方差是 $V_\infty/n$，却使用 $2V_\infty/n$，绝对误差虽然趋零，Wald 值仍只有正确尺度的一半。同样，若资料给的是 $\sqrt n$ 尺度矩阵的估计 $\widehat V_\infty$，应使用 $n\,d^\top[R\widehat V_\infty R^\top]^{-1}d$，不能漏掉外面的 $n$。

有了渐近依据后，使用 $\chi_q^2$ 上尾临界值，或报告相应的近似[[p值条件解释|p 值]]。二维例子若有有效的 $\chi_2^2$ 近似，则两个尾概率分别为 $e^{-10/2}\approx0.0067$ 和 $e^{-(10/9)/2}\approx0.574$；这些数依赖那套近似依据，不是看见协方差矩阵就自动成立。

<!-- bilingual-en:start -->
## Separate algebraic identities from exact calibration
<!-- bilingual-en:end -->

*把代数恒等式与精确校准分开*

<!-- bilingual-en:start -->
For one restriction, using the same positive variance estimate as the [[回归t检验|single-contrast t statistic]], $W=t^2$ algebraically. For $q$ restrictions in ordinary least squares with $n>k$ and $s^2>0$, using the same classical covariance $\widehat V=s^2(X^\top X)^{-1}$ gives $W/q$ equal to the [[嵌套模型F检验|nested-model RSS F statistic]]. Here $X$ has $n$ rows and $k$ independent columns, $RSS_U$ is the sum of squared unrestricted residuals, and $s^2=RSS_U/(n-k)$ is the unrestricted residual-variance estimate.

Write $y=X\beta+u$, with response vector $y$ and error vector $u$; $I_n$ below is the identity matrix. For fixed $X$, or conditional on $X$, the joint model $u\mid X\sim N_n(0,\sigma^2I_n)$ with $0<\sigma^2<\infty$, full column rank, $n>k$, and the stated full-row-rank restrictions gives the **exact** null law $W/q\sim F_{q,n-k}$. The identity $W/q=F$ is algebra; these additional Gaussian-model conditions supply one sufficient exact calibration. In this case $W$ itself is generally not an exact chi-square variable. For $q=1$, the upper-tail F test matches the two-sided t test; the square has lost the sign.
<!-- bilingual-en:end -->

只有一条限制，并使用与[[回归t检验|单组合 t 统计量]]相同的正方差估计时，代数上 $W=t^2$。普通最小二乘有 $q$ 条限制且 $n>k$、$s^2>0$ 时，若使用同一经典协方差 $\widehat V=s^2(X^\top X)^{-1}$，则 $W/q$ 等于[[嵌套模型F检验|嵌套模型的 RSS F 统计量]]。这里 $X$ 有 $n$ 行、$k$ 个独立列，$RSS_U$ 是无约束拟合的残差平方和，$s^2=RSS_U/(n-k)$ 是无约束残差方差估计。

把模型写成 $y=X\beta+u$，其中 $y$ 是响应向量，$u$ 是误差向量，下式 $I_n$ 是单位矩阵。在固定 $X$ 或条件于 $X$ 时，联合模型 $u\mid X\sim N_n(0,\sigma^2I_n)$，配合 $0<\sigma^2<\infty$、满列秩、$n>k$ 及已规定的满行秩限制，给出**精确**零假设分布 $W/q\sim F_{q,n-k}$。$W/q=F$ 是代数关系；额外 Gaussian 模型条件提供一套充分的精确校准。这时 $W$ 自身通常并非精确卡方变量。$q=1$ 时，F 上尾检验对应双侧 t 检验，因为平方已丢掉符号。

<!-- bilingual-en:start -->
A covariance estimator allowing heteroskedasticity, serial dependence or clustering may instead support an asymptotic Wald test, provided its own limit and sampling assumptions hold. The correct number of growing independent units may differ from the number of rows. Software may report $W/q$ or another adjusted F form with specified degrees of freedom; division by $q$ alone does not prove an exact F law. Record the covariance estimator, scaling and reference distribution together, following [[标准误口径匹配|the dependence-matching rule]].

Choose $R,r$ before searching the same outcomes for a significant direction. A data-dependent selection needs [[选择后推断|selection-aware calibration]]. Finally, a precisely computed Wald distance cannot repair a coefficient that fails to estimate the intended parameter: an incorrect mean model or a missing identification argument is not cured by changing the covariance matrix.
<!-- bilingual-en:end -->

允许异方差、时间依赖或聚类的协方差估计，在其极限与抽样前提成立时，可以支持渐近 Wald 检验。真正增长的独立单元数可能不同于观测行数。软件可能报告 $W/q$ 或另外调整过的 F 形式及自由度；单纯除以 $q$ 不会证明精确 F 分布。应按照[[标准误口径匹配|依赖结构匹配规则]]，把协方差估计、缩放与参考分布一同记录。

应在同一数据中搜索显著方向之前确定 $R,r$；根据数据挑选限制，需要[[选择后推断|选择后的校准]]。最后，如果系数估计没有指向预期参数，Wald 距离算得再精确也没有修复这个问题：错误均值模型或缺失的识别依据，不会由更换协方差矩阵治好。

## 来源与核验

<!-- bilingual-en:start -->
- [Hansen, *Econometrics* (2010), §5.7, PDF pp.84–86; §6.3, p.98](https://users.ssc.wisc.edu/~xshi28/Econometrics_Hansen.pdf#page=98): transformed parameter covariance, the Wald statistic, its joint-normal limit and covariance consistency. The source uses an asymptotic covariance on the $\sqrt n$ scale; the two equivalent scaling conventions are stated explicitly here. Restriction-covariance positive definiteness is made explicit rather than inferred from row rank alone.
- [Hansen, §4.13, PDF pp.68–70; §6.4, pp.99–100](https://users.ssc.wisc.edu/~xshi28/Econometrics_Hansen.pdf#page=69): conditional Gaussian coefficient/residual independence and the classical RSS/Wald relation. The general fixed affine restriction follows by the restricted least-squares identity explained in the linked F card.
- [[02_Economy/01_Econometrics/05_多元回归模型的矩阵表达.md#4.2. Wald 检验|Permanent course Wald section]] preserves the matrix notation and covariance-matching context. The two-dimensional distances, tail values, singular-covariance examples and factor-of-two scaling example are direct checks.
<!-- bilingual-en:end -->

- Hansen 第 5.7、6.3 节核对变换后的参数协方差、Wald 统计量、联合正态极限与协方差一致性。来源使用 $\sqrt n$ 尺度的渐近协方差，本卡明确区分两套等价缩放；限制协方差的正定性单独列出，不仅凭行秩推断。
- Hansen 第 4.13、6.4 节核对条件 Gaussian 系数/残差独立及经典 RSS/Wald 关系；一般固定仿射限制由链接 F 卡的受限最小二乘恒等式承接。
- 永久课程 Wald 段保留矩阵记号与协方差匹配语境。二维距离、尾概率、奇异协方差与两倍尺度例子均直接核算。

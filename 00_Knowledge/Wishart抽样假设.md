---
aliases:
  - "Sample-covariance Wishart assumptions"
  - "Wishart 精确抽样律的假设边界"
student_os: knowledge-atom
atom_id: STAT-WISH-006
atom_set: wishart-sample-covariance
atom_type: assumption-boundary
status: source-checked
requires:
  - "[[样本协方差Wishart律]]"
part_of:
  - "[[Wishart 分布与样本协方差推断.canvas]]"
related:
  - "[[正态均值协方差独立]]"
---

<!-- bilingual-en:start -->
# Assumptions behind the sample-covariance Wishart law
<!-- bilingual-en:end -->

*样本协方差 Wishart 律的抽样假设*

<!-- bilingual-en:start -->
An estimator's average can be correct while its probability distribution is different from the one used to calculate a confidence interval. For sample covariance, [[样本协方差无偏分母|unbiasedness]] says $E[S]=\Sigma$. The [[样本协方差Wishart律|Wishart sampling law]] says much more: it specifies the entire distribution of $(n-1)S$. Knowing the first statement does not establish the second.
<!-- bilingual-en:end -->

估计量平均而言可以是正确的，但它的概率分布未必是计算置信区间时所采用的那个分布。对于样本协方差，[[样本协方差无偏分母|无偏性]]只说 $E[S]=\Sigma$；[[样本协方差Wishart律|Wishart 抽样律]]则指定整个 $(n-1)S$ 的分布。知道前一条，并不等于证明了后一条。

<!-- bilingual-en:start -->
The standard sufficient assumptions are $n\ge2$ independent observations from the same [[联合Gaussian|joint multivariate Gaussian]] population:
$$X_i\overset{\mathrm{iid}}{\sim}N_p(\mu,\Sigma),\qquad \Sigma\succ0.$$
Each observation is a $p$-coordinate vector, and all observations share the fixed mean $\mu$ and covariance $\Sigma$. With $S$ using denominator $n-1$, these assumptions imply
$$(n-1)S\sim W_p(\Sigma,n-1),\qquad \bar X\ \text{independent of }S.$$
The [[Wishart参数化|notation here puts scale first]]. This is a sufficient route to the theorem, not a claim that no other specially constructed model could give the same law.
<!-- bilingual-en:end -->

一组标准的充分条件是：$n\ge2$ 个观测相互独立，来自同一个[[联合Gaussian|联合多元 Gaussian]]总体，如上式。每个观测都是 $p$ 维向量，共享固定均值 $\mu$ 与协方差 $\Sigma$；若 $S$ 的分母为 $n-1$，就有所列 Wishart 律及均值—协方差独立性。本卡[[Wishart参数化|把尺度写在前面]]。这是得到定理的一条充分路径，不是声称其他特殊构造绝不可能得到相同分布。

> [!example] Same moments, different sampling law · 前两阶矩相同，抽样律不同
> <!-- bilingual-en:start -->
> Let $X_1,X_2$ be independent, each equal to $-1$ or $1$ with probability $1/2$. Each has mean zero and variance one, just like a standard normal. In this one-dimensional example, write $S$ for the sample variance itself:
> $$\bar X=\frac{X_1+X_2}{2},\qquad S=\frac{(X_1-X_2)^2}{2}.$$
> Equal signs occur with probability $1/2$ and give $S=0$; opposite signs occur with probability $1/2$ and give $S=2$. Thus $E[S]=1$, but $S$ takes only two values. The normal-sampling theorem would give the continuous law $\chi_1^2$. Matching means, variances and unbiasedness has not matched the distribution.
> <!-- bilingual-en:end -->
>
> 两次独立观测各以一半概率取 $-1$ 或 $1$，每个的均值、方差都与标准正态相同。本例用 $S$ 直接表示标量样本方差。同号的概率为 $1/2$，此时 $S=0$；异号的概率也为 $1/2$，此时 $S=2$。因此 $E[S]=1$，却只有两个可能值；正态抽样定理会给出连续的 $\chi_1^2$ 律。相同前两阶矩与无偏性没有决定完整分布。

<!-- bilingual-en:start -->
## Where the Gaussian sampling proof uses its assumptions
<!-- bilingual-en:end -->

*证明在哪些地方用到了假设*

<!-- bilingual-en:start -->
The proof rotates the **observation index**, separating a mean direction from $n-1$ zero-sum contrasts. Independence between observations and a common covariance make these contrasts uncorrelated with equal covariance. Joint Gaussianity then turns that zero cross-covariance into independence. The scatter becomes an outer-product sum of independent, zero-mean Gaussian vectors with one common scale, exactly the [[Wishart分布|Wishart construction]].
<!-- bilingual-en:end -->

证明在**观测索引**上改变坐标，把一个均值方向与 $n-1$ 个权重和为零的对比分开。不同观测独立且协方差相同，使这些对比不相关并具有同一协方差；联合 Gaussian 再把零交叉协方差转成独立。最终散布矩阵成为一组独立、零均值、同尺度 Gaussian 向量的外积和，恰好符合[[Wishart分布]]的构造。

<!-- bilingual-en:start -->
The original centered residuals are not themselves independent: they sum to zero even under iid sampling. Conversely, independence of raw observations cannot be replaced by equal moments. If every observation copies the same Gaussian vector $Z$, then all observations have the same Gaussian marginal law, but their centered scatter is identically zero. It cannot have the nondegenerate-scale Wishart law above.
<!-- bilingual-en:end -->

原始中心化残差本身并不独立：即使 iid 抽样，它们也相加为零。另一方面，也不能用相同的矩来代替原始观测之间的独立性。如果每个观测都复制同一个 Gaussian 向量 $Z$，各观测的 Gaussian 边际分布相同，中心化散布却恒等于零，不可能符合上述非退化尺度的 Wishart 律。

<!-- bilingual-en:start -->
Different observation covariances, changing means, temporal or cluster dependence, and non-Gaussian tails or asymmetry can change the contrast distribution. They therefore require a new argument; none is automatically fixed by keeping denominator $n-1$. In particular, “uncorrelated contrasts” is not the same as independent Gaussian contrasts, and marginal normality of each variable is not joint multivariate normality.
<!-- bilingual-en:end -->

观测协方差不同、均值变化、时间或组内依赖、非 Gaussian 尾部或偏态，都可能改变对比的分布，需要重新推导；保留分母 $n-1$ 不会自动修复这些问题。“对比不相关”不等于“对比是独立 Gaussian”，各变量边际正态也不等于联合多元正态。

<!-- bilingual-en:start -->
## Estimation and calibration need separate justifications
<!-- bilingual-en:end -->

*估计与校准分别需要依据*

<!-- bilingual-en:start -->
Non-Gaussian data can still support covariance estimation. Under iid sampling with finite second moments, the usual $S$ is unbiased, as in the two-point example. Consistency, large-sample approximations, resampling, robust covariance methods and dependence-aware models have their own assumptions and targets. They may offer an alternative inferential justification, but they do not retroactively establish the exact finite-sample Wishart law.
<!-- bilingual-en:end -->

非 Gaussian 数据仍然可以估计协方差。iid 且二阶矩有限时，通常的 $S$ 仍无偏，前面的两点分布就是例子。一致性、大样本近似、重抽样、稳健协方差方法和考虑依赖的模型，各有自己的条件与目标；它们可以提供其他推断依据，但不会反过来证明精确的有限样本 Wishart 律。

<!-- bilingual-en:start -->
Before using an inverse covariance or a reference quantile, check both the sampling-law assumptions and [[Wishart秩与可逆性|the rank conditions]]. A sensible covariance estimate does not by itself justify a particular tail probability. [[正态均值协方差独立|Mean–covariance independence]] is another separate distributional property, not a consequence of unbiasedness.
<!-- bilingual-en:end -->

使用逆协方差或参考分位数之前，要同时检查抽样分布条件与[[Wishart秩与可逆性|秩条件]]。协方差估计合理，本身并不能支持某个具体尾概率。[[正态均值协方差独立|均值—协方差独立性]]也是额外的分布性质，并非无偏性的后果。

## 来源与核验

<!-- bilingual-en:start -->
- [Peter Hoff, Multivariate Analysis notes, PDF pp.67–69](https://www2.stat.duke.edu/~pdh10/Teaching/832/Materials/notes.pdf): Gaussian residual contrasts, their independence and the exact Wishart construction.
- [DasGupta, Graduate Probability, Theorem 1.91, PDF p.162, printed p.172](https://www.stat.purdue.edu/~dasgupta/gradprob.pdf#page=162): common multivariate-normal sampling, mean distribution and mean–covariance independence; the source states the density-based Wishart claim for $n>p$.
- [[01_Math/04_多元统计分析/04_多元正态分布The Multivariate Normal Distribution.md#1.6. Wishart 分布|Course §1.6]]: course scope and conventions. The independent-sign and copied-observation counterexamples are computed directly above.
<!-- bilingual-en:end -->

- Hoff 第67–69页核对独立 Gaussian 对比及精确 Wishart 构造；DasGupta 定理1.91核对共同多元正态抽样与独立性，其密度版本的 Wishart 结论要求 $n>p$。
- 课程 §1.6 核对范围与参数约定；独立符号及复制观测反例在正文中直接计算。

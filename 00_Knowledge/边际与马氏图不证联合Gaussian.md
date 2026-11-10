---
aliases:
  - "多元正态性诊断边界"
  - "Limits of Gaussian diagnostic plots"
student_os: knowledge-atom
atom_id: STAT-MVN-006
atom_set: multivariate-normal
atom_type: diagnostic-boundary
status: source-checked
requires:
  - "[[联合Gaussian]]"
  - "[[Gaussian马氏平方律]]"
related:
  - "[[联合Gaussian诊断]]"
  - "[[联合高斯独立判据]]"
  - "[[马氏距离不要求正态]]"
part_of:
  - "[[多元正态分布.canvas]]"
---

<!-- bilingual-en:start -->
# Limits of Gaussian diagnostic plots

A diagnostic plot can display a disagreement with a model, or fail to reveal one. Neither a normal-looking set of margins nor a convincing Mahalanobis plot identifies the full joint distribution. Understanding this boundary makes the plots more useful: each answers a narrower question, and the conclusion should match that question.

For a [[随机向量|random vector]] $X$ with $p$ coordinates, [[联合Gaussian|joint Gaussianity]] means that **every** fixed real projection $a^TX$ is normal, with constants allowed as degenerate cases. Here $a$ is a vector of weights. Margins examine only one coordinate at a time; a squared distance compresses an entire vector into one nonnegative number. These summaries discard different information.
<!-- bilingual-en:end -->

*Gaussian 诊断图的推断边界*

诊断图可以显示某种模型不相容，也可能没有显示出问题。但一组看似正态的边际，或一张令人信服的马氏图，都不能确定完整联合分布。明确这个边界，才能恰当地使用图形：每张图回答的是一个较窄的问题，结论也应限定在那个问题上。

对有 $p$ 个坐标的[[随机向量]] $X$，[[联合Gaussian|联合 Gaussian]] 意味着**每个**固定实投影 $a^TX$ 都正态，常数包括在退化情形中；其中 $a$ 是一组权重。边际每次只看一个坐标，而距离平方把整个向量压成一个非负数，两类摘要丢失的是不同信息。

<!-- bilingual-en:start -->
## Normal margins inspect only a few directions
<!-- bilingual-en:end -->

## 边际正态为什么不够

<!-- bilingual-en:start -->
Checking coordinate $X_j$ is the special case where the weights select that coordinate and set all other weights to zero. It says nothing by itself about a sum or difference of coordinates. This is a logical issue even if the population margins are exactly normal, before any finite-sample plotting uncertainty enters.
<!-- bilingual-en:end -->

检查坐标 $X_j$，只是令权重选中这一坐标、把其余权重设为零的特殊情形。单靠它不能判断坐标和或差的分布。这是逻辑上的不足；即使总体边际确实正态、完全没有有限样本画图误差，也仍然存在。

> [!example] Two normal margins on two lines · 两个正态边际，却集中在两条线上
> <!-- bilingual-en:start -->
> Let $Z\sim N(0,1)$ be [[正态分布|standard normal]], and independently let $B$ be equally likely to be $1$ or $-1$. Define $X=Z$ and $Y=BZ$. Symmetry makes $Y$ standard normal too. Yet when $B=-1$, $X+Y=0$, so
> $$P(X+Y=0)=\tfrac12.$$
> The sum is not constant, and a nondegenerate normal variable has no point masses. Thus the pair is not jointly Gaussian. Its points lie on the two lines $y=x$ and $y=-x$, although both coordinate distributions are normal. The [[联合高斯独立判据|Gaussian independence criterion]] develops this same example's dependence implications.
> <!-- bilingual-en:end -->
>
> 设 $Z\sim N(0,1)$ 为[[正态分布|标准正态]]，另取与它独立、等概率取 $1$ 和 $-1$ 的 $B$，定义 $X=Z$、$Y=BZ$。由对称性，$Y$ 也服从标准正态。但 $B=-1$ 时，$X+Y=0$，因此
> $$P(X+Y=0)=\tfrac12.$$
> 这个和并非恒定，而非退化正态没有单点概率质量，所以这一对变量并不联合 Gaussian。它的点集中在 $y=x$ 与 $y=-x$ 两条线上，尽管两个坐标分布都正态。同一例子的依赖含义见[[联合高斯独立判据]]。

<!-- bilingual-en:start -->
Pairwise scatter plots would reveal this particular problem. In higher dimensions, however, pairwise views and a finite selection of weighted sums still do not inspect every projection. A plot's usefulness for finding a departure does not make passing that plot a sufficient condition for the full model.
<!-- bilingual-en:end -->

成对散点图能够发现上面这个具体问题。但在更高维下，成对视角和有限几个加权和仍未检查所有投影。某种图善于发现偏离，并不意味着“通过这张图”就足以满足完整模型。

<!-- bilingual-en:start -->
## Fitting changes the distance reference
<!-- bilingual-en:end -->

## 样本 Mahalanobis 图为什么不是证明

<!-- bilingual-en:start -->
There are two separate limitations: fitted distances do not have the known-parameter law, and even exact distance information does not describe direction.

The [[Gaussian马氏平方律|Gaussian Mahalanobis law]] states that, for $X\sim N_p(\mu,\Sigma)$ with positive-definite covariance and true fixed population parameters,
$$D^2=(X-\mu)^T\Sigma^{-1}(X-\mu)\sim\chi_p^2.$$
This is a forward implication from the full Gaussian model. If several observations are independent draws from that model, their known-parameter distances are independent too; normality within each vector alone does not establish independence between observations.

A sample plot instead usually uses the [[样本均值向量|sample mean]] $\bar x$ and [[样本协方差矩阵|sample covariance]] $S$ fitted to those same observations:
$$d_i^2=(x_i-\bar x)^TS^{-1}(x_i-\bar x).$$
Every observation helps determine the common center and scale. The distances therefore have fitting constraints and cannot be treated as independent exact $\chi_p^2$ variables. For two distinct scalar observations, $\bar x=(x_1+x_2)/2$ and $S=(x_1-x_2)^2/2$; both fitted squared distances are exactly $1/2$, whatever the two values are. A $\chi_1^2$ variable is not constant.
<!-- bilingual-en:end -->

这里有两项不同限制：拟合距离不再具有已知参数下的分布；即使精确知道距离分布，也还不知道方向。

[[Gaussian马氏平方律]]说明：若 $X\sim N_p(\mu,\Sigma)$，协方差正定，且使用真实、固定的总体参数，则
$$D^2=(X-\mu)^T\Sigma^{-1}(X-\mu)\sim\chi_p^2.$$
这是从完整 Gaussian 模型出发的正向推论。若多次观测还是从该模型独立抽取的，它们的已知参数距离也相互独立；仅仅每个向量内部联合正态，并不能建立不同观测之间的独立性。

实际样本图通常使用由同一批观测拟合的[[样本均值向量|样本均值]] $\bar x$ 与[[样本协方差矩阵|样本协方差]] $S$：
$$d_i^2=(x_i-\bar x)^TS^{-1}(x_i-\bar x).$$
每个观测都参与确定共同的中心与尺度，所以距离受到拟合约束，不能当作相互独立、精确服从 $\chi_p^2$ 的变量。例如，只有两个不同的一维观测时，$\bar x=(x_1+x_2)/2$、$S=(x_1-x_2)^2/2$，无论观测值具体是什么，两个拟合距离平方都恰好为 $1/2$；而 $\chi_1^2$ 变量并不是常数。

<!-- bilingual-en:start -->
The formula also requires $S$ to be invertible. With $n$ observations in $p$ dimensions, the [[样本协方差秩边界|rank bound]] is $\operatorname{rank}(S)\le\min(p,n-1)$ for $n\ge2$. Thus $p\ge n$ forces singularity, and exact affine redundancy can cause singularity even when $p<n$. A pseudoinverse or regularized distance needs separately stated conditions and calibration.
<!-- bilingual-en:end -->

公式还要求 $S$ 可逆。对 $n\ge2$ 个 $p$ 维观测，[[样本协方差秩边界|秩界]]为 $\operatorname{rank}(S)\le\min(p,n-1)$。因此 $p\ge n$ 时必定奇异；即使 $p<n$，精确仿射冗余也能造成奇异。使用伪逆或正则化距离，需要另行说明条件和校准方式。

> [!example]- Correct radii can still hide wrong directions · 距离分布正确，也可能方向错误
> <!-- bilingual-en:start -->
> Take a nonnegative $R$ with $R^2\sim\chi_2^2$. Independently choose $U$ uniformly from $(1,0)^T,(-1,0)^T,(0,1)^T,(0,-1)^T$, and put $W=RU$. The four directions give $E[W]=0$. Since $E[R^2]=2$, each coordinate has variance $2/2=1$ and their product is always zero, so $\operatorname{Cov}(W)=I_2$.
>
> Its true-parameter squared Mahalanobis distance is exactly $W^TW=R^2\sim\chi_2^2$. Nevertheless, $W$ lies on the coordinate axes and its first coordinate is zero with probability $1/2$, so it is not Gaussian. The radius records how far a point is from the center, but loses its direction. The [[卡方分布|chi-square definition and mean]] suffice to verify this construction directly.
> <!-- bilingual-en:end -->
>
> 取非负的 $R$，使 $R^2\sim\chi_2^2$；独立地从 $(1,0)^T,(-1,0)^T,(0,1)^T,(0,-1)^T$ 中等概率选出 $U$，令 $W=RU$。四个方向对称，所以 $E[W]=0$。由于 $E[R^2]=2$，每个坐标的方差为 $2/2=1$，两坐标乘积恒为零，因此 $\operatorname{Cov}(W)=I_2$。
>
> 它使用真实参数的马氏距离平方恰好为 $W^TW=R^2\sim\chi_2^2$，然而 $W$ 集中在两条坐标轴上，第一个坐标以概率 $1/2$ 等于零，所以它不是 Gaussian。半径只记下离中心多远，丢失了方向。这个构造可以用[[卡方分布|卡方定义与均值]]直接核算。

<!-- bilingual-en:start -->
## Non-rejection remains a limited conclusion
<!-- bilingual-en:end -->

## “没有拒绝”仍不是总体证明

<!-- bilingual-en:start -->
A finite sample adds uncertainty to all these views. A [[总体分位数|population quantile]] is a feature of a distribution; a plotted sample value fluctuates between samples. Near-linearity therefore supplies compatibility evidence, not equality of the unknown population distribution to the reference. Nor does a formal test's non-rejection prove its null hypothesis—the distributional claim being tested. The test may have low power: a low probability of detecting the particular departure that matters. Conversely, a large sample may reveal a small discrepancy with little practical effect.

The defensible conclusion states what was checked and what remains open: “No material departure was detected in these margins and selected projections; the fitted distance plot was broadly compatible with the Gaussian reference under the stated sampling assumptions.” Then examine whether the intended estimate, interval, prediction or tail probability changes under reasonable alternatives. The [[联合Gaussian诊断|joint Gaussian diagnostic workflow]] organizes that investigation. A checklist can support a modeling decision; it cannot establish all unexamined directions and tail behavior.
<!-- bilingual-en:end -->

有限样本还给这些视角都加上了不确定性。[[总体分位数]]是分布的特征，而画在图上的样本值会随抽样变化。因此，大致成直线提供的是相容证据，而不是证明未知总体分布等于参照分布。形式检验“不拒绝”也不证明原假设，即正在检验的分布主张。检验的功效可能较低，也就是发现所关心的某种偏离的概率较低；反过来，大样本也可能检出实际影响很小的差别。

可辩护的结论需要说明检查范围与尚未确定之处，例如：“在这些边际和选定投影中，没有发现明显偏离；在所述抽样假设下，拟合距离图与 Gaussian 参照大体相容。”随后检查合理替代选择是否会改变所关心的估计、区间、预测或尾部概率。[[联合Gaussian诊断|联合 Gaussian 诊断流程]]组织了这项调查。检查清单能支持建模决策，却不能确立所有未检查方向和尾部的行为。

## 来源与核验

<!-- bilingual-en:start -->
- [MIT 6.436J Lecture 14, PDF pp. 3–5, Definition 4 and Theorem 1](https://ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/ceb423b6a703abd60429367b733f45bf_MIT6_436JF18_lec14.pdf): joint Gaussianity requires every fixed linear projection to be normal. The two-line example is verified directly and linked to the existing independence-boundary card.
- [Penn State STAT 505, Lesson 4, §4.4](https://online.stat.psu.edu/stat505/Lesson04): fitted Mahalanobis Q–Q plots use an approximate chi-square reference. [[Gaussian马氏平方律|The Gaussian Mahalanobis law]] provides the true-parameter and fitted-parameter distinction and the two-observation calculation.
- [NIST/SEMATECH, Chi-Square Distribution](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3666.htm): independent normal-square construction and mean $2$ for $\chi_2^2$; the four-direction counterexample is derived explicitly above, not quoted as a source theorem.
- [NIST/SEMATECH, Normal Probability Plot](https://www.itl.nist.gov/div898/handbook/eda/section3/normprpl.htm) and [Quantile–Quantile Plot](https://www.itl.nist.gov/div898/handbook/eda/section3/eda33o.htm): graphical distribution comparisons. [Penn State STAT 200, Lesson 6, §§6.1 and 6.5](https://online.stat.psu.edu/stat200/Lesson06): non-rejection, error and power limitations.
<!-- bilingual-en:end -->

- MIT第3–5页核对“所有固定线性投影”的联合 Gaussian 定义；两条直线反例在正文直接验证，并与既有独立性边界卡衔接。
- Penn State STAT 505 §4.4 支持拟合马氏 Q–Q 图的近似卡方参照；真实参数与拟合参数的区别、两个观测的计算见已核验的 Gaussian 马氏平方律。
- NIST卡方条目支持独立正态平方和定义及 $\chi_2^2$ 的均值为 $2$；四方向反例在正文显式推导，没有冒称为来源直接给出的定理。
- NIST图形条目支持分布比较的解释；Penn State STAT 200 支持不拒绝、错误与功效限制。

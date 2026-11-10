---
aliases:
  - Regression t test
  - 单个回归限制检验
student_os: knowledge-atom
atom_id: ECON-OLS-018
atom_set: regression-inference
atom_type: procedure
status: source-checked
part_of:
  - "[[回归推断.canvas|回归推断]]"
requires:
  - "[[标准误口径匹配]]"
  - "[[BLUE正态性边界]]"
leads_to:
  - "[[p值条件解释]]"
  - "[[置信区间覆盖率]]"
related:
  - "[[嵌套模型F检验]]"
  - "[[Wald联合检验]]"
---

<!-- bilingual-en:start -->
# Regression t test
<!-- bilingual-en:end -->

*回归 t 检验*

<!-- bilingual-en:start -->
A regression t test asks whether **one specified coefficient or coefficient combination** is compatible with a proposed value. A fitted slope will usually differ from zero even when the population slope is zero, because the sample contains random noise. The test compares that difference with its [[标准误含义|standard error]]: the estimated scale of the estimate's variation across repeated samples.

For example, in $y_i=\beta_0+\beta_1x_i+u_i$, $\beta_1$ is the change in the model's conditional mean of $y$ for a one-unit increase in $x$. The [[零假设与备择假设|null and two-sided alternative]] $H_0:\beta_1=0$ and $H_1:\beta_1\ne0$ concern this population slope, not whether the fitted line happens to be perfectly horizontal. Decide the target and direction before inspecting which result looks significant.
<!-- bilingual-en:end -->

回归 t 检验问的是：**某一个预先指定的系数或系数组合**，是否与一个提出的数值相容。即使总体斜率为零，样本里的随机噪声通常也会使拟合斜率偏离零。检验把这种偏离与它的[[标准误含义|标准误]]比较：后者估计的是反复抽样时，这个估计量波动的尺度。

例如 $y_i=\beta_0+\beta_1x_i+u_i$ 中，$\beta_1$ 表示 $x$ 增加一个单位时，模型里 $y$ 的条件均值改变多少。[[零假设与备择假设|零假设与双侧备择]] $H_0:\beta_1=0$、$H_1:\beta_1\ne0$ 问的是这个总体系数，不是本次拟合线是否恰好水平。应先确定目标和检验方向，再看结果是否显著。

> [!example] A slope measured in standard-error units · 把斜率偏离换成标准误单位
> <!-- bilingual-en:start -->
> Take five illustrative observations with $x=(0,1,2,3,4)$ and $y=(0,0,1,2,2)$. [[普通最小二乘|Ordinary least squares]] chooses the coefficients that minimize the squared residuals; here it fits $\hat y=-0.2+0.6x$. The [[回归残差|residuals]], observed minus fitted values, are $(0.2,-0.4,0,0.4,-0.2)$, so their squared sum is $RSS=0.4$.
>
> Two coefficients were fitted, leaving $5-2=3$ residual degrees of freedom. The [[OLS经典协方差估计|classical variance estimate]] is $s^2=0.4/3=2/15$. Since $\bar x=2$ and $\sum_i(x_i-\bar x)^2=10$, the slope's standard error is $\sqrt{(2/15)/10}=1/\sqrt{75}\approx0.1155$. For $H_0:\beta_1=0$,
> $$t_{\rm obs}=\frac{0.6-0}{1/\sqrt{75}}=3\sqrt3\approx5.196.$$
> The fitted slope lies about $5.196$ estimated standard errors above zero. This arithmetic alone does not select a reference distribution; the assumptions below supply the classical $t_3$ calibration.
> <!-- bilingual-en:end -->
>
> 取五个示意观测 $x=(0,1,2,3,4)$、$y=(0,0,1,2,2)$。[[普通最小二乘]]选择让残差平方和最小的系数，此处拟合出 $\hat y=-0.2+0.6x$。观测减拟合值的[[回归残差|残差]]为 $(0.2,-0.4,0,0.4,-0.2)$，平方和 $RSS=0.4$。
>
> 估计了两个系数，所以剩 $5-2=3$ 个残差自由度。[[OLS经典协方差估计|经典方差估计]]为 $s^2=0.4/3=2/15$。又有 $\bar x=2$、$\sum_i(x_i-\bar x)^2=10$，因此斜率标准误为 $\sqrt{(2/15)/10}=1/\sqrt{75}\approx0.1155$。检验斜率为零时，得到 $t_{\rm obs}=3\sqrt3\approx5.196$，表示拟合斜率比零高约 $5.196$ 个估计标准误。算出这个数本身还没有指定参考分布；下面的条件才提供经典 $t_3$ 校准。

<!-- bilingual-en:start -->
## State the one restriction
<!-- bilingual-en:end -->

*把要检验的一条限制写清楚*

<!-- bilingual-en:start -->
Write the regression as $y=X\beta+u$. Here $y$ contains $n$ responses; the $n\times k$ design matrix $X$ contains all $k\ge1$ regression terms, including an intercept column if present; $\beta\in\mathbb R^k$ contains the coefficients; and $u$ is the unobserved error vector. Require $X$ to have [[满列秩与OLS唯一性|full column rank]], meaning its columns are [[线性无关|linearly independent]], so the OLS coefficients are uniquely determined. In this card $k$ always includes the intercept; a source that counts only slopes will write a different-looking degrees-of-freedom formula.

A single linear restriction has the form $H_0:a^\top\beta=c$, with a specified nonzero vector $a\in\mathbb R^k$ and specified real number $c$. The superscript $\top$ means [[矩阵转置|transpose]]. Each entry of $a$ is a weight on one coefficient. For $\beta=(\beta_0,\beta_1,\beta_2)^\top$, choosing $a=(0,1,-1)^\top$ tests $\beta_1-\beta_2=c$; choosing $a=(0,1,0)^\top$ selects just $\beta_1$. Given a coefficient covariance estimate $\widehat V$, [[线性组合推断|covariance propagation for a linear combination]] gives
<!-- bilingual-en:end -->

把回归写成 $y=X\beta+u$。$y$ 包含 $n$ 个响应；$n\times k$ 设计矩阵 $X$ 包含全部 $k\ge1$ 个回归项，若有截距也计入；$\beta\in\mathbb R^k$ 是系数向量，$u$ 是不可观测误差。要求 $X$ [[满列秩与OLS唯一性|满列秩]]，即各列[[线性无关]]，使 OLS 系数唯一。本卡的 $k$ 始终包括截距；若资料只把斜率数记为 $k$，自由度公式的写法就会不同。

单个线性限制写成 $H_0:a^\top\beta=c$，其中 $a\in\mathbb R^k$ 是事先指定的非零权重向量，$c$ 是指定实数。上标 $\top$ 表示[[矩阵转置|转置]]；$a$ 的每个元素说明给对应系数多少权重。若 $\beta=(\beta_0,\beta_1,\beta_2)^\top$，$a=(0,1,-1)^\top$ 检验 $\beta_1-\beta_2=c$；$a=(0,1,0)^\top$ 则只选 $\beta_1$。给定系数协方差估计 $\widehat V$，[[线性组合推断|线性组合的协方差传播]]给出

$$
se(a^\top\hat\beta)=\sqrt{a^\top\widehat V a},\qquad
t=\frac{a^\top\hat\beta-c}{\sqrt{a^\top\widehat V a}}.
$$

<!-- bilingual-en:start -->
The numerator is the estimated departure from the null. The denominator is the estimated uncertainty of that **same combination**, which must be positive for this ratio. Off-diagonal covariances matter: the standard error of $\hat\beta_1-\hat\beta_2$ is not generally obtained by adding their standard errors. The numerator and denominator have matching units, so $t$ is dimensionless. A zero estimated denominator does not supply an ordinary t test.
<!-- bilingual-en:end -->

分子是估计值偏离零假设的距离；分母是**同一个组合**的不确定性估计，必须为正才能使用这个比值。非对角协方差也参与计算，因此 $\hat\beta_1-\hat\beta_2$ 的标准误通常不能由两项标准误相加得到。分子、分母单位一致，$t$ 因而无量纲。估计分母为零时，不能当作一次普通 t 检验。

<!-- bilingual-en:start -->
## When the reference law is exactly Student t
<!-- bilingual-en:end -->

*何时有精确的 Student t 参考分布*

<!-- bilingual-en:start -->
One sufficient classical setup is fixed $X$, or inference conditional on $X$, with $n>k$ and the **entire error vector** satisfying $u\mid X\sim N_n(0,\sigma^2I_n)$ for $0<\sigma^2<\infty$. Here $I_n$ is the identity matrix, with ones on its diagonal and zeros elsewhere. This means a correctly specified conditional mean $X\beta$, [[联合Gaussian|joint Gaussian errors]], common variance and zero cross-covariances. Under this joint Gaussian assumption the errors are conditionally independent. Merely having normal-looking marginal outcomes does not state these conditions.

Use $\hat\beta=(X^\top X)^{-1}X^\top y$, residuals $\hat u=y-X\hat\beta$, and the classical estimate
<!-- bilingual-en:end -->

一套充分的经典条件是：固定 $X$，或条件于 $X$ 进行推断；$n>k$；**整个误差向量**满足 $u\mid X\sim N_n(0,\sigma^2I_n)$，其中 $0<\sigma^2<\infty$。这里 $I_n$ 是对角元为一、其余为零的单位矩阵。这包含正确的条件均值 $X\beta$、[[联合Gaussian]]误差、共同方差和零交叉协方差；联合 Gaussian 条件下，各误差才进一步条件独立。观测结果的边际分布“看起来正态”没有说明这些条件。

使用 $\hat\beta=(X^\top X)^{-1}X^\top y$、残差 $\hat u=y-X\hat\beta$，以及经典估计

$$
RSS=\sum_{i=1}^n\hat u_i^2,\qquad
s^2=\frac{RSS}{n-k},\qquad
\widehat V=s^2(X^\top X)^{-1}.
$$

<!-- bilingual-en:start -->
Under $H_0$, the departure $a^\top\hat\beta-c$ divided by its true standard deviation is standard normal; independently, $RSS/\sigma^2$ is [[卡方分布|chi-square]] with $n-k$ degrees of freedom. Replacing $\sigma$ by $s=\sqrt{s^2}$ therefore divides that normal variable by the square root of an independent chi-square variable divided by $n-k$. This is precisely the construction of the [[Student t分布|Student t distribution]], so
<!-- bilingual-en:end -->

零假设下，系数组合的偏离除以其真实标准差后为标准正态；与之独立的 $RSS/\sigma^2$ 服从 $n-k$ 自由度的[[卡方分布]]。把 $\sigma$ 换成 $s=\sqrt{s^2}$，相当于再除以“独立卡方变量除以 $n-k$ 后的平方根”，恰是[[Student t分布]]的构造。因此

$$
t\mid X,H_0\sim t_{n-k}.
$$

<!-- bilingual-en:start -->
Full rank and $a\ne0$ make the true contrast variance positive; $n>k$ leaves positive residual degrees of freedom and gives $s^2>0$ almost surely under this Gaussian model. If $n=k$, there is no residual degree of freedom from which to estimate $\sigma^2$. The [[经典 Gauss–Markov 定理|Gauss–Markov theorem]] alone supplies neither the normal numerator nor its independent chi-square denominator; see [[BLUE正态性边界|the distinct role of normality in OLS inference]].
<!-- bilingual-en:end -->

满秩和 $a\ne0$ 保证组合的真实方差为正；$n>k$ 留下正的残差自由度，并使这个 Gaussian 模型下 $s^2$ 几乎必然为正。若 $n=k$，就没有用于估计 $\sigma^2$ 的残差自由度。[[经典 Gauss–Markov 定理]]本身不提供正态分子，也不提供独立的卡方分母，见[[BLUE正态性边界|正态性在 OLS 推断中的独立作用]]。

<!-- bilingual-en:start -->
## Turn the statistic into a stated decision
<!-- bilingual-en:end -->

*用匹配的参考分布解释统计量*

<!-- bilingual-en:start -->
For a prespecified two-sided level $0<\alpha<1$, reject when $|t_{\rm obs}|>t_{n-k,1-\alpha/2}$, where $t_{\nu,p}$ denotes the lower-tail $p$ quantile of $t_\nu$. Equivalently use the [[p值条件解释|p-value]] $P(|T|\ge|t_{\rm obs}|)$ for a reference $T\sim t_{n-k}$. For the five-observation example, the classical assumptions give $p\approx0.0138$, so a prespecified $5\%$ two-sided test rejects the zero-slope null. It does not establish a causal effect, a large practical effect, or a correct model. Failure to reject would not prove equality either. A directional alternative needs the corresponding one-sided tail, chosen with the question rather than after seeing the sign.

If heteroskedasticity, serial dependence or within-group dependence is allowed, use a [[标准误口径匹配|covariance estimator and calibration suited to that structure]]. A studentized ratio can have an asymptotic standard-normal law when the estimator is asymptotically normal and the standard error consistently estimates the appropriate asymptotic scale, with positive limiting variance. Some methods use justified finite-sample or cluster degrees-of-freedom adjustments instead. The label “robust standard error” alone does not establish any of these results, and it does not preserve the classical $t_{n-k}$ law automatically.
<!-- bilingual-en:end -->

对预先选定的双侧水平 $0<\alpha<1$，当 $|t_{\rm obs}|>t_{n-k,1-\alpha/2}$ 时拒绝；$t_{\nu,p}$ 表示 $t_\nu$ 的下尾 $p$ 分位数。也可以计算参考变量 $T\sim t_{n-k}$ 下的[[p值条件解释|p 值]] $P(|T|\ge|t_{\rm obs}|)$。五观测例子若满足经典条件，$p\approx0.0138$，所以预先指定的 $5\%$ 双侧检验拒绝零斜率。它没有因此证明因果效应、很大的实际效应或模型正确；反过来，不拒绝也不证明相等。方向性的备择须使用相应单侧尾部，并应由问题决定，而不是看见符号后才选择。

若允许异方差、时间依赖或组内依赖，应使用[[标准误口径匹配|与该结构匹配的协方差估计和校准]]。在估计量渐近正态、标准误对相应渐近尺度一致、极限方差为正等条件下，学生化比值可有渐近标准正态分布；某些方法则采用有依据的有限样本或聚类自由度修正。“稳健标准误”这个标签本身没有证明这些结论，也不会自动保留经典 $t_{n-k}$ 分布。

> [!question]- Changing the denominator changes the test · 标准误换了，检验也要跟着换
> <!-- bilingual-en:start -->
> A coefficient gives $t=2.4$ with classical standard errors and $t=1.6$ with cluster standard errors. Can you keep $2.4$ and simply label it “cluster robust”? No: the numerator may be unchanged, but the denominator changed. Use $1.6$ with the reference distribution or degrees-of-freedom rule justified for the cluster procedure.
> <!-- bilingual-en:end -->
>
> 同一系数用经典标准误得到 $t=2.4$，换成聚类标准误得到 $t=1.6$，能否保留 $2.4$ 只改标签？不能。分子可能未变，分母已经改变；应使用 $1.6$，并按聚类程序有依据的参考分布或自由度规则解释。

<!-- bilingual-en:start -->
For the same single restriction, data, OLS fit and classical variance estimate, the [[嵌套模型F检验|nested-model F statistic]] satisfies $F=t^2$. Its upper-tail test matches the **two-sided** t test; squaring loses the sign and does not reproduce a one-sided t test. More generally a one-restriction [[Wald联合检验|Wald statistic]] formed from the same positive estimated contrast variance satisfies $W=t^2$ algebraically. This identity does not decide which reference law or software correction is valid.
<!-- bilingual-en:end -->

同一个单限制、同一数据、同一 OLS 拟合和经典方差估计下，[[嵌套模型F检验|嵌套模型 F 统计量]]满足 $F=t^2$。它的上尾检验对应的是**双侧** t 检验；平方丢掉了符号，不会重现单侧 t 检验。更一般地，只要使用同一个正的组合方差估计，单限制的[[Wald联合检验|Wald 统计量]]在代数上满足 $W=t^2$；这个恒等式不决定参考分布或软件修正是否有效。

## 来源与核验

<!-- bilingual-en:start -->
- [Hansen, *Econometrics* (2010), §4.13, PDF pp.68–70; §5.8, p.86; §§6.1–6.4, pp.96–100](https://users.ssc.wisc.edu/~xshi28/Econometrics_Hansen.pdf#page=69): the conditional Gaussian coefficient/residual decomposition, chi-square and exact t laws, asymptotic studentization, and the single-restriction/Wald/F connection. The text's asymptotic covariance notation is scaled by $n$; $\widehat V$ here estimates the covariance of $\hat\beta$ itself.
- [MIT 14.310x, Lecture 17, PDF pp.53–56](https://ocw.mit.edu/courses/14-310x-data-analysis-for-social-scientists-spring-2023/mit14_310x_s23_week08_lec17.pdf#page=53): the additional normal-error assumption, unknown variance and the $n-2$ denominator for a line with an intercept and slope.
- [[02_Economy/01_Econometrics/02_一元线性回归.md#4.4. 参数的约束检验|Local simple-regression hypothesis tests]] and [[02_Economy/01_Econometrics/03_多元线性回归.md#3.1. 单参数检验：t 检验|local multiple-regression t tests]] preserve course notation and context. The five-point fit, standard error and t-tail probability are independently calculated examples; exact distributional conditions are checked against the original sources above.
<!-- bilingual-en:end -->

- Hansen 对应原文：核对条件 Gaussian 系数/残差分解、卡方与精确 t 分布、渐近学生化，以及单限制、Wald 与 F 的关系。来源的渐近协方差记号带 $n$ 缩放；本卡 $\widehat V$ 估计的是 $\hat\beta$ 自身的协方差。
- MIT 第17讲第53–56页：核对额外正态误差条件、未知方差和截距加斜率时的 $n-2$ 分母。
- 本地一元/多元回归对应段保留课程符号与单参数语境。五点拟合、标准误及 t 尾概率是独立核算的例子；精确分布条件依据上述原始来源核验。

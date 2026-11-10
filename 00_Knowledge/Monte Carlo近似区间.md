---
aliases:
- Monte Carlo normal interval
- Normal confidence interval for a Monte Carlo mean
- Monte Carlo 正态近似区间
student_os: knowledge-atom
atom_id: PROB-MC-023
atom_set: monte-carlo-methods
atom_type: inference-boundary
status: source-checked
part_of:
- '[[Monte Carlo.canvas|Monte Carlo]]'
requires:
- '[[Monte Carlo均值标准误]]'
- '[[经典中心极限定理]]'
- '[[Slutsky定理]]'
related:
- '[[标准误含义]]'
- '[[矩存在性的使用边界]]'
---

<!-- bilingual-en:start -->
# Normal confidence intervals for Monte Carlo means
<!-- bilingual-en:end -->

*Monte Carlo 均值的正态近似区间*

<!-- bilingual-en:start -->
A simulation may report an average such as $5.00$, but a different collection of random draws would usually give a different average. A **Monte Carlo confidence interval** adds a range around the answer to express that sampling uncertainty. This card explains the common large-sample normal interval for the mean of independent outputs: where its width comes from, what its confidence level means, and when a narrow interval can be misleading.
<!-- bilingual-en:end -->

模拟可能报告平均值 $5.00$，但换一批随机抽样，平均值通常会变化。**Monte Carlo 置信区间**在答案周围加上一个范围，表达这份抽样不确定性。本卡讨论独立输出均值的常用大样本正态近似区间：宽度从哪里来，置信水平是什么意思，以及为什么区间很窄仍可能不可靠。

<!-- bilingual-en:start -->
## From simulated outputs to an interval
<!-- bilingual-en:end -->

*从模拟输出算到区间*

<!-- bilingual-en:start -->
Fix the simulation model and the output rule. Let $Y_1,\ldots,Y_N$ be [[独立同分布|iid]] real outputs with mean $\mu$ and $0<\operatorname{Var}(Y_i)=\sigma^2<\infty$. The target $\mu$ is the model's expected output. For a fixed, prespecified integer $N\ge2$, compute the [[Monte Carlo估计|Monte Carlo average]] and sample variance:
$$\widehat\mu_N=\frac1N\sum_{i=1}^N Y_i,\qquad
s_N^2=\frac1{N-1}\sum_{i=1}^N(Y_i-\widehat\mu_N)^2.$$
The sample standard deviation $s_N=\sqrt{s_N^2}$ estimates the spread of individual outputs. The estimated [[Monte Carlo均值标准误|standard error of the average]] is $\widehat{SE}_N=s_N/\sqrt N$. These are different scales: $s_N$ does not itself measure the uncertainty of the average.
<!-- bilingual-en:end -->

固定模拟模型和输出规则。令实数输出 $Y_1,\ldots,Y_N$ [[独立同分布]]，均值为 $\mu$，方差正且有限，即 $0<\sigma^2<\infty$。目标 $\mu$ 是模型的期望输出。预先固定整数 $N\ge2$，按上式计算[[Monte Carlo估计|模拟平均]]和样本方差。样本标准差 $s_N$ 估计单次输出的散布，而平均值的估计[[Monte Carlo均值标准误|标准误]]是 $s_N/\sqrt N$；二者的尺度不同，不能把 $s_N$ 本身当成平均值的不确定性。

<!-- bilingual-en:start -->
Choose a confidence level $1-\alpha$, where $0<\alpha<1$. Let $z_{1-\alpha/2}$ be the $(1-\alpha/2)$ [[总体分位数|quantile]] of the [[正态分布|standard normal distribution]]: it leaves probability $\alpha/2$ above the cutoff. Symmetry leaves the same amount below its negative. The interval is
<!-- bilingual-en:end -->

选择置信水平 $1-\alpha$，其中 $0<\alpha<1$。令 $z_{1-\alpha/2}$ 为[[正态分布|标准正态分布]]的 $1-\alpha/2$ [[总体分位数|分位数]]：截点上方留出 $\alpha/2$ 概率，对称的负截点下方也留出同样概率。区间为

> [!important] Estimate plus or minus a margin · 估计值加减误差半宽
> $$I_N=\left[\widehat\mu_N-z_{1-\alpha/2}\frac{s_N}{\sqrt N},\quad
> \widehat\mu_N+z_{1-\alpha/2}\frac{s_N}{\sqrt N}\right].$$
> <!-- bilingual-en:start -->
> For 95% confidence, $\alpha=0.05$ and $z_{0.975}\approx1.96$. The half-width is $1.96s_N/\sqrt N$; the full width is twice that. The justification is asymptotic under the stated assumptions, not an exact finite-sample guarantee for arbitrary outputs.
> <!-- bilingual-en:end -->
>
> 95% 时，$\alpha=0.05$、$z_{0.975}\approx1.96$。半宽为 $1.96s_N/\sqrt N$，完整宽度是它的两倍。依据是在上述假设下的大样本极限，并不保证任意输出分布、任意有限样本量都精确覆盖 95%。

> [!example] An average of five · 平均值为五
> <!-- bilingual-en:start -->
> Suppose $N=10{,}000$, $\widehat\mu_N=5$ and $s_N=2$. Then $\widehat{SE}_N=2/100=0.02$, the approximate 95% half-width is $1.96(0.02)=0.0392$, and the interval is $[4.9608,5.0392]$. This interval concerns the model mean, not the range containing 95% of individual outputs; the latter have estimated standard deviation $2$, not $0.02$.
> <!-- bilingual-en:end -->
>
> 若 $N=10{,}000$、平均为 $5$、$s_N=2$，估计标准误是 $2/100=0.02$，近似95%半宽是 $0.0392$，区间为 $[4.9608,5.0392]$。它针对模型均值，不是包含95%单次输出的范围；单次输出的估计标准差是 $2$，不是 $0.02$。

<!-- bilingual-en:start -->
## Why the estimated standard deviation can replace the unknown one
<!-- bilingual-en:end -->

*为什么能用样本标准差替代未知尺度*

<!-- bilingual-en:start -->
The [[经典中心极限定理|classical central limit theorem]] states that the mean's standardized error $\sqrt N(\widehat\mu_N-\mu)/\sigma$ approaches a standard normal distribution. The outputs themselves need not be normal. Under the same finite-second-moment assumption, the [[弱大数定律|law of large numbers]] applies both to $Y_i$ and to $Y_i^2$. Indeed,
$$s_N^2=\frac{N}{N-1}\left(\frac1N\sum_iY_i^2-\widehat\mu_N^2\right)
\xrightarrow p E[Y_i^2]-\mu^2=\sigma^2.$$
Thus $s_N\xrightarrow p\sigma>0$. The [[Slutsky定理|Slutsky theorem]] permits substitution of this convergent scale, yielding
$$\frac{\sqrt N(\widehat\mu_N-\mu)}{s_N}\Rightarrow N(0,1).$$
This is called studentization. It does not require a finite fourth moment or independence between $s_N$ and the sample mean. On samples with $s_N=0$, the ratio is undefined; assigning it any fixed value there does not change the limiting law because $P(s_N=0)\to0$ under these assumptions.
<!-- bilingual-en:end -->

[[经典中心极限定理]]说明，平均值的标准化误差 $\sqrt N(\widehat\mu_N-\mu)/\sigma$ 趋向标准正态分布，单次输出本身不必正态。同一有限二阶矩假设下，[[弱大数定律]]同时适用于 $Y_i$ 与 $Y_i^2$，因此上式给出 $s_N^2\xrightarrow p\sigma^2$，继而 $s_N\xrightarrow p\sigma>0$。[[Slutsky定理]]允许替换这个趋于正确常数的尺度，得到第二个式子，称为学生化。这不要求有限四阶矩，也不要求样本标准差与均值独立。若某份样本 $s_N=0$，比值无定义；由于这些假设下 $P(s_N=0)\to0$，在该事件上任意规定一个固定值，不会改变极限分布。

<!-- bilingual-en:start -->
The normal limit gives $P\{\mu\in I_N\}\to1-\alpha$ for a fixed output distribution as $N\to\infty$. This is [[置信区间覆盖率|repeated-simulation coverage]]: rerun the entire procedure and intervals fluctuate around a fixed $\mu$. It is not a posterior probability for $\mu$ inside the one observed interval. Nor does the limit specify a universal sample size at which the approximation is accurate.
<!-- bilingual-en:end -->

正态极限给出：固定输出分布，$N\to\infty$ 时 $P\{\mu\in I_N\}\to1-\alpha$。这是[[置信区间覆盖率|重复模拟覆盖率]]：反复重做整个程序，区间变化，而 $\mu$ 固定；它不是已观察到的这一个区间中参数的后验概率。这个极限也没有提供一个对所有分布通用的“足够大”样本量。

<!-- bilingual-en:start -->
## When the displayed range needs more scrutiny
<!-- bilingual-en:end -->

*什么时候不能只看区间宽度*

> [!warning] Zero observed variation can hide a rare event · 没看见波动，可能只是没遇到稀有事件
> <!-- bilingual-en:start -->
> Let each output be [[Bernoulli分布|Bernoulli]] with success probability $p=0.001$. With $N=1000$, the probability of seeing only zeros is $(0.999)^{1000}\approx0.368$. On those runs, both the sample mean and $s_N$ are zero, so the displayed normal interval collapses to $[0,0]$ and misses the positive true mean. The population variance is nevertheless positive and finite. This illustrates slow finite-sample approximation; observing zero successes does not establish that the model probability is exactly zero.
> <!-- bilingual-en:end -->
>
> 设单次输出服从成功概率 $p=0.001$ 的[[Bernoulli分布]]。$N=1000$ 时，全部抽到零的概率为 $(0.999)^{1000}\approx0.368$。这些模拟中，样本均值与 $s_N$ 都为零，正态区间缩成 $[0,0]$，却漏掉正的真均值；总体方差仍然正且有限。这展示有限样本近似可能很慢，零次成功不能证明模型概率恰好为零。

<!-- bilingual-en:start -->
Strong skewness, heavy tails and rare outputs can delay the approximation. Infinite variance invalidates this finite-variance CLT argument. Correlated simulations, including MCMC, need a variance estimate appropriate to their dependence; counting all draws as independent is not justified. The derivation here is for a fixed planned $N$; a rule that repeatedly checks results and stops based on them needs its own coverage argument.
<!-- bilingual-en:end -->

强偏态、厚尾和稀有输出会拖慢近似。无穷方差会破坏这里的有限方差 CLT 论证。MCMC 等相关模拟需要与依赖结构相符的方差估计，不能把每次抽样都直接计作独立。这里按预先固定的 $N$ 推导；若反复检查结果并据此停机，需要另证该停止规则的覆盖性质。

<!-- bilingual-en:start -->
If the outputs are actually iid normal with positive variance, the studentized mean has an exact [[Student t分布|Student t distribution]] with $N-1$ degrees of freedom. Replacing the normal cutoff by the corresponding t cutoff then gives an exact finite-sample interval under that model. Choosing a t cutoff alone does not make a general nonnormal simulation exact. Both kinds of interval quantify sampling error for the specified model and implementation; [[Monte Carlo误差分层|model, coding and discretization errors]] require separate checks.
<!-- bilingual-en:end -->

若输出确实 iid 正态且方差为正，学生化均值精确服从自由度 $N-1$ 的[[Student t分布]]。此时把正态截点换成对应 t 截点，才得到该模型下的精确有限样本区间；仅仅选用 t 截点，不会让一般非正态模拟也变得精确。两种区间都量化指定模型和实现下的抽样误差，[[Monte Carlo误差分层|模型、代码及离散化错误]]需要另行检查。

## 来源与核验

<!-- bilingual-en:start -->
- [Art B. Owen, Monte Carlo Theory, Methods and Examples, §2.2, pp.18–20](https://artowen.su.domains/mc/Ch-intro.pdf#page=18): sample variance, CLT, consistent scale substitution, normal intervals (2.12–2.14), and the exact normal-data t interval (2.15). §2.4, p.23, explains the all-zero-output failure of a normal probability interval.
- [MIT18.05, Reading6b, §§5.2–5.3, PDF pp.7–8](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/mit18_05_s22_class06-prep-b.pdf#page=7): standardized iid-mean limit, the $\sigma/\sqrt N$ scale, and the 1.96 normal cutoff. The variance-consistency algebra and the two numerical examples above are checked directly.
<!-- bilingual-en:end -->

- Owen 第18–20页核对样本方差、CLT、尺度一致代入、正态区间公式2.12–2.14，以及正态数据下精确t区间2.15；第23页说明全零输出导致正态概率区间失效。
- MIT第7–8页核对 iid 均值的标准化极限、标准误尺度与1.96截点；样本方差一致性的代数及两个数值例子直接计算核验。

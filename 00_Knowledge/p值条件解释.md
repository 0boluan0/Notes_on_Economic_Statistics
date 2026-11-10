---
aliases:
- Conditional interpretation of p-values
- p-value interpretation
- p 值解释
student_os: knowledge-atom
atom_id: STAT-INF-003
atom_type: interpretation-boundary
status: source-checked
part_of:
- '[[回归推断.canvas|回归推断]]'
related:
- '[[回归t检验]]'
- '[[Wald联合检验]]'
- '[[置信区间覆盖率]]'
leads_to:
- '[[显著性解释边界]]'
---

<!-- bilingual-en:start -->
# Interpreting a p-value
<!-- bilingual-en:end -->

*p 值的条件解释*

<!-- bilingual-en:start -->
A **p-value** describes where an observed result lies in the reference distribution supplied by a statistical model and a null hypothesis. It answers a question about what that model would generate on repeated sampling. To read it, we need to know which result was measured, what counted as more extreme, and which assumptions produced the reference distribution.
<!-- bilingual-en:end -->

**p 值**描述一个观察结果在“统计模型加上零假设”给出的参考分布中处于什么位置。它回答的是：如果按这个模型反复取样，会产生怎样的结果。读懂它，需要知道测量了哪个量、怎样算更极端，以及参考分布依赖哪些假设。

<!-- bilingual-en:start -->
## Count the whole tail, not just the observed outcome
<!-- bilingual-en:end -->

*把整个尾部算进去，而不是只算恰好这一次*

<!-- bilingual-en:start -->
Consider ten [[独立同分布|iid]] coin tosses: independent trials with the same heads probability $\theta$. The [[零假设与备择假设|null hypothesis]] $H_0:\theta=1/2$ supplies a fair-coin reference. Let $K$ count heads; this is a [[统计量|statistic]], a number computed from the data. Its null [[抽样分布|sampling distribution]] is [[二项分布|binomial]] with ten trials and probability $1/2$. Observing nine heads does not contradict fairness logically: it is possible, but its tail position may be unusual.
<!-- bilingual-en:end -->

考虑十次[[独立同分布|iid]]硬币抛掷：各次独立，正面概率都是 $\theta$。[[零假设与备择假设|零假设]] $H_0:\theta=1/2$ 提供公平硬币的参考。用 $K$ 计数正面，它是由数据算出的[[统计量]]；零假设下的[[抽样分布]]是试验次数十、成功概率 $1/2$ 的[[二项分布]]。观察到九次正面并没有在逻辑上推翻公平性：它仍然可能发生，但所在尾部可能比较少见。

> [!example] Nine heads in ten tosses · 十次中九次正面
> <!-- bilingual-en:start -->
> For a prespecified upper-tail comparison, nine or more heads means $K=9$ or $K=10$. Of the $2^{10}=1024$ equally likely sequences under the fair-coin model, ten contain nine heads and one contains ten. Thus
> $$p_{\rm upper}=P_{1/2}(K\ge9)=\frac{10+1}{1024}=\frac{11}{1024}\approx0.01074.$$
> The probability of **exactly** nine heads is $10/1024$; it leaves out the still more extreme all-heads outcome and is not this p-value.
> <!-- bilingual-en:end -->
>
> 预先规定的上尾比较，把九次或更多正面算进去，也就是 $K=9$ 或 $K=10$。公平硬币模型下，$2^{10}=1024$ 种序列等可能，其中十种恰含九次正面，一种全为正面，所以这个上尾 p 值为 $11/1024\approx0.01074$。**恰好**九次正面的概率是 $10/1024$；它漏掉了更极端的全正面结果，不是这里的 p 值。

<!-- bilingual-en:start -->
In general, write $t_{\rm obs}$ for the observed value of a chosen statistic $T$, and $T^*$ for that statistic in a hypothetical fresh sample from the null model. If larger values are more extreme, the tail calculation is
$$p=P_{H_0,\mathcal M}(T^*\ge t_{\rm obs}).$$
The symbol $\mathcal M$ records the sampling and model conditions. The subscript says **use the null model to calculate the probability**; it does not assign a probability to the null hypothesis itself. The observed threshold is held fixed while the hypothetical sample varies.
<!-- bilingual-en:end -->

一般地，$t_{\rm obs}$ 是选定统计量 $T$ 的本次观察值，$T^*$ 表示从零假设模型假想重新抽一份样本时得到的同一统计量。若数值越大越极端，就用上式计算尾概率。$\mathcal M$ 记录抽样和模型条件；下标表示**采用零假设模型计算概率**，不是给零假设本身赋予概率。计算时，已经观察到的门槛固定，假想的新样本变化。

<!-- bilingual-en:start -->
## The question determines what counts as more extreme
<!-- bilingual-en:end -->

## “更极端”由检验问题决定

<!-- bilingual-en:start -->
If the question is whether the coin differs from fairness in either direction, use the two-sided discrepancy $|K-5|$. Nine heads has discrepancy four, so the outcomes at least this far from five are $K=0,1,9,10$. Their total probability is
$$p_{\rm two}=\frac{1+10+10+1}{1024}=\frac{11}{512}\approx0.02148.$$
The two answers differ because they answer different directional questions. Here symmetry makes the two-sided value twice the smaller one-sided tail. For asymmetric or discrete null distributions, “double a tail” is not a universal definition: the chosen test must specify its ordering or two-tail convention.
<!-- bilingual-en:end -->

如果问题是硬币在任一方向上是否偏离公平性，就可以用双侧偏离量 $|K-5|$。九次正面离五有四，因此至少同样远的结果是 $K=0,1,9,10$，总概率为 $11/512\approx0.02148$。两个答案不同，是因为方向问题不同。这个例子具有对称性，双侧值恰好是较小单侧尾概率的两倍；面对非对称或离散零假设分布，“尾概率乘二”不是普遍定义，必须说明具体检验的排序或双尾约定。

<!-- bilingual-en:start -->
Likewise, a symmetric [[Student t分布|Student t]] reference often uses $|t|$ for a two-sided coefficient question. An [[F分布|F]] statistic may use only its upper tail even when the underlying question concerns departures in many coefficient directions. “Two-sided” concerns the hypothesis and statistic, not a rule to double every reported number. Choose the direction before seeking a favorable result in the same data.
<!-- bilingual-en:end -->

同样，对称的 [[Student t分布|Student t]] 参考分布通常用 $|t|$ 处理双侧系数问题；[[F分布|F]] 统计量即使检查多个系数方向的偏离，也可能只用上尾。“双侧”涉及假设与统计量，不是把所有报告数字都乘二的规则。应在同一份数据中寻找有利结果之前确定方向。

<!-- bilingual-en:start -->
A composite null permits more than one distribution, so writing $P_{H_0}$ alone does not finish the calibration. For the one-sided coin question $H_0:\theta\le1/2$, $H_1:\theta>1/2$, the probability of $K\ge9$ increases with $\theta$. The fair-coin boundary therefore gives the largest null tail probability, and $11/1024$ is valid for that composite null too. Other tests must justify how they handle unknown nuisance parameters or the other allowed null distributions; arbitrary substitution is not sufficient.
<!-- bilingual-en:end -->

复合零假设允许不止一种分布，所以只写 $P_{H_0}$ 还没有完成校准。对单侧硬币问题 $H_0:\theta\le1/2$、$H_1:\theta>1/2$，$K\ge9$ 的概率随 $\theta$ 增大，因此公平硬币这个边界给出零假设范围内最大的尾概率；$11/1024$ 对这个复合零假设也有效。其他检验同样需要解释怎样处理未知干扰参数或其他允许的零假设分布，不能任意代入一个值就算完成。

<!-- bilingual-en:start -->
## Read the probability in its original direction
<!-- bilingual-en:end -->

*按原来的方向读这个概率*

> [!important] What p = 0.03 says · p = 0.03 说明什么
> <!-- bilingual-en:start -->
> For a correctly calibrated simple-null tail calculation, say: “If the null and the stated model hold, a result at least this extreme under the chosen statistic has probability $3\%$.” This is not “the null has a $3\%$ probability of being true.” A posterior probability of a hypothesis requires an additional model for hypotheses, such as a prior and likelihood; it is not obtained by reversing this tail calculation.
> <!-- bilingual-en:end -->
>
> 对正确校准的简单零假设尾概率，可以说：“若零假设及所用模型成立，按已选统计量得到至少这么极端结果的概率为 $3\%$。”这不是“零假设为真的概率为 $3\%$”。假设的后验概率还需要为假设建立额外的概率模型，例如先验与似然，不能把这里的尾概率方向倒过来就得到。

<!-- bilingual-en:start -->
A small p-value flags a tail discrepancy with the combined null and model. It does not isolate which assumption is responsible, measure an effect in its original units, or give a probability of replication. Nor does a large p-value prove equality: imprecise data can fail to distinguish meaningful departures. Report the effect estimate and its uncertainty as well; [[显著性解释边界|statistical significance and practical importance]] have different roles. A p-value alone also cannot establish a causal interpretation.
<!-- bilingual-en:end -->

小 p 值提示数据与“零假设加模型”的组合存在尾部不相容，却不能单独指出哪项假设出了问题，也不衡量原单位下的效应大小，更不给出复现概率。大 p 值同样不能证明相等：不精确的数据可能分辨不出有意义的偏离。因此还应报告效应估计及不确定性；[[显著性解释边界|统计显著性与实际重要性]]承担不同职责。p 值本身也不能确立因果解释。

<!-- bilingual-en:start -->
## Calibration includes the whole analysis rule
<!-- bilingual-en:end -->

*校准针对整个分析规则*

<!-- bilingual-en:start -->
A valid p-value procedure controls false rejections: for every allowed null distribution and a prespecified threshold $0\le\alpha\le1$,
$$P_{H_0,\mathcal M}\{p(X)\le\alpha\}\le\alpha.$$
Here $X$ denotes the data before observation, and a false rejection means rejecting a true null. In a simple null with a continuous statistic and the exact tail distribution, this probability equals $\alpha$. With discrete data it can be smaller. For the two-sided ten-toss rule above, rejection at $0.05$ occurs only for $K=0,1,9,10$, so its actual null rejection probability is $11/512\approx2.15\%$, not exactly $5\%$.
<!-- bilingual-en:end -->

有效的 p 值程序控制错误拒绝：对每个允许的零假设分布，以及预先选定的门槛 $0\le\alpha\le1$，上式要求把真实零假设拒绝的概率不超过 $\alpha$。$X$ 表示观察前的数据。在简单零假设、连续统计量及精确尾分布下，可以取等号；离散数据则可能更保守。上面的十次抛掷双侧规则在 $0.05$ 下只会因 $K=0,1,9,10$ 而拒绝，因此实际零假设拒绝概率为 $11/512\approx2.15\%$，不是恰好 $5\%$。

<!-- bilingual-en:start -->
A p-value based on a large-sample reference is approximate, so finite-sample error control needs its own justification. Searching many outcomes, models or directions and reporting only the smallest p-value also changes the procedure. For example, if twenty true-null tests have independent uniform p-values, the probability that at least one is at most $0.05$ is $1-0.95^{20}\approx0.642$. The smallest value cannot be read as the result of one prespecified test. These selection issues continue in [[选择后推断|post-selection inference]]; they are not repaired by a smaller printed number.
<!-- bilingual-en:end -->

用大样本参考分布得到的 p 值是近似的，有限样本错误率还需另外论证。搜索多个结果、模型或方向，只报告最小 p 值，也改变了程序。例如，二十个真实零假设检验的 p 值若相互独立且均匀分布，至少一个不超过 $0.05$ 的概率为 $1-0.95^{20}\approx0.642$，不能把最小值按一次预先指定检验来读。选择问题由[[选择后推断]]继续解释，打印出更小的数字不会修复它。

<!-- visual-lab:links:start -->
## 交互理解

- [[07_Programme/可视化实验室/实验/testing|检验、p 值与统计功效]]
<!-- visual-lab:links:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [NIST, “Critical values and p values”](https://www.itl.nist.gov/div898/handbook/prc/section1/prc131.htm): the tail-event definition and a prespecified rejection threshold; its equality shorthand for size is refined here for discrete valid tests.
- [UC Berkeley Stat 210A, “p-values, confidence regions, and (mis-)interpreting Tests”, p-Values sections](https://stat210a.berkeley.edu/fall-2025/reader/testing-interpretation.html): composite-null upper-tail calibration, the test-dependent two-sided convention, and the null inequality $P(p\le\alpha)\le\alpha$.
- [ASA, official 2016 statement announcement, p. 2, principles 1–6](https://www.amstat.org/asa/files/pdfs/P-ValueStatement.pdf#page=2): hypothesis probability, effect size and interpretation/reporting boundaries. This is the official summary of the principles, not the full journal discussion.
- [MIT 14.30, Lecture 20, §2.1, PDF pp. 4–5](https://ocw.mit.edu/courses/14-30-introduction-to-statistical-methods-in-economics-spring-2009/39b3f42fb38a16c2f6c95d53eced4eba_MIT14_30s09_lec20.pdf#page=4): null and alternative sets, maintained sampling assumptions and Type I errors.
- [[02_Economy/01_Econometrics/02_一元线性回归.md#4.4.2. 参数检验|Local regression course, §4.4.2]]: the course's conditional interpretation. The coin counts, discrete rejection probability and twenty-test example are computed directly.
<!-- bilingual-en:end -->

- NIST 对应页核对尾事件定义与预先规定的拒绝门槛；其错误率等号简写在这里对离散有效检验作了精化。
- Berkeley Stat 210A 的 p-Values 各节核对复合零假设上尾校准、依赖具体检验的双侧约定，以及零假设下 $P(p\le\alpha)\le\alpha$。
- ASA 官方声明公告第 2 页六项原则核对假设概率、效应大小以及解释与报告边界；这是原则的官方摘要，不冒充完整期刊论述。
- MIT 第 20 讲 PDF 第 4–5 页核对零/备择集合、抽样前提与第一类错误。本地课程 §4.4.2 保留回归中的条件解释；硬币、离散错误率和二十检验例子直接计算。

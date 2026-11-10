---
aliases:
  - Pillai trace robustness boundary
  - Pillai–Bartlett trace robustness
  - Pillai trace 的稳健性边界
student_os: knowledge-atom
atom_id: STAT-MAN-006
atom_set: manova
atom_type: robustness-boundary
status: source-checked
requires:
  - "[[MANOVA四种统计量]]"
  - "[[MANOVA经典假设]]"
part_of:
  - "[[MANOVA 多元方差分析.canvas|MANOVA 多元方差分析]]"
related:
  - "[[MANOVA检验校准]]"
  - "[[多响应联合显著边界]]"
---

<!-- bilingual-en:start -->
# How robust is Pillai's trace?
<!-- bilingual-en:end -->
Pillai trace 的稳健性边界

<!-- bilingual-en:start -->
Pillai's trace is one of the statistics used by [[MANOVA|MANOVA]] to compare several response means jointly. It is often recommended when deciding among the classical criteria. Here, **more robust** means that its testing behavior can deteriorate less in certain departures from the assumed model. It does not mean that switching to Pillai makes any dataset suitable for the classical test.
<!-- bilingual-en:end -->
Pillai trace 是 [[MANOVA]] 中用来联合比较多个响应均值的一种统计量，在几个经典选项中经常受到推荐。这里的**更稳健**，指模型发生某些偏离时，它的检验表现可能恶化得较少；并不意味着换成 Pillai 后，任何数据都能使用经典检验。

<!-- bilingual-en:start -->
For orientation, let $H$ summarize the fitted hypothesis differences and $E$ the residual variation, as in [[MANOVA SSCP分解|the SSCP decomposition]]. When $E$ is positive definite, let $\lambda_i\ge0$ be the generalized roots comparing $H$ with $E$. The [[MANOVA四种统计量|four-criterion card]] explains how these roots are obtained. Pillai's statistic is
<!-- bilingual-en:end -->
先定位它是什么：用 $H$ 概括拟合出的假设差异，用 $E$ 概括残差变动，具体见[[MANOVA SSCP分解]]。当 $E$ 正定时，设比较 $H$ 与 $E$ 的广义特征根为 $\lambda_i\ge0$；这些根的计算见[[MANOVA四种统计量]]。Pillai 统计量为

$$V=\sum_i\frac{\lambda_i}{1+\lambda_i}.$$

<!-- bilingual-en:start -->
A larger value favors departure from the joint null. Each root contributes less than one, but this algebraic bound alone proves neither resistance to outliers nor valid p-values under arbitrary distributions: unusual observations can change both $H$ and $E$, and the reference distribution still matters. That second step is [[MANOVA检验校准|MANOVA calibration]].
<!-- bilingual-en:end -->
数值越大越倾向于反对联合零假设。每个根的贡献小于一，但仅凭这个代数界，既不能证明它抵抗离群点，也不能证明任意分布下的 p 值有效：异常观测会改变 $H$ 和 $E$，参考分布仍然重要。这后一层属于[[MANOVA检验校准]]。

<!-- bilingual-en:start -->
## What “better” is being measured?
<!-- bilingual-en:end -->
### “表现更好”具体衡量什么

<!-- bilingual-en:start -->
Suppose the population means really satisfy the joint null $H_0$. A **Type I error** occurs when a sample nevertheless leads us to reject it. A nominal level $\alpha=0.05$ is the intended false-rejection limit; the actual rejection probability is determined by the data-generating process and the complete analysis rule. When model assumptions fail, it can exceed $0.05$ even if the program uses that threshold correctly. [[检验功效|Size and power]] also distinguish this error from failing to detect a real difference: power is the rejection probability at a specified alternative.
<!-- bilingual-en:end -->
假设总体均值确实满足联合零假设 $H_0$，却因一次样本而拒绝它，这叫**第一类错误**。名义水平 $\alpha=0.05$ 是预期控制的错误拒绝上限；实际拒绝概率由数据生成机制和完整分析规则决定。模型条件不成立时，即使程序正确使用了0.05阈值，实际错误率也可能超过它。[[检验功效|检验大小与功效]]还要区分另一件事：在某个给定真实差异下，功效是发现差异、拒绝零假设的概率。

<!-- bilingual-en:start -->
A robustness comparison must therefore say which departures were studied, at what sample sizes and group balance, against which competing procedures, and using which error or power measure. Low rejection under the null is not enough by itself: a rule that never rejects has zero false rejections and no power. Similarly, high power gained by rejecting too often under the null is not a free advantage.
<!-- bilingual-en:end -->
因此，比较稳健性时，必须说明研究了什么偏离、样本量与组间平衡如何、对手是什么程序，以及比较错误率还是功效。零假设下拒绝少本身还不够：永远不拒绝的规则虽然没有错误拒绝，却也没有功效。靠零假设下过度拒绝换来的高功效，同样不能当成无代价优势。

<!-- bilingual-en:start -->
## The evidence supports a qualified preference
<!-- bilingual-en:end -->
### 证据支持的是有限定的优先考虑

<!-- bilingual-en:start -->
Olson's 1974 article reports Monte Carlo comparisons within fixed-effects MANOVA under nonnormality and covariance heterogeneity; its abstract recommends Pillai–Bartlett for the cases studied and cautions against the largest-root procedure. R's documentation separately makes Pillai the default and cites Hand and Taylor's recommendation. These are reasons to consider Pillai, not a theorem covering every pattern of dependence, heterogeneity or tail behavior.
<!-- bilingual-en:end -->
Olson在1974年论文中报告了固定效应 MANOVA 下非正态、协方差异质情形的 Monte Carlo 比较；其摘要对所研究情形推荐Pillai–Bartlett，并提醒谨慎使用最大根程序。R文档也把Pillai作为默认值，并引用Hand与Taylor的推荐。这些是考虑Pillai的理由，不是覆盖任意依赖、异质性或尾部行为的定理。

> [!example] Better than rivals can still miss the nominal level / 比对手好，仍可能达不到名义水平
> <!-- bilingual-en:start -->
> Stevens' 1979 comment, p.355, discusses an example with three groups of five observations, six responses, and one group's population variances 36 times those of the others. At nominal $\alpha=0.05$, the reported false-rejection rates are $0.09$ for Pillai, $0.49$ for Wilks, $0.58$ for Hotelling–Lawley and $0.62$ for Roy. Pillai is much closer to $0.05$, yet $0.09$ is still above it. These reported simulation results illustrate relative performance; they are not universal error rates for those tests.
> <!-- bilingual-en:end -->
>
> Stevens在1979年评论第355页讨论了一个例子：三组、每组五个观测、六个响应，其中一组的总体方差为其他组的36倍。在名义 $\alpha=0.05$ 下，文中报告的错误拒绝率为Pillai $0.09$、Wilks $0.49$、Hotelling–Lawley $0.58$、Roy $0.62$。Pillai明显更接近0.05，但0.09依然高于它。这组文献仿真数值说明相对表现，不是各检验通用的错误率。

<!-- bilingual-en:start -->
The comparison can also depend on whether true mean differences are concentrated in one combination or spread across several. Stevens' discussion challenges a universal ranking by considering both robustness and power. This does not establish that another criterion is always preferable; it tells us to match the comparison to the scientific design and target.
<!-- bilingual-en:end -->
比较结果还可能取决于真实均值差集中在一个组合上，还是分散到多个组合。Stevens的讨论同时考虑稳健性与功效，质疑不加范围的统一排序；这并不证明另一个统计量永远更好，而是提醒我们把比较依据与实际设计、目标对应起来。

<!-- bilingual-en:start -->
## What changing the criterion cannot repair
<!-- bilingual-en:end -->
### 更换统计量不能修复什么

<!-- bilingual-en:start -->
Pillai uses the data and hypothesis supplied to it. If repeated rows from the same person are treated as independent observations, changing the criterion does not remove their dependence. If a contrast is not estimable from the design, a different criterion cannot identify it. If the error SSCP matrix $E$ is singular, the usual full-rank reference law cannot be justified merely because some expression involving $H+E$ can still be evaluated. Start with [[MANOVA经典假设|the model, estimability and rank conditions]].
<!-- bilingual-en:end -->
Pillai仍使用输入的数据与假设。同一个人的重复观测若被当作独立行，更换统计量不会消除相关性；设计中不可估的对比，也不会因此变得可识别。误差 SSCP 矩阵 $E$ 奇异时，即使某个含 $H+E$ 的表达式还能计算，也不能据此套用通常的满秩参考分布。应先核对[[MANOVA经典假设|模型、可估性与秩条件]]。

<!-- bilingual-en:start -->
Choose and justify a criterion before examining competing p-values. Looking at all four and reporting the smallest is a new selection rule, not the same prespecified test. A significant joint result also does not imply [[多响应联合显著边界|a separate effect for every response]] or [[多响应因果边界|a causal effect]]. If the classical model is unsuitable, a robust, resampling or high-dimensional method needs its own valid sampling assumptions and calibration; its name alone is insufficient.
<!-- bilingual-en:end -->
应在查看几个竞争性p值之前选择并说明统计量。看完四个结果只报最小值，是新的选择规则，不再是原来预设的一次检验。联合显著也不推出[[多响应联合显著边界|每个响应分别都有差异]]，或[[多响应因果边界|存在因果效应]]。经典模型不适用时，稳健、重抽样或高维方法也必须有各自有效的抽样条件与校准，不能只靠方法名称获得保证。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [Olson (1974), Comparative Robustness of Six Tests in Multivariate Analysis of Variance](https://doi.org/10.1080/01621459.1974.10480224), [accessible article abstract](https://www.researchgate.net/publication/236982019_Comparative_Robustness_of_Six_Tests_in_Multivariate_Analysis_of_Variance): the qualitative recommendation and fixed-effects simulation scope. Verification here is limited to the actual abstract, not the unavailable full simulation tables.
- [Stevens (1979), Comment on Olson](https://doi.org/10.1037/0033-2909.86.2.355), [APA publisher preview, p.355](https://www.researchgate.net/publication/232569893_Comment_on_Olson_Choosing_a_test_statistic_in_multivariate_analysis_of_variance): the reported 36-fold-variance example and the distinction between relative robustness, nominal error and power. The example is read directly from that preview, not reconstructed from later citations.
- [R `summary.manova`, Details](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/summary.manova.html): default/recommended Pillai and approximate F output. [SAS GLM MANOVA](https://support.sas.com/documentation/cdl/en/statug/66103/HTML/default/statug_glm_details45.htm): classical Gaussian error model and criterion definitions.
- [MIT 14.30 Lecture20](https://ocw.mit.edu/courses/14-30-introduction-to-statistical-methods-in-economics-spring-2009/39b3f42fb38a16c2f6c95d53eced4eba_MIT14_30s09_lec20.pdf), printed pp.3–4 / PDF4–5: false rejection and power. Model-repair and selection cautions follow from keeping that rejection rule and its reference assumptions explicit.
<!-- bilingual-en:end -->
- Olson核对范围限于已打开的原文摘要；没有声称重审其全文仿真表。Stevens的例子与比较范围来自已打开的APA第355页预览。
- R与SAS官方文档核对默认值、统计量和校准；MIT第3–4印刷页核对第一类错误与功效。结构条件、选择规则与结论范围分别通过对应精确知识卡片衔接。

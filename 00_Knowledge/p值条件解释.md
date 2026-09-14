---
aliases:
  - "p 值是在原假设与所用模型条件成立时得到至少同样极端统计量的尾概率，而不是原假设为真的概率"
  - A p-value is a conditional tail probability under the null and model rather than the probability that the null is true
  - Conditional interpretation of p-values
student_os: knowledge-atom
atom_id: STAT-INF-003
atom_type: interpretation-boundary
status: source-checked
mastery_state: unassessed
part_of:
  - "[[回归推断.canvas|回归推断]]"
related:
  - "[[回归t检验]]"
  - "[[Wald联合检验]]"
  - "[[置信区间覆盖率]]"
leads_to:
  - "[[显著性解释边界]]"
---

# p 值是在原假设与所用模型条件成立时得到至少同样极端统计量的尾概率，而不是原假设为真的概率
<!-- bilingual-en:start -->
*A p-value is the tail probability of a statistic at least as extreme as the observed one under the null hypothesis and the stated model, not the probability that the null is true*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> 先指定原假设 $H_0$、检验统计量 $T$ 以及什么方向算作“更极端”。若数值越大越不利于 $H_0$，观察值为 $t_{obs}$，则上尾 p 值是
> $$
> p=P_{H_0,\,\mathcal M}\!\left(T\ge t_{obs}\right),
> $$
> 其中 $\mathcal M$ 代表为参考分布提供依据的模型与抽样条件。p 值把 $H_0$ 当作计算前提；它没有计算 $P(H_0\mid data)$。
> <!-- bilingual-en:start -->
> First specify the null hypothesis $H_0$, the statistic $T$, and which directions count as more extreme. If larger values count against $H_0$, the upper-tail p-value is $P_{H_0,\mathcal M}(T\ge t_{obs})$, where $\mathcal M$ denotes the model and sampling conditions that justify the reference distribution. The calculation conditions on $H_0$; it does not calculate $P(H_0\mid data)$.
> <!-- bilingual-en:end -->

## “更极端”由检验问题决定
<!-- bilingual-en:start -->
*The test question determines what counts as more extreme*
<!-- bilingual-en:end -->

双侧检验 $H_1:\theta\ne\theta_0$ 通常把两个方向都计入尾部；有理论依据且预先指定的单侧检验只计算一个方向。不能看到估计方向后才把双侧问题改成单侧，以便缩小 p 值。对 $t$、$F$、Wald 等不同统计量，“更极端”的集合也由各自参考分布定义，不是统一地把原始数据作大小比较。
<!-- bilingual-en:start -->
A two-sided alternative $H_1:\theta\ne\theta_0$ usually counts both tails, whereas a theoretically justified and prespecified one-sided test counts one direction. The direction must not be chosen after observing the estimate merely to obtain a smaller p-value. For $t$, $F$, Wald, and other statistics, the relevant extreme set is defined by the corresponding reference distribution rather than by a universal ordering of the raw data.
<!-- bilingual-en:end -->

较小的 p 值表示：如果 $H_0$ 与这些条件成立，当前统计量所处位置较少见，因而数据与这一整套假设较不相容。它本身不提供以下结论：

- $H_0$ 为真的概率；
- 效应的大小或实际重要性；
- 结果再次研究时复现的概率；
- 模型、标准误口径或因果识别一定正确。

<!-- bilingual-en:start -->
A smaller p-value says that the observed statistic occupies a less common tail position if the null and its stated conditions hold. It does not by itself give the probability that the null is true, the size or practical importance of an effect, the probability of replication, or proof that the model, covariance estimator, and causal identification are correct.
<!-- bilingual-en:end -->

p 值的校准还取决于分析路径。若在许多变量、方向或模型中搜索后只报告最小值，最终数字不再等同于一次预先指定检验的 p 值；这种选择造成的额外不确定性由 [[选择后推断]] 单独处理。
<!-- bilingual-en:start -->
P-value calibration also depends on the analysis path. Searching across many variables, directions, or models and reporting only the smallest value does not produce the p-value of one prespecified test. The additional uncertainty created by that selection is handled in [[选择后推断|post-selection inference]].
<!-- bilingual-en:end -->

> [!question]- 自检
> 某研究报告 $p=0.03$。请写出一句完整而合法的频率学派解释，并指出为什么“原假设为真的概率是 3%”不合法。
> <!-- bilingual-en:start -->
> A study reports $p=0.03$. Give one complete valid frequentist interpretation and explain why “the probability that the null is true is 3%” is invalid.
> <!-- bilingual-en:end -->
>
> **答案：** 合法解释是：“若原假设、所用模型和参考分布条件成立，得到当前这样或更极端统计量的概率为 3%。”非法说法调换了条件概率的方向：p 值计算的是 $P(\text{至少同样极端的数据}\mid H_0,\mathcal M)$，不是 $P(H_0\mid \text{data})$；后者还需要对假设的先验分布与完整概率模型作额外规定。
> <!-- bilingual-en:start -->
> **Answer:** A valid statement is: “If the null, the stated model, and the reference-distribution conditions hold, a statistic at least this extreme has probability 3%.” The invalid statement reverses the conditional probability: a p-value is $P(\text{data at least this extreme}\mid H_0,\mathcal M)$, not $P(H_0\mid \text{data})$. The latter requires additional assumptions such as a prior distribution over hypotheses and a complete probability model.
> <!-- bilingual-en:end -->

## 来源与核验

- [[02_Economy/01_Econometrics/02_一元线性回归.md#4.4.2. 参数检验|本地课程：p 值]]：核对“在原假设下得到当前或更极端统计量的概率”这一条件解释。
- NIST/SEMATECH, [Hypothesis testing and p-values](https://itl.nist.gov/div898/handbook/prc/section1/prc131.htm)：核验 p 值、显著性水平与拒绝规则的定义。
- MIT OpenCourseWare 14.30, [Lecture Notes 20](https://ocw.mit.edu/courses/14-30-introduction-to-statistical-methods-in-economics-spring-2009/39b3f42fb38a16c2f6c95d53eced4eba_MIT14_30s09_lec20.pdf)：核验原假设、备择、拒绝域与 Type I/II error 的频率学派框架。
<!-- bilingual-en:start -->
- The local course note, NIST/SEMATECH handbook, and MIT 14.30 notes support the conditional tail-probability interpretation and its calibration within a prespecified testing procedure.
<!-- bilingual-en:end -->

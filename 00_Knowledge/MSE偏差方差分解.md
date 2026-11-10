---
student_os: knowledge-atom
atom_id: 71c13e1d-a0fe-40c0-ae66-2056c0e72534
status: source-checked
aliases:
  - "Bias–variance decomposition of MSE"
  - "MSE bias-variance decomposition"
  - "均方误差的偏差方差分解"
---

<!-- bilingual-en:start -->
# Bias–variance decomposition of MSE
<!-- bilingual-en:end -->

*均方误差的偏差—方差分解*

<!-- bilingual-en:start -->
An estimator can miss its target because its average answer is displaced, because its answers fluctuate, or both. The **bias–variance decomposition** separates these two contributions to [[均方误差|mean squared error]]. It explains why an unbiased rule can still have large errors, and why accepting some bias can sometimes improve overall squared-error performance.
<!-- bilingual-en:end -->

估计量偏离目标，可能因为平均答案就有偏移，也可能因为每次答案波动很大，或者两者兼有。**偏差—方差分解**把这两种来源在[[均方误差]]中的贡献分开，解释为什么无偏规则仍可能误差很大，以及为什么接受一些偏差有时反而能改善总平方误差。

<!-- bilingual-en:start -->
Let $\tau$ be a fixed finite real target, and let $T$ be an estimator with $E[T^2]<\infty$ under the stated sampling model. Put $m=E[T]$, the estimator's own sampling mean. Its [[估计量偏差|bias]] is $b=m-\tau$, and its [[方差|variance]] is $E[(T-m)^2]$. Then
<!-- bilingual-en:end -->

固定有限实数目标为 $\tau$，估计量 $T$ 在指定抽样模型下满足 $E[T^2]<\infty$。令 $m=E[T]$ 为它自己的抽样均值，[[估计量偏差|偏差]]为 $b=m-\tau$，[[方差]]为 $E[(T-m)^2]$。于是

> [!important] Total squared error has two nonnegative parts · 两个非负部分
> $$\underbrace{E[(T-\tau)^2]}_{\operatorname{MSE}(T;\tau)}
> =\underbrace{E[(T-m)^2]}_{\operatorname{Var}(T)}
> +\underbrace{(m-\tau)^2}_{\operatorname{Bias}(T;\tau)^2}.$$
> <!-- bilingual-en:start -->
> Bias is squared. A negative and positive bias of the same magnitude contribute equally. All terms use the same target and sampling model; for a parameter-dependent target $g(\theta)$, hold $\theta$ fixed and replace $\tau$ by $g(\theta)$.
> <!-- bilingual-en:end -->
>
> 偏差必须平方，同样大小的正偏差与负偏差贡献相同。各项使用同一目标与抽样模型；若目标为 $g(\theta)$，就固定 $\theta$，再把 $\tau$ 换成 $g(\theta)$。

^core

<!-- bilingual-en:start -->
## Why the cross term disappears
<!-- bilingual-en:end -->

*为什么中间的交叉项消失*

<!-- bilingual-en:start -->
Split the estimation error at the estimator's own mean:
$$T-\tau=(T-m)+(m-\tau).$$
The first term is the random deviation around $m$; the second is the fixed displacement of $m$ from the target. Squaring and taking [[期望|expectations]] gives
$$E[(T-\tau)^2]=E[(T-m)^2]+2(m-\tau)E[T-m]+(m-\tau)^2.$$
By [[期望线性性|linearity]], $E[T-m]=E[T]-m=0$, so the middle term vanishes. The other terms are exactly variance and squared bias. No independence or normality assumption is used in this algebra.
<!-- bilingual-en:end -->

在估计量自己的均值处分开误差：$T-\tau=(T-m)+(m-\tau)$。第一项是在 $m$ 周围的随机偏离，第二项是 $m$ 与目标之间的固定距离。平方后取[[期望]]得到上式；由[[期望线性性]]，$E[T-m]=E[T]-m=0$，所以交叉项为零，剩下方差与偏差平方。这个代数推导不需要独立或正态假设。

> [!example] A biased rule can have smaller MSE · 有偏规则的 MSE 可以更小
> <!-- bilingual-en:start -->
> At one model value, set the target to $10$. Rule $T$ returns $10$ or $14$ with probability $1/2$ each. Its mean is $12$, bias is $2$, variance is $4$, and
> $$\operatorname{MSE}(T)=2^2+4=8=\frac{(10-10)^2+(14-10)^2}{2}.$$
> Rule $U$ returns $6$ or $14$ with equal probabilities. It has mean $10$, bias $0$ and variance $16$, so its MSE is $16$. The biased rule has the smaller MSE here because its reduction in variance outweighs its squared bias. This comparison is at the stated model value; it does not establish superiority at every possible target.
> <!-- bilingual-en:end -->
>
> 固定一个模型值，目标为 $10$。规则 $T$ 等概率给出 $10$ 或 $14$，均值 $12$、偏差 $2$、方差 $4$，所以 MSE 为 $2^2+4=8$。规则 $U$ 等概率给出 $6$ 或 $14$，均值 $10$、偏差零、方差 $16$，MSE 反而为 $16$。前者减少的方差超过新增的偏差平方，所以这里有偏规则更好；但这并没有证明它在所有可能目标处都更好。

<!-- bilingual-en:start -->
## What follows from the identity
<!-- bilingual-en:end -->

*这个恒等式能告诉我们什么*

<!-- bilingual-en:start -->
If the rule is [[估计量无偏性|unbiased]] at this model value, MSE equals variance. Its [[标准误含义|standard error]] then equals RMSE; without zero bias, RMSE is at least the standard error. Zero variance alone does not imply zero MSE: a rule that always returns $12$ for target $10$ has variance zero and MSE four. Its output is perfectly stable and systematically wrong.
<!-- bilingual-en:end -->

若规则在这个模型值处[[估计量无偏性|无偏]]，MSE 就等于方差，此时[[标准误含义|标准误]]等于 RMSE；偏差不为零时，RMSE 至少等于标准误。仅有零方差不保证零 MSE：目标为 $10$ 而规则总返回 $12$，方差为零，MSE 却为四；答案完全稳定，仍然有系统误差。

<!-- bilingual-en:start -->
For a sequence $T_n$ with finite second moments at each $n$, MSE tends to zero **if and only if** both its variance and squared bias tend to zero, because both summands are nonnegative. Vanishing MSE then implies [[估计量一致性|consistency]] through a probability bound. The reverse implication from consistency to vanishing MSE is false in general: rare huge errors can prevent moments from shrinking.
<!-- bilingual-en:end -->

对每个 $n$ 都有有限二阶矩的序列 $T_n$，MSE 趋零**当且仅当**方差与偏差平方都趋零，因为两个加项都非负。MSE 趋零再通过概率界推出[[估计量一致性|一致性]]；反过来，一致性一般不保证 MSE 趋零，罕见但巨大的误差可以阻止矩缩小。

<!-- bilingual-en:start -->
This identity itself does not say that reducing bias must increase variance; that depends on the family of rules being compared. It also does not add an “irreducible noise” term. Such a term appears in a prediction-error decomposition when the target is a new noisy outcome and the required conditioning assumptions are supplied. Here the target is fixed, and the displayed two-term identity is the complete decomposition for estimation error.
<!-- bilingual-en:end -->

恒等式本身没有说“减少偏差必然增加方差”，这取决于所比较的规则族。这里也没有额外的“不可约噪声”项；若预测对象是新的带噪声结果，并给出相应条件化假设，预测误差分解才会出现该项。本卡目标固定，显示的两项就是估计误差的完整分解。

<!-- bilingual-en:start -->
Finite second moments make every term above finite and the cancellation valid in ordinary real arithmetic. If they fail, a nonnegative squared-error expectation can be infinite, but expressions involving an undefined mean or subtracting infinities cannot be justified by this finite-moment proof.
<!-- bilingual-en:end -->

有限二阶矩确保上面各项有限，可以在普通实数运算中展开与抵消。若条件失败，非负平方误差的期望可以为无穷；但不能借这个有限矩证明，去操作不存在的均值或无穷减无穷。

<!-- bilingual-en:start -->
## Explore the decomposition
<!-- bilingual-en:end -->

<!-- visual-lab:links:start -->
## 交互理解

- [[07_Programme/可视化实验室/实验/biasvariance|真正的偏差—方差分解]]
<!-- visual-lab:links:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/08_EC400_MathsCamp/2026_Course_Materials/04_PSI/Lectures/Lecture 1 - Foundations.pdf#page=66|LSE EC400 PSI Lecture1, slide66]] states the MSE, bias and variance identity; slides64–67 supply the estimation context.
- [Will Fithian, Berkeley Stat210A, Statistical decision theory, squared-error risk and comparison of estimators](https://www.stat.berkeley.edu/~wfithian/courses/stat210a/models.html): fixed-target squared-error risk and examples where some bias can improve MSE at some parameter values.
- The full finite-second-moment proof and two-point examples are derived above from expectation and variance definitions. The prediction comparison identifies a different target; it is not used as an alternative formula for this estimation problem.
<!-- bilingual-en:end -->

- LSE 第66页给出 MSE、偏差和方差恒等式，第64–67页提供估计语境。
- Berkeley 的平方损失风险及估计量比较段落支持固定目标风险，以及某些参数值下接受偏差可以降低 MSE。
- 正文按期望与方差定义完整推导有限二阶矩下的分解，直接计算两个二点分布；预测比较用于说明对象不同，不是替换本卡公式。

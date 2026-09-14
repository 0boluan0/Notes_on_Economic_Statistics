---
aliases:
  - "2×2 DID 在平行趋势与无预期下识别 ATT"
  - 2×2 DID identification of ATT
  - DID identifies ATT
  - 双重差分识别 ATT
student_os: knowledge-atom
atom_id: ECON-DID-002
atom_type: identification
status: source-checked
mastery_state: unassessed
part_of:
  - "[[双重差分法（DID）.canvas]]"
requires:
  - "[[双重差分法]]"
  - "[[平行趋势]]"
  - "[[DID无预期条件]]"
  - "[[DID 的外溢污染]]"
  - "[[DID 的样本构成变化]]"
contrasts_with:
  - "[[2×2 DID回归等价]]"
---

# 2×2 DID 在平行趋势与无预期下识别 ATT

<!-- bilingual-en:start -->
*Under parallel trends and no anticipation, the $2\times2$ DID identifies the ATT*
<!-- bilingual-en:end -->

> [!summary] 识别结论
> 四格均值永远可以相减。在处理定义稳定、对照组未受外溢、四格均值代表同一目标总体的前提下，还必须保证处理前尚未受政策影响，而且处理组与对照组的**未处理潜在结果**本会有相同平均变化；这时该差才等于处理组平均处理效应（ATT）。
>
> <!-- bilingual-en:start -->
> Four cell means can always be subtracted. Provided treatment is well defined, the comparison group is uncontaminated, and the four cells represent the same target populations over time, the contrast equals the average treatment effect on the treated only when outcomes are not yet affected before treatment and the groups would have had the same average change in their **untreated potential outcomes**.
> <!-- bilingual-en:end -->

令 $t=0$ 为处理前，$t=1$ 为处理后，$G=1$ 为处理组。处理组在处理后的目标是

$$
ATT=E\bigl[Y_1(1)-Y_1(0)\mid G=1\bigr].
$$

其中 $Y_1(1)$ 可观察，真正缺的是同一批处理单位在未处理时的 $Y_1(0)$。若无预期处理使 $Y_0=Y_0(0)$，并且平行趋势给出

$$
E[Y_1(0)-Y_0(0)\mid G=1]
=E[Y_1(0)-Y_0(0)\mid G=0],
$$

那么对照组的实际变化就可代替处理组缺失的未处理变化。把这个替代量代回 ATT，正好得到四格 DID。
<!-- bilingual-en:start -->
Let $t=0$ denote the pre-period, $t=1$ the post-period, and $G=1$ the treated group. The target after treatment is

$$
ATT=E\bigl[Y_1(1)-Y_1(0)\mid G=1\bigr].
$$

$Y_1(1)$ is observed. The missing quantity is $Y_1(0)$ for those same treated units. No anticipation lets the pre-period outcome equal $Y_0(0)$, while parallel trends makes the comparison group's untreated change stand in for the treated group's missing untreated change. Substituting that change into the ATT expression yields the four-cell DID exactly.
<!-- bilingual-en:end -->

## 识别的是谁的效应

这个结论通常识别 ATT，而不是自动识别全体单位的 ATE。对照组在这里负责恢复反事实趋势；它的处理效应没有被观察，也不必等于处理组的处理效应。若研究问题改成“政策若施加给所有单位会怎样”，需要额外外推条件，不能把 DID 系数换个名字就当作 ATE。
<!-- bilingual-en:start -->
The usual target is the ATT, not automatically the ATE for the whole population. The comparison group supplies a counterfactual trend; its own treatment effect is unobserved and need not equal the treated group's effect. Answering what would happen if everyone were treated requires additional extrapolation assumptions, not a relabelling of the DID coefficient.
<!-- bilingual-en:end -->

## 这个结论没有包办什么

平行趋势与无预期是中心条件，但完整设计还要处理样本构成、跨组干扰、处理定义和推断。识别等式成立也不等于有限样本估计精确；点估计的目标、标准误和外部有效性是另外的问题。
<!-- bilingual-en:start -->
Parallel trends and no anticipation are central, but a complete design must also address sample composition, interference across groups, treatment definition, and inference. Identification in the population does not make a finite-sample estimate precise; the estimand, standard error, and external validity remain separate questions.
<!-- bilingual-en:end -->

> [!question]- 回忆与辨别
> 为什么“对照组处理后没有接受政策”还不足以让四格 DID 等于 ATT？
>
> **答案：** 因为还要相信对照组的未处理变化能代表处理组缺失的未处理变化，并排除处理前预期反应。未受政策不等于自动成为可信反事实。

## 来源与核验

- Callaway & Sant’Anna (2021), [*Difference-in-Differences with Multiple Time Periods*](https://psantanna.com/files/Callaway_SantAnna_2020.pdf)，第 2 节：核验 ATT、无预期、条件平行趋势和由对照组路径识别反事实的表达。
- Goodman-Bacon (2021), [*Difference-in-Differences with Variation in Treatment Timing*](https://doi.org/10.1016/j.jeconom.2021.03.014)：核验两组两期 DID 在共同趋势下识别处理组平均效应的基准结论。

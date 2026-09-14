---
aliases:
  - "错位处理下传统 TWFE 会混入已处理组作对照的比较"
  - Staggered DID TWFE contamination
  - Already-treated controls in TWFE
  - 错位处理双向固定效应问题
student_os: knowledge-atom
atom_id: ECON-DID-010
atom_type: failure-mode
status: source-checked
mastery_state: unassessed
part_of:
  - "[[双重差分法（DID）.canvas]]"
requires:
  - "[[2×2 DID回归等价]]"
  - "[[事件研究系数语境]]"
contrasts_with:
  - "[[Group-time ATT 聚合]]"
---

# 错位处理下传统 TWFE 会混入已处理组作对照的比较

<!-- bilingual-en:start -->
*With staggered adoption, conventional TWFE mixes in comparisons that use already-treated groups as controls*
<!-- bilingual-en:end -->

> [!summary] 核心风险
> 当不同组在不同时间开始处理时，一个包含单位固定效应、时间固定效应和单个处理指标的回归，不再对应一个清楚的 $2\times2$ DID。它会加权组合许多两组两期比较，其中一部分把早已处理的组当作后来处理组的对照。
>
> <!-- bilingual-en:start -->
> When groups adopt treatment at different times, a regression with unit effects, time effects, and one treatment indicator no longer represents one transparent $2\times2$ DID. It combines many two-group/two-period comparisons, some of which use earlier-treated groups as controls for later-treated groups.
> <!-- bilingual-en:end -->

不同单位不同时间受处理且效应随 cohort 或暴露时间变化时，传统 two-way fixed effects 可能把已处理单位当对照。Goodman–Bacon 分解对各个 $2\times2$ DID 使用非负权重；问题在于某些组成比较已经减去了早期处理组不断变化的处理效应。把最终系数进一步写成底层处理效应的加权平均时，才可能出现负权重和符号反转。应改用按 group-time ATT 构造并透明聚合的方法，或为另一种估计框架明确给出假设。
<!-- bilingual-en:start -->
When units adopt treatment at different times and effects vary across cohorts or exposure lengths, conventional two-way fixed effects may use already-treated units as controls. The Goodman–Bacon decomposition assigns non-negative weights to its component $2\times2$ DIDs; the problem is that some components subtract the evolving effect of an earlier-treated group. Negative weights and sign reversals can appear when the final coefficient is instead decomposed over underlying treatment effects. Use estimators built from group-time ATTs with transparent aggregation, or justify another framework with explicit assumptions.
<!-- bilingual-en:end -->

## 为什么“已处理对照”会出错

设 A 组先处理、B 组后处理。B 尚未处理时，A 对 B 的比较可以利用 B 作对照；B 处理以后，传统 TWFE 还可能反过来用已经受到政策影响的 A 作为 B 的对照。若 A 的政策效应随暴露时间继续增长，A 在这段时期的变化不是未处理趋势。第二次比较会把 A 的动态效应减进 B 的效应。
<!-- bilingual-en:start -->
Let cohort A be treated before cohort B. While B remains untreated, it may serve as a comparison for A. After B adopts, conventional TWFE may reverse the roles and use the already-treated A as a control for B. If A's effect evolves with exposure length, A's subsequent change is not an untreated trend. That comparison subtracts A's dynamic treatment effect from B's effect.
<!-- bilingual-en:end -->

在标准的二元吸收式处理设定中，Goodman–Bacon 的分解说明静态 TWFE 是所有可构造 $2\times2$ DID 的加权平均。这个结论本身不是说每个 TWFE 都必然错；若处理效应在 cohort 和时间上满足足够强的同质限制，各比较可能仍指向同一参数。危险出现在研究者允许效应异质，却仍把一个由样本份额和处理方差决定的回归加权平均解释成自己想要的 ATT。
<!-- bilingual-en:start -->
Under the standard binary, absorbing-treatment setup, Goodman-Bacon's decomposition shows that the static TWFE coefficient is a weighted average of all available $2\times2$ DIDs. This does not mean every TWFE estimate is necessarily wrong: under sufficiently strong homogeneity across cohorts and time, the comparisons may still target the same effect. The problem arises when heterogeneous effects are allowed but a regression-weighted average—whose weights depend on sample shares and treatment variation—is interpreted as the researcher's intended ATT.
<!-- bilingual-en:end -->

## 事件研究还有第二层污染

在动态 TWFE 中，一个 lead 或 lag 的系数也可能吸收其他相对时期的效应。Sun 与 Abraham 说明，处理效应异质性本身就能制造表面的处理前系数。因此，检查静态 Bacon 权重不能替代检查动态事件研究的估计对象。
<!-- bilingual-en:start -->
Dynamic TWFE adds another problem: a lead or lag coefficient can absorb effects from other relative periods. Sun and Abraham show that treatment-effect heterogeneity alone can create apparent pretreatment coefficients. Inspecting the static Bacon decomposition therefore does not settle the estimand of a dynamic event-study regression.
<!-- bilingual-en:end -->

> [!question]- 回忆与应用
> 为什么“模型已经加入单位 FE 和年份 FE”不能回答错位处理下估计是否可信？
>
> **答案：** 因为还要知道每个比较使用谁作对照、处理效应是否随 cohort 或暴露时间变化，以及回归权重对应什么 estimand。固定效应名称不会自动排除已处理对照。

## 来源与核验

- Goodman-Bacon (2021), [*Difference-in-Differences with Variation in Treatment Timing*](https://doi.org/10.1016/j.jeconom.2021.03.014)：核验错位处理 TWFE 的 $2\times2$ 分解、权重来源和时间变化效应下的偏差。
- Sun & Abraham (2021), [*Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects*](https://arxiv.org/abs/1804.05785)：核验动态 TWFE 的跨时期污染和表面预趋势问题。

---
aliases:
  - "DID 的协变量调整改变平行趋势条件与目标总体"
  - Covariate adjustment in DID
  - Conditional parallel trends in DID
  - 双重差分协变量调整
student_os: knowledge-atom
atom_id: ECON-DID-012
atom_type: distinction
status: source-checked
mastery_state: unassessed
part_of:
  - "[[双重差分法（DID）.canvas]]"
requires:
  - "[[平行趋势]]"
  - "[[DID 的样本构成变化]]"
leads_to:
  - "[[Group-time ATT 聚合]]"
---

# DID 的协变量调整改变平行趋势条件与目标总体

<!-- bilingual-en:start -->
*Covariate adjustment in DID changes the parallel-trends condition and potentially the target population*
<!-- bilingual-en:end -->

> [!summary] 核心区别
> 在 DID 里加入协变量，不只是“控制得更多”。它可能把无条件平行趋势改成条件平行趋势，改变哪些单位互相比较、使用什么权重，以及最后效应代表哪个有共同支持的总体。
>
> <!-- bilingual-en:start -->
> Adding covariates to DID is not merely “controlling for more.” It may replace unconditional with conditional parallel trends, change which units are compared and how they are weighted, and alter the population represented by the final effect when overlap is limited.
> <!-- bilingual-en:end -->

若不同年龄、行业或处理前结果历史有不同的自然趋势，而处理组与对照组的这些特征分布不同，无条件 DID 会把构成差异带来的趋势混进政策效应。条件平行趋势要求在同一处理前协变量 $X$ 下，处理组和所选对照组的未处理结果变化相同。
<!-- bilingual-en:start -->
Suppose age, industry, or pretreatment outcome histories imply different natural trends, and those characteristics are distributed differently across treated and comparison groups. Unconditional DID then mixes composition-driven trends into the policy contrast. Conditional parallel trends instead requires equal untreated changes between treated and chosen comparison units at the same pretreatment $X$.
<!-- bilingual-en:end -->

## 三个必须分开的作用

协变量可能用于提高精度、使识别假设更可信，或重新加权到某个目标总体。三者不是同一件事。一个强预测变量可以降低方差，却不修复不可比趋势；重新加权可以使处理组与对照组在 $X$ 上对齐，却可能把结果限制到两组都有观察值的共同支持（overlap）区域；某个变量在回归中显著，也不说明它足以使平行趋势成立。
<!-- bilingual-en:start -->
Covariates may improve precision, make the identifying assumption more plausible, or reweight observations toward a target population. These roles are distinct. A strong predictor may reduce variance without repairing incomparable trends. Reweighting can align treated and comparison units on $X$ but may restrict the result to their overlap region. Statistical significance of a control does not show that conditioning on it is sufficient for parallel trends.
<!-- bilingual-en:end -->

## 只能用处理前信息建立反事实

处理后的收入、就业、迁移或企业规模可能已经被政策改变。把它们当控制变量，会截断政策作用路径或引入选择，目标也不再是原来的总效应。Callaway 与 Sant’Anna 的识别结果明确使用处理前协变量；若要研究中介机制，应另行定义直接/间接效应，而不是悄悄把中介放进 DID 回归。
<!-- bilingual-en:start -->
Post-treatment income, employment, migration, or firm size may already be affected by the policy. Controlling for them can block part of the treatment pathway or induce selection, so the estimand is no longer the original total effect. Callaway and Sant’Anna's identification results use pretreatment covariates. Mediation questions require separately defined direct and indirect effects rather than silently inserting a mediator into a DID regression.
<!-- bilingual-en:end -->

## 共同支持是实质条件

若所有处理单位都是大型企业、所有对照单位都是小型企业，就没有数据告诉我们“同样规模下”的趋势差。极端外推不是条件平行趋势的证据。应展示倾向得分（propensity score）或关键协变量的重叠、说明裁剪规则，并明确裁剪后 ATT 代表谁。
<!-- bilingual-en:start -->
If every treated unit is a large firm and every comparison unit is small, the data contain no evidence about trend differences at a common size. Extreme extrapolation is not support for conditional parallel trends. Show overlap in propensity scores or key covariates, state any trimming rule, and identify the population represented by the resulting ATT.
<!-- bilingual-en:end -->

> [!question]- 回忆与应用
> 在 DID 回归中加入政策实施后的就业作为控制变量，为什么可能有问题？
>
> **答案：** 就业可能是政策作用的一部分。控制它会截断效应路径或引入处理后选择，使目标从总效应变成另一个、通常未明确的参数。

## 来源与核验

- Callaway & Sant’Anna (2021), [*Difference-in-Differences with Multiple Time Periods*](https://psantanna.com/files/Callaway_SantAnna_2020.pdf)，Assumptions 4–6 与 Remarks 1、7：核验条件平行趋势、overlap、处理前协变量和排除处理后协变量。
- Baker et al. (2026), [*Difference-in-Differences Designs: A Practitioner’s Guide*](https://www.aeaweb.org/articles?id=10.1257/jel.20251650)：交叉核验协变量、权重和目标参数必须按设计解释。

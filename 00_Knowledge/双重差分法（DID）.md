---
aliases:
  - "Difference-in-Differences"
  - "Difference in Differences"
  - "DID"
  - "双重差分"
status: source-checked
---

# 双重差分法（DID）
<!-- bilingual-en:start -->
*Difference-in-differences (DID)*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 当一个组经历政策、另一个组没有时，用两组“前后变化的差”构造处理组未受政策时的反事实变化。
> **具体锚点：** 新泽西上调最低工资、宾夕法尼亚未上调；DID 比较两地就业前后变化，而不是只比政策后水平。
> **核心难点：** 平行趋势说的是未处理潜在结果的趋势，而不是处理前水平相同；它不能被有限的处理前数据彻底证明。
> **为什么重要：** DID 的可信度来自研究设计和对照组，不来自在回归里放入两个固定效应。
> **继续：** 先掌握 2×2 ATT，再读事件研究、错位处理与异质效应；不要直接把传统 TWFE 当通用答案。
> <!-- bilingual-en:start -->
> **What it solves:** When one group experiences a policy and another does not, DID uses the difference between their before–after changes to construct the treated group's counterfactual change without the policy.
> **Concrete anchor:** New Jersey raised its minimum wage while Pennsylvania did not. DID compares the change in employment across the two states rather than comparing only their post-policy levels.
> **Central difficulty:** Parallel trends concerns the trajectory of untreated potential outcomes, not equality of pretreatment levels. A finite pretreatment sample can never prove it conclusively.
> **Why it matters:** DID is credible because of its research design and comparison group, not because a regression contains two fixed effects.
> **Continue with:** First master the $2\times2$ ATT, then study event studies, staggered adoption, and heterogeneous effects. Do not treat conventional TWFE as a universal solution.
> <!-- bilingual-en:end -->

> [!source] 本节依据
> - [[02_Economy/01_Econometrics/太白金星v计量.pdf]]：支持课程范围、初级计量顺序与示例；核心结论另与权威教材或原始论文交叉核验。
> - Wooldridge, *Introductory Econometrics: A Modern Approach*：核验线性模型、推断、内生性、面板与时序计量。
> <!-- bilingual-en:start -->
> - [[02_Economy/01_Econometrics/太白金星v计量.pdf|Local econometrics course notes]] support the course scope, introductory sequence, and examples; the core results were also cross-checked against authoritative textbooks or original research.
> - Wooldridge, *Introductory Econometrics: A Modern Approach*, supports the treatment of linear models, inference, endogeneity, panel data, and time-series econometrics.
> <!-- bilingual-en:end -->

## 2×2 反事实结构
<!-- bilingual-en:start -->
*The $2\times2$ counterfactual structure*
<!-- bilingual-en:end -->

处理组 G=1 在 post 期受处理，对照组 G=0 始终未处理。DID 为
$[E(Y\mid G=1,post)-E(Y\mid G=1,pre)]-[E(Y\mid G=0,post)-E(Y\mid G=0,pre)]$。
它用对照组变化估计处理组在无处理时本会发生的共同变化，目标通常是 treated 的 ATT。
<!-- bilingual-en:start -->
Group $G=1$ receives treatment in the post period, while group $G=0$ remains untreated. DID is
$[E(Y\mid G=1,post)-E(Y\mid G=1,pre)]-[E(Y\mid G=0,post)-E(Y\mid G=0,pre)]$.
It uses the comparison group's change to estimate the common change the treated group would have experienced without treatment. The usual target is the ATT for the treated group.
<!-- bilingual-en:end -->

## 回归表达与 ATT
<!-- bilingual-en:start -->
*Regression representation and the ATT*
<!-- bilingual-en:end -->

$Y_{it}=\alpha+\gamma G_i+\lambda Post_t+\tau(G_i\times Post_t)+u_{it}$ 中 $\tau$ 等于 2×2 DID。面板、重复截面都可使用，但抽样和构成假设不同。加入协变量应明确是提高精度、使条件平行趋势更可信，还是改变目标总体。
<!-- bilingual-en:start -->
In $Y_{it}=\alpha+\gamma G_i+\lambda Post_t+\tau(G_i\times Post_t)+u_{it}$, $\tau$ equals the $2\times2$ DID. Both panel data and repeated cross-sections can be used, but they require different sampling and composition assumptions. When adding covariates, state whether they improve precision, make conditional parallel trends more plausible, or change the target population.
<!-- bilingual-en:end -->

## 平行趋势与辅助假设
<!-- bilingual-en:start -->
*Parallel trends and supporting assumptions*
<!-- bilingual-en:end -->

核心是 $E[Y_t(0)-Y_{t-1}(0)\mid G=1]=E[\cdot\mid G=0]$，另需无预期处理、稳定样本构成/SUTVA 等。处理前趋势图和 placebo 能发现明显不相容，却无法证明处理后未处理趋势本会平行。
<!-- bilingual-en:start -->
The core condition is $E[Y_t(0)-Y_{t-1}(0)\mid G=1]=E[\cdot\mid G=0]$. Additional assumptions include no anticipation, stable sample composition, and SUTVA. Pretreatment trend plots and placebo tests can reveal clear incompatibilities, but they cannot prove that untreated trends would have remained parallel after treatment.
<!-- bilingual-en:end -->

## 事件研究
<!-- bilingual-en:start -->
*Event studies*
<!-- bilingual-en:end -->

相对处理时点的 leads/lags 展示动态效应和处理前差异。系数依赖基准期、窗口和样本构成；处理前系数不显著可能只是功效低。置信区间应考虑多个 event-time 系数的同时推断。
<!-- bilingual-en:start -->
Leads and lags relative to treatment timing display dynamic effects and pretreatment differences. The coefficients depend on the reference period, event window, and sample composition; insignificant pretreatment coefficients may merely reflect low power. Confidence intervals should account for simultaneous inference across multiple event-time coefficients.
<!-- bilingual-en:end -->

## 错位处理与异质效应
<!-- bilingual-en:start -->
*Staggered treatment and heterogeneous effects*
<!-- bilingual-en:end -->

不同单位不同时间受处理且效应随 cohort/时间异质时，传统 two-way fixed effects 可能把已处理单位当对照，并形成难解释甚至负权重比较。应使用按 group-time ATT 构造并透明聚合的现代估计量，或在明确假设下选择其他方法。
<!-- bilingual-en:start -->
When units are treated at different times and effects vary across cohorts or over time, conventional two-way fixed effects may use already-treated units as controls and combine comparisons with opaque or even negative weights. Use modern estimators built from group-time ATTs with transparent aggregation, or another method justified by explicit assumptions.
<!-- bilingual-en:end -->

## DDD 与变种
<!-- bilingual-en:start -->
*DDD and other variants*
<!-- bilingual-en:end -->

三重差分再引入一个不受政策机制影响的维度，净化另一个共同变化；它需要相应的“差中差仍可比”假设，不是自动更可信。连续处理、处理可逆和 spillover 需要专门设计。
<!-- bilingual-en:start -->
Triple differences introduce another dimension that should not be affected by the policy mechanism in order to remove an additional common change. This requires a corresponding comparability assumption for the difference-in-differences; it is not automatically more credible. Continuous treatments, reversible treatments, and spillovers require specialized designs.
<!-- bilingual-en:end -->

## 实操与报告
<!-- bilingual-en:start -->
*Implementation and reporting*
<!-- bilingual-en:end -->

明确处理、对照、时点和 estimand；画原始趋势与样本构成；解释平行趋势机制；选择与处理时序相容的估计量和聚类层级；报告动态、预趋势、替代对照、placebo、窗口和异质性。
<!-- bilingual-en:start -->
Define the treatment, comparison group, timing, and estimand. Plot raw trends and sample composition. Explain the mechanism behind parallel trends. Choose an estimator and clustering level compatible with treatment timing. Report dynamics, pretrends, alternative controls, placebos, event windows, and heterogeneity.
<!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum self-check*
<!-- bilingual-en:end -->

### 用自己的话解释 DID 为什么要‘再差一次’。
<!-- bilingual-en:start -->
*Explain in your own words why DID takes a second difference.*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 处理组前后差含政策效应和共同时间变化；减去对照组前后差，尝试扣除处理组未处理时也会经历的共同变化。
> <!-- bilingual-en:start -->
> The treated group's before–after difference contains both the policy effect and common time changes. Subtracting the comparison group's before–after difference attempts to remove the common change the treated group would also have experienced without treatment.
> <!-- bilingual-en:end -->
### 平行趋势是否要求两组处理前结果水平相同？
<!-- bilingual-en:start -->
*Does parallel trends require the two groups to have the same pretreatment outcome level?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 不要求水平相同，要求未处理潜在结果的平均变化路径相同；固定水平差可被组效应吸收。
> <!-- bilingual-en:start -->
> No. It requires equal average changes in untreated potential outcomes, not equal levels. A fixed level difference can be absorbed by the group effect.
> <!-- bilingual-en:end -->
### 错位处理下传统 TWFE 的主要危险是什么？
<!-- bilingual-en:start -->
*What is the main danger of conventional TWFE under staggered treatment?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 已处理组可能充当后来处理组的对照，异质动态效应被混成不透明权重平均，甚至出现负权重和反直觉系数。
> <!-- bilingual-en:start -->
> Already-treated groups may serve as controls for groups treated later. Heterogeneous dynamic effects are then mixed through opaque weights, potentially producing negative weights and counterintuitive coefficients.
> <!-- bilingual-en:end -->
### 处理前 event-study 系数都不显著是否证明平行趋势？
<!-- bilingual-en:start -->
*Do insignificant pretreatment event-study coefficients prove parallel trends?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 不能；有限处理前期和噪声可能缺乏功效，且平行趋势是未处理潜在结果在处理后本会如何变化的假设。
> <!-- bilingual-en:start -->
> No. A short pretreatment period and noisy data may provide little power, and parallel trends concerns how untreated potential outcomes would have evolved after treatment.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [[02_Economy/01_Econometrics/太白金星v计量.pdf]]：支持课程范围、初级计量顺序与示例；核心结论另与权威教材或原始论文交叉核验。
- Wooldridge, *Introductory Econometrics: A Modern Approach*：核验线性模型、推断、内生性、面板与时序计量。
- Card & Krueger (1994)：经典 2×2 应用与研究设计语境。
- Callaway & Sant’Anna (2021), [Difference-in-Differences with Multiple Time Periods](https://doi.org/10.1016/j.jeconom.2020.12.001)：核验错位处理与组别—时期 ATT。
- Sun & Abraham (2021) 与 Goodman-Bacon (2021)：核验异质效应下传统 TWFE 事件研究和加权分解的问题。
<!-- bilingual-en:start -->
- [[02_Economy/01_Econometrics/太白金星v计量.pdf|Local econometrics course notes]] support the course scope, introductory sequence, and examples; the core results were also cross-checked against authoritative textbooks or original research.
- Wooldridge, *Introductory Econometrics: A Modern Approach*, was used to verify the linear model, inference, endogeneity, panel data, and time-series econometrics.
- Card and Krueger (1994) supports the classic $2\times2$ application and its research-design context.
- Callaway and Sant’Anna (2021), [Difference-in-Differences with Multiple Time Periods](https://doi.org/10.1016/j.jeconom.2020.12.001), was used to verify the treatment of staggered adoption and group-time ATTs.
- Sun and Abraham (2021), together with Goodman-Bacon (2021), supports the discussion of conventional TWFE event studies and weighting decompositions under heterogeneous effects.
<!-- bilingual-en:end -->

---
aliases:
  - "DID 研究应从 estimand 与对照组开始而不是从回归式开始"
  - DID research workflow
  - Difference-in-differences design checklist
  - 双重差分实证流程
student_os: knowledge-atom
atom_id: ECON-DID-014
atom_type: procedure
status: source-checked
mastery_state: unassessed
part_of:
  - "[[双重差分法（DID）.canvas]]"
requires:
  - "[[2×2 DID 识别 ATT]]"
  - "[[前趋势检验边界]]"
  - "[[Group-time ATT 聚合]]"
  - "[[DID 的外溢污染]]"
  - "[[DID结果尺度]]"
related:
  - "[[聚类标准误边界]]"
---

# DID 研究应从 estimand 与对照组开始而不是从回归式开始

<!-- bilingual-en:start -->
*A DID study should begin with the estimand and comparison group, not with a regression specification*
<!-- bilingual-en:end -->

> [!summary] 决策顺序
> 先说清楚要估计谁、哪种处理、哪个时间范围下的什么效应，再说明谁在每个时期提供反事实。只有这些问题有答案后，才能选择估计量、事件窗口、协变量和推断方式。
>
> <!-- bilingual-en:start -->
> First define whose effect is sought, which treatment is being studied, and over what horizon. Then identify who supplies the counterfactual in each period. Only after those questions are answered should the researcher choose an estimator, event window, covariates, and inference procedure.
> <!-- bilingual-en:end -->

把回归放在后面不是形式主义。DID 的可信度来自研究设计和对照组；两个固定效应只能执行比较，不能替研究者创造可信的反事实。
<!-- bilingual-en:start -->
Putting the regression later is not a matter of presentation. DID derives credibility from the research design and comparison group. Two fixed effects can implement a comparison; they cannot create a credible counterfactual for the researcher.
<!-- bilingual-en:end -->

## 一条可执行的顺序

1. **定义处理。** 写清生效、公告与可能预期反应的时间；说明处理是一次进入后持续、可退出、连续剂量，还是会外溢。
2. **定义 estimand。** 区分 ATT、某 cohort-time 效应、按暴露时长的动态效应、直接效应与总效应。
3. **逐期指定对照。** 说明使用 never-treated、not-yet-treated 还是别的比较组，以及为何它们在这些时期未被处理或污染。
4. **为平行趋势给出机制。** 讨论制度、共同冲击、样本构成和处理前结果；图和伪处理检验（placebo）是证据，不是替代论证。
5. **再选估计量。** 两组两期可用四格/饱和回归；错位处理与异质效应用处理批次—时期（cohort-time）相容的方法；协变量只使用处理前信息并检查共同支持。
6. **按依赖结构推断。** 面板与政策结果常有组内序列相关。聚类层级应反映处理分配和误差相关的层级；独立 cluster 很少时，常规大样本 cluster 标准误也可能失真，需要适合小 cluster 的方法或随机化/置换思路。
7. **让读者看见估计对象。** 报告每期样本、处理 cohort、原始趋势、权重或聚合规则、事件基准期、稳健性和不能识别的范围。

<!-- bilingual-en:start -->
1. **Define treatment.** State implementation, announcement, and plausible anticipation dates; say whether treatment is absorbing, reversible, continuous, or capable of spilling over.
2. **Define the estimand.** Distinguish the ATT, cohort-time effects, exposure-length dynamics, and direct versus total effects.
3. **Specify controls period by period.** State whether controls are never treated, not yet treated, or another group, and why they remain untreated and uncontaminated at those dates.
4. **Make the case for parallel trends.** Use institutions, common shocks, composition, and pretreatment outcomes. Plots and placebos are evidence, not a substitute for the design argument.
5. **Then choose the estimator.** A saturated four-cell regression may suit a $2\times2$ design. Staggered heterogeneous effects require cohort-time-compatible methods. Covariate adjustment should use pretreatment information and examine overlap.
6. **Align inference with dependence.** Panel and policy outcomes are often serially correlated within groups. Clustering should reflect the level of treatment assignment and residual dependence. With few independent clusters, ordinary large-sample clustered standard errors may also fail, requiring a small-cluster or randomisation-based method.
7. **Expose the estimand to the reader.** Report period-specific samples, treatment cohorts, raw trends, weights or aggregation rules, the event-study reference period, robustness checks, and what the design cannot identify.
<!-- bilingual-en:end -->

## 报告不能退化成检验清单

替代对照和 placebo 只有在对应具体威胁时才有信息。例如，把一个理论上也会受政策影响的结果当 placebo，失败并不诊断平行趋势；任意换窗口得到相似系数，也不能证明没有预期反应。每项稳健性检查都应先说明：它在什么失效机制下应该改变结果？
<!-- bilingual-en:start -->
Alternative controls and placebos are informative only when they target a specific threat. An outcome that theory says may also respond to the policy is not a valid placebo for parallel trends. Similar coefficients under arbitrary window changes do not rule out anticipation. Every robustness check should begin by stating which failure mechanism would make its result change.
<!-- bilingual-en:end -->

> [!question]- 回忆与应用
> 拿到一份地区—年份面板后，为什么第一步不应是跑 `unit FE + year FE + treated`？
>
> **答案：** 因为还不知道目标效应、处理何时真正开始、每期谁能提供未处理反事实、是否有错位处理或外溢。回归形式必须由这些设计事实决定。

## 来源与核验

- Baker et al. (2026), [*Difference-in-Differences Designs: A Practitioner’s Guide*](https://www.aeaweb.org/articles?id=10.1257/jel.20251650)：核验按设计类型组织 estimand、协变量、权重、多期与错位处理选择的总体框架。
- Bertrand, Duflo & Mullainathan (2004), [*How Much Should We Trust Differences-in-Differences Estimates?*](https://doi.org/10.1162/003355304772839588)：核验 DID 中序列相关会使常规标准误严重偏小，以及聚类/压缩时期/随机化推断等修正方向。
- Roth (2022), [*Pretest with Caution*](https://www.aeaweb.org/articles?id=10.1257/aeri.20210236)：核验 pretrend 预检不能承担设计放行功能。

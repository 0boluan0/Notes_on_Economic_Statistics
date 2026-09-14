---
aliases:
  - "Group-time ATT 先估计可比效应再按问题透明聚合"
  - Group-time average treatment effects
  - ATT(g,t)
  - 组别时期平均处理效应
student_os: knowledge-atom
atom_id: ECON-DID-011
atom_type: method
status: source-checked
mastery_state: unassessed
part_of:
  - "[[双重差分法（DID）.canvas]]"
requires:
  - "[[错位 DID 的已处理对照]]"
  - "[[平行趋势]]"
  - "[[错位 DID 的吸收处理条件]]"
leads_to:
  - "[[事件研究系数语境]]"
  - "[[DID 估计目标与对照组]]"
---

# Group-time ATT 先估计可比效应再按问题透明聚合

<!-- bilingual-en:start -->
*Group-time ATTs estimate comparable effects first and aggregate them transparently for the question*
<!-- bilingual-en:end -->

> [!summary] 核心方法
> 错位处理时，不先强迫所有 cohort 和时期共用一个回归系数。先对“在 $g$ 期首次处理的组，于 $t$ 期受到什么影响”分别估计 $ATT(g,t)$，再用研究问题决定怎样汇总。
>
> <!-- bilingual-en:start -->
> With staggered adoption, do not first force every cohort and period into one regression coefficient. Estimate $ATT(g,t)$—the effect at time $t$ for units first treated at time $g$—and then aggregate those effects according to the research question.
> <!-- bilingual-en:end -->

对首次在 $g$ 期接受处理的单位，定义

$$
ATT(g,t)=E\bigl[Y_t(g)-Y_t(0)\mid G=g\bigr].
$$

这个对象允许效应随首次处理 cohort、日历时间和暴露时长 $e=t-g$ 改变。它把传统 TWFE 隐藏在一个系数里的异质性展开了：研究者可以直接看哪一组、哪一期有多少效应，而不必先接受由回归几何自动生成的权重。
<!-- bilingual-en:start -->
For units first treated in period $g$,

$$
ATT(g,t)=E\bigl[Y_t(g)-Y_t(0)\mid G=g\bigr].
$$

This object permits effects to vary by adoption cohort, calendar time, and exposure length $e=t-g$. It unpacks heterogeneity that conventional TWFE hides inside one coefficient, letting the researcher inspect the effect for each cohort and period before accepting any aggregate.
<!-- bilingual-en:end -->

## 对照组由假设决定

若存在可信的 never-treated 单位，它们可在所有时期作为固定对照。若所有单位最终都会处理，可以在某一期使用当时尚未处理的 cohort，但可识别范围会在最后一组开始受影响后收缩。选择 never-treated 还是 not-yet-treated 不是软件选项：它对应不同的平行趋势假设，也决定每个 $ATT(g,t)$ 到底由哪些比较识别。
<!-- bilingual-en:start -->
Credible never-treated units can serve as a fixed comparison group throughout. If everyone is eventually treated, cohorts not yet treated at a given time may be used, but the identifiable horizon shrinks once the last cohort becomes affected. Choosing never-treated versus not-yet-treated controls is not merely a software option: it invokes a different parallel-trends assumption and determines which comparisons identify each $ATT(g,t)$.
<!-- bilingual-en:end -->

## 聚合不是最后随手取平均

将 $ATT(g,t)$ 写成一个总数，需要权重 $w(g,t)$：

$$
\theta=\sum_g\sum_t w(g,t)ATT(g,t).
$$

先在每个 cohort 内对其可观察处理后时期取平均、再按 cohort 人数加权，可以回答“所有曾处理单位所经历的平均效应”；按暴露时长聚合回答“处理后第 $e$ 期的动态效应”；按日历时间聚合回答“某一时期已处理单位的平均效应”。这些都是合理但不同的问题。透明聚合的价值不在于保证结果更好看，而在于让一个总数的含义可以被审查。
<!-- bilingual-en:start -->
Reducing $ATT(g,t)$ to one number requires weights $w(g,t)$. Averaging over the observed post-treatment periods within each cohort and then weighting cohorts by size can summarize the effect experienced by all ever-treated units. Aggregating by exposure length can answer the effect $e$ periods after treatment, while aggregating by calendar time can answer the average effect among units treated in a particular period. These are all legitimate but different questions. Transparent aggregation does not guarantee a more attractive result; it makes the meaning of the summary auditable.
<!-- bilingual-en:end -->

长暴露期的动态效应只由足够早接受处理、且样本中仍可观察的 cohort 提供。若 $e=1$ 和 $e=5$ 的 cohort 构成不同，两点之差会同时含动态变化与组别构成变化。需要比较同一批 cohort 时，应限制在共同可观察的事件窗口，再明确报告被保留的目标总体。
<!-- bilingual-en:start -->
Dynamic effects at long exposure horizons are supplied only by cohorts treated early enough and still observed in the sample. If the cohorts contributing at $e=1$ differ from those at $e=5$, the contrast combines dynamics with changing cohort composition. To compare the same cohorts, restrict attention to a common event window and state the retained target population explicitly.
<!-- bilingual-en:end -->

## 它也不是自动修复器

Group-time 方法仍依赖正确的处理时点、无预期、相应对照组的（可能为条件）平行趋势、共同支持和可靠推断。若这些条件不可信，把传统 TWFE 换成现代命令不会创造识别。
<!-- bilingual-en:start -->
Group-time methods still require correct treatment timing, limited anticipation, parallel trends for the chosen controls—possibly conditional on covariates—common support, and valid inference. Replacing a conventional TWFE command with a modern estimator cannot create identification when those conditions are implausible.
<!-- bilingual-en:end -->

> [!question]- 回忆与应用
> “处理后两年平均效应”和“2028 年所有已处理单位的平均效应”为何可能不同？
>
> **答案：** 前者按相对处理时间聚合不同 cohort，后者按同一日历时间聚合不同暴露时长。它们使用不同 $ATT(g,t)$ 与权重，回答不同问题。

## 来源与核验

- Callaway & Sant’Anna (2021), [*Difference-in-Differences with Multiple Time Periods*](https://psantanna.com/files/Callaway_SantAnna_2020.pdf)，第 2–3 节：核验 $ATT(g,t)$、never-treated/not-yet-treated 对照、条件平行趋势和问题导向的聚合。
- Sun & Abraham (2021), [*Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects*](https://arxiv.org/abs/1804.05785)：交叉核验 cohort-event-time 效应与 interaction-weighted 动态聚合。

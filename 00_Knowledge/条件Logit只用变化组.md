---
aliases:
  - "条件固定效应 Logit 的共同斜率只由结果在单位内发生变化的组识别"
  - Conditional logit uses outcome-switching groups
student_os: knowledge-atom
atom_id: ECON-BIN-026
atom_set: binary-outcome-models
atom_type: identification-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[条件固定效应Logit]]"
  - "[[组内变异边界]]"
related:
  - "[[非线性固定效应偏误]]"
part_of:
  - "[[二元结果模型.canvas|二元结果模型]]"
---

# 条件固定效应 Logit 的共同斜率只由结果在单位内发生变化的组识别

<!-- bilingual-en:start -->
*The common slopes in conditional fixed-effects logit are identified only by groups whose outcomes change within unit*
<!-- bilingual-en:end -->

> [!summary] 条件化改变了实际估计样本
> 条件固定效应 logit 对单位成功次数 $S_i=\sum_tY_{it}$ 条件化。若某单位始终 $Y_{it}=0$ 或始终 $Y_{it}=1$，给定 $S_i$ 后只有一种结果排列，对共同斜率 $\beta$ 不提供信息，因此整组退出条件似然。只有 $0<S_i<T_i$ 的变化组参与斜率估计。
> <!-- bilingual-en:start -->
> Conditional fixed-effects logit conditions on the unit success count $S_i=\sum_tY_{it}$. If a unit is always zero or always one, conditioning on $S_i$ leaves only one possible outcome sequence. The group contains no information about the common slope $\beta$ and drops out of the conditional likelihood. Only switching groups with $0<S_i<T_i$ contribute to slope estimation.
> <!-- bilingual-en:end -->

这不是随机删行。结果从未变化的组可能在原始样本中占很大比例，因此条件 logit 的估计样本和目标人群可能与全样本明显不同。报告时应给出原始组数、全 0 组、全 1 组、最终变化组数和实际观测数。

<!-- bilingual-en:start -->
This is not arbitrary row deletion. Groups whose outcomes never change may form a large share of the original sample, so the conditional-logit estimation sample and target population can differ sharply from the full sample. A report should state the original number of groups, the all-zero and all-one groups, the remaining switching groups, and the actual number of observations used.
<!-- bilingual-en:end -->

结果变化仍不够：某个解释变量若在单位内不变，就与单位截距无法区分，不能估计其系数。这里要同时检查“$Y$ 是否变化”和“目标 $X$ 是否变化”；二者是不同的组内信息边界。

<!-- bilingual-en:start -->
Outcome switching is still not sufficient for every coefficient. A regressor that is constant within unit cannot be distinguished from the unit intercept, so its coefficient cannot be estimated. Both outcome variation and variation in the target regressor must be checked; they are distinct within-unit information boundaries.
<!-- bilingual-en:end -->

> [!example] 哪些组会进入
> 三名个体各观察三期：A 的结果为 $(0,0,0)$，B 为 $(1,1,1)$，C 为 $(0,1,1)$。A 与 B 在给定成功次数后都只剩一种序列；共同斜率只从 C 这类结果发生变化的个体获得信息。若 C 的某个协变量三期都相同，该协变量的斜率仍无法由 C 识别。
> <!-- bilingual-en:start -->
> Three individuals are observed for three periods. A has outcomes $(0,0,0)$, B has $(1,1,1)$, and C has $(0,1,1)$. Once the success count is fixed, A and B each have only one possible sequence, so the common slope receives information only from switchers such as C. If one of C's regressors is constant across all three periods, C still cannot identify that regressor's slope.
> <!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- Chamberlain (1980), [“Analysis of Covariance with Qualitative Data”](https://doi.org/10.2307/2297110), *Review of Economic Studies* 47(1): 225–238：核对条件似然中成功次数的作用及变化组提供斜率信息的边界。
  <!-- bilingual-en:start -->
  Chamberlain (1980), [“Analysis of Covariance with Qualitative Data”](https://doi.org/10.2307/2297110), *Review of Economic Studies* 47(1): 225–238, supports the role of the success count in the conditional likelihood and the boundary that slope information comes from switching groups.
  <!-- bilingual-en:end -->
- StataCorp, [*clogit — Conditional (fixed-effects) logistic regression*](https://www.stata.com/manuals/rclogit.pdf), “Fixed-effects logit” 与 Methods and formulas：核对全 0/全 1 组退出、单位内不变变量不可估计及报告实际估计样本。
  <!-- bilingual-en:start -->
  StataCorp, [*clogit — Conditional (fixed-effects) logistic regression*](https://www.stata.com/manuals/rclogit.pdf), “Fixed-effects logit” and Methods and formulas, supports omission of all-zero and all-one groups, non-estimability of within-unit constant regressors, and reporting the actual estimation sample.
  <!-- bilingual-en:end -->
- [[组内变异边界]]：复用一般固定效应中“时间不变解释变量没有组内识别信息”的边界。
  <!-- bilingual-en:start -->
  [[组内变异边界|The within-unit variation boundary]] supplies the general fixed-effects rule that a time-invariant regressor contains no within-unit identifying variation.
  <!-- bilingual-en:end -->

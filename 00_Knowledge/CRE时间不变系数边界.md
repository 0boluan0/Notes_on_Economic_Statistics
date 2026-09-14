---
aliases:
  - "CRE 能估计时间不变变量但其系数仍依赖额外的组间外生性而不自动具有因果解释"
  - CRE time-invariant coefficient boundary
student_os: knowledge-atom
atom_id: ECON-PANEL-021
atom_set: panel-data
atom_type: interpretation-boundary
status: source-checked
mastery_state: unassessed
part_of:
  - "[[面板数据：Pooled OLS、固定效应与随机效应.canvas]]"
requires:
  - "[[相关随机效应]]"
contrasts_with:
  - "[[组内变异边界]]"
related:
  - "[[固定效应因果边界]]"
---

# CRE 能估计时间不变变量但其系数仍依赖额外的组间外生性而不自动具有因果解释

<!-- bilingual-en:start -->
*CRE can estimate coefficients on time-invariant variables, but their interpretation still requires additional between-unit exogeneity and is not automatically causal*
<!-- bilingual-en:end -->

> [!summary] 原子命题
> 单位固定效应会消去时间不变变量，而 CRE 保留组间信息，因此能在数值上报告这类变量的系数。但该系数由单位之间的差异识别；若时间不变变量仍与 CRE 投影后的剩余单位异质性 $r_i$ 相关，它就不具有通常的一致性或因果解释。
>
> <!-- bilingual-en:start -->
> Unit fixed effects remove time-invariant regressors, whereas CRE retains between-unit information and can therefore report their coefficients. Those coefficients are identified from differences across units; if a time-invariant regressor still correlates with the residual unit heterogeneity $r_i$ after the CRE projection, the usual consistency or causal interpretation does not follow.
> <!-- bilingual-en:end -->

因此“FE 估不了、CRE 估得了”只回答可计算性，不回答识别。研究者仍需为组间比较提供额外正交限制、外生变化或设计依据；更换估计器不会创造可信反事实。
<!-- bilingual-en:start -->
Thus “FE cannot estimate it, CRE can” answers a computability question, not an identification question. The between-unit comparison still needs an orthogonality restriction, exogenous variation, or design justification; changing estimators does not create a credible counterfactual.
<!-- bilingual-en:end -->

> [!question]- 回忆与应用
> CRE 报出了“固定地区属性”的系数，为什么还不能直接称为因果效应？
>
> **答案：** 该系数来自地区之间的差异；若地区属性与剩余单位异质性相关，组间比较仍然混杂。

## 来源与核验

- StataCorp, [xtreg manual](https://www.stata.com/manuals/xtxtreg.pdf), CRE remarks and examples：说明 CRE 可以估计时间不变回归量，同时警告其系数解释依赖时间不变变量与剩余个体异质性的额外不相关限制。
- [[组内变异边界]]：对照 FE 为何不能从单位内部识别时间不变变量。

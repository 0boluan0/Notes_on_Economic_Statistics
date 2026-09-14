---
aliases:
  - "在未加权线性模型的同一样本与同一规格下组内变换和 LSDV 给出相同共同斜率"
  - Within-LSDV slope equivalence
student_os: knowledge-atom
atom_id: ECON-PANEL-017
atom_set: panel-data
atom_type: equivalence
status: source-checked
mastery_state: unassessed
part_of:
  - "[[面板数据：Pooled OLS、固定效应与随机效应.canvas]]"
requires:
  - "[[组内变换]]"
  - "[[FWL残差化定理]]"
---

# 在未加权线性模型的同一样本与同一规格下组内变换和 LSDV 给出相同共同斜率

<!-- bilingual-en:start -->
*In an unweighted linear model with the same sample and specification, within transformation and LSDV give the same common slopes*
<!-- bilingual-en:end -->

> [!summary] 原子命题
> 在未加权线性加性单位效应模型中，若估计样本、回归量和时间效应规格相同，先按单位去均值再做 OLS，与在原方程中加入完整单位 indicators 的 LSDV，对共同斜率 $\beta$ 给出相同估计值。
>
> <!-- bilingual-en:start -->
> In an unweighted linear additive-unit-effect model, demeaning by unit and then applying OLS gives the same estimate of the common slopes $\beta$ as least-squares dummy variables with a complete set of unit indicators, provided that the estimation sample, regressors, and time-effects specification are identical.
> <!-- bilingual-en:end -->

等价只针对共同斜率。LSDV 还显式参数化单位截距；软件展示的截距归一化、单位效应、总平方和、$R^2$ 和自由度处理可以不同。两次命令若因缺失值或附加 controls 使用了不同样本或规格，不能援引这条等价；带权估计还必须改用与权重相容的投影，不能直接套本卡的算术去均值。
<!-- bilingual-en:start -->
The equivalence concerns the common slopes only. LSDV also parameterises unit intercepts, and software may report different intercept normalisations, unit effects, sums of squares, $R^2$ measures, or degrees of freedom. Different missing-data samples or additional controls fall outside the equivalence; weighted estimation requires the corresponding weighted projection rather than the arithmetic demeaning used here.
<!-- bilingual-en:end -->

> [!question]- 回忆与应用
> 一个命令删除缺失值后再去均值，另一个在更大样本上跑 LSDV，斜率是否必然相同？
>
> **答案：** 不必然。相同斜率要求同一估计样本和同一规格。

## 来源与核验

- MIT OpenCourseWare, [14.382 Lecture 8](https://ocw.mit.edu/courses/14-382-econometrics-spring-2017/049489cf2ae5d050409ee2c5cbd5346f_MIT14_382S17_lec8.pdf), §2.1：说明 within 估计与加入单位 indicators 的 OLS 对共同斜率数值等价。
- StataCorp, [Interpreting the intercept in the fixed-effects model](https://www.stata.com/support/faqs/statistics/intercept-in-fixed-effects-model/)：区分共同斜率等价与截距、单位效应归一化。
- [[FWL残差化定理]]：给出先对单位 indicators 残差化再估计共同斜率的通用代数基础。

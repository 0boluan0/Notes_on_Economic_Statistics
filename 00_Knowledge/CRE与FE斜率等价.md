---
aliases:
  - "在未加权线性加性模型的同一样本与规格下 CRE 和 within FE 给出相同的时变回归量斜率"
  - CRE-FE slope equivalence
student_os: knowledge-atom
atom_id: ECON-PANEL-020
atom_set: panel-data
atom_type: equivalence
status: source-checked
mastery_state: unassessed
part_of:
  - "[[面板数据：Pooled OLS、固定效应与随机效应.canvas]]"
requires:
  - "[[相关随机效应]]"
  - "[[个体固定效应]]"
  - "[[非平衡面板均值]]"
---

# 在未加权线性加性模型的同一样本与规格下 CRE 和 within FE 给出相同的时变回归量斜率

<!-- bilingual-en:start -->
*For the same sample and specification in an unweighted linear additive model, CRE and within FE give the same slopes on time-varying regressors*
<!-- bilingual-en:end -->

> [!summary] 原子命题
> 在未加权线性加性单位效应模型中，若 CRE 与 within FE 使用同一完整估计样本、同一组时变回归量和同一规格，并在该共同样本内构造所有单位均值，则 CRE 中 $x_{it}$ 的斜率与 within FE 的共同斜率数值相同。
> 若 $x_{it}$ 含时期 indicators 或其他 aggregate time variables，它们也属于这组回归量，其单位均值必须进入 CRE；在非平衡面板中，这些均值会因单位观察到的时期不同而变化。
>
> <!-- bilingual-en:start -->
> In an unweighted linear additive-unit-effect model, CRE and within FE give numerically identical slopes on $x_{it}$ when they use the same complete-case estimation sample, the same time-varying regressors and specification, and all unit means are constructed within that common sample.
> If $x_{it}$ includes period indicators or other aggregate time variables, their unit means must also enter CRE. In an unbalanced panel, those means vary across units because units are observed in different periods.
> <!-- bilingual-en:end -->

等价对象只是时变回归量的 within 斜率，不是两个模型的全部输出。样本、回归量集合、规格或均值计算一旦不同，等价前提就不再成立；尤其不能让每个变量按自己的非缺失期计算均值。
<!-- bilingual-en:start -->
The equivalence concerns only the within slopes on time-varying regressors, not every model output. It no longer applies when the sample, regressor set, specification, or mean construction differs; in particular, variable-specific pairwise means are not allowed.
<!-- bilingual-en:end -->

> [!question]- 回忆与应用
> CRE 与 FE 使用相同变量，但 CRE 的单位均值按 pairwise 非缺失值计算，能否保证斜率相同？
>
> **答案：** 不能。等价要求所有均值来自同一个完整估计样本。

## 来源与核验

- Mundlak (1978), [“On the Pooling of Time Series and Cross Section Data”](https://people.stern.nyu.edu/wgreene/Econometrics/Mundlak-1978.pdf)：建立平衡线性面板中 CRE 与 within 斜率的代数联系。
- Wooldridge (2019), [“Correlated random effects models with unbalanced panels”](https://doi.org/10.1016/j.jeconom.2018.12.010), Proposition 2.1：把时变斜率等价扩展到使用共同完整估计样本均值的非平衡面板。
- StataCorp, [xtreg manual](https://www.stata.com/manuals/xtxtreg.pdf), CRE methods：核验非平衡样本均值与 FE 斜率等价。

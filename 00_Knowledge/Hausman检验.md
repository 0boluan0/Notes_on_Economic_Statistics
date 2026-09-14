---
aliases:
  - "Hausman 检验通过比较同一参数的两个估计量是否系统分离来检验模型限制"
  - Hausman specification test
  - 豪斯曼检验
student_os: knowledge-atom
atom_id: ECON-PANEL-023
atom_set: panel-data
atom_type: definition
status: source-checked
mastery_state: unassessed
part_of:
  - "[[面板数据：Pooled OLS、固定效应与随机效应.canvas]]"
requires:
  - "[[个体固定效应]]"
  - "[[随机效应]]"
leads_to:
  - "[[Hausman检验边界]]"
related:
  - "[[Mundlak检验]]"
---

# Hausman 检验通过比较同一参数的两个估计量是否系统分离来检验模型限制

<!-- bilingual-en:start -->
*The Hausman test evaluates model restrictions by asking whether two estimators of the same parameter separate systematically*
<!-- bilingual-en:end -->

> [!summary] 原子命题
> Hausman 检验比较两个针对同一参数的估计量：一个在原假设与备择下都一致，另一个只在原假设下才一致、并在原假设下更有效率。若原假设成立，两者的差异应只来自抽样波动；系统差异则反对使第二个估计量成立的限制。
>
> <!-- bilingual-en:start -->
> The Hausman test compares two estimators of the same parameter: one consistent under both the null and the alternative, and another consistent only under the null and more efficient there. Under the null, their difference should be attributable to sampling variation; a systematic difference counts against the restrictions supporting the second estimator.
> <!-- bilingual-en:end -->

在线性面板的 FE–RE 应用中，FE 通常扮演“备择下仍一致”的估计量，RE 则依赖额外的单位效应正交限制。检验比较两者共有的时变回归量斜率，而不是把所有输出逐项相减；具体使用前提见 [[Hausman检验边界]]。
<!-- bilingual-en:start -->
In the linear-panel FE–RE application, FE commonly plays the estimator that remains consistent under the alternative, whereas RE depends on an additional unit-effect orthogonality restriction. The test compares slopes on parameters common to both estimators, not every reported output; its implementation premises are recorded separately.
<!-- bilingual-en:end -->

> [!question]- 回忆与应用
> FE–RE Hausman 检验拒绝时，直接被反对的是什么？
>
> **答案：** 是让 RE 与 FE 估计同一共同斜率且 RE 仍一致的附加限制；拒绝本身不是 FE 因果性的证明。

## 来源与核验

- Hausman (1978), [“Specification Tests in Econometrics”](https://mit-dspace.eks.prod.4science.cloud/bitstreams/3870feac-1282-468c-bde3-6d18741013ba/download)：建立“备择下仍一致”与“仅原假设下相容且更有效率”估计量差异的 specification-test 逻辑。
- StataCorp, [hausman manual](https://www.stata.com/manuals/rhausman.pdf), p.3：核验 FE–RE 应用中共同参数的比较对象。

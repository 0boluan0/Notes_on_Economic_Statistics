---
aliases:
  - "Mundlak 检验通过联合检验 CRE 中的回归量单位均值系数诊断标准 RE 正交限制"
  - Mundlak test
  - Correlated-random-effects test
student_os: knowledge-atom
atom_id: ECON-PANEL-022
atom_set: panel-data
atom_type: definition
status: source-checked
mastery_state: unassessed
part_of:
  - "[[面板数据：Pooled OLS、固定效应与随机效应.canvas]]"
requires:
  - "[[相关随机效应]]"
  - "[[随机效应正交假设]]"
related:
  - "[[Hausman检验]]"
---

# Mundlak 检验通过联合检验 CRE 中的回归量单位均值系数诊断标准 RE 正交限制

<!-- bilingual-en:start -->
*The Mundlak test diagnoses the standard RE orthogonality restriction by jointly testing the coefficients on regressor unit means in CRE*
<!-- bilingual-en:end -->

> [!summary] 原子命题
> 在 CRE 方程中加入各时变回归量的单位均值 $\bar x_i$ 后，Mundlak 检验对
> $$
> H_0:\pi=0
> $$
> 作联合 Wald 检验。拒绝说明单位效应中存在与回归量路径系统相关、不能由标准 RE 正交限制忽略的部分。
>
> <!-- bilingual-en:start -->
> After adding the unit means $\bar x_i$ of the time-varying regressors to a CRE equation, the Mundlak test conducts a joint Wald test of $H_0:\pi=0$. Rejection indicates a systematic component of the unit effect related to the regressor path that standard RE orthogonality cannot ignore.
> <!-- bilingual-en:end -->

这个检验可在 augmented/CRE regression 中使用适合问题的 robust 或 cluster-robust VCE。它没有把“不拒绝”变成假设证明：有限样本或弱信息可能使检验没有发现差异，所以结论只能是当前规格下未检出均值项的联合作用。
<!-- bilingual-en:start -->
The test can be implemented in the augmented CRE regression with a robust or cluster-robust VCE appropriate to the problem. Nonrejection is not proof of the assumption: finite samples or weak information may conceal a difference, so the conclusion is only that the joint mean terms were not detected in the current specification.
<!-- bilingual-en:end -->

> [!question]- 回忆与应用
> Mundlak 检验 $p=0.40$ 最多说明什么？
>
> **答案：** 当前样本、均值构造、规格和 VCE 下没有检出 $\pi$ 的联合偏离；不能证明 $E(c_i\mid X_i)=0$。

## 来源与核验

- StataCorp, [xtreg manual](https://www.stata.com/manuals/xtxtreg.pdf), CRE remarks and Methods：把 Mundlak test 定义为 CRE 中单位均值系数联合为 0 的检验，并给出 robust/cluster-robust 实现。
- Wooldridge (2019), [“Correlated random effects models with unbalanced panels”](https://doi.org/10.1016/j.jeconom.2018.12.010), eqs. (2.9)–(2.10)：给出非平衡面板下单位均值项联合检验及预检验解释边界。

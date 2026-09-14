---
aliases:
  - "Johansen trace 与最大特征值统计量检验不同的秩假设"
  - Johansen trace and maximum-eigenvalue tests
  - Johansen 两种秩检验
student_os: knowledge-atom
atom_id: TS-CI-020
atom_set: cointegration-error-correction
atom_type: test-comparison
status: source-checked
mastery_state: unassessed
requires:
  - "[[Johansen广义特征值]]"
related:
  - "[[Johansen规格选择]]"
part_of:
  - "[[协整与误差修正模型.canvas]]"
---

# Johansen trace 与最大特征值统计量检验不同的秩假设
<!-- bilingual-en:start -->
*Johansen trace and maximum-eigenvalue statistics test different rank hypotheses*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> trace 检验累计考察 $r$ 之后的所有典型根，原假设为协整秩至多 $r$；最大特征值检验只考察下一个根，比较秩 $r$ 与 $r+1$。它们不是同一统计量的两个名字。

按 $\hat\lambda_1\ge\cdots\ge\hat\lambda_n$ 排序，常用形式是
$$
LR_{\text{trace}}(r)=-T\sum_{i=r+1}^{n}\log(1-\hat\lambda_i),
$$
$$
LR_{\max}(r,r+1)=-T\log(1-\hat\lambda_{r+1}).
$$
实际常从 $r=0$ 起顺序检验，第一次未拒绝的位置形成秩选择证据；多次顺序检验、有限样本校正和软件的有效样本量定义都应记录。

两种统计量可能给出不同秩。最大特征值的备择更窄，不等于它普遍更“准确”；应检查滞后、确定性项、残差、样本区间和经济可解释性，并透明报告分歧。

> [!question]- 自检
> trace 在 $r=1$ 未拒绝而 max-eigen 在相邻步骤给出不同结论，能否只挑支持预期的一个？
>
> **答案：** 不能。应报告两者，并检查规格与稳健性，而不是事后择优。

## 来源与核验

- [Johansen (1988)](https://doi.org/10.1016/0165-1889(88)90041-3)：核对似然比秩检验。
- [Johansen (1991)](https://researchprofiles.ku.dk/en/publications/estimation-and-hypothesis-testing-of-cointegration-vectors-in-gau/)：核对高斯 VAR 中的秩与参数检验。

---
aliases:
  - "Hausman 检验只诊断相容估计量差异而不能用 p 值机械选择 FE 或 RE"
  - Hausman specification-test boundary
student_os: knowledge-atom
atom_id: ECON-PANEL-012
atom_set: panel-data
atom_type: diagnostic-boundary
status: source-checked
mastery_state: unassessed
part_of:
  - "[[面板数据：Pooled OLS、固定效应与随机效应.canvas]]"
requires:
  - "[[Hausman检验]]"
related:
  - "[[固定效应因果边界]]"
  - "[[Mundlak检验]]"
---

# Hausman 检验只诊断相容估计量差异而不能用 p 值机械选择 FE 或 RE

<!-- bilingual-en:start -->
*The Hausman test diagnoses differences between compatible estimators and cannot mechanically select FE or RE from a p-value*
<!-- bilingual-en:end -->

> [!summary] 原子命题
> FE–RE Hausman 检验只有在两次估计针对同一组共同参数、使用同一样本与可比规格，而且检验所需的协方差差具有相应效率解释时，才诊断 RE 限制。把结果缩成“$p$ 小选 FE，$p$ 大选 RE”会丢掉这些前提。
>
> <!-- bilingual-en:start -->
> An FE–RE Hausman test diagnoses the restrictions supporting RE only when both estimators target the same common parameters on the same sample with compatible specifications, and when the covariance difference has the required efficiency interpretation. The mnemonic “small $p$: FE; large $p$: RE” discards those premises.
> <!-- bilingual-en:end -->

经典 covariance-difference 实现不能机械接收彼此不相容的 robust、clustered 或 weighted VCE。协方差差非正定、共同系数集合不同或估计噪声很大时，检验可能没有可用的判别力；应先检查实现和比较对象，而不是只读最后一行 $p$ 值。
<!-- bilingual-en:start -->
The classical covariance-difference implementation cannot mechanically combine incompatible robust, clustered, or weighted covariance estimates. A non-positive-definite covariance difference, different sets of common coefficients, or noisy estimates can leave the test without useful discriminatory power; the implementation and comparison target must be checked before reading the final p-value.
<!-- bilingual-en:end -->

不拒绝只表示当前数据和规格没有检出系统差异，不证明 [[随机效应正交假设]]；拒绝也不使 FE 系数自动具有因果解释，后者仍受 [[固定效应因果边界]] 约束。需要 cluster-robust 诊断时，可考虑在 CRE 增广回归中实施 [[Mundlak检验]]。
<!-- bilingual-en:start -->
Nonrejection means only that the current data and specification did not detect a systematic difference; it does not prove RE orthogonality. Rejection likewise does not make the FE slope causal. When a cluster-robust diagnostic is needed, the Mundlak test in an augmented CRE regression may be more transparent.
<!-- bilingual-en:end -->

> [!question]- 回忆与应用
> Hausman 检验 $p=0.40$ 最多允许说什么？
>
> **答案：** 在当前样本、共同系数、规格和检验实现下，没有检测到足够大的系统差异；不能说 RE 限制已经被证明。

## 来源与核验

- StataCorp, [hausman manual](https://www.stata.com/manuals/rhausman.pdf), pp.2–3, 8–9：核验共同参数、效率前提、协方差差与 robust/clustered 实现边界。
- [[固定效应因果边界]]：区分“更偏好 FE 规格”与“FE 已经识别因果效应”。
- [[Mundlak检验]]：提供能在 augmented CRE regression 中配合适当 VCE 的相关性诊断。

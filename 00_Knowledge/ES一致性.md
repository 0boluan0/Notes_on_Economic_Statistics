---
aliases:
  - "对可积损失正确定义的 ES 是一致风险度量"
  - Expected Shortfall is coherent for integrable losses
student_os: knowledge-atom
atom_id: RM-VAR-021
atom_set: var-es-backtesting
atom_type: proposition
status: source-checked
mastery_state: unassessed
requires:
  - "[[ES定义]]"
  - "[[一致风险度量]]"
contrasts_with:
  - "[[VaR非次可加]]"
related:
  - "[[光谱风险度量]]"
  - "[[ES严格超越均值边界]]"
  - "[[ES是光谱特例]]"
part_of:
  - "[[VaR、ES与回测.canvas|VaR、ES与回测]]"
---

# 对可积损失正确定义的 ES 是一致风险度量
<!-- bilingual-en:start -->
*Properly defined Expected Shortfall is a coherent risk measure for integrable losses*
<!-- bilingual-en:end -->

> [!summary] ES 的尾部平均满足四项一致性公理
> 以分位积分或等价的一般分布定义计算的 ES 具有单调性、现金平移、正齐次与次可加。离散分布中若错误地把 ES 定义为严格超越 VaR 的条件均值，这个结论可能失去适用对象。
>
> <!-- bilingual-en:start -->
> ES defined by the quantile integral, or an equivalent general-distribution formula, is monotone, cash-translation invariant, positively homogeneous, and subadditive. Replacing it in a discrete distribution by a strict-exceedance conditional mean can change the object.
> <!-- bilingual-en:end -->

对可积损失 $L$，

$$
\operatorname{ES}_{\alpha}(L)
=\frac{1}{1-\alpha}\int_\alpha^1 q_u(L)\,du.
$$

于是任意可积 $L_1,L_2$ 满足

$$
\operatorname{ES}_{\alpha}(L_1+L_2)
\le
\operatorname{ES}_{\alpha}(L_1)
+\operatorname{ES}_{\alpha}(L_2).
$$

这条次可加性允许把组合风险控制在分项 ES 之和以内，但不证明所用联合分布或尾部参数正确。

<!-- bilingual-en:start -->
Subadditivity bounds portfolio ES by the sum of stand-alone ES, but it does not validate the fitted joint distribution or tail parameters.
<!-- bilingual-en:end -->

> [!question]- 自检
> “ES 一致”是否意味着 ES 估计比 VaR 更稳定？
>
> **答案：** 不意味着。一致性描述聚合公理；高置信度 ES 仍可能因尾部样本少而不稳定。
>
> <!-- bilingual-en:start -->
> **Self-check:** Does the coherence of ES imply that ES estimates are more stable than VaR estimates?
>
> **Answer:** No. Coherence describes aggregation axioms; high-confidence ES can remain unstable because tail observations are scarce.
> <!-- bilingual-en:end -->

## 边界

- 需要相应尾部可积；ES 可能为无穷。
- 一致性不等于模型准确、监管合格或易回测。
- 分位点有概率质量时必须使用 [[ES定义]] 的一般公式。

<!-- bilingual-en:start -->
- Tail integrability is required; ES may be infinite.
- Coherence is not model accuracy, regulatory approval, or easy backtesting.
- Probability mass at the quantile requires the general [[ES定义|ES definition]].
<!-- bilingual-en:end -->

## 来源与核验

- [Acerbi & Tasche, *On the Coherence of Expected Shortfall*](https://arxiv.org/abs/cond-mat/0104295)：核对一般分布下 ES 的稳健定义与一致性。
- [Artzner et al., *Coherent Measures of Risk*](https://doi.org/10.1111/1467-9965.00068)：核对四项一致性公理。

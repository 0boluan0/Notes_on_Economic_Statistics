---
aliases:
  - "VaR 在一般损失分布下可能违反次可加"
  - VaR may fail subadditivity
  - VaR 次可加边界
student_os: knowledge-atom
atom_id: RM-VAR-005
atom_set: var-es-backtesting
atom_type: boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[VaR定义]]"
  - "[[一致风险度量]]"
related:
  - "[[相同EL不同尾部]]"
contrasts_with:
  - "[[ES一致性]]"
part_of:
  - "[[VaR、ES与回测.canvas|VaR、ES与回测]]"
---

# VaR 在一般损失分布下可能违反次可加
<!-- bilingual-en:start -->
*VaR can fail subadditivity for general loss distributions*
<!-- bilingual-en:end -->

> [!summary] 分项 VaR 为零，组合 VaR 仍可能为正
> VaR 只读取一个分位点。两个头寸各自的罕见损失可以落在分位点之外，但合并后“至少一个损失发生”的概率可能跨过分位阈值，使组合 VaR 大于分项 VaR 之和。
>
> <!-- bilingual-en:start -->
> VaR reads one quantile. Each position's rare loss can lie beyond that quantile, while the probability that at least one loss occurs can cross the threshold after aggregation, making portfolio VaR exceed the sum of stand-alone VaRs.
> <!-- bilingual-en:end -->

令 $L_1,L_2$ 相互独立，且各自以 4% 概率损失 100、以 96% 概率损失 0。则

<!-- bilingual-en:start -->
Let independent $L_1,L_2$ each equal 100 with probability 4% and 0 with probability 96%. Then
<!-- bilingual-en:end -->

$$
\operatorname{VaR}_{.95}(L_1)
=\operatorname{VaR}_{.95}(L_2)=0.
$$

但组合零损失概率只有 $0.96^2=0.9216<0.95$，而 $P(L_1+L_2\le100)=0.9984$，所以

<!-- bilingual-en:start -->
But the portfolio has zero loss with probability only $0.96^2=0.9216<0.95$, while $P(L_1+L_2\le100)=0.9984$. Therefore
<!-- bilingual-en:end -->

$$
\operatorname{VaR}_{.95}(L_1+L_2)=100
>
\operatorname{VaR}_{.95}(L_1)+\operatorname{VaR}_{.95}(L_2)=0.
$$

这违反 [[一致风险度量]] 的次可加公理。结论是 VaR **可能**违反次可加，不是 VaR 对所有分布和组合都违反次可加。

<!-- bilingual-en:start -->
This violates the subadditivity axiom of a [[一致风险度量|coherent risk measure]]. The conclusion is that VaR can fail subadditivity, not that it fails for every distribution and portfolio.
<!-- bilingual-en:end -->

> [!question]- 自检
> 为什么两个 95% VaR 都为 0 的头寸，组合 95% VaR 可以为 100？
>
> **答案：** 单项零损失概率为 96%，但两者同时零损失的概率只有 92.16%；组合的 95% 分位因此上移到 100。
>
> <!-- bilingual-en:start -->
> **Self-check:** Why can two positions with individual 95% VaR equal to zero have portfolio 95% VaR equal to 100?
>
> **Answer:** Each position has 96% probability of zero loss, but both are zero together only 92.16% of the time, so the portfolio's 95% quantile moves to 100.
> <!-- bilingual-en:end -->

## 边界

- 次可加失效是风险函数的聚合边界，不证明联合分布估计正确或错误。
- VaR 在椭圆分布等特定条件下可以满足次可加。
- 不得把“可能违反”改写成“VaR 总是惩罚分散化”。

<!-- bilingual-en:start -->
- Failure of subadditivity is a property of the risk functional, not proof about the fitted joint distribution.
- VaR can be subadditive under particular structures such as elliptical distributions.
- “Can fail” must not be rewritten as “always penalises diversification.”
<!-- bilingual-en:end -->

## 来源与核验

- [Artzner et al., *Coherent Measures of Risk*](https://doi.org/10.1111/1467-9965.00068)：核对次可加公理与分位型风险度量可能失效的结论。
- [Acerbi & Tasche, *On the Coherence of Expected Shortfall*](https://arxiv.org/abs/cond-mat/0104295)：交叉核对 VaR 与 ES 的聚合性质边界。

---
aliases:
  - Alpha 是相对于指定定价模型的回归截距而不是无条件能力
  - Alpha依赖基准模型
  - Jensen's alpha is model relative
  - Alpha benchmark dependence
student_os: knowledge-atom
atom_id: INV-CAPM-010
atom_set: capm-systematic-risk
atom_type: performance-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[证券市场线]]"
  - "[[Beta历史估计]]"
part_of:
  - "[[CAPM、系统风险与资本成本.canvas]]"
related:
  - "[[市场效率联合假设]]"
  - "[[异常收益]]"
  - "[[HAC协方差]]"
---

# Alpha 是相对于指定定价模型的回归截距而不是无条件能力
<!-- bilingual-en:start -->
*Alpha is a regression intercept relative to a specified pricing model, not unconditional skill*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> Jensen alpha 来自回归
> $$R_{p,t}-R_{f,t}=\alpha_p+\beta_p(R_{m,t}-R_{f,t})+\varepsilon_{p,t}.$$
> $\alpha_p$ 表示在所选 CAPM 基准、市场代理、样本期和收益口径下，组合平均超额收益中没有被 beta 项解释的截距。换成多因子模型、不同市场代理或不同费用口径，alpha 可以改变。
> <!-- bilingual-en:start -->
> Jensen's alpha is the intercept in $R_{p,t}-R_{f,t}=\alpha_p+\beta_p(R_{m,t}-R_{f,t})+\varepsilon_{p,t}$. It is the average excess return not explained by beta under the selected CAPM benchmark, market proxy, sample period, and return convention. A multifactor model, another proxy, or another fee convention can change alpha.
> <!-- bilingual-en:end -->

单期“实际收益减 CAPM 要求收益”是一次已实现异常收益，不等于已经估准的长期 alpha。把它解释为能力，需要足够长的有效样本、对序列相关与异方差相容的不确定性估计、与策略相符的风险基准、费用和交易成本后的回报，以及样本外持续性证据；回归截距显著也可能来自遗漏因子、偶然性或选择性报告。
<!-- bilingual-en:start -->
A one-period realized return minus the CAPM-implied return is one realized abnormal return, not a precisely estimated long-run alpha. Interpreting it as skill requires a sufficiently long effective sample, uncertainty estimates robust to serial dependence and heteroskedasticity, a risk benchmark appropriate to the strategy, returns after fees and trading costs, and out-of-sample persistence. A significant intercept can still reflect omitted factors, chance, or selective reporting.
<!-- bilingual-en:end -->

当同时搜索许多基金、因子或参数时，总会有一部分仅凭运气显著。应报告搜索范围并做多重检验或留出样本校正。alpha 的正确结论格式不是“经理有能力”，而是“在某模型、某代理、某期间和某成本口径下，证据是否支持一个非零截距”。
<!-- bilingual-en:start -->
When many funds, factors, or specifications are searched, some will appear significant by chance. Report the search universe and use multiple-testing or holdout corrections. The proper conclusion is not “the manager has skill,” but whether evidence supports a nonzero intercept under a stated model, proxy, period, and cost convention.
<!-- bilingual-en:end -->

> [!question]- 自检
> 市场跌 30% 时，一个低 beta 基金只跌 10%。为什么“比市场跌得少”还不能证明正 alpha？
>
> **答案：** 要先按该基金的 beta 和无风险利率算模型基准；低 beta 本来就意味着较小市场暴露。即使当期异常收益为正，还要靠多期、扣费且模型相容的证据区分能力与运气。

## 来源与核验

- [Jensen (1968), “The Performance of Mutual Funds in the Period 1945–1964”](https://doi.org/10.1111/j.1540-6261.1968.tb00815.x)：核对风险调整回归、alpha 截距、费用口径和抽样显著性。
- [Harvey, Liu & Zhu (2016), “…and the Cross-Section of Expected Returns”](https://doi.org/10.1093/rfs/hhv059)：核对大规模策略/因子搜索中的多重检验边界。
- [[HAC协方差]]：核对时间序列回归中标准误如何同时允许异方差与弱序列依赖。
- [[02_Economy/07_金融机构与风险管理/01_引言.md#1.18|课程绩效例题]]：核对低 beta 组合必须先做风险调整；本卡把单期异常收益与长期 alpha 估计分开。

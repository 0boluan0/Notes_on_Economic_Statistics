---
aliases:
  - "历史模拟 Bootstrap 通过重抽历史情景估计风险统计量的抽样分布"
  - Historical-simulation bootstrap
student_os: knowledge-atom
atom_id: RM-VAR-031
atom_set: var-es-backtesting
atom_type: resampling-method
status: source-checked
mastery_state: unassessed
requires:
  - "[[历史模拟法]]"
  - "[[VaR采样误差]]"
related:
  - "[[伪随机复现]]"
part_of:
  - "[[风险模拟.canvas|风险模拟]]"
---

# 历史模拟 Bootstrap 通过重抽历史情景估计风险统计量的抽样分布
<!-- bilingual-en:start -->
*Historical-simulation bootstrap estimates the sampling distribution of a risk statistic by resampling historical scenarios*
<!-- bilingual-en:end -->

> [!summary] 重抽已有信息，不创造新尾部
> 普通 Bootstrap 从已有历史行有放回抽样，在每个重抽样样本上重算 VaR 或 ES，以近似估计量的抽样分布、偏差或置信区间。
>
> <!-- bilingual-en:start -->
> An ordinary bootstrap samples historical rows with replacement and recomputes VaR or ES in every replicate to approximate the estimator's sampling distribution, bias, or confidence interval.
> <!-- bilingual-en:end -->

若原样本为 $Z_1,\ldots,Z_n$，第 $b$ 个 replicate 从经验分布 $\widehat F_n$ 抽取

$$
Z_1^{*(b)},\ldots,Z_n^{*(b)}
\overset{iid}{\sim}\widehat F_n,
$$

再计算 $\widehat q_\alpha^{*(b)}$。多个 replicate 的分位数可构造 percentile 区间。重抽会重复已有冲击，但不会生成原窗口之外的风险因子尾部。

<!-- bilingual-en:start -->
Compute $\widehat q_\alpha^{*(b)}$ in each replicate and use the replicate distribution, for example, to form a percentile interval. Resampling repeats observed shocks but does not generate factor tails outside the original window.
<!-- bilingual-en:end -->

若历史序列有依赖，逐日 iid 重抽会破坏波动聚集和持续性；应采用与目标依赖结构匹配的 block 或 stationary bootstrap。

<!-- bilingual-en:start -->
When history is dependent, iid row resampling destroys volatility clustering and persistence; use a block or stationary bootstrap suited to the target dependence.
<!-- bilingual-en:end -->

> [!question]- 自检
> 把 500 个历史日 Bootstrap 成一万条记录，是否得到一万天新历史证据？
>
> **答案：** 没有。只是在重复原 500 天，用于描述给定重抽机制下的估计不确定性。
>
> <!-- bilingual-en:start -->
> **Self-check:** If 500 historical days are bootstrapped into ten thousand records, does that create ten thousand days of new historical evidence?
>
> **Answer:** No. The procedure only repeats the original 500 days to describe estimation uncertainty under the chosen resampling mechanism.
> <!-- bilingual-en:end -->

## 边界

- Bootstrap 不能把未观察到的制度或极端冲击变成证据。
- 参数化或过滤管线若要纳入不确定性，每个 replicate 应重跑相应估计步骤。
- 置信区间有效性仍依赖统计量与重抽方案的条件。

<!-- bilingual-en:start -->
- Bootstrap cannot turn unobserved regimes or extremes into evidence.
- To propagate parameter or filtering uncertainty, rerun those estimation steps in each replicate.
- Interval validity still depends on the statistic and resampling design.
<!-- bilingual-en:end -->

## 来源与核验

- Efron, [*Bootstrap Methods: Another Look at the Jackknife*](https://doi.org/10.1214/aos/1176344552)：核对经验分布有放回重抽与抽样分布估计。
- Politis & Romano, [*The Stationary Bootstrap*](https://doi.org/10.1080/01621459.1994.10476870)：核对时间依赖下的区块化重抽。

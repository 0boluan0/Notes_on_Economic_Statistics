---
aliases:
  - "绝对 VaR 以零 P&L 为基准，相对 VaR 以预先确定的基准 P&L 为基准"
  - Absolute and relative VaR use different P&L benchmarks
student_os: knowledge-atom
atom_id: RM-VAR-019
atom_set: var-es-backtesting
atom_type: distinction
status: source-checked
mastery_state: unassessed
requires:
  - "[[风险度量口径]]"
  - "[[VaR定义]]"
part_of:
  - "[[VaR、ES与回测.canvas|VaR、ES与回测]]"
---

# 绝对 VaR 以零 P&L 为基准，相对 VaR 以预先确定的基准 P&L 为基准
<!-- bilingual-en:start -->
*Absolute VaR uses zero P&L as its benchmark, while relative VaR uses a pre-specified benchmark P&L*
<!-- bilingual-en:end -->

> [!summary] 两者差在比较基准
> 若 $X$ 是未来 P&L、$L=-X$ 是正损失，绝对 VaR 读取 $L$ 的分位点；相对 VaR 读取相对确定性基准 P&L $b$ 的不足额 $b-X$ 的分位点。
>
> <!-- bilingual-en:start -->
> If $X$ is future P&L and $L=-X$ is positive loss, absolute VaR is a quantile of $L$. Relative VaR is a quantile of the shortfall $b-X$ from a deterministic benchmark P&L $b$.
> <!-- bilingual-en:end -->

$$
\operatorname{VaR}^{\mathrm{abs}}_\alpha=q_\alpha(L),
\qquad
\operatorname{VaR}^{\mathrm{rel}}_\alpha(b)
=q_\alpha(b-X)
=q_\alpha(L)+b.
$$

若绝对 VaR 为 10，预先确定的预期利润基准为 2，则相对 VaR 为 12：分位情形不但亏损 10，还比预期少赚 2。

<!-- bilingual-en:start -->
If absolute VaR is 10 and the pre-specified expected-profit benchmark is 2, relative VaR is 12: the quantile outcome loses 10 and also forgoes the expected gain of 2.
<!-- bilingual-en:end -->

若基准 $B$ 本身随机，相对损失是 $B-X$，必须使用 $(B,X)$ 的联合分布；不能把随机基准当作常数平移。

<!-- bilingual-en:start -->
If the benchmark $B$ is random, relative loss is $B-X$ and requires the joint distribution of $(B,X)$; it is not a constant shift.
<!-- bilingual-en:end -->

> [!question]- 自检
> 绝对 VaR 为 10、确定性利润基准为 2 时，相对 VaR 是多少？
>
> **答案：** 12。若基准随机，则不能用这个加法。
>
> <!-- bilingual-en:start -->
> **Self-check:** If absolute VaR is 10 and the deterministic profit benchmark is 2, what is relative VaR?
>
> **Answer:** It is 12. This addition is invalid when the benchmark is random.
> <!-- bilingual-en:end -->

## 边界

- 基准必须在风险预测前确定。
- “相对”描述基准，不表示风险数字已经中心化、标准化或验证。
- 不同符号约定下应从相对损失重新推导，不机械套用加号。

<!-- bilingual-en:start -->
- The benchmark must be fixed before the forecast.
- “Relative” names the benchmark; it does not imply validation or standardisation.
- Under another sign convention, re-derive the shortfall rather than copying the plus sign.
<!-- bilingual-en:end -->

## 来源与核验

- [Rockafellar & Uryasev, *Conditional Value-at-Risk for General Loss Distributions*](https://sites.math.washington.edu/~rtr/papers/rtr187-CVaR2.pdf)：核对损失变量与分位型风险度量；确定性基准关系由分位点平移直接推出。
- [[02_Economy/07_金融机构与风险管理/12_VAR风险]]：核对课程中绝对与相对 VaR 的数值语境。

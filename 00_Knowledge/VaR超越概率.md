---
aliases:
  - "分位点存在概率质量时，严格超过 VaR 的概率可以小于一减置信水平"
  - VaR exceedance probability at a mass point
  - VaR 例外概率边界
student_os: knowledge-atom
atom_id: RM-VAR-037
atom_set: var-es-backtesting
atom_type: boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[VaR定义]]"
  - "[[累积分布函数]]"
related:
  - "[[VaR回测损益口径]]"
  - "[[Kupiec无条件覆盖]]"
  - "[[尾界含义]]"
part_of:
  - "[[VaR、ES与回测.canvas|VaR、ES与回测]]"
---

# 分位点存在概率质量时，严格超过 VaR 的概率可以小于一减置信水平
<!-- bilingual-en:start -->
*When probability mass lies at the quantile, the probability of strictly exceeding VaR can be smaller than one minus the confidence level*
<!-- bilingual-en:end -->

> [!summary] VaR 固定分位位置，不保证严格超越率恰好命中
> 对左分位点 $q_\alpha=\operatorname{VaR}_\alpha(L)$，定义只保证严格超越概率不超过 $1-\alpha$。若分位点上积聚了概率质量，累计概率会跨过 $\alpha$，于是严格超越率可以更小。
>
> <!-- bilingual-en:start -->
> For the left quantile $q_\alpha=\operatorname{VaR}_\alpha(L)$, the definition guarantees only that the strict exceedance probability is no greater than $1-\alpha$. Probability mass at the quantile can make cumulative probability jump across $\alpha$, leaving a smaller strict exceedance rate.
> <!-- bilingual-en:end -->

精确边界是

$$
P(L>q_\alpha)
\le 1-\alpha
\le P(L\ge q_\alpha).
$$

只有当 $F_L(q_\alpha)=\alpha$ 时，才有

$$
P(L>q_\alpha)=1-\alpha.
$$

例如

$$
P(L=0)=0.995,
\qquad
P(L=100)=0.005.
$$

在 $\alpha=0.99$ 时，$q_{0.99}=0$，但

$$
P(L>q_{0.99})=0.005<0.01.
$$

因此，“99% VaR”不能仅凭定义被翻译成“损失严格超过 VaR 的概率恰好是 1%”。实际回测还必须事先固定使用 $>$ 还是 $\ge$，并让例外定义与统计检验一致。

<!-- bilingual-en:start -->
Thus, “99% VaR” cannot be translated by definition alone into “loss strictly exceeds VaR with probability exactly 1%.” Backtesting must pre-specify whether an exception uses $>$ or $\ge$ and keep that convention aligned with the test.
<!-- bilingual-en:end -->

> [!question]- 自检
> 上例的 99% VaR 为 0，为什么严格超越率不是 1%？
>
> **答案：** 0 点累计了 99.5% 的概率，累计分布在该点跨过 99%；剩在其上的概率只有 0.5%。
>
> <!-- bilingual-en:start -->
> **Self-check:** In the example, 99% VaR is zero. Why is the strict exceedance rate not 1%?
>
> **Answer:** The mass at zero raises cumulative probability to 99.5%, jumping across 99%; only 0.5% remains strictly above the quantile.
> <!-- bilingual-en:end -->

## 来源与核验

- [Rockafellar & Uryasev, *Conditional Value-at-Risk for General Loss Distributions*](https://sites.math.washington.edu/~rtr/papers/rtr187-CVaR2.pdf)：核对一般分布中的左分位点与分位点概率质量。
- [[累积分布函数]]：承载左分位点及其概率夹界。

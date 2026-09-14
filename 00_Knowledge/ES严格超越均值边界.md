---
aliases:
  - "ES 只有在 VaR 分位点的累计概率恰好等于 alpha 时才等于严格超越 VaR 的条件均值"
  - ES strict-exceedance conditional-mean boundary
  - 离散分布中 ES 不等于严格超越均值
student_os: knowledge-atom
atom_id: RM-VAR-039
atom_set: var-es-backtesting
atom_type: boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[ES定义]]"
  - "[[VaR定义]]"
  - "[[条件期望]]"
related:
  - "[[VaR超越概率]]"
  - "[[超越均值不足以验证ES]]"
part_of:
  - "[[VaR、ES与回测.canvas|VaR、ES与回测]]"
---

# ES 只有在 VaR 分位点的累计概率恰好等于 alpha 时才等于严格超越 VaR 的条件均值
<!-- bilingual-en:start -->
*ES equals the conditional mean strictly above VaR only when cumulative probability at the VaR quantile is exactly alpha*
<!-- bilingual-en:end -->

> [!summary] 条件均值简写需要尾部概率恰好对齐
> 令 $q=q_\alpha(L)$。一般 ES 平均的是完整的最坏 $1-\alpha$ 概率质量；只有当 $F_L(q)=\alpha$ 时，严格超越事件本身才恰好占据这部分概率，因而可以写成 $E[L\mid L>q]$。
>
> <!-- bilingual-en:start -->
> Let $q=q_\alpha(L)$. General ES averages the complete worst $1-\alpha$ probability mass. Only when $F_L(q)=\alpha$ does the strict exceedance event itself contain exactly that mass, allowing ES to be written as $E[L\mid L>q]$.
> <!-- bilingual-en:end -->

一般公式是

$$
\operatorname{ES}_{\alpha}(L)
=\frac{
E\!\left[L\mathbf 1_{\{L>q\}}\right]
+q\left(F_L(q)-\alpha\right)
}{1-\alpha}.
$$

若 $F_L(q)=\alpha$，第二项为零且 $P(L>q)=1-\alpha$，于是

$$
\operatorname{ES}_{\alpha}(L)=E[L\mid L>q].
$$

反之，考虑

$$
P(L=0)=0.95,
\qquad
P(L=100)=0.04,
\qquad
P(L=1000)=0.01.
$$

取 $\alpha=0.975$，有 $q=100$。最坏 2.5% 包含全部 1% 的 1000 损失和 1.5% 的 100 损失，因此

$$
\operatorname{ES}_{0.975}(L)
=\frac{0.01(1000)+0.015(100)}{0.025}
=460.
$$

但 $E[L\mid L>100]=1000$，因为严格超越事件只选中了最坏 1%，没有补足定义要求的 2.5%。机械改成 $L\ge100$ 又会选入全部 5% 的尾部质量，同样不对。

<!-- bilingual-en:start -->
However, $E[L\mid L>100]=1000$ because strict exceedance selects only the worst 1%, not the required 2.5%. Replacing $>$ mechanically by $\ge$ includes the entire 5% tail mass and is also incorrect.
<!-- bilingual-en:end -->

> [!question]- 自检
> 上例中为什么 ES 既不是 $E[L\mid L>100]$，也不是 $E[L\mid L\ge100]$？
>
> **答案：** 前者只平均 1%，后者平均 5%；ES 必须平均恰好最坏 2.5% 的概率质量。
>
> <!-- bilingual-en:start -->
> **Self-check:** Why is ES in the example neither $E[L\mid L>100]$ nor $E[L\mid L\ge100]$?
>
> **Answer:** The former averages only 1%, while the latter averages 5%. ES must average exactly the worst 2.5% probability mass.
> <!-- bilingual-en:end -->

## 来源与核验

- [Acerbi & Tasche, *On the Coherence of Expected Shortfall*](https://arxiv.org/abs/cond-mat/0104295)：核对非连续分布中 ES 与尾部条件均值的区别。
- [Rockafellar & Uryasev, *Conditional Value-at-Risk for General Loss Distributions*](https://sites.math.washington.edu/~rtr/papers/rtr187-CVaR2.pdf)：核对分位点概率质量的补足公式。

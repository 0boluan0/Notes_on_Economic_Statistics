---
aliases:
  - "ES 是在上尾使用常数权重的光谱风险度量"
  - Expected Shortfall is a spectral risk measure
  - ES 的阶梯光谱权重
student_os: knowledge-atom
atom_id: RM-VAR-040
atom_set: var-es-backtesting
atom_type: proposition
status: source-checked
mastery_state: unassessed
requires:
  - "[[ES定义]]"
  - "[[光谱风险度量]]"
related:
  - "[[ES一致性]]"
part_of:
  - "[[VaR、ES与回测.canvas|VaR、ES与回测]]"
---

# ES 是在上尾使用常数权重的光谱风险度量
<!-- bilingual-en:start -->
*Expected Shortfall is a spectral risk measure with constant weight over the upper loss tail*
<!-- bilingual-en:end -->

> [!summary] ES 对阈值以上的分位位置等权
> 置信水平 $\alpha$ 下的 ES 对 $u\le\alpha$ 的损失分位点赋零权重，对 $u>\alpha$ 的分位点赋相同正权重 $1/(1-\alpha)$。这个阶梯权重非负、归一且不减，因此属于光谱风险度量允许的权重类。
>
> <!-- bilingual-en:start -->
> ES at confidence level $\alpha$ assigns zero weight to loss quantiles with $u\le\alpha$ and the same positive weight $1/(1-\alpha)$ to quantiles with $u>\alpha$. This step weight is non-negative, normalized, and non-decreasing, so it belongs to the admissible spectral class.
> <!-- bilingual-en:end -->

定义

$$
\phi_\alpha(u)
=
\begin{cases}
0,&u\le\alpha,\\[4pt]
\dfrac{1}{1-\alpha},&u>\alpha.
\end{cases}
$$

则

$$
\int_0^1\phi_\alpha(u)\,du=1
$$

且 $\phi_\alpha$ 随 $u$ 不减。代入 [[光谱风险度量]] 的定义得到

$$
\rho_{\phi_\alpha}(L)
=\int_0^1\phi_\alpha(u)q_u(L)\,du
=\frac{1}{1-\alpha}\int_\alpha^1q_u(L)\,du
=\operatorname{ES}_\alpha(L).
$$

单点 $u=\alpha$ 取左值还是右值不会改变积分。这里的“等权”是对**分位水平**等权，不是说有限样本中每个不同损失数值都必须出现相同次数。

<!-- bilingual-en:start -->
The value assigned at the single point $u=\alpha$ does not change the integral. “Equal weight” here means equal weight per quantile level, not equal frequency for every distinct numerical loss in a finite sample.
<!-- bilingual-en:end -->

> [!question]- 自检
> ES 的光谱权重为什么满足归一化？
>
> **答案：** 权重只在长度为 $1-\alpha$ 的区间上取 $1/(1-\alpha)$，积分正好为一。
>
> <!-- bilingual-en:start -->
> **Self-check:** Why is the spectral weight for ES normalized?
>
> **Answer:** It equals $1/(1-\alpha)$ on an interval of length $1-\alpha$, so its integral is exactly one.
> <!-- bilingual-en:end -->

## 来源与核验

- [Acerbi, *Spectral Measures of Risk: A Coherent Representation of Subjective Risk Aversion*](https://doi.org/10.1016/S0378-4266(02)00281-9)：核对光谱权重条件及 ES 的阶梯权重表示。
- [Acerbi & Tasche, *On the Coherence of Expected Shortfall*](https://arxiv.org/abs/cond-mat/0104295)：核对 ES 的分位积分表示。

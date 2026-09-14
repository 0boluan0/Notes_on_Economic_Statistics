---
aliases:
  - "Christoffersen 独立性检验比较上一期是否例外后的两种一阶转移概率"
  - Christoffersen independence test
  - VaR 例外聚集检验
student_os: knowledge-atom
atom_id: RM-VAR-023
atom_set: var-es-backtesting
atom_type: diagnostic-test
status: source-checked
mastery_state: unassessed
requires:
  - "[[VaR回测损益口径]]"
related:
  - "[[GARCH选择与预测评估]]"
  - "[[独立同分布]]"
leads_to:
  - "[[Christoffersen条件覆盖]]"
part_of:
  - "[[VaR、ES与回测.canvas|VaR、ES与回测]]"
---

# Christoffersen 独立性检验比较上一期是否例外后的两种一阶转移概率
<!-- bilingual-en:start -->
*The Christoffersen independence test compares first-order exception probabilities after a hit and after no hit*
<!-- bilingual-en:end -->

> [!summary] 它看例外顺序，不固定平均例外率
> 纯独立性检验问 $P(I_t=1\mid I_{t-1}=0)$ 是否等于 $P(I_t=1\mid I_{t-1}=1)$。共同概率由转移样本估计，不固定为名义尾概率。
>
> <!-- bilingual-en:start -->
> The pure independence test asks whether $P(I_t=1\mid I_{t-1}=0)$ equals $P(I_t=1\mid I_{t-1}=1)$. Their common probability is estimated from the transition sample, not fixed at the nominal tail probability.
> <!-- bilingual-en:end -->

令 $n_{ij}$ 统计 $I_{t-1}=i,I_t=j$ 的转移。备择下

$$
\widehat\pi_0=\frac{n_{01}}{n_{00}+n_{01}},
\qquad
\widehat\pi_1=\frac{n_{11}}{n_{10}+n_{11}},
$$

纯独立性原假设下

$$
\widehat\pi
=\frac{n_{01}+n_{11}}
{n_{00}+n_{01}+n_{10}+n_{11}}.
$$

将共同率似然 $L_{\mathrm{ind},0}$ 与分开率似然 $L_1$ 比较：

$$
LR_{\mathrm{ind}}
=-2\log\!\left(\frac{L_{\mathrm{ind},0}}{L_1}\right)
\overset{a}{\sim}\chi_1^2.
$$

把共同率直接换成 $1-\alpha$ 会同时施加正确覆盖率，变成 [[Christoffersen条件覆盖]] 的联合约束。

<!-- bilingual-en:start -->
Replacing the common fitted rate by $1-\alpha$ also imposes correct coverage and therefore belongs to the [[Christoffersen条件覆盖|conditional-coverage]] null.
<!-- bilingual-en:end -->

> [!question]- 自检
> 纯独立性受限似然中的共同率应取 1%，还是由样本估计？
>
> **答案：** 由转移样本估计；固定为 1% 就不再是纯独立性检验。
>
> <!-- bilingual-en:start -->
> **Self-check:** Should the common hit rate in the restricted likelihood for the pure independence test be fixed at 1%, or estimated from the sample?
>
> **Answer:** It is estimated from the transition sample. Fixing it at 1% changes the hypothesis from pure independence.
> <!-- bilingual-en:end -->

## 边界

- 一阶 Markov 备择只能捕捉相邻期聚集。
- 极少例外或空转移行会使估计落在边界。
- 重叠多日损益可机械制造相邻依赖。

<!-- bilingual-en:start -->
- A first-order Markov alternative detects only adjacent clustering.
- Sparse hits or an empty transition row create boundary estimates.
- Overlapping multi-day losses can mechanically induce dependence.
<!-- bilingual-en:end -->

## 来源与核验

- Christoffersen, [*Evaluating Interval Forecasts*](https://doi.org/10.2307/2527341)：核对转移计数、纯独立性似然与渐近自由度。
- Christoffersen, [公开教学镜像全文](https://users.ssc.wisc.edu/~behansen/718/Christoffersen1998.pdf)：逐式核对共同样本率。

---
aliases:
  - "潜在状态满足一阶 Markov 性不推出边际观测序列也满足一阶 Markov 性"
  - A hidden Markov state does not make observations Markov
  - Marginal observations need not be first-order Markov
  - 潜在 Markov 不推观测 Markov
student_os: knowledge-atom
atom_id: PROB-DTMC-043
atom_set: discrete-time-markov-chains
atom_type: implication-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[Markov switching模型]]"
  - "[[Markov性]]"
related:
  - "[[Markov充分状态]]"
  - "[[状态扩充]]"
leads_to: []
part_of:
  - "[[离散时间马尔可夫链.canvas]]"
---

# 潜在状态满足一阶 Markov 性不推出边际观测序列也满足一阶 Markov 性
<!-- bilingual-en:start -->
*A first-order Markov latent state does not imply that the marginal observation sequence is first-order Markov*
<!-- bilingual-en:end -->

> [!summary] 积分掉潜在状态会保留更早历史的信息
> 在 [[Markov switching模型]] 中，$S_t$ 可以满足
> $$S_{t+1}\perp S_{1:t-1}\mid S_t,$$
> 但观测 $Y_t$ 通常不能满足
> $$Y_{t+1}\perp Y_{1:t-1}\mid Y_t.$$
> 更早的观测会改变对当前潜在状态的后验判断，从而继续影响下一期观测分布。
> <!-- bilingual-en:start -->
> Earlier observations can refine the posterior distribution of the current hidden state, so their predictive effect need not disappear after conditioning on the latest observation.
> <!-- bilingual-en:end -->

一个两状态反例即可看出差别。令 $S_t\in\{0,1\}$ 从平稳分布开始，并满足
$$
\Pr(S_t=S_{t-1})=0.9.
$$
给定 $S_t$，令二元观测以概率 $0.8$ 等于当前状态，以概率 $0.2$ 取另一个值。直接用 Bayes 公式计算得到
$$
\Pr(Y_3=1\mid Y_2=1,Y_1=1)\approx0.701,
$$
而
$$
\Pr(Y_3=1\mid Y_2=1,Y_1=0)\approx0.540.
$$
两式已经固定相同的 $Y_2=1$，却因 $Y_1$ 不同而给出不同预测，所以 $\{Y_t\}$ 不是一阶 Markov 链。

> [!question]- 自检
> 为什么知道最新观测 $Y_t$ 后，更早的 $Y_{t-1}$ 仍可能有预测价值？
>
> **答案：** $Y_t$ 只是潜在状态 $S_t$ 的带噪信号；更早观测可继续更新 $S_t$ 的后验分布，而 $S_t$ 决定下一期状态和观测。

## 来源与核验

- [Rabiner (1989), A Tutorial on Hidden Markov Models](https://doi.org/10.1109/5.18626)：核对观测是潜在 Markov 状态的概率函数这一双重随机过程结构；卡片中的两组条件概率由所列参数直接作 Bayes 计算。
- [Hamilton (1989), A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle](https://doi.org/10.2307/1912559)：核对经济时间序列中潜在 Markov 制度与状态依赖观测方程的区分。

---
aliases:
  - "Markov switching 模型由潜在 Markov 状态与状态依赖观测模型共同定义"
  - Markov switching model
  - Markov regime-switching model
  - 马尔可夫区制转换
student_os: knowledge-atom
atom_id: PROB-DTMC-023
atom_set: discrete-time-markov-chains
atom_type: model-definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[Markov性]]"
related:
  - "[[Markov充分状态]]"
  - "[[状态扩充]]"
  - "[[DTMC时间齐次转移核]]"
  - "[[ARMA 模型：识别、估计、诊断与预测.canvas]]"
leads_to:
  - "[[观测序列不必Markov]]"
part_of:
  - "[[离散时间马尔可夫链.canvas]]"
---

# Markov switching 模型由潜在 Markov 状态与状态依赖观测模型共同定义
<!-- bilingual-en:start -->
*A Markov-switching model combines a latent Markov state with a state-dependent observation model*
<!-- bilingual-en:end -->

> [!summary] 潜在状态决定当前采用哪套观测参数
> Markov switching 模型包含两部分：不可直接观测的状态 $S_t\in\{1,\ldots,K\}$ 按 Markov 转移规律演化；观测变量 $Y_t$ 的条件分布或回归参数随 $S_t$ 改变。例如
> $$
> \Pr(S_t=j\mid S_{t-1}=i,S_{0:t-2})=p_{ij,t},
> $$
> $$
> Y_t=\mu_{S_t}+\phi_{S_t}Y_{t-1}+\varepsilon_t.
> $$
> <!-- bilingual-en:start -->
> A regime-switching model combines a latent-state transition law with an observation distribution whose parameters depend on the current latent state.
> <!-- bilingual-en:end -->

$p_{ij,t}$ 可以不随时间变化，此时潜在链时间齐次；也可以由观测到的协变量驱动而随 $t$ 改变，此时是在给定协变量后满足一阶 Markov 条件。Markov switching 的必要结构是潜在状态按这种条件转移并控制观测模型，不是必须采用同一个固定转移矩阵。

只给 $p_{ij,t}$ 仍不能计算观测数据的似然，因为还缺少每个状态下的观测分布、参数和初始状态分布。经济周期中的“扩张/衰退”只是 $S_t$ 的一种解释；同一结构也可表示高波动/低波动或其他制度。

> [!question]- 自检
> 只给出 $S_t$ 的转移矩阵，能否计算观测数据 $Y_{1:T}$ 的似然？
>
> **答案：** 不能；还需每个状态下的观测分布或回归方程，以及初始状态分布。

## 来源与核验

- [Hamilton (1989), A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle](https://doi.org/10.2307/1912559)：核对 autoregression 参数由离散潜在 Markov 状态驱动及对未观测制度的滤波。
- [statsmodels MarkovRegression documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.regime_switching.markov_regression.MarkovRegression.html)：核对状态依赖回归与 time-varying transition probabilities 的现代实现边界。

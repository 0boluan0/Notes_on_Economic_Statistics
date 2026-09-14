---
aliases:
  - "AR(p) 模型用前 p 期观测与当前创新递归生成本期值"
  - "AR(p) 用过去观测递归传播当前冲击"
  - AR冲击递归传播
  - Autoregressive Model
  - AR model
  - AR process
  - 自回归模型
student_os: knowledge-atom
atom_id: TS-ARMA-003
atom_set: arma-modeling
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[创新]]"
  - "[[白噪声二阶定义]]"
  - "[[滞后算子约定]]"
related:
  - "[[AR因果根条件]]"
  - "[[ARMA无限MA表示]]"
  - "[[线性常系数差分方程]]"
part_of:
  - "[[ARMA 模型：识别、估计、诊断与预测.canvas]]"
  - "[[差分方程与滞后算子.canvas]]"
---

# AR(p) 模型用前 p 期观测与当前创新递归生成本期值
<!-- bilingual-en:start -->
*An AR(p) model recursively generates the current value from the previous p observations and a new innovation*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 中心化 AR($p$) 模型写成
> $$y_t-\mu=\phi_1(y_{t-1}-\mu)+\cdots+\phi_p(y_{t-p}-\mu)+\varepsilon_t,$$
> 或 $\phi(B)(y_t-\mu)=\varepsilon_t$。它用序列自己的前 $p$ 个观测形成本期的线性可预测部分，再加上本期新到达的[[创新]] $\varepsilon_t$。
> <!-- bilingual-en:start -->
> A centered AR($p$) model uses the series' previous $p$ observations to form the linearly predictable part of the current value, then adds the new innovation $\varepsilon_t$.
> <!-- bilingual-en:end -->

若改写为带截距形式
$$y_t=c+\sum_{i=1}^p\phi_i y_{t-i}+\varepsilon_t,$$
在平稳均值存在且 $1-\sum_i\phi_i\ne0$ 时，
$$c=\mu\left(1-\sum_{i=1}^p\phi_i\right).$$
所以截距 $c$ 一般不等于长期均值 $\mu$。
<!-- bilingual-en:start -->
In intercept form, $c=\mu(1-\sum_i\phi_i)$ whenever the stationary mean exists. The intercept is therefore generally not the long-run mean.
<!-- bilingual-en:end -->

“阶数为 $p$”只表示线性递推显式使用最近 $p$ 个观测，不表示一次冲击只持续 $p$ 期。以 AR(1) 为例，
$$y_t-\mu=\phi(y_{t-1}-\mu)+\varepsilon_t.$$
若今天的创新增加 1，其他创新不变，它对 $y_t,y_{t+1},y_{t+2}$ 的作用依次是 $1,\phi,\phi^2$。只要 $\phi\ne0$，影响就能传播超过一期；[[AR因果根条件]]决定这条传播是衰减、持续还是爆炸。
<!-- bilingual-en:start -->
The order $p$ limits the observations that enter the recursion directly; it does not limit a shock's lifetime. In AR(1), a unit innovation affects successive observations by $1,\phi,\phi^2,\ldots$. The causal-root condition determines whether those effects decay, persist, or explode.
<!-- bilingual-en:end -->

若 $\varepsilon_t$ 只满足二阶白噪声条件，这个模型首先刻画基于过去观测的最佳线性预测。要把右侧直接称为完整条件均值，还需要 $\varepsilon_t$ 相对于所用信息集是鞅差、独立，或有 joint Gaussian 等足够条件。
<!-- bilingual-en:start -->
With merely second-order white noise, the equation first describes the best linear prediction from past observations. Interpreting it as the full conditional mean requires an MDS, independence, joint Gaussianity, or another sufficient condition relative to the stated information set.
<!-- bilingual-en:end -->

> [!question]- 自检
> AR(1) 只含一个滞后，是否意味着今天的冲击在两期后完全消失？
>
> **答案：** 不意味着。它会经 $y_t\to y_{t+1}\to y_{t+2}$ 继续传播；稳定时权重按 $\phi^h$ 衰减。

## 来源与核验

- [[01_Math/06_时间序列分析/lecture.pdf#page=73|课程讲义 pp. 73–74, 87–89]]：核对 AR($p$) 方程、截距、创新记号与 AR(1) 递归传播。
- [Hyndman & Athanasopoulos, FPP3 §9.3](https://otexts.com/fpp3/AR.html)：核对 AR($p$) 定义、平稳参数区间与长期均值解释。

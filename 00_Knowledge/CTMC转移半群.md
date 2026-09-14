---
aliases:
  - "齐次 CTMC 的转移矩阵族构成 Chapman-Kolmogorov 半群"
  - CTMC Chapman-Kolmogorov equation
  - Transition semigroup
  - Continuous-time transition function
student_os: knowledge-atom
atom_id: PROB-CTMC-007
atom_set: continuous-time-markov-chains
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[CTMC定义]]"
  - "[[CTMC时间齐次转移函数]]"
related:
  - "[[DTMC转移复合]]"
leads_to:
  - "[[Kolmogorov前后向方程]]"
  - "[[CTMC矩阵指数]]"
  - "[[CTMC平稳分布]]"
part_of:
  - "[[连续时间马尔可夫链.canvas]]"
---

# 齐次 CTMC 的转移矩阵族构成 Chapman-Kolmogorov 半群
<!-- bilingual-en:start -->
*The transition matrices of a homogeneous CTMC form a Chapman-Kolmogorov semigroup*
<!-- bilingual-en:end -->

> [!summary] 连续参数复合
> 令 $P(t)=(p_{ij}(t))$，其中 $p_{ij}(t)=\Pr_i(X_t=j)$。时间齐次 Markov 性给出
> $$
> P(0)=I,\qquad P(s+t)=P(s)P(t),
> $$
> 即
> $$
> p_{ij}(s+t)=\sum_k p_{ik}(s)p_{kj}(t).
> $$
> 这是连续时间 Chapman–Kolmogorov 方程；之所以叫半群，是因为只要求非负时间参数，不要求 $P(t)$ 可逆。
> <!-- bilingual-en:start -->
> Conditioning on the state at an intermediate time composes two transition intervals. The family is a semigroup, not generally a group.
> <!-- bilingual-en:end -->

对保守非爆炸链，每个 $P(t)$ 行随机。若使用爆炸后被杀死的 minimal process 而不把 cemetery state 列入矩阵，某些行和可小于一：缺失质量正是此前已经爆炸的概率。

时间非齐次链需要两参数转移矩阵 $P(r,t)$，满足 $P(r,t)=P(r,s)P(s,t)$；通常不能压成只依赖 $t-r$ 的单参数半群。

> [!example] 中间状态必须求和
> 从 $i$ 在时间 $s+t$ 到达 $j$，时间 $s$ 时可能处于任何状态 $k$。这些事件互斥且穷尽，因此先到 $k$、再由 $k$ 到 $j$ 的概率必须对 $k$ 求和。

> [!question]- 自检
> 为什么 $P(2t)=P(t)^2$ 不表示过程每隔 $t$ 才能跳一次？
>
> **答案：** 矩阵乘法只是按中间时点分解条件概率；每个长度为 $t$ 的区间内部仍可发生任意次跳跃。

## 来源与核验

- [Ward Whitt, Continuous-Time Markov Chains, Lemma 2.1](https://www.columbia.edu/~ww2040/4106S11/CTMCchapter121906.pdf#page=3)：核对连续时间 Chapman–Kolmogorov 求和式与半群性质。
- [Cambridge Applied Probability notes](https://www.statslab.cam.ac.uk/~ps422/notes-new2015.pdf)：核对 minimal transition semigroup 在爆炸情形的质量边界。

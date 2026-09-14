---
aliases:
  - "在截断线性逆需求 P(Q)=max{a-bQ,0}、0<c<a、b>0 与相同恒定边际成本 c 的 Stackelberg 数量双寡头中，一家可观察且已承诺的领导者产量为 (a-c)/(2b)，跟随者产量为 (a-c)/(4b)"
  - In a Stackelberg quantity duopoly under P(Q)=max{a-bQ,0}, 0<c<a, b>0, and identical constant marginal cost c, one observable committed leader produces (a-c)/(2b) and the follower produces (a-c)/(4b)
  - 线性Stackelberg均衡
student_os: knowledge-atom
atom_id: GT-OLI-016
atom_set: oligopoly-competition
atom_type: sequential-equilibrium-result
status: source-checked
mastery_state: unassessed
requires:
  - "[[Stackelberg竞争]]"
  - "[[Cournot反应函数]]"
  - "[[逆向归纳]]"
related:
  - "[[可信威胁]]"
  - "[[战略替代与互补]]"
  - "[[Cournot均衡]]"
part_of:
  - "[[寡头竞争.canvas|寡头竞争]]"
---

# 在截断线性逆需求 P(Q)=max{a-bQ,0}、0<c<a、b>0 与相同恒定边际成本 c 的 Stackelberg 数量双寡头中，一家可观察且已承诺的领导者产量为 (a-c)/(2b)，跟随者产量为 (a-c)/(4b)
<!-- bilingual-en:start -->
*In a Stackelberg quantity duopoly under $P(Q)=\max\{a-bQ,0\}$, $0<c<a$, $b>0$, and identical constant marginal cost $c$, one observable committed leader produces $(a-c)/(2b)$ and the follower produces $(a-c)/(4b)$*
<!-- bilingual-en:end -->

> [!summary] 领导者不是猜对手，而是预见一整条续局反应
> 企业 1 先锁定产量，企业 2 观察后选择自己的利润最大产量。企业 1 因而不能把 $q_2$ 当作固定常数；它要把跟随者对每个可能 $q_1$ 的最佳反应代入自身利润，再选择最优先行量。
>
> <!-- bilingual-en:start -->
> Firm 1 commits to output first and firm 2 observes it before optimizing. The leader cannot treat $q_2$ as fixed; it substitutes the follower's complete best-response function into its own profit and then chooses the initial quantity.
> <!-- bilingual-en:end -->

## 从第二阶段倒推

设逆需求 $P(Q)=\max\{a-bQ,0\}$，$a>c>0$、$b>0$，两家边际成本均为 $c$。正边际成本排除零价格区间里无成本过量生产造成的退化最佳反应。给定领导者产量 $q_L$，跟随者第二阶段的最佳反应是

$$
q_F(q_L)=\max\left\{0,\frac{a-c-bq_L}{2b}\right\}.
$$

在将得到的内点区域，领导者把正值分支代入利润：

$$
\pi_L(q_L)
=\bigl[a-b(q_L+q_F(q_L))-c\bigr]q_L
=\frac{a-c-bq_L}{2}q_L.
$$

一阶和二阶条件给出

$$
q_L^*=\frac{a-c}{2b},
\qquad
\frac{d^2\pi_L}{dq_L^2}=-b<0.
$$

这个内点候选还要和另外两个全局区域比较。若

$$
\frac{a-c}{b}\le q_L<\frac ab,
$$

跟随者退出，领导者利润变为 $(a-c-bq_L)q_L\le0$；若 $q_L\ge a/b$，市场价格为 0，而 $c>0$ 使领导者利润为 $-cq_L<0$。内点候选利润 $(a-c)^2/(8b)>0$，因此两个边界区域都不能优于它，局部二阶条件得到的候选也是全局最优先行量。

跟随者随后选择

$$
q_F^*=q_F(q_L^*)=\frac{a-c}{4b}.
$$

所以

$$
Q^*=\frac{3(a-c)}{4b},
\qquad
P^*=\frac{a+3c}{4},
$$

$$
\pi_L^*=\frac{(a-c)^2}{8b},
\qquad
\pi_F^*=\frac{(a-c)^2}{16b}.
$$

<!-- bilingual-en:start -->
Backward induction first derives the follower's response $q_F(q_L)=\max\{0,(a-c-bq_L)/(2b)\}$. The interior branch gives $q_L^*=(a-c)/(2b)$. If the leader instead makes the follower exit while price remains positive, its profit is nonpositive; in the zero-price region, positive marginal cost makes profit strictly negative. Thus the interior candidate is globally optimal, and $q_F^*=(a-c)/(4b)$. Total output is $3(a-c)/(4b)$, price is $(a+3c)/4$, and profits are $(a-c)^2/(8b)$ for the leader and $(a-c)^2/(16b)$ for the follower.
<!-- bilingual-en:end -->

## 先行优势来自承诺与战略替代

在 [[Cournot均衡|同时行动基准]] 中，每家利润是 $(a-c)^2/(9b)$。本模型里领导者利润更高、跟随者利润更低，总产量更高且价格更低。机制是产量为 [[战略替代与互补|战略替代]]：领导者先承诺较大产量，跟随者的最优产量随之收缩。

这个比较要求 $q_L$ 在跟随者行动前可观察且难以撤回。若跟随者看不到先行量，它不能按 $q_L$ 调整；若领导者可以在跟随者行动后无成本撤回，原先数量未形成真正承诺。此时不能只保留这组公式，必须重写 [[博弈模型]]。先手优势也不是所有序贯博弈的定理。

<!-- bilingual-en:start -->
Relative to simultaneous Cournot, the leader earns more, the follower less, total output is higher, and price lower. The mechanism is commitment with strategic substitutes. Observability and irreversibility are essential; an unseen or freely reversible early move does not generate the same continuation response. Moving first is not inherently advantageous in every sequential game.
<!-- bilingual-en:end -->

> [!question]- 自检
> 为什么这里的领导者不能像 Cournot 企业一样，把跟随者产量当作一个固定数再求自己的最佳反应？
>
> **答案：** 因为跟随者会观察每个可能的先行产量并据此优化。领导者选择 $q_L$ 时已知道 $q_F$ 会随它变化，必须优化复合函数 $\pi_L(q_L,q_F(q_L))$。

## 来源与核验

- MIT 15.010/15.011, [*The Basics of Game Theory*, pp. 2 and 9–10](https://ocw.mit.edu/courses/15-010-economic-analysis-for-business-decisions-fall-2004/807ba86e100d349ef73c294b9e720931_the_bsc_game_thy.pdf)：核对可观察的序贯数量选择、跟随者反应、领导者倒推与线性数值例题。
- MIT 14.01SC, [*Exam 2 Review*, p. 5](https://ocw.mit.edu/courses/14-01sc-principles-of-microeconomics-fall-2011/66a11c228027ef6cd3dbc53a30e64921_MIT14_01SCF11_rev_midtm02.pdf)：交叉核对领导者必须把跟随者对其产量的反应纳入选择。
- [[Stackelberg竞争]]拥有序贯数量领导的定义；本卡只拥有标准线性同成本双寡头的具体均衡、利润比较与承诺边界。本地课程正文未单独教授这组公式，因此未把该推导错误归因给课程页。

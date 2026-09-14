---
aliases:
  - "Stackelberg 竞争是领导者先作出可观察且可信的行动承诺，跟随者观察后选择最佳反应的序贯博弈模型"
  - Stackelberg competition is a sequential game in which a leader makes an observable and credible action commitment before a follower chooses a best response
  - 斯塔克尔伯格竞争
student_os: knowledge-atom
atom_id: GT-OLI-011
atom_set: oligopoly-competition
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[扩展式博弈]]"
  - "[[逆向归纳]]"
related:
  - "[[Cournot竞争]]"
  - "[[可信威胁]]"
  - "[[寡头模型选择]]"
leads_to:
  - "[[Stackelberg线性均衡]]"
part_of:
  - "[[寡头竞争.canvas|寡头竞争]]"
---

# Stackelberg 竞争是领导者先作出可观察且可信的行动承诺，跟随者观察后选择最佳反应的序贯博弈模型
<!-- bilingual-en:start -->
*Stackelberg competition is a sequential game in which a leader makes an observable and credible action commitment before a follower chooses a best response*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> Stackelberg 竞争不是“先猜一个对手行动”的同时行动模型。领导者先选择一个能被跟随者观察、且在跟随者行动前不能随意撤回的行动；跟随者看见该行动后优化。领导者因此要预见整条跟随者反应，再决定自己的先行动作。
>
> <!-- bilingual-en:start -->
> Stackelberg competition is not simultaneous play with a guessed rival action. The leader first chooses an observable action that cannot be freely withdrawn before the follower moves. The follower then optimizes after observing it, so the leader must anticipate the follower's entire response when choosing the initial action.
> <!-- bilingual-en:end -->

## 模型怎样写

以数量领导为例，领导者先选择 $q_L$。跟随者观察后，在每个可能的 $q_L$ 下选择

$$
q_F(q_L)\in BR_F(q_L)
=\arg\max_{q_F}\pi_F(q_L,q_F).
$$

领导者知道自己的选择会改变后续反应，所以求解

$$
\max_{q_L}\;\pi_L\bigl(q_L,q_F(q_L)\bigr).
$$

这就是 [[逆向归纳]] 的两步：先求跟随者在每个续局的最优行动，再把完整反应代回领导者问题。解是序贯策略，而不只是均衡路径上的两个数。

<!-- bilingual-en:start -->
In quantity leadership, the follower first supplies a best response $q_F(q_L)$ for every possible leader quantity. The leader then maximizes its own payoff after substituting that complete response. Backward induction therefore solves the follower's continuation problem before the leader's initial problem, and the solution specifies continuation play rather than only two on-path quantities.
<!-- bilingual-en:end -->

## 名称不自动带来先手优势

模型成立至少需要三件事：行动有先后顺序；跟随者能在自己行动前观察领导者的选择；领导者的先行量构成真实承诺。若先行行动不可见，跟随者无法据此调整；若领导者可以无成本撤回，原行动也没有改变续局。此时应重写 [[博弈模型]]，不能机械套用 Stackelberg 结果。

先行动是否更有利还取决于收益和最佳反应的形状。在线性同成本数量双寡头中可以推出具体产量与先行者利润优势，见 [[Stackelberg线性均衡]]；这不是所有序贯博弈的定义性结论。

<!-- bilingual-en:start -->
The model requires sequential moves, observation before the follower acts, and a genuine commitment. An unseen or freely reversible early action does not generate the same continuation response. Moving first is not advantageous by definition; the familiar linear-duopoly quantities and profit ranking belong to the separate [[Stackelberg线性均衡|linear Stackelberg equilibrium]] result.
<!-- bilingual-en:end -->

> [!question]- 最小例子
> 一家企业先公开并锁定产量，另一家随后观察该产量再生产。后手产量随先手量变化时，先手企业应该优化什么？
>
> **答案：** 它要优化复合收益 $\pi_L(q_L,q_F(q_L))$，而不是把后手产量当作固定常数。

## 来源与核验

- MIT 15.010/15.011, [*The Basics of Game Theory*, pp. 2 and 9–10](https://ocw.mit.edu/courses/15-010-economic-analysis-for-business-decisions-fall-2004/807ba86e100d349ef73c294b9e720931_the_bsc_game_thy.pdf)：核对可观察的序贯数量选择、领导者、跟随者与倒推结构。
- MIT 14.01SC, [*Exam 2 Review*, p. 5](https://ocw.mit.edu/courses/14-01sc-principles-of-microeconomics-fall-2011/66a11c228027ef6cd3dbc53a30e64921_MIT14_01SCF11_rev_midtm02.pdf)：交叉核对领导者必须把跟随者对先行产量的反应纳入选择。
- [[Stackelberg线性均衡]]拥有标准截断线性需求下的具体均衡、利润比较与边界核验；本卡只拥有 Stackelberg 序贯竞争的模型定义与通用求解结构。

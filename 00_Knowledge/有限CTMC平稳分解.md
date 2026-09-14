---
aliases:
  - "有限 CTMC 的全部平稳分布是各闭沟通类平稳分布的凸组合"
  - "有限 CTMC 的平稳分布由闭沟通类承载且唯一性由闭类数决定"
  - Finite CTMC stationary decomposition
  - Closed-class stationary distributions
student_os: knowledge-atom
atom_id: PROB-CTMC-015
atom_set: continuous-time-markov-chains
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[跳链决定CTMC沟通类]]"
  - "[[CTMC平稳分布]]"
  - "[[有限不可约CTMC稳态]]"
related:
  - "[[有限链平稳分解]]"
leads_to:
  - "[[有限CTMC全局收敛]]"
part_of:
  - "[[连续时间马尔可夫链.canvas]]"
---

# 有限 CTMC 的全部平稳分布是各闭沟通类平稳分布的凸组合
<!-- bilingual-en:start -->
*Every stationary distribution of a finite CTMC is a convex combination of the stationary distributions on its closed communicating classes*
<!-- bilingual-en:end -->

> [!summary] 用闭类描述全链的全部平稳分布
> 把有限 CTMC 分解为暂态状态和闭沟通类 $C_1,\ldots,C_m$。由[[有限不可约CTMC稳态]]，每个 $C_r$ 都有一个只支撑在该类上的唯一平稳分布 $\pi^{(r)}$。全链的全部平稳分布恰是
> $$
> \pi=\sum_{r=1}^m a_r\pi^{(r)},
> \qquad a_r\ge0,
> \qquad \sum_{r=1}^m a_r=1.
> $$
> 暂态状态在任何平稳分布下质量都为零。因此，全链的平稳分布唯一当且仅当只有一个闭沟通类。
> <!-- bilingual-en:start -->
> Closed classes carry all stationary mass. The full set of stationary distributions is their convex hull, and uniqueness is equivalent to having exactly one closed class.
> <!-- bilingual-en:end -->

全链分解只回答各闭类的平稳分布怎样拼成全链的平稳分布；每个闭类内部为何存在且只有一个平稳分布，由[[有限不可约CTMC稳态]]给出。

> [!example] 两个吸收态造成不唯一
> 状态 0 可跳到吸收态 1 或 2。闭类是 $\{1\}$ 与 $\{2\}$，所以 $\delta_1$、$\delta_2$ 以及任意
> $$a\delta_1+(1-a)\delta_2,\qquad 0\le a\le1,$$
> 都是平稳分布；暂态状态 0 的平稳质量始终为零。

> [!question]- 自检
> 一条有限 CTMC 有三个暂态状态和一个闭沟通类。全链的平稳分布是否唯一？
>
> **答案：** 唯一。暂态状态不承载平稳质量，而唯一闭类只提供一个类内平稳分布。

## 来源与核验

- [MIT OCW 6.436J, Lecture 24](https://ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/087af3cedbc9def5b156c5e1665ac79c_MIT6_436JF18_lec24.pdf#page=7)：核对有限 CTMC 的闭类分解、平稳质量支撑与唯一 recurrent class 判据。
- [[有限链平稳分解]]：交叉核对有限 Markov 链按闭沟通类分解平稳分布的结构。

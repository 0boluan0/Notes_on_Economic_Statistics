---
aliases:
  - "离散时间 Markov 链是在离散时点、离散状态空间上满足 Markov 性的随机过程"
  - Discrete-time Markov chain
  - DTMC
  - 离散时间马尔可夫链
student_os: knowledge-atom
atom_id: PROB-DTMC-024
atom_set: discrete-time-markov-chains
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[随机过程]]"
  - "[[Markov性]]"
related:
  - "[[Markov矩阵]]"
leads_to:
  - "[[Markov充分状态]]"
  - "[[DTMC时间齐次转移核]]"
part_of:
  - "[[离散时间马尔可夫链.canvas]]"
---

# 离散时间 Markov 链是在离散时点、离散状态空间上满足 Markov 性的随机过程
<!-- bilingual-en:start -->
*A discrete-time Markov chain is a stochastic process on discrete time and a discrete state space that satisfies the Markov property*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 离散时间 Markov 链是一列随机变量
> $$
> \{X_n:n=0,1,2,\ldots\},
> $$
> 其状态空间 $S$ 有限或可数，并且给定当前状态 $X_n$ 后，下一步的条件分布不再依赖更早的状态。对有正概率的离散历史，
> $$
> \Pr(X_{n+1}=j\mid X_0=i_0,\ldots,X_n=i)
> =\Pr(X_{n+1}=j\mid X_n=i).
> $$
> <!-- bilingual-en:start -->
> A DTMC is a finite- or countable-state stochastic process indexed by discrete time whose next-state law, conditional on the present state, does not depend on the earlier path.
> <!-- bilingual-en:end -->

这里的“链”同时约束时间和状态：时间按 $0,1,2,\ldots$ 推进，状态取自离散集合。只有时间离散而状态连续的 Markov 模型，通常称离散时间 Markov 过程，不属于上述 DTMC 定义。

时间齐次不是上述定义的一部分。一般 DTMC 的一步转移可以随 $n$ 改变；只有再加入[[DTMC时间齐次转移核|时间齐次性]]，才能用同一个转移矩阵反复描述每一步。初始分布与这些一步转移核共同确定链的有限维分布。

> [!example] 天气状态链
> 令 $S=\{\text{晴},\text{雨}\}$，每天记录一次天气。若明天的天气分布在已知今天后不再因前天而改变，这列天气状态就是一条 DTMC；至于冬季与夏季是否使用同一套概率，是另一个时间齐次问题。

> [!question]- 自检
> 一个过程每小时观测一次，但每个时点取任意实数。它仅凭“时间离散”就符合上述 DTMC 定义吗？
>
> **答案：** 不是。它还需要离散状态空间并满足 Markov 性；连续状态模型不属于这里的 Markov 链口径。

## 来源与核验

- [MIT OCW 6.262, Chapter 3, Definition 3.1.1](https://ocw.mit.edu/courses/6-262-discrete-stochastic-processes-spring-2011/3558b08622765d26c2b0a7d2eeeac885_MIT6_262S11_chap03.pdf#page=2)：核对离散时间、离散状态与 Markov 条件概率。
- [Cambridge Markov Chains notes, §1](https://www.statslab.cam.ac.uk/~rrw1/markov/M.pdf#page=2)：核对初始分布、转移概率与链的有限维分布。

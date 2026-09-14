---
aliases:
  - "不可约、非爆炸且正常返的可数 CTMC 收敛到唯一平稳分布"
  - Countable CTMC convergence to equilibrium
  - Positive recurrent CTMC convergence
student_os: knowledge-atom
atom_id: PROB-CTMC-033
atom_set: continuous-time-markov-chains
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[可数CTMC稳态存在]]"
related:
  - "[[有限CTMC全局收敛]]"
  - "[[有限链逐步收敛]]"
  - "[[有限CTMC平稳生成矩阵判据]]"
part_of:
  - "[[连续时间马尔可夫链.canvas]]"
---

# 不可约、非爆炸且正常返的可数 CTMC 收敛到唯一平稳分布
<!-- bilingual-en:start -->
*An irreducible non-explosive positive recurrent countable CTMC converges to its unique stationary distribution*
<!-- bilingual-en:end -->

> [!summary] 可数连续时间链的平衡收敛
> 设可数状态 CTMC 不可约、非爆炸且正常返，并令 $\pi$ 为[[可数CTMC稳态存在|其唯一平稳分布]]。在所给生成矩阵对应保守、regular 转移半群的前提下，
> $$
> p_{ij}(t)\longrightarrow\pi_j
> \qquad(t\to\infty)
> $$
> 对所有状态 $i,j$ 成立。连续时间中的随机停留消除了 embedded jump chain 的固定步数周期阻塞，因此不需另加离散链式的非周期条件。
> <!-- bilingual-en:start -->
> Irreducibility, non-explosion, and positive recurrence yield convergence to equilibrium in calendar time; no separate aperiodicity assumption is needed.
> <!-- bilingual-en:end -->

这个结论的方向不能倒置成“只要形式上解出 $\pi Q=0$ 就收敛”。[[有限CTMC平稳生成矩阵判据|有限状态判据]]只判断平稳性，也不直接给出收敛；在这里还要先由[[可数CTMC稳态存在]]确认所讨论过程确有唯一平稳概率，再由 regular CTMC 的平衡收敛定理得到极限。

embedded jump chain 即使有周期，CTMC 仍可能收敛。周期描述每次真跳后的相位；$P(t)$ 还混合了截至时刻 $t$ 已发生的不同跳数，所以不会保留固定奇偶振荡。

> [!question]- 自检
> 一条不可约可数 CTMC 已找到归一化的 $\pi Q=0$ 解，能否立即断言 $p_{ij}(t)\to\pi_j$？
>
> **答案：** 不能。还必须确认过程非爆炸、regular 且正常返；形式零空间解不能替代这些过程条件。

## 来源与核验

- [Cambridge Applied Probability notes, Theorem 2.22](https://www.statslab.cam.ac.uk/~ps422/notes-new2015.pdf)：核对不可约、非爆炸、正常返 CTMC 的 convergence to equilibrium。
- [James Norris, Markov Chains, §§3.5–3.6](https://www.statslab.cam.ac.uk/~jrn10/Markov/)：交叉核对可数 CTMC 的平稳存在与连续时间收敛。

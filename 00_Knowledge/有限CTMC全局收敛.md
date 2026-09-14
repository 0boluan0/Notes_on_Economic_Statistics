---
aliases:
  - "有限 CTMC 有唯一闭类便从任意初态收敛而无需非周期条件"
  - Finite CTMC convergence to equilibrium
  - CTMC has no periodicity obstruction
  - 连续时间链收敛
student_os: knowledge-atom
atom_id: PROB-CTMC-016
atom_set: continuous-time-markov-chains
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[有限不可约CTMC稳态]]"
  - "[[有限CTMC平稳分解]]"
related:
  - "[[有限链逐步收敛]]"
  - "[[周期链Cesaro平均]]"
  - "[[稳态唯一不推收敛]]"
part_of:
  - "[[连续时间马尔可夫链.canvas]]"
---

# 有限 CTMC 有唯一闭类便从任意初态收敛而无需非周期条件
<!-- bilingual-en:start -->
*A finite CTMC with one closed class converges from every initial state without an aperiodicity condition*
<!-- bilingual-en:end -->

> [!summary] 连续等待打破固定步长振荡
> 若有限 CTMC 只有一个闭沟通类，令 $\pi$ 为其唯一平稳分布，则
> $$
> p_{ij}(t)\longrightarrow \pi_j\qquad(t\to\infty)
> $$
> 对所有初始状态 $i$ 成立。与离散链不同，不需要另加 aperiodicity；任意正时间内“尚未跳跃”的正概率使固定日历采样失去 embedded chain 的周期阻塞。
> <!-- bilingual-en:start -->
> Continuous holding times remove the fixed-step periodic obstruction. With one closed class, a finite CTMC converges to its unique stationary law from every state.
> <!-- bilingual-en:end -->

embedded jump chain 可以有周期。最简单的两状态真跳链严格交替，周期为 2；但 CTMC 在任一观察时刻可能仍停在当前状态，因此其日历时间分布平滑收敛。

> [!example] 周期为二的跳链仍收敛
> 对 $0\rightleftarrows1$，速率分别为 $\lambda,\mu>0$，embedded chain 每一步必换状态。CTMC 却满足
> $$
> p_{00}(t)=\frac\mu{\lambda+\mu}+\frac\lambda{\lambda+\mu}e^{-(\lambda+\mu)t},
> $$
> 因而收敛到 $\pi_0=\mu/(\lambda+\mu)$，没有奇偶振荡。

> [!question]- 自检
> embedded jump chain 的周期是 3，能否据此断言 CTMC 的 $P(t)$ 不收敛？
>
> **答案：** 不能。周期是固定步数现象；有限 CTMC 在唯一闭类条件下仍随连续时间收敛。

## 来源与核验

- [MIT OCW 6.436J, Lecture 24](https://ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/087af3cedbc9def5b156c5e1665ac79c_MIT6_436JF18_lec24.pdf#page=7)：核对有限 CTMC 在唯一 recurrent class 下的 mixing 结论及无周期障碍。
- [Cambridge Applied Probability notes, Theorem 2.22](https://www.statslab.cam.ac.uk/~ps422/notes-new2015.pdf)：核对 irreducible non-explosive CTMC 的 convergence-to-equilibrium 论证。

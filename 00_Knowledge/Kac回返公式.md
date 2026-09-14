---
aliases:
  - "可数不可约正常返链的平稳质量等于平均回返时间的倒数"
  - Kac return-time formula
  - Mean recurrence time formula
  - Kac 回返公式
student_os: knowledge-atom
atom_id: PROB-DTMC-015
atom_set: discrete-time-markov-chains
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[正常返与零常返]]"
  - "[[不可约链]]"
  - "[[Markov稳态分布]]"
related:
  - "[[Markov时间平均收敛]]"
part_of:
  - "[[离散时间马尔可夫链.canvas]]"
---

# 可数不可约正常返链的平稳质量等于平均回返时间的倒数
<!-- bilingual-en:start -->
*For a countable irreducible positive recurrent chain, stationary mass is the reciprocal of mean return time*
<!-- bilingual-en:end -->

> [!summary] Kac 回返关系
> 设 DTMC 的状态空间可数，链不可约且正常返，并令 $\tau_i^+=\inf\{n\ge1:X_n=i\}$。对其唯一平稳分布 $\pi$，每个状态 $i$ 都满足
> $$
> \pi_i=\frac{1}{E_i\tau_i^+}.
> $$
> 直觉上，每次回到 $i$ 开启一个更新周期；一个周期平均长 $E_i\tau_i^+$，每周期在起点计一次 $i$，所以长期访问率为其倒数。
> <!-- bilingual-en:start -->
> Kac's return relation identifies stationary mass with the reciprocal of the expected first return time in an irreducible positive recurrent chain.
> <!-- bilingual-en:end -->

式子不要求非周期，因为它描述长期访问率而非逐步分布极限。也不能把它越过“正常返”条件直接延伸成零常返链的平稳分布：零常返时平均回返时间无限，形式上的倒数全为 0，无法归一化成概率分布。

> [!question]- 自检
> 若 $\pi_i=0.2$，从 $i$ 出发的平均首次回返时间是多少？
>
> **答案：** $1/0.2=5$ 步；前提是链不可约且正常返。

## 来源与核验

- [Cambridge Markov Chains notes, Theorem 9.1](https://www.statslab.cam.ac.uk/~rrw1/markov/M.pdf#page=37)：核对可数不可约链正常返、平稳分布与 $m_i=1/\pi_i$ 的关系。
- [MIT OCW 6.262, Chapter 5, Theorem 5.1.4](https://ocw.mit.edu/courses/6-262-discrete-stochastic-processes-spring-2011/01d0892549619cb25d928f15ec7230ed_MIT6_262S11_chap05.pdf#page=10)：核对时间平均访问率与平均回返时间。

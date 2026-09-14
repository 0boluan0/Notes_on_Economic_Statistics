---
aliases:
  - "爆炸是跳时序列在有限时间内聚积"
  - CTMC explosion
  - Explosion time
  - 爆炸时间
student_os: knowledge-atom
atom_id: PROB-CTMC-010
atom_set: continuous-time-markov-chains
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[CTMC指数停留时间]]"
  - "[[出口率]]"
  - "[[嵌入跳链]]"
related:
  - "[[有界率非爆炸]]"
  - "[[纯生链爆炸判据]]"
leads_to:
  - "[[最小CTMC]]"
  - "[[爆炸后延拓]]"
part_of:
  - "[[连续时间马尔可夫链.canvas]]"
---

# 爆炸是跳时序列在有限时间内聚积
<!-- bilingual-en:start -->
*Explosion occurs when infinitely many jump times accumulate in finite time*
<!-- bilingual-en:end -->

> [!summary] 可数状态的新边界
> 令 $T_n$ 为第 $n$ 次真实跳跃的时刻；若这次跳跃不存在，则约定 $T_n=\infty$。再定义
> $$
> \zeta:=\lim_{n\to\infty}T_n.
> $$
> 若 $\Pr_i(\zeta<\infty)>0$，链从 $i$ 出发可能在有限日历时间内完成无限次跳跃，称为**爆炸**。每个状态的出口率都有限，并不能排除沿路径进入越来越快的状态。
> <!-- bilingual-en:start -->
> Explosion is accumulation of infinitely many genuine jumps before a finite time. Finite rate at each individual state does not imply a uniform rate bound along the path.
> <!-- bilingual-en:end -->

> [!example] 速率越来越快
> 纯生链从 $n$ 只跳到 $n+1$，若速率为 $2^n$，则第 $n$ 个状态的停留时间 $H_n\sim\operatorname{Exp}(2^n)$，且
> $$
> \mathbb E\!\left[\sum_{n\ge n_0}H_n\right]
> =\sum_{n\ge n_0}2^{-n}<\infty.
> $$
> 这个非负总和因而几乎处处有限：链会在有限时间内依次越过无限多个状态。

> [!question]- 自检
> “每个 $q_i$ 都小于无穷”是否足以保证不爆炸？
>
> **答案：** 不足。还需控制沿路径遇到的速率；例如 $q_n=2^n$ 每项有限却可爆炸。

## 来源与核验

- [James Norris, Markov Chains, §§2.7–2.9](https://www.statslab.cam.ac.uk/~jrn10/Markov/)：核对 explosion time 与逐跳构造。
- [Cambridge Applied Probability notes, §§1.7–1.9](https://www.statslab.cam.ac.uk/~ps422/notes-new2015.pdf)：核对点态有限出口率不足以排除爆炸，以及纯生链的爆炸例子。

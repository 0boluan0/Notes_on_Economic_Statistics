---
aliases:
  - "CTMC 的沟通类由嵌入跳链的正率路径决定"
  - Positive-rate paths determine CTMC communicating classes
student_os: knowledge-atom
atom_id: PROB-CTMC-012
atom_set: continuous-time-markov-chains
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[嵌入跳链]]"
  - "[[马氏链沟通类]]"
  - "[[闭沟通类]]"
  - "[[不可约链]]"
related:
  - "[[CTMC爆炸]]"
leads_to:
  - "[[常返与停留时间]]"
  - "[[有限不可约CTMC稳态]]"
part_of:
  - "[[连续时间马尔可夫链.canvas]]"
---

# CTMC 的沟通类由嵌入跳链的正率路径决定
<!-- bilingual-en:start -->
*The communicating classes of a CTMC are determined by positive-rate paths in its embedded jump chain*
<!-- bilingual-en:end -->

> [!summary] 时间改变而有向图不变
> 对生成矩阵 $Q$ 的 minimal CTMC，状态 $j$ 从 $i$ 可达，当且仅当存在有限路径
> $$
> i=x_0,x_1,\ldots,x_m=j
> $$
> 使每条边都有 $q_{x_rx_{r+1}}>0$。这些正速率边恰好是[[嵌入跳链]]的正概率边，因此两者有相同的可达关系、沟通类、闭沟通类与不可约性。
> <!-- bilingual-en:start -->
> Exit rates change calendar time, but the directed graph of possible pre-explosion jumps is the graph of the embedded chain.
> <!-- bilingual-en:end -->

一个集合闭合，意味着类内任一状态都没有正速率边通向类外。若 $q_i=0$，状态 $i$ 自身构成闭吸收类。改变正速率的大小不会改变“能否沿有限路径到达”，但会改变给定时间内到达的概率和速度。

爆炸可能使 minimal process 无法在原状态空间中运行到任意大的日历时刻，却不改变爆炸前由 $Q$ 的正速率边定义的 class graph。若另行加入[[爆炸后延拓]]，延拓产生的爆炸后重入行为不属于这个由 $Q$ 单独决定的图结构。

> [!example] 同图不同速度
> 把所有正速率同时乘 100，沟通类完全不变；所有有限正平均停留时间则缩短为原来的 $1/100$。

> [!question]- 自检
> 若 $q_{12}>0$、$q_{23}>0$，但 $q_{13}=0$，状态 3 是否仍从状态 1 可达？
>
> **答案：** 可达，因为存在正速率路径 $1\to2\to3$；可达不要求一次直接跳转。

## 来源与核验

- [Cambridge Applied Probability notes, Theorem 2.1](https://www.statslab.cam.ac.uk/~ps422/notes-new2015.pdf)：核对 CTMC、embedded jump chain、正速率路径与 $p_{ij}(t)>0$ 的等价可达条件。
- [[马氏链沟通类]]、[[闭沟通类]]与[[不可约链]]：CTMC 的正速率图沿用这些可达、闭合与全空间单沟通类的定义；出口率大小只改变日历时间，不改变类结构。

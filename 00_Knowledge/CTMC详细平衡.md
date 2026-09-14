---
aliases:
  - "相对于非负权重 π，CTMC 的详细平衡要求每对状态的双向速率流相等"
  - CTMC detailed balance
  - Rate detailed balance
  - 速率详细平衡
student_os: knowledge-atom
atom_id: PROB-CTMC-019
atom_set: continuous-time-markov-chains
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[CTMC平稳分布]]"
related:
  - "[[DTMC详细平衡]]"
leads_to:
  - "[[CTMC详细平衡与可逆]]"
part_of:
  - "[[连续时间马尔可夫链.canvas]]"
---

# 相对于非负权重 π，CTMC 的详细平衡要求每对状态的双向速率流相等
<!-- bilingual-en:start -->
*Relative to nonnegative weights π, detailed balance for a CTMC requires equal rate flow in both directions for every pair of states*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 给定 CTMC 的生成矩阵 $Q=(q_{ij})$ 和一组不全为零的非负权重 $\pi=(\pi_i)$。若对所有不同状态 $i,j$ 都有
> $$
> \pi_iq_{ij}=\pi_jq_{ji},
> $$
> 就称 $Q$ **相对于 $\pi$ 满足详细平衡**。等式两边分别是按 $\pi$ 加权后从 $i$ 流向 $j$ 与从 $j$ 流向 $i$ 的瞬时概率流率。
> <!-- bilingual-en:start -->
> Detailed balance relative to π means that every pair of states carries equal stationary-weighted transition-rate flow in both directions.
> <!-- bilingual-en:end -->

这里的权重不必预先归一化。若 $\sum_i\pi_i<\infty$，才可除以总质量得到概率分布；详细平衡方程在整体缩放下不变。这个区分使[[生灭链平稳递推]]可以先求相对权重，再另行检查能否归一化。涉及平稳分布或可逆性时，仍须使用归一化后的概率分布，并满足相应的过程条件。

详细平衡不是 $q_{ij}=q_{ji}$。例如两状态生成矩阵
$$
Q=\begin{pmatrix}-\lambda&\lambda\\ \mu&-\mu\end{pmatrix},
\qquad \lambda,\mu>0,
$$
相对于 $\pi=(\mu/(\lambda+\mu),\lambda/(\lambda+\mu))$ 满足 $\pi_0\lambda=\pi_1\mu$，即使 $\lambda\ne\mu$。

逐对速率流条件本身不等于平稳性的一般定义。它与可逆性的关系、归一化后为何推出平稳，以及平稳为何不必满足该条件，见 [[CTMC详细平衡与可逆]]。

> [!question]- 自检
> 只检查每个状态的总流入等于总流出，是否已经验证 CTMC 详细平衡？
>
> **答案：** 没有。详细平衡要求每一对状态的双向流逐对相等，而不仅是总量守恒。

## 来源与核验

- [Cambridge Applied Probability notes, §2.6](https://www.statslab.cam.ac.uk/~ps422/notes-new2015.pdf)：核对 rate detailed balance 的逐对速率流定义。
- [Ward Whitt, Continuous-Time Markov Chains, §§6, 10](https://www.columbia.edu/~ww2040/Whitt_CTMCnotes121312.pdf)：交叉核对生成率详细平衡方程。
- [[DTMC详细平衡]]：对照离散时间中逐对概率流的定义，同时保留速率与一步概率的区别。

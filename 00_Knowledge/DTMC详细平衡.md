---
aliases:
  - "相对于分布 π，DTMC 的详细平衡要求每对状态的双向概率流相等"
  - Detailed balance
  - DTMC detailed balance
  - 详细平衡
student_os: knowledge-atom
atom_id: PROB-DTMC-020
atom_set: discrete-time-markov-chains
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[Markov稳态分布]]"
related:
  - "[[DTMC可逆性]]"
leads_to:
  - "[[详细平衡与可逆]]"
part_of:
  - "[[离散时间马尔可夫链.canvas]]"
---

# 相对于分布 π，DTMC 的详细平衡要求每对状态的双向概率流相等
<!-- bilingual-en:start -->
*Relative to a distribution π, detailed balance for a DTMC requires equal probability flow in both directions for every pair of states*
<!-- bilingual-en:end -->

> [!summary] 定义
> 给定 DTMC 的转移矩阵 $P=(p_{ij})$ 和状态空间上的概率分布 $\pi$。若对所有状态对 $i,j$ 都有
> $$
> \pi_i p_{ij}=\pi_jp_{ji},
> $$
> 就称 $P$ **相对于 $\pi$ 满足详细平衡**。左、右两边分别是按 $\pi$ 加权后从 $i$ 流向 $j$ 和从 $j$ 流向 $i$ 的一步概率流。
> <!-- bilingual-en:start -->
> Detailed balance relative to π means that the one-step probability flow from i to j equals the reverse flow from j to i for every state pair.
> <!-- bilingual-en:end -->

“概率流相等”不等于 $p_{ij}=p_{ji}$。例如二状态链
$$
P=\begin{pmatrix}1-a&a\\ b&1-b\end{pmatrix},\qquad a,b>0,
$$
相对于 $\pi=(b/(a+b),\,a/(a+b))$ 满足 $\pi_1a=\pi_2b$，即使 $a\ne b$。

这里的等式只规定逐对概率流平衡。它对平稳性和时间反演的后果见 [[详细平衡与可逆]]。

> [!question]- 自检
> 只检查每个状态的总流入等于总流出，是否已经验证详细平衡？
>
> **答案：** 没有。详细平衡要求每一对状态都逐对抵消，而不只是总量守恒。

## 来源与核验

- [Cambridge Markov Chains notes, §11.2](https://www.statslab.cam.ac.uk/~rrw1/markov/M.pdf#page=46)：核对详细平衡的逐对概率流定义。
- [Levin and Peres, Markov Chains and Mixing Times, Proposition 1.20](https://pages.uoregon.edu/dlevin/MARKOV/mcmt2e.pdf#page=26)：交叉核对 detailed balance 方程。

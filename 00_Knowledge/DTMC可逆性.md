---
aliases:
  - "相对于分布 π，DTMC 可逆是指平稳路径在时间反演后保持同一联合分布"
  - Reversible Markov chain
  - DTMC reversibility
  - 时间可逆 Markov 链
student_os: knowledge-atom
atom_id: PROB-DTMC-031
atom_set: discrete-time-markov-chains
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[离散时间Markov链]]"
  - "[[Markov稳态分布]]"
related:
  - "[[DTMC详细平衡]]"
leads_to:
  - "[[详细平衡与可逆]]"
part_of:
  - "[[离散时间马尔可夫链.canvas]]"
---

# 相对于分布 π，DTMC 可逆是指平稳路径在时间反演后保持同一联合分布
<!-- bilingual-en:start -->
*Relative to π, a DTMC is reversible when its stationary path has the same joint distribution after time reversal*
<!-- bilingual-en:end -->

> [!summary] 定义
> 设 $\pi$ 是 DTMC 的平稳分布，并从 $X_0\sim\pi$ 启动。若对每个 $m\ge1$，
> $$
> (X_0,X_1,\ldots,X_m)
> \overset d=
> (X_m,X_{m-1},\ldots,X_0),
> $$
> 就称该链**相对于 $\pi$ 可逆**。也就是说，在平稳状态下观察任意有限路径，正放和倒放具有相同的联合分布。
> <!-- bilingual-en:start -->
> A chain is reversible relative to π when, after starting in stationarity, every finite path has the same joint law as the path read backwards.
> <!-- bilingual-en:end -->

把一段平稳运行的录像倒放，若仅凭状态序列的概率规律无法判断播放方向，这就是可逆性。它是关于整段路径分布的性质，不只是某一个时刻的边际分布保持为 $\pi$。

二状态确定性交替链从 $\pi=(1/2,1/2)$ 启动时，可能路径只有 $1,2,1,2,\ldots$ 和 $2,1,2,1,\ldots$，两者概率相同；倒放只会把这两种等概率路径互换或保持，因此它相对于 $\pi$ 可逆。

> [!question]- 自检
> 只知道每个时刻都有 $X_n\sim\pi$，是否已经说明链相对于 $\pi$ 可逆？
>
> **答案：** 没有。平稳性只固定单时刻分布；可逆性还要求任意有限路径的联合分布在倒放后不变。

## 来源与核验

- [Cambridge Markov Chains notes, §11.3](https://www.statslab.cam.ac.uk/~rrw1/markov/M.pdf#page=47)：核对从平稳分布启动时，以有限路径时间反演不变来定义可逆性。
- [Levin and Peres, Markov Chains and Mixing Times, §1.6](https://pages.uoregon.edu/dlevin/MARKOV/mcmt2e.pdf#page=26)：交叉核对 reversible Markov chain 的路径解释。

---
aliases:
  - "相对于分布 π，CTMC 可逆是指平稳路径在时间反演下保持相同的有限维分布"
  - CTMC reversibility
  - Reversible continuous-time Markov chain
  - 连续时间链可逆性
student_os: knowledge-atom
atom_id: PROB-CTMC-034
atom_set: continuous-time-markov-chains
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[CTMC定义]]"
  - "[[CTMC平稳分布]]"
related:
  - "[[DTMC可逆性]]"
leads_to:
  - "[[CTMC详细平衡与可逆]]"
part_of:
  - "[[连续时间马尔可夫链.canvas]]"
---

# 相对于分布 π，CTMC 可逆是指平稳路径在时间反演下保持相同的有限维分布
<!-- bilingual-en:start -->
*Relative to π, a CTMC is reversible when its stationary path has the same finite-dimensional distributions after time reversal*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 设 $\pi$ 是 CTMC 的平稳分布，并从 $X_0\sim\pi$ 启动。若对任意 $m\ge1$ 与任意
> $$0=t_0<t_1<\cdots<t_m,$$
> 都有
> $$
> (X_{t_0},X_{t_1},\ldots,X_{t_m})
> \overset d=
> (X_{t_m-t_0},X_{t_m-t_1},\ldots,X_{t_m-t_m}),
> $$
> 就称该链**相对于 $\pi$ 可逆**。也就是说，平稳路径正放与倒放具有相同的有限维概率规律。
> <!-- bilingual-en:start -->
> A stationary CTMC is reversible relative to π when every finite collection of observations has the same joint law after the time axis is reversed.
> <!-- bilingual-en:end -->

在生成矩阵语言中，这表示平稳时间反演得到的 CTMC 与正向链具有同一个生成矩阵。可逆性讨论的是整段路径的联合分布，不只是每个单独时刻仍服从 $\pi$。

可逆性是一条关于整个平稳路径分布的对称性。怎样用逐对速率流检查它，以及它与平稳性的逻辑关系，见 [[CTMC详细平衡与可逆]]。

> [!question]- 自检
> 只知道每个时刻都有 $X_t\sim\pi$，是否已经说明 CTMC 相对于 $\pi$ 可逆？
>
> **答案：** 没有。平稳性只固定单时刻分布；可逆性还要求任意有限维路径分布在时间反演后不变。

## 来源与核验

- [Cambridge Applied Probability notes, §2.6](https://www.statslab.cam.ac.uk/~ps422/notes-new2015.pdf)：核对平稳时间反演过程与 CTMC 可逆性的定义。
- [Ward Whitt, Continuous-Time Markov Chains, §10](https://www.columbia.edu/~ww2040/Whitt_CTMCnotes121312.pdf)：交叉核对反演生成矩阵与路径可逆性。
- [[DTMC可逆性]]：对照离散时间的有限路径倒放定义。

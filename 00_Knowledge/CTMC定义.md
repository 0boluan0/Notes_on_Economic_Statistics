---
aliases:
  - "CTMC 是以连续时间为参数、取值于有限或可数离散状态空间的 Markov 纯跳过程"
  - Continuous-time Markov chain
  - CTMC
  - 连续时间 Markov 链
student_os: knowledge-atom
atom_id: PROB-CTMC-001
atom_set: continuous-time-markov-chains
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[Markov性]]"
  - "[[随机过程]]"
related:
  - "[[Markov充分状态]]"
  - "[[状态扩充]]"
  - "[[CTMC时间齐次转移函数]]"
leads_to:
  - "[[CTMC指数停留时间]]"
  - "[[生成矩阵约束]]"
  - "[[CTMC转移半群]]"
part_of:
  - "[[连续时间马尔可夫链.canvas]]"
---

# CTMC 是以连续时间为参数、取值于有限或可数离散状态空间的 Markov 纯跳过程
<!-- bilingual-en:start -->
*A CTMC is a Markov pure-jump process in continuous time on a finite or countable discrete state space*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 标准 CTMC 是定义在连续时间 $t\ge 0$ 上、取值于有限或可数状态空间 $S$ 的 Markov 纯跳过程。给定当前状态 $X_t$ 后，未来与过去条件独立；若再假设**[[CTMC时间齐次转移函数|时间齐次]]**，转移概率才只依赖时间差：
> $$
> \Pr(X_{t+s}=j\mid X_t=i)=p_{ij}(s).
> $$
> “时间连续”“状态离散”“Markov 性”和“时间齐次”是四件不同的事。
> <!-- bilingual-en:start -->
> A standard CTMC has continuous time, a finite or countable state space, jump paths, and the Markov property. Time homogeneity is an additional assumption, not part of conditional independence itself.
> <!-- bilingual-en:end -->

在更严格的路径表述中，通常取右连续、具有左极限的分段常值路径：过程在一个状态停留一段时间，然后跳到另一个状态。这里“chain”指离散状态结构；Brownian motion 虽然是连续时间 Markov 过程，却有连续状态和连续路径，不属于本主题的 CTMC。

时间非齐次 Markov 链仍可能满足
$$
\Pr(X_{t+s}=j\mid\mathcal F_t)=\Pr(X_{t+s}=j\mid X_t),
$$
但右侧可同时依赖 $t$ 与 $s$。因此不能从“未来只看现在”直接推出一个固定的 $P(s)$ 或固定生成矩阵 $Q$。

> [!example] 最小反例
> 设两状态设备白天故障率高、夜间故障率低。只要日历时点也决定未来速率，它可以是时间非齐次 Markov 过程；但不能用同一个只依赖时间差的 $p_{ij}(s)$ 描述全天。

> [!question]- 自检
> “过程在连续时间上观察”是否足以说明它是 CTMC？
>
> **答案：** 不足。还要说明状态空间、路径类型与 Markov 性；连续状态扩散也是连续时间过程，却不是这里的离散状态纯跳链。

## 来源与核验

- [MIT OCW 6.436J, Lecture 24](https://ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/087af3cedbc9def5b156c5e1665ac79c_MIT6_436JF18_lec24.pdf)：核对有限/可数状态、连续时间 Markov 定义及时间齐次的独立定义。
- [Cambridge Applied Probability notes](https://www.statslab.cam.ac.uk/~ps422/notes-new2015.pdf)：核对 countable-state pure-jump CTMC、跳时与半群的标准范围。
- [[01_Math/05_随机过程/02_随机过程的概念和分类.docx]]：仅确认课程使用“连续时间马尔可夫链”和“无后效性”的术语。

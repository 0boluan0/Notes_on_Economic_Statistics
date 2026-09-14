---
aliases:
  - "CTMC 平稳分布是在所有时长转移下保持不变的概率分布"
  - CTMC stationary distribution
  - CTMC invariant distribution
  - 连续时间链平稳分布
student_os: knowledge-atom
atom_id: PROB-CTMC-014
atom_set: continuous-time-markov-chains
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[CTMC转移半群]]"
  - "[[Markov矩阵左右约定]]"
related:
  - "[[Markov稳态分布]]"
leads_to:
  - "[[有限CTMC平稳生成矩阵判据]]"
  - "[[有限不可约CTMC稳态]]"
  - "[[有限CTMC平稳分解]]"
  - "[[可数CTMC稳态存在]]"
  - "[[CTMC详细平衡]]"
part_of:
  - "[[连续时间马尔可夫链.canvas]]"
---

# CTMC 平稳分布是在所有时长转移下保持不变的概率分布
<!-- bilingual-en:start -->
*A CTMC stationary distribution is a probability distribution left unchanged by transitions over every time horizon*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 在行概率约定下，概率向量 $\pi$ 是 CTMC 的平稳分布，当且仅当
> $$
> \pi P(t)=\pi\quad\text{对每个 }t\ge0,
> \qquad \pi_i\ge0,
> \qquad \sum_i\pi_i=1.
> $$
> <!-- bilingual-en:start -->
> Stationarity means that starting from $\pi$ leaves the distribution equal to $\pi$ after every elapsed time $t$.
> <!-- bilingual-en:end -->

## 直觉
<!-- bilingual-en:start -->
*Intuition*
<!-- bilingual-en:end -->

若 $X_0\sim\pi$，那么对每个固定时长 $t$ 都有 $X_t\sim\pi$。这表示每个观察时点的状态分布保持不变，不表示样本路径停在原状态不跳，也不把“平稳分布存在”自动升级为唯一性或长期收敛。

<!-- bilingual-en:start -->
If $X_0\sim\pi$, then $X_t\sim\pi$ at every fixed time $t$. This keeps the one-time state distribution unchanged; it does not say that sample paths never jump, nor does it by itself imply uniqueness or convergence from other initial laws.
<!-- bilingual-en:end -->

> [!example] 吸收态提供最小例子
> 若状态 $a$ 吸收，则从 $a$ 出发在任意时长后仍位于 $a$，所以点质量 $\delta_a$ 满足
> $$
> \delta_aP(t)=\delta_a\qquad(t\ge0).
> $$
> 因而 $\delta_a$ 是平稳分布；过程从其他状态出发时仍可能先发生跳跃。
>
> <!-- bilingual-en:start -->
> If state $a$ is absorbing, a chain started at $a$ remains there at every horizon. Hence the point mass $\delta_a$ is stationary, even though paths started elsewhere may still jump before reaching $a$.
> <!-- bilingual-en:end -->

> [!question]- 自检
> 若 $X_0\sim\pi$ 且 $X_t\sim\pi$ 对每个 $t$ 都成立，是否表示几乎每条路径都保持常数？
>
> **答案：** 不表示。平稳性保持的是各时点的分布；个体路径仍可在状态之间跳转。
>
> <!-- bilingual-en:start -->
> **Self-check:** If $X_0\sim\pi$ and $X_t\sim\pi$ for every $t$, must almost every path be constant?
>
> **Answer:** No. Stationarity preserves the distribution at each time; individual paths may still jump between states.
> <!-- bilingual-en:end -->

## 边界
<!-- bilingual-en:start -->
*Boundaries*
<!-- bilingual-en:end -->

- 平稳分布只回答“从该分布启动后是否保持不变”；存在性、唯一性以及从其他初始分布收敛是另外的命题。
- 有限状态下可用[[有限CTMC平稳生成矩阵判据|生成矩阵左零空间判据]]计算；该判据不能无条件外推到可数状态链。
- 可数状态下最终仍应针对所讨论的保守转移半群核对 $\pi P(t)=\pi$，并单独处理非爆炸与 regularity。

<!-- bilingual-en:start -->
- A stationary distribution answers only whether that initial law is preserved; existence, uniqueness, and convergence from other initial laws are separate claims.
- A finite-state chain can be checked through [[有限CTMC平稳生成矩阵判据|the generator left-null-vector criterion]], which does not extend unconditionally to countable state spaces.
- In a countable state space, stationarity ultimately remains a statement about the specified conservative transition semigroup, with non-explosion and regularity handled separately.
<!-- bilingual-en:end -->

## 来源与核验

- [MIT OCW 6.436J, Lecture 24, Proposition 2](https://ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/087af3cedbc9def5b156c5e1665ac79c_MIT6_436JF18_lec24.pdf#page=7)：核对 CTMC 平稳分布是随时间保持不变的概率分布。
- [Cambridge Applied Probability notes, §2.4](https://www.statslab.cam.ac.uk/~ps422/notes-new2015.pdf)：核对 invariant distribution 的含义及可数状态下需保留的过程条件。
- [[Markov稳态分布]]：对照离散时间的固定分布，同时保留连续时间对所有 $t\ge0$ 的半群不变要求。

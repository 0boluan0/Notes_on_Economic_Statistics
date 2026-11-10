---
aliases:
  - Finite irreducible CTMC stationary distribution
  - Unique stationary law of a finite irreducible CTMC
student_os: knowledge-atom
atom_id: PROB-CTMC-032
atom_set: continuous-time-markov-chains
atom_type: theorem
status: source-checked
requires:
  - "[[CTMC平稳分布]]"
  - "[[跳链决定CTMC沟通类]]"
related:
  - "[[有限不可约链稳态]]"
  - "[[有限CTMC平稳生成矩阵判据]]"
leads_to:
  - "[[有限CTMC平稳分解]]"
  - "[[有限CTMC全局收敛]]"
part_of:
  - "[[连续时间马尔可夫链.canvas]]"
---

<!-- bilingual-en:start -->
# The stationary law of a finite irreducible CTMC
<!-- bilingual-en:end -->

*有限不可约 CTMC 的唯一正平稳分布*

<!-- bilingual-en:start -->
Consider a machine that can fail and then be repaired. If both changes have positive rates, neither state can be cut off from the other. For a finite chain, this kind of mutual reachability guarantees exactly one stationary probability allocation, and no state is assigned probability zero. The theorem explains why solving the balance equations yields one meaningful distribution rather than several incompatible choices.
<!-- bilingual-en:end -->

考虑一台能够发生故障、也能够被修好的机器。若两种变化的速率都为正，任何一个状态都不会与另一个隔绝。对有限链，这种互相可达性保证平稳概率分配恰好只有一个，而且每个状态的概率都严格为正。该定理说明，为什么平衡方程会确定唯一的分布，而不会出现多种不同选择。

<!-- bilingual-en:start -->
## Two states that can reach each other
<!-- bilingual-en:end -->

*能够互达的两个状态*

<!-- bilingual-en:start -->
Let the rates $\lambda>0$ and $\mu>0$ be finite, measured in the same inverse-time unit. In the [[Markov矩阵左右约定|row convention]], the [[生成矩阵约束|generator]] and a probability row are
<!-- bilingual-en:end -->

设速率 $\lambda>0$、$\mu>0$ 均有限，并采用相同的时间倒数单位。在[[Markov矩阵左右约定|行约定]]下，[[生成矩阵约束|生成矩阵]]与概率行分别为

$$Q=\begin{pmatrix}-\lambda&\lambda\\\mu&-\mu\end{pmatrix},\qquad \pi=(a,1-a).$$

<!-- bilingual-en:start -->
The [[有限CTMC平稳生成矩阵判据|stationary equation]] is $a\lambda=(1-a)\mu$, so
<!-- bilingual-en:end -->

由[[有限CTMC平稳生成矩阵判据|平稳方程]] $a\lambda=(1-a)\mu$，得到

$$\pi=\left(\frac{\mu}{\lambda+\mu},\frac{\lambda}{\lambda+\mu}\right).$$

<!-- bilingual-en:start -->
Both entries are positive. For $\lambda=2$, $\mu=1$, the result is $(1/3,2/3)$: the state with the slower exit rate receives more stationary probability. A [[CTMC平稳分布|stationary law]] means that starting with these probabilities preserves them at all later times. It does not assert that a machine known to start in the first state already has distribution $\pi$ at time zero.
<!-- bilingual-en:end -->

两个分量都为正。若 $\lambda=2$、$\mu=1$，得到 $(1/3,2/3)$：出口较慢的状态获得更多平稳概率。[[CTMC平稳分布|平稳分布]]指从这些概率启动后，所有以后时刻仍保持这些概率；它没有声称，一台确定从第一个状态出发的机器在零时刻就已经服从 $\pi$。

> [!important] One positive stationary law · 唯一且处处为正
> <!-- bilingual-en:start -->
> Let a time-homogeneous CTMC have a finite nonempty state set $S$ and a conservative generator $Q$ with finite entries. Conservative here means nonnegative off-diagonal rates and zero row sums. Assume it is [[不可约链|irreducible]]: for every pair of distinct states, there is a finite directed path whose transition rates are all positive. Then there is exactly one stationary probability row $\pi$, and $\pi_i>0$ for every $i\in S$.
> <!-- bilingual-en:end -->
>
> 设时间齐次 CTMC 的状态集 $S$ 有限非空，生成矩阵 $Q$ 元素有限且保守；这里保守指非对角速率非负、每行和为零。假设链[[不可约链|不可约]]：每一对不同状态之间都有一条有限有向路径，沿途每次转移速率都为正。那么平稳概率行 $\pi$ 恰好只有一个，而且每个 $i\in S$ 都满足 $\pi_i>0$。

<!-- bilingual-en:start -->
Such a chain is automatically non-explosive because finitely many finite exit rates have a finite maximum. Irreducibility concerns paths, not necessarily direct transitions. A directed three-state cycle is irreducible even though half of its off-diagonal rates are zero. The [[跳链决定CTMC沟通类|positive-rate path criterion]] explains why this is the same communication structure as that of the embedded jump chain.
<!-- bilingual-en:end -->

这样的链自动非爆炸，因为有限个有限出口率有有限最大值。不可约说的是路径，不要求一步直达。例如单向三状态循环不可约，尽管一半非对角速率为零。[[跳链决定CTMC沟通类|正率路径判据]]说明，这与嵌入跳链具有相同的沟通结构。

<!-- bilingual-en:start -->
## Reduce existence and uniqueness to a finite discrete chain
<!-- bilingual-en:end -->

*把存在和唯一性转成有限离散链的结论*

<!-- bilingual-en:start -->
Write $q_i=-q_{ii}$ for the [[出口率|exit rate]]. Choose a number $\nu>\max_iq_i$ and let
<!-- bilingual-en:end -->

记 $q_i=-q_{ii}$ 为[[出口率]]。选 $\nu>\max_iq_i$，并令

$$R=I+Q/\nu,$$

<!-- bilingual-en:start -->
where $I$ is the [[单位矩阵|identity matrix]]. For $i\ne j$, $R_{ij}=q_{ij}/\nu\ge0$, and $R_{ii}=1-q_i/\nu>0$. Each row sums to one, so $R$ is a [[Markov矩阵|transition matrix]]. Its off-diagonal positive entries are exactly the positive-rate edges of $Q$, hence $R$ is irreducible as well. This is the candidate-event matrix of [[CTMC均匀化|uniformization]], not generally the matrix recording only genuine jumps.
<!-- bilingual-en:end -->

其中 $I$ 是[[单位矩阵]]。对 $i\ne j$，有 $R_{ij}=q_{ij}/\nu\ge0$；对角元 $R_{ii}=1-q_i/\nu>0$。每行和为一，因此 $R$ 是[[Markov矩阵|转移矩阵]]。它的非对角正元素恰好对应 $Q$ 的正率边，所以 $R$ 也不可约。这是[[CTMC均匀化]]中的候选事件矩阵，一般不是只记录真实跳跃的矩阵。

<!-- bilingual-en:start -->
The [[有限不可约链稳态|finite irreducible discrete-chain theorem]] now supplies exactly one probability row $\pi$ with $\pi R=\pi$, and every entry is positive. That theorem needs no aperiodicity for stationary existence or uniqueness. For any probability row $x$, elementary multiplication gives
<!-- bilingual-en:end -->

由[[有限不可约链稳态|有限不可约离散链定理]]，满足 $\pi R=\pi$ 的概率行恰好只有一个，而且每项都为正。该定理的平稳存在和唯一性本来就不需要非周期性。对任意概率行 $x$，直接乘法给出

$$xR=x+\frac{xQ}{\nu},\qquad xR=x\ \Longleftrightarrow\ xQ=0.$$

<!-- bilingual-en:start -->
By the [[有限CTMC平稳生成矩阵判据|finite generator criterion]], the probability solutions of $xQ=0$ are precisely the CTMC's stationary laws. Thus the CTMC and $R$ have exactly the same stationary probability rows. Existence, uniqueness and positivity transfer together. This also covers the one-state chain: $Q=(0)$, $R=(1)$ and $\pi=(1)$.
<!-- bilingual-en:end -->

再由[[有限CTMC平稳生成矩阵判据]]，$xQ=0$ 的概率解恰好就是 CTMC 的平稳分布。所以 CTMC 与 $R$ 具有完全相同的平稳概率行，存在性、唯一性和正性一起转移过来。单状态链也包括在内：$Q=(0)$、$R=(1)$、$\pi=(1)$。

> [!note] A periodic jump chain causes no obstruction · 跳链有周期也不妨碍
> <!-- bilingual-en:start -->
> In the two-state example, the genuine [[嵌入跳链|jump chain]] alternates deterministically between the states and has [[状态周期|period]] two. Nevertheless the CTMC has the unique positive law displayed above. No periodicity assumption belongs in this stationary-law theorem. The jump chain's own stationary row is $(1/2,1/2)$, which differs from the CTMC's when $\lambda\ne\mu$ because jump counts do not weight unequal waiting times.
> <!-- bilingual-en:end -->
>
> 两状态例的真实[[嵌入跳链]]在两状态间确定性交替，[[状态周期|周期]]为二，但 CTMC 仍具有上面唯一的正平稳分布。本平稳分布定理不需要任何周期条件。跳链自己的平稳行是 $(1/2,1/2)$；当 $\lambda\ne\mu$ 时，它不同于 CTMC 的平稳分布，因为按跳跃次数计数没有给不同停留时长加权。

<!-- bilingual-en:start -->
## What the assumptions do
<!-- bilingual-en:end -->

*这些条件各自排除了什么*

<!-- bilingual-en:start -->
Irreducibility is sufficient, not necessary, for uniqueness. If $Q=\begin{pmatrix}-1&1\\0&0\end{pmatrix}$, the chain is reducible but its only stationary row is $(0,1)$, concentrated on the absorbing second state. It lacks positivity on the whole state set. If instead $Q=0$ on two states, every probability row is stationary. The general finite case is organized by [[有限CTMC平稳分解|closed-class stationary decomposition]].
<!-- bilingual-en:end -->

不可约是唯一性的充分条件，不是必要条件。若 $Q=\begin{pmatrix}-1&1\\0&0\end{pmatrix}$，链可约，但唯一平稳行是 $(0,1)$，集中于第二个吸收态；它不在整个状态集上处处为正。若两状态的 $Q=0$，则每个概率行都平稳。一般有限情形由[[有限CTMC平稳分解|闭类平稳分解]]统一描述。

<!-- bilingual-en:start -->
The law $\pi$ in this theorem is determined by $Q$, not by the chosen initial distribution. This is distinct from proving that every initial law converges to $\pi$, which is handled in [[有限CTMC全局收敛|finite CTMC convergence]]. Finiteness also cannot be silently dropped: on an infinite state set, irreducibility alone need not give a stationary probability distribution; [[可数CTMC稳态存在|non-explosion and positive recurrence]] become substantive conditions.
<!-- bilingual-en:end -->

本定理的 $\pi$ 由 $Q$ 决定，而不是由所选初始分布决定。这与证明每个初始分布最终都趋近 $\pi$ 是不同的任务，后者见[[有限CTMC全局收敛]]。有限性也不能暗中删去：无限状态集上，仅不可约不保证平稳概率分布存在，[[可数CTMC稳态存在|非爆炸与正常返]]成为实质条件。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [Whitt, Continuous-Time Markov Chains, Theorem 4.1 and Theorem 4.2(a),(c), PDF pp.18–19 and 21 / printed pp.17–18 and 20](https://www.columbia.edu/~ww2040/4106S11/CTMCchapter121906.pdf#page=18): finite irreducible stationary uniqueness and the uniformization identity. The proof here isolates stationary existence/uniqueness from the stronger convergence result in the source.
- [Berestycki–Sousi, Applied Probability, Theorem 2.1, PDF pp.20–21; §2.4, pp.23–26](https://www.statslab.cam.ac.uk/~ps422/notes-new2015.pdf#page=20): positive-rate communication, invariant measures, the singleton case and finite-generator equivalence. [MIT 6.436J, Lecture 24, PDF p.7](https://ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/087af3cedbc9def5b156c5e1665ac79c_MIT6_436JF18_lec24.pdf#page=7): finite existence/uniqueness context.
- [[有限不可约链稳态|Finite irreducible discrete-chain theorem]] supplies the precise intermediate theorem; [[01_Math/05_随机过程/05_连续时间的马尔可夫链#5. 平稳分布、唯一性与收敛|permanent course §5]] supplies scope and notation. The two-state, singleton and reducible examples are checked directly.
<!-- bilingual-en:end -->

- Whitt 定理4.1及4.2(a),(c)（PDF18–19、21／印刷17–18、20页）：有限不可约稳态唯一与均匀化恒等式；本卡把平稳存在／唯一性从原文更强的收敛结论中单独证明。
- Berestycki–Sousi 定理2.1（PDF20–21页）及 §2.4（23–26页）：正率沟通、不变测度、单状态边界与有限生成矩阵等价。MIT6.436J 第24讲 PDF7页提供有限存在／唯一性语境。
- 既有有限不可约离散链定理提供精确的中间结论，永久课程第5节提供范围与记号；两状态、单状态、可约例直接核算。

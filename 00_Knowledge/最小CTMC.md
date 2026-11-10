---
aliases:
- Minimal CTMC
- Minimal process
- 墓地状态构造
student_os: knowledge-atom
atom_id: PROB-CTMC-027
atom_set: continuous-time-markov-chains
atom_type: definition
status: source-checked
requires:
- '[[CTMC爆炸]]'
- '[[CTMC转移半群]]'
leads_to:
- '[[爆炸后延拓]]'
part_of:
- '[[连续时间马尔可夫链.canvas]]'
---

<!-- bilingual-en:start -->
# Minimal continuous-time Markov chain
<!-- bilingual-en:end -->
*最小连续时间 Markov 链*

<!-- bilingual-en:start -->
A **minimal CTMC** runs the prescribed jump mechanism for as long as that construction lasts, then records termination if infinitely many jumps accumulate in finite time. It never sends an exploded path back into an ordinary state. The word “minimal” describes this choice of continuation; it does not mean a model with the fewest states or the smallest jump rates.
<!-- bilingual-en:end -->

**最小 CTMC**按指定的跳跃机制运行，直到该构造无法继续：如果无限次跳跃在有限时刻聚积，就记录终止，不把已经爆炸的路径送回普通状态。“最小”描述的是这种继续方式，不是说模型的状态数最少或跳转速率最小。

<!-- bilingual-en:start -->
Take a finite or countable state set $S$ and a [[生成矩阵约束|stable conservative generator]] $Q$: each state's total exit rate is finite, off-diagonal rates are nonnegative, and each row sums to zero. The [[嵌入跳链|jump chain]] selects destinations, while [[CTMC指数停留时间|exponential holding times]] supply the clock. Write $X_t$ for the state at time $t$, and let $\zeta$ be the [[CTMC爆炸|explosion time]], the limit of the genuine jump times. This card assumes that no separate killing mechanism is imposed before $\zeta$.
<!-- bilingual-en:end -->

取有限或可数状态集合 $S$，以及[[生成矩阵约束|稳定且保守的生成矩阵]] $Q$：各状态总出口率分别有限，非对角率非负，每行和为零。[[嵌入跳链]]选择去向，[[CTMC指数停留时间|指数停留时间]]提供时钟。用 $X_t$ 表示时刻 $t$ 的状态，$\zeta$ 表示[[CTMC爆炸|爆炸时刻]]，即真实跳时的极限。本卡假定在 $\zeta$ 之前没有另外设置杀死机制。

> [!example] Give termination its own label · 为终止设置单独标签
> <!-- bilingual-en:start -->
> For jumps $n\to n+1$ at rates $2^n$, starting from $1$, the successive states run through $1,2,3,\ldots$ before a finite random time $\zeta$ almost surely. Add a label $\Delta$ outside $S$ and set $X_t=\Delta$ from that time onwards. The label means “this run has terminated”. It is not an integer reached after a final jump from some largest integer.
> <!-- bilingual-en:end -->
>
> 对以速率 $2^n$ 从 $n$ 跳到 $n+1$ 的链，从 $1$ 出发后，几乎必然会在有限随机时刻 $\zeta$ 前依次走过 $1,2,3,\ldots$。在 $S$ 外增加标签 $\Delta$，从该时刻起令 $X_t=\Delta$，表示“这次运行已终止”。它不是从某个最大整数再跳一次所到达的整数。

<!-- bilingual-en:start -->
## The cemetery convention
<!-- bilingual-en:end -->
*墓地状态约定*

<!-- bilingual-en:start -->
The added **cemetery state** $\Delta\notin S$ is absorbing. The minimal construction is
<!-- bilingual-en:end -->

额外的**墓地状态** $\Delta\notin S$ 是吸收态。最小构造规定

$$X_t=\begin{cases}
\text{the state given by the jump construction},&0\le t<\zeta,\\
\Delta,&t\ge\zeta.
\end{cases}$$

<!-- bilingual-en:start -->
If $\zeta=\infty$, the second case is never used at a finite observation time. A zero-rate state already in $S$ is a different kind of absorption: the chain stays in that ordinary state forever without exploding. In either case the model is defined at every finite time after we include $\Delta$.

Before explosion, the usual paths are right-continuous with left limits. Assigning $X_\zeta=\Delta$ does **not** claim a left limit at $\zeta$ inside the original discrete space. In the increasing-state example the path keeps leaving every finite state as time approaches $\zeta$.
<!-- bilingual-en:end -->

若 $\zeta=\infty$，在任何有限观测时点都不会用到第二种情况。$S$ 中原有的零速率状态则是另一种吸收：链永久停在那个普通状态，没有爆炸。把 $\Delta$ 纳入之后，无论哪种情况，模型在每个有限时刻都有取值。

爆炸前，通常的路径右连续且有左极限；指定 $X_\zeta=\Delta$ **不代表**原离散空间内存在 $\zeta$ 处的左极限。在状态递增的例子中，时间越接近 $\zeta$，路径越不断离开每一个有限状态。

<!-- bilingual-en:start -->
## Where did the missing probability go?
<!-- bilingual-en:end -->
*缺失的概率去了哪里？*

<!-- bilingual-en:start -->
For $i,j\in S$, write $p_{ij}(t)=\Pr_i(X_t=j)$, where $\Pr_i$ means that $X_0=i$. This is a row-convention transition probability: the row fixes the starting state. Because a path in $S$ at time $t$ has not yet been killed, summing over the original states gives
<!-- bilingual-en:end -->

对 $i,j\in S$，记 $p_{ij}(t)=\Pr_i(X_t=j)$，其中 $\Pr_i$ 表示 $X_0=i$。这是行约定的转移概率：每一行固定出发状态。由于时刻 $t$ 仍在 $S$ 中的路径尚未被杀死，对原状态求和得到

$$\sum_{j\in S}p_{ij}(t)=\Pr_i(t<\zeta),\qquad
1-\sum_{j\in S}p_{ij}(t)=\Pr_i(\zeta\le t).$$

<!-- bilingual-en:start -->
The deficit is exactly the probability of explosion by time $t$, including explosion at $t$. Thus $P(t)=(p_{ij}(t))_{i,j\in S}$ has nonnegative entries and row sums at most one: a **subprobability** transition kernel. Its [[CTMC转移半群|semigroup identity]] still holds, because a path killed before the intermediate time can never return to contribute to a later state in $S$.
<!-- bilingual-en:end -->

缺额恰好是截至 $t$ 已爆炸的概率，也包括恰好在 $t$ 爆炸。因此 $P(t)=(p_{ij}(t))_{i,j\in S}$ 的元素非负，行和至多为一，是**次概率**转移核。它仍满足[[CTMC转移半群|半群恒等式]]：在中间时点之前已被杀死的路径，不可能重新返回并对后来处于 $S$ 的概率作贡献。

<!-- bilingual-en:start -->
If we also list the cemetery column, its entry is $p_{i\Delta}(t)=\Pr_i(\zeta\le t)$. The full row now sums to one; starting at $\Delta$ gives $p_{\Delta\Delta}(t)=1$ and $p_{\Delta j}(t)=0$ for $j\in S$. Probability has not disappeared from the full model. It has left the set of states we were displaying.
<!-- bilingual-en:end -->

如果同时列出墓地状态那一列，其元素是 $p_{i\Delta}(t)=\Pr_i(\zeta\le t)$，完整的行和又变成一。从 $\Delta$ 出发时，$p_{\Delta\Delta}(t)=1$，且对 $j\in S$ 有 $p_{\Delta j}(t)=0$。概率没有从完整模型中消失，只是离开了原先展示的状态集合。

> [!warning] A conservative Q is a local statement · Q 的保守性是局部条件
> <!-- bilingual-en:start -->
> Even though every row of $Q$ sums to zero, the minimal kernel on a countably infinite $S$ can lose mass through explosion. There need not be any direct finite-state jump rate to $\Delta$: an infinite sequence of ordinary jumps can end the construction. Do not infer unit row sums of $P(t)$ merely from zero row sums of $Q$.
> <!-- bilingual-en:end -->
>
> 即使 $Q$ 每行和为零，可数无穷 $S$ 上的最小核仍可能因爆炸而损失质量。不必有某个普通状态直接跳到 $\Delta$ 的速率：无限次普通跳跃本身就能使构造终止。因此，不能仅凭 $Q$ 行和为零就断言 $P(t)$ 行和为一。

<!-- bilingual-en:start -->
## Why it is called minimal
<!-- bilingual-en:end -->
*为什么叫“最小”*

<!-- bilingual-en:start -->
For a fixed starting state, couple another admissible [[爆炸后延拓|extension]] to use the same path until the first explosion. Paths not yet exploded contribute exactly the minimal probability $p_{ij}(t)$. An extension can additionally contribute paths that exploded and then returned to $j$. Therefore its original-state transition probability $\widetilde p_{ij}(t)$ satisfies
<!-- bilingual-en:end -->

固定出发状态，让另一个合法的[[爆炸后延拓|延拓过程]]在首次爆炸之前使用同一条路径。尚未爆炸的路径恰好贡献最小过程的概率 $p_{ij}(t)$；延拓还可能额外计入已经爆炸、后来又回到 $j$ 的路径。因此它在原状态上的转移概率 $\widetilde p_{ij}(t)$ 满足

$$\widetilde p_{ij}(t)
=p_{ij}(t)+\Pr_i(\widetilde X_t=j,\ \zeta\le t)
\ge p_{ij}(t).$$

<!-- bilingual-en:start -->
This entrywise comparison explains the adjective. Analytically, the minimal kernel is the minimal nonnegative solution of the [[Kolmogorov前后向方程|backward equations]] with the identity initial condition (at time zero the state is still its starting state); the source below proves this by successively including paths with finitely many jumps. Equality is possible when an extension adds no returning paths. For a nonexplosive construction, every finite observation is already before explosion, so this post-explosion freedom has no effect on its law.
<!-- bilingual-en:end -->

这个逐元素比较解释了“最小”的含义。在方程层面，最小核是以恒等初值（零时长后仍在起点）出发的[[Kolmogorov前后向方程|后向方程]]的最小非负解；下列教材通过逐步纳入有限次跳跃的路径来证明。若延拓没有增加返回路径，比较可以取等号。若构造非爆炸，每个有限观测时刻原本就在爆炸之前，这种爆炸后的选择便不影响概率律。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [Cambridge Applied Probability (2015), §1.2 p.5; §§1.6–1.7 pp.12–16; §1.8 pp.19–20](https://www.statslab.cam.ac.uk/~ps422/notes-new2015.pdf#page=5): the cemetery convention, jump construction, minimal kernel and additional returning paths. The survival/deficit identities follow by partitioning the full state space into $S$ and $\{\Delta\}$.
- [Norris, Markov Chains, Theorems 2.8.3–2.8.4 and proof, printed pp.85–87 / preview PDF pp.3–5](https://www.statslab.cam.ac.uk/~jrn10/Markov/s28.pdf#page=3): the minimal nonnegative backward solution, its semigroup, and the jump-chain/holding-time characterisation. The path comparison above explains this minimality without requiring the reader to solve an infinite differential system.
<!-- bilingual-en:end -->

- Cambridge 第1.2节第5页、第1.6–1.7节第12–16页、第1.8节第19–20页：核对墓地约定、逐跳构造、最小核及返回路径增加的概率。存活与缺额恒等式由把完整状态空间分成 $S$ 和 $\{\Delta\}$ 直接得到。
- Norris 定理2.8.3–2.8.4及证明，印刷第85–87页／预览PDF第3–5页：核对最小非负后向解、半群及跳链—停留时间刻画。正文通过路径比较解释最小性，无需先解无限微分方程组。

---
student_os: knowledge-atom
atom_id: PROB-DTMC-046
atom_type: theorem
aliases:
  - Countable-state Markov chain convergence
  - Discrete Markov chain convergence to equilibrium
  - 可数离散链平衡收敛
status: source-checked
---

<!-- bilingual-en:start -->
# Convergence of countable-state Markov chains
<!-- bilingual-en:end -->

*可数状态 Markov 链的平衡收敛*

<!-- bilingual-en:start -->
A chain may have infinitely many possible states and still approach a stable probability distribution. The crucial point is that this probability distribution must actually exist: infinitely many possible destinations do not automatically share one unit of probability in a stationary way. Once such a distribution exists, irreducibility and absence of a fixed-step cycle guarantee convergence to it.
<!-- bilingual-en:end -->

一条链有无限多个可能状态，也仍然可以趋近稳定的概率分布。关键在于：这个概率分布必须确实存在；无限多个去处并不会自动以某种平稳方式分享总共一单位的概率。有了这样的分布，再加上不可约和没有固定步数的周期阻塞，就能保证趋近它。

<!-- bilingual-en:start -->
Use a [[DTMC时间齐次转移核|time-homogeneous discrete-time Markov chain]] on a nonempty finite or countably infinite state set $S$. Its transition matrix $P=(p_{ij})$ uses rows: $p_{ij}$ is the probability of the next state being $j$ when the current state is $i$. The [[DTMC转移复合|matrix power]] entry $p_{ij}^{(n)}=(P^n)_{ij}$ describes the state after exactly $n$ steps, allowing intervening visits and returns.
<!-- bilingual-en:end -->

考虑非空有限或可数无穷状态集 $S$ 上的[[DTMC时间齐次转移核|时间齐次离散 Markov 链]]。转移矩阵 $P=(p_{ij})$ 采用行约定：$p_{ij}$ 是当前处于 $i$ 时，下一步到 $j$ 的概率。[[DTMC转移复合|矩阵幂]]的条目 $p_{ij}^{(n)}=(P^n)_{ij}$ 描述恰好 $n$ 步后的状态，允许中间访问和返回。

> [!important] Convergence theorem · 收敛定理
> <!-- bilingual-en:start -->
> Assume the chain is [[不可约链|irreducible]] (every state can reach every other), [[状态周期|aperiodic]] (return times have greatest common divisor one), and has a [[Markov稳态分布|stationary probability vector]] $\pi$: $\pi P=\pi$, $\pi_i\ge0$, $\sum_i\pi_i=1$. Then for every initial probability law $\mu$ and every state $j$, $\Pr_\mu(X_n=j)\to\pi_j$. In particular, $p_{ij}^{(n)}\to\pi_j$ for every fixed $i,j$. The stationary probability is unique.
> <!-- bilingual-en:end -->
>
> 假设链[[不可约链|不可约]]，即任意状态都能到达其他状态；[[状态周期|非周期]]，即可能回返步数的最大公约数为一；并存在[[Markov稳态分布|平稳概率向量]] $\pi$，满足 $\pi P=\pi$、$\pi_i\ge0$、$\sum_i\pi_i=1$。那么对任意初始概率律 $\mu$ 和任意状态 $j$，有 $\Pr_\mu(X_n=j)\to\pi_j$；特别地，每个固定 $i,j$ 都有 $p_{ij}^{(n)}\to\pi_j$。平稳概率分布唯一。

<!-- bilingual-en:start -->
Here $X_n$ is the state after $n$ steps, and $\Pr_\mu$ means that $X_0$ was drawn from $\mu$. The conclusion compares distributions across repeated runs at one late observation time. It does not say a path eventually stops, or that a time average has been taken. The [[Markov时间平均收敛|time-average theorem]] is a different statement.
<!-- bilingual-en:end -->

这里 $X_n$ 是第 $n$ 步后的状态，$\Pr_\mu$ 表示 $X_0$ 按 $\mu$ 抽取。结论比较的是反复运行时，在某个较晚观测时点的概率分布；它没有说一条路径最终停止，也没有先对时间求平均。[[Markov时间平均收敛]]是另一条结论。

<!-- bilingual-en:start -->
## An infinite chain whose limit is explicit
<!-- bilingual-en:end -->

*可以直接算出极限的无限状态例子*

<!-- bilingual-en:start -->
Use states $0,1,2,\ldots$ and let $\pi_j=2^{-(j+1)}$. These positive probabilities sum to one. At each step, keep the current state with probability $2/3$; with probability $1/3$, independently draw a state from $\pi$. The new draw may equal the old state. Thus
<!-- bilingual-en:end -->

取状态 $0,1,2,\ldots$，令 $\pi_j=2^{-(j+1)}$。这些正概率之和为一。每一步以概率 $2/3$ 保持当前状态，以概率 $1/3$ 独立按 $\pi$ 重抽一个状态；重抽结果也可能恰好等于旧状态。因此

$$
p_{ij}=\frac23\mathbf1_{\{i=j\}}+\frac13\pi_j.
$$

<!-- bilingual-en:start -->
The [[事件指示变量|indicator]] is one when $i=j$ and zero otherwise. Every transition probability is positive, so the chain is irreducible and has a one-step self-return. Also $\sum_i\pi_i p_{ij}=(2/3)\pi_j+(1/3)\pi_j=\pi_j$, establishing stationarity directly.
<!-- bilingual-en:end -->

[[事件指示变量|指示量]]在 $i=j$ 时为一，否则为零。每个转移概率都为正，因此链不可约，且有一步自返。再计算 $\sum_i\pi_i p_{ij}=(2/3)\pi_j+(1/3)\pi_j=\pi_j$，就直接验证了平稳性。

<!-- bilingual-en:start -->
Starting from $i$, no fresh draw occurs in the first $n$ steps with probability $(2/3)^n$; then the state is still $i$. If at least one draw occurs, the state has distribution $\pi$, which later steps preserve. Therefore
<!-- bilingual-en:end -->

从 $i$ 出发，前 $n$ 步一次重抽也没有发生的概率是 $(2/3)^n$，这时状态仍是 $i$；若至少重抽一次，状态分布就是 $\pi$，后续步骤也会保持它。因此

$$
p_{ij}^{(n)}
=\left(\frac23\right)^n\mathbf1_{\{i=j\}}
+\left[1-\left(\frac23\right)^n\right]\pi_j
\longrightarrow\pi_j.
$$

<!-- bilingual-en:start -->
## Why the assumptions have separate jobs
<!-- bilingual-en:end -->

*各项条件分别解决什么问题*

<!-- bilingual-en:start -->
Irreducibility prevents different closed regions from keeping different initial masses forever. Aperiodicity prevents fixed-step oscillation: the two-state matrix $\begin{pmatrix}0&1\\1&0\end{pmatrix}$ is irreducible and has stationary law $(1/2,1/2)$, but from a fixed state it alternates forever. Existence of a probability stationary law is the extra requirement that finite irreducible chains get automatically; on an infinite space, a nonnegative invariant weight sequence with infinite total cannot serve as $\pi$. See [[正常返与零常返|positive versus null recurrence]] for the return-time distinction.
<!-- bilingual-en:end -->

不可约防止多个闭区域永远保留各自的初始质量。非周期防止固定步数的振荡：两状态矩阵 $\begin{pmatrix}0&1\\1&0\end{pmatrix}$ 不可约，也有平稳律 $(1/2,1/2)$，但从固定状态开始会一直交替。存在平稳**概率**律是有限不可约链自动满足的额外条件；无限空间上，总质量无穷的非负不变权重不能充当 $\pi$。回返时间的区别见[[正常返与零常返]]。

<!-- bilingual-en:start -->
One proof constructs two chains on a common probability space: one starts from the chosen law, the other from $\pi$. Under the theorem’s hypotheses they can be coupled to agree from an almost surely finite time $T$ onward. Before that time they may differ, so for every state $j$ the discrepancy is bounded by $\Pr(T>n)$, which tends to zero. Norris gives the construction using a pair chain and the Markov property; the inequality explains how eventual agreement turns into distributional convergence.
<!-- bilingual-en:end -->

一种证明把两条链放在同一个概率空间：一条从指定分布开始，另一条从 $\pi$ 开始。在定理条件下，可以构造它们，使它们在某个几乎必然有限的时刻 $T$ 以后始终一致。在这之前它们可能不同，所以任一状态 $j$ 的概率差至多为 $\Pr(T>n)$，它趋于零。Norris 的证明利用成对状态链与 Markov 性构造这一过程；这个界解释了“最终一致”如何带来分布收敛。

<!-- bilingual-en:start -->
Uniqueness also follows from the conclusion: if another stationary probability $\rho$ existed, a chain started from $\rho$ would stay at $\rho$ at every step but would have to converge componentwise to $\pi$, forcing $\rho=\pi$. This theorem supplies the discrete-time step in [[可数CTMC平衡收敛|countable CTMC convergence]] when a continuous-time chain is observed at equally spaced times.
<!-- bilingual-en:end -->

唯一性也能由结论看出：若另有平稳概率 $\rho$，从它启动后，每一步分布都仍是 $\rho$，但定理又要求各分量趋于 $\pi$，只能有 $\rho=\pi$。当连续时间链按等间隔采样时，本定理为[[可数CTMC平衡收敛]]提供离散时间的证明步骤。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [Norris, Markov Chains, §1.8, Theorem 1.8.3, preview PDF pp. 1–3](https://www.statslab.cam.ac.uk/~jrn10/Markov/s18.pdf): the finite/countable irreducible aperiodic theorem with an invariant probability law, arbitrary initial distribution, and the coupling proof. The original pages were inspected as images because their text extraction is garbled.
- The refresh-chain formula and the two-state periodic counterexample above are calculated directly; neither is used as a substitute for the general theorem.
<!-- bilingual-en:end -->

- [Norris, Markov Chains，§1.8，定理1.8.3，预览PDF第1–3页](https://www.statslab.cam.ac.uk/~jrn10/Markov/s18.pdf)：有限／可数、不可约、非周期、有不变概率律时的收敛，任意初始分布，以及耦合证明。原页文字提取乱码，已直接阅读原图。
- 重抽链公式与两状态周期反例直接计算；这些算例不替代一般定理。

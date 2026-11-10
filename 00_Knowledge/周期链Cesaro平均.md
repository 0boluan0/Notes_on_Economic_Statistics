---
aliases:
- Cesaro convergence of periodic Markov chains
- Cesàro convergence of transition probabilities
- 转移概率的 Cesàro 平均
student_os: knowledge-atom
atom_id: PROB-DTMC-014
atom_set: discrete-time-markov-chains
atom_type: theorem
status: source-checked
requires:
- '[[正常返与零常返]]'
- '[[不可约链]]'
- '[[Markov稳态分布]]'
- '[[状态周期]]'
related:
- '[[有限链逐步收敛]]'
- '[[稳态唯一不推收敛]]'
- '[[Markov时间平均收敛]]'
part_of:
- '[[离散时间马尔可夫链.canvas]]'
---

<!-- bilingual-en:start -->
# Cesàro averages of Markov transition probabilities
<!-- bilingual-en:end -->

*周期链的 Cesàro 平均*

<!-- bilingual-en:start -->
A probability sequence can oscillate forever while its running arithmetic mean settles. A **Cesàro average** simply averages the first $N$ terms, with each term weighted by $1/N$. Here the terms are probabilities of being in a chosen state at different times. Averaging those probabilities is a different operation from observing one path and counting its visits.
<!-- bilingual-en:end -->

一列概率可以永远振荡，但它的累计算术平均仍能稳定。**Cesàro 平均**就是把前 $N$ 项相加后除以 $N$，每项权重都是 $1/N$。本卡平均的是不同时刻处于某状态的概率；这与观察一条路径、清点其中的访问次数，是不同的操作。

> [!example] Averaging the alternating snapshots · 平均交替的概率快照
> <!-- bilingual-en:start -->
> Let the two states swap deterministically, with row matrix $P=\begin{pmatrix}0&1\\1&0\end{pmatrix}$, and start at state $1$. At times $0,1,2,\ldots$, the probabilities of state $1$ are $1,0,1,0,\ldots$. For $N=1,2,3,4,5$, their averages are $1,1/2,2/3,1/2,3/5$. If $N$ is even, the average is exactly $1/2$; if $N$ is odd, it is $1/2+1/(2N)$. Hence the average approaches $1/2$ although the individual probabilities do not.
> <!-- bilingual-en:end -->
>
> 令两个状态确定性交换，行矩阵为 $P=\begin{pmatrix}0&1\\1&0\end{pmatrix}$，并从状态 $1$ 出发。在时刻 $0,1,2,\ldots$，位于状态 $1$ 的概率为 $1,0,1,0,\ldots$。取 $N=1,2,3,4,5$ 时，平均依次为 $1,1/2,2/3,1/2,3/5$。若 $N$ 偶数，平均恰为 $1/2$；若 $N$ 奇数，则为 $1/2+1/(2N)$。所以平均趋于 $1/2$，单项概率却不收敛。

<!-- bilingual-en:start -->
## Which probabilities are being averaged?
<!-- bilingual-en:end -->

*究竟在平均什么概率？*

<!-- bilingual-en:start -->
Let $(X_n)_{n\ge0}$ be a [[DTMC时间齐次转移核|time-homogeneous Markov chain]] on a finite or countably infinite state set $S$. Use [[Markov矩阵左右约定|row notation]], with $p_{ij}^{(n)}=\Pr_i(X_n=j)$ and $p_{ij}^{(0)}$ equal to one when $i=j$ and zero otherwise. Assume [[不可约链|irreducibility]], meaning mutual reachability of all states, and [[正常返与零常返|positive recurrence]], meaning finite mean positive return times. These conditions give a unique [[Markov稳态分布|stationary probability distribution]] $(\pi_j)$: its entries are nonnegative, sum to one, and satisfy $\pi_j=\sum_i\pi_i p_{ij}$.
<!-- bilingual-en:end -->

设 $(X_n)_{n\ge0}$ 为有限或可数无限状态集 $S$ 上的[[DTMC时间齐次转移核|时间齐次 Markov 链]]。采用[[Markov矩阵左右约定|行约定]]，记 $p_{ij}^{(n)}=\Pr_i(X_n=j)$；$p_{ij}^{(0)}$ 在 $i=j$ 时为一，否则为零。假设[[不可约链|不可约]]，即所有状态互达，并且[[正常返与零常返|正常返]]，即平均正回返时间有限。这些条件给出唯一[[Markov稳态分布|平稳概率分布]] $(\pi_j)$：各项非负、总和为一，且满足 $\pi_j=\sum_i\pi_i p_{ij}$。

> [!important] Average convergence does not need aperiodicity · 平均收敛不要求非周期
> <!-- bilingual-en:start -->
> For every fixed starting state $i$ and destination $j$, define the average over times $0$ through $N-1$, where $N\ge1$ is an integer. Then
> <!-- bilingual-en:end -->
>
> 对每个固定初态 $i$ 和目标状态 $j$，把时刻 $0$ 到 $N-1$ 的概率平均，其中整数 $N\ge1$。有
>
> $$\overline p_{ij,N}=\frac1N\sum_{n=0}^{N-1}p_{ij}^{(n)}\longrightarrow\pi_j.$$
>
> <!-- bilingual-en:start -->
> No condition on the [[状态周期|period]] is imposed. The bar denotes averaging, not a new transition rule. For a finite irreducible chain positive recurrence is automatic, so this applies whether or not the chain is aperiodic.
> <!-- bilingual-en:end -->
>
> 不对[[状态周期]]另加条件。横线表示取平均，不是在重新规定转移规则。有限不可约链自动正常返，因此无论是否非周期，都能使用此结论。

<!-- bilingual-en:start -->
The same coordinate limit holds for any initial distribution $\mu$: the averaged row $N^{-1}\sum_{n=0}^{N-1}\mu P^n$ approaches $\pi$ at every coordinate. On a finite state space, summing the finitely many coordinate errors also gives convergence of the entire probability row in absolute-error sum. The countable-state statement above is made coordinate by coordinate; it does not assert one bound uniform over every starting state and destination.
<!-- bilingual-en:end -->

任意初始分布 $\mu$ 也有相同的逐坐标极限：平均概率行 $N^{-1}\sum_{n=0}^{N-1}\mu P^n$ 的每个坐标趋于 $\pi$。有限状态下，把有限多个坐标误差相加，也得到整个概率行的绝对误差和趋零。上面的可数状态陈述逐个坐标成立，并未给出对全部初态和目标同时有效的统一误差界。

<!-- bilingual-en:start -->
## Why the average tends to stationary mass
<!-- bilingual-en:end -->

*平均为什么趋于平稳质量？*

<!-- bilingual-en:start -->
For a fixed destination $j$, let $V_{j,N}=\sum_{n=0}^{N-1}\mathbf1_{\{X_n=j\}}$ count visits along one realised path. The [[事件指示变量|indicator]] is one on a visit and zero otherwise. The [[Markov时间平均收敛|Markov time-average theorem]] gives $V_{j,N}/N\to\pi_j$ almost surely under the same irreducibility and positive-recurrence assumptions, even if the chain is periodic.
<!-- bilingual-en:end -->

固定目标 $j$，令 $V_{j,N}=\sum_{n=0}^{N-1}\mathbf1_{\{X_n=j\}}$ 统计一条实际路径上的访问次数；[[事件指示变量]]在访问时取一，否则取零。由[[Markov时间平均收敛|Markov 时间平均定理]]，在相同的不可约、正常返条件下，即使链有周期，仍有 $V_{j,N}/N\to\pi_j$ 几乎必然成立。

<!-- bilingual-en:start -->
The visit fraction is always between zero and one. This common integrable bound lets us pass to expectations using [[控制收敛定理|dominated convergence]]. Also, [[期望线性性|linearity of expectation]] does not require the visits to be independent. With $E_i$ denoting expectation for a start at $i$,
<!-- bilingual-en:end -->

访问比例始终介于零和一之间。这个共同可积上界允许使用[[控制收敛定理]]把极限传到期望中；而[[期望线性性]]不要求各次访问独立。用 $E_i$ 表示从 $i$ 出发时的期望，有

$$E_i\!\left[\frac{V_{j,N}}N\right]
=\frac1N\sum_{n=0}^{N-1}\Pr_i(X_n=j)
=\overline p_{ij,N}\longrightarrow\pi_j.$$

<!-- bilingual-en:start -->
The time-average theorem also holds from any initial law $\mu$, so the same argument proves the claim for $\mu$. This identifies the connection between the two kinds of average without identifying the objects themselves: the path fraction is random before observing the run; its expectation is a deterministic average of probabilities.
<!-- bilingual-en:end -->

时间平均定理也适用于任意初始律 $\mu$，同一论证便得到 $\mu$ 的结论。这样说明了两种平均之间的关系，却没有把它们当成同一个对象：路径比例在运行前是随机变量；其期望才是确定的概率平均。

<!-- bilingual-en:start -->
## What averaging does and does not fix
<!-- bilingual-en:end -->

*取平均能解决什么，不能解决什么？*

<!-- bilingual-en:start -->
If the unaveraged probabilities already converge, their arithmetic averages have the same limit. The useful extra case is oscillation: the swap has [[稳态唯一不推收敛|a unique stationary distribution without stepwise convergence]], while averaging balances its two phases. It does not change the original chain or make its late-time marginal settle.
<!-- bilingual-en:end -->

如果未平均的概率已经收敛，算术平均也有同一极限。平均新增的用途在于振荡：交换链[[稳态唯一不推收敛|平稳分布唯一，但不逐步收敛]]，平均则平衡了两个相位。它没有改变原链，也没有让原来的较晚时点边际分布停止振荡。

<!-- bilingual-en:start -->
Do not drop the class and recurrence conditions when demanding the same stationary probability limit. For two absorbing states, $P$ is the identity and the average law remains the chosen initial law forever. There is no common limit for all starts. In an irreducible null recurrent chain, individual long-run visit proportions can all be zero; a list of zeros is not a probability distribution. The formula here concerns a normalized stationary law in the positive recurrent case, not just existence of coordinate limits.
<!-- bilingual-en:end -->

若要求共同的平稳概率极限，不能删去类结构和回返条件。对两个吸收状态，$P$ 是单位矩阵，平均律永远等于所选初始律，不存在所有初态共用的极限。不可约零常返链中，每个单独状态的长期访问比例都可能为零；一串零不是概率分布。本公式针对正常返情形中归一化的平稳律，不只是逐坐标极限是否存在。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [Gallager, MIT 6.262 Chapter 5, Theorem 5.1.2, equations (5.15)–(5.16), PDF p.9; Theorem 5.1.4, PDF pp.10–11](https://ocw.mit.edu/courses/6-262-discrete-stochastic-processes-spring-2011/01d0892549619cb25d928f15ec7230ed_MIT6_262S11_chap05.pdf#page=9): visitation frequencies and their expectations, and identification of reciprocal mean return with stationary mass. Its count uses times $1$ through $N$; shifting to $0$ through $N-1$ changes the fraction by at most $1/N$.
- [Weber, Cambridge Markov Chains, Theorem 10.2, PDF pp.41–42](https://www.statslab.cam.ac.uk/~rrw1/markov/M.pdf#page=41): arbitrary initial laws and absence of an aperiodicity requirement. The expectation step and the swap/absorbing examples are checked directly.
<!-- bilingual-en:end -->

- Gallager 定理5.1.2、式(5.15)–(5.16)（PDF9）及定理5.1.4（PDF10–11）：核对访问频率及其期望，并把平均回返倒数识别为平稳质量。原文数时刻 $1$ 至 $N$；改为 $0$ 至 $N-1$，比例最多相差 $1/N$。
- Weber 定理10.2（PDF41–42）：核对任意初始律且无需非周期。正文取期望步骤、交换链和吸收例直接核验。

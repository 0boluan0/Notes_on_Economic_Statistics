---
aliases:
- Kac return-time formula
- Mean recurrence time formula
- Kac 回返公式
student_os: knowledge-atom
atom_id: PROB-DTMC-015
atom_set: discrete-time-markov-chains
atom_type: theorem
status: source-checked
requires:
- '[[正常返与零常返]]'
- '[[不可约链]]'
- '[[Markov稳态分布]]'
related:
- '[[Markov时间平均收敛]]'
part_of:
- '[[离散时间马尔可夫链.canvas]]'
---

<!-- bilingual-en:start -->
# Kac’s return-time formula
<!-- bilingual-en:end -->

*Kac 回返公式*

<!-- bilingual-en:start -->
A state that occupies a large share of a chain’s stationary probability must be revisited frequently. **Kac’s return-time formula** makes this precise: in an irreducible positive recurrent discrete-time chain, the stationary probability of a state is the reciprocal of its mean first positive return time. The average is measured in steps, starting from that state.
<!-- bilingual-en:end -->

平稳概率中占比大的状态，平均会更频繁地被重访。**Kac 回返公式**把它写成精确关系：在不可约、正常返的离散时间链中，一个状态的平稳概率等于从该状态出发的平均首次正回返步数的倒数。

> [!example] A failure state with mean return time five · 平均五步回到故障状态
> <!-- bilingual-en:start -->
> Order two states as working $W$ and failed $F$, and use the [[Markov矩阵左右约定|row transition matrix]] $P=\begin{pmatrix}0.9&0.1\\0.4&0.6\end{pmatrix}$. From $W$, reaching $F$ takes mean $h$ satisfying $h=1+0.9h$, so $h=10$. Starting at $F$, a first step that stays there is already a return; a first step to $W$ requires the additional mean $h$. Thus $m_F=1+0.4h=5$. The formula predicts stationary failure probability $1/5$, not $1/10$: hitting from $W$ and returning from $F$ are different questions.
> <!-- bilingual-en:end -->
>
> 按正常 $W$、故障 $F$ 排列状态，采用上面的[[Markov矩阵左右约定|行转移矩阵]]。从 $W$ 命中 $F$ 的平均时间 $h$ 满足 $h=1+0.9h$，所以 $h=10$。但从 $F$ 出发，第一步若仍在 $F$ 就已经回返；若去了 $W$，才需要再等平均 $h$ 步。因此 $m_F=1+0.4h=5$。公式给出平稳故障概率 $1/5$，不是 $1/10$：从 $W$ 命中与从 $F$ 回返是两个问题。

<!-- bilingual-en:start -->
## Conditions and the probability being computed
<!-- bilingual-en:end -->

*先说明条件与所求概率*

<!-- bilingual-en:start -->
Let the state set $S$ be finite or countably infinite, with a fixed [[DTMC时间齐次转移核|time-homogeneous transition kernel]] $p_{ij}$. Assume the chain is [[不可约链|irreducible]], so every state can reach every other, and [[正常返与零常返|positive recurrent]], so its first positive returns have finite means. Let $E_i$ denote expectation when starting at state $i$, and write
<!-- bilingual-en:end -->

令状态集 $S$ 有限或可数无限，转移核 $p_{ij}$ [[DTMC时间齐次转移核|不随时间改变]]。假设链[[不可约链|不可约]]，即所有状态互达，并且[[正常返与零常返|正常返]]，即首次正回返具有有限均值。用 $E_i$ 表示从状态 $i$ 出发的期望，记

$$\tau_i^+=\inf\{n\ge1:X_n=i\},\qquad m_i=E_i\tau_i^+.$$

<!-- bilingual-en:start -->
The unique [[Markov稳态分布|stationary probability distribution]] is a nonnegative list $(\pi_j)_{j\in S}$ with total mass one and $\pi_j=\sum_{k\in S}\pi_kp_{kj}$. These equations extend the finite row-vector definition by countable nonnegative sums. The return formula is
<!-- bilingual-en:end -->

唯一的[[Markov稳态分布|平稳概率分布]]是一组非负数 $(\pi_j)_{j\in S}$，总和为一，并满足 $\pi_j=\sum_{k\in S}\pi_kp_{kj}$。这里用非负可数和把有限概率行的定义扩展到可数状态。回返公式为

$$\boxed{\pi_i=\frac1{m_i}=\frac1{E_i\tau_i^+}>0\qquad(i\in S).}$$

<!-- bilingual-en:start -->
For a finite irreducible chain, positive recurrence follows automatically; for an infinite chain it is an essential extra condition. A [[状态周期|period]] greater than one is allowed. In column notation the stationary equation changes side, but the state probability $\pi_i$ and the mean return time remain the same.
<!-- bilingual-en:end -->

有限不可约链自动正常返；无限链则需要额外验证。允许[[状态周期]]大于一。若采用概率列，平稳方程的乘法方向会改变，但状态概率 $\pi_i$ 与平均回返时间都不变。

<!-- bilingual-en:start -->
## Count one cycle without counting its endpoint twice
<!-- bilingual-en:end -->

*数一个回返周期，避免重复计算端点*

<!-- bilingual-en:start -->
Start at $i$. One cycle consists of times $0,1,\ldots,\tau_i^+-1$; the return at time $\tau_i^+$ begins the next cycle. Let $\nu_j$ be the expected number of visits to state $j$ during this cycle:
<!-- bilingual-en:end -->

从 $i$ 出发，把时刻 $0,1,\ldots,\tau_i^+-1$ 视作一个周期；时刻 $\tau_i^+$ 的回返开启下一周期。令 $\nu_j$ 表示本周期内访问状态 $j$ 的平均次数：

$$\nu_j=E_i\!\left[\sum_{n=0}^{\tau_i^+-1}\mathbf1_{\{X_n=j\}}\right],\qquad \nu_i=1,\qquad \sum_{j\in S}\nu_j=m_i.$$

<!-- bilingual-en:start -->
The [[事件指示变量|indicator]] is one when the state is $j$ and zero otherwise. There is exactly one visit to $i$ in the half-open cycle: its initial visit. Summing the visit counts over all states counts every step of the cycle once, explaining the total $m_i$. Nonnegative sums justify this interchange even when $S$ is countably infinite; here the final total is finite.
<!-- bilingual-en:end -->

[[事件指示变量]]在状态为 $j$ 时取一，否则为零。这个左闭右开的周期里恰好访问一次 $i$，即初始访问；把所有状态的次数加起来，每个周期时点恰好计一次，所以总期望为 $m_i$。非负求和保证可数无限状态下仍可交换这些和；这里最终总量有限。

<!-- bilingual-en:start -->
Advancing every counted time by one replaces the initial state $i$ with the final state $i$, so no state's count changes. The Markov rule says the expected next count at $j$ is $\sum_k\nu_kp_{kj}$. Thus $\nu_j=\sum_k\nu_kp_{kj}$. More formally, for each fixed $n$, whether $n<\tau_i^+$ is already known from the history through $n$, so ordinary one-step conditioning applies before summing over $n$. Dividing by $m_i$ produces a stationary probability distribution. By uniqueness in an irreducible positive recurrent chain it is $\pi$, and its coordinate at $i$ is $\nu_i/m_i=1/m_i$.
<!-- bilingual-en:end -->

把每个计数时点向后推进一步，相当于删去初始状态 $i$、加上最后返回的同一个状态 $i$，各状态次数都不变。Markov 规则又给出下一步访问 $j$ 的期望为 $\sum_k\nu_kp_{kj}$，所以 $\nu_j=\sum_k\nu_kp_{kj}$。更正式地，对每个固定 $n$，事件 $n<\tau_i^+$ 已由截至 $n$ 的历史决定，可以先作普通一步条件分解，再对 $n$ 求和。除以 $m_i$ 就得到平稳概率分布；不可约正常返链中平稳分布唯一，因此它就是 $\pi$，在 $i$ 处为 $\nu_i/m_i=1/m_i$。

<!-- bilingual-en:start -->
For the machine, $\pi=(4/5,1/5)$ directly satisfies $\pi P=\pi$. From $F$, the mean time to hit $W$ is $1/0.4=2.5$; hence the mean positive return to $W$ is $1+0.1(2.5)=1.25$, whose reciprocal is $4/5$. Both coordinates agree with the cycle formula. More generally, under its conditions a stationary mass $0.2$ means a five-step mean positive return.
<!-- bilingual-en:end -->

设备例子中，$\pi=(4/5,1/5)$ 直接满足 $\pi P=\pi$。从 $F$ 命中 $W$ 平均需要 $1/0.4=2.5$ 步，所以从 $W$ 首次正回返平均需要 $1+0.1(2.5)=1.25$ 步，倒数为 $4/5$。两个坐标都与周期公式一致。一般地，在公式条件下，平稳质量 $0.2$ 就对应平均五步首次正回返。

> [!warning] Stationary mass is not a stepwise limit · 平稳质量不等于逐步极限
> <!-- bilingual-en:start -->
> In the alternating chain $0\leftrightarrow1$, each mean positive return is two and the stationary distribution is $(1/2,1/2)$. Starting at $0$, however, the distribution alternates forever. Kac's formula needs no aperiodicity. Its cycle-frequency interpretation is formalized by the separate [[Markov时间平均收敛|time-average theorem]], not by claiming the distribution at each late step has already converged.
> <!-- bilingual-en:end -->
>
> 在交替链 $0\leftrightarrow1$ 中，每个平均正回返时间都是二，平稳分布为 $(1/2,1/2)$；但从 $0$ 出发，逐步分布仍永久交替。Kac 公式不要求非周期。它的周期频率解释由另一个[[Markov时间平均收敛|时间平均定理]]形式化，并没有声称每个较晚时点的分布已经收敛。

<!-- bilingual-en:start -->
The normalization matters. For two absorbing states, any mixture $(a,1-a)$ is stationary, but each state's return time is one. The reciprocal one cannot equal an arbitrary mixture weight $a$. Restricting to a closed positive recurrent communicating class and normalizing its stationary law to total mass one restores the formula within that class. In an irreducible null recurrent chain, all $m_i$ are infinite and no stationary probability distribution exists: a list of zeros cannot be normalized. Finally, these return times count discrete steps; a continuous-time model's stationary proportions also depend on [[常返与停留时间|holding times]].
<!-- bilingual-en:end -->

归一化条件很重要。对两个吸收状态，任意混合 $(a,1-a)$ 都平稳，但每个状态一步就回返；倒数一不能等于任意混合权重 $a$。若限制在某个闭的正常返沟通类，并把该类的平稳律归一化为总质量一，公式就在类内恢复。不可约零常返链的全部 $m_i$ 都无穷，也不存在平稳概率分布，不能把一串零归一化。最后，这里的时间按离散步数计算；连续时间模型的平稳比例还取决于[[常返与停留时间|停留时间]]。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [Weber, Cambridge Markov Chains, Theorems 8.4–8.5 and 9.1, PDF pp.33–35 and37](https://www.statslab.cam.ac.uk/~rrw1/markov/M.pdf#page=33): expected visits during one return cycle, invariant-measure construction and uniqueness, and the countable irreducible positive-recurrence formula.
- [Gallager, MIT 6.262 Chapter 5, Theorem 5.1.4, PDF pp.10–11](https://ocw.mit.edu/courses/6-262-discrete-stochastic-processes-spring-2011/01d0892549619cb25d928f15ec7230ed_MIT6_262S11_chap05.pdf#page=10): equivalence of positive recurrence and a stationary probability distribution, with reciprocal mean return times. The machine, alternating and two-absorbing-state calculations are checked directly.
- [Berestycki and Sousi, Cambridge Applied Probability, §2.4, PDF pp.23–24](https://www.statslab.cam.ac.uk/~ps422/notes-new2015.pdf#page=23): the closing continuous-time boundary; stationary weights depend on holding times.
<!-- bilingual-en:end -->

- Weber 定理8.4–8.5、9.1（PDF33–35、37）支持回返周期中的平均访问次数、不变测度构造与唯一性，以及可数不可约正常返链的回返公式。
- Gallager 定理5.1.4（PDF10–11）支持正常返与平稳概率分布存在的等价及平均回返倒数公式；设备、交替和双吸收状态例子直接核验。
- Berestycki 与 Sousi 第2.4节（PDF23–24）核对结尾的连续时间边界：平稳权重还取决于停留时间。

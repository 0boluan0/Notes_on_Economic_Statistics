---
aliases:
  - Countable CTMC convergence to equilibrium
  - Positive recurrent CTMC convergence
  - 可数连续时间链平衡收敛
student_os: knowledge-atom
atom_id: PROB-CTMC-033
atom_set: continuous-time-markov-chains
atom_type: theorem
status: source-checked
requires:
  - "[[可数CTMC稳态存在]]"
related:
  - "[[有限CTMC全局收敛]]"
  - "[[有限链逐步收敛]]"
  - "[[有限CTMC平稳生成矩阵判据]]"
part_of:
  - "[[连续时间马尔可夫链.canvas]]"
---

<!-- bilingual-en:start -->
# Convergence to equilibrium for countable CTMCs
<!-- bilingual-en:end -->

*可数 CTMC 向平稳分布收敛*

<!-- bilingual-en:start -->
A stationary distribution stays unchanged if the process starts from it. Convergence to equilibrium answers a different question: if the process starts somewhere else, do its probabilities eventually approach that same distribution? For an irreducible non-explosive countable continuous-time Markov chain, existence of a stationary probability is enough. No additional discrete-time aperiodicity condition is required.
<!-- bilingual-en:end -->

平稳分布说明：过程若从这个分布启动，分布会保持不变。平衡收敛回答另一件事：若从其他地方启动，各状态的概率最终会不会接近同一个分布？对不可约、非爆炸的可数连续时间 Markov 链，有平稳概率就足够；无需另加离散时间的非周期条件。

<!-- bilingual-en:start -->
Use a [[CTMC时间齐次转移函数|time-homogeneous chain]] $X=(X_t)_{t\ge0}$ on a finite or countably infinite state set $S$. Its [[生成矩阵约束|conservative generator]] has finite exit rates $q_i=-q_{ii}=\sum_{j\ne i}q_{ij}$. Assume the [[最小CTMC|minimal process]] is [[CTMC爆炸|non-explosive]] and [[不可约链|irreducible]]. Write $p_{ij}(t)=\Pr_i(X_t=j)$ for the probability at elapsed time $t$ when starting at $i$.
<!-- bilingual-en:end -->

考虑有限或可数无穷状态集 $S$ 上的[[CTMC时间齐次转移函数|时间齐次链]] $X=(X_t)_{t\ge0}$。其[[生成矩阵约束|保守生成矩阵]]具有有限出口率 $q_i=-q_{ii}=\sum_{j\ne i}q_{ij}$。假设[[最小CTMC|最小过程]][[CTMC爆炸|非爆炸]]且[[不可约链|不可约]]。记 $p_{ij}(t)=\Pr_i(X_t=j)$，表示从 $i$ 启动、经过时长 $t$ 后位于 $j$ 的概率。

> [!important] The same limit from every start · 每个初态都有相同的概率极限
> <!-- bilingual-en:start -->
> If the chain has a [[CTMC平稳分布|stationary probability law]] $\pi$, then
> <!-- bilingual-en:end -->
>
> $$p_{ij}(t)\longrightarrow\pi_j\qquad(t\to\infty)$$
>
> <!-- bilingual-en:start -->
> for all $i,j\in S$. More generally, any initial probability law $\mu$ satisfies $\Pr_\mu(X_t=j)\to\pi_j$ for every $j$. The law $\pi$ is unique. For a nontrivial irreducible chain, its existence is equivalent to positive recurrence by [[可数CTMC稳态存在|the stationary-existence theorem]].
> <!-- bilingual-en:end -->
>
> 若链有[[CTMC平稳分布|平稳概率律]] $\pi$，则所有 $i,j\in S$ 都满足上式。更一般地，对任意初始概率律 $\mu$ 和每个 $j$，都有 $\Pr_\mu(X_t=j)\to\pi_j$，且 $\pi$ 唯一。对非平凡不可约链，由[[可数CTMC稳态存在|平稳存在定理]]，它的存在等价于正常返。

<!-- bilingual-en:start -->
This compares probabilities at late observation times across repeated runs. It does not say the sample path stops moving or that a particular run eventually has state $\pi$; $\pi$ is a probability vector, not a state.
<!-- bilingual-en:end -->

这里比较的是反复运行时，在较晚观察时点的概率。它没有说样本路径会停止运动，也没有说某次运行最终处于“状态 $\pi$”；$\pi$ 是概率向量，不是一个状态。

<!-- bilingual-en:start -->
## Equal-time observations are different from counting jumps
<!-- bilingual-en:end -->

*等间隔观察与数跳跃次数不同*

<!-- bilingual-en:start -->
Fix any duration $h>0$ and observe $Z_n=X_{nh}$ for $n=0,1,2,\ldots$. This discrete chain, often called a skeleton chain, records the state every $h$ time units. Its transition matrix is $P(h)$, by the [[CTMC转移半群|semigroup identity]]. It is different from the [[嵌入跳链|embedded jump chain]], which records a new state only after a genuine jump and ignores the length of each wait.
<!-- bilingual-en:end -->

固定任意时长 $h>0$，在 $n=0,1,2,\ldots$ 时观察 $Z_n=X_{nh}$。这条离散链常称骨架链，每隔 $h$ 个时间单位记录一次状态。由[[CTMC转移半群|半群恒等式]]，它的转移矩阵为 $P(h)$。它不同于[[嵌入跳链]]：后者只在真实跳跃后记录新状态，并忽略每次等待有多长。

<!-- bilingual-en:start -->
Irreducibility and the [[跳链决定CTMC沟通类|positive-rate route argument]] give $p_{ij}(h)>0$ for every pair $i,j$. In particular, staying at $i$ throughout the interval has probability $e^{-q_i h}>0$, so the skeleton has a one-step self-return and is [[状态周期|aperiodic]]. Also $\pi P(h)=\pi$ by stationarity. The [[可数Markov链平衡收敛|countable discrete-chain convergence theorem]] therefore gives
<!-- bilingual-en:end -->

不可约加上[[跳链决定CTMC沟通类|正率路线论证]]，给出每对 $i,j$ 都有 $p_{ij}(h)>0$。特别地，在整个区间内都不离开 $i$ 的概率为 $e^{-q_i h}>0$，所以骨架链有一步自返，因而[[状态周期|非周期]]。平稳性又给出 $\pi P(h)=\pi$。于是可用[[可数Markov链平衡收敛|可数离散链收敛定理]]，得到

$$p_{ij}(nh)\longrightarrow\pi_j\qquad(n\to\infty).$$

<!-- bilingual-en:start -->
The genuine jump chain may still have period two or another period. The convergence theorem is applied to the equal-time skeleton, not to that jump chain.
<!-- bilingual-en:end -->

真实跳链仍可能具有周期二或其他周期。这里把收敛定理用于等时骨架链，而不是那条跳链。

<!-- bilingual-en:start -->
## Fill the gaps between observation times
<!-- bilingual-en:end -->

*补上相邻观察时点之间的空隙*

<!-- bilingual-en:start -->
A limit along $h,2h,3h,\ldots$ alone does not yet prove a limit along all real times. For any $t,s\ge0$, the semigroup identity gives a useful local bound:
<!-- bilingual-en:end -->

只沿 $h,2h,3h,\ldots$ 收敛，还不足以证明沿所有实数时间收敛。对任意 $t,s\ge0$，半群恒等式给出一个局部界：

$$
p_{ij}(t+s)-p_{ij}(t)
=\sum_{k\ne i}p_{ik}(s)p_{kj}(t)
-\bigl(1-p_{ii}(s)\bigr)p_{ij}(t).
$$

<!-- bilingual-en:start -->
Both terms on the right before subtraction lie between zero and $1-p_{ii}(s)$. Consequently,
<!-- bilingual-en:end -->

右侧相减的两项都在零与 $1-p_{ii}(s)$ 之间，因此

$$
|p_{ij}(t+s)-p_{ij}(t)|
\le 1-p_{ii}(s)\le 1-e^{-q_i s}.
$$

<!-- bilingual-en:start -->
Given a tolerance $\varepsilon>0$, choose $h>0$ small enough that $1-e^{-q_i h}<\varepsilon/2$. For a late time $t$, write $t=nh+s$ with integer $n=\lfloor t/h\rfloor$ and $0\le s<h$. The skeleton theorem makes $|p_{ij}(nh)-\pi_j|<\varepsilon/2$ for large $n$, while the local bound controls the remaining interval. Adding the two errors gives $|p_{ij}(t)-\pi_j|<\varepsilon$. Only the fixed starting state's finite rate $q_i$ is used; a common upper bound on all exit rates is unnecessary.
<!-- bilingual-en:end -->

给定容差 $\varepsilon>0$，选足够小的 $h>0$，使 $1-e^{-q_i h}<\varepsilon/2$。对较晚时刻 $t$，写成 $t=nh+s$，其中整数 $n=\lfloor t/h\rfloor$、$0\le s<h$。骨架链定理使充分大的 $n$ 满足 $|p_{ij}(nh)-\pi_j|<\varepsilon/2$，局部界则控制剩余小区间。两部分误差相加便得 $|p_{ij}(t)-\pi_j|<\varepsilon$。这里只用固定起点的有限速率 $q_i$，不需要所有出口率有共同上界。

<!-- bilingual-en:start -->
For an arbitrary initial law, $\Pr_\mu(X_t=j)=\sum_i\mu_i p_{ij}(t)$. Choose finitely many initial states carrying all but an arbitrarily small amount of $\mu$'s mass. Their finitely many probability differences all vanish; the remaining weighted difference is bounded by the omitted mass because probabilities lie in $[0,1]$. This proves the stated conclusion for $\mu$ without an unjustified interchange of an infinite sum and a limit.
<!-- bilingual-en:end -->

任意初始律满足 $\Pr_\mu(X_t=j)=\sum_i\mu_i p_{ij}(t)$。选有限多个起始状态，使它们以外的 $\mu$ 质量任意小。这有限项的概率差都趋于零；其余加权差由于概率在 $[0,1]$ 中，至多为被省略的质量。这就证明了任意 $\mu$ 的结论，无须未经论证交换无穷和与极限。

> [!example] Period-two jumps, convergent clock-time probabilities · 跳链周期二，时钟概率仍收敛
> <!-- bilingual-en:start -->
> In an [[M-M-1稳态人数|M/M/1 queue]] with arrival rate $1$ and service-completion rate $2$, the system size jumps only by $+1$ or $-1$, so genuine jumps switch parity and the jump chain has period two. Exit rates are at most $3$, giving [[有界率非爆炸|non-explosion]], and its stationary system-size law is $\pi_j=2^{-(j+1)}$, $j\ge0$. The theorem gives $p_{ij}(t)\to2^{-(j+1)}$ from every initial size $i$. At a fixed clock time the number of completed jumps is random, so a forced jump-parity oscillation does not remain.
> <!-- bilingual-en:end -->
>
> [[M-M-1稳态人数|M/M/1 队列]]若到达率为 $1$、服务完成率为 $2$，系统人数每次真跳只增减一，因此真跳切换奇偶，跳链周期为二。出口率不超过 $3$，从而[[有界率非爆炸|非爆炸]]；系统人数的平稳律为 $\pi_j=2^{-(j+1)}$，$j\ge0$。本定理给出任意初始人数 $i$ 下 $p_{ij}(t)\to2^{-(j+1)}$。在固定钟表时刻，已完成的跳数是随机的，所以固定跳数的奇偶振荡不会保留下来。

<!-- bilingual-en:start -->
The assumptions still matter. A formal normalized solution of $\pi Q=0$ needs the process conditions in [[可数CTMC稳态存在|the existence theorem]] before this result applies. Without irreducibility, distinct closed classes can retain different initial masses, as [[有限CTMC全局收敛|finite-chain limits]] already illustrate. The theorem supplies a limit, not a numerical convergence rate or a uniform error bound over all initial states.
<!-- bilingual-en:end -->

前提仍然重要。形式上的归一化解 $\pi Q=0$ 必须配合[[可数CTMC稳态存在|存在定理]]的过程条件，才能应用本结果。删去不可约，不同闭类可保留不同初始质量，[[有限CTMC全局收敛|有限链极限]]已说明这一点。定理给出极限，没有给出数值收敛速率，也没有保证对所有初态统一的误差界。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [Berestycki–Sousi, Applied Probability, Lemma 2.21 and Theorem 2.22, PDF p.28](https://www.statslab.cam.ac.uk/~ps422/notes-new2015.pdf#page=28): irreducible non-explosive equilibrium convergence, the local continuity bound, and the equal-time skeleton proof. Theorem 2.1 (pp.20–21) gives positive transition probabilities.
- [Norris, Markov Chains, §1.8, Theorem 1.8.3, preview PDF pp.1–3](https://www.statslab.cam.ac.uk/~jrn10/Markov/s18.pdf): the finite/countable discrete theorem used for the skeleton, read from original page images and explained in its linked card. The arbitrary-initial-law extension here is also justified directly by a finite-mass truncation.
- [Berestycki–Sousi, §3.2, PDF p.34](https://www.statslab.cam.ac.uk/~ps422/notes-new2015.pdf#page=34): M/M/1 stationary law; the rate bound, parity and numerical specialization are checked directly.
<!-- bilingual-en:end -->

- Berestycki–Sousi 引理2.21、定理2.22（PDF28页）：不可约非爆炸链的平衡收敛、局部连续界及等时骨架证明；定理2.1（20–21页）给出转移概率正性。
- Norris §1.8 定理1.8.3（预览PDF1–3页）：骨架所需的有限／可数离散定理，已直接阅读原页图并在对应卡片解释；本卡另外用有限质量截断论证任意初始律。
- Berestycki–Sousi §3.2（PDF34页）：M/M/1 平稳律；速率界、奇偶与数值特例直接核算。

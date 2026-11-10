---
student_os: knowledge-atom
atom_id: PROB-BM-012
atom_type: theorem
aliases:
- Strong Markov property of Brownian motion
- Brownian strong Markov property
- 布朗运动强马尔可夫性
status: source-checked
---

<!-- bilingual-en:start -->
# The strong Markov property of Brownian motion
<!-- bilingual-en:end -->
*布朗运动的强 Markov 性*

<!-- bilingual-en:start -->
Brownian motion can be restarted when a rule based on observations so far tells us to stop. Measure future positions relative to the position at that random time: the resulting displacements form a fresh standard Brownian motion, independent of everything known at stopping. This permits rules such as “stop at the first hit of a barrier, or at the deadline if no hit occurs”, rather than only restarting at a predetermined clock time.
<!-- bilingual-en:end -->

布朗运动可以在“根据已经观察到的信息决定停止”的随机时刻重新开始。把未来位置都减去停止时的位置，得到的位移就是一条新的标准布朗运动，并独立于停止时已知的全部信息。因此，可以使用“首次达到某水平时停止，若一直没达到则在截止时刻停止”等规则，而不必只在预先确定的钟表时刻重新开始。

<!-- bilingual-en:start -->
Let $W=(W_t)_{t\ge0}$ be a [[标准布朗运动|standard Brownian motion]]. Use its [[自然滤过|natural filtration]] $\mathcal F_t=\sigma(W_s:0\le s\le t)$, the events determined by observing the whole path up to time $t$. The usual augmentation by null events and right-continuity is also permitted. These information choices do not reveal future Brownian increments.
<!-- bilingual-en:end -->

设 $W=(W_t)_{t\ge0}$ 是[[标准布朗运动]]。采用它的[[自然滤过]] $\mathcal F_t=\sigma(W_s:0\le s\le t)$，即观察截至 $t$ 的整段路径后能够判断的事件；也可以使用补入零概率事件并作右连续化后的通常增广。这些信息选择不会提前泄露未来的布朗增量。

<!-- bilingual-en:start -->
A random time $T$ is a [[停时定义|stopping time]] when $\{T\le t\}\in\mathcal F_t$ for every $t$: by time $t$ we can decide whether stopping has occurred. The [[停时信息集|stopped information]] $\mathcal F_T$ contains the events known when that rule stops. Assume $P(T<\infty)=1$; $T$ need not have a finite mean or a fixed upper bound.
<!-- bilingual-en:end -->

随机时间 $T$ 是[[停时定义|停时]]，是指每个 $t$ 都有 $\{T\le t\}\in\mathcal F_t$：到了 $t$，能够判断是否已经停止。[[停时信息集]] $\mathcal F_T$ 收集按这个规则停止时已经知道的事件。假设 $P(T<\infty)=1$；不要求 $T$ 有有限期望，也不要求存在固定上界。

> [!important] Restart the displacements, not the absolute position · 重新开始的是位移
> <!-- bilingual-en:start -->
> Define, for $u\ge0$,
> <!-- bilingual-en:end -->
>
> 对 $u\ge0$ 定义
>
> $$B^{(T)}_u=W_{T+u}-W_T.$$
>
> <!-- bilingual-en:start -->
> The process $B^{(T)}$ is standard Brownian motion and is independent of $\mathcal F_T$. In particular, $B^{(T)}_0=0$, it has continuous paths, and for $0\le u<v$ its increment has law $N(0,v-u)$. Disjoint future increments are mutually independent. The superscript $(T)$ labels the restarted process; it is not an exponent. On the null event $T=\infty$, its values may be assigned arbitrarily without changing the conclusion.
> <!-- bilingual-en:end -->
>
> 过程 $B^{(T)}$ 是标准布朗运动，并独立于 $\mathcal F_T$。特别地，$B^{(T)}_0=0$，路径连续；当 $0\le u<v$ 时，其增量服从 $N(0,v-u)$，不交的未来增量相互独立。上标 $(T)$ 是重新开始过程的标签，不是幂。在零概率事件 $T=\infty$ 上任意规定取值，不影响结论。

<!-- bilingual-en:start -->
Independence applies to the **whole future displacement process** and the stopped past, not merely to one future increment. It does not make $W_T$ independent of $T$, or make the absolute future position $W_{T+u}$ independent of $W_T$: the latter still includes the random starting position.
<!-- bilingual-en:end -->

独立性针对的是**整条未来位移过程**与停止时的过去信息，不只某一个增量。它没有使 $W_T$ 与 $T$ 独立，也没有使绝对位置 $W_{T+u}$ 与 $W_T$ 独立，因为绝对位置仍包含随机起点。

> [!example] First hit or deadline · 首次触及或截止
> <!-- bilingual-en:start -->
> Let $\tau_1=\inf\{s\ge0:W_s=1\}$, with $\inf\varnothing=\infty$, and $T=\min(\tau_1,2)$. By observing up to time $t$, we know whether the path has hit $1$ or the deadline has arrived, so $T$ is a stopping time bounded by $2$. If the hit occurs first, $W_T=1$; otherwise $W_T$ is the position at time $2$. In both cases, $W_{T+1}-W_T$ has law $N(0,1)$ independently of the stopped past. The theorem uses the same rule across both branches.
> <!-- bilingual-en:end -->
>
> 令 $\tau_1=\inf\{s\ge0:W_s=1\}$，并约定 $\inf\varnothing=\infty$，再令 $T=\min(\tau_1,2)$。观察到 $t$ 时，就知道是否已经到达 $1$ 或截止时刻，因此 $T$ 是不超过 $2$ 的停时。如果先触及水平，则 $W_T=1$；否则 $W_T$ 是时刻 $2$ 的位置。无论哪种情况，$W_{T+1}-W_T$ 都服从 $N(0,1)$，并独立于停止时的过去。同一个停止规则同时覆盖这两个分支。

<!-- bilingual-en:start -->
## Why a random stopping time is allowed
<!-- bilingual-en:end -->
*为什么允许随机停时*

<!-- bilingual-en:start -->
For a fixed time $s$, independent increments already give the restart property. A random $T$ is different: it was selected using the path. The stopping-time condition is exactly what lets us extend the fixed-time calculation, rather than simply substituting a random number into a deterministic-time result.
<!-- bilingual-en:end -->

对确定时刻 $s$，独立增量已经给出重新开始的性质。但随机 $T$ 是根据路径选择出来的，问题不同。停时条件正是把确定时刻计算延伸过来的关键，不能仅把确定时刻公式里的数字换成随机量就算证明。

<!-- bilingual-en:start -->
First suppose $T$ takes values on a discrete time grid. On the branch $\{T=s\}$, any event $A\in\mathcal F_T$ gives $A\cap\{T=s\}\in\mathcal F_s$. Fixed-time independence therefore applies separately on each branch. Summing the branches leaves the same Brownian future law and factors out $P(A)$.
<!-- bilingual-en:end -->

先设 $T$ 只取离散网格上的值。在分支 $\{T=s\}$ 上，任何 $A\in\mathcal F_T$ 都满足 $A\cap\{T=s\}\in\mathcal F_s$，所以每个分支都能使用确定时刻的独立性。把分支相加，未来仍具有相同的布朗律，并能把 $P(A)$ 因子提出来。

<!-- bilingual-en:start -->
For a general almost surely finite $T$, round it up to the next grid point:
<!-- bilingual-en:end -->

对一般几乎必然有限的 $T$，把它向上取整到网格点：

$$T_n=2^{-n}\lceil2^nT\rceil\downarrow T.$$

<!-- bilingual-en:start -->
Here $\lceil x\rceil$ is the smallest integer at least $x$. Each $T_n$ is a stopping time, since at a grid point $s=k2^{-n}$ its event $\{T_n\le s\}$ is $\{T\le s\}$. For $A\in\mathcal F_T$, the event $A\cap\{T_n=s\}$ is known by $s$, directly from the definition of $\mathcal F_T$. Thus the same branch calculation works for $T_n$, even if its countably many possible values are unbounded.
<!-- bilingual-en:end -->

$\lceil x\rceil$ 是不小于 $x$ 的最小整数。每个 $T_n$ 仍是停时，因为在网格点 $s=k2^{-n}$ 上，事件 $\{T_n\le s\}$ 就是 $\{T\le s\}$。对 $A\in\mathcal F_T$，由停时信息集的定义，事件 $A\cap\{T_n=s\}$ 在 $s$ 时已知。因此同样的分支计算适用于 $T_n$，即使它可能取可数多个、没有共同上界的值。

<!-- bilingual-en:start -->
To see what passes to the limit, fix finitely many future times $u_1,\ldots,u_m\ge0$ and any bounded continuous function $f$ of $m$ real numbers. The branch calculation says
<!-- bilingual-en:end -->

为了看清怎样取极限，固定有限个未来时点 $u_1,\ldots,u_m\ge0$，以及任意作用于 $m$ 个实数的有界连续函数 $f$。分支计算给出

$$E\!\left[\mathbf1_A f\bigl(W_{T_n+u_j}-W_{T_n}:1\le j\le m\bigr)\right]
=P(A)\,E\!\left[f\bigl(W_{u_j}:1\le j\le m\bigr)\right].$$

<!-- bilingual-en:start -->
The [[事件指示变量|indicator]] $\mathbf1_A$ is one on $A$ and zero otherwise. Path continuity makes each argument on the left converge to $W_{T+u_j}-W_T$. Boundedness of $f$ allows [[控制收敛定理|dominated convergence]], so the identity remains true with $T$ in place of $T_n$. Bounded continuous tests determine the finite-dimensional law and its independence from every $A\in\mathcal F_T$. The shifted path is continuous as well; together these facts give the stated Brownian process and independence of its whole future path. This argument needs almost-sure finiteness, not a finite $E[T]$.
<!-- bilingual-en:end -->

[[事件指示变量|指示变量]] $\mathbf1_A$ 在 $A$ 上为一，否则为零。路径连续性使左侧每个输入趋于 $W_{T+u_j}-W_T$。由于 $f$ 有界，可以用[[控制收敛定理]]把极限放进期望，所以把 $T_n$ 换成 $T$ 后恒等式仍成立。有界连续测试决定有限维分布，以及它与每个 $A\in\mathcal F_T$ 的独立性；平移后的路径也连续，合起来就得到所述布朗过程及整条未来路径的独立性。这里需要的是几乎必然有限，而不是 $E[T]$ 有限。

<!-- bilingual-en:start -->
## What this permits, and what it does not
<!-- bilingual-en:end -->
*它允许什么，不能直接推出什么*

<!-- bilingual-en:start -->
In the [[布朗反射原理|reflection principle]], restart at a first-hit time bounded by the observation deadline. The fresh future can then be replaced by its negative because Brownian motion is symmetric; strong Markov supplies the independence, while symmetry supplies the reflection. The strong Markov property itself only restarts the process and does not reflect it.
<!-- bilingual-en:end -->

在[[布朗反射原理]]中，先在被观察截止时刻截断的首次到达时重新开始，再利用布朗运动的对称性把新的未来换成其负数。强Markov性提供独立性，对称性提供反射；强Markov性本身只负责重新开始，并不负责反射。

<!-- bilingual-en:start -->
A rule that needs future observations is not automatically a stopping time. Nor may we replace the natural filtration by arbitrary extra information about the future: if the initial information already reveals $W_1$, the increment to time $1$ cannot remain independent of that information. Finally, this theorem concerns Brownian motion; a [[CTMC强Markov性|CTMC strong Markov theorem]] restarts a jump process with its own law, not a Brownian path.
<!-- bilingual-en:end -->

需要未来观察才能决定的规则，不会自动成为停时；也不能随意把未来信息加进自然滤过。如果初始信息已经泄露 $W_1$，到时刻 $1$ 的增量就不可能仍独立于这些信息。最后，本定理针对布朗运动；[[CTMC强Markov性]]重新开始的是遵循自身规律的跳跃过程，不是布朗路径。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [Pitman–Yor, *A Guide to Brownian Motion and Related Stochastic Processes*, §4.2, printed p.14 / PDF p.15](https://www.stat.berkeley.edu/~aldous/205B/pitman_yor_guide_bm.pdf#page=15): Brownian strong Markov property, stopped information and independence of post-stopping displacements. The grid-rounding argument above explicitly verifies the almost-surely-finite case used here.
- [MIT 15.070J, Lecture 7, §§1–2, PDF pp.1–3](https://ocw.mit.edu/courses/15-070j-advanced-stochastic-processes-fall-2013/aca1518a09539a09ddd37428ab0d0268_MIT15_070JF13_Lec7.pdf#page=1): stopping rules, strong independent increments after a first hit, and the reflection application; the capped-hit example avoids presupposing that an uncapped hit is finite.
<!-- bilingual-en:end -->

- Pitman–Yor 第4.2节，印刷14页／PDF15页：布朗强Markov性、停时信息与停止后位移的独立性；上述网格取整论证明确核验这里使用的几乎必然有限情形。
- MIT 第7讲第1–2节，PDF第1–3页：停止规则、首次到达后的强独立增量，以及反射应用；截断到达的例子不预先假设未截断的到达时间有限。

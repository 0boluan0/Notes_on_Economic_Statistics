---
aliases:
- Finite CTMC convergence to equilibrium
- Long-time limits of finite CTMCs
- 有限连续时间链收敛
student_os: knowledge-atom
atom_id: PROB-CTMC-016
atom_set: continuous-time-markov-chains
atom_type: theorem
status: source-checked
requires:
- '[[有限不可约CTMC稳态]]'
- '[[有限CTMC平稳分解]]'
related:
- '[[有限链逐步收敛]]'
- '[[周期链Cesaro平均]]'
- '[[稳态唯一不推收敛]]'
part_of:
- '[[连续时间马尔可夫链.canvas]]'
---

<!-- bilingual-en:start -->
# Long-time limits of finite CTMCs
<!-- bilingual-en:end -->

*有限 CTMC 的全局收敛*

<!-- bilingual-en:start -->
A finite continuous-time Markov chain can keep jumping forever while its probabilities settle. The important distinction is between **having a limit from each starting state** and **having the same limit from all starting states**. A time-homogeneous finite chain with finite rates always has the first property. It has the second exactly when there is only one closed communicating class. With several closed classes, the limit records the probabilities of eventually entering each one.
<!-- bilingual-en:end -->

有限连续时间 Markov 链可以一直跳动，但各状态的概率逐渐稳定。关键要区分**每个初态各自有极限**与**所有初态都有同一个极限**。时间齐次、速率有限的有限链总有第一种性质；恰好只有一个闭沟通类时，才有第二种性质。如果闭类有多个，极限会保留最终进入各类的概率。

<!-- bilingual-en:start -->
Throughout, use a nonempty finite state set $S$, a [[CTMC时间齐次转移函数|time-homogeneous CTMC]] with [[生成矩阵约束|generator]] $Q$, and the [[Markov矩阵左右约定|row convention]]. Write $P_{ij}(t)=\Pr_i(X(t)=j)$ for the probability of being at $j$ at time $t$ when starting at $i$. An initial probability row $\alpha$ has time-$t$ law $\alpha P(t)$. All off-diagonal rates are finite and nonnegative, and rows of $Q$ sum to zero. These finite-state assumptions give [[有界率非爆炸|non-explosion]] and keep all probability inside $S$.
<!-- bilingual-en:end -->

全篇采用非空有限状态集 $S$、[[CTMC时间齐次转移函数|时间齐次 CTMC]]、[[生成矩阵约束|生成矩阵]] $Q$ 和[[Markov矩阵左右约定|行约定]]。记 $P_{ij}(t)=\Pr_i(X(t)=j)$，表示从 $i$ 出发，时刻 $t$ 位于 $j$ 的概率。初始概率行 $\alpha$ 在时刻 $t$ 的分布为 $\alpha P(t)$。全部非对角速率有限且非负，$Q$ 每行和为零。这些有限状态条件保证[[有界率非爆炸|非爆炸]]，全部概率始终留在 $S$ 内。

<!-- bilingual-en:start -->
## Alternating jumps need not produce alternating snapshots
<!-- bilingual-en:end -->

*跳跃交替不等于固定时刻的概率振荡*

<!-- bilingual-en:start -->
Take states $0,1$, with rate $\lambda>0$ from $0$ to $1$ and $\mu>0$ in reverse. The [[嵌入跳链|embedded jump chain]] records only genuine state changes, so it alternates $0,1,0,1,\ldots$ and has [[状态周期|period]] two. A calendar-time observation is different: it does not fix the number of completed jumps.
<!-- bilingual-en:end -->

取状态 $0,1$，从 $0$ 到 $1$ 的速率为 $\lambda>0$，反向速率为 $\mu>0$。[[嵌入跳链]]只记录真实状态改变，所以按 $0,1,0,1,\ldots$ 交替，[[状态周期|周期]]为二。但在某个日历时刻观察，并没有固定此前完成的跳跃次数。

> [!example] A two-state probability limit / 两状态概率极限
> <!-- bilingual-en:start -->
> The generator and stationary row are
> <!-- bilingual-en:end -->
>
> 生成矩阵与平稳概率行为
>
> $$Q=\begin{pmatrix}-\lambda&\lambda\\\mu&-\mu\end{pmatrix},\qquad
> \pi=\left(\frac\mu{\lambda+\mu},\frac\lambda{\lambda+\mu}\right).$$
>
> <!-- bilingual-en:start -->
> Starting at $0$, let $f(t)=P_{00}(t)$. The [[Kolmogorov前后向方程|forward equation]] gives $f'(t)=-\lambda f(t)+\mu[1-f(t)]$, with $f(0)=1$. Subtract the equilibrium value $\mu/(\lambda+\mu)$ and solve the resulting exponential decay:
> <!-- bilingual-en:end -->
>
> 从 $0$ 出发，令 $f(t)=P_{00}(t)$。由[[Kolmogorov前后向方程|前向方程]]，$f'(t)=-\lambda f(t)+\mu[1-f(t)]$，初值 $f(0)=1$。减去平衡值 $\mu/(\lambda+\mu)$，剩下指数衰减，得到
>
> $$P_{00}(t)=\frac\mu{\lambda+\mu}+\frac\lambda{\lambda+\mu}e^{-(\lambda+\mu)t}\longrightarrow\frac\mu{\lambda+\mu}.$$
>
> <!-- bilingual-en:start -->
> For $\lambda=2,\mu=1$, this is $1/3+(2/3)e^{-3t}$. There is no odd/even oscillation in calendar time. The probabilities converge even though neither state is absorbing and the process continues to jump.
> <!-- bilingual-en:end -->
>
> 例如 $\lambda=2,\mu=1$ 时，得到 $1/3+(2/3)e^{-3t}$。日历时间里没有奇偶振荡。两个状态都不是吸收状态，过程仍会继续跳动，概率却已经收敛。

<!-- bilingual-en:start -->
## Which limit is selected when there are several closed classes?
<!-- bilingual-en:end -->

*有多个闭类时，初态选择哪个极限？*

<!-- bilingual-en:start -->
A [[闭沟通类|closed communicating class]] is a maximal mutually reachable region with no exit. For a CTMC, reachability is determined by finite paths of positive off-diagonal rates. List its closed classes as $C_1,\ldots,C_m$. Let $\pi^{(r)}$ be the unique stationary probability law on $C_r$, extended by zero outside it, as in [[有限CTMC平稳分解|stationary decomposition]]. Define the first entry time and its eventual entry probability by
<!-- bilingual-en:end -->

[[闭沟通类]]是内部互相可达、又没有出口的最大状态组。对 CTMC，可达由有限的非对角正速率路径决定。列出闭类 $C_1,\ldots,C_m$，并像[[有限CTMC平稳分解]]中那样，令 $\pi^{(r)}$ 为 $C_r$ 上唯一平稳概率律，在类外补零。定义首次进入时刻及最终进入概率：

$$\tau_r=\inf\{t\ge0:X(t)\in C_r\},\qquad
h_r(i)=\Pr_i(\tau_r<\infty).$$

<!-- bilingual-en:start -->
The infimum is the first such time, and is $\infty$ when the class is never visited. If $i\in C_r$, entry has already occurred at time zero, so $h_r(i)=1$. Finite size ensures eventual entry into some closed class with probability one; different closed classes cannot both be visited. Therefore $\sum_rh_r(i)=1$.
<!-- bilingual-en:end -->

这里的下确界表示首次这样的时刻；若永不进入，就取 $\infty$。如果 $i\in C_r$，时刻零就已进入，所以 $h_r(i)=1$。有限性保证以概率一最终进入某个闭类，而不同闭类不可能先后都访问，因此 $\sum_rh_r(i)=1$。

> [!important] The limit remembers entry probabilities / 极限保留进入各类的概率
> <!-- bilingual-en:start -->
> For every pair of states $i,j$, the limit exists and equals
> <!-- bilingual-en:end -->
>
> 对任意状态对 $i,j$，极限存在，且为
>
> $$P_{ij}(t)\longrightarrow L_{ij}=\sum_{r=1}^m h_r(i)\pi_j^{(r)}.$$
>
> <!-- bilingual-en:start -->
> For an arbitrary initial probability row $\alpha$, average over the starting state:
> <!-- bilingual-en:end -->
>
> 对任意初始概率行 $\alpha$，再按初态求平均：
>
> $$\alpha P(t)\longrightarrow\sum_{r=1}^m w_r(\alpha)\pi^{(r)},\qquad
> w_r(\alpha)=\sum_{i\in S}\alpha_i h_r(i).$$
>
> <!-- bilingual-en:start -->
> The coefficients $w_r(\alpha)$ are nonnegative and sum to one. They depend on the initial law through the eventual destination class. These are probability limits, not a statement that each individual path ends at a fixed state.
> <!-- bilingual-en:end -->
>
> 系数 $w_r(\alpha)$ 非负且和为一。它们通过最终进入哪个类而依赖初始律。这是概率分布的极限，不是说每条轨迹最后都停在一个固定状态。

<!-- bilingual-en:start -->
Return to the four-state generator in [[有限CTMC平稳分解|the decomposition example]]. State $0$ has total exit rate $3$ and jumps to $1$ with probability $1/3$, or to the absorbing state $3$ with probability $2/3$. Hence $h_1(0)=1/3$ for $C_1=\{1,2\}$, and $h_2(0)=2/3$ for $C_2=\{3\}$. The limit from $0$ is
<!-- bilingual-en:end -->

回到[[有限CTMC平稳分解|分解卡的四状态生成矩阵]]。状态 $0$ 的总出口率为 $3$，以概率 $1/3$ 跳到 $1$，以概率 $2/3$ 跳到吸收状态 $3$。所以对 $C_1=\{1,2\}$、$C_2=\{3\}$，有 $h_1(0)=1/3$、$h_2(0)=2/3$。从 $0$ 出发的极限为

$$\frac13(0,1/3,2/3,0)+\frac23(0,0,0,1)=(0,1/9,2/9,2/3).$$

<!-- bilingual-en:start -->
Starting at $1$ instead gives $(0,1/3,2/3,0)$; starting at $3$ gives $(0,0,0,1)$. All starts converge, but their limits differ. Each path eventually stays in only one class; the mixture weights describe probabilities across paths, not time spent alternating between distinct closed classes.
<!-- bilingual-en:end -->

改从 $1$ 出发，极限是 $(0,1/3,2/3,0)$；从 $3$ 出发，极限是 $(0,0,0,1)$。所有初态都收敛，但极限不同。每条路径最终只会留在一个闭类；混合权重描述不同路径进入各类的概率，不是在不同闭类间来回切换的时间比例。

<!-- bilingual-en:start -->
## Why continuous time removes the periodic obstruction
<!-- bilingual-en:end -->

*为什么连续时间不受离散周期阻塞*

<!-- bilingual-en:start -->
Let $q_i=-q_{ii}$ and choose $\nu>\max_i q_i$. The [[CTMC均匀化|uniformized transition matrix]] $R=I+Q/\nu$, where $I$ is the [[单位矩阵|identity]], has the same class graph as $Q$ and a positive self-loop at every state. Within each closed class it is finite, irreducible and aperiodic, so [[有限链逐步收敛|the discrete convergence theorem]] applies. The self-loops are virtual candidate events, not additional genuine jumps.
<!-- bilingual-en:end -->

令 $q_i=-q_{ii}$，选 $\nu>\max_i q_i$。[[CTMC均匀化|均匀化转移矩阵]] $R=I+Q/\nu$ 中，$I$ 是[[单位矩阵]]。$R$ 与 $Q$ 的类图相同，而且每个状态都有正自环。限制在任一闭类内，它有限、不可约且非周期，所以可以使用[[有限链逐步收敛|离散收敛定理]]。这些自环是候选事件中的虚事件，不是额外的真实跳跃。

<!-- bilingual-en:start -->
The discrete chain enters a closed class almost surely by [[有限链类分解|finite class decomposition]]. To obtain its full limit, first choose a large step bound $M$ so that the chance of not entering by $M$ is small. Conditional on an entry step at most $M$ and its entry state, the remaining transitions approach that class's stationary law. There are only finitely many such entry-step/state possibilities, so they can be handled together. The leftover probability is bounded by the small chance of late entry. This gives $(R^n)_{ij}\to L_{ij}$ with the entry weights above. Uniformization keeps the same genuine path, so these entry-class probabilities are also those of the CTMC.
<!-- bilingual-en:end -->

由[[有限链类分解]]，离散链几乎必然进入某个闭类。要得到它的全链极限，先选足够大的步数门槛 $M$，使尚未进入的概率很小。条件于不超过 $M$ 的进入步数和进入状态，后续转移趋于该类的平稳律。这样的进入步数／状态组合只有有限个，可以一并处理；剩余误差由很小的晚进入概率控制。于是 $(R^n)_{ij}\to L_{ij}$，权重正是上面的进入概率。均匀化保留同一真实路径，因此进入各类的概率也与 CTMC 相同。

<!-- bilingual-en:start -->
Now let $N(t)$ count an independent [[齐次泊松过程|Poisson clock]] of rate $\nu$. The matrix power $R^n$ records $n$ candidate transitions, with $R^0=I$. The factor $n!=1\cdot2\cdots n$ (and $0!=1$) is the factorial in the Poisson probability. The exact uniformization formula is
<!-- bilingual-en:end -->

现在令 $N(t)$ 计数一个独立的、速率为 $\nu$ 的[[齐次泊松过程|Poisson 时钟]]。矩阵幂 $R^n$ 记录 $n$ 次候选转移，约定 $R^0=I$；$n!=1\cdot2\cdots n$（以及 $0!=1$）是 Poisson 概率中的阶乘。精确均匀化公式为

$$P_{ij}(t)=\sum_{n=0}^{\infty}e^{-\nu t}\frac{(\nu t)^n}{n!}(R^n)_{ij}.$$

<!-- bilingual-en:start -->
For any $\varepsilon>0$, choose an integer $K\ge1$ such that $|(R^n)_{ij}-L_{ij}|\le\varepsilon$ for all $n\ge K$. Each probability entry lies between zero and one. Splitting the sum into small and large counts gives
<!-- bilingual-en:end -->

任取 $\varepsilon>0$，选整数 $K\ge1$，使所有 $n\ge K$ 都满足 $|(R^n)_{ij}-L_{ij}|\le\varepsilon$。每个概率项都在零与一之间，把求和分成小计数和大计数两部分，得到

$$|P_{ij}(t)-L_{ij}|\le\Pr(N(t)<K)+\varepsilon,
\qquad
\Pr(N(t)<K)=e^{-\nu t}\sum_{n=0}^{K-1}\frac{(\nu t)^n}{n!}\longrightarrow0.$$

<!-- bilingual-en:start -->
This proves convergence for every real time $t\to\infty$, not only along a sampled subsequence. Strict inequality $\nu>\max_iq_i$ is a convenient proof choice to make $R$ aperiodic; uniformization itself also permits equality, provided $\nu>0$. An embedded jump chain of period two or three therefore does not obstruct the finite CTMC limit.
<!-- bilingual-en:end -->

这证明的是所有实数时间 $t\to\infty$ 的收敛，而不只是某个采样子序列的收敛。取严格不等号 $\nu>\max_iq_i$ 是为了方便证明，使 $R$ 非周期；均匀化本身也允许等号，但仍须 $\nu>0$。因此，嵌入跳链的周期为二或三，都不能阻碍有限 CTMC 的极限。

<!-- bilingual-en:start -->
## When does every initial law have the same limit?
<!-- bilingual-en:end -->

*什么时候所有初始律都有同一个极限？*

<!-- bilingual-en:start -->
With exactly one closed class, its entry probability is one from every state, so every row of $L$ is the same stationary row $\pi$. With two or more closed classes, starting inside two different classes gives stationary limits supported on disjoint sets; the limits cannot agree. Thus a unique closed class is necessary and sufficient for a common limit from all starts. [[不可约链|Irreducibility]] is sufficient, but not necessary: transient states may feed one closed class. If $Q=0$, every state is an absorbing singleton and $P(t)=I$; each initial law is already constant, but there is no common limit unless $S$ has one state.
<!-- bilingual-en:end -->

恰好只有一个闭类时，从每个状态进入它的概率都为一，所以 $L$ 每一行都是同一个平稳行 $\pi$。有两个或更多闭类时，分别从不同类内出发，平稳极限支撑在互不相交的集合上，不可能相同。因此，唯一闭类是全部初态具有共同极限的充要条件。[[不可约链|不可约]]是充分条件，却不是必要条件：可以有暂态状态流向唯一闭类。若 $Q=0$，每个状态都是吸收单点，$P(t)=I$；每个初始律已经不变，但除非 $S$ 只有一个状态，否则不存在共同极限。

<!-- bilingual-en:start -->
The finite, homogeneous assumptions organize this entire conclusion. For a countably infinite chain, non-explosion alone does not supply a stationary probability or convergence to one. Nor does a stationary law by itself mean that a sample path eventually stops. Keep the [[有限CTMC平稳分解|set of invariant probability laws]], the limit selected by an initial law, and individual path behavior separate.
<!-- bilingual-en:end -->

整个结论建立在有限状态、时间齐次的范围内。对可数无限链，非爆炸本身不能保证存在平稳概率律，更不能直接保证趋近它。存在平稳律也不表示单条路径最终停止。需要分别理解[[有限CTMC平稳分解|全部不变概率律]]、初始律选择的极限，以及单条路径的行为。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [MIT 6.436J Lecture 24, PDF p. 7](https://ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/087af3cedbc9def5b156c5e1665ac79c_MIT6_436JF18_lec24.pdf#page=7): finite CTMC convergence from every start when there is one closed recurrent class, without a separate aperiodicity condition.
- [Cambridge Applied Probability, Theorem 2.1, PDF pp. 20–21; Theorem 2.22 and proof, PDF p. 28](https://www.statslab.cam.ac.uk/~ps422/notes-new2015.pdf#page=28): the positive-rate reachability relation and convergence within an irreducible non-explosive class with an invariant distribution. Its sampled-chain proof explains why continuous waiting removes the discrete timing obstruction.
- [Levin–Peres with Wilmer, §1.7, PDF pp. 31–34; Theorem 4.9, PDF pp. 68–69](https://pages.uoregon.edu/dlevin/MARKOV/mcmt2e.pdf#page=68): finite class structure and discrete irreducible aperiodic convergence. The entry-time split, Poisson-mixture limit and two-state equation above give the reduction and explicit calculations used here.
<!-- bilingual-en:end -->

- MIT 第 24 讲 PDF 第 7 页：核对唯一闭常返类时从全部初态收敛，不另加非周期条件，范围明确为有限 CTMC。
- Cambridge 定理 2.1（PDF20–21）、定理 2.22 及证明（PDF28）：核对正速率可达性，以及有不变分布的不可约非爆炸链收敛；其中的采样链证明解释连续等待如何消除离散的时间阻塞。
- Levin–Peres 等第 1.7 节（PDF31–34）和定理 4.9（PDF68–69）：核对有限类结构与离散不可约非周期收敛。进入时刻分割、Poisson 混合极限和两状态方程在正文直接推导。

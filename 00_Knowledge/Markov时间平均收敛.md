---
aliases:
- Markov chain ergodic time average
- Markov chain ergodic theorem
- 周期链时间平均
student_os: knowledge-atom
atom_id: PROB-DTMC-030
atom_set: discrete-time-markov-chains
atom_type: theorem
status: source-checked
requires:
- '[[正常返与零常返]]'
- '[[不可约链]]'
- '[[Markov稳态分布]]'
related:
- '[[周期链Cesaro平均]]'
- '[[Kac回返公式]]'
part_of:
- '[[离散时间马尔可夫链.canvas]]'
---

<!-- bilingual-en:start -->
# Time averages along a Markov chain
<!-- bilingual-en:end -->

*Markov 时间平均收敛*

<!-- bilingual-en:start -->
Suppose a machine is observed repeatedly along **one run**. We may want the fraction of observations in failure, or the average cost per observation. These are time averages along a [[样本路径|sample path]], not averages over many independently restarted machines. Successive observations can be dependent, yet suitable Markov-chain conditions still make their long-run average settle.
<!-- bilingual-en:end -->

设沿**一次运行**反复观察设备。我们可能关心故障观测所占的比例，也可能关心每次观测的平均成本。这是在一条[[样本路径]]上取时间平均，并不是把许多独立重启的设备平均。前后观测可以相互依赖；适当的 Markov 链条件仍能保证长期平均稳定。

> [!example] A periodic path can have a stable average · 周期路径也能有稳定平均
> <!-- bilingual-en:start -->
> In the deterministic two-state chain $1\to2\to1\to2\to\cdots$, give state $1$ cost $0$ and state $2$ cost $4$. Starting at $1$, the first five observed costs are $0,4,0,4,0$, with average $8/5$. For every even number $N$ of observations the average is exactly $2$; for odd $N$ it is $2-2/N$. Starting at $2$ changes the odd case to $2+2/N$. Both limits are $2$, although the state at each step keeps alternating.
> <!-- bilingual-en:end -->
>
> 对确定性交替链 $1\to2\to1\to2\to\cdots$，设状态 $1$ 的成本为 $0$，状态 $2$ 的成本为 $4$。从 $1$ 开始，前五个成本为 $0,4,0,4,0$，平均 $8/5$。若观测数 $N$ 为偶数，平均恰为 $2$；若为奇数，则为 $2-2/N$。从 $2$ 开始时，奇数情形改为 $2+2/N$。两种情况都趋于 $2$，虽然每一步的状态始终交替。

<!-- bilingual-en:start -->
## The path-average theorem
<!-- bilingual-en:end -->

*路径时间平均定理*

<!-- bilingual-en:start -->
Let $(X_n)_{n\ge0}$ be a [[DTMC时间齐次转移核|time-homogeneous Markov chain]] on a finite or countably infinite state set $S$. Assume it is [[不可约链|irreducible]], so all states communicate, and [[正常返与零常返|positive recurrent]], so mean first positive return times are finite. Its unique [[Markov稳态分布|stationary probability distribution]] is $(\pi_j)$, with $\pi_j\ge0$, $\sum_j\pi_j=1$ and $\pi_j=\sum_i\pi_i p_{ij}$ for row transition probabilities $p_{ij}$.
<!-- bilingual-en:end -->

设 $(X_n)_{n\ge0}$ 为有限或可数无限状态集 $S$ 上的[[DTMC时间齐次转移核|时间齐次 Markov 链]]。假设它[[不可约链|不可约]]，即所有状态互通，并且[[正常返与零常返|正常返]]，即平均首次正回返时间有限。唯一[[Markov稳态分布|平稳概率分布]]记为 $(\pi_j)$，满足 $\pi_j\ge0$、$\sum_j\pi_j=1$ 及 $\pi_j=\sum_i\pi_i p_{ij}$，这里 $p_{ij}$ 是行转移概率。

<!-- bilingual-en:start -->
A reward function $f:S\to\mathbb R$ assigns a finite real observation or cost to each state. In this version it is **bounded**: one finite constant $M$ satisfies $|f(j)|\le M$ for every state. For integer $N\ge1$, define the observed average $A_N(f)$ and its stationary target $\pi(f)$ by
<!-- bilingual-en:end -->

观测或成本函数 $f:S\to\mathbb R$ 为每个状态赋予一个有限实数。本版本要求它**有界**：存在共同有限常数 $M$，使每个状态都满足 $|f(j)|\le M$。对整数 $N\ge1$，定义实际平均 $A_N(f)$ 与平稳目标 $\pi(f)$：

$$A_N(f)=\frac1N\sum_{n=0}^{N-1}f(X_n),\qquad
\pi(f)=\sum_{j\in S}\pi_jf(j).$$

> [!important] From any initial law, with probability one · 任意初始律下，以概率一成立
> <!-- bilingual-en:start -->
> For every initial distribution $\mu$ and every fixed bounded $f$,
> <!-- bilingual-en:end -->
>
> 对任意初始分布 $\mu$ 和任意固定有界函数 $f$，
>
> $$A_N(f)\xrightarrow{a.s.}\pi(f).$$
>
> <!-- bilingual-en:start -->
> [[几乎必然收敛|Almost-sure convergence]] means that, under the path law determined by $\mu$ and the transition rule, a set of paths of probability one has this limit. The chain need not start at $\pi$, and it need not be aperiodic. The target series is absolutely convergent because $\sum_j\pi_j|f(j)|\le M$.
> <!-- bilingual-en:end -->
>
> [[几乎必然收敛]]表示：在 $\mu$ 与转移规则确定的路径概率下，一组概率为一的路径都具有该极限。不需要从 $\pi$ 开始，也不需要非周期。目标级数绝对收敛，因为 $\sum_j\pi_j|f(j)|\le M$。

<!-- bilingual-en:start -->
Take $f(x)=\mathbf1_{\{x=j\}}$, the [[事件指示变量|indicator]] of a visit to state $j$. Then $A_N(f)$ is the empirical visit fraction and the limit is $\pi_j$. On a finite state space every real-valued $f$ is bounded, and finite irreducibility already guarantees positive recurrence. These observations make the theorem especially simple for finite irreducible chains, including periodic ones.
<!-- bilingual-en:end -->

取 $f(x)=\mathbf1_{\{x=j\}}$，即访问状态 $j$ 的[[事件指示变量]]，则 $A_N(f)$ 就是经验访问比例，极限为 $\pi_j$。有限状态空间上的任意实值函数都自动有界，有限不可约也已保证正常返。因此定理对有限不可约链尤其方便，包括有周期的链。

<!-- bilingual-en:start -->
## Why return cycles produce the limit
<!-- bilingual-en:end -->

*回返周期为什么给出长期比例？*

<!-- bilingual-en:start -->
Fix a state $j$ and first start there. Divide the path into excursions from one visit to $j$ up to, but not including, the next visit. Each excursion starts afresh in the same state. The Markov rule at these return times makes the successive excursion lengths [[独立同分布|independent with the same distribution]]; the observations at successive ordinary times are not thereby independent. One can justify the restart by conditioning on each possible integer return time, whose occurrence is known from the history up to that time.
<!-- bilingual-en:end -->

固定状态 $j$，先从它出发。把路径分成一段段远游：从一次访问 $j$ 开始，到下一次访问之前为止。每段都从同一个状态重新开始；在这些回返时刻使用 Markov 规则，使各段长度[[独立同分布]]，但并没有让相邻普通时点的观测变成独立。这个重启性质可以通过对每个可能的整数回返时刻分别条件化来说明：是否在该时刻返回，已经由此前历史确定。

<!-- bilingual-en:start -->
Let $L_r$ be the length of excursion $r$ and let $m_j=E_jL_1<\infty$ be its mean, where $E_j$ means [[期望|expectation]] from state $j$. The [[强大数定律|iid strong law]] gives $(L_1+\cdots+L_r)/r\to m_j$ almost surely. The cumulative lengths bracket the observation time between the last completed excursion and the next one. Dividing this bracket by the number of excursions gives the reciprocal frequency
<!-- bilingual-en:end -->

令 $L_r$ 为第 $r$ 段长度，$m_j=E_jL_1<\infty$ 为其均值，$E_j$ 表示从 $j$ 出发的[[期望]]。由 [[强大数定律|iid 强大数定律]]，$(L_1+\cdots+L_r)/r\to m_j$ 几乎必然成立。累计长度把当前观测时点夹在最后一个完整周期与下一个周期之间；用周期数除这个夹界，就得到倒数频率：

$$\frac1N\sum_{n=0}^{N-1}\mathbf1_{\{X_n=j\}}
\longrightarrow\frac1{m_j}=\pi_j.$$

<!-- bilingual-en:start -->
The last equality is [[Kac回返公式|Kac's return-time formula]]. Starting elsewhere only adds the initial segment before first reaching $j$. In an irreducible recurrent chain this first arrival is finite almost surely: repeated returns to the starting state give repeated chances to follow a fixed positive-probability route to $j$. Its finite realised initial segment vanishes after division by $N$; finite expected initial waiting is not needed for this step. Mixing over a countable set of initial states proves the same probability-one conclusion for any initial law.
<!-- bilingual-en:end -->

最后一个等号是 [[Kac回返公式]]。从别处开始，只是多出首次到达 $j$ 之前的初始片段。在不可约常返链中，该首次到达几乎必然有限：反复回到初态，就反复获得沿固定正概率路线去往 $j$ 的机会。实际路径上有限的初始片段除以 $N$ 后消失；这一步不要求初始等待时间的期望也有限。再对可数个初态混合，即得到任意初始律下的概率一结论。

<!-- bilingual-en:start -->
For finite $S$, multiply each limiting visit fraction by $f(j)$ and add the finitely many terms. For countably infinite $S$, this exchange needs a tail argument. Choose a finite subset $F$ carrying stationary mass close to one. Its empirical mass tends to its stationary mass because it is a finite sum of visit fractions. The contribution outside $F$ to the difference between $A_N(f)$ and $\pi(f)$ has absolute value at most $M$ times the sum of the empirical and stationary masses outside $F$. Its limiting bound is therefore $2M\pi(S\setminus F)$, as small as desired. Countably many state-frequency statements hold together on one probability-one event. This proves the bounded-function conclusion without interchanging an uncontrolled infinite sum and limit.
<!-- bilingual-en:end -->

若 $S$ 有限，把各状态的极限访问比例乘以 $f(j)$，再有限求和即可。若 $S$ 可数无限，交换需要控制尾部。选有限子集 $F$，使其平稳质量接近一；由于是访问比例的有限和，$F$ 的经验质量趋于它的平稳质量。$F$ 之外对 $A_N(f)-\pi(f)$ 的贡献，绝对值不超过 $M$ 乘以其经验质量与平稳质量之和，极限上界因而为 $2M\pi(S\setminus F)$，可以任意小。可数个状态频率结论能在同一个概率一事件上同时成立。这样就证明了有界函数结论，没有不加控制地交换无限求和与极限。

<!-- bilingual-en:start -->
## What a long path tells us
<!-- bilingual-en:end -->

*一条长路径能说明什么？*

<!-- bilingual-en:start -->
In the opening swap, the stationary law is $(1/2,1/2)$, so the long-run cost is $(1/2)0+(1/2)4=2$. Choosing instead the indicator of state $1$ gives long-run fraction $1/2$ from either start, even though a point-start marginal continues to oscillate. The [[周期链Cesaro平均|Cesàro average of transition probabilities]] is its expectation-based companion, not the same random object.
<!-- bilingual-en:end -->

开头交换链的平稳律为 $(1/2,1/2)$，所以长期成本是 $(1/2)0+(1/2)4=2$。若改用状态 $1$ 的指示函数，则从任一初态出发，长期比例都为 $1/2$，尽管单点起始的逐步边际仍会振荡。[[周期链Cesaro平均|转移概率的 Cesàro 平均]]是与它相联系的期望对象，并非同一个随机对象。

> [!warning] A stationary mixture need not describe one realised path · 平稳混合不一定描述一条实际路径
> <!-- bilingual-en:start -->
> With two absorbing states and initial law $(1/2,1/2)$, every one-time distribution remains $(1/2,1/2)$. A realised path, however, stays forever in whichever state was initially selected. Its state-$1$ visit fraction is either zero or one, not $1/2$. Irreducibility is missing. Stationarity alone does not justify replacing a time average by an expectation under the full mixture.
> <!-- bilingual-en:end -->
>
> 对两个吸收状态，从 $(1/2,1/2)$ 初始律开始时，每个时点的分布始终都是 $(1/2,1/2)$；但一条实际路径永远停留在最初选中的状态，访问状态 $1$ 的比例只能为零或一，而不是 $1/2$。这里缺少不可约性。仅凭平稳，不能把时间平均替换为整个混合分布下的期望。

<!-- bilingual-en:start -->
On an infinite state space, do not remove the boundedness assumption without an appropriate integrability theorem. Also, almost-sure convergence supplies neither a universal finite run length nor independent-observation standard errors. It need not be monotone, and probability zero exceptional paths may remain. The result concerns the limiting average under the stated chain model; it does not certify that the model fits an observed system.
<!-- bilingual-en:end -->

在无限状态空间上，没有适当的可积性定理，不能直接删去有界假设。几乎必然收敛也不提供通用的有限运行长度，更不自动提供独立观测的标准误；逼近不必单调，仍可能存在概率为零的例外路径。结论针对所设链模型下的极限平均，并没有检验模型是否符合实际系统。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [Weber, Cambridge Markov Chains, Theorem 10.2 and proof, PDF pp.41–42](https://www.statslab.cam.ac.uk/~rrw1/markov/M.pdf#page=41): arbitrary initial distribution, bounded rewards in the positive recurrent case, no aperiodicity assumption, and the return-excursion strong-law proof for visit fractions. The finite-set tail argument above explains the bounded-reward extension stated there.
- [Gallager, MIT 6.262 Chapter 5, Lemmas 5.1.3–5.1.4 and Theorem 5.1.2, PDF pp.8–9; Theorem 5.1.4, PDF pp.10–11](https://ocw.mit.edu/courses/6-262-discrete-stochastic-processes-spring-2011/01d0892549619cb25d928f15ec7230ed_MIT6_262S11_chap05.pdf#page=8): almost-sure first arrival within a recurrent class, independent return cycles, visit-frequency limits and the stationary normalization. The two-state cost and absorbing-mixture examples are computed directly.
<!-- bilingual-en:end -->

- Weber 定理10.2与证明（PDF41–42）：任意初始分布、正常返下的有界观测、无需非周期，以及访问比例的回返周期强大数证明；正文用有限集合尾部界解释原文陈述的有界观测推广。
- Gallager 引理5.1.3–5.1.4、定理5.1.2（PDF8–9）及定理5.1.4（PDF10–11）：常返类内几乎必然到达、独立回返周期、访问频率极限及平稳归一化。两状态成本与吸收混合例直接计算。

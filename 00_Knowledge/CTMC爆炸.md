---
aliases:
- CTMC explosion
- Explosion time
- 爆炸时间
student_os: knowledge-atom
atom_id: PROB-CTMC-010
atom_set: continuous-time-markov-chains
atom_type: definition
status: source-checked
requires:
- '[[CTMC指数停留时间]]'
- '[[出口率]]'
- '[[嵌入跳链]]'
related:
- '[[有界率非爆炸]]'
- '[[纯生链爆炸判据]]'
leads_to:
- '[[最小CTMC]]'
- '[[爆炸后延拓]]'
part_of:
- '[[连续时间马尔可夫链.canvas]]'
---

<!-- bilingual-en:start -->
# Explosion in a continuous-time Markov chain
<!-- bilingual-en:end -->
*连续时间 Markov 链的爆炸*

<!-- bilingual-en:start -->
**Explosion** means that a chain makes infinitely many genuine changes of state before a finite time has elapsed. Imagine visiting states $1,2,3,\ldots$, with shorter and shorter random waits. Each wait can be positive, yet their total can be finite. The problem is not that one jump takes zero time; it is that an infinite sequence of waits can fit inside a bounded interval.
<!-- bilingual-en:end -->

**爆炸**指链在有限时长内完成了无限次真正的状态改变。想象依次访问状态 $1,2,3,\ldots$，每次随机等待越来越短：每段等待都可以为正，但总和仍可能有限。问题不在于某一次跳跃无需等待，而在于无限多段等待可能装进一个有界时间区间。

<!-- bilingual-en:start -->
A [[CTMC定义|continuous-time Markov chain]] records both states and their holding times. Its [[嵌入跳链|embedded jump chain]] retains the successive states but removes the clock. Explosion is a question about that clock: an infinite state sequence alone does not tell us whether it takes finite or infinite time to run through it.
<!-- bilingual-en:end -->

[[CTMC定义|连续时间 Markov 链]]同时记录状态和停留时长；[[嵌入跳链]]只保留依次访问的状态，去掉时钟。爆炸问的是时钟这一层：只知道状态序列无限长，无法判断走完它需要有限还是无限时间。

<!-- bilingual-en:start -->
## Locate the accumulation time
<!-- bilingual-en:end -->
*找出跳时的聚积点*

<!-- bilingual-en:start -->
Use a standard time-homogeneous pure-jump model on a finite or countable state set $S$, with finite [[出口率|exit rate]] at each state. Set $T_0=0$, and let $T_n$ be the time of the $n$th genuine state change. If that jump never occurs, set $T_n=\infty$, and likewise for later missing jumps. The **explosion time** is
<!-- bilingual-en:end -->

考虑有限或可数状态集合 $S$ 上的标准时间齐次纯跳模型，每个状态的[[出口率]]有限。令 $T_0=0$，$T_n$ 为第 $n$ 次真实状态改变的时刻。如果这次跳跃从不发生，就令 $T_n=\infty$，后面不存在的跳时也同样处理。**爆炸时刻**定义为

$$\zeta=\lim_{n\to\infty}T_n=\sup_{n\ge0}T_n.$$

<!-- bilingual-en:start -->
The sequence is increasing, so the limit exists in $[0,\infty]$. When there are infinitely many genuine jumps, write $H_n=T_n-T_{n-1}$ for the wait before jump $n$; then $\zeta=\sum_{n\ge1}H_n$. If the chain instead reaches an absorbing state, its next wait is infinite and $\zeta=\infty$: repeating the absorbing label in a completed jump-chain record does not create genuine jumps.
<!-- bilingual-en:end -->

这个序列递增，所以极限在 $[0,\infty]$ 中存在。若有无限次真实跳跃，记 $H_n=T_n-T_{n-1}$ 为第 $n$ 次跳跃前的等待，就有 $\zeta=\sum_{n\ge1}H_n$。若链进入吸收态，下一段等待为无穷，因而 $\zeta=\infty$；在补全的跳链记录中不断重复吸收态标签，并不产生真实跳跃。

> [!important] Specify the starting state · 说明从哪里出发
> <!-- bilingual-en:start -->
> Write $\Pr_i$ for probability when starting in $i$. Explosion occurs on the event $\{\zeta<\infty\}$. The model is explosive from $i$ if $\Pr_i(\zeta<\infty)>0$, and nonexplosive from $i$ if this probability is zero. Nonexplosion from every state means that every bounded time interval contains only finitely many genuine jumps almost surely. Infinitely many jumps over an infinite lifetime are compatible with nonexplosion.
> <!-- bilingual-en:end -->
>
> 用 $\Pr_i$ 表示从 $i$ 出发时的概率。爆炸事件是 $\{\zeta<\infty\}$；若 $\Pr_i(\zeta<\infty)>0$，就说从 $i$ 出发可能爆炸，若该概率为零，则从 $i$ 出发非爆炸。从所有状态都非爆炸，意味着每个有界时间区间内几乎必然只有有限次真实跳跃。在无限长的寿命中累计无限次跳跃，仍然可以非爆炸。

<!-- bilingual-en:start -->
## A finite rate at every state is not enough
<!-- bilingual-en:end -->
*每个状态分别有有限速率还不够*

<!-- bilingual-en:start -->
Fix one time unit. Start at state $1$ and allow only jumps $n\to n+1$, at rate $q_n=2^n$. This is a [[纯生链|pure-birth chain]]: its next state is fixed, while its wait is random. The [[CTMC指数停留时间|holding-time construction]] gives independent waits in the successive states,
<!-- bilingual-en:end -->

固定一个时间单位。从状态 $1$ 出发，只允许以速率 $q_n=2^n$ 从 $n$ 跳到 $n+1$。这是一条[[纯生链]]：下一站确定，等待时间随机。按照[[CTMC指数停留时间|停留时间构造]]，各状态中的等待相互独立，并且

$$H_n\sim\operatorname{Exp}(2^n),\qquad E[H_n]=2^{-n},\qquad n\ge1.$$

<!-- bilingual-en:start -->
Here $\operatorname{Exp}(2^n)$ uses the rate parameter of the [[指数分布|exponential distribution]]. The first three mean waits are $1/2,1/4,1/8$ time units. They are means, not a deterministic schedule. All rates are finite, but there is no single finite ceiling on them.
<!-- bilingual-en:end -->

这里 $\operatorname{Exp}(2^n)$ 使用[[指数分布]]的速率参数。前三段平均等待为 $1/2,1/4,1/8$ 个时间单位。这些是均值，不是确定的时间表。每个速率分别有限，但不存在一个共同的有限上界。

> [!example] Why the infinite sequence finishes in finite time · 为什么无限序列能在有限时间内走完
> <!-- bilingual-en:start -->
> The partial sums $Z_m=H_1+\cdots+H_m$ are nonnegative and increase to $\zeta$. The [[单调收敛定理|monotone convergence theorem]] therefore permits the expectation limit, and the [[几何级数|geometric series]] gives
> <!-- bilingual-en:end -->
>
> 部分和 $Z_m=H_1+\cdots+H_m$ 非负且递增到 $\zeta$。因此[[单调收敛定理]]允许交换极限与期望，再由[[几何级数]]得到
>
> $$E[\zeta]=\lim_{m\to\infty}E[Z_m]
> =\sum_{n=1}^{\infty}2^{-n}=1.$$
>
> <!-- bilingual-en:start -->
> A nonnegative quantity with finite expectation cannot be infinite with positive probability. To see why, $\zeta\ge M\mathbf1_{\{\zeta=\infty\}}$ for every $M>0$, where the [[事件指示变量|indicator]] is one on the event and zero otherwise. Hence $1=E[\zeta]\ge M\Pr(\zeta=\infty)$ for every $M$, forcing that probability to zero. Thus this chain explodes almost surely. Its **mean** explosion time is one; individual paths do not all explode at time one.
> <!-- bilingual-en:end -->
>
> 非负量若期望有限，就不能以正概率取无穷。因为对每个 $M>0$，都有 $\zeta\ge M\mathbf1_{\{\zeta=\infty\}}$，其中[[事件指示变量]]在事件发生时为一，否则为零。因此每个 $M$ 都满足 $1=E[\zeta]\ge M\Pr(\zeta=\infty)$，只能有该概率为零。所以这条链几乎必然爆炸。它的**平均**爆炸时间是一，并不是每条路径都在时刻一爆炸。

<!-- bilingual-en:start -->
Starting instead at a finite state $n_0\ge0$ gives $E_{n_0}[\zeta]=\sum_{n=n_0}^{\infty}2^{-n}=2^{1-n_0}$ by the same argument. In particular, starting at $0$ gives mean lifetime $2$ rather than $1$.
<!-- bilingual-en:end -->

若改为从有限状态 $n_0\ge0$ 出发，同一论证给出 $E_{n_0}[\zeta]=\sum_{n=n_0}^{\infty}2^{-n}=2^{1-n_0}$。特别地，从 $0$ 出发的平均寿命为 $2$，而不是 $1$。

<!-- bilingual-en:start -->
## What the definition does not decide
<!-- bilingual-en:end -->
*定义没有替我们决定什么*

<!-- bilingual-en:start -->
A uniform bound on exit rates is a [[有界率非爆炸|sufficient nonexplosion condition]], automatically available for a finite state space with finite rates. Unbounded rates alone do not prove explosion: the [[纯生链爆炸判据|pure-birth criterion]] gives a nonexplosive chain with rates $q_n=n$, started at $1$. The route and the accumulated waiting times matter.

Explosion also does not assign a state at $\zeta$. In the increasing-state example there is no last finite state, and no left limit in the original discrete state space at that time. A [[最小CTMC|minimal process]] records termination using an added cemetery label. Returning to an ordinary state requires an extra [[爆炸后延拓|post-explosion rule]].
<!-- bilingual-en:end -->

出口率有统一有限上界是[[有界率非爆炸|非爆炸充分条件]]；有限状态且各速率有限时自动满足。单凭速率无界也不能证明爆炸：由[[纯生链爆炸判据]]，从 $1$ 出发、速率为 $q_n=n$ 的链就非爆炸。实际经过的路线与等待时间的累积才是关键。

爆炸的定义也没有给 $\zeta$ 时刻指定状态。在状态不断递增的例子中，没有最后一个有限状态，且在原离散状态空间中不存在该时刻的左极限。[[最小CTMC|最小过程]]用额外的墓地标签记录终止；如果要返回普通状态，就必须增加[[爆炸后延拓|爆炸后规则]]。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [Berestycki and Sousi, Applied Probability (2015), §1.2 p.5; §1.5, Proposition 1.12 p.11; §1.6, Theorem 1.19 pp.13–14](https://www.statslab.cam.ac.uk/~ps422/notes-new2015.pdf#page=11): jump-time accumulation, the pure-birth waiting-time sum, the finite-expectation argument and uniform-rate nonexplosion. The powers-of-two sums above are computed explicitly.
- [Norris, Markov Chains, §2.9, Example 2.9.1, printed p.91 / preview PDF p.9](https://www.statslab.cam.ac.uk/~jrn10/Markov/s28.pdf#page=9): the same $2^n$ explosive birth process started at $0$, and the distinction between explosion and a continuation rule.
<!-- bilingual-en:end -->

- Berestycki–Sousi 第1.2节第5页、第1.5节命题1.12第11页、第1.6节定理1.19第13–14页：核对跳时聚积、纯生等待时间和、有限期望论证及一致有界率非爆炸。正文直接计算二的幂对应的几何和。
- Norris 第2.9节例2.9.1，印刷第91页／预览PDF第9页：核对从 $0$ 出发的同一个 $2^n$ 爆炸纯生模型，并区分爆炸与之后的继续规则。

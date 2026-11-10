---
student_os: knowledge-atom
atom_id: PROB-CTMC-039
aliases:
  - Strong Markov property for CTMCs
  - CTMC strong Markov property
  - 连续时间马尔可夫链的强马尔可夫性
atom_type: theorem
status: source-checked
requires:
  - "[[CTMC定义]]"
  - "[[CTMC时间齐次转移函数]]"
  - "[[停时定义]]"
  - "[[停时信息集]]"
related:
  - "[[Markov性]]"
  - "[[条件独立]]"
leads_to:
  - "[[CTMC指数停留时间]]"
  - "[[嵌入跳链]]"
  - "[[生成矩阵分解]]"
part_of:
  - "[[连续时间马尔可夫链.canvas]]"
---

<!-- bilingual-en:start -->
# The strong Markov property for CTMCs
<!-- bilingual-en:end -->

*CTMC 的强 Markov 性*

<!-- bilingual-en:start -->
A machine fails at a random time. Can we begin a fresh prediction at the moment of failure, using only the state it has just entered? The **strong Markov property** says yes for a standard continuous-time Markov chain when that observation time is a stopping time. It extends the usual fixed-time Markov rule to times selected from the observed path.
<!-- bilingual-en:end -->

机器在随机时刻故障。能否从“刚刚故障”的那个时刻开始，只用新进入的状态重新预测？对标准连续时间 Markov 链，只要这个观察时刻是停时，**强 Markov 性**就允许这样做。它把通常的固定时刻 Markov 规则，推广到根据已观察路径选出的时刻。

<!-- bilingual-en:start -->
The ordinary [[Markov性|Markov property]] makes a prediction at a specified clock time, such as 10 a.m. A [[停时定义|stopping time]] $T$ can instead be “the first time the device fails”. By time $t$, observing the path up to $t$ is enough to decide whether $T\le t$; no future observations are needed. Genuine jump times have this property in the standard right-continuous jump model.
<!-- bilingual-en:end -->

普通[[Markov性]]在指定日历时点作预测，例如上午10点。[[停时定义|停时]] $T$ 则可以是“设备第一次故障的时刻”：到了 $t$ 时，只观察截至 $t$ 的路径，就足以判断是否已有 $T\le t$，不需要查看未来。在标准右连续跳跃模型中，真实跳时就是这种时刻。

<!-- bilingual-en:start -->
## Restarting the probability law
<!-- bilingual-en:end -->

*重新开始的是概率规律*

<!-- bilingual-en:start -->
First consider a nonexplosive, [[CTMC时间齐次转移函数|time-homogeneous]] [[CTMC定义|pure-jump CTMC]] $X$ on a finite or countable state space, with right-continuous paths and a stable conservative generator: each exit rate is finite and the generator rows sum to zero. Nonexplosion means only finitely many genuine jumps occur on each bounded time interval. Let $T$ be a stopping time, finite with probability one, for the process's natural information.
<!-- bilingual-en:end -->

先考虑一个非爆炸、[[CTMC时间齐次转移函数|时间齐次]]的[[CTMC定义|纯跳 CTMC]] $X$，状态空间有限或可数，路径右连续，生成矩阵稳定且保守，即每个出口率有限、各行和为零。非爆炸表示每个有界时间区间内只有有限次真实跳跃。令 $T$ 是相对于该过程自然信息的停时，并且以概率一有限。

<!-- bilingual-en:start -->
Write $\mathcal F_T^X$ for the [[停时信息集|information available at $T$]], including the path seen so far and the observed stopping time. Let $p_{ij}(u)$ be the probability of reaching state $j$ after a further duration $u$ when starting at $i$. Then, for each $u\ge0$ and state $j$,
<!-- bilingual-en:end -->

用 $\mathcal F_T^X$ 表示[[停时信息集|到 $T$ 为止掌握的信息]]，包括已看到的路径以及发生停止的时刻。$p_{ij}(u)$ 表示从 $i$ 出发、再经过时长 $u$ 后处于 $j$ 的概率。对每个 $u\ge0$ 与状态 $j$，有

$$
\Pr(X_{T+u}=j\mid\mathcal F_T^X)=p_{X_T,j}(u)
\qquad\text{almost surely}.
$$

<!-- bilingual-en:start -->
The right side selects the transition row belonging to the state $X_T$ actually reached. More strongly, conditional on $X_T=i$, the entire future process $(X_{T+u})_{u\ge0}$ has the same law as a fresh chain started at $i$, independently of the earlier path under that condition. Time homogeneity makes the fresh law depend on elapsed time rather than the random calendar time $T$.
<!-- bilingual-en:end -->

右边选取实际到达状态 $X_T$ 所对应的转移行。更强地说，给定 $X_T=i$，整个未来过程 $(X_{T+u})_{u\ge0}$ 与一条从 $i$ 新启动的链同分布，并在这个条件下独立于此前路径。时间齐次使这个新规律只依赖经过时长，不再依赖随机日历时点 $T$。

> [!important] Conditional restart · 条件下重新开始
> <!-- bilingual-en:start -->
> The independence is [[条件独立|conditional on the state reached]]. Different reached states can have different future laws. The theorem does not say that $X_T$, $T$ and the future are unconditionally independent, nor does it reset the physical system or erase elapsed time.
> <!-- bilingual-en:end -->
>
> 独立性是[[条件独立|给定到达状态后的独立性]]，不同到达状态可以对应不同未来规律。定理没有说 $X_T$、$T$ 与未来无条件独立，也没有把物理系统重置或抹去已过去的时间。

<!-- bilingual-en:start -->
## Use it at a failure or a jump
<!-- bilingual-en:end -->

*在故障或跳跃时使用*

<!-- bilingual-en:start -->
Take states working $W$ and failed $F$, with positive failure rate $2$ and repair rate $3$ per hour. Start working, and let $T$ be the first failure time. It is a finite stopping time and $X_T=F$. The law after $T$ is that of the same chain started failed. In particular, the next repair wait has an [[CTMC指数停留时间|exponential holding-time law]] with mean $1/3$ hour, or twenty minutes, independently of the pre-failure path.
<!-- bilingual-en:end -->

取正常 $W$、故障 $F$ 两个状态，故障率为每小时 $2$、维修率为每小时 $3$，从正常出发。令 $T$ 为第一次故障时刻，它是有限停时且 $X_T=F$。$T$ 之后的规律与从故障状态启动同一条链相同。特别是，下一次维修完成的等待服从[[CTMC指数停留时间|指数停留时间分布]]，平均为 $1/3$ 小时，即二十分钟，并独立于此前的故障前路径。

<!-- bilingual-en:start -->
After waiting twenty minutes from that failure, the probability of still having had **no repair** is $e^{-1}$. This is not generally the probability of being failed at the endpoint: the machine might have been repaired and then failed again. Strong Markov restarts the whole chain; the no-repair event specifically selects its first holding time.
<!-- bilingual-en:end -->

从这次故障开始再等二十分钟，**始终没有修好**的概率是 $e^{-1}$。它一般不等于终点仍处于故障的概率，因为设备可能先被修好、又再次故障。强 Markov 性重新给出整条链的规律；“一直没修好”则专门选取其中的第一次停留事件。

<!-- bilingual-en:start -->
Apply the same result at every genuine jump time before explosion. Given the entered state, the future next-state law does not require all previous states. This is the step that makes the [[嵌入跳链|embedded jump chain]] Markov and supports the local [[生成矩阵分解|waiting-time/destination decomposition]]. A fixed-time Markov identity alone should not be silently applied to a random entry time without this theorem.
<!-- bilingual-en:end -->

在爆炸前的每个真实跳时都可以使用同一结果。给定刚进入的状态，后续去向规律不需要再保留此前全部状态。这一步让[[嵌入跳链]]具有 Markov 性，也支持局部的[[生成矩阵分解|等待与去向分解]]。在随机进入时刻使用固定时刻的 Markov 等式，需要本定理来连接。

<!-- bilingual-en:start -->
## Why observing the time without looking ahead matters
<!-- bilingual-en:end -->

*为什么不能偷看未来来选时刻*

<!-- bilingual-en:start -->
“The last jump before 1 p.m.” is usually not a stopping time. At 12:30, even if we have seen several jumps, we cannot know whether the last jump before 1 p.m. has already occurred; a later jump might still happen. Selecting a last jump therefore uses future information. Its post-jump segment is biased toward having no further jumps before the deadline, so a fresh-law restart is not licensed merely because the chosen time is random.
<!-- bilingual-en:end -->

“下午1点以前的最后一次跳跃”通常不是停时。到了12:30，即使已经观察到几次跳跃，也不能判断1点前的最后一次是否已发生，因为之后还可能再跳。选择最后一次跳跃使用了未来信息，它后面的路径片段会偏向于截止前不再跳。因此不能仅因为选的是一个随机时刻，就套用重新开始的规律。

<!-- bilingual-en:start -->
## The idea behind the extension from fixed times
<!-- bilingual-en:end -->

*从固定时刻推广的思路*

<!-- bilingual-en:start -->
For a bounded stopping time, approximate $T$ from above by the next point on a fine grid: $T_m=2^{-m}\lceil2^mT\rceil$, where the ceiling rounds upward to an integer. Each $T_m$ is again a stopping time. On a case $T_m=k2^{-m}$, apply the ordinary Markov rule at that fixed grid time; the case is determined by information already available there. Combining the disjoint cases gives the restart identity at $T_m$.
<!-- bilingual-en:end -->

对有界停时，可以从上方用细网格的下一格逼近：$T_m=2^{-m}\lceil2^mT\rceil$，其中上取整符号表示向上取到整数。每个 $T_m$ 仍是停时。在 $T_m=k2^{-m}$ 这个情形上，于固定网格时点使用普通 Markov 性；这个情形本身由当时已经掌握的信息确定。合并互斥情形后，就得到 $T_m$ 处的重新开始等式。

<!-- bilingual-en:start -->
As $m$ grows, $T_m$ decreases to $T$. Right continuity lets the observed states and future states approach those at $T$ and $T+u$; in a discrete state space, sufficiently nearby states on the right eventually agree. For an event $A\in\mathcal F_T$, the same event belongs to every $\mathcal F_{T_m}$, so we can test each identity by integrating over $A$. Bounded functions then allow a [[控制收敛定理|dominated-convergence]] limit. Finally, for an almost surely finite $T$, use the bounded-time result at $T\wedge n$ on the event $A\cap\{T\le n\}$, which belongs to $\mathcal F_{T\wedge n}$. On that event the truncated time is exactly $T$; letting $n$ increase recovers all finite stopping paths. This explains the role of the stopping-time and path assumptions, rather than treating the extension as a change of symbols.
<!-- bilingual-en:end -->

随着 $m$ 增加，$T_m$ 从上方下降到 $T$。右连续性使观察状态和未来状态分别趋于 $T$ 与 $T+u$ 处的状态；离散状态空间中，右侧足够近时最终会与该状态相同。对 $A\in\mathcal F_T$，同一个事件也属于每个 $\mathcal F_{T_m}$，因此可以在 $A$ 上对各个等式积分。使用有界函数后，可以通过[[控制收敛定理]]取极限。最后，对几乎必然有限的 $T$，在属于 $\mathcal F_{T\wedge n}$ 的事件 $A\cap\{T\le n\}$ 上，于有界时刻 $T\wedge n$ 使用结论。该事件上截断时刻恰好等于 $T$，再令 $n$ 增大，就覆盖所有有限停止的路径。这说明停时与路径假设为何必要，而不是单纯把固定时间符号换成随机符号。

<!-- bilingual-en:start -->
## An explosive chain needs a lifetime boundary
<!-- bilingual-en:end -->

*爆炸链需要存在时间的边界*

<!-- bilingual-en:start -->
For the [[最小CTMC|minimal construction]] of a countable stable conservative Q-matrix, let $\zeta$ be its [[CTMC爆炸|explosion time]]. The corresponding restart statement applies on $T<\zeta$, conditional on the reached state. The restarted law includes any possible future explosion; it is not conditioned to survive forever. After explosion the minimal process remains at its cemetery state. Another [[爆炸后延拓|continuation rule]] must be specified and checked separately, so the theorem should not be extended to an unspecified post-explosion model.
<!-- bilingual-en:end -->

对可数状态、稳定保守 Q 矩阵的[[最小CTMC|最小构造]]，记 $\zeta$ 为[[CTMC爆炸|爆炸时刻]]。相应的重新开始结论用于 $T<\zeta$，并给定到达状态。重新开始的概率律包括此后可能发生的爆炸，并没有条件化为永远存活。爆炸后最小过程停留在墓地状态；其他[[爆炸后延拓|继续规则]]需要另行规定并检查，不能把定理自动套到尚未指定的爆炸后模型上。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [Berestycki and Sousi, Applied Probability, Theorem 1.22, PDF p.14; §1.6 pp.12–14](https://www.statslab.cam.ac.uk/~ps422/notes-new2015.pdf#page=14): strong Markov restart for a CTMC before explosion, generator construction and lifetime boundary.
- [Durrett, Probability: Theory and Examples, §7.3, PDF pp.375–377 / printed pp.367–369](https://sites.math.duke.edu/~rtd/PTE/PTE5_011119.pdf#page=375): stopped information and the grid-approximation proof for continuous-path processes; above, right continuity and a discrete state space supply the needed limiting step for a nonexplosive jump chain.
- [MIT 6.436J, Lecture 24, PDF pp.5–7](https://ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/087af3cedbc9def5b156c5e1665ac79c_MIT6_436JF18_lec24.pdf#page=5): ordinary fixed-time Markov identity, holding-time proof and its path-event qualification. The fine-grid argument above explains the extension; the device and last-jump examples are direct applications and boundary checks.
<!-- bilingual-en:end -->

- Berestycki–Sousi 定理1.22及第1.6节（PDF12–14页）：CTMC 爆炸前的强 Markov 重新开始结论、生成构造与存在时间边界。
- Durrett 第7.3节（PDF375–377页／印刷367–369页）：停时信息和连续路径过程的网格逼近证明；正文中非爆炸跳链所需的极限步骤由右连续性与离散状态空间保证。
- MIT 6.436J 第24讲（PDF5–7页）：固定时刻的普通 Markov 等式、停留时间证明及整段路径事件的条件；正文网格逼近解释推广思路，设备与最后跳时例直接检查应用和边界。

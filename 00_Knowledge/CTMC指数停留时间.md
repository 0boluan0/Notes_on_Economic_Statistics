---
aliases:
- Exponential holding time
- CTMC sojourn time
- CTMC dwell time
- 指数停留时间
student_os: knowledge-atom
atom_id: PROB-CTMC-002
atom_set: continuous-time-markov-chains
atom_type: theorem
status: source-checked
requires:
- '[[CTMC定义]]'
- '[[CTMC时间齐次转移函数]]'
- '[[指数分布]]'
- '[[指数分布无记忆性]]'
related:
- '[[Markov充分状态]]'
- '[[状态扩充]]'
- '[[普通更新模型边界]]'
leads_to:
- '[[出口率]]'
- '[[生成矩阵分解]]'
- '[[有界率非爆炸]]'
part_of:
- '[[连续时间马尔可夫链.canvas]]'
---

<!-- bilingual-en:start -->
# Exponential holding times in a CTMC
<!-- bilingual-en:end -->

*CTMC 的指数停留时间*

<!-- bilingual-en:start -->
A machine is currently working. A **holding time** is how long it remains in that state before its next actual change, such as a failure. In a standard [[CTMC定义|continuous-time Markov chain]], the state is constant between jumps. If the model is also [[CTMC时间齐次转移函数|time homogeneous]], its transition rules do not change with the calendar time. These assumptions make a nonabsorbing state's holding time exponential: after surviving in that state for a while, the remaining wait has the same distribution as a fresh wait.
<!-- bilingual-en:end -->

一台机器目前正常工作。**停留时间**指它在这个状态中持续多久，才第一次真正改变状态，例如发生故障。标准 [[CTMC定义|连续时间 Markov 链]]在两次跳跃之间保持状态不变；再加上[[CTMC时间齐次转移函数|时间齐次]]，转移规则就不随日历时点改变。这些条件使非吸收状态的停留时间服从指数分布：已经在该状态中等了一段时间后，剩余等待的分布与刚开始等待时相同。

<!-- bilingual-en:start -->
## What the rate tells us

Let $i$ denote the current state. Its [[出口率|exit rate]] is $q_i$, measured in inverse time units. In the row convention for a [[生成矩阵约束|generator]] $Q=(q_{ij})$, the off-diagonal entry $q_{ij}$ is the rate of a direct jump from $i$ to $j$, and
<!-- bilingual-en:end -->

*速率告诉我们什么*

用 $i$ 表示当前状态，$q_i$ 是该状态的[[出口率]]，单位是时间单位的倒数。采用[[生成矩阵约束|生成矩阵]] $Q=(q_{ij})$ 的行约定，非对角元 $q_{ij}$ 表示从 $i$ 直接跳到 $j$ 的速率，且

$$
q_i=\sum_{j\ne i}q_{ij}=-q_{ii}.
$$

<!-- bilingual-en:start -->
Work with a standard right-continuous, time-homogeneous pure-jump chain on a finite or countable state space, with $0\le q_i<\infty$ at every state and zero generator row sums. Write $X_t$ for the state at time $t$. Start in $i$ at time $0$, and write $H_i=\inf\{t>0:X_t\ne i\}$ for the first departure time; the infimum of the empty set is $\infty$. If $q_i>0$, then
<!-- bilingual-en:end -->

这里讨论有限或可数状态空间上的标准右连续、时间齐次纯跳链：每个状态满足 $0\le q_i<\infty$，生成矩阵各行和为零。用 $X_t$ 表示时刻 $t$ 的状态。从时刻 $0$ 的状态 $i$ 出发，令 $H_i=\inf\{t>0:X_t\ne i\}$ 为首次离开时间；若从不离开，空集合的下确界约定为 $\infty$。当 $q_i>0$ 时，

$$
H_i\sim\operatorname{Exp}(q_i),\qquad
\Pr_i(H_i>t)=e^{-q_it}\quad(t\ge0),\qquad
\mathbb E_iH_i=\frac1{q_i}.
$$

<!-- bilingual-en:start -->
The subscript $i$ means that the chain starts in state $i$. The notation $\operatorname{Exp}(q_i)$ uses the **rate** parameter of the [[指数分布|exponential distribution]], not its mean. A larger rate means a shorter average wait.
<!-- bilingual-en:end -->

下标 $i$ 表示从状态 $i$ 出发。$\operatorname{Exp}(q_i)$ 采用[[指数分布]]的**速率参数**，不是均值参数；速率越大，平均等待越短。

> [!example] Five departures per unit of exposure / 单位暴露时间的离开率为五
> <!-- bilingual-en:start -->
> Suppose the rates from $i$ to two other states are $2$ and $3$ per hour. Then $q_i=5$ per hour, so the mean wait is $1/5$ hour, or $12$ minutes. The probability of no departure during the next $12$ minutes is $e^{-1}\approx0.368$. This does not mean a departure occurs every $12$ minutes: the wait is random.
> <!-- bilingual-en:end -->
>
> 若从 $i$ 去往另外两个状态的速率分别是每小时 $2$ 和 $3$，则 $q_i=5$ 每小时，平均等待为 $1/5$ 小时，即 $12$ 分钟。未来 $12$ 分钟内一直不离开的概率为 $e^{-1}\approx0.368$。这不表示每隔 $12$ 分钟必定离开一次，等待时间本身是随机的。

<!-- bilingual-en:start -->
## Why the clock has no age

Put $G_i(t)=\Pr_i(H_i>t)$. On the event $H_i>s$, the chain has remained in $i$ throughout the first $s$ time units. The Markov property says its subsequent evolution depends on its current state rather than that past history; time homogeneity removes dependence on the starting clock time. Thus, whenever the conditional probability is defined,
<!-- bilingual-en:end -->

*为什么这个时钟没有年龄*

记 $G_i(t)=\Pr_i(H_i>t)$。在事件 $H_i>s$ 上，链在前 $s$ 个时间单位里一直停留在 $i$。Markov 性使后续演化只依赖当前状态，不再依赖这段历史；时间齐次再去掉对开始时点的依赖。因此，在条件概率有定义时，

$$
\Pr_i(H_i>s+t\mid H_i>s)=\Pr_i(H_i>t),
\qquad G_i(s+t)=G_i(s)G_i(t).
$$

<!-- bilingual-en:start -->
This is [[指数分布无记忆性|the memoryless property]]. A survival function that is nonincreasing, right-continuous at $0$, and satisfies $G_i(0)=1$ and this multiplicative rule has the form $e^{-q_it}$; the exponential-characterisation card explains the rational-time argument and its extension to real times. Positive finite exit rate gives a finite exponential wait. Zero rate instead gives $G_i(t)=1$ for every finite $t$.

The event here is **no departure during an interval**. It differs from $X_t=i$: the chain could leave $i$ and return before $t$. In general the transition probability $\Pr_i(X_t=i)$ is not $e^{-q_it}$.
<!-- bilingual-en:end -->

这就是[[指数分布无记忆性|无记忆性]]。一个满足 $G_i(0)=1$、在 $0$ 处右连续、单调不增且具有上述乘法规则的生存函数必为 $e^{-q_it}$；指数分布的表征卡解释了有理时刻的论证及向实数时刻的延伸。正的有限出口率给出有限的指数等待；零出口率则使每个有限 $t$ 上都有 $G_i(t)=1$。

这里的事件是**整段时间都没有离开**，与 $X_t=i$ 不同：链可能先离开再在 $t$ 之前返回。因此一般不能把转移概率 $\Pr_i(X_t=i)$ 写成 $e^{-q_it}$。

<!-- bilingual-en:start -->
## Starting the clock at an entry time

At a fixed observation time, conditional on the current state being $i$, the wait until the next departure has the same exponential law. At a random entry time, an extra theorem is being used: the **[[CTMC强Markov性|strong Markov property]]** says the process restarts from the entered state at a [[停时定义|stopping time]], a time whose occurrence can be recognized from the path already observed. Genuine jump times have this property. Conditional on an entry into $i$ before any explosion, the subsequent holding time therefore has the same law as $H_i$, independently of the earlier path given that state. Ordinary Markov statements made only at fixed times should not be silently substituted for this stopping-time result.
<!-- bilingual-en:end -->

在固定观察时刻，给定当前状态为 $i$，到下一次离开的剩余等待仍具有相同的指数分布。在随机进入时刻则还用到了**[[CTMC强Markov性|强 Markov 性]]**：在[[停时定义|停时]]——只看已经观察到的路径就能判断是否发生的时刻——过程可以从到达的状态重新开始。真实跳时属于这种时刻。因此，给定在爆炸之前进入 $i$，后续停留与 $H_i$ 同分布，并且在给定该状态后独立于此前路径。只针对固定时刻陈述的普通 Markov 性，不能不加说明地替代这个停时结论。

> [!warning] Absorption and the model boundary / 吸收与模型边界
> <!-- bilingual-en:start -->
> If $q_i=0$, every outgoing rate is zero: state $i$ is absorbing and $H_i=\infty$ almost surely. This is not an ordinary $\operatorname{Exp}(0)$ distribution. Pointwise finite rates also do not by themselves exclude [[CTMC爆炸|explosion]], meaning infinitely many jumps in finite time. Statements about a countable chain apply before explosion; finite state spaces with finite rates are nonexplosive.
> <!-- bilingual-en:end -->
>
> 若 $q_i=0$，所有离开方向的速率都为零：$i$ 是吸收态，$H_i=\infty$ 几乎必然成立。这不是普通的 $\operatorname{Exp}(0)$ 分布。每个状态的速率分别有限，也不自动排除[[CTMC爆炸|爆炸]]，即有限时间内发生无限次跳跃；可数链的上述结论用于爆炸之前，有限状态且速率有限则不会爆炸。

<!-- bilingual-en:start -->
For example, a repair duration with survival function $\Pr(H>t)=e^{-t^2}$ for $t\ge0$ (a Weibull law with shape $2$, in the chosen time units) satisfies $\Pr(H>s+t\mid H>s)=e^{-2st-t^2}$. The remaining-wait law depends on elapsed age $s$, so the label “under repair” omits relevant information. It cannot by itself be the repair state of this homogeneous CTMC. An appropriate [[状态扩充|state augmentation]] may include elapsed repair time, producing a Markov model with a continuous age coordinate; it is not automatically a countable-state CTMC. A suitable phase model or a semi-Markov model, which allows more general holding times, may be another modelling choice. Merely observing changes between labels does not establish the exponential-wait assumption.
<!-- bilingual-en:end -->

例如，维修时长的生存函数为 $\Pr(H>t)=e^{-t^2}$（$t\ge0$，在选定时间单位下为形状参数 $2$ 的 Weibull 分布）时，$\Pr(H>s+t\mid H>s)=e^{-2st-t^2}$。剩余等待的分布取决于已经过去的年龄 $s$，所以“维修中”这个标签遗漏了信息，不能单独作为上述齐次 CTMC 的维修状态。合适的[[状态扩充]]可以加入已经维修多久，得到含连续年龄坐标的 Markov 模型，但它不自动成为可数状态 CTMC。也可考虑合适的分阶段模型，或允许一般停留分布的 semi-Markov 模型。仅仅观察到状态标签之间有变化，并不能确立指数等待假设。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [MIT 6.436J, Lecture 24, pp. 5–7](https://ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/087af3cedbc9def5b156c5e1665ac79c_MIT6_436JF18_lec24.pdf#page=5): time homogeneity, Proposition 1 on holding times, and the path-event subtlety in its proof.
- [Berestycki and Sousi, Applied Probability, §§1.2–1.3 and 1.6, pp. 4–6 and 12–14](https://www.statslab.cam.ac.uk/~ps422/notes-new2015.pdf#page=4): right-continuous jump paths, memorylessness, finite exit rates, absorption, nonexplosion criteria and Theorem 1.22 on the strong Markov property.
- [NIST, Weibull distribution, Survival Function](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3668.htm): the shape-2 survival function used for the age-dependent repair counterexample; its conditional survival is calculated directly.
- [Whitt, Continuous-Time Markov Chains, §§3.1 and 3.3, PDF pp. 6–7 and 12–14](https://www.columbia.edu/~ww2040/4106S11/CTMCchapter121906.pdf#page=6): exponential clocks, the state-information requirement, and the distinction from more general holding-time models. The numerical example is calculated from the displayed survival function.
<!-- bilingual-en:end -->

- MIT 6.436J 第 24 讲第 5–7 页：时间齐次、停留时间命题及证明中整段路径事件的技术条件。
- Berestycki–Sousi 讲义第 4–6、12–14 页：右连续跳路径、无记忆性、有限出口率、吸收与非爆炸条件，以及强 Markov 性定理 1.22。
- NIST 的 Weibull 分布页面核对形状参数 2 的生存函数；维修反例的条件生存概率直接由比值计算。
- Whitt §3.1、§3.3，PDF 第 6–7、12–14 页：指数时钟、状态信息及一般停留模型的区别。数值例直接由生存函数计算。

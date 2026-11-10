---
aliases:
- Continuous-time Markov chain
- CTMC
- 连续时间 Markov 链
- 连续时间马尔可夫链
student_os: knowledge-atom
atom_id: PROB-CTMC-001
atom_set: continuous-time-markov-chains
atom_type: definition
status: source-checked
requires:
- '[[Markov性]]'
- '[[随机过程]]'
related:
- '[[Markov充分状态]]'
- '[[状态扩充]]'
- '[[CTMC时间齐次转移函数]]'
leads_to:
- '[[CTMC指数停留时间]]'
- '[[生成矩阵约束]]'
- '[[CTMC转移半群]]'
part_of:
- '[[连续时间马尔可夫链.canvas]]'
---

<!-- bilingual-en:start -->
# Continuous-time Markov chain
<!-- bilingual-en:end -->
*连续时间 Markov 链（CTMC）*

<!-- bilingual-en:start -->
A **continuous-time Markov chain** models a system that stays in one of a discrete collection of states, then changes state at a possibly random time. Think of a device that is working, then fails, then is repaired. We can ask for its state at 10:00, 10:00:00.1 or any other time. The clock is continuous; the state still takes only the labels “working” and “failed”. The Markov assumption says that the current state, at the current clock time, contains the information from the device's history needed to predict its future.
<!-- bilingual-en:end -->

**连续时间 Markov 链**描述这样一个系统：它在某个离散状态停留一段时间，再于可能随机的时刻改变状态。例如设备先正常运行，随后故障，再被修好。可以询问它在10:00、10:00:00.1或其他任意时刻的状态：时钟连续，状态仍然只有“正常”“故障”这些标签。Markov 假设则要求：在当前日历时点，当前状态已经保留了设备历史中预测未来所需要的信息。

<!-- bilingual-en:start -->
A [[随机过程|stochastic process]] is a family of random variables on one common probability space. Write $X=(X_t)_{t\ge0}$, where $t$ is any nonnegative real time and $X_t$ is the state then. Its state space $S$ is finite or countably infinite: for example, $S=\{W,F\}$ for working/failed, or $S=\{0,1,2,\ldots\}$ for a queue length. The numbers used as state labels need not measure physical distances.
<!-- bilingual-en:end -->

[[随机过程]]是在同一概率空间上的一族随机变量。写成 $X=(X_t)_{t\ge0}$：$t$ 可以是任意非负实数时间，$X_t$ 是那个时刻的状态。状态空间 $S$ 有限或可数无穷，例如设备的 $S=\{W,F\}$，或排队人数的 $S=\{0,1,2,\ldots\}$。用来标记状态的数字，不一定表示物理距离。

> [!example] Read one path before the probability law · 先读一条路径
> <!-- bilingual-en:start -->
> One possible record is $W$ until hour $0.7$, $F$ from $0.7$ until $1.9$, then $W$ again. At the failure time itself we record the new state: $X_{0.7}=F$. This is one [[样本路径|sample path]], not a schedule that every run must follow. The model assigns probabilities to possible records. Nothing requires jumps to occur only at integer hours.
> <!-- bilingual-en:end -->
>
> 一条可能的记录是：第0.7小时前为 $W$，从0.7到1.9小时为 $F$，然后恢复为 $W$。在故障发生的那个时刻就记录新状态，即 $X_{0.7}=F$。这是一个[[样本路径|样本路径]]，不是每次运行都必须照做的时间表；模型描述各种可能记录的概率。跳跃不必只发生在整数小时。

<!-- bilingual-en:start -->
## What “Markov” adds
<!-- bilingual-en:end -->
*Markov 性增加了什么条件*

<!-- bilingual-en:start -->
The [[Markov性|Markov property]] compares forecasts made at the same time. Let $\mathcal F_t^X$ denote all information obtained by observing this process up to $t$, its [[自然滤过|natural filtration]]. Then for every later time $t+h$, $h\ge0$, and state $j$,
<!-- bilingual-en:end -->

[[Markov性]]比较的是同一时点作出的预测。用 $\mathcal F_t^X$ 表示观察这个过程直到 $t$ 所得到的全部历史信息，即它的[[自然滤过]]。对每个后续时点 $t+h$、$h\ge0$，以及每个状态 $j$，要求

$$\Pr(X_{t+h}=j\mid\mathcal F_t^X)
=\Pr(X_{t+h}=j\mid X_t)\qquad\text{almost surely}.$$

<!-- bilingual-en:start -->
The left side uses the whole history; the right side uses the current state. The equality is an equality of conditional probabilities as random variables, holding with probability one. With finitely many observed past times and a positive-probability history, it says that deleting the earlier state observations does not change the forecast. A zero-probability history cannot be handled by dividing two event probabilities. The condition concerns a **distribution** of possible futures, not a deterministic future path.
<!-- bilingual-en:end -->

左边使用完整历史，右边只使用当前状态。这是作为随机变量的条件概率之间的等式，以概率一成立。若只写有限个过去观测时点，并且该历史具有正概率，它就表示：删去更早的状态观测，不改变预测。零概率历史不能靠两个事件概率相除来处理。条件约束的是可能未来的**概率分布**，并不是说未来路径已经确定。

<!-- bilingual-en:start -->
The state choice matters. If every repair lasts exactly one hour, knowing only “under repair” may be insufficient: a repair that has already lasted $0.8$ hours will finish within the next $0.3$ hours, while one that has lasted $0.1$ hours will not. If both histories can occur at the same observation time, the current label has discarded predictive information. One must reconsider the [[Markov充分状态|state description]], for example by recording repair age; a continuously varying age would also change the discrete-state scope of this card.
<!-- bilingual-en:end -->

状态怎样选择很重要。若每次维修恰好持续一小时，只知道“正在维修”可能不够：已经修了0.8小时的设备会在接下来0.3小时内修好，才修了0.1小时的却不会。若同一观测时点可能出现这两种历史，当前标签就遗漏了预测信息。此时要重新考虑[[Markov充分状态|状态描述]]，例如记录维修年龄；连续变化的年龄也会超出本卡的离散状态范围。

<!-- bilingual-en:start -->
## Jump paths and their lifetime
<!-- bilingual-en:end -->
*跳跃路径及其存在时间*

<!-- bilingual-en:start -->
Here we use the standard pure-jump path convention: up to a possible explosion, paths are piecewise constant, take their new value at a jump, and are right-continuous with left limits. “Right-continuous” means that immediately after a time, the path agrees with its value at that time; the left limit records the state just before a jump. A nonexplosive path has only finitely many genuine state changes on each bounded time interval. An absorbing state can be occupied forever.
<!-- bilingual-en:end -->

这里采用标准纯跳路径约定：在可能的爆炸之前，路径分段常值，在跳时取跳后的新值，并且右连续、有左极限。“右连续”表示某时刻刚刚之后的状态与该时刻的取值一致；左极限记录跳跃前的状态。非爆炸路径在每个有界时间区间内只发生有限次真正的状态改变；也允许进入吸收状态后永远不再离开。

<!-- bilingual-en:start -->
On a countably infinite state space, infinitely many jumps can accumulate before a finite time. This is [[CTMC爆炸|explosion]]. If $T_n$ is the time of the $n$th genuine jump, the accumulation time is $\zeta=\lim_n T_n$; set missing jump times to $\infty$. A [[最小CTMC|minimal process]] assigns a new cemetery label $\Delta\notin S$ at and after $\zeta$ and never returns to $S$. The cemetery label records that the original construction has ended. It does not assert a left limit at explosion within the original discrete space. If one wants a different continuation, that is an additional [[爆炸后延拓|post-explosion rule]].
<!-- bilingual-en:end -->

可数无穷状态空间上，可能在有限时刻之前完成无限次跳跃，这叫[[CTMC爆炸|爆炸]]。若 $T_n$ 是第 $n$ 次真正跳跃的时刻，其聚积时刻是 $\zeta=\lim_n T_n$；不存在的跳时记为 $\infty$。[[最小CTMC|最小过程]]在 $\zeta$ 及以后取一个新的墓地标签 $\Delta\notin S$，不再回到 $S$。这个标签表示原来的构造已经结束，并没有声称爆炸时在原离散空间中仍存在左极限。若要采用其他继续方式，就需要额外规定[[爆炸后延拓]]。

> [!warning] Four separate choices · 四个不同条件
> <!-- bilingual-en:start -->
> Continuous time, discrete states, jump paths and the Markov property are separate specifications. [[标准布朗运动|Brownian motion]] has continuous time and is Markov, but its states range over the real line and its paths are continuous, so it is not this kind of chain. A jump path alone does not establish the Markov property either.
> <!-- bilingual-en:end -->
>
> 连续时间、离散状态、跳跃路径与 Markov 性是不同的规定。[[标准布朗运动|布朗运动]]在连续时间上运行且具有 Markov 性，但状态取遍实数、路径连续，因此不属于这里的链。单凭一条跳跃路径，同样不能证明 Markov 性。

<!-- bilingual-en:start -->
Finally, [[CTMC时间齐次转移函数|time homogeneity]] is an extra restriction: the transition law over an interval depends only on its length, not its starting clock time. A device can be Markov while the failure rule changes between day and night. Without homogeneity, the current time can still affect the forecast; the Markov property removes dependence on earlier states, not on the known clock. Exponential holding times under suitable homogeneous assumptions are treated in [[CTMC指数停留时间|the holding-time theorem]].
<!-- bilingual-en:end -->

最后，[[CTMC时间齐次转移函数|时间齐次]]是附加限制：一段时间内的转移规律只依赖这段时间的长度，不依赖起始日历时点。设备的白天和夜间故障规则可以不同，同时仍满足 Markov 性。不齐次时，当前时间仍可影响预测；Markov 性排除的是对更早状态的额外依赖，不是对已知时钟的依赖。适当齐次条件下的指数停留时间见[[CTMC指数停留时间|停留时间定理]]。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [MIT 6.436J, Lecture 24, Definition 2, PDF p.5](https://ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/087af3cedbc9def5b156c5e1665ac79c_MIT6_436JF18_lec24.pdf#page=5): finite/countable states, arbitrary real observation times, the Markov identity and the separate homogeneity condition.
- [Berestycki and Sousi, Cambridge Applied Probability (2015), §§1.1–1.2, PDF pp.3–5; §§1.6–1.8, pp.12–19](https://www.statslab.cam.ac.uk/~ps422/notes-new2015.pdf#page=3): right-continuous jump paths, jump/holding times, explosion, the minimal construction and the distinction from a post-explosion extension. The device records illustrate these definitions; the repair-age comparison is calculated directly.
- [[01_Math/05_随机过程/02_随机过程的概念和分类.docx|Local course terminology]]: used only to check the course's Markov and “memorylessness” terminology.
<!-- bilingual-en:end -->

- MIT 第5页定义2核对有限/可数状态、实数观测时间、Markov 等式与单独的齐次条件。
- Cambridge 第1.1–1.2节（PDF第3–5页）、第1.6–1.8节（第12–19页）核对右连续跳跃路径、跳时与停留时间、爆炸、最小构造及爆炸后延拓。设备记录用于解释定义，维修年龄比较直接计算。
- 本地课程文件仅核对 Markov 与“无后效性”术语。

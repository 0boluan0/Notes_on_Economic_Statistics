---
aliases:
  - CTMC stationary distribution
  - CTMC invariant distribution
  - 连续时间链平稳分布
student_os: knowledge-atom
atom_id: PROB-CTMC-014
atom_set: continuous-time-markov-chains
atom_type: definition
status: source-checked
requires:
  - "[[CTMC转移半群]]"
  - "[[Markov矩阵左右约定]]"
related:
  - "[[Markov稳态分布]]"
leads_to:
  - "[[有限CTMC平稳生成矩阵判据]]"
  - "[[有限不可约CTMC稳态]]"
  - "[[有限CTMC平稳分解]]"
  - "[[可数CTMC稳态存在]]"
  - "[[CTMC详细平衡]]"
part_of:
  - "[[连续时间马尔可夫链.canvas]]"
---

<!-- bilingual-en:start -->
# Stationary distributions of a CTMC
<!-- bilingual-en:end -->

*连续时间 Markov 链的平稳分布*

<!-- bilingual-en:start -->
Imagine many identical machines, each switching randomly between working and being repaired. Individual machines keep changing state, but the fraction expected to be working can stay constant. A **stationary distribution** describes just such an unchanged probability allocation across states. It is a distribution from which to start the process, not a command for each path to remain still.
<!-- bilingual-en:end -->

设想许多相同的机器，各自在工作与维修之间随机切换。每台机器都在变化，但预期处于工作状态的比例可以保持不变。**平稳分布**描述的就是这种不变的状态概率分配。它规定怎样随机选择过程的初始状态，不要求每条路径原地不动。

<!-- bilingual-en:start -->
## An unchanged distribution at every time
<!-- bilingual-en:end -->

## 直觉

<!-- bilingual-en:start -->
Take a [[CTMC定义|continuous-time Markov chain]] $X=(X_t)_{t\ge0}$ on a finite or countable state set $S$. Here $X_t$ is the state at time $t$. Assume the chain is [[CTMC时间齐次转移函数|time-homogeneous]] and remains in $S$ at every finite time. Its [[CTMC转移半群|transition semigroup]] $P(t)$ has entry $p_{ij}(t)=\Pr(X_t=j\mid X_0=i)$: the probability of ending at $j$ after duration $t$ when starting at $i$. Each row sums to one. In the [[Markov矩阵左右约定|row-vector convention]], an initial probability row $\mu=(\mu_i)_{i\in S}$ becomes $\mu P(t)$.
<!-- bilingual-en:end -->

考虑有限或可数状态集 $S$ 上的[[CTMC定义|连续时间 Markov 链]] $X=(X_t)_{t\ge0}$，其中 $X_t$ 表示时刻 $t$ 的状态。假设链[[CTMC时间齐次转移函数|时间齐次]]，并且每个有限时刻仍位于 $S$。其[[CTMC转移半群]] $P(t)$ 的元素 $p_{ij}(t)=\Pr(X_t=j\mid X_0=i)$，表示从 $i$ 出发、经过时长 $t$ 后处于 $j$ 的概率；每行和为一。按[[Markov矩阵左右约定|行向量约定]]，初始概率行 $\mu=(\mu_i)_{i\in S}$ 经过时长 $t$ 后变成 $\mu P(t)$。

> [!important] Invariant under every elapsed duration · 任意时长后都不变
> <!-- bilingual-en:start -->
> A probability row $\pi=(\pi_i)_{i\in S}$ is **stationary**, or **invariant**, if
> <!-- bilingual-en:end -->
>
> $$\pi_i\ge0,\qquad\sum_{i\in S}\pi_i=1,\qquad \pi P(t)=\pi\quad\text{for every }t\ge0.$$
>
> <!-- bilingual-en:start -->
> Coordinate $j$ of the last equality is $\sum_i\pi_i p_{ij}(t)=\pi_j$. The left side adds all ways to start somewhere and end at $j$. Thus $X_0\sim\pi$ implies $X_t\sim\pi$ at each time; the symbol $\sim$ here means “has distribution.”
> <!-- bilingual-en:end -->
>
> 概率行 $\pi=(\pi_i)_{i\in S}$ 称为**平稳分布**或**不变分布**，当它各项非负、总和为一，并对每个 $t\ge0$ 满足 $\pi P(t)=\pi$。最后一个等式的第 $j$ 项是 $\sum_i\pi_i p_{ij}(t)=\pi_j$：把从各个可能初态到达 $j$ 的概率加起来，仍为原来的质量。因此 $X_0\sim\pi$ 推出每个时刻都有 $X_t\sim\pi$；这里 $\sim$ 表示“服从该分布”。

<!-- bilingual-en:start -->
The quantifier “every $t$” is part of the definition. It asks for invariance under the whole family $P(t)$, rather than under the generator itself. The discrete-time analogue is [[Markov稳态分布|a probability row fixed by a one-step transition matrix]].
<!-- bilingual-en:end -->

“每个 $t$”是定义的一部分：要求在整族 $P(t)$ 的作用下不变，而不是把生成矩阵直接当转移概率使用。离散时间的对应概念是[[Markov稳态分布|被一步转移矩阵保持不变的概率行]]。

> [!example] Machines still switch · 机器仍在切换
> <!-- bilingual-en:start -->
> Let state $0$ mean working and state $1$ mean under repair. Suppose the failure rate is $2$ per hour and the repair-completion rate is $1$ per hour. The [[生成矩阵约束|generator]], which records rates, and its [[CTMC矩阵指数|transition matrix]] are
> <!-- bilingual-en:end -->
>
> $$Q=\begin{pmatrix}-2&2\\1&-1\end{pmatrix},\qquad
> P(t)=\begin{pmatrix}
> \frac13+\frac23e^{-3t}&\frac23(1-e^{-3t})\\
> \frac13(1-e^{-3t})&\frac23+\frac13e^{-3t}
> \end{pmatrix},$$
>
> <!-- bilingual-en:start -->
> with $t$ measured in hours. Multiply $\pi=(1/3,2/3)$ by $P(t)$: the exponential terms cancel and leave $(1/3,2/3)$ for every $t$. Yet a working machine has probability $e^{-2t}$ of making no departure during the next $t$ hours, which is less than one for $t>0$. Stationarity holds for the distribution while paths continue to move.
> <!-- bilingual-en:end -->
>
> 状态 $0$ 表示工作、状态 $1$ 表示维修。故障率为每小时 $2$，维修完成率为每小时 $1$；上式给出记录速率的[[生成矩阵约束|生成矩阵]]与相应[[CTMC矩阵指数|转移矩阵]]，其中 $t$ 以小时计。将 $\pi=(1/3,2/3)$ 左乘 $P(t)$，指数项抵消，任意 $t$ 都留下 $(1/3,2/3)$。但工作中的机器在随后 $t$ 小时内一次也不离开该状态的概率是 $e^{-2t}$；$t>0$ 时它小于一。分布可以平稳，路径仍然变化。

<!-- bilingual-en:start -->
## Existence, uniqueness and convergence are different questions
<!-- bilingual-en:end -->

## 边界

<!-- bilingual-en:start -->
An [[吸收状态|absorbing state]] $a$ cannot be left. Its point mass $\delta_a$, meaning probability one at $a$ and zero elsewhere, is stationary: $\delta_aP(t)=\delta_a$. If two states are both absorbing and there are no others, $P(t)=I$, the [[单位矩阵|identity matrix]], and every probability row $(c,1-c)$, $0\le c\le1$, is stationary. There is no single stationary law selected independently of the starting distribution in that example.
<!-- bilingual-en:end -->

[[吸收状态]] $a$ 无法被离开，因此点质量 $\delta_a$——在 $a$ 上概率为一、其他状态上为零——满足 $\delta_aP(t)=\delta_a$，是平稳分布。若仅有两个状态且它们都吸收，则 $P(t)=I$（[[单位矩阵]]），每个 $0\le c\le1$ 对应的概率行 $(c,1-c)$ 都平稳。在这个例子中，没有一个与初始分布无关、被全链唯一选出的平稳分布。

<!-- bilingual-en:start -->
The definition therefore does not settle whether a stationary law exists, whether it is unique, or whether other initial laws approach it. For finite chains, [[有限CTMC平稳生成矩阵判据|the generator criterion]] computes stationary candidates and [[有限不可约CTMC稳态|finite irreducibility]] guarantees a unique positive law; [[有限CTMC全局收敛|global convergence]] is a separate theorem. A claim about a single path's long-run fraction of time also needs its own ergodic assumptions, rather than just the equation defining stationarity.
<!-- bilingual-en:end -->

所以，定义本身没有解决平稳分布是否存在、是否唯一、其他初始分布是否趋近它。有限链可用[[有限CTMC平稳生成矩阵判据|生成矩阵判据]]计算平稳候选；[[有限不可约CTMC稳态|有限不可约性]]保证唯一正分布；[[有限CTMC全局收敛|全局收敛]]是另一条定理。把某个概率解释为单条路径的长期时间比例，也需要相应的遍历条件，不能只凭平稳性定义式。

<!-- bilingual-en:start -->
For a countably infinite state space, specify which transition semigroup is being discussed. A [[最小CTMC|minimal chain]] may lose probability mass from $S$ after [[CTMC爆炸|explosion]], and an extension must specify what happens then. A formal solution of $\pi Q=0$ does not replace checking the actual semigroup and its probability conservation; see [[可数CTMC稳态存在|stationary existence in the countable case]].
<!-- bilingual-en:end -->

对可数无限状态空间，必须指定讨论的是哪个转移半群。[[最小CTMC]]可能在[[CTMC爆炸|爆炸]]后从 $S$ 中流失概率质量，延拓则要说明之后发生什么。形式上的 $\pi Q=0$ 解不能替代对实际半群及其概率守恒的核查，见[[可数CTMC稳态存在]]。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [Whitt, Continuous-Time Markov Chains, §4 introduction, PDF p.17 / printed p.16](https://www.columbia.edu/~ww2040/4106S11/CTMCchapter121906.pdf#page=17): time-invariant probability definition and distinction from a limiting law. The two-state transition matrix and absorbing examples above are checked directly.
- [MIT 6.436J, Lecture 24, PDF p.7](https://ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/087af3cedbc9def5b156c5e1665ac79c_MIT6_436JF18_lec24.pdf#page=7): stationary probabilities and finite-state context. [Berestycki–Sousi, Applied Probability, §2.4, PDF pp.23–27](https://www.statslab.cam.ac.uk/~ps422/notes-new2015.pdf#page=23): generator-invariant measures and the extra issues in the countable/explosive case; this card defines stationarity by the transition semigroup.
- [[01_Math/05_随机过程/05_连续时间的马尔可夫链#5. 平稳分布、唯一性与收敛|Permanent course notes, §5]]: scope and row-vector notation.
<!-- bilingual-en:end -->

- Whitt §4 开头（PDF17／印刷16页）：随时间不变的概率分布定义，以及与极限分布的区别；两状态转移矩阵和吸收例直接核算。
- MIT6.436J 第24讲 PDF7页：平稳概率与有限状态语境。Berestycki–Sousi §2.4（PDF23–27页）：生成矩阵不变测度及可数／爆炸情形的额外问题；本卡以转移半群定义平稳性。
- 永久课程笔记第5节保留课程范围与行向量记号。

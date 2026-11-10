---
aliases:
  - "CTMC complete-path MLE"
  - "Complete-path generator rate estimation"
  - "Complete-path transition counts and exposure times"
  - "CTMC 完整路径生成率估计"
student_os: knowledge-atom
atom_id: PROB-CTMC-024
atom_set: continuous-time-markov-chains
atom_type: method
status: source-checked
requires:
  - "[[生成矩阵约束]]"
  - "[[生成矩阵分解]]"
  - "[[出口率]]"
related:
  - "[[嵌入跳链]]"
  - "[[Markov充分状态]]"
leads_to:
  - "[[CTMC共同率估计]]"
part_of:
  - "[[连续时间马尔可夫链.canvas]]"
---

<!-- bilingual-en:start -->
# Estimating CTMC rates from a complete path
<!-- bilingual-en:end -->

*从完整路径估计 CTMC 生成率*

<!-- bilingual-en:start -->
A jump count tells us how often a transition happened. To estimate **how fast** it happens, we also need the time during which it could have happened. Complete-path estimation uses both: every genuine jump, and the total time spent in each starting state. A state observed for ten hours without a departure supplies information even though its jump count is zero.
<!-- bilingual-en:end -->

跳转计数告诉我们某种变化发生了几次；要估计它发生得**有多快**，还要知道有多少时间可能发生这种变化。完整路径估计同时使用每次真实跳跃，以及在各出发状态累计停留的时间。某状态观察了十小时却没有离开，虽然跳数为零，仍然提供信息。

<!-- bilingual-en:start -->
## Read a timeline before dividing
<!-- bilingual-en:end -->

*先读时间线，再做除法*

<!-- bilingual-en:start -->
Observe one system continuously for eight hours. It starts in state $0$, jumps to $1$ at hour $2$, returns to $0$ at hour $3$, and jumps to $2$ at hour $7$. Observation stops at hour $8$, while the system is still in $2$. The recorded path is
<!-- bilingual-en:end -->

连续观察一个系统八小时。它从状态 $0$ 出发，在第 $2$ 小时跳到 $1$，第 $3$ 小时回到 $0$，第 $7$ 小时跳到 $2$。第 $8$ 小时结束观察时，它仍在 $2$。记录的路径为

$$0\ \xrightarrow{\ t=2\ }\ 1\ \xrightarrow{\ t=3\ }\ 0\ \xrightarrow{\ t=7\ }\ 2,\qquad T=8.$$

<!-- bilingual-en:start -->
Let $N_{ij}$ count genuine jumps from $i$ to $j$, and let $S_i$ be the total **exposure time** in state $i$. Repeated visits are added together. Here $N_{01}=N_{10}=N_{02}=1$, all other off-diagonal counts are zero, and
<!-- bilingual-en:end -->

令 $N_{ij}$ 计数从 $i$ 到 $j$ 的真实跳跃，$S_i$ 表示在状态 $i$ 的总**暴露时间**；多次访问的时间需要相加。这里 $N_{01}=N_{10}=N_{02}=1$，其余非对角计数为零，并有

$$S_0=2+(7-3)=6,\qquad S_1=3-2=1,\qquad S_2=8-7=1.$$

<!-- bilingual-en:start -->
For freely varying rates, the estimates are $\widehat q_{01}=\widehat q_{02}=1/6$ per hour and $\widehat q_{10}=1$ per hour. The remaining off-diagonal estimates are zero. The row diagonals are minus the total exit rates:
<!-- bilingual-en:end -->

若各速率可以独立取值，估计为 $\widehat q_{01}=\widehat q_{02}=1/6$ 每小时，$\widehat q_{10}=1$ 每小时；其余非对角估计为零。对角元取该行总出口率的负数：

$$\widehat Q=\begin{pmatrix}-1/3&1/6&1/6\\1&-1&0\\0&0&0\end{pmatrix}.$$

<!-- bilingual-en:start -->
The estimated last row is zero because this model allows zero rates and the record contains positive exposure at $2$ but no departure. It does not establish that state $2$ is truly absorbing. The last hour must not disappear merely because observation ended before the next jump.
<!-- bilingual-en:end -->

最后一行估计为零，是因为模型允许零速率，且记录在 $2$ 中有正暴露时间、没有离开；这并不能证明真实状态 $2$ 是吸收态。不能因为观察在下一次跳跃前结束，就删掉最后这一小时。

<!-- bilingual-en:start -->
## What “complete” observation means
<!-- bilingual-en:end -->

*“完整”观察究竟包含什么*

<!-- bilingual-en:start -->
Use a [[CTMC时间齐次转移函数|time-homogeneous CTMC]] on a finite state set $\mathcal S$, with a fixed [[生成矩阵约束|generator]] $Q=(q_{ij})$. Rates are finite, $q_{ij}\ge0$ for $i\ne j$, and the [[出口率|exit rate]] is $q_i=\sum_{j\ne i}q_{ij}=-q_{ii}$. Fix the observation window $[0,T]$ with $T>0$, and condition on the recorded initial state. Every state change and its time must be observed; the times of artificial self-loops in a simulation are not genuine jumps.
<!-- bilingual-en:end -->

使用有限状态集 $\mathcal S$ 上的[[CTMC时间齐次转移函数|时间齐次 CTMC]]，其固定[[生成矩阵约束|生成矩阵]]为 $Q=(q_{ij})$。各速率有限，$i\ne j$ 时 $q_{ij}\ge0$，[[出口率]]为 $q_i=\sum_{j\ne i}q_{ij}=-q_{ii}$。观察窗固定为 $[0,T]$，其中 $T>0$，并条件于记录的初始状态。必须看到每次状态变化及其时刻；模拟中的人工自环不是真实跳跃。

<!-- bilingual-en:start -->
Suppose there are $m$ jumps at $0<t_1<\cdots<t_m<T$. Let $t_0=0$, $t_{m+1}=T$, and $x_k$ be the state occupied between $t_k$ and $t_{k+1}$. A jump at the predetermined endpoint has probability zero under this model. Then exposure can be calculated directly as
<!-- bilingual-en:end -->

设窗内有 $m$ 次跳跃，时刻为 $0<t_1<\cdots<t_m<T$。令 $t_0=0$、$t_{m+1}=T$，$x_k$ 是 $t_k$ 到 $t_{k+1}$ 之间所处的状态。在这个模型下，恰好在预定终点跳跃的概率为零。暴露时间可直接计算为

$$S_i=\sum_{k=0}^{m}(t_{k+1}-t_k)\,\mathbf1\{x_k=i\},\qquad \sum_iS_i=T.$$

<!-- bilingual-en:start -->
The [[事件指示变量|indicator]] $\mathbf1\{x_k=i\}$ is one when the interval belongs to state $i$, and zero otherwise. This includes the last interval and also works when $m=0$.
<!-- bilingual-en:end -->

[[事件指示变量|指示量]] $\mathbf1\{x_k=i\}$ 在该区间属于状态 $i$ 时取一，否则取零。求和包含最后一个区间，$m=0$ 时也适用。

<!-- bilingual-en:start -->
## The final wait contributes a survival factor
<!-- bilingual-en:end -->

*最后的等待贡献一个生存概率因子*

<!-- bilingual-en:start -->
A [[似然函数|likelihood]] fixes the observed path and compares candidate generators. For a completed wait of length $h$ in $i$ followed by a jump to $j$, [[生成矩阵分解|the holding-time/destination construction]] gives the density $q_{ij}e^{-q_i h}$. If observation ends after $h$ hours in $i$ without a departure, the contribution is instead $e^{-q_i h}$. For $q_i>0$, this is the probability that the [[指数分布|exponential waiting time]] exceeds $h$; for $q_i=0$, the state is absorbing and the factor equals one. The unfinished observation is **right censoring**: we know a lower bound on that final wait, not its completed length.
<!-- bilingual-en:end -->

[[似然函数]]把已观察路径固定，比较不同候选生成矩阵。若在 $i$ 完整等待 $h$ 后跳到 $j$，[[生成矩阵分解|停留时间与去向的构造]]给出密度 $q_{ij}e^{-q_i h}$。若在 $i$ 等待了 $h$ 小时仍没离开，观察便结束，则贡献是 $e^{-q_i h}$。当 $q_i>0$，它是[[指数分布|指数等待时间]]超过 $h$ 的概率；当 $q_i=0$，该状态吸收，因子等于一。对未完成等待的观察叫**右删失**：知道最后一段等待的下界，却不知道它完整的长度。

$$
\begin{aligned}
L(Q)&=\left[\prod_{k=1}^{m}q_{x_{k-1}x_k}e^{-q_{x_{k-1}}(t_k-t_{k-1})}\right]e^{-q_{x_m}(T-t_m)}\\
&=\prod_{i\in\mathcal S}\prod_{j\ne i}q_{ij}^{N_{ij}}e^{-q_{ij}S_i}.
\end{aligned}
$$

<!-- bilingual-en:start -->
This is a [[概率密度函数|density]] in the jump times, not the probability of observing those exact real numbers. Empty products equal one; a factor with $N_{ij}=0$ contributes no power of $q_{ij}$. A positive count on an edge forced to have rate zero has likelihood zero. The terminal exponential factor is why unfinished waits still enter $S_i$.
<!-- bilingual-en:end -->

这是关于跳时的[[概率密度函数|密度]]，不是观察到这些精确实数的点概率。空乘积取一；$N_{ij}=0$ 的因子不贡献 $q_{ij}$ 的幂。若模型强制某条边速率为零，却观察到该边正的跳数，则似然为零。最后的指数因子正是未完成等待仍计入 $S_i$ 的原因。

<!-- bilingual-en:start -->
## Maximization and its boundary cases
<!-- bilingual-en:end -->

*最大化与边界情况*

<!-- bilingual-en:start -->
For separately variable off-diagonal rates, take the [[自然对数|natural logarithm]] to turn products into sums. The term for one edge is $N_{ij}\log q_{ij}-S_iq_{ij}$. When $S_i>0$ and $N_{ij}>0$, its [[导数|derivative]], the local rate of change, is $N_{ij}/q_{ij}-S_i$, changing from positive to negative at $N_{ij}/S_i$. Thus the [[极大似然估计|maximum likelihood estimate]] is
<!-- bilingual-en:end -->

若非对角参数可分别自由变化，取[[自然对数]]把乘积变为求和。每条边的项为 $N_{ij}\log q_{ij}-S_iq_{ij}$。当 $S_i>0$ 且 $N_{ij}>0$ 时，其[[导数]]（局部变化率）为 $N_{ij}/q_{ij}-S_i$，在 $N_{ij}/S_i$ 处从正变负。因此[[极大似然估计]]为

$$\widehat q_{ij}=\frac{N_{ij}}{S_i}\quad(j\ne i),\qquad \widehat q_{ii}=-\sum_{j\ne i}\widehat q_{ij}.$$

<!-- bilingual-en:start -->
If $S_i>0$ but $N_{ij}=0$, the factor $e^{-S_iq_{ij}}$ is maximized at zero when zero is allowed. If the parameter domain requires $q_{ij}>0$, the supremum is approached as $q_{ij}\downarrow0$ and no maximum is attained on that edge. If $S_i=0$, a valid finite-jump record almost surely also has no departures from $i$; this row does not enter the conditional likelihood, so its rates are not identified by that record. Do not define $0/0$ as a zero-rate estimate.
<!-- bilingual-en:end -->

若 $S_i>0$ 但 $N_{ij}=0$，因子 $e^{-S_iq_{ij}}$ 在允许零速率时于零达到最大。若参数域强制 $q_{ij}>0$，则只能在 $q_{ij}\downarrow0$ 时逼近上确界，该边没有取到最大值。若 $S_i=0$，合法有限跳路径几乎必然也没有从 $i$ 离开的记录；这一行不进入条件似然，其速率不能由这份记录识别。不能把 $0/0$ 定义成零速率估计。

> [!example] The denominator carries the time scale / 分母带来时间尺度
> <!-- bilingual-en:start -->
> Ten $i\to j$ jumps over two hours of exposure give $5$ per hour; the same ten jumps over twenty hours give $0.5$ per hour. In contrast, $N_{ij}/\sum_{k\ne i}N_{ik}$, when its denominator is positive, estimates only a [[嵌入跳链|jump destination probability]]. It removes the clock information.
> <!-- bilingual-en:end -->
>
> 十次 $i\to j$ 跳跃若对应两小时暴露，估计为每小时 $5$；若对应二十小时暴露，则为每小时 $0.5$。相比之下，分母为正时，$N_{ij}/\sum_{k\ne i}N_{ik}$ 只估计[[嵌入跳链|跳跃去向概率]]，丢掉了时钟信息。

<!-- bilingual-en:start -->
## Which observation model are we fitting?
<!-- bilingual-en:end -->

*正在拟合哪种观察模型？*

<!-- bilingual-en:start -->
If rates share a parameter or obey other constraints, maximize within that parameterization; see [[CTMC共同率估计|shared-rate estimation]]. Independent paths governed by the same $Q$, each with its own fixed window, multiply their conditional likelihoods, so their counts and exposures add. Neither this factorization nor the MLE definition alone guarantees finite-sample unbiasedness.
<!-- bilingual-en:end -->

若多条边共享参数或满足其他约束，应在该参数化内部最大化，见[[CTMC共同率估计]]。多个独立路径若由同一 $Q$ 控制、各自使用固定窗口，其条件似然相乘，计数与暴露时间因而相加。这种因子分解和 MLE 定义本身，都不保证有限样本无偏。

<!-- bilingual-en:start -->
If we observe only states $z_0,\ldots,z_r$ at fixed times $s_0<\cdots<s_r$, missing jumps prevent us from reconstructing $N_{ij}$ and $S_i$. The conditional likelihood instead multiplies $[P(s_k-s_{k-1})]_{z_{k-1}z_k}$, where [[CTMC矩阵指数|$P(\Delta)=e^{\Delta Q}$]] in the finite model. Five changes between hourly snapshots are not necessarily five genuine jumps, and unchanged endpoints can conceal departures and returns.
<!-- bilingual-en:end -->

若只在固定时刻 $s_0<\cdots<s_r$ 看到状态 $z_0,\ldots,z_r$，漏掉的跳跃使我们无法重建 $N_{ij}$ 与 $S_i$。条件似然应改为相乘 $[P(s_k-s_{k-1})]_{z_{k-1}z_k}$；有限模型下有[[CTMC矩阵指数|$P(\Delta)=e^{\Delta Q}$]]。每小时快照中五次状态变化，不一定就是五次真实跳跃；两端相同也可能隐藏离开再返回。

<!-- bilingual-en:start -->
Conditioning on the initial state is part of the setup. If its distribution depends on $Q$ and is included in the joint likelihood, that additional factor must be retained. On a countable state set the same product describes a finite path before explosion, but candidate rates must satisfy the chosen process model. Selecting only paths that survive to $T$ and then conditioning on survival can add a parameter-dependent normalization; it is not justified merely by observing a non-exploded record.
<!-- bilingual-en:end -->

条件于初态是这里的设定之一。若初始分布依赖 $Q$，且把它纳入联合似然，就必须保留该额外因子。在可数状态集上，同样乘积可以描述爆炸前的有限路径，但候选速率仍须满足所选过程模型。若只保留存活至 $T$ 的路径，再条件于存活，可能增加依赖参数的归一化因子；仅仅手中这条路径未爆炸，并不能省掉它。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [Tataru and Hobolth (2011), Background, equation (1)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3329461/): complete observation on a fixed interval, conditional on the initial state, with counts and occupation times as sufficient data for the likelihood.
- [Oxford Statistical Lifetime-Models, §10.1.2, PDF pp.71–72 / printed pp.63–64; §10.2, PDF p.72](https://www.stats.ox.ac.uk/~winkel/bs3b10_14.pdf#page=71): free-rate maximization and aggregated independent paths; the exposure construction includes waits ended by observation. The explicit terminal survival factor and boundary cases are derived above.
- [McGibbon and Pande (2015), §II and §III.A, equations (3), (6)–(8)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4514821/): finite-model matrix exponential and the conditional likelihood for discrete-time observations.
<!-- bilingual-en:end -->

- Tataru 与 Hobolth（2011）Background 式(1)：核对固定区间完整观察、条件于初态，以及计数与占用时间决定似然。
- Oxford §10.1.2，PDF71–72／印刷63–64页，及 §10.2 PDF72：核对自由速率最大化、独立路径汇总，以及观察中止的等待仍计入暴露。正文直接推导终点生存因子与参数边界。
- McGibbon 与 Pande（2015）§II、§III.A 式(3)、(6)–(8)：核对有限模型矩阵指数和离散时点观察的条件似然。

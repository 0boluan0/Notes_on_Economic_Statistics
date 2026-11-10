---
aliases:
  - Countable CTMC stationary existence
  - Positive recurrence and CTMC stationary distributions
  - 可数连续时间链平稳分布存在性
student_os: knowledge-atom
atom_id: PROB-CTMC-017
atom_set: continuous-time-markov-chains
atom_type: theorem
status: source-checked
requires:
  - "[[CTMC爆炸]]"
  - "[[常返与停留时间]]"
  - "[[CTMC平稳分布]]"
related:
  - "[[正常返与零常返]]"
  - "[[Kac回返公式]]"
  - "[[有限CTMC平稳生成矩阵判据]]"
leads_to:
  - "[[可数CTMC平衡收敛]]"
  - "[[M-M-1稳态人数]]"
part_of:
  - "[[连续时间马尔可夫链.canvas]]"
---

<!-- bilingual-en:start -->
# Stationary distributions of countable CTMCs
<!-- bilingual-en:end -->

*可数 CTMC 的平稳分布存在性*

<!-- bilingual-en:start -->
A system with infinitely many states can keep returning to familiar states yet spend such long periods away that no stationary probability distribution exists. What matters is not only whether it returns, but the **mean calendar time for a complete return cycle**. Positive recurrence makes this mean finite and allows the time spent in each state to be normalized into a probability distribution.
<!-- bilingual-en:end -->

一个具有无限多个状态的系统，可以不断返回熟悉的状态，却每次在外面花费太久，以至于没有平稳概率分布。关键不只是会不会回来，而是一次完整回返周期的**平均日历时间**是否有限。正常返保证这个均值有限，使各状态上的停留时间能够归一化成概率分布。

<!-- bilingual-en:start -->
## Which process and which return time?
<!-- bilingual-en:end -->

*先明确过程与回返时间*

<!-- bilingual-en:start -->
Consider a [[CTMC时间齐次转移函数|time-homogeneous CTMC]] on a finite or countably infinite state set $S$ with at least two states. Its [[生成矩阵约束|generator]] $Q=(q_{ij})$ is conservative, meaning $q_{ij}\ge0$ for $i\ne j$ and $q_i=-q_{ii}=\sum_{j\ne i}q_{ij}<\infty$. Assume [[不可约链|irreducibility]]: all states communicate by positive-rate routes. Also assume the [[最小CTMC|minimal process]] is [[CTMC爆炸|non-explosive]], so it remains in $S$ at every finite time with probability one.
<!-- bilingual-en:end -->

考虑状态集 $S$ 有限或可数无穷、且至少包含两个状态的[[CTMC时间齐次转移函数|时间齐次 CTMC]]。[[生成矩阵约束|生成矩阵]] $Q=(q_{ij})$ 保守：$i\ne j$ 时 $q_{ij}\ge0$，且 $q_i=-q_{ii}=\sum_{j\ne i}q_{ij}<\infty$。假设[[不可约链|不可约]]，即所有状态通过正率路线互达；另假设[[最小CTMC|最小过程]][[CTMC爆炸|非爆炸]]，因而每个有限时刻都以概率一留在 $S$ 中。

<!-- bilingual-en:start -->
Start at state $i$, let $J_1$ be the first genuine departure time, and define
<!-- bilingual-en:end -->

从状态 $i$ 出发，令 $J_1$ 为第一次真实离开时间，定义

$$C_i=\inf\{t\ge J_1:X_t=i\},\qquad m_i=E_i[C_i].$$

<!-- bilingual-en:start -->
Here $X_t$ is the state at time $t$, and $E_i$ means expectation when $X_0=i$. Set $C_i=\infty$ if no return occurs. The interval from $0$ to $C_i$ includes the initial wait in $i$ and all waits away from it; it is not merely a number of jumps. [[常返与停留时间|Positive recurrence in continuous time]] means $m_i<\infty$. Irreducibility and at least two states ensure $0<q_i<\infty$ for every $i$.
<!-- bilingual-en:end -->

这里 $X_t$ 是时刻 $t$ 的状态，$E_i$ 表示从 $X_0=i$ 出发取期望；若永不返回，规定 $C_i=\infty$。从 $0$ 到 $C_i$ 的区间包括最初在 $i$ 的停留和离开后的全部等待，不只是跳跃次数。[[常返与停留时间|连续时间正常返]]指 $m_i<\infty$。不可约且至少两个状态保证每个 $i$ 的出口率满足 $0<q_i<\infty$。

> [!important] Finite mean return time gives equilibrium · 平均回返时间有限才有平稳概率
> <!-- bilingual-en:start -->
> Under these assumptions, a [[CTMC平稳分布|stationary probability distribution]] exists if and only if one state, equivalently every state, is positive recurrent. When it exists it is unique, is positive at every state, and satisfies
> <!-- bilingual-en:end -->
>
> $$\pi_i=\frac{1}{q_i m_i}\qquad(i\in S).$$
>
> <!-- bilingual-en:start -->
> Stationary means $\pi P(t)=\pi$ for every $t\ge0$, where $P(t)$ is the transition matrix after duration $t$. The product $q_i m_i$ is dimensionless: rate times time.
> <!-- bilingual-en:end -->
>
> 在这些条件下，存在[[CTMC平稳分布|平稳概率分布]]，当且仅当某个状态正常返；这也等价于每个状态正常返。存在时它唯一、各项严格为正，并满足上式。平稳指每个 $t\ge0$ 都有 $\pi P(t)=\pi$，其中 $P(t)$ 为时长 $t$ 的转移矩阵。$q_i m_i$ 是速率乘时间，因此没有量纲。

<!-- bilingual-en:start -->
An absorbing singleton has stationary law $(1)$ but no genuine departure-and-return cycle. Handle it directly instead of inserting $q_i=0$ into this formula.
<!-- bilingual-en:end -->

只有一个吸收态的链具有平稳律 $(1)$，却没有真实的离开—返回周期；应直接处理，不能把 $q_i=0$ 代入公式。

<!-- bilingual-en:start -->
## Normalize time spent during a cycle
<!-- bilingual-en:end -->

*把一个周期中的停留时间归一化*

<!-- bilingual-en:start -->
Fix a reference state $i$ with $m_i<\infty$. For each state $j$, let
<!-- bilingual-en:end -->

固定满足 $m_i<\infty$ 的参考状态 $i$。对每个状态 $j$，令

$$u_j=E_i\!\left[\int_0^{C_i}\mathbf1_{\{X_s=j\}}\,ds\right].$$

<!-- bilingual-en:start -->
The [[事件指示变量|indicator]] is one when the process is at $j$ and zero otherwise, so the integral counts time spent at $j$ during the cycle. The states account for the whole cycle, giving $\sum_j u_j=m_i$. Thus $\pi_j=u_j/m_i$ is nonnegative and sums to one. The [[CTMC强Markov性|strong Markov property]] restarts the same experiment at each return to $i$; the occupation-cycle argument in the cited theorem shows that this normalized allocation is stationary. It is the same time-per-cycle logic used by the [[更新奖励定理|renewal reward theorem]].
<!-- bilingual-en:end -->

[[事件指示变量|指示量]]在过程位于 $j$ 时为一，否则为零，所以积分计算一个周期内在 $j$ 停留的时间。所有状态分配了整个周期，因此 $\sum_j u_j=m_i$，于是 $\pi_j=u_j/m_i$ 非负且总和为一。[[CTMC强Markov性]]让过程每次返回 $i$ 后重新开始同样的实验；来源定理的占据周期论证证明，这种归一化分配确实平稳。其直觉与[[更新奖励定理]]的“每周期收益除以每周期时长”相同。

<!-- bilingual-en:start -->
During this cycle, time at $i$ consists only of the initial holding period; the next visit ends the cycle. That period has [[CTMC指数停留时间|exponential mean]] $1/q_i$. Hence $u_i=1/q_i$ and $\pi_i=(1/q_i)/m_i$. The converse is substantive: in an irreducible non-explosive chain a stationary probability law forces finite mean return times. Recurrence alone, with an infinite mean, is insufficient.
<!-- bilingual-en:end -->

在这个周期里，在 $i$ 的停留只有最初那一段；下一次访问 $i$ 时周期就结束了。这段[[CTMC指数停留时间|指数等待]]的均值为 $1/q_i$，所以 $u_i=1/q_i$，得到 $\pi_i=(1/q_i)/m_i$。反方向同样是定理的实质：不可约非爆炸链若有平稳概率，就必有有限平均回返时间。仅仅常返、均值却无穷，还不够。

> [!example] A two-state check · 两状态核算
> <!-- bilingual-en:start -->
> Suppose $0\to1$ has rate $2$ and $1\to0$ rate $1$. A return cycle from $0$ contains one wait of mean $1/2$ and one of mean $1$, so $m_0=3/2$ and $\pi_0=1/(2\cdot3/2)=1/3$. Starting at $1$ gives the same cycle mean, hence $\pi_1=1/(1\cdot3/2)=2/3$. Longer occupancy of state $1$ produces greater stationary mass.
> <!-- bilingual-en:end -->
>
> 设 $0\to1$ 的速率为 $2$，$1\to0$ 的速率为 $1$。从 $0$ 开始的回返周期含均值 $1/2$ 与 $1$ 的两次等待，因此 $m_0=3/2$，$\pi_0=1/(2\cdot3/2)=1/3$。从 $1$ 开始也有相同周期均值，于是 $\pi_1=1/(1\cdot3/2)=2/3$。在状态 $1$ 停留更久，就获得更多平稳质量。

<!-- bilingual-en:start -->
## What fails on an infinite state space
<!-- bilingual-en:end -->

*无限状态下缺少什么*

<!-- bilingual-en:start -->
The star-shaped chain in [[常返与停留时间|the two-jump return example]] goes from $0$ to $j\ge1$ at rate $2^{-j}$ and from $j$ to $0$ at the same rate. It returns to $0$ after exactly two jumps, but $m_0=1+\sum_{j\ge1}1=\infty$. Its exit rates are at most one, so explosion is not the issue. Balance at each leaf would require $\pi_j=\pi_0$; infinitely many equal nonnegative masses cannot sum to one. The chain is recurrent but has no stationary probability.
<!-- bilingual-en:end -->

[[常返与停留时间|两跳回返例]]中的星形链，从 $0$ 到 $j\ge1$ 的速率为 $2^{-j}$，从 $j$ 回 $0$ 也为这个速率。它恰好两跳就返回 $0$，但 $m_0=1+\sum_{j\ge1}1=\infty$。出口率都不超过一，因此问题不是爆炸。每个叶节点的平衡要求 $\pi_j=\pi_0$；无限多个相同的非负质量无法总和为一。这条链常返，却没有平稳概率。

<!-- bilingual-en:start -->
For the [[M-M-1稳态人数|M/M/1 system-size chain]], arrivals occur at rate $\lambda>0$ and service completions at rate $\mu>0$ when the system is nonempty. Both rates are finite. It is positive recurrent when $\rho=\lambda/\mu<1$, with $\pi_j=(1-\rho)\rho^j$ for $j\ge0$; at $\rho=1$ it is null recurrent, and at $\rho>1$ it is transient. System size includes a customer currently in service. Legal generator rows alone do not select the stable regime.
<!-- bilingual-en:end -->

对[[M-M-1稳态人数|M/M/1 系统人数链]]，到达率为有限正数 $\lambda$，系统非空时的服务完成率为有限正数 $\mu$。当 $\rho=\lambda/\mu<1$ 时正常返，且 $j\ge0$ 时 $\pi_j=(1-\rho)\rho^j$；$\rho=1$ 时零常返，$\rho>1$ 时暂态。系统人数包括正在接受服务的顾客。生成矩阵逐行合法，本身还不能确定系统处于稳定区域。

<!-- bilingual-en:start -->
In the present irreducible **non-explosive** setting, finding a nonnegative normalized solution of $\pi Q=0$ does establish stationarity and positive recurrence, by the countable-state theorem. It is unnecessary to assume positive recurrence again after applying that theorem. Without the non-explosion condition, a normalizable formal balance solution alone does not certify the minimal process's stationary law. Infinite sums also require a valid argument; the finite matrix proof cannot simply be copied. Once existence is established, [[可数CTMC平衡收敛|convergence from other initial laws]] is a further theorem.
<!-- bilingual-en:end -->

在本卡不可约、**非爆炸**的设定下，找到非负且归一化的 $\pi Q=0$ 解，确实可由可数状态定理推出平稳性与正常返；应用这个定理之后，无须把正常返再额外假设一次。但删去非爆炸条件，能归一化的形式平衡解仍不能认证最小过程的平稳律。无穷求和也需要有效论证，不能照抄有限矩阵证明。确认存在之后，[[可数CTMC平衡收敛|其他初始律向它收敛]]是进一步的定理。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [Berestycki–Sousi, Applied Probability, Theorems 2.13–2.15 and 2.19, PDF pp.24–27](https://www.statslab.cam.ac.uk/~ps422/notes-new2015.pdf#page=24): return-cycle definition, finite mean versus stationary probability, occupation measure, uniqueness and $\pi_i=1/(q_im_i)$. Its use of “invariant” initially means generator balance; the semigroup argument is read as well. Irreducibility is retained explicitly.
- [Whitt, Continuous-Time Markov Chains (2012), Theorem 10.5, PDF pp.52–53 / printed pp.50–51](https://www.columbia.edu/~ww2040/Whitt_CTMCnotes121312.pdf#page=52): for irreducible regular chains, a probability solution of the balance equations is sufficient and equivalent to positive recurrence. Here regularity is supplied by non-explosion of the conservative minimal process.
- [Berestycki–Sousi, §3.2, PDF p.34](https://www.statslab.cam.ac.uk/~ps422/notes-new2015.pdf#page=34): the M/M/1 recurrence regimes and geometric law. The two-state and star-chain calculations are checked directly.
<!-- bilingual-en:end -->

- Berestycki–Sousi 定理2.13–2.15、2.19（PDF24–27页）：回返周期定义、平均时间有限与平稳概率、占据测度、唯一性及回返公式。原文起初的 invariant 指生成矩阵平衡，已同时核对半群论证；本卡明确保留不可约。
- Whitt（2012）定理10.5（PDF52–53／印刷50–51页）：不可约 regular 链的概率平衡解足以且等价于正常返；本卡通过保守最小过程非爆炸落实该过程条件。
- Berestycki–Sousi §3.2（PDF34页）：M/M/1 的常返分类与几何律；两状态和星形链算式直接核算。

---
aliases:
  - Finite-state CTMC stationary generator criterion
  - CTMC stationary equation
student_os: knowledge-atom
atom_id: PROB-CTMC-038
atom_set: continuous-time-markov-chains
atom_type: theorem
status: source-checked
requires:
  - "[[CTMC平稳分布]]"
  - "[[生成矩阵约束]]"
  - "[[CTMC矩阵指数]]"
  - "[[Markov矩阵左右约定]]"
related:
  - "[[有限不可约CTMC稳态]]"
  - "[[可数CTMC稳态存在]]"
  - "[[跳链频率时间加权]]"
leads_to:
  - "[[CTMC详细平衡与可逆]]"
part_of:
  - "[[连续时间马尔可夫链.canvas]]"
---

<!-- bilingual-en:start -->
# The finite-state stationary generator criterion
<!-- bilingual-en:end -->

*有限 CTMC 的平稳生成矩阵判据*

<!-- bilingual-en:start -->
A stationary allocation must replenish each state's probability at the same rate that probability leaves it. In a finite continuous-time Markov chain, these balance equations form a finite linear system. Solving it, together with nonnegativity and total mass one, is enough to check stationarity for every elapsed time.
<!-- bilingual-en:end -->

平稳的概率分配要求：每个状态流入概率的速率，恰好补回流出概率的速率。对有限状态连续时间 Markov 链，这些平衡条件组成一个有限线性方程组。配合非负和总质量为一，解这个方程组就足以检验所有时长上的平稳性。

<!-- bilingual-en:start -->
Let $S=\{1,\ldots,d\}$ be a finite nonempty state set, where $d$ is the number of states. Consider a [[CTMC时间齐次转移函数|time-homogeneous CTMC]] with [[生成矩阵约束|generator]] $Q=(q_{ij})$: its entries are finite, $q_{ij}\ge0$ for $i\ne j$, and $q_{ii}=-\sum_{j\ne i}q_{ij}$. Write $q_i=-q_{ii}$ for the [[出口率|exit rate]] from $i$. Such a finite chain is non-explosive. Use the [[Markov矩阵左右约定|row convention]] for probability vectors and $P(t)$ for the transition matrix after duration $t$.
<!-- bilingual-en:end -->

设 $S=\{1,\ldots,d\}$ 是有限非空状态集，$d$ 是状态数。考虑[[CTMC时间齐次转移函数|时间齐次 CTMC]]，其[[生成矩阵约束|生成矩阵]] $Q=(q_{ij})$ 的元素有限，$i\ne j$ 时 $q_{ij}\ge0$，且 $q_{ii}=-\sum_{j\ne i}q_{ij}$。记 $q_i=-q_{ii}$ 为从 $i$ 离开的[[出口率]]。这样的有限链不会爆炸。概率向量采用[[Markov矩阵左右约定|行约定]]，$P(t)$ 表示经过时长 $t$ 的转移矩阵。

> [!important] Rates determine stationary probabilities · 用速率判定平稳概率
> <!-- bilingual-en:start -->
> A row $\pi=(\pi_1,\ldots,\pi_d)$ is a [[CTMC平稳分布|stationary probability distribution]] exactly when
> <!-- bilingual-en:end -->
>
> $$\pi Q=0,\qquad \pi_i\ge0\ (1\le i\le d),\qquad \sum_{i=1}^{d}\pi_i=1.$$
>
> <!-- bilingual-en:start -->
> Here $0$ is the zero row. Thus $\pi$ is a [[左零空间|left null vector]] of $Q$, written as a row; the probability restrictions are essential. No irreducibility assumption is needed for this equivalence.
> <!-- bilingual-en:end -->
>
> 行 $\pi=(\pi_1,\ldots,\pi_d)$ 是[[CTMC平稳分布|平稳概率分布]]，当且仅当满足上面三项条件；$0$ 表示零行。因此 $\pi$ 是以行表示的 $Q$ 的[[左零空间|左零向量]]，但概率限制不能省略。这一等价不要求不可约。

<!-- bilingual-en:start -->
To read the equation physically, examine its $j$th coordinate and move the diagonal term to the other side:
<!-- bilingual-en:end -->

要理解方程的含义，取第 $j$ 项，把对角项移到另一侧：

$$\sum_{i\ne j}\pi_i q_{ij}=\pi_j q_j.$$

<!-- bilingual-en:start -->
The left side is the total stationary flow into $j$ from all other states; the right side is the total flow out. These quantities are rates, not probabilities of a transition over a whole hour. For example, when $q_{ij}$ is measured per hour, the flows are probability mass per hour.
<!-- bilingual-en:end -->

左侧是从其他各状态流入 $j$ 的总平稳流量，右侧是流出 $j$ 的总流量。它们是速率，不是整整一小时内的转移概率；例如 $q_{ij}$ 的单位为每小时，流量的单位就是概率质量每小时。

> [!example] Solve one balance equation and normalize · 解平衡式并归一化
> <!-- bilingual-en:start -->
> With $Q=\begin{pmatrix}-2&2\\1&-1\end{pmatrix}$, write $\pi=(a,b)$. Then $\pi Q=(-2a+b,2a-b)$. The two zero equations are the same constraint $b=2a$; normalization $a+b=1$ gives $a=1/3$, $b=2/3$. The opposite flows are both $2/3$ per unit time. The second state has a slower exit rate, so the unequal stationary allocation is compatible with alternating genuine jumps. [[跳链频率时间加权|Jump frequencies and calendar-time probabilities]] need not agree.
> <!-- bilingual-en:end -->
>
> 对 $Q=\begin{pmatrix}-2&2\\1&-1\end{pmatrix}$，令 $\pi=(a,b)$，则 $\pi Q=(-2a+b,2a-b)$。两个零方程其实都是 $b=2a$；结合 $a+b=1$，得到 $a=1/3$、$b=2/3$。两个方向的流量都是每单位时间 $2/3$。第二个状态的出口率较慢，所以真实跳跃即使交替进行，平稳概率仍不相等；[[跳链频率时间加权|按跳跃次数与按钟表时间观察]]并不相同。

<!-- bilingual-en:start -->
## Why the two definitions agree
<!-- bilingual-en:end -->

## 为什么等价

<!-- bilingual-en:start -->
Suppose first that $\pi$ is stationary. By definition, $\pi P(t)=\pi$ for every $t\ge0$. The [[Kolmogorov前后向方程|finite-state transition equations]] give the right derivative $P'(0)=Q$. Differentiating the constant row $\pi P(t)$ at zero therefore yields $\pi Q=0$.
<!-- bilingual-en:end -->

先假设 $\pi$ 平稳。按定义，对所有 $t\ge0$ 有 $\pi P(t)=\pi$。[[Kolmogorov前后向方程|有限状态转移方程]]给出零时刻右导数 $P'(0)=Q$。因此对恒定的行 $\pi P(t)$ 在零处求右导数，就得到 $\pi Q=0$。

<!-- bilingual-en:start -->
Conversely, assume the three displayed conditions. The [[CTMC矩阵指数|finite-state matrix-exponential formula]] is $P(t)=I+\sum_{k=1}^{\infty}t^kQ^k/k!$, where $I$ is the [[单位矩阵|identity]], $Q^k$ is the $k$fold matrix product and $k!$ the factorial. Since $\pi Q=0$, associativity gives $\pi Q^k=(\pi Q)Q^{k-1}=0$ for every integer $k\ge1$. The convergent series then gives
<!-- bilingual-en:end -->

反过来，假设满足上面三项条件。[[CTMC矩阵指数|有限状态矩阵指数公式]]是 $P(t)=I+\sum_{k=1}^{\infty}t^kQ^k/k!$，其中 $I$ 是[[单位矩阵]]，$Q^k$ 是 $k$ 次矩阵乘积，$k!$ 是阶乘。因为 $\pi Q=0$，由结合律，对每个整数 $k\ge1$ 都有 $\pi Q^k=(\pi Q)Q^{k-1}=0$。于是收敛级数给出

$$\pi P(t)=\pi I+\sum_{k=1}^{\infty}\frac{t^k}{k!}\pi Q^k=\pi.$$

<!-- bilingual-en:start -->
Nonnegative entries summing to one ensure that this invariant row is a probability distribution. This proves both directions for finite chains, including reducible chains and absorbing states.
<!-- bilingual-en:end -->

分量非负且总和为一，保证这个不变行确实是概率分布。至此完成有限链的双向证明，包括可约链与含吸收态的链。

<!-- bilingual-en:start -->
## What the equation does and does not say
<!-- bilingual-en:end -->

## 边界

<!-- bilingual-en:start -->
The row-sum identity $Q\mathbf1=0$, where $\mathbf1$ is the all-ones column, is a different equation: it holds for every generator and does not find a stationary probability row. Nor is every left null vector a probability vector. For the two-state zero matrix $Q=0$, the signed row $(2,-1)$ has sum one and satisfies $xQ=0$, yet its negative entry disqualifies it. Every genuine probability row is stationary in that example, so the linear criterion itself does not ensure uniqueness.
<!-- bilingual-en:end -->

零行和恒等式 $Q\mathbf1=0$ 是另一条方程，其中 $\mathbf1$ 是全一列；它对每个生成矩阵都成立，不能用来求平稳概率行。也不是每个左零向量都是概率向量。对两状态零矩阵 $Q=0$，有符号行 $(2,-1)$ 总和为一并满足 $xQ=0$，但负分量使它不是概率分布。该例的每个真正概率行都平稳，所以线性判据本身不保证唯一性。

<!-- bilingual-en:start -->
Total flow balance is weaker than [[CTMC详细平衡|detailed balance]], which matches each pair separately. A three-state directed cycle $1\to2\to3\to1$, all at rate one, has uniform stationary probabilities: each state receives and sends rate $1/3$. But flow $1\to2$ is $1/3$ while reverse flow is zero. Pairwise balance is not required by $\pi Q=0$.
<!-- bilingual-en:end -->

总流量平衡弱于逐对匹配的[[CTMC详细平衡]]。三状态单向循环 $1\to2\to3\to1$，每条边速率为一，具有均匀平稳概率：各状态流入和流出速率都是 $1/3$。但 $1\to2$ 的流量是 $1/3$，反向为零。$\pi Q=0$ 并不要求逐对平衡。

<!-- bilingual-en:start -->
The finite-state qualification also matters. With infinitely many states, differentiating through an infinite sum and identifying the actual conservative process require justification. A normalized formal $\pi Q=0$ solution is not, by this finite proof alone, a stationary law of an explosive minimal process. See [[可数CTMC稳态存在|the countable-state existence conditions]].
<!-- bilingual-en:end -->

有限状态限制也很重要。状态无限时，穿过无穷和求导，以及确定实际的保守过程，都需要论证。仅靠这里的有限维证明，不能把归一化的形式解 $\pi Q=0$ 当成爆炸最小过程的平稳分布，见[[可数CTMC稳态存在|可数状态的存在条件]]。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [Berestycki–Sousi, Applied Probability, Theorem 2.17, PDF p.26; Theorems 1.27–1.29, pp.17–18](https://www.statslab.cam.ac.uk/~ps422/notes-new2015.pdf#page=26): the finite-state equivalence and matrix-exponential prerequisites. Example 2.16 on p.25 explains why an explosive countable chain needs additional care.
- [Whitt, Continuous-Time Markov Chains, Theorem 4.2(c) and its proof, PDF pp.19–21 / printed pp.18–20](https://www.columbia.edu/~ww2040/4106S11/CTMCchapter121906.pdf#page=19): generator balance equations and flow interpretation. Its uniqueness statement is within the finite irreducible setting; this card's general finite equivalence does not assert uniqueness.
- [MIT 6.436J, Lecture 24, Proposition 2 and finite-state discussion, PDF p.7](https://ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/087af3cedbc9def5b156c5e1665ac79c_MIT6_436JF18_lec24.pdf#page=7): normalized generator equations, used here with explicit finite-state scope. [[01_Math/05_随机过程/05_连续时间的马尔可夫链#5. 平稳分布、唯一性与收敛|Permanent course §5]] fixes notation. Both examples and the proof are checked directly.
<!-- bilingual-en:end -->

- Berestycki–Sousi 定理2.17（PDF26页）与定理1.27–1.29（17–18页）：有限状态等价及矩阵指数前置；25页例2.16说明爆炸可数链为什么需要另加条件。
- Whitt 定理4.2(c)及证明（PDF19–21／印刷18–20页）：生成矩阵平衡方程及流量解释。原文唯一性处于有限不可约语境，本卡的一般有限等价不声称唯一。
- MIT6.436J 第24讲命题2及有限状态讨论（PDF7页）：归一化生成矩阵方程，本卡明确限定有限状态。永久课程第5节用于记号；例子与证明均直接核算。

---
aliases:
- CTMC uniformization
- Jensen method
- Randomization method
- 均匀化
student_os: knowledge-atom
atom_id: PROB-CTMC-020
atom_set: continuous-time-markov-chains
atom_type: method
status: source-checked
requires:
- '[[生成矩阵约束]]'
- '[[CTMC转移半群]]'
- '[[有界率非爆炸]]'
related:
- '[[生成矩阵分解]]'
- '[[嵌入跳链]]'
- '[[CTMC矩阵指数]]'
leads_to:
- '[[CTMC平稳分布]]'
part_of:
- '[[连续时间马尔可夫链.canvas]]'
---

<!-- bilingual-en:start -->
# CTMC uniformization
<!-- bilingual-en:end -->

*连续时间 Markov 链的均匀化*

<!-- bilingual-en:start -->
A continuous-time Markov chain usually waits at different speeds in different states. **Uniformization** represents the same process using one constant-rate clock. When that clock rings, the chain either changes state or stays where it is. These extra “nothing happened” events let a single clock accommodate both fast and slow states without changing the genuine state path's probability law.
<!-- bilingual-en:end -->

连续时间 Markov 链通常在不同状态以不同速度等待。**均匀化**用一个固定速率时钟表示同一个过程。时钟响时，链可以改变状态，也可以留在原处。加入这些“什么都没变”的候选事件，就能让一个时钟同时适配快、慢状态，而不改变真实状态路径的概率律。

<!-- bilingual-en:start -->
Let $Q=(q_{ij})$ be the [[生成矩阵约束|generator]] on a finite or countable state set $S$, using the [[Markov矩阵左右约定|row convention]]: $q_{ij}$ points from $i$ to $j$. The [[出口率|exit rates]] are $q_i=-q_{ii}=\sum_{j\ne i}q_{ij}$. Assume one finite bound works for all $q_i$; this guarantees [[有界率非爆炸|non-explosion of the minimal process]]. Choose a finite candidate rate $\nu>0$ with $\nu\ge\sup_iq_i$.
<!-- bilingual-en:end -->

令 $Q=(q_{ij})$ 是有限或可数状态集合 $S$ 上的[[生成矩阵约束|生成矩阵]]，采用[[Markov矩阵左右约定|行约定]]：$q_{ij}$ 指向从 $i$ 到 $j$。[[出口率]]为 $q_i=-q_{ii}=\sum_{j\ne i}q_{ij}$。假设所有 $q_i$ 共享一个有限上界，这保证[[有界率非爆炸|最小过程非爆炸]]。选取有限候选速率 $\nu>0$，满足 $\nu\ge\sup_iq_i$。

<!-- bilingual-en:start -->
## What to do at one clock ring
<!-- bilingual-en:end -->

*每次时钟响时做什么*

<!-- bilingual-en:start -->
Define a discrete [[DTMC时间齐次转移核|transition matrix]] $R$ by the following entries. The [[单位矩阵|identity]] $I$ has diagonal entries one and all other entries zero.
<!-- bilingual-en:end -->

按下列各项定义一个离散[[DTMC时间齐次转移核|转移矩阵]] $R$。[[单位矩阵]] $I$ 的对角元为一，其余项为零。

$$R=I+\frac Q\nu,\qquad R_{ij}=\frac{q_{ij}}\nu\ (j\ne i),\qquad R_{ii}=1-\frac{q_i}\nu.$$

<!-- bilingual-en:start -->
Every entry is nonnegative and every row sums to one. From state $i$, a candidate event causes a genuine departure with probability $q_i/\nu$, and keeps the state unchanged with probability $1-q_i/\nu$. A self-loop of $R$ is a **virtual event**: it is not a real departure followed by an immediate return. Thus $R$ is generally different from the [[嵌入跳链|embedded genuine-jump chain]], which deletes such events.
<!-- bilingual-en:end -->

每项都非负，每行和都是一。处于 $i$ 时，一次候选事件以概率 $q_i/\nu$ 导致真实离开，以概率 $1-q_i/\nu$ 保持原状态。$R$ 的自环是**虚事件**，不是先真的离开又立即返回。因此 $R$ 通常不同于去掉这些事件的[[嵌入跳链|嵌入真实跳链]]。

<!-- bilingual-en:start -->
Take a discrete-time chain $Y_0,Y_1,\ldots$ with transition matrix $R$, and an independent [[齐次泊松过程|homogeneous Poisson process]] $N(t)$ of rate $\nu$, starting at $N(0)=0$. The count $N(t)$ tells us how many candidate events have occurred by time $t$. Set $X(t)=Y_{N(t)}$, with the required initial distribution for $Y_0$. This specifies the whole process, including its joint distributions at several times, not just one marginal at time $t$.
<!-- bilingual-en:end -->

取转移矩阵为 $R$ 的离散时间链 $Y_0,Y_1,\ldots$，以及与它独立、速率为 $\nu$ 的[[齐次泊松过程]] $N(t)$，从 $N(0)=0$ 开始。计数 $N(t)$ 告诉我们截至 $t$ 发生了多少候选事件。令 $X(t)=Y_{N(t)}$，并给 $Y_0$ 所需的初始分布。这规定的是整个过程，包括多个时刻的联合分布，而不只是时刻 $t$ 的一个边际分布。

> [!important] Exact Poisson mixture / 精确 Poisson 混合
> <!-- bilingual-en:start -->
> Write $P_{ij}(t)=P_i(X(t)=j)$, where $P_i$ means that the process starts at $i$. Given $N(t)=k$, the discrete chain has taken $k$ steps, so its transition probabilities are $(R^k)_{ij}$. Averaging over the Poisson count gives the formula below. In a countable state space it is read entry by entry; all terms are nonnegative and the weights sum to one. $R^0=I$ includes the possibility of no candidate event.
> <!-- bilingual-en:end -->
>
> 记 $P_{ij}(t)=P_i(X(t)=j)$，其中 $P_i$ 表示过程从 $i$ 出发时的概率。给定 $N(t)=k$，离散链已经走了 $k$ 步，转移概率就是 $(R^k)_{ij}$。对 Poisson 计数取平均，得到下式。状态可数时，逐个矩阵项理解这个级数；各项非负，权重和为一。$R^0=I$ 包含一次候选事件也没有发生的可能。
>
> $$P_{ij}(t)=e^{-\nu t}\sum_{k=0}^{\infty}\frac{(\nu t)^k}{k!}(R^k)_{ij},\qquad t\ge0.$$
> $$P(t)=e^{-\nu t}\sum_{k=0}^{\infty}\frac{(\nu t)^k}{k!}R^k.$$

<!-- bilingual-en:start -->
## Why the genuine waiting time is unchanged
<!-- bilingual-en:end -->

*为什么真实等待时间不变*

<!-- bilingual-en:start -->
Start at $i$ and let $H$ be the time of the first genuine departure. No genuine departure by $t$ means that every candidate event up to $t$ was a self-loop. Conditional on $N(t)=k$, this probability is $R_{ii}^k$. Hence
<!-- bilingual-en:end -->

从 $i$ 开始，令 $H$ 为首次真实离开的时间。截至 $t$ 还没真正离开，表示此前每次候选事件都是自环。给定 $N(t)=k$，其概率为 $R_{ii}^k$，因此

$$P_i(H>t)=e^{-\nu t}\sum_{k=0}^{\infty}\frac{(\nu t R_{ii})^k}{k!}=e^{-\nu(1-R_{ii})t}=e^{-q_it}.$$

<!-- bilingual-en:start -->
For $q_i>0$, conditional on an event being a genuine departure, its destination is $j\ne i$ with probability $R_{ij}/(1-R_{ii})=q_{ij}/q_i$. The independent clock intervals and fresh transition choices give the same waiting-time/destination independence as the [[生成矩阵分解|usual CTMC construction]]. Repeating after each change therefore gives exactly the same process law. If $q_i=0$, every event is virtual and $H=\infty$. The Poisson clock makes only finitely many candidate events in each finite interval, so this construction is non-explosive.
<!-- bilingual-en:end -->

当 $q_i>0$ 时，条件于一次事件确实离开，去向为 $j\ne i$ 的概率是 $R_{ij}/(1-R_{ii})=q_{ij}/q_i$。独立的时钟间隔和每次重新作出的转移选择，给出与[[生成矩阵分解|标准 CTMC 构造]]相同的等待时间与去向独立性。每次变化后重复，就得到完全相同的过程律。若 $q_i=0$，每次事件都是虚事件，$H=\infty$。Poisson 时钟在每个有限区间中只有有限个候选事件，所以这个构造非爆炸。

<!-- bilingual-en:start -->
## A two-state calculation
<!-- bilingual-en:end -->

*一个两状态计算*

<!-- bilingual-en:start -->
Suppose the exit rates from states $0$ and $1$ are respectively $2$ and $1$ per unit time, with each departure going to the other state. Choose $\nu=3$. Then
<!-- bilingual-en:end -->

设状态 $0$ 和 $1$ 的出口率分别为每单位时间 $2$ 和 $1$，每次离开都去另一个状态。取 $\nu=3$，则

$$Q=\begin{pmatrix}-2&2\\1&-1\end{pmatrix},\qquad R=\begin{pmatrix}1/3&2/3\\1/3&2/3\end{pmatrix}.$$

<!-- bilingual-en:start -->
At state $0$, one third of candidate events are virtual; at state $1$, two thirds are virtual. Direct multiplication gives $R^2=R$, hence $R^k=R$ for every $k\ge1$. Separating the zero-event term in the mixture yields
<!-- bilingual-en:end -->

在状态 $0$，三分之一的候选事件是虚事件；在状态 $1$，三分之二是虚事件。直接相乘可得 $R^2=R$，所以对每个 $k\ge1$ 都有 $R^k=R$。把混合式中零次事件的项分开，得到

$$P(t)=e^{-3t}I+(1-e^{-3t})R,\qquad P_{01}(1)=\frac23(1-e^{-3})\approx0.633475.$$

> [!warning] Not an Euler time step / 不是 Euler 时间步
> <!-- bilingual-en:start -->
> Although $R=I+Q/\nu$ resembles an Euler step of size $1/\nu$, $R$ is a transition at a **candidate event**, not over a fixed interval of that length. In the example, $P(1/3)=e^{-1}I+(1-e^{-1})R\ne R$. Fixed-time evolution must average over zero, one, two, and all larger possible candidate counts. Uniformization is exact; truncating its infinite sum for numerical work introduces a separate approximation.
> <!-- bilingual-en:end -->
>
> 虽然 $R=I+Q/\nu$ 看起来像步长 $1/\nu$ 的 Euler 一步，但 $R$ 描述的是一次**候选事件**时的转移，不是经过这么长的固定时间后的转移。上例有 $P(1/3)=e^{-1}I+(1-e^{-1})R\ne R$。固定时间的演化必须平均零次、一次、两次及更多候选事件的可能。均匀化本身精确；数值计算时截断无穷级数，才另外引入近似。

<!-- bilingual-en:start -->
## Choosing the candidate rate
<!-- bilingual-en:end -->

*如何选择候选速率*

<!-- bilingual-en:start -->
Any finite $\nu\ge\sup_iq_i$ with $\nu>0$ works, including equality. A larger $\nu$ produces more candidates and more self-loops, leaving the genuine path law unchanged while usually increasing computational work. If all rates vanish, $Q=0$ and $P(t)=I$; one can still take any $\nu>0$ and obtain $R=I$. Taking $\nu=0$ in $Q/\nu$ is not a valid formula.
<!-- bilingual-en:end -->

任何满足 $\nu>0$ 且 $\nu\ge\sup_iq_i$ 的有限值都可以，包括等号。更大的 $\nu$ 产生更多候选事件和更多自环，真实路径律保持不变，但通常增加计算工作。如果全部速率为零，$Q=0$ 且 $P(t)=I$；仍可任取 $\nu>0$，得到 $R=I$。不能在 $Q/\nu$ 中直接代入 $\nu=0$。

<!-- bilingual-en:start -->
A finite state space with finite rates always allows a choice. A countable state space with unbounded exit rates does not allow this single global finite-rate clock, even when the chain is non-explosive. For finite matrices the same mixture equals the [[CTMC矩阵指数|matrix exponential]] $e^{tQ}$. That finite-dimensional algebra is not needed to justify the entrywise Poisson construction on a countable state space.
<!-- bilingual-en:end -->

状态有限且速率有限时，总能作出这种选择。状态可数但出口率无界时，无法使用这里的单个全局有限速率时钟，即使过程本身非爆炸。对有限矩阵，这个混合式也等于[[CTMC矩阵指数|矩阵指数]] $e^{tQ}$。而可数状态下逐项成立的 Poisson 构造，不需要依靠有限维矩阵代数来证明。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [Whitt, §3.4, equations (3.22)–(3.30) and Theorem 3.4, PDF pp. 15–16](https://www.columbia.edu/~ww2040/4106S11/CTMCchapter121906.pdf#page=15): independent Poisson time change, candidate self-loops, the exact mixture and the finite-matrix exponential representation.
- [Rao and Teh (2013), §3.1 and Proposition 1, PDF pp. 5–6](https://jmlr.csail.mit.edu/papers/volume14/rao13a/rao13a.pdf#page=5): exact finite-dimensional law and virtual jumps for finite-state processes. Their column convention is transposed to the row convention used here.
- [Cambridge Applied Probability, Definition 1.15 and Theorem 1.19, PDF pp. 12–14](https://www.statslab.cam.ac.uk/~ps422/notes-new2015.pdf#page=12): the countable-state holding-time construction and bounded-rate non-explosion used to extend the argument. The two-state matrix and fixed-time comparison are calculated directly above.
<!-- bilingual-en:end -->

- Whitt 第 3.4 节、式 (3.22)–(3.30) 和定理 3.4，PDF 页 15–16：核对独立 Poisson 时间改变、候选自环、精确混合式及有限矩阵指数表示。
- Rao 与 Teh（2013）第 3.1 节及命题 1、PDF 页 5–6：核对有限状态过程的精确有限维分布与虚跳；其列约定已转换为本卡的行约定。
- Cambridge 定义 1.15 及定理 1.19，PDF 页 12–14：核对可数状态停留时间构造与有界率非爆炸，支持上述推广；两状态矩阵与固定时间比较直接计算。

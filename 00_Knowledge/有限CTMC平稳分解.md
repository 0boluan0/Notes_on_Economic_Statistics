---
aliases:
- Finite CTMC stationary decomposition
- Closed-class stationary distributions
- 有限连续时间链平稳分解
student_os: knowledge-atom
atom_id: PROB-CTMC-015
atom_set: continuous-time-markov-chains
atom_type: theorem
status: source-checked
requires:
- '[[跳链决定CTMC沟通类]]'
- '[[CTMC平稳分布]]'
- '[[有限不可约CTMC稳态]]'
related:
- '[[有限链平稳分解]]'
leads_to:
- '[[有限CTMC全局收敛]]'
part_of:
- '[[连续时间马尔可夫链.canvas]]'
---

<!-- bilingual-en:start -->
# Stationary decomposition of a finite CTMC
<!-- bilingual-en:end -->

*有限 CTMC 的平稳分布分解*

<!-- bilingual-en:start -->
A system can have several regions that it can enter but never leave. Within each region, its probabilities may settle into a characteristic balance. **Stationary decomposition** answers a different question: how can we distribute probability across all those regions so that the entire distribution remains unchanged? For a finite continuous-time Markov chain, we choose how much probability belongs to each closed region, then use that region's own stationary proportions.
<!-- bilingual-en:end -->

一个系统可能有几个进入以后就出不来的区域。每个区域内部，状态概率可以保持某种特定平衡。**平稳分解**回答的是：怎样把概率分配到这些区域，才能使全链的分布保持不变？对于有限连续时间 Markov 链，先选择每个闭合区域分到多少概率，再按该区域自身的平稳比例分配即可。

<!-- bilingual-en:start -->
Throughout, the state set $S$ is nonempty and finite, and the chain is [[CTMC时间齐次转移函数|time-homogeneous]]: its transition rules do not depend on the calendar time. Its [[生成矩阵约束|generator]] $Q=(q_{ij})$ has finite nonnegative off-diagonal rates, with $q_{ii}=-\sum_{j\ne i}q_{ij}$. We use the [[Markov矩阵左右约定|row convention]], so $q_{ij}$ is the rate from $i$ to $j$, and an initial probability row $\alpha$ evolves to $\alpha P(t)$. Here $P_{ij}(t)=\Pr_i(X(t)=j)$, where $\Pr_i$ means starting at $i$. Finite size and finite rates guarantee [[有界率非爆炸|non-explosion]]: infinitely many genuine jumps cannot occur in a finite time interval.
<!-- bilingual-en:end -->

全篇假设状态集 $S$ 非空且有限，链[[CTMC时间齐次转移函数|时间齐次]]，即转移规则不随日历时刻改变。[[生成矩阵约束|生成矩阵]] $Q=(q_{ij})$ 的非对角速率有限且非负，并有 $q_{ii}=-\sum_{j\ne i}q_{ij}$。采用[[Markov矩阵左右约定|行约定]]，$q_{ij}$ 是从 $i$ 到 $j$ 的速率，初始概率行 $\alpha$ 演化为 $\alpha P(t)$。其中 $P_{ij}(t)=\Pr_i(X(t)=j)$，$\Pr_i$ 表示从 $i$ 出发时的概率。有限状态和有限速率保证[[有界率非爆炸|非爆炸]]：有限时间内不可能发生无穷次真实跳跃。

<!-- bilingual-en:start -->
## An entrance, a moving pair and an absorbing state
<!-- bilingual-en:end -->

*一个入口、一个仍在运动的状态对和一个吸收点*

<!-- bilingual-en:start -->
Order four states as $0,1,2,3$, and take
<!-- bilingual-en:end -->

按 $0,1,2,3$ 排列四个状态，取

$$Q=\begin{pmatrix}
-3&1&0&2\\
0&-2&2&0\\
0&1&-1&0\\
0&0&0&0
\end{pmatrix}.$$

<!-- bilingual-en:start -->
From $0$, the process can jump to $1$ or $3$, but cannot return to $0$. States $1$ and $2$ communicate: each can reach the other, with rates $2$ from $1$ to $2$ and $1$ in the reverse direction. They have no exit outside the pair. State $3$ is [[吸收状态|absorbing]], because its exit rate is zero. Thus the [[闭沟通类|closed communicating classes]] are $C_1=\{1,2\}$ and $C_2=\{3\}$. The class definition uses positive-rate paths for a CTMC, as explained by [[跳链决定CTMC沟通类|the connection with the jump graph]]. A closed class need not be a single state or stop moving.
<!-- bilingual-en:end -->

从 $0$ 可以跳到 $1$ 或 $3$，却不能回到 $0$。状态 $1$、$2$ 互通：$1$ 到 $2$ 的速率为 $2$，反向速率为 $1$，而这两个状态没有通向其他状态的出口。状态 $3$ 的出口率为零，是[[吸收状态]]。因此[[闭沟通类]]为 $C_1=\{1,2\}$ 与 $C_2=\{3\}$。CTMC 用正速率路径判断类，见[[跳链决定CTMC沟通类|跳转图与沟通类的关系]]。闭类不一定是单个状态，也不表示停止运动。

<!-- bilingual-en:start -->
Within the pair, balance requires $2\pi_1=\pi_2$: the probability flow from $1$ to $2$ equals the reverse flow. Normalizing the pair to total mass one gives $(1/3,2/3)$. Extend each class distribution to the whole state set by assigning zero outside its class:
<!-- bilingual-en:end -->

在状态对内部，平衡要求 $2\pi_1=\pi_2$，即从 $1$ 流向 $2$ 的概率流与反向流相等。把该状态对的总概率归一为一，得到 $(1/3,2/3)$。在类外补零，就得到全状态集上的两个概率行：

$$\pi^{(1)}=(0,1/3,2/3,0),\qquad \pi^{(2)}=(0,0,0,1).$$

> [!example] Choosing the mass of each class / 选择每个闭类的概率质量
> <!-- bilingual-en:start -->
> Allocate a fraction $a$ of probability to the moving pair and $1-a$ to the absorbing state, with $0\le a\le1$. Then
> <!-- bilingual-en:end -->
>
> 给仍在运动的状态对分配概率 $a$，给吸收状态分配 $1-a$，其中 $0\le a\le1$。于是
>
> $$\pi=a\pi^{(1)}+(1-a)\pi^{(2)}=(0,a/3,2a/3,1-a).$$
>
> <!-- bilingual-en:start -->
> For example, $a=3/4$ gives $(0,1/4,1/2,1/4)$. Every such row is stationary. There is no stationary mass at $0$: it can send probability out but receives none back. The allowed values of $a$ describe all stationary distributions, not the limit selected by a specified initial state.
> <!-- bilingual-en:end -->
>
> 例如，$a=3/4$ 给出 $(0,1/4,1/2,1/4)$。这些概率行全部平稳。状态 $0$ 不承载平稳质量：它能向外输出概率，却收不到返回的概率。这里允许的所有 $a$ 描述全部平稳分布，还没有指定某个初态会选择哪个极限。

<!-- bilingual-en:start -->
## Every stationary law is a mixture of class laws
<!-- bilingual-en:end -->

*全部平稳分布都是类内平稳律的混合*

<!-- bilingual-en:start -->
A [[CTMC平稳分布|stationary distribution]] is a probability row $\pi$ satisfying $\pi P(t)=\pi$ for every $t\ge0$. In the finite setting this is equivalent to [[有限CTMC平稳生成矩阵判据|the balance equation]] $\pi Q=0$. List all closed communicating classes as $C_1,\ldots,C_m$, with $m\ge1$. Each restricted chain is finite and irreducible, so it has a [[有限不可约CTMC稳态|unique stationary probability law]], positive on that class. Write $\pi^{(r)}$ for that law extended by zero to $S\setminus C_r$.
<!-- bilingual-en:end -->

[[CTMC平稳分布|平稳分布]]是对每个 $t\ge0$ 都满足 $\pi P(t)=\pi$ 的概率行。在有限状态范围，这等价于[[有限CTMC平稳生成矩阵判据|平衡方程]] $\pi Q=0$。列出全部闭沟通类 $C_1,\ldots,C_m$，其中 $m\ge1$。每个限制后的链都有限且不可约，因此有[[有限不可约CTMC稳态|唯一平稳概率律]]，在该类上处处为正。记 $\pi^{(r)}$ 为这一概率律，并在 $S\setminus C_r$ 上补零。

> [!important] The complete decomposition / 完整分解
> <!-- bilingual-en:start -->
> A probability row $\pi$ is stationary exactly when it can be written as
> <!-- bilingual-en:end -->
>
> 概率行 $\pi$ 平稳，当且仅当它能写为
>
> $$\pi=\sum_{r=1}^m a_r\pi^{(r)},\qquad a_r\ge0,\qquad \sum_{r=1}^m a_r=1.$$
>
> <!-- bilingual-en:start -->
> These weights are unique: $a_r=\pi(C_r)=\sum_{i\in C_r}\pi_i$. All other states have stationary mass zero. Consequently, the full chain has exactly one stationary distribution if and only if it has exactly one closed communicating class; it may still have transient states outside that class.
> <!-- bilingual-en:end -->
>
> 权重唯一：$a_r=\pi(C_r)=\sum_{i\in C_r}\pi_i$。其余状态的平稳质量全部为零。因此，全链平稳分布唯一，当且仅当只有一个闭沟通类；这个闭类外仍然可以有暂态状态。

<!-- bilingual-en:start -->
The stationary laws form a **simplex**: each $\pi^{(r)}$ is a corner, and all other stationary laws are the probability-weighted mixtures of these corners. The disjoint class supports make the weights recoverable from the resulting law. This is not a vector space: multiplying a stationary probability row by two loses total mass one, and subtracting two class laws produces negative entries.
<!-- bilingual-en:end -->

全部平稳律构成一个**单纯形**：每个 $\pi^{(r)}$ 是一个顶点，其余平稳律都是这些顶点的概率加权混合。类的支撑互不重叠，所以能从混合结果恢复各权重。这不是向量空间：把概率行乘二会破坏总质量一；把两个类内平稳律相减，还会产生负条目。

<!-- bilingual-en:start -->
## Why transient states cannot carry stationary mass
<!-- bilingual-en:end -->

*为什么暂态状态不能承载平稳质量*

<!-- bilingual-en:start -->
Use [[CTMC均匀化|uniformization]] to compare with a discrete chain. Let $q_i=-q_{ii}$ be the exit rate and choose a finite $\nu>\max_i q_i$. With $I$ the [[单位矩阵|identity matrix]], set $R=I+Q/\nu$. Then $R$ is row-stochastic, has positive diagonal entries, and has exactly the same off-diagonal positive routes as $Q$. Thus it has the same communicating and closed classes. Moreover,
<!-- bilingual-en:end -->

用[[CTMC均匀化|均匀化]]与离散链比较。令 $q_i=-q_{ii}$ 为出口率，选有限的 $\nu>\max_i q_i$。以 $I$ 表示[[单位矩阵]]，令 $R=I+Q/\nu$。$R$ 是行随机矩阵，对角项为正，非对角正路线与 $Q$ 完全相同，因此沟通类和闭类也相同。并且

$$\pi R=\pi\quad\Longleftrightarrow\quad \pi Q=0.$$

<!-- bilingual-en:start -->
The [[有限链类分解|finite-chain class decomposition]] says that the discrete chain enters some closed class in finite steps with probability one. Let $\mathcal T$ be the states outside those classes. Then $\Pr_i(Y_n\in\mathcal T)\to0$ for every start $i$, where $Y_n$ follows $R$. If $\pi$ is stationary, however, its probability of being in $\mathcal T$ stays equal to $\pi(\mathcal T)$ at every step. Averaging the preceding limit over the finitely many initial states gives $\pi(\mathcal T)=0$.
<!-- bilingual-en:end -->

由[[有限链类分解]]，这条离散链以概率一在有限步内进入某个闭类。设 $\mathcal T$ 为所有闭类之外的状态，$Y_n$ 按 $R$ 转移，则对每个初态 $i$ 都有 $\Pr_i(Y_n\in\mathcal T)\to0$。但若从平稳律 $\pi$ 出发，每一步位于 $\mathcal T$ 的概率都等于 $\pi(\mathcal T)$。对有限多个初态把前面的极限加权求和，便得 $\pi(\mathcal T)=0$。

<!-- bilingual-en:start -->
No closed class can send mass to another. Once the transient mass is zero, the balance equation restricted to a class with $a_r>0$, divided by $a_r$, is a stationary probability equation on that class. Its unique solution is the class law. A class with $a_r=0$ contributes nothing. Conversely, each class law extended by zero remains stationary for the full chain, and any probability-weighted mixture remains stationary by linearity. This proves the [[有限链平稳分解|same decomposition structure]] in continuous time without assuming global convergence first.
<!-- bilingual-en:end -->

闭类不能向另一个闭类输出质量。暂态质量为零以后，把平衡方程限制在 $a_r>0$ 的类上，再除以 $a_r$，就得到该类上的平稳概率方程；其唯一解就是类内平稳律。$a_r=0$ 的类没有贡献。反过来，类内平稳律在类外补零后仍对全链平稳，按概率加权混合后也因线性而平稳。这就把[[有限链平稳分解|相同的分解结构]]带到连续时间，不必先假定全链收敛。

<!-- bilingual-en:start -->
As a simpler special case, if state $0$ can jump to either absorbing state $1$ or $2$, the class laws are $\delta_1$ and $\delta_2$, where $\delta_j$ puts all probability at $j$. Every $a\delta_1+(1-a)\delta_2$ is stationary. Conversely, three transient states feeding one closed class still give only one stationary law. The [[有限CTMC全局收敛|long-time limit from a chosen initial law]] additionally determines the class weights through entry probabilities. On an infinite state space, a closed irreducible class need not have a stationary probability law at all; the finite hypothesis is essential here.
<!-- bilingual-en:end -->

更简单的特例是：状态 $0$ 可以跳向两个吸收状态 $1$、$2$，类内平稳律就是 $\delta_1$、$\delta_2$，其中 $\delta_j$ 表示全部概率集中在 $j$。任意 $a\delta_1+(1-a)\delta_2$ 都平稳。反过来，三个暂态状态通向一个闭类时，仍只有一个平稳律。若要找[[有限CTMC全局收敛|指定初始律下的长期极限]]，还要用进入各类的概率确定权重。无限状态下，闭的不可约类甚至可能没有平稳概率律，因此这里的有限假设不能省略。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [MIT 6.436J Lecture 24, PDF p. 7, finite-state paragraph](https://ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/087af3cedbc9def5b156c5e1665ac79c_MIT6_436JF18_lec24.pdf#page=7): existence for a finite CTMC, the unique closed recurrent-class criterion, and the separation of stationary and limiting laws.
- [Cambridge Applied Probability, Theorem 2.1, PDF pp. 20–21; Theorems 2.13, 2.15 and 2.17, PDF pp. 24–26](https://www.statslab.cam.ac.uk/~ps422/notes-new2015.pdf#page=20): positive-rate reachability, classwise invariant laws, and the finite-state equivalence between stationarity and $\pi Q=0$.
- [Levin–Peres with Wilmer, §1.7, Propositions 1.28–1.29 and Exercise 1.14, PDF pp. 31–34](https://pages.uoregon.edu/dlevin/MARKOV/mcmt2e.pdf#page=31): stationary mass on closed classes and their mixture structure. The uniformization reduction, class-weight recovery and four-state balance equations are derived explicitly above.
<!-- bilingual-en:end -->

- MIT 第 24 讲 PDF 第 7 页有限状态段：核对有限 CTMC 平稳律存在、唯一闭常返类判据，以及平稳律与极限律的区别。
- Cambridge 定理 2.1（PDF20–21）、2.13、2.15、2.17（PDF24–26）：核对正速率可达性、类内不变律，以及有限状态下平稳性与 $\pi Q=0$ 的等价。
- Levin–Peres 等第 1.7 节、命题 1.28–1.29 和习题 1.14（PDF31–34）：核对平稳质量所在闭类和混合结构；均匀化转换、权重恢复与四状态平衡方程在正文直接推导。

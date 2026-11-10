---
aliases:
  - CTMC communicating classes
  - Positive-rate accessibility
student_os: knowledge-atom
atom_id: PROB-CTMC-012
atom_set: continuous-time-markov-chains
atom_type: theorem
status: source-checked
requires:
  - "[[嵌入跳链]]"
  - "[[马氏链沟通类]]"
  - "[[闭沟通类]]"
  - "[[不可约链]]"
related:
  - "[[CTMC爆炸]]"
leads_to:
  - "[[常返与停留时间]]"
  - "[[有限不可约CTMC稳态]]"
part_of:
  - "[[连续时间马尔可夫链.canvas]]"
---

<!-- bilingual-en:start -->
# Communicating classes of a CTMC
<!-- bilingual-en:end -->

*连续时间链的沟通类*

<!-- bilingual-en:start -->
To decide which states can reach each other, first draw the possible routes and put the clock aside. A [[CTMC定义|continuous-time Markov chain]] can reach a state through several genuine jumps even when no direct jump connects the endpoints. Its [[嵌入跳链|embedded jump chain]] records exactly those destinations. For the minimal construction, both processes therefore have the same communicating classes: changing waiting times does not change which finite routes are possible.
<!-- bilingual-en:end -->

判断哪些状态能互相到达，可以先画出可能的路线，暂时放下时钟。[[CTMC定义|连续时间 Markov 链]]可以经过多次真实跳跃到达某状态，不要求起点与终点之间有直接跳跃。[[嵌入跳链]]恰好记录这些去向。对于最小构造，两者因此具有相同的沟通类：改变等待时长，不改变哪些有限路线可能发生。

> [!example] An indirect route is still a route · 间接到达也算可达
> <!-- bilingual-en:start -->
> Order three states as $1,2,3$, and take the row-convention [[生成矩阵约束|generator]]
> <!-- bilingual-en:end -->
>
> 按 $1,2,3$ 排列状态，采用行约定的[[生成矩阵约束|生成矩阵]]
>
> $$Q=\begin{pmatrix}-2&2&0\\0&-3&3\\0&0&0\end{pmatrix}.$$
>
> <!-- bilingual-en:start -->
> The only route is $1\to2\to3$. Its mean waits at 1 and 2 are $1/2$ and $1/3$ time unit; state 3 is absorbing. Although $q_{13}=0$, state 3 is accessible from 1 through state 2. No reverse route exists, so the [[马氏链沟通类|communicating classes]] are the three singletons $\{1\},\{2\},\{3\}$. Only $\{3\}$ is closed.
> <!-- bilingual-en:end -->
>
> 唯一的前进路线是 $1\to2\to3$。在 1、2 的平均等待分别为 $1/2$、$1/3$ 个时间单位，状态 3 吸收。虽然 $q_{13}=0$，从 1 仍能经 2 到达 3。反向路线都不存在，所以[[马氏链沟通类|沟通类]]是三个单点 $\{1\},\{2\},\{3\}$，其中只有 $\{3\}$ 闭合。

<!-- bilingual-en:start -->
## The precise route criterion

Let $S$ be a nonempty finite or countable state set. Use the time-homogeneous [[最小CTMC|minimal CTMC]] with a stable conservative generator $Q=(q_{ij})$: off-diagonal rates are nonnegative, and the [[出口率|exit rate]] $q_i=\sum_{j\ne i}q_{ij}=-q_{ii}$ is finite at each state. “Minimal” means that an exploded path goes to a cemetery state outside $S$. Write $p_{ij}(t)$ for its probability of being at $j\in S$ at time $t$ when started at $i$.

For $q_i>0$, its jump matrix has $r_{ij}=q_{ij}/q_i$ for $j\ne i$ and $r_{ii}=0$. If $q_i=0$, complete the discrete record with $r_{ii}=1$: no physical jump is created by this convention.
<!-- bilingual-en:end -->

*准确的路线判据*

设 $S$ 非空，有限或可数。采用时间齐次的[[最小CTMC|最小 CTMC]]，其稳定、保守生成矩阵为 $Q=(q_{ij})$：非对角率非负，每个状态的[[出口率]] $q_i=\sum_{j\ne i}q_{ij}=-q_{ii}$ 有限。“最小”表示爆炸后进入 $S$ 外的墓地状态。记 $p_{ij}(t)$ 为从 $i$ 出发，时刻 $t$ 仍在原空间的状态 $j$ 的概率。

若 $q_i>0$，跳链矩阵满足 $r_{ij}=q_{ij}/q_i$（$j\ne i$）、$r_{ii}=0$；若 $q_i=0$，用 $r_{ii}=1$ 补全离散记录，这个约定不产生真实跳跃。

> [!important] The clock changes probabilities, not accessibility · 时钟改变概率，不改变可达性
> <!-- bilingual-en:start -->
> For $i,j\in S$, the following are equivalent: the jump chain can reach $j$ from $i$; there is a finite positive-rate route from $i$ to $j$; $p_{ij}(t)>0$ for some $t>0$; and $p_{ij}(t)>0$ for every $t>0$.
> A positive-rate route means
> <!-- bilingual-en:end -->
>
> $$i=x_0,x_1,\ldots,x_m=j,\qquad
> x_{\ell+1}\ne x_\ell,\quad q_{x_\ell x_{\ell+1}}>0\quad(0\le\ell<m).$$
>
> <!-- bilingual-en:start -->
> Here $m$ is the number of jumps. Allow $m=0$ when $i=j$; this records self-accessibility without asserting departure and return.
> <!-- bilingual-en:end -->
>
> 对 $i,j\in S$，以下条件等价：跳链能从 $i$ 到达 $j$；存在上式的有限正速率路线；某个 $t>0$ 有 $p_{ij}(t)>0$；以及每个 $t>0$ 都有 $p_{ij}(t)>0$。$m$ 计数跳跃次数。$i=j$ 时允许 $m=0$，表示自身可达，并不声称已经离开又返回。

<!-- bilingual-en:start -->
## Why a route can fit inside any positive duration

Fix a route of $m\ge1$ jumps and any $t>0$. Each prescribed next destination has positive probability. Conditional on this finite destination sequence, the [[CTMC指数停留时间|holding times]] are independent exponentials with the corresponding exit rates. Require each of the first $m$ waits to be shorter than $t/(2m)$, and then require the final wait at $j$ to exceed $t$. These events jointly have positive probability: all jumps finish before $t/2$, and the path is still at $j$ at time $t$. The last requirement has probability $e^{-q_jt}>0$, also when $q_j=0$.

Conversely, a minimal path that is in $S$ at time $t$ has not exploded. It has made only finitely many genuine jumps, so it reached $j$ along a finite route. There are only countably many finite routes; a positive total probability requires a route with positive probability, hence positive rates on its edges. For $i=j$, simply making no jump gives $p_{ii}(t)\ge e^{-q_it}>0$.
<!-- bilingual-en:end -->

*为什么任意正时长都可能走完这条路线*

固定一条含 $m\ge1$ 次跳跃的路线，以及任意 $t>0$。每次指定去向都有正概率。给定这段有限的去向序列，各段[[CTMC指数停留时间|停留时间]]是相应出口率的独立指数变量。要求前 $m$ 段等待各自小于 $t/(2m)$，再要求到达 $j$ 后等待超过 $t$。这些事件同时发生的概率为正：全部跳跃在 $t/2$ 前完成，时刻 $t$ 仍在 $j$。最后一项概率为 $e^{-q_jt}>0$，$q_j=0$ 时也成立。

反过来，最小过程在时刻 $t$ 仍位于 $S$，就说明尚未爆炸，因此此前只有有限次真实跳跃，到达 $j$ 必经一条有限路线。有限路线只有可数多条，总概率为正就要求至少一条路线具有正概率，其各条边的速率必为正。对于 $i=j$，仅“不跳跃”便给出 $p_{ii}(t)\ge e^{-q_it}>0$。

<!-- bilingual-en:start -->
## What the graph tells us, and where it stops

States communicate when each is accessible from the other; the resulting maximal groups are the [[马氏链沟通类|communicating classes]]. A [[闭沟通类|closed class]] has no positive-rate edge to its complement in $S$. The chain is [[不可约链|irreducible]] when all of $S$ is one class. An absorbing state is a closed singleton. Thus these graph properties agree for the CTMC and its jump chain.

Multiplying every positive rate by $100$ preserves the jump probabilities and all these classes, while dividing each finite mean holding time by $100$. More generally, changing positive rate magnitudes while keeping their zero pattern preserves the classes, even if it changes jump probabilities.
<!-- bilingual-en:end -->

*图结构能告诉什么，不能告诉什么*

两状态双向可达就互通，由此得到的最大状态组是[[马氏链沟通类|沟通类]]。[[闭沟通类]]没有通向 $S$ 中类外状态的正速率边；整个 $S$ 是一个类时，链便[[不可约链|不可约]]。吸收态是闭单点类。因此，CTMC 与跳链的这些图结构相同。

把所有正速率同时乘 $100$，跳跃概率和类划分不变，各个有限平均停留时间变为原来的 $1/100$。更一般地，只要正率与零率的位置不变，改变正率大小也不改变类划分，即使去向概率会改变。

> [!warning] Closed in the graph does not exclude explosion · 图上闭合不等于排除爆炸
> <!-- bilingual-en:start -->
> A closed class prevents a finite genuine jump from leaving the class. On a countably infinite class it does not, by itself, prevent [[CTMC爆炸|infinitely many jumps in finite time]] and loss to the cemetery state. The criterion concerns the minimal process on $S$. A separately specified [[爆炸后延拓|post-explosion extension]] can introduce re-entry routes that are not finite positive-rate paths of $Q$.
> <!-- bilingual-en:end -->
>
> 闭类禁止通过一次真实跳跃离开类。在可数无限类上，这本身不排除[[CTMC爆炸|有限时间内无限次跳跃]]，以及随后进入墓地状态。判据针对 $S$ 上的最小过程；另行指定的[[爆炸后延拓]]可能加入不属于 $Q$ 的有限正率路线的重入行为。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [Berestycki and Sousi, Applied Probability, §2.1, Theorem 2.1, PDF pp. 20–21](https://www.statslab.cam.ac.uk/~ps422/notes-new2015.pdf#page=20): equivalence of minimal-CTMC accessibility, jump-chain accessibility, positive-rate routes and positive finite-time probabilities; the chapter explicitly uses finite or countable state spaces and minimal chains. The zero-step case and finite-route argument are spelled out above.
- [[马氏链沟通类|Communicating classes]], [[闭沟通类|closed classes]] and [[不可约链|irreducibility]] provide the graph definitions. The displayed three-state generator and the rate-rescaling example are checked directly.
<!-- bilingual-en:end -->

- Berestycki–Sousi §2.1，定理 2.1，PDF 第 20–21 页：最小 CTMC 可达、跳链可达、正率路线与有限时长正转移概率的等价；该章明确限定有限或可数状态及最小过程。正文补明零步情形和有限路线论证。
- [[马氏链沟通类]]、[[闭沟通类]]与[[不可约链]]给出图定义；三状态生成矩阵与统一速率缩放例直接核算。

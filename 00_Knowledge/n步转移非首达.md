---
aliases:
- n-step occupancy is not first passage
- n-step transition versus first passage
- n 步转移非首达
student_os: knowledge-atom
atom_id: PROB-DTMC-027
atom_set: discrete-time-markov-chains
atom_type: distinction
status: source-checked
requires:
- '[[DTMC转移复合]]'
- '[[Markov性]]'
related:
- '[[吸收链]]'
leads_to:
- '[[命中概率第一步方程]]'
- '[[期望命中时间方程]]'
part_of:
- '[[离散时间马尔可夫链.canvas]]'
---
<!-- bilingual-en:start -->
# Occupancy and first passage
<!-- bilingual-en:end -->
*第 n 步在目标，不等于第 n 步才第一次到达*

<!-- bilingual-en:start -->
“Is the machine failed after two hours?” and “Does it fail for the first time after exactly two hours?” are different questions. The first counts a machine that failed after one hour and stayed failed. The second excludes it. An **occupancy probability** checks the state at one chosen time; a **first-passage probability** also checks the earlier path.
<!-- bilingual-en:end -->

“两小时后设备是否故障”和“设备是否恰好在两小时后第一次故障”是两个问题。第一问会计入第一小时已经故障、随后保持故障的设备，第二问会排除它。**占据概率**只检查指定时刻的状态；**首达概率**还要检查之前经过的路径。

> [!example] The difference is already visible in two steps · 两步就能看出差别
> <!-- bilingual-en:start -->
> Use a time-homogeneous machine model with states working $W$, failed $F$, starting at $W$, and the [[Markov矩阵左右约定|row transition matrix]]
> <!-- bilingual-en:end -->
>
> 设备状态为正常 $W$、故障 $F$，从 $W$ 出发，每小时使用同一套[[Markov矩阵左右约定|行转移矩阵]]：
>
> $$P=\begin{pmatrix}0.9&0.1\\0.4&0.6\end{pmatrix}.$$
>
> <!-- bilingual-en:start -->
> At hour two, failure can arise through $W\to W\to F$ or $W\to F\to F$. Their probabilities are $0.09$ and $0.06$, so occupancy is $0.15$. Only the first path reaches $F$ for the first time at hour two, so the first-passage probability is $0.09$.
> <!-- bilingual-en:end -->
>
> 第二小时故障可以经过 $W\to W\to F$ 或 $W\to F\to F$，概率分别为 $0.09$ 与 $0.06$，因此占据概率是 $0.15$。只有第一条路径在第二小时才第一次到达 $F$，所以首达概率是 $0.09$。

<!-- bilingual-en:start -->
## What the two events require
<!-- bilingual-en:end -->
*两个事件分别检查什么*

<!-- bilingual-en:start -->
For a finite or countable [[离散时间Markov链|Markov chain]], $\Pr_i$ means probability when the starting state is fixed at $i$. For distinct states $i\ne j$, define the first hitting time
<!-- bilingual-en:end -->

对有限或可数状态的[[离散时间Markov链]]，$\Pr_i$ 表示从指定状态 $i$ 出发时的概率。对不同状态 $i\ne j$，定义首次命中时刻

$$\tau_j=\inf\{m\ge0:X_m=j\}.$$

<!-- bilingual-en:start -->
The infimum here selects the earliest nonnegative integer in the set. If the chain never reaches $j$, set $\tau_j=\infty$. In a [[DTMC时间齐次转移核|time-homogeneous]] model, for an integer $n\ge1$,
<!-- bilingual-en:end -->

这里的下确界就是集合中最早的非负整数；若永远到不了 $j$，约定 $\tau_j=\infty$。在[[DTMC时间齐次转移核|时间齐次]]模型中，对整数 $n\ge1$，

$$\begin{aligned}
p_{ij}^{(n)}&=\Pr_i(X_n=j),\\
f_{ij}^{(n)}&=\Pr_i(\tau_j=n)
=\Pr_i(X_1\ne j,\ldots,X_{n-1}\ne j,X_n=j).
\end{aligned}$$

<!-- bilingual-en:start -->
The first event imposes no restrictions on earlier visits. The second forbids them. Thus $f_{ij}^{(n)}\le p_{ij}^{(n)}$, but equality can occur—for example at $n=1$. The finite-time first-passage probabilities can sum to less than one; the missing probability is the chance of never hitting $j$.
<!-- bilingual-en:end -->

第一个事件不限制早先是否访问过目标，第二个事件禁止早先到访。因此 $f_{ij}^{(n)}\le p_{ij}^{(n)}$，但两者也可能相等，例如 $n=1$。所有有限时刻首达概率的和可以小于一，剩余概率就是永远无法命中的概率。

<!-- bilingual-en:start -->
Under the row convention, [[DTMC转移复合|transition composition]] gives $p_{ij}^{(n)}=(P^n)_{ij}$. Reading that matrix entry supplies occupancy. It includes early arrival followed by staying, and early arrival followed by leaving and returning. In the machine example, $W\to F\to W\to F$ contributes $0.1(0.4)(0.1)=0.004$ to occupancy at hour three, although the first failure was at hour one.
<!-- bilingual-en:end -->

行随机约定下，[[DTMC转移复合]]给出 $p_{ij}^{(n)}=(P^n)_{ij}$。读取这个矩阵元素得到的是占据概率：它包括早到后停留，也包括早到、离开、再返回。设备例子中的 $W\to F\to W\to F$ 为第三小时的占据概率贡献 $0.1(0.4)(0.1)=0.004$，但第一次故障发生在第一小时。

<!-- bilingual-en:start -->
## Split occupancy by the first arrival
<!-- bilingual-en:end -->
*按第一次到达的时点拆开占据概率*

<!-- bilingual-en:start -->
If a path starting away from $j$ is at $j$ at time $n$, it must have first reached $j$ at exactly one time $m\in\{1,\ldots,n\}$. Split into these disjoint cases. Given a positive-probability event $\tau_j=m$, the state at time $m$ is $j$. The [[Markov性|Markov property]] at this fixed time makes the next $n-m$ steps depend on that state rather than the earlier avoidance history. Homogeneity makes their return probability $p_{jj}^{(n-m)}$. Hence
<!-- bilingual-en:end -->

一条从 $j$ 之外出发、在时刻 $n$ 位于 $j$ 的路径，必定在某个唯一的 $m\in\{1,\ldots,n\}$ 第一次到达 $j$。按这些互斥情况拆开：当事件 $\tau_j=m$ 的概率为正时，时刻 $m$ 的状态就是 $j$；在这个固定时点应用 [[Markov性]]，后续 $n-m$ 步就只依赖当前状态，不再依赖此前如何避开目标。齐次性使这段返回概率等于 $p_{jj}^{(n-m)}$。因此

$$p_{ij}^{(n)}=\sum_{m=1}^{n}f_{ij}^{(m)}p_{jj}^{(n-m)}\qquad(i\ne j).$$

<!-- bilingual-en:start -->
Zero-probability first-arrival cases contribute zero. The final term uses $p_{jj}^{(0)}=1$: if the first arrival is at time $n$, no further transition is required. For the two-hour example, the terms are $0.1(0.6)+0.09(1)=0.15$. This also gives a recursive way to recover first-passage probabilities from occupancy:
<!-- bilingual-en:end -->

零概率的首达情况贡献为零。最后一项使用 $p_{jj}^{(0)}=1$：若在时刻 $n$ 才首达，之后不需要再转移。两小时例子各项为 $0.1(0.6)+0.09(1)=0.15$。它也给出了由占据概率递推首达概率的方法：

$$f_{ij}^{(n)}=p_{ij}^{(n)}-\sum_{m=1}^{n-1}f_{ij}^{(m)}p_{jj}^{(n-m)}.$$

<!-- bilingual-en:start -->
For a nonhomogeneous Markov chain, the event distinction remains the same, but the continuation probability is the transition from calendar time $m$ to $n$, not a function of $n-m$ alone. The displayed homogeneous convolution cannot be copied without that change.
<!-- bilingual-en:end -->

非齐次 Markov 链也有同样的事件区别，但后半段应使用从日历时刻 $m$ 到 $n$ 的转移概率，不能只由步数差 $n-m$ 决定。上面的齐次卷积公式不能不加修改地照搬。

> [!warning] Starting at the target and absorbing targets · 起点就是目标，以及吸收目标
> <!-- bilingual-en:start -->
> If $i=j$, the hitting time defined above is already zero. To ask about a subsequent return, define instead $\tau_j^+=\inf\{m\ge1:X_m=j\}$; this is the first positive return time. Keep this convention separate from first hitting including time zero.
> <!-- bilingual-en:end -->
>
> 若 $i=j$，上面定义的首次命中时刻已经是零。要问之后何时第一次返回，应另定义 $\tau_j^+=\inf\{m\ge1:X_m=j\}$，即首次正时刻回返；它与包含时刻零的首次命中不是同一约定。
>
> <!-- bilingual-en:start -->
> If $j$ is [[吸收状态|absorbing]], entry is permanent, so $\Pr_i(X_n=j)=\Pr_i(\tau_j\le n)$: occupancy becomes the cumulative first-arrival probability, not the probability of arriving exactly at $n$. In particular, if state $0$ moves to absorbing state $1$ in one step with certainty, then $p_{01}^{(3)}=1$ but $f_{01}^{(3)}=0$.
> <!-- bilingual-en:end -->
>
> 若 $j$ 是[[吸收状态]]，进入以后不会离开，于是 $\Pr_i(X_n=j)=\Pr_i(\tau_j\le n)$：占据概率成为截至 $n$ 的累计首达概率，仍不是恰在 $n$ 首达的概率。特别地，若状态 $0$ 一步必到吸收状态 $1$，则 $p_{01}^{(3)}=1$，但 $f_{01}^{(3)}=0$。

<!-- bilingual-en:start -->
The probability of ever reaching the target and the expected waiting time are further questions, handled by [[命中概率第一步方程|hitting-probability equations]] and [[期望命中时间方程|expected-hitting-time equations]]. Neither is obtained merely by reading one matrix-power entry.
<!-- bilingual-en:end -->

最终能否命中、平均要等多久，是另外两个问题，分别由[[命中概率第一步方程]]和[[期望命中时间方程]]处理，不能只读一个矩阵幂元素来回答。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [Gallager, MIT 6.262 Chapter 5, §5.1.1, PDF pp. 4–5 / printed pp. 231–232](https://ocw.mit.edu/courses/6-262-discrete-stochastic-processes-spring-2011/01d0892549619cb25d928f15ec7230ed_MIT6_262S11_chap05.pdf#page=4): Definition 5.1.1 distinguishes first subsequent entry from occupancy; the following page distinguishes exact-time and cumulative probabilities. Its positive-time convention is separated here from hitting at time zero.
- [Weber, Cambridge Markov Chains, §§3.2 and 4.4, PDF pp. 13 and 19](https://www.statslab.cam.ac.uk/~rrw1/markov/M.pdf#page=13): supports the hitting-time convention, infinity when no hit occurs, and restarting from a hit state. The finite sum here is derived by splitting over fixed possible first-arrival times and checked directly in the displayed examples.
<!-- bilingual-en:end -->

- Gallager 第 5.1.1 节（PDF 第 4–5 页，印刷第 231–232 页）定义 5.1.1 区分正时刻首次到达与占据，下一页区分恰时和累计概率；本卡明确区分其正时刻约定与包含时刻零的命中约定。
- Weber 第 3.2、4.4 节（PDF 第 13、19 页）支持命中时刻、从不命中时取无穷，以及到达某状态后的重新开始。本文有限求和按每个固定首达时点直接推导，展示例子逐项复算。

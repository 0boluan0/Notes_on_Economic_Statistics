---
aliases:
- CTMC Chapman-Kolmogorov equation
- Transition semigroup
- CTMC transition semigroup
- Continuous-time Chapman-Kolmogorov equation
student_os: knowledge-atom
atom_id: PROB-CTMC-007
atom_set: continuous-time-markov-chains
atom_type: theorem
status: source-checked
requires:
- '[[CTMC定义]]'
- '[[CTMC时间齐次转移函数]]'
related:
- '[[DTMC转移复合]]'
leads_to:
- '[[Kolmogorov前后向方程]]'
- '[[CTMC矩阵指数]]'
- '[[CTMC平稳分布]]'
part_of:
- '[[连续时间马尔可夫链.canvas]]'
---

<!-- bilingual-en:start -->
# CTMC transition semigroup
<!-- bilingual-en:end -->
*连续时间转移半群与 Chapman–Kolmogorov 方程*

<!-- bilingual-en:start -->
To find where a continuous-time Markov chain will be after two hours, first consider where it can be after one hour. From each possible intermediate state, continue for another hour and add the contributions. The **transition semigroup** expresses this way of joining time intervals. It describes observation times; it does not require the chain to jump on an hourly grid.
<!-- bilingual-en:end -->

要计算连续时间 Markov 链两小时后在哪里，可以先考虑一小时后可能处于哪些状态，再从各个中间状态继续运行一小时，把概率贡献加起来。**转移半群**表达的就是这种时间区间拼接方式。它在分解观测时点，并没有要求链只能在整点跳跃。

> [!example] Failure during either hour · 两个小时中任意一段故障
> <!-- bilingual-en:start -->
> Take a [[CTMC时间齐次转移函数|homogeneous device model]] with states working $W$, failed $F$, exponential time to failure, and no repair. Suppose its one-hour survival probability is $1/2$. Then
> <!-- bilingual-en:end -->
>
> 考虑一个[[CTMC时间齐次转移函数|齐次设备模型]]：状态为正常 $W$、故障 $F$，等待指数时间后故障且不再维修。设存活一小时的概率为 $1/2$，则
>
> $$P(1)=\begin{pmatrix}1/2&1/2\\0&1\end{pmatrix}.$$
>
> <!-- bilingual-en:start -->
> Starting at $W$, the device can be failed at hour two by still working at hour one and failing afterwards, or by already being failed at hour one. These cases are disjoint:
> <!-- bilingual-en:end -->
>
> 从 $W$ 出发，两小时后处于故障状态，可能是一小时后仍正常、随后故障，也可能是一小时后已经故障。两种情况互斥：
>
> $$p_{WF}(2)=\underbrace{\tfrac12\cdot\tfrac12}_{W\text{ at hour }1}
> +\underbrace{\tfrac12\cdot1}_{F\text{ at hour }1}=\tfrac34.$$
>
> <!-- bilingual-en:start -->
> The intermediate state labels what is seen at hour one, not the exact failure time within either hour. The same reasoning works with any two nonnegative lengths.
> <!-- bilingual-en:end -->
>
> 中间状态记录一小时那个时点看到了什么，而不指定任一小时内部的确切故障时刻。同一思路适用于任意两段非负时长。

<!-- bilingual-en:start -->
## From the intermediate-state sum to a matrix product
<!-- bilingual-en:end -->
*从中间状态求和到矩阵乘积*

<!-- bilingual-en:start -->
Consider a time-homogeneous [[CTMC定义|CTMC]] on a finite or countable state set $S$. First suppose it retains all probability in $S$ at every finite time. Write $\Pr_i$ for its law when started at $i$, and $p_{ij}(t)=\Pr_i(X_t=j)$. Here $i$ is the departure state, $j$ the destination, and $t\ge0$ an elapsed time. Starting at a specified state defines the transition law even when that state had probability zero under a different initial distribution.
<!-- bilingual-en:end -->

考虑有限或可数状态集 $S$ 上的时间齐次 [[CTMC定义|CTMC]]，先假设全部概率都保留在这个集合中。记 $\Pr_i$ 为指定从 $i$ 出发的概率律，$p_{ij}(t)=\Pr_i(X_t=j)$。这里 $i$ 是出发状态，$j$ 是到达状态，$t\ge0$ 是经过时长。指定一个出发状态就能定义转移律，不必要求该状态在另一初始分布下具有正概率。

<!-- bilingual-en:start -->
At the intermediate time $s$, the events $X_s=k$ for $k\in S$ partition all possibilities. The [[全概率公式|law of total probability]] splits the event of ending at $j$ according to these cases. For a positive-probability case, [[Markov性|the Markov property]] removes the earlier history from the continuation, and homogeneity removes the starting clock time. Consequently,
<!-- bilingual-en:end -->

在中间时刻 $s$，事件 $X_s=k$（$k\in S$）把全部可能情况分开。[[全概率公式]]允许按这些情况拆开最终处于 $j$ 的事件。对具有正概率的情况，[[Markov性]]让后半段不再依赖更早历史，齐次性又去掉起始日历时点的影响。因此

$$\begin{aligned}
p_{ij}(s+t)
&=\sum_{k\in S}\Pr_i(X_s=k,\ X_{s+t}=j)\\
&=\sum_{k\in S}p_{ik}(s)p_{kj}(t),\qquad s,t\ge0.
\end{aligned}$$

<!-- bilingual-en:start -->
Each term means “reach $k$ over length $s$, then reach $j$ over length $t$”. A case with zero probability contributes zero through the intersection expression; no division by zero is needed. If $S$ is countably infinite, this is a countable sum of nonnegative terms, bounded by one.
<!-- bilingual-en:end -->

每一项表示“先经过 $s$ 到 $k$，再经过 $t$ 到 $j$”。零概率情况在交集表达式中贡献零，不需要除以零。$S$ 可数无穷时，这是非负项的可数和，总和不超过一。

> [!abstract] Chapman–Kolmogorov semigroup identity · 半群恒等式
> <!-- bilingual-en:start -->
> Arrange $p_{ij}(t)$ in a matrix $P(t)$, with the starting state in the row. The row-by-column sum is exactly [[矩阵乘法|matrix multiplication]], so
> <!-- bilingual-en:end -->
>
> 把 $p_{ij}(t)$ 排成矩阵 $P(t)$，用行固定出发状态。逐行逐列的求和正是[[矩阵乘法]]，因此
>
> $$P(s+t)=P(s)P(t),\qquad P(0)=I.$$
>
> <!-- bilingual-en:start -->
> $I$ is the [[单位矩阵|identity]]: at zero elapsed time every state is unchanged. For infinitely many states, the displayed product means the entrywise kernel sum just proved, rather than an appeal to finite-dimensional matrix theory.
> <!-- bilingual-en:end -->
>
> $I$ 是[[单位矩阵|恒等矩阵]]：零时长后每个状态仍是自身。状态无穷时，上式乘积表示刚证明的逐元素核求和，不能直接搬用有限维矩阵理论。

<!-- bilingual-en:start -->
The parameter set is nonnegative time, and combining intervals is associative. This is why the family is called a semigroup. It need not contain a valid reverse-time transition for each member. In the device example, $P(1)$ does have an ordinary [[逆矩阵|inverse]], $\begin{pmatrix}2&-1\\0&1\end{pmatrix}$, but its negative entry prevents it from being a probability transition matrix. Thus an algebraic inverse, when available, is not automatically an operation for running the stochastic model backwards.
<!-- bilingual-en:end -->

参数取非负时间，区间拼接满足结合律，因此这族对象称为半群。它不要求为每个成员都配备合法的逆向时间转移。设备例子中的 $P(1)$ 确实存在通常的[[逆矩阵]] $\begin{pmatrix}2&-1\\0&1\end{pmatrix}$，但其中有负数，不能当作概率转移矩阵。因此，即使代数逆存在，也不自动表示把随机模型倒着运行。

<!-- bilingual-en:start -->
## Keep time order and vector direction together
<!-- bilingual-en:end -->
*时间顺序与向量方向要一起保持*

<!-- bilingual-en:start -->
With the [[Markov矩阵左右约定|row convention]], a probability row $\mu$ evolves as $\mu P(s)P(t)$: the first interval is on the left. With probability columns, define $C(t)=P(t)^T$, where $T$ denotes [[矩阵转置|transpose]]. Then the update is $C(t)C(s)x$ for the initial column $x$: the first interval acts nearest to $x$. Members of this one homogeneous family commute, since both $P(s)P(t)$ and $P(t)P(s)$ equal $P(s+t)$. Unrelated transition matrices need not commute.
<!-- bilingual-en:end -->

采用[[Markov矩阵左右约定|行约定]]时，概率行 $\mu$ 更新为 $\mu P(s)P(t)$，先发生的区间在左。若采用概率列，令 $C(t)=P(t)^T$，其中 $T$ 表示[[矩阵转置|转置]]；初始列 $x$ 的更新为 $C(t)C(s)x$，先发生的区间要靠近 $x$。同一个齐次族中的成员可交换，因为 $P(s)P(t)$ 与 $P(t)P(s)$ 都等于 $P(s+t)$；不相关的转移矩阵则未必可交换。

<!-- bilingual-en:start -->
For a nonhomogeneous chain, let $P(r,t)$ describe the interval from clock time $r$ to $t$. The same conditioning argument gives, for $r\le u\le t$,
<!-- bilingual-en:end -->

对非齐次链，令 $P(r,t)$ 描述从日历时点 $r$ 到 $t$ 的区间。同样按中间状态条件分解，对 $r\le u\le t$ 可得

$$P(r,t)=P(r,u)P(u,t).$$

<!-- bilingual-en:start -->
This two-parameter composition law does not usually reduce to a single function of $t-r$. For example, allow only failures in the first hour and only repairs in the second, each with one-hour survival probability $1/2$ in the state that can leave. The interval matrices and their ordered product are
<!-- bilingual-en:end -->

这种两参数复合规律通常不能简化为只依赖 $t-r$ 的单一函数。例如第一小时只允许故障，第二小时只允许维修；在允许离开的状态中，停留满一小时的概率都为 $1/2$。两段矩阵及按顺序的乘积为

$$A=\begin{pmatrix}1/2&1/2\\0&1\end{pmatrix},\quad
B=\begin{pmatrix}1&0\\1/2&1/2\end{pmatrix},\quad
AB=\begin{pmatrix}3/4&1/4\\1/2&1/2\end{pmatrix}.$$

<!-- bilingual-en:start -->
Reversing the regimes gives $BA=\begin{pmatrix}1/2&1/2\\1/4&3/4\end{pmatrix}$, a different answer. Transposing to the column convention also reverses the written product: $(AB)^T=B^TA^T$.
<!-- bilingual-en:end -->

若把两段规则倒过来，得到 $BA=\begin{pmatrix}1/2&1/2\\1/4&3/4\end{pmatrix}$，结果不同。换成列约定时，转置也要把书写次序倒过来：$(AB)^T=B^TA^T$。

<!-- bilingual-en:start -->
## What happens to probability at explosion?
<!-- bilingual-en:end -->
*爆炸时的概率质量怎样处理？*

<!-- bilingual-en:start -->
For a [[最小CTMC|minimal chain]] killed only at its explosion time $\zeta$, retain just the original states $S$ in $P(t)$. Then
<!-- bilingual-en:end -->

对只在爆炸时刻 $\zeta$ 被杀死的[[最小CTMC|最小链]]，若 $P(t)$ 只列原状态集 $S$，则

$$\sum_{j\in S}p_{ij}(t)=\Pr_i(t<\zeta),\qquad
1-\sum_{j\in S}p_{ij}(t)=\Pr_i(\zeta\le t).$$

<!-- bilingual-en:start -->
The semigroup identity still holds: a path already killed at the splitting time cannot contribute to a later destination in $S$. The rows are now allowed to sum to less than one, giving a subprobability transition semigroup. If the cemetery state is included as an absorbing state, total probability is again one. A nonexplosive model without killing has unit row sums throughout. These statements concern a specified process; they do not by themselves supply a generator, differentiability, or a rule for [[爆炸后延拓|returning after explosion]].
<!-- bilingual-en:end -->

半群等式仍然成立：在分割时刻已经被杀死的路径，不会再对后来处于 $S$ 中某状态的概率作贡献。此时允许行和小于一，得到次概率转移半群。若把墓地状态作为吸收态列入，总概率又为一；不爆炸且没有杀死机制的模型始终具有单位行和。这些结论针对已经规定好的过程，本身并没有额外提供生成矩阵、可微性或[[爆炸后延拓|爆炸后返回]]规则。

<!-- bilingual-en:start -->
The formula $P(2t)=P(t)^2$ is therefore a way to split an observation interval, not a restriction on jump times or jump counts within it. [[Kolmogorov前后向方程|Forward and backward equations]] and [[CTMC矩阵指数|matrix-exponential formulas]] build on this composition identity with additional rate and regularity assumptions.
<!-- bilingual-en:end -->

因此，$P(2t)=P(t)^2$ 是拆分观测区间的方法，不是在限制区间内部的跳时或跳跃次数。[[Kolmogorov前后向方程|前向、后向方程]]与[[CTMC矩阵指数|矩阵指数公式]]在这个复合恒等式之上，再使用相应的速率和正则条件。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [Whitt, Continuous-Time Markov Chains, Lemma 2.1, PDF p.4 / printed p.3](https://www.columbia.edu/~ww2040/4106S11/CTMCchapter121906.pdf#page=4): intermediate-state conditioning, Chapman–Kolmogorov sums, the matrix semigroup and row-law propagation. The nonhomogeneous version follows by retaining both clock times in the same proof; the examples and column products are checked directly.
- [Cambridge Applied Probability (2015), §1.1 PDF p.3; §1.2 p.5; §§1.6–1.8 pp.12–19](https://www.statslab.cam.ac.uk/~ps422/notes-new2015.pdf#page=3): the semigroup on a countable state space, explosion/cemetery conventions and minimal transition probabilities. Omitting the absorbing cemetery state gives the survival row sum displayed above.
<!-- bilingual-en:end -->

- Whitt 引理2.1（PDF第4页，印刷第3页）核对按中间状态条件分解、Chapman–Kolmogorov 求和、矩阵半群与概率行更新。非齐次形式在同一证明中保留两个日历时点即可得到；算例与列约定乘积直接核算。
- Cambridge 第1.1节（PDF第3页）、第1.2节（第5页）、第1.6–1.8节（第12–19页）核对可数状态半群、爆炸与墓地约定、最小转移概率；省略吸收墓地状态后，行和就是正文中的存活概率。

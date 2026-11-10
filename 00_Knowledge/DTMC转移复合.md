---
aliases:
- Chapman-Kolmogorov equation
- n-step transition probabilities
- C-K 方程
student_os: knowledge-atom
atom_id: PROB-DTMC-004
atom_set: discrete-time-markov-chains
atom_type: identity
status: source-checked
requires:
- '[[DTMC时间齐次转移核]]'
- '[[Markov矩阵左右约定]]'
- '[[全概率公式]]'
related:
- '[[n步转移非首达]]'
- '[[CTMC转移半群]]'
leads_to:
- '[[马氏链沟通类]]'
- '[[状态周期]]'
part_of:
- '[[离散时间马尔可夫链.canvas]]'
---
<!-- bilingual-en:start -->
# Chapman–Kolmogorov composition
<!-- bilingual-en:end -->
*把较短的转移区间拼成较长区间*

<!-- bilingual-en:start -->
To find where a Markov chain can be after two steps, list where it could be after the first step. For each intermediate state, multiply the chance of reaching it by the chance of continuing to the destination. Then add these contributions. The **Chapman–Kolmogorov equation** applies this same idea to two intervals of any lengths.
<!-- bilingual-en:end -->

要算 Markov 链两步后可能在哪里，先列出第一步后可能处于的状态。对每个中间状态，把“到这里的概率”乘“从这里继续到目标的概率”，最后把这些贡献相加。**Chapman–Kolmogorov 方程**把同一个思路用于任意两段步数。

> [!example] Two routes to failure · 两条通向故障的路线
> <!-- bilingual-en:start -->
> Order a machine's states as working $W$, failed $F$. Use the same [[DTMC时间齐次转移核|transition rule]] each hour, with rows giving the current state and columns the next state:
> <!-- bilingual-en:end -->
>
> 把设备状态按正常 $W$、故障 $F$ 排列，每小时使用同一套[[DTMC时间齐次转移核|转移规则]]；行表示当前状态，列表示下一状态：
>
> $$P=\begin{pmatrix}0.9&0.1\\0.4&0.6\end{pmatrix}.$$
>
> <!-- bilingual-en:start -->
> Starting at $W$, being at $F$ two hours later can happen through $W\to W\to F$ or $W\to F\to F$. These alternatives are disjoint, so their probabilities add:
> <!-- bilingual-en:end -->
>
> 从 $W$ 出发，两小时后在 $F$，可能经过 $W\to W\to F$，也可能经过 $W\to F\to F$。两条路线互斥，因此概率相加：
>
> $$p_{WF}^{(2)}=0.9\times0.1+0.1\times0.6=0.15.$$
>
> <!-- bilingual-en:start -->
> The second route already reached the target after one hour. This calculation asks where the machine is at hour two, not when it [[n步转移非首达|first reached]] the target.
> <!-- bilingual-en:end -->
>
> 第二条路线在第一小时就已经到过目标。这里计算的是第二小时在哪里，不是[[n步转移非首达|第一次何时到达]]。

<!-- bilingual-en:start -->
## The intermediate-state sum
<!-- bilingual-en:end -->
*按中间状态分解*

<!-- bilingual-en:start -->
Consider a time-homogeneous [[离散时间Markov链|Markov chain]] on a finite or countable state set $S$. Write $\Pr_i$ for the law when we start the chain at state $i$, and
<!-- bilingual-en:end -->

考虑有限或可数状态集 $S$ 上的时间齐次[[离散时间Markov链]]。用 $\Pr_i$ 表示把初始状态指定为 $i$ 时的概率律，并记

$$p_{ij}^{(n)}=\Pr_i(X_n=j).$$

<!-- bilingual-en:start -->
Here $n$ is a nonnegative integer number of steps, $i$ is the start, and $j$ is the destination. Specifying a starting law avoids trying to condition on an initial state that had probability zero under some other law. Under a given law, the equivalent expression $\Pr(X_{r+n}=j\mid X_r=i)$ is used when $\Pr(X_r=i)>0$; homogeneity removes dependence on the starting time $r$.
<!-- bilingual-en:end -->

这里 $n$ 是非负整数步数，$i$ 是起点，$j$ 是终点。直接指定从 $i$ 出发，可以避免在另一初始分布的零概率状态上硬做条件除法。在某个给定概率律下，若 $\Pr(X_r=i)>0$，也可以写成 $\Pr(X_{r+n}=j\mid X_r=i)$；齐次性使它不依赖起始时点 $r$。

> [!abstract] Chapman–Kolmogorov equation · 转移复合公式
> <!-- bilingual-en:start -->
> For nonnegative integers $m,n$,
> <!-- bilingual-en:end -->
>
> 对非负整数 $m,n$，有
>
> $$p_{ij}^{(m+n)}=\sum_{k\in S}p_{ik}^{(m)}p_{kj}^{(n)}.$$
>
> <!-- bilingual-en:start -->
> Each term follows a route through state $k$ at the splitting time: $m$ steps to $k$, then $n$ steps to $j$.
> <!-- bilingual-en:end -->
>
> 每一项对应在分割时刻经过状态 $k$：先用 $m$ 步到 $k$，再用 $n$ 步到 $j$。

<!-- bilingual-en:start -->
At time $m$, exactly one of the events $X_m=k$ occurs. The [[全概率公式|law of total probability]] therefore splits the destination event into these cases. In a case of positive probability, the [[Markov性|Markov property]] removes the earlier history from the law of the next $n$ steps, and homogeneity makes that continuation probability $p_{kj}^{(n)}$. A zero-probability intermediate case contributes zero; no undefined conditional probability is multiplied by zero. This proves the identity. If $S$ is infinite, the sum is a countable sum of nonnegative contributions bounded by one.
<!-- bilingual-en:end -->

在时刻 $m$，事件 $X_m=k$ 恰好有一个发生，因此[[全概率公式]]允许按这些情况拆开终点事件。对具有正概率的情况，[[Markov性]]使后续 $n$ 步的规律不再依赖更早历史，齐次性又使这段概率等于 $p_{kj}^{(n)}$。零概率的中间情况贡献为零，不需要把一个未定义的条件概率乘零。这样便得到恒等式；若 $S$ 无穷，则求和是非负项的可数和，总和不超过一。

<!-- bilingual-en:start -->
## Matrix multiplication and its direction
<!-- bilingual-en:end -->
*矩阵乘法与方向约定*

<!-- bilingual-en:start -->
For finitely many states, use the [[Markov矩阵左右约定|row-stochastic convention]] $P_{ij}=p_{ij}^{(1)}$. A row of the first matrix and a column of the second produce exactly the sum above; this is [[矩阵乘法|matrix multiplication]]. If $P^{(n)}$ denotes the matrix of $n$-step probabilities, then
<!-- bilingual-en:end -->

状态有限时，采用[[Markov矩阵左右约定|行随机约定]] $P_{ij}=p_{ij}^{(1)}$。第一个矩阵的一行与第二个矩阵的一列相乘求和，恰好就是上式，这正是[[矩阵乘法]]。若 $P^{(n)}$ 表示 $n$ 步转移矩阵，则

$$P^{(m+n)}=P^{(m)}P^{(n)},\qquad P^{(n)}=P^n,\qquad P^{(0)}=I.$$

<!-- bilingual-en:start -->
The [[单位矩阵|identity matrix]] $I$ records that zero steps leave every state unchanged. The parentheses in $P^{(n)}$ label an interval; the equation says this interval matrix equals the ordinary matrix power. A probability row $\mu$ evolves to $\mu P^n$.
<!-- bilingual-en:end -->

[[单位矩阵]] $I$ 表示零步后每个状态都留在原处。$P^{(n)}$ 中的括号是在标记步数区间，而等式说明这个区间矩阵等于通常的矩阵幂。概率行 $\mu$ 更新为 $\mu P^n$。

<!-- bilingual-en:start -->
With probability columns, set $C=P^T$, where $T$ denotes [[矩阵转置|transpose]]. The same dynamics become $x_{n+1}=Cx_n$, so the first interval acts nearest to $x$: the two-interval expression is $C^nC^m x$. For one fixed matrix, $C^nC^m=C^{m+n}$, so this order is easy to overlook. It matters when the rule changes over time.
<!-- bilingual-en:end -->

若使用概率列，就令 $C=P^T$，其中 $T$ 表示[[矩阵转置]]。同一动态变成 $x_{n+1}=Cx_n$，先发生的区间要放在离 $x$ 最近的位置，所以两段更新是 $C^nC^m x$。同一个固定矩阵的幂满足 $C^nC^m=C^{m+n}$，容易掩盖这个顺序；规则随时间变化时，顺序就不能忽略。

<!-- bilingual-en:start -->
## When the rule changes with the calendar
<!-- bilingual-en:end -->
*规则随日历变化时怎样拼接*

<!-- bilingual-en:start -->
A nonhomogeneous chain is still Markov. Let $P_s$ be its row matrix for the step from time $s$ to $s+1$, and let $P(s,t)$ describe the whole interval from $s$ to $t$. For integers $s\le u\le t$, the same intermediate-state argument gives
<!-- bilingual-en:end -->

非齐次链仍可满足 Markov 性。令 $P_s$ 是从时刻 $s$ 到 $s+1$ 的行随机矩阵，$P(s,t)$ 描述从 $s$ 到 $t$ 的整段转移。对整数 $s\le u\le t$，同样按中间状态求和可得

$$P(s,t)=P(s,u)P(u,t)=P_sP_{s+1}\cdots P_{t-1},\qquad P(s,s)=I.$$

<!-- bilingual-en:start -->
For column matrices $C_s=P_s^T$, the total matrix is $C_{t-1}\cdots C_{s+1}C_s$, with the earliest step on the right. These finite products also describe countable-state kernels entry by entry using nonnegative sums; they do not require finite-dimensional spectral results.
<!-- bilingual-en:end -->

换成列矩阵 $C_s=P_s^T$，总矩阵就是 $C_{t-1}\cdots C_{s+1}C_s$，最早一步放在最右边。可数状态核也能逐元素用非负求和作这样的有限次复合，不需要有限维谱结论。

<!-- bilingual-en:start -->
For example, use the machine matrix $P$ first, then change its working row to $(0.8,0.2)$, producing $Q$. The two-step failure probability is $(PQ)_{WF}=0.9(0.2)+0.1(0.6)=0.24$. Reversing the dates gives $(QP)_{WF}=0.8(0.1)+0.2(0.6)=0.20$. A correct multiplication with the wrong order answers a different model.
<!-- bilingual-en:end -->

例如先用设备矩阵 $P$，下一小时把正常状态那一行改成 $(0.8,0.2)$，得到 $Q$。两步故障概率为 $(PQ)_{WF}=0.9(0.2)+0.1(0.6)=0.24$；若把日期顺序倒过来，就变成 $(QP)_{WF}=0.8(0.1)+0.2(0.6)=0.20$。乘法算对了但顺序错了，回答的是另一个模型。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [Weber, Cambridge Markov Chains, §1.5, PDF p. 7 / printed p. 3](https://www.statslab.cam.ac.uk/~rrw1/markov/M.pdf#page=7): supports the intermediate-state sum, row propagation and matrix-power identity for finite or countable states.
- [Gallager, MIT 6.262 Chapter 3, (3.3) and §3.3, PDF pp. 2 and 8](https://ocw.mit.edu/courses/6-262-discrete-stochastic-processes-spring-2011/3558b08622765d26c2b0a7d2eeeac885_MIT6_262S11_chap03.pdf#page=8): supports time-dependent rules and the homogeneous Chapman–Kolmogorov equation. The ordered nonhomogeneous formula follows by the same conditioning argument; its column version follows by transposing the product. The displayed numbers are direct calculations.
<!-- bilingual-en:end -->

- Weber 第 1.5 节（PDF 第 7 页，印刷第 3 页）支持中间状态求和、概率行更新与矩阵幂；适用于有限或可数状态。
- Gallager 式 (3.3) 与第 3.3 节（PDF 第 2、8 页）支持非齐次转移规则与齐次 C–K 方程。非齐次乘积由相同条件分解推出，列形式由乘积转置得到；数值例子直接复算。

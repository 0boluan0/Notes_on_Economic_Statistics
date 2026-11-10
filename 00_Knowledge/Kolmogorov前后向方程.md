---
aliases:
  - Kolmogorov forward equation
  - Kolmogorov backward equation
  - CTMC master equation
  - 前向方程与后向方程
student_os: knowledge-atom
atom_id: PROB-CTMC-009
atom_set: continuous-time-markov-chains
atom_type: theorem
status: source-checked
requires:
  - "[[生成矩阵约束]]"
  - "[[CTMC转移半群]]"
  - "[[Markov矩阵左右约定]]"
related:
  - "[[CTMC短时概率展开]]"
leads_to:
  - "[[CTMC矩阵指数]]"
part_of:
  - "[[连续时间马尔可夫链.canvas]]"
---

<!-- bilingual-en:start -->
# Kolmogorov forward and backward equations
<!-- bilingual-en:end -->

*Kolmogorov 前向与后向方程*

<!-- bilingual-en:start -->
The **Kolmogorov equations** describe how the transition probabilities of a continuous-time Markov chain change as we allow more time to pass. A generator gives local rates; the equations turn those rates into a rule for the evolution of probabilities. “Forward” and “backward” refer to which end of a time interval we examine first.
<!-- bilingual-en:end -->

**Kolmogorov 方程**描述连续时间 Markov 链的转移概率怎样随着允许经过的时长而变化。生成矩阵给出局部速率，方程则把速率变成概率随时间演化的规则。“前向”和“后向”指从时间区间的哪一端开始分析。

<!-- bilingual-en:start -->
Consider a **finite-state, time-homogeneous** [[CTMC定义|CTMC]] with [[生成矩阵约束|generator]] $Q=(q_{ij})$. In the [[Markov矩阵左右约定|row convention]], $p_{ij}(t)=\Pr_i(X_t=j)$ starts at $i$ and ends at $j$ after duration $t$, and $P(t)=(p_{ij}(t))$. The subscript on $\Pr_i$ specifies the initial state. The matrix $P(0)=I$ is the [[单位矩阵|identity]], because no state changes in zero time.
<!-- bilingual-en:end -->

考虑**有限状态、时间齐次**的 [[CTMC定义|CTMC]]，其[[生成矩阵约束|生成矩阵]]为 $Q=(q_{ij})$。采用[[Markov矩阵左右约定|行约定]]，$p_{ij}(t)=\Pr_i(X_t=j)$ 表示从 $i$ 出发、经过时长 $t$ 后处于 $j$；把它们组成 $P(t)$。$\Pr_i$ 的下标指定起始状态。$P(0)=I$ 是[[单位矩阵]]，因为零时长内状态不变。

<!-- bilingual-en:start -->
## A short interval at the end gives the forward equation
<!-- bilingual-en:end -->

*把短区间加在末端：前向方程*

<!-- bilingual-en:start -->
The [[CTMC转移半群|semigroup identity]] splits duration $t+h$ into a first part of length $t$ and a final short part of length $h$: $P(t+h)=P(t)P(h)$. At the intermediate time, the process may be in any state. The final short transition uses the rates in $Q$. Subtract $P(t)$ and divide by $h>0$:
<!-- bilingual-en:end -->

由[[CTMC转移半群|半群等式]]，把 $t+h$ 分成先经过 $t$、最后再经过很短的 $h$，得到 $P(t+h)=P(t)P(h)$。中间时点可以处于任何状态，最后一小段使用 $Q$ 中的速率。减去 $P(t)$，再除以 $h>0$：

$$
\frac{P(t+h)-P(t)}h
=P(t)\frac{P(h)-I}{h}
\longrightarrow P(t)Q.
$$

<!-- bilingual-en:start -->
The limit uses the [[CTMC短时概率展开|right derivative at zero]]. In finite dimensions the products have only finitely many terms, so this limit passes through the product. It gives the **forward equation**:
<!-- bilingual-en:end -->

这个极限使用了[[CTMC短时概率展开|零点处的右导数]]。有限维乘积只含有限项，极限可以逐项传入，得到**前向方程**：

$$P'(t)=P(t)Q,\qquad p'_{ij}(t)=\sum_k p_{ik}(t)q_{kj}.$$

<!-- bilingual-en:start -->
For a fixed starting state $i$, the off-diagonal terms with $k\ne j$ are probability flowing into destination $j$. The diagonal contribution is $p_{ij}(t)q_{jj}=-p_{ij}(t)q_j$, probability flowing out of $j$. Thus the derivative is inflow minus outflow. The prime denotes a [[导数|time derivative]], not an extra transition step.
<!-- bilingual-en:end -->

固定起点 $i$ 后，$k\ne j$ 的项表示从其他状态流入终点 $j$ 的概率率；对角项 $p_{ij}(t)q_{jj}=-p_{ij}(t)q_j$ 则表示从 $j$ 流出的概率率。因此导数就是流入减流出。撇号表示对时间的[[导数]]，不是多走了一步。

<!-- bilingual-en:start -->
## A short interval at the beginning gives the backward equation
<!-- bilingual-en:end -->

*把短区间放在开头：后向方程*

<!-- bilingual-en:start -->
We can instead split the same duration as a short first interval followed by length $t$: $P(t+h)=P(h)P(t)$. Now the first departure from the starting state is resolved before the rest of the evolution:
<!-- bilingual-en:end -->

也可以把同一时长分成先经过短区间、再经过 $t$，即 $P(t+h)=P(h)P(t)$。这次先分析从起点开始的局部变化，再接上后面的演化：

$$
\frac{P(t+h)-P(t)}h
=\frac{P(h)-I}{h}P(t)
\longrightarrow QP(t).
$$

$$P'(t)=QP(t),\qquad p'_{ij}(t)=\sum_k q_{ik}p_{kj}(t).$$

<!-- bilingual-en:start -->
This is the **backward equation**. Its rates act on the starting-state index $i$: make a possible initial change to $k$, then continue from $k$ to $j$. It still predicts a later state from an earlier one. It is not a formula for conditioning on the endpoint or for reversing a sample path.
<!-- bilingual-en:end -->

这就是**后向方程**。速率作用在起始状态下标 $i$ 上：先考虑开始时可能转到 $k$，再从 $k$ 继续到 $j$。它仍然是在从早到晚预测，不是在给定终点后反推起点，也不是把样本路径倒放。

> [!abstract] The same family solves both · 同一转移族满足两式
> <!-- bilingual-en:start -->
> For the finite homogeneous chain, $P(0)=I$ and $P'(t)=QP(t)=P(t)Q$. The [[CTMC矩阵指数|matrix exponential]] gives their unique solution. Its differentiability supplies the usual two-sided derivative at $t>0$; at $0$ the derivative is from the right.
> <!-- bilingual-en:end -->
>
> 有限齐次链满足 $P(0)=I$ 与 $P'(t)=QP(t)=P(t)Q$，[[CTMC矩阵指数|矩阵指数]]给出同一个唯一解。其可微性保证 $t>0$ 时通常的双侧导数；在 $0$ 处使用右导数。

<!-- bilingual-en:start -->
## Propagate a probability distribution
<!-- bilingual-en:end -->

*让状态分布随时间演化*

<!-- bilingual-en:start -->
An initial probability row $\alpha(0)$ gives $\alpha(t)=\alpha(0)P(t)$. Using the forward equation yields $\alpha'(t)=\alpha(t)Q$, often called the **master equation**. A probability column $x(t)=\alpha(t)^T$ instead satisfies $x'(t)=Q^Tx(t)$, where $T$ denotes [[矩阵转置|transpose]]. Keep the chosen row or column convention throughout.
<!-- bilingual-en:end -->

初始概率行 $\alpha(0)$ 在时刻 $t$ 给出 $\alpha(t)=\alpha(0)P(t)$。使用前向方程得 $\alpha'(t)=\alpha(t)Q$，常称为 **master equation（主方程）**。若改用概率列 $x(t)=\alpha(t)^T$，则是 $x'(t)=Q^Tx(t)$；$T$ 表示[[矩阵转置]]。行、列约定要始终一致。

> [!example] A working and failed device · 正常与故障设备
> <!-- bilingual-en:start -->
> Order the states as working $0$, failed $1$. Failure has rate $\lambda\ge0$, repair rate $\mu\ge0$, and $Q=\begin{pmatrix}-\lambda&\lambda\\\mu&-\mu\end{pmatrix}$. If $p_1(t)$ is the probability of being failed and $p_0(t)=1-p_1(t)$, then $p_1'(t)=\lambda p_0(t)-\mu p_1(t)$. With $\lambda=2,\mu=3$ per hour, this becomes $p_1'=2-5p_1$. Starting surely working gives $p_1(t)=\frac25(1-e^{-5t})$: direct differentiation verifies both the equation and $p_1(0)=0$.
> <!-- bilingual-en:end -->
>
> 按正常 $0$、故障 $1$ 排列，故障率为 $\lambda\ge0$，维修率为 $\mu\ge0$，于是 $Q=\begin{pmatrix}-\lambda&\lambda\\\mu&-\mu\end{pmatrix}$。若 $p_1(t)$ 是处于故障的概率，$p_0(t)=1-p_1(t)$，则 $p_1'(t)=\lambda p_0(t)-\mu p_1(t)$。取每小时 $\lambda=2,\mu=3$，得到 $p_1'=2-5p_1$；从确定正常出发，解为 $p_1(t)=\frac25(1-e^{-5t})$。直接求导即可同时检验方程与初值 $p_1(0)=0$。

<!-- bilingual-en:start -->
The equality $QP(t)=P(t)Q$ concerns this generator and its own homogeneous transition family. It does not make arbitrary matrices commute. Nor does the finite-state derivation justify unrestricted infinite sums: for countably many states, derivatives, convergence of sums, and uniqueness of the minimal solution require their own arguments. An unbounded rate matrix cannot simply be handled as a finite array. Time-inhomogeneous chains also require equations retaining the starting and ending clock times.
<!-- bilingual-en:end -->

$QP(t)=P(t)Q$ 针对的是这个生成矩阵及其自身的齐次转移族，并不使任意矩阵都可交换。有限状态推导也不能直接保证无穷求和合法：状态可数无穷时，导数、求和收敛以及最小解的唯一性都需要各自的论证。无界速率矩阵不能直接当作有限数组处理；时间不齐次链则还要保留起始、终止日历时点。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [Whitt, Continuous-Time Markov Chains, Theorem 3.1, PDF pp.9–10 / printed pp.8–9](https://www.columbia.edu/~ww2040/4106S11/CTMCchapter121906.pdf#page=9): forward/backward derivations, row convention and finite-state scope.
- [Berestycki and Sousi, Applied Probability, §1.7, PDF pp.15–18](https://www.statslab.cam.ac.uk/~ps422/notes-new2015.pdf#page=15): countable minimal backward equation and finite-dimensional solution/uniqueness. The device equation and solution are calculated directly.
- [[01_Math/05_随机过程/05_连续时间的马尔可夫链.md|Permanent course note]]: course scope and notation.
<!-- bilingual-en:end -->

- Whitt 定理3.1（PDF9–10页／印刷8–9页）：前后向推导、行约定与有限状态范围。
- Berestycki–Sousi 第1.7节（PDF15–18页）：可数最小后向方程及有限维解与唯一性；设备方程和解直接核算。
- 永久课程笔记：课程范围与记号。

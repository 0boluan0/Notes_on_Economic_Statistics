---
aliases:
  - CTMC matrix exponential
  - 生成矩阵的矩阵指数
student_os: knowledge-atom
atom_id: PROB-CTMC-008
atom_set: continuous-time-markov-chains
atom_type: theorem
status: source-checked
requires:
  - "[[生成矩阵约束]]"
  - "[[CTMC转移半群]]"
  - "[[矩阵指数]]"
  - "[[常系数线性系统的矩阵指数解]]"
related:
  - "[[Kolmogorov前后向方程]]"
leads_to:
  - "[[CTMC均匀化]]"
  - "[[有限CTMC平稳生成矩阵判据]]"
part_of:
  - "[[连续时间马尔可夫链.canvas]]"
---

<!-- bilingual-en:start -->
# The matrix exponential of a CTMC generator
<!-- bilingual-en:end -->

*CTMC 生成矩阵的矩阵指数*

<!-- bilingual-en:start -->
A generator tells us the local speed of each possible state change. To predict where the system will be after a whole hour, we must combine all possible jumps and waits during that hour. For a finite, time-homogeneous continuous-time Markov chain, the **matrix exponential** performs exactly this conversion from rates to finite-duration probabilities.
<!-- bilingual-en:end -->

生成矩阵告诉我们每种状态变化的局部速度。要预测整整一小时后系统在哪里，必须把这一小时中所有可能的跳跃和等待组合起来。对有限状态、时间齐次的连续时间 Markov 链，**矩阵指数**恰好完成从速率到有限时长转移概率的转换。

<!-- bilingual-en:start -->
Let $Q=(q_{ij})$ be a [[生成矩阵约束|finite generator]]: $q_{ij}\ge0$ for $i\ne j$, with finite entries and each row summing to zero. In the [[Markov矩阵左右约定|row convention]], $P(t)$ has entry $p_{ij}(t)=\Pr_i(X_t=j)$. The initial state is $i$, the elapsed duration is $t\ge0$, and the terminal state is $j$. Then
<!-- bilingual-en:end -->

设 $Q=(q_{ij})$ 是[[生成矩阵约束|有限生成矩阵]]：$i\ne j$ 时 $q_{ij}\ge0$，元素有限且每行和为零。采用[[Markov矩阵左右约定|行约定]]，$P(t)$ 的元素为 $p_{ij}(t)=\Pr_i(X_t=j)$，即从 $i$ 出发、经过 $t\ge0$ 后处于 $j$ 的概率。于是

$$
P(t)=e^{tQ}:=I+tQ+\frac{t^2Q^2}{2!}+\frac{t^3Q^3}{3!}+\cdots.
$$

<!-- bilingual-en:start -->
The [[矩阵指数|matrix exponential]] is defined by this convergent power series. Here $I$ is the [[单位矩阵|identity]], $Q^2=QQ$ uses [[矩阵乘法|matrix multiplication]], and $n!=1\cdot2\cdots n$ is the factorial. Each term combines matrix products; this is not the array obtained by exponentiating entries separately. At $t=0$ it gives $I$, whereas entrywise exponentiation would give an array full of ones.
<!-- bilingual-en:end -->

[[矩阵指数]]由这个收敛幂级数定义。$I$ 是[[单位矩阵]]，$Q^2=QQ$ 使用[[矩阵乘法]]，$n!=1\cdot2\cdots n$ 是阶乘。各项都使用矩阵乘积，不能把元素逐个取指数来替代。$t=0$ 时结果为 $I$，而逐元素取指数会得到全一数组。

<!-- bilingual-en:start -->
## Why it solves the transition equations
<!-- bilingual-en:end -->

*为什么它解出了转移方程*

<!-- bilingual-en:start -->
In finite dimensions, the exponential series can be differentiated term by term. Each power of $Q$ commutes with $Q$, giving both [[Kolmogorov前后向方程|Kolmogorov equations]]:
<!-- bilingual-en:end -->

有限维中，指数级数可以逐项求导。$Q$ 的各次幂都与 $Q$ 可交换，因此同时得到两条 [[Kolmogorov前后向方程|Kolmogorov 方程]]：

$$
\frac{d}{dt}e^{tQ}=Qe^{tQ}=e^{tQ}Q,\qquad e^{0Q}=I.
$$

<!-- bilingual-en:start -->
The solution is unique. If a differentiable matrix function $F(t)$ satisfies $F'(t)=F(t)Q$ and $F(0)=I$, the [[乘积法则|product rule]] gives $\frac{d}{dt}(F(t)e^{-tQ})=0$. Thus $F(t)e^{-tQ}=I$, so $F(t)=e^{tQ}$. This is the [[常系数线性系统的矩阵指数解|constant-coefficient linear-system solution]] applied to transition probabilities. The algebraic inverse $e^{-tQ}$ used in the proof need not itself be a probability transition matrix.
<!-- bilingual-en:end -->

解还是唯一的。若可微矩阵函数 $F(t)$ 满足 $F'(t)=F(t)Q$、$F(0)=I$，由[[乘积法则]]，$\frac{d}{dt}(F(t)e^{-tQ})=0$。因此 $F(t)e^{-tQ}=I$，从而 $F(t)=e^{tQ}$。这是把[[常系数线性系统的矩阵指数解]]用于转移概率。证明中使用的代数逆 $e^{-tQ}$ 本身不必是合法的概率转移矩阵。

<!-- bilingual-en:start -->
## Why the result contains valid probabilities
<!-- bilingual-en:end -->

*为什么算出来是合法概率*

<!-- bilingual-en:start -->
The powers $Q^n$ can contain negative entries, so positivity is not obvious from the first series. If $Q\ne0$, choose a finite number $\nu>0$ at least as large as every [[出口率|exit rate]] $q_i=-q_{ii}$. The matrix $R=I+Q/\nu$ is nonnegative with row sums one. Since $Q=\nu(R-I)$ and $R$ commutes with $I$,
<!-- bilingual-en:end -->

$Q^n$ 中可能有负数，所以不能直接从最初的级数看出最终非负。若 $Q\ne0$，选一个有限的 $\nu>0$，不小于每个[[出口率]] $q_i=-q_{ii}$。矩阵 $R=I+Q/\nu$ 就元素非负、每行和为一。因为 $Q=\nu(R-I)$，而 $R$ 与 $I$ 可交换，

$$e^{tQ}=\sum_{n=0}^{\infty}e^{-\nu t}\frac{(\nu t)^n}{n!}R^n.$$

<!-- bilingual-en:start -->
Every $R^n$ is a transition matrix, and the scalar weights are nonnegative and sum to $e^{-\nu t}e^{\nu t}=1$. This proves that $e^{tQ}$ is nonnegative with row sums one. When $Q=0$, the answer is simply $P(t)=I$. The representation also leads to [[CTMC均匀化|uniformization]], where $R$ includes candidate events that may leave the state unchanged.
<!-- bilingual-en:end -->

每个 $R^n$ 都是转移矩阵，前面的标量权重非负且和为 $e^{-\nu t}e^{\nu t}=1$。所以 $e^{tQ}$ 元素非负、每行和为一。若 $Q=0$，直接得到 $P(t)=I$。这个表示也导向[[CTMC均匀化]]；其中 $R$ 允许候选事件发生后状态仍不变。

> [!example] One irreversible failure · 一次不可逆故障
> <!-- bilingual-en:start -->
> For $Q=\begin{pmatrix}-\lambda&\lambda\\0&0\end{pmatrix}$ with $\lambda\ge0$, the transition matrix is $P(t)=\begin{pmatrix}e^{-\lambda t}&1-e^{-\lambda t}\\0&1\end{pmatrix}$. Differentiate it to check $P'=PQ$, and set $t=0$ to check the initial condition. At rate $2$ per hour and $t=1/2$ hour, failure probability is $1-e^{-1}\approx0.632$. It is not $e^{\lambda t}$, the incorrect entrywise exponential of the off-diagonal entry.
> <!-- bilingual-en:end -->
>
> 对 $\lambda\ge0$ 的 $Q=\begin{pmatrix}-\lambda&\lambda\\0&0\end{pmatrix}$，转移矩阵为 $P(t)=\begin{pmatrix}e^{-\lambda t}&1-e^{-\lambda t}\\0&1\end{pmatrix}$。直接求导可检验 $P'=PQ$，取 $t=0$ 可检验初值。故障率为每小时 $2$、经过半小时时，故障概率为 $1-e^{-1}\approx0.632$，不是把非对角元单独取指数所得的 $e^{\lambda t}$。

<!-- bilingual-en:start -->
## Diagonalization is optional
<!-- bilingual-en:end -->

*对角化不是必要条件*

<!-- bilingual-en:start -->
The power-series definition works for every finite square matrix. A [[对角化|diagonalization]], when available, can simplify computation, but a generator need not have one. For instance, the three-state chain that moves $0\to1\to2$ at unit rates and then stops has
<!-- bilingual-en:end -->

幂级数定义适用于任意有限方阵。若能[[对角化|对角化]]，可以简化计算，但生成矩阵未必可对角化。例如，以单位速率沿 $0\to1\to2$ 跳跃，然后停止的三状态链有

$$
Q=\begin{pmatrix}-1&1&0\\0&-1&1\\0&0&0\end{pmatrix},
\qquad
P(t)=\begin{pmatrix}
e^{-t}&te^{-t}&1-(1+t)e^{-t}\\
0&e^{-t}&1-e^{-t}\\
0&0&1
\end{pmatrix}.
$$

<!-- bilingual-en:start -->
The eigenvalue $-1$ occurs twice, but solving $(Q+I)v=0$ gives only a one-dimensional [[特征空间|eigenspace]], so $Q$ is not diagonalizable. Nevertheless the displayed $P(t)$ satisfies the differential equation and $P(0)=I$ entry by entry. The term $te^{-t}$ is a valid probability: it corresponds to having completed the first change but not the second by time $t$. A Jordan or Schur representation and uniformization are other computational routes to the same matrix exponential.
<!-- bilingual-en:end -->

特征值 $-1$ 出现两次，但解 $(Q+I)v=0$ 只得到一维[[特征空间]]，所以 $Q$ 不可对角化。上面的 $P(t)$ 仍逐项满足微分方程与初值。$te^{-t}$ 是合法概率，对应截至 $t$ 完成了第一次变化、却尚未完成第二次。Jordan、Schur 表示及均匀化，是计算同一个矩阵指数的其他途径。

> [!warning] The finite-state qualification matters · 有限状态条件有用
> <!-- bilingual-en:start -->
> For a countable state space, a rate matrix can be unbounded and the process can explode. Infinite matrix products, differentiation and the displayed series then need separate justification. Zero row sums alone do not identify a unique stochastic transition family after explosion; see [[最小CTMC|minimal chains]] and [[爆炸后延拓|post-explosion extensions]]. The finite-dimensional proof above makes no such unrestricted claim.
> <!-- bilingual-en:end -->
>
> 可数状态空间中，速率矩阵可能无界，过程也可能爆炸。这时无穷矩阵乘积、求导和上述级数都需要另外论证。仅凭零行和不能确定爆炸之后唯一的随机转移族，见[[最小CTMC]]与[[爆炸后延拓]]；上面的有限维证明没有作这种无限制推广。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [Whitt, Continuous-Time Markov Chains, Theorem 3.2 and §3.4, PDF pp.10 and 15–18](https://www.columbia.edu/~ww2040/4106S11/CTMCchapter121906.pdf#page=10): matrix-exponential solution and the nonnegative stochastic-series representation.
- [Berestycki and Sousi, Applied Probability, Theorems 1.27–1.29, PDF pp.17–18](https://www.statslab.cam.ac.uk/~ps422/notes-new2015.pdf#page=17): finite generator, exponential, differential equations and uniqueness.
- [[矩阵指数|Matrix exponential]] and [[常系数线性系统的矩阵指数解|linear-system solution]]: existing algebraic prerequisites. Both displayed examples are independently checked by differentiation and initial values.
<!-- bilingual-en:end -->

- Whitt 定理3.2及第3.4节（PDF10、15–18页）：矩阵指数解与非负随机矩阵级数表示。
- Berestycki–Sousi 定理1.27–1.29（PDF17–18页）：有限生成矩阵、指数、微分方程与唯一性。
- 既有矩阵指数和常系数线性系统解提供代数前置；两个展示例由求导和初值独立核算。

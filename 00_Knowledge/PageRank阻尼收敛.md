---
student_os: knowledge-atom
atom_id: PROB-DTMC-045
aliases:
- PageRank damping convergence
- Damped PageRank convergence
atom_type: theorem
status: source-checked
requires:
- '[[PageRank随机传送]]'
- '[[Markov稳态分布]]'
- '[[几何级数]]'
related:
- '[[全支持传送收敛]]'
- '[[PageRank模型依赖]]'
part_of:
- '[[离散时间马尔可夫链.canvas]]'
---

<!-- bilingual-en:start -->
# Convergence of damped PageRank
<!-- bilingual-en:end -->

*PageRank 阻尼收敛*

<!-- bilingual-en:start -->
Two people can start the same PageRank walk from different pages. **Damping convergence** says that repeated restarts make their probability distributions forget that initial difference. The restart distribution may favor only a few pages: giving every page a positive restart probability is not required for the uniqueness or convergence proved here.
<!-- bilingual-en:end -->

两个人可以从不同页面开始同一种 PageRank 游走。**阻尼收敛**说的是：反复加入重新开始的机会，会使他们的概率分布逐渐忘记初始差别。传送分布可以只偏向少数页面；本卡证明的唯一性与收敛，不要求每个页面都有正的传送概率。

<!-- bilingual-en:start -->
Use the finite [[PageRank随机传送|teleportation model]] on $n\ge1$ states. Its already-repaired [[Markov矩阵|row-stochastic]] matrix is $P$, its probability column is $v\ge0$ with $\sum_jv_j=1$, and its fixed link-following probability is $0\le\alpha<1$. Write $G=\alpha P+(1-\alpha)\mathbf1v^T$. The superscript $T$ is [[矩阵转置|transpose]], and $\mathbf1$ is the column of ones. The evolving distributions and stationary law below are written as rows.
<!-- bilingual-en:end -->

采用 $n\ge1$ 个状态上的有限[[PageRank随机传送|随机传送模型]]。$P$ 是已完成悬空修补的[[Markov矩阵|行随机矩阵]]，$v\ge0$ 是满足 $\sum_jv_j=1$ 的概率列，固定的跟随链接概率为 $0\le\alpha<1$。记 $G=\alpha P+(1-\alpha)\mathbf1v^T$；$T$ 表示[[矩阵转置|转置]]，$\mathbf1$ 是全一列。下面的逐步分布与稳态分布都写作行。

> [!important] One stationary law, from every starting law · 任意起点趋于同一稳态
> <!-- bilingual-en:start -->
> There is exactly one [[Markov稳态分布|stationary probability row]] $\pi$. For every initial probability row $x_0$,
>
> $$x_k=x_0G^k\longrightarrow\pi.$$
>
> For $0<\alpha<1$, a quantitative bound is
>
> $$\|x_k-\pi\|_1\le\alpha^k\|x_0-\pi\|_1\le2\alpha^k\qquad(k\ge0).$$
>
> Here $\|z\|_1=\sum_j|z_j|$ is the sum of the absolute coordinate differences. When $\alpha=0$, $x_k=\pi=v^T$ for every $k\ge1$, so convergence takes one step.
> <!-- bilingual-en:end -->
>
> 恰好存在一个[[Markov稳态分布|稳态概率行]] $\pi$，并且任意初始概率行 $x_0$ 都满足上面的收敛。$0<\alpha<1$ 时，误差不超过 $2\alpha^k$；这里 $\|z\|_1=\sum_j|z_j|$ 表示各坐标差的绝对值之和。$\alpha=0$ 时，全部 $k\ge1$ 都有 $x_k=\pi=v^T$，一步就完成收敛。

<!-- bilingual-en:start -->
## Why the starting distributions come together
<!-- bilingual-en:end -->

*为什么不同起点会靠拢*

<!-- bilingual-en:start -->
For any probability row $x$, multiplication gives $xG=\alpha xP+(1-\alpha)v^T$, because $x\mathbf1=1$. Subtract the updates from two probability rows $x,y$. Their restart terms cancel, leaving

$$(x-y)G=\alpha(x-y)P.$$

A stochastic step cannot increase the sum of absolute differences. To see this, put $z=x-y$ and use the nonnegative entries and unit row sums of $P$:
<!-- bilingual-en:end -->

对任意概率行 $x$，因为 $x\mathbf1=1$，所以 $xG=\alpha xP+(1-\alpha)v^T$。把两个概率行 $x,y$ 的更新相减，重新开始的部分相消，留下 $(x-y)G=\alpha(x-y)P$。随机转移不会增加绝对差之和：令 $z=x-y$，利用 $P$ 的元素非负与每行和为一，得到

$$
\|zP\|_1
=\sum_j\left|\sum_i z_iP_{ij}\right|
\le\sum_j\sum_i|z_i|P_{ij}
=\sum_i|z_i|=\|z\|_1.
$$

<!-- bilingual-en:start -->
Consequently $\|xG-yG\|_1\le\alpha\|x-y\|_1$. Each step multiplies the remaining difference by at most $\alpha<1$; this is the contraction. The argument uses no strict positivity of $v$. The bound may be conservative, but it also explains why taking $\alpha$ very close to one weakens this worst-case guarantee.
<!-- bilingual-en:end -->

因此 $\|xG-yG\|_1\le\alpha\|x-y\|_1$。每一步把剩余差距最多乘以 $\alpha<1$，这就是收缩。整个论证没有用到 $v$ 严格为正。这个界可能不紧，但它说明把 $\alpha$ 取到非常接近一时，最坏情况下的保证会变弱。

<!-- bilingual-en:start -->
## Build the limit by averaging walk lengths
<!-- bilingual-en:end -->

*把不同游走长度混合，构造极限*

<!-- bilingual-en:start -->
We can construct the fixed distribution rather than merely assume it exists:

$$\pi=(1-\alpha)\sum_{m=0}^{\infty}\alpha^m v^TP^m.$$

The row $v^TP^m$ is the distribution after starting from $v^T$ and taking $m$ ordinary $P$ steps. The zero-step term uses $P^0=I$, the [[单位矩阵|identity matrix]]. The weights $(1-\alpha)\alpha^m$ sum to one by the [[几何级数|geometric series]]. Every summand is nonnegative, and each of its coordinates is bounded by the corresponding weight. These weights are summable, so the series defines a probability row. At $\alpha=0$, keep only the zero-step term.
<!-- bilingual-en:end -->

可以直接构造固定分布，而不只是假设它存在：$\pi=(1-\alpha)\sum_{m=0}^{\infty}\alpha^m v^TP^m$。其中 $v^TP^m$ 是从 $v^T$ 开始，按普通 $P$ 转移走 $m$ 步后的分布；零步使用 $P^0=I$，即[[单位矩阵]]。由[[几何级数]]，权重 $(1-\alpha)\alpha^m$ 的和为一。各坐标都非负，且每项不超过对应的可求和权重，因此这个级数定义了概率行。$\alpha=0$ 时只保留零步项。

<!-- bilingual-en:start -->
Shifting the summation index gives $\alpha\pi P+(1-\alpha)v^T=\pi$, so $\pi G=\pi$. If another stationary probability row $\rho$ existed, contraction would give $\|\pi-\rho\|_1\le\alpha\|\pi-\rho\|_1$, forcing the difference to be zero. Applying the same inequality repeatedly to $x_0$ and $\pi$ proves the stated error bound; two probability rows have distance at most two.
<!-- bilingual-en:end -->

把求和下标平移一位，就得到 $\alpha\pi P+(1-\alpha)v^T=\pi$，所以 $\pi G=\pi$。若还有另一个稳态概率行 $\rho$，收缩便给出 $\|\pi-\rho\|_1\le\alpha\|\pi-\rho\|_1$，迫使差距为零。再把同一不等式反复用于 $x_0$ 与 $\pi$，就得到误差界；两个概率行的距离至多为二。

<!-- bilingual-en:start -->
The same series also has a linear-system form. The matrix series $\sum_{m\ge0}\alpha^mP^m$ converges entry by entry, and multiplication by $I-\alpha P$ telescopes to $I$ since the remainder $\alpha^{M+1}P^{M+1}$ tends to zero. Hence it is the [[逆矩阵|inverse]] of $I-\alpha P$, and

$$\pi=(1-\alpha)v^T(I-\alpha P)^{-1}.$$

This is often called a Neumann-series representation. It is an exact characterization; numerical computation need not form an explicit inverse.
<!-- bilingual-en:end -->

同一级数还能写成线性方程的形式。矩阵级数 $\sum_{m\ge0}\alpha^mP^m$ 逐元素收敛；乘以 $I-\alpha P$ 后中间项相消，由于余项 $\alpha^{M+1}P^{M+1}$ 趋于零，结果为 $I$。因此它就是 $I-\alpha P$ 的[[逆矩阵]]，从而 $\pi=(1-\alpha)v^T(I-\alpha P)^{-1}$。这常称为 Neumann 级数表示，是精确刻画；数值计算不必显式形成逆矩阵。

> [!example] Zero teleportation mass does not prevent convergence · 传送概率为零不妨碍收敛
> <!-- bilingual-en:start -->
> Take $P=I_2$, $v=(1,0)^T$ and $\alpha=1/2$. Then
>
> $$G=\begin{pmatrix}1&0\\1/2&1/2\end{pmatrix},\qquad
> (0,1)G^k=(1-2^{-k},2^{-k}).$$
>
> The limit is the unique stationary row $(1,0)$. State 1 can never reach state 2, and the latter has zero stationary mass. Thus convergence and uniqueness do not require full support or irreducibility. [[全支持传送收敛|Full support]] provides stronger positivity properties.
> <!-- bilingual-en:end -->
>
> 取 $P=I_2$、$v=(1,0)^T$、$\alpha=1/2$，就得到上面的 $G$。从 $(0,1)$ 开始，第 $k$ 步分布为 $(1-2^{-k},2^{-k})$，趋于唯一稳态 $(1,0)$。状态 1 永远到不了状态 2，后者的稳态质量为零。因此，唯一与收敛不要求全支持或不可约；[[全支持传送收敛|全支持]]提供的是更强的正性性质。

<!-- bilingual-en:start -->
The endpoint $\alpha=1$ is different: there is no strict contraction. With $P=I_2$, every probability row is stationary; with $P=\begin{pmatrix}0&1\\1&0\end{pmatrix}$, the unique stationary row is $(1/2,1/2)$ but $(1,0)$ alternates forever. Whether an undamped chain converges must therefore be checked separately.
<!-- bilingual-en:end -->

端点 $\alpha=1$ 不同：不再有严格收缩。若 $P=I_2$，每个概率行都平稳；若 $P=\begin{pmatrix}0&1\\1&0\end{pmatrix}$，虽然唯一稳态为 $(1/2,1/2)$，但 $(1,0)$ 会永远交替。因此，没有阻尼时是否收敛，需要另行检查。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [Gleich, *PageRank Beyond the Web*, Definition 2.1, equation (2.3), Theorem 2.2 and the following Neumann-series discussion, PDF pp.3–4 / printed pp.323–324](https://www.cs.cornell.edu/courses/cs6241/2019sp/readings/Gleich-2015-PageRank.pdf#page=3): arbitrary probability teleportation vectors, unique solutions, contraction and the series representation. The source uses probability columns; the row proof and examples above are checked directly. Its stated range is $0<\alpha<1$; the $\alpha=0$ endpoint is verified separately here.
- [Weber, Cambridge *Markov Chains*, Example 8.6, PDF p.36 / printed p.32](https://www.statslab.cam.ac.uk/~rrw1/markov/M.pdf#page=36): the row-based random-surfer model with $\alpha\in[0,1)$. Its uniform-teleportation example is a special case, not a restriction on the general theorem.
<!-- bilingual-en:end -->

- Gleich，定义 2.1、式 (2.3)、定理 2.2 及其后的 Neumann 级数讨论，PDF3–4页／印刷323–324页：支持任意概率传送向量、解的唯一性、收缩与级数表示。原文用概率列；上面的行版本证明与例子直接核验。原文定理范围为 $0<\alpha<1$，本文单独验证 $\alpha=0$ 端点。
- Weber 例 8.6，PDF36页／印刷32页：支持 $\alpha\in[0,1)$ 的行约定随机浏览者模型。其均匀传送例子是一般定理的特例，不是一般定理的限制。

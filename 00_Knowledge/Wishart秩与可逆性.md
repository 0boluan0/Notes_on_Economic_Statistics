---
aliases:
  - "Wishart rank and invertibility"
  - "Wishart 秩与可逆性"
student_os: knowledge-atom
atom_id: STAT-WISH-004
atom_set: wishart-sample-covariance
atom_type: theorem-boundary
status: source-checked
requires:
  - "[[Wishart分布]]"
  - "[[正定与半正定可逆性]]"
part_of:
  - "[[Wishart 分布与样本协方差推断.canvas]]"
related:
  - "[[样本协方差Wishart律]]"
---

<!-- bilingual-en:start -->
# Wishart rank and invertibility
<!-- bilingual-en:end -->

*Wishart 矩阵的秩与可逆性*

<!-- bilingual-en:start -->
In the integer-degree construction, a [[Wishart分布|Wishart matrix]] adds Gaussian outer products. Each generating vector contributes at most one direction, so too few vectors cannot fill all $p$ coordinate directions. This card explains when the sum reaches its largest possible [[矩阵秩|rank]]—the number of independent directions—and when an ordinary [[逆矩阵|inverse]] exists.
<!-- bilingual-en:end -->

在整数自由度构造中，[[Wishart分布|Wishart 矩阵]]由 Gaussian 外积相加得到。每个生成向量至多贡献一个方向，向量太少便不可能覆盖全部 $p$ 个坐标方向。本卡解释：这个和何时达到最大可能的[[矩阵秩|秩]]，也就是独立方向个数；普通[[逆矩阵]]又何时存在。

> [!example] One direction cannot fill a plane · 一个方向填不满平面
> <!-- bilingual-en:start -->
> In two dimensions, a single vector $z=(1,2)^T$ gives
> $$zz^T=\begin{bmatrix}1&2\\2&4\end{bmatrix},\qquad
> zz^T\begin{bmatrix}2\\-1\end{bmatrix}=0.$$
> The second column is twice the first. The matrix has rank one and loses the nonzero direction $(2,-1)^T$, so it cannot be inverted. Adding the outer product of $(2,-1)^T$ gives $\operatorname{diag}(5,5)$, which has rank two and inverse $\operatorname{diag}(1/5,1/5)$. The issue is how many independent directions the vectors supply.
> <!-- bilingual-en:end -->
>
> 二维中，一个向量 $z=(1,2)^T$ 产生上述外积。第二列是第一列的两倍，秩为一；非零方向 $(2,-1)^T$ 被映成零，所以矩阵不可逆。再加上 $(2,-1)^T$ 的外积，就得到 $\operatorname{diag}(5,5)$，秩为二，逆为 $\operatorname{diag}(1/5,1/5)$。关键是生成向量实际提供了多少个独立方向。

<!-- bilingual-en:start -->
## Integer degrees of freedom count generating vectors
<!-- bilingual-en:end -->

*整数自由度计算生成向量的个数*

<!-- bilingual-en:start -->
Take $\nu\ge1$ independent $N_p(0,\Sigma)$ vectors, with integer $\nu$ and real [[正定矩阵|positive-definite]] $\Sigma$. Positive definiteness means every nonzero linear combination has positive variance; the Gaussian vector is not confined to a lower-dimensional subspace. Write $W=\sum_{i=1}^{\nu}Z_iZ_i^T$.
<!-- bilingual-en:end -->

取整数 $\nu\ge1$ 个独立的 $N_p(0,\Sigma)$ 向量，实矩阵 $\Sigma$ [[正定矩阵|正定]]。这意味着每个非零线性组合都有正方差，Gaussian 向量不局限在更低维的子空间中。令 $W=\sum_{i=1}^{\nu}Z_iZ_i^T$。

> [!important] Rank with probability one · 概率为一的秩
> <!-- bilingual-en:start -->
> Under these assumptions,
> $$\operatorname{rank}(W)=\min(p,\nu)\quad\text{almost surely}.$$
> Consequently, $W$ is positive definite and invertible almost surely exactly when $\nu\ge p$. “Almost surely” means the exceptional draws have probability zero; it does not say every imaginable list of vectors has maximal rank.
> <!-- bilingual-en:end -->
>
> 在这些条件下，$\operatorname{rank}(W)=\min(p,\nu)$ 几乎必然成立。因此，$W$ 几乎必然正定且可逆，当且仅当 $\nu\ge p$。“几乎必然”指例外抽取的概率为零，并不是说每一组想象得出的向量都达到最大秩。

<!-- bilingual-en:start -->
To see the mechanism, put the generating vectors into the columns of a $p\times\nu$ matrix $Z=[Z_1\ \cdots\ Z_\nu]$. Then $W=ZZ^T$. For any $a\in\mathbb R^p$,
<!-- bilingual-en:end -->

把生成向量排成 $p\times\nu$ 矩阵的列：$Z=[Z_1\ \cdots\ Z_\nu]$，就有 $W=ZZ^T$。对任意 $a\in\mathbb R^p$，

$$
a^TWa=\|Z^Ta\|_2^2.
$$

<!-- bilingual-en:start -->
Here $\|\cdot\|_2^2$ adds coordinate squares. Thus $Wa=0$ exactly when $Z^Ta=0$: the two maps lose precisely the same directions. The [[Gram矩阵正定判据|Gram rank identity]] gives $\operatorname{rank}(ZZ^T)=\operatorname{rank}(Z)\le\min(p,\nu)$.
<!-- bilingual-en:end -->

这里 $\|\cdot\|_2^2$ 将各坐标平方相加。因此 $Wa=0$ 当且仅当 $Z^Ta=0$，两者丢失的方向完全相同。由 [[Gram矩阵正定判据|Gram 秩恒等式]]，$\operatorname{rank}(ZZ^T)=\operatorname{rank}(Z)\le\min(p,\nu)$。

<!-- bilingual-en:start -->
Why is this bound reached? A Gaussian vector with positive-definite covariance has a [[Gaussian密度与奇异支撑|full-dimensional density]]. A fixed proper subspace—a line in a plane, for example—has probability zero. Conditional on the vectors already drawn, their [[张成|span]] is fixed, while the next independent vector keeps its Gaussian law. Until that span has dimension $p$, the next vector therefore adds a new direction with probability one. Thus the $\nu$ columns span $\min(p,\nu)$ directions. Finally, $W$ is always positive semidefinite, so [[正定与半正定可逆性|full rank, positive definiteness and invertibility]] coincide for it.
<!-- bilingual-en:end -->

为什么能达到这个上界？协方差正定的 Gaussian 向量有[[Gaussian密度与奇异支撑|全维密度]]，落在固定真子空间里的概率为零，例如平面里的一条直线。给定此前已抽出的向量，它们的[[张成]]空间就固定了，而下一个独立向量仍保持原 Gaussian 分布。在张成空间达到 $p$ 维之前，下一个向量都以概率一贡献新方向；所以这 $\nu$ 列共张成 $\min(p,\nu)$ 个方向。最后，$W$ 总是半正定，因此对它而言[[正定与半正定可逆性|满秩、正定和可逆]]等价。

<!-- bilingual-en:start -->
## Singular scale leaves fewer available directions
<!-- bilingual-en:end -->

*尺度奇异时，可用方向本来就更少*

<!-- bilingual-en:start -->
If $\Sigma$ is only positive semidefinite with rank $r<p$, the centered Gaussian vectors lie in its column space, an $r$-dimensional subspace. In coordinates along that subspace, the nonzero covariance directions form a positive-definite $r$-dimensional Gaussian model. Applying the same argument there gives
<!-- bilingual-en:end -->

若 $\Sigma$ 仅半正定，秩为 $r<p$，中心 Gaussian 向量会落在它的列空间，即一个 $r$ 维子空间中。沿这个子空间选取坐标后，非零协方差方向构成一个正定的 $r$ 维 Gaussian 模型。在那里应用同样论证，得到

$$
\operatorname{rank}(W)=\min(r,\nu)\qquad\text{almost surely}.
$$

<!-- bilingual-en:start -->
In particular, no number of draws makes $W$ invertible as a $p\times p$ matrix when $r<p$. If $r=0$, every generating vector and $W$ are zero almost surely. This is still the integer outer-product construction; it does not use the nonsingular Wishart density.
<!-- bilingual-en:end -->

所以只要 $r<p$，无论抽取多少个向量，都不能使 $W$ 作为 $p\times p$ 矩阵可逆。$r=0$ 时，每个生成向量及 $W$ 都几乎必然为零。这里仍然使用整数外积和构造，没有使用非奇异 Wishart 密度。

> [!warning] Noninteger degrees of freedom have a different boundary · 非整数自由度的边界不同
> <!-- bilingual-en:start -->
> The nonsingular [[Wishart分布|Wishart density]] requires positive-definite scale and real $\nu>p-1$, and assigns probability one to positive-definite matrices. For $p=2,\nu=1.5$, it therefore gives rank two almost surely. The integer rule “invertible iff $\nu\ge p$” counts generating vectors and must not be applied to this noninteger model. For $p=2,\nu=1$, by contrast, the integer construction is valid and has rank one; it is outside that density's domain.
> <!-- bilingual-en:end -->
>
> 非奇异 [[Wishart分布|Wishart 密度]]要求尺度正定、实数 $\nu>p-1$，并把概率一赋给正定矩阵。因此 $p=2,\nu=1.5$ 时，秩几乎必然为二。整数规则“可逆当且仅当 $\nu\ge p$”在计算生成向量的个数，不能套到这个非整数模型上。反之，$p=2,\nu=1$ 的整数构造有效且秩为一，只是不属于上述密度的参数域。

<!-- bilingual-en:start -->
## The sample-covariance threshold
<!-- bilingual-en:end -->

*样本协方差的可逆门槛*

<!-- bilingual-en:start -->
For $n\ge2$ iid $N_p(\mu,\Sigma)$ observations with positive-definite $\Sigma$, let $S$ be the [[样本协方差矩阵|sample covariance]] formed by subtracting the sample mean and dividing by $n-1$. The [[样本协方差Wishart律|sampling law]] gives $(n-1)S\sim W_p(\Sigma,n-1)$, so
<!-- bilingual-en:end -->

对 $n\ge2$ 个 iid $N_p(\mu,\Sigma)$ 观测，假设 $\Sigma$ 正定。令 $S$ 为减去样本均值后、以 $n-1$ 为分母的[[样本协方差矩阵]]。[[样本协方差Wishart律|抽样分布]]给出 $(n-1)S\sim W_p(\Sigma,n-1)$，因此

$$
\operatorname{rank}(S)=\min(p,n-1)\quad\text{almost surely},
\qquad
S^{-1}\text{ exists almost surely}\iff n\ge p+1.
$$

<!-- bilingual-en:start -->
At $n=p$, the rank is at most $p-1$: centering makes the $n$ centered observation vectors sum to zero, leaving at most $n-1$ independent directions. This upper bound is deterministic. The Gaussian model supplies the stronger assertion that the upper bound is reached almost surely. For arbitrary data, merely having $n\ge p+1$ does not guarantee invertibility; exact duplicate variables can still reduce the rank. See the [[样本协方差秩边界|sample-covariance rank bound]] for that distinction.
<!-- bilingual-en:end -->

$n=p$ 时，秩至多为 $p-1$：中心化后的 $n$ 个观测向量之和为零，至多剩下 $n-1$ 个独立方向。这个上界是确定性的；Gaussian 模型额外保证几乎必然达到上界。对任意数据，仅有 $n\ge p+1$ 并不保证可逆，例如重复变量仍会降低秩。[[样本协方差秩边界]]专门区分这两种陈述。

<!-- bilingual-en:start -->
Invertibility is an algebraic threshold, not a numerical-stability guarantee. For example, $\operatorname{diag}(1,\varepsilon)$ is positive definite for every $\varepsilon>0$, yet its inverse contains $1/\varepsilon$, which can greatly amplify error when $\varepsilon$ is small. A covariance estimate near the rank threshold may likewise have a small minimum eigenvalue. A successful inverse alone does not establish reliable inference.
<!-- bilingual-en:end -->

可逆是代数门槛，不是数值稳定性保证。例如每个 $\varepsilon>0$ 都使 $\operatorname{diag}(1,\varepsilon)$ 正定，但它的逆包含 $1/\varepsilon$，$\varepsilon$ 很小时可能大幅放大误差。接近秩门槛的协方差估计也可能有很小的最小特征值。成功求出逆矩阵本身，不能证明推断可靠。

## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.1sum.pdf#page=2|MIT 18.06SC, Session 2.1 summary, p. 2]]: Gram nullspace and rank identity. The sequential Gaussian-span argument and singular-scale reduction are spelled out above.
- [MIT 6.436J, Lecture 14, pp. 3–6](https://ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/ceb423b6a703abd60429367b733f45bf_MIT6_436JF18_lec14.pdf#page=3): Gaussian generating factors, positive-definite densities and singular support.
- [Purdue, DasGupta, Graduate Probability, Definition 1.76 and Theorem 1.91, PDF p. 162 / printed p. 172](https://www.stat.purdue.edu/~dasgupta/gradprob.pdf#page=162): nonsingular integer Wishart density and the Gaussian sample-covariance theorem when $n>p$.
- [Stan Functions Reference, “Wishart distribution”](https://mc-stan.org/docs/functions-reference/covariance_matrix_distributions.html#wishart-distribution): positive-definite support for real $\nu>p-1$, kept separate from the integer rank theorem.
- [[01_Math/04_多元统计分析/04_多元正态分布The Multivariate Normal Distribution.md#1.6. Wishart 分布|Multivariate Statistics §1.6.1]]: the $n-1$ sample-covariance degrees of freedom. The displayed rank-one, rank-two and small-diagonal examples are direct calculations.
<!-- bilingual-en:end -->

- MIT 18.06SC 第 2.1 节摘要第 2 页：核对 Gram 零空间与秩恒等式；正文展开逐个 Gaussian 方向的论证与奇异尺度约化。
- MIT 6.436J 第 14 讲第 3–6 页：核对 Gaussian 生成因子、正定密度与奇异支撑。
- Purdue 的定义 1.76、定理 1.91，PDF 第 162 页、印刷第 172 页：核对非奇异整数 Wishart 密度，以及 $n>p$ 时的 Gaussian 样本协方差定理。
- Stan 第 “Wishart distribution” 节：核对实数 $\nu>p-1$ 时的正定支持，并与整数秩定理区分。
- 多元统计课程第 1.6.1 节：核对样本协方差的 $n-1$ 自由度。正文直接计算秩一、秩二及小对角元例子。

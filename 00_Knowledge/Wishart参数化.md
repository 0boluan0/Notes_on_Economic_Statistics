---
aliases:
  - "Wishart scale and precision conventions"
  - "Wishart parameterization"
  - "Wishart 参数化约定"
student_os: knowledge-atom
atom_id: STAT-WISH-005
atom_set: wishart-sample-covariance
atom_type: convention-boundary
status: source-checked
requires:
  - "[[Wishart分布]]"
part_of:
  - "[[Wishart 分布与样本协方差推断.canvas]]"
related:
  - "[[样本协方差Wishart律]]"
---

<!-- bilingual-en:start -->
# Wishart scale and precision conventions
<!-- bilingual-en:end -->

*Wishart 的尺度与精度约定*

<!-- bilingual-en:start -->
Before using a [[Wishart分布|Wishart distribution]], identify what its matrix parameter means. In the scale convention, it is the covariance of the Gaussian vectors used to form the outer-product sum. A precision convention instead supplies the **inverse of that covariance**. The same numerical input therefore describes different distributions if it is interpreted in these two different ways.
<!-- bilingual-en:end -->

使用 [[Wishart分布|Wishart 分布]]前，要先确定矩阵参数代表什么。尺度约定使用生成外积和的 Gaussian 向量的协方差；精度约定则使用**这个协方差的逆矩阵**。同一个数值输入若分别按这两种含义解释，会得到不同分布。

> [!example] The scalar case exposes a wrong convention · 一维就能看出约定用错
> <!-- bilingual-en:start -->
> Take three independent $N(0,4)$ numbers. Each has variance $4$, so their squared sum is $W=4Q$ with a [[卡方分布|chi-square variable]] $Q\sim\chi_3^2$ and $E[W]=12$. The scale is $V=4$, while the precision is $\Omega=V^{-1}=1/4$. These two inputs describe the same law when sent to their respective parameter slots.
>
> Passing $4$ into a **precision** slot instead means variance $1/4$. The resulting squared sum has mean $3/4$, not $12$. In two dimensions the same distinction is visible with $V=\operatorname{diag}(4,1)$, $\Omega=\operatorname{diag}(1/4,1)$ and $E[W]=\operatorname{diag}(12,3)$; [[对角矩阵|diag]] places the listed numbers on the diagonal and zeros elsewhere.
> <!-- bilingual-en:end -->
>
> 取三个独立的 $N(0,4)$ 数，每个方差为 $4$。平方和为 $W=4Q$，其中 $Q\sim\chi_3^2$ 是[[卡方分布|卡方变量]]，所以均值是 $12$。尺度为 $V=4$，精度为 $\Omega=V^{-1}=1/4$；分别填入对应的参数位置，表示同一个分布。
>
> 若把 $4$ 填入**精度**位置，实际方差便是 $1/4$，平方和均值变成 $3/4$。二维时，$V=\operatorname{diag}(4,1)$、$\Omega=\operatorname{diag}(1/4,1)$ 和 $E[W]=\operatorname{diag}(12,3)$ 展示同一区别；[[对角矩阵|diag]] 将所列数放在对角线上，其余位置为零。

<!-- bilingual-en:start -->
## Calibrate the meaning before reading the argument order
<!-- bilingual-en:end -->

*先校准含义，再看参数顺序*

<!-- bilingual-en:start -->
For $p$ coordinates and an integer $\nu\ge1$, the scale convention is fixed by the generating statement
<!-- bilingual-en:end -->

有 $p$ 个坐标、整数 $\nu\ge1$ 时，尺度约定由下面的生成式明确确定：

$$
Z_i\overset{\mathrm{iid}}{\sim}N_p(0,V),\qquad
W=\sum_{i=1}^{\nu}Z_iZ_i^T,\qquad
E[W]=\nu V.
$$

<!-- bilingual-en:start -->
The notation “iid” means [[独立同分布|independent with a common distribution]]. With positive-definite $V$, set $\Omega=V^{-1}$, using the [[逆矩阵|matrix inverse]]. The same law then has $E[W]=\nu\Omega^{-1}$. These mean formulas also hold for the nonsingular real-degree extension $\nu>p-1$. The precision here belongs to the generating Gaussian vector; it is not the inverse of a covariance matrix of all the random entries of $W$.
<!-- bilingual-en:end -->

“iid” 表示[[独立同分布]]。$V$ 正定时，用[[逆矩阵]]定义 $\Omega=V^{-1}$，同一个分布就满足 $E[W]=\nu\Omega^{-1}$。这些均值公式也适用于 $\nu>p-1$ 的非奇异实数自由度延拓。这里的精度属于生成 Gaussian 向量，并不是把 $W$ 所有随机元素组成的协方差矩阵取逆。

<!-- bilingual-en:start -->
This vault writes $W_p(V,\nu)$ with **scale first**. The course record and some references write $W_p(\nu,V)$ with **degrees of freedom first**. Neither letter choice nor position alone establishes whether a matrix is a scale or a precision: check the definition. A stated expectation helps identify an author's convention within an already specified Wishart family; matching an arbitrary random matrix's mean does not prove that its distribution is Wishart.
<!-- bilingual-en:end -->

本库写 $W_p(V,\nu)$，把**尺度放在前面**；课程记录及部分文献写 $W_p(\nu,V)$，把**自由度放在前面**。字母和位置本身都不能确定矩阵是尺度还是精度，必须读定义。已明确属于 Wishart 族时，作者给出的期望可帮助辨认约定；仅匹配任意随机矩阵的均值，不能证明它服从 Wishart 分布。

<!-- bilingual-en:start -->
## The density shows where the inverse belongs
<!-- bilingual-en:end -->

*从密度看逆矩阵的位置*

<!-- bilingual-en:start -->
Suppose $V$ is positive definite and $\nu>p-1$ is real. At a positive-definite symmetric value $w$, the density can be written
<!-- bilingual-en:end -->

设 $V$ 正定，实数 $\nu>p-1$。在正定对称取值 $w$ 处，密度可写成

$$
f(w\mid V,\nu)\propto
|V|^{-\nu/2}|w|^{(\nu-p-1)/2}
\exp\!\left[-\frac12\operatorname{tr}(V^{-1}w)\right].
$$

<!-- bilingual-en:start -->
Vertical bars denote the [[行列式|determinant]], not absolute values of individual entries. The trace $\operatorname{tr}(A)$ is the sum of a square matrix's diagonal entries. The proportionality symbol omits a positive factor depending only on $p$ and $\nu$, retaining all scale dependence. Substituting $V=\Omega^{-1}$ gives the same density as
<!-- bilingual-en:end -->

竖线表示[[行列式]]，不是逐个元素的绝对值；$\operatorname{tr}(A)$ 表示方阵的对角元之和。比例号省略只依赖 $p,\nu$ 的正常数，保留全部尺度依赖。代入 $V=\Omega^{-1}$，同一个密度变成

$$
f(w\mid\Omega,\nu)\propto
|\Omega|^{\nu/2}|w|^{(\nu-p-1)/2}
\exp\!\left[-\frac12\operatorname{tr}(\Omega w)\right].
$$

<!-- bilingual-en:start -->
The determinant power changes sign because $|V|=|\Omega|^{-1}$, and the matrix multiplying $w$ is now $\Omega$. This reparameterizes the same random matrix $W$. It does not replace $W$ with $W^{-1}$; that latter operation, when the inverse exists, produces an inverse-Wishart distribution. These densities require positive-definite scale and support; a singular integer outer-product law must be handled through its construction instead.
<!-- bilingual-en:end -->

因为 $|V|=|\Omega|^{-1}$，尺度行列式的幂改变符号；与 $w$ 相乘的矩阵则变成 $\Omega$。这只是在重新参数化同一个随机矩阵 $W$，没有把 $W$ 换成 $W^{-1}$；后者在逆存在时产生逆 Wishart 分布。这两个密度要求尺度和取值都正定，奇异的整数外积和应从构造本身理解。

<!-- bilingual-en:start -->
## Apply the convention to a software call
<!-- bilingual-en:end -->

*把约定落实到函数调用*

<!-- bilingual-en:start -->
Read the generating law or density first, check the stated mean second, and only then map the degrees of freedom and matrix to the function's argument names. Official interfaces illustrate why this order matters:
<!-- bilingual-en:end -->

先读生成式或密度，再核对期望，最后才把自由度和矩阵放入函数的具体参数。官方接口正好说明这种核对顺序为什么必要：

<!-- bilingual-en:start -->
| Interface | Meaning |
|---|---|
| R rWishart(n, df, Sigma) | Sigma is covariance scale; df is degrees of freedom; n counts returned matrices. |
| SciPy wishart(df=..., scale=...) | scale is $V$ in the displayed density. Its documentation writes $W_p(\nu,V)$. |
| PyTorch Wishart(df, covariance_matrix=V) or Wishart(df, precision_matrix=Omega) | Supply exactly one matrix representation; precision_matrix is $V^{-1}$. |
<!-- bilingual-en:end -->

R 的 Sigma 是协方差型尺度，df 是自由度，n 是要返回的矩阵个数。SciPy 的 scale 是上述密度中的 $V$，其文档采用自由度在前的记号。PyTorch 的协方差与精度两种矩阵表示只能选一种输入；precision_matrix 是 $V^{-1}$。

> [!warning] Mathematical existence and numerical sampling · 分布存在与数值抽样
> <!-- bilingual-en:start -->
> A nonsingular Wishart density exists for every real $\nu>p-1$, but a sampler's accepted inputs and numerical reliability are implementation-specific. In particular, a theoretically positive-definite draw can be returned as numerically singular near the boundary. Check the current interface documentation as well as the mathematical convention.
> <!-- bilingual-en:end -->
>
> 非奇异 Wishart 密度对每个实数 $\nu>p-1$ 都存在，但抽样器接受什么输入、数值上是否可靠，要看具体实现。特别是在边界附近，理论上正定的抽样结果也可能被数值算法返回为奇异矩阵。数学约定和当前接口文档都需要核对。

## 来源与核验

<!-- bilingual-en:start -->
- [R, stats::rWishart](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/rWishart.html): scale-first notation, covariance-scale construction, mean, and the separate meanings of output count and degrees of freedom.
- [SciPy, scipy.stats.wishart, Notes](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wishart.html): degrees-first notation, exact covariance-scale density, inverse-Wishart relation, and numerical sampling caveat. Its [official implementation, parameter check and mean](https://github.com/scipy/scipy/blob/v1.18.0/scipy/stats/_multivariate.py#L2916) confirms $E[W]=\nu V$ throughout the accepted real domain $\nu>p-1$.
- [PyTorch, Wishart](https://docs.pytorch.org/docs/2.14/distributions.html#torch.distributions.wishart.Wishart): mutually exclusive covariance/precision inputs, their inverse relation, real-degree domain, and numerical caveat.
- [Purdue, DasGupta, Graduate Probability, Definition 1.76, PDF p. 162 / printed p. 172](https://www.stat.purdue.edu/~dasgupta/gradprob.pdf#page=162): covariance-scale density and determinant/trace positions in its stated $k\ge p$ domain. The real-degree domain is checked separately against the software references above.
- [[01_Math/04_多元统计分析/04_多元正态分布The Multivariate Normal Distribution.md#1.6. Wishart 分布|Multivariate Statistics §1.6.1]]: the course's degrees-first, covariance-scale convention. The scalar and diagonal examples and the precision substitution are calculated in this card.
<!-- bilingual-en:end -->

- R 文档：核对尺度在前记号、协方差型尺度构造、均值，以及输出矩阵个数与自由度的区别。
- SciPy 的 Notes：核对自由度在前记号、精确尺度密度、逆 Wishart 关系与数值抽样边界；另核对其官方实现的参数检查和 mean，确认在实数域 $\nu>p-1$ 内仍有 $E[W]=\nu V$。
- PyTorch 的 Wishart：核对协方差与精度输入互斥且互逆、实数自由度域与数值边界。
- Purdue 的定义 1.76，PDF 第 162 页、印刷第 172 页：核对其所列 $k\ge p$ 域内的尺度密度，以及行列式和迹的位置。实数自由度域另由上述软件文档核验。
- 多元统计课程第 1.6.1 节：核对本课自由度在前、协方差型尺度约定。正文直接计算一维、对角例子及精度代换。

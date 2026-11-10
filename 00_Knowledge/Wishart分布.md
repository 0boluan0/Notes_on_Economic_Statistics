---
aliases:
  - "Wishart distribution"
  - "Wishart definition"
  - "Wishart 分布定义"
student_os: knowledge-atom
atom_id: STAT-WISH-001
atom_set: wishart-sample-covariance
atom_type: definition
status: source-checked
part_of:
  - "[[Wishart 分布与样本协方差推断.canvas]]"
implies:
  - "[[Wishart秩与可逆性]]"
related:
  - "[[样本协方差Wishart律]]"
  - "[[协方差矩阵半正定性]]"
---

<!-- bilingual-en:start -->
# Wishart distribution
<!-- bilingual-en:end -->

*Wishart 分布*

<!-- bilingual-en:start -->
With positive integer degrees of freedom—a count of generating vectors—a Wishart distribution describes a random matrix made by adding products of Gaussian coordinates. A density extends the family to noninteger degrees of freedom, as explained below. It is the matrix counterpart of a [[卡方分布|chi-square distribution]]: a chi-square variable adds squares of independent standard normal numbers; a Wishart matrix also keeps the cross-products between different coordinates.
<!-- bilingual-en:end -->

在正整数自由度下，自由度计算生成向量的个数，Wishart 分布描述由 Gaussian 坐标乘积累加而成的随机矩阵。密度还可将这个分布族延拓到非整数自由度，后文会单独解释。它是[[卡方分布]]的矩阵对应物：卡方变量累加独立标准正态数的平方；Wishart 矩阵除了平方，还记录不同坐标的交叉乘积。

<!-- bilingual-en:start -->
Start with one real column vector $z=(a,b)^T$. Its [[矩阵转置|transpose]] $z^T$ is a row, so [[矩阵乘法|multiplying]] the column by the row gives an **outer product**:
<!-- bilingual-en:end -->

先看实列向量 $z=(a,b)^T$。它的[[矩阵转置|转置]] $z^T$ 是一行；将列与行做[[矩阵乘法]]，就得到一个**外积**：

$$
zz^T=
\begin{bmatrix}a\\b\end{bmatrix}
\begin{bmatrix}a&b\end{bmatrix}
=
\begin{bmatrix}a^2&ab\\ab&b^2\end{bmatrix}.
$$

> [!example] Adding two outer products · 两个外积相加
> <!-- bilingual-en:start -->
> For the illustrative vectors $z_1=(1,2)^T$ and $z_2=(2,-1)^T$,
> $$z_1z_1^T+z_2z_2^T
> =\begin{bmatrix}1&2\\2&4\end{bmatrix}
> +\begin{bmatrix}4&-2\\-2&1\end{bmatrix}
> =\begin{bmatrix}5&0\\0&5\end{bmatrix}.$$
> The diagonal adds squares; the off-diagonal adds cross-products, which can be negative. These fixed vectors illustrate the operation. To obtain a Wishart **distribution**, we repeatedly draw the vectors from the Gaussian model below; the resulting whole matrix varies from draw to draw.
> <!-- bilingual-en:end -->
>
> 用 $z_1=(1,2)^T$、$z_2=(2,-1)^T$ 演示，两个外积相加得到上面的矩阵。对角元累加平方，非对角元累加可能为负的交叉乘积。这两个固定向量只演示运算。要得到 Wishart **分布**，需要按下面的 Gaussian 模型反复抽取向量；每次得到的整张矩阵会随抽取而改变。

<!-- bilingual-en:start -->
## The Gaussian construction
<!-- bilingual-en:end -->

*由 Gaussian 向量生成*

<!-- bilingual-en:start -->
Let $p\ge1$ be the number of coordinates and let $\nu\ge1$ be an integer. Take $\nu$ [[独立同分布|independent, identically distributed]] real [[联合Gaussian|Gaussian vectors]] $Z_i\sim N_p(0,\Sigma)$. Here $0$ is the mean vector, and the common [[协方差矩阵|covariance matrix]] $\Sigma$ specifies the coordinate variances and covariances. With the **scale-first convention used here**, define
<!-- bilingual-en:end -->

设每个向量有 $p\ge1$ 个坐标，$\nu\ge1$ 是整数。取 $\nu$ 个[[独立同分布]]的实[[联合Gaussian|Gaussian 向量]] $Z_i\sim N_p(0,\Sigma)$。$0$ 是均值向量，共同的[[协方差矩阵]] $\Sigma$ 指定各坐标的方差与协方差。本卡采用**尺度在前**的记号，定义

$$
W=\sum_{i=1}^{\nu}Z_iZ_i^T
\sim W_p(\Sigma,\nu).
$$

<!-- bilingual-en:start -->
The result $W$ is a random $p\times p$ symmetric matrix. The integer $\nu$ counts the independent vectors and is called the **degrees of freedom**. The scale $\Sigma$ may be any real symmetric positive-semidefinite matrix for this construction, including a singular one. The zero means are part of this central Wishart definition. Arbitrary nonzero means do not give the same central law.
<!-- bilingual-en:end -->

结果 $W$ 是一个随机的 $p\times p$ 对称矩阵。整数 $\nu$ 计算独立向量的个数，称为**自由度**。在这个构造中，尺度 $\Sigma$ 可以是任意实对称半正定矩阵，包括奇异矩阵。均值为零是这里的中心 Wishart 定义的一部分；任意非零均值不能直接得到同一个中心 Wishart 律。

> [!important] What the construction guarantees · 构造直接给出的性质
> <!-- bilingual-en:start -->
> For every real coefficient vector $c$,
> $$c^TWc=\sum_{i=1}^{\nu}(c^TZ_i)^2\ge0.$$
> Thus $W$ is [[半正定矩阵|positive semidefinite]]: every quadratic combination is nonnegative. This is the [[外积和的半正定性|outer-product sum rule]], not a claim that every matrix entry is nonnegative. Also, taking the expectation of each entry gives
> $$E[W]=\nu\Sigma,$$
> because $E[Z_{ij}Z_{ik}]=\Sigma_{jk}$ for centered vectors. The scale is the covariance of each generating vector; it is not the mean matrix of $W$.
> <!-- bilingual-en:end -->
>
> 对每个实系数向量 $c$，上面的二次型都是平方和，所以 $W$ [[半正定矩阵|半正定]]。这是[[外积和的半正定性]]，并不是说每个矩阵元素都非负。对每个元素分别取期望，又得到 $E[W]=\nu\Sigma$，因为中心向量满足 $E[Z_{ij}Z_{ik}]=\Sigma_{jk}$。尺度是每个生成向量的协方差，不是 $W$ 自身的均值矩阵。

<!-- bilingual-en:start -->
When $p=1$ and $\Sigma=\sigma^2>0$, each $Z_i/\sigma$ is standard normal, so $W/\sigma^2\sim\chi_\nu^2$. In several dimensions, $W$ still contains a **sum**: no division by $\nu$ or subtraction of an estimated mean is built into the definition. The [[样本协方差Wishart律|sample-covariance Wishart law]] explains how centering observations and dividing by $n-1$ connect a sample covariance to this sum.
<!-- bilingual-en:end -->

$p=1$、$\Sigma=\sigma^2>0$ 时，每个 $Z_i/\sigma$ 都是标准正态变量，因此 $W/\sigma^2\sim\chi_\nu^2$。多维定义仍然是一个**和**，并未自动除以 $\nu$，也没有减去估计均值。[[样本协方差Wishart律]]另外解释：观测中心化、除以 $n-1$ 后的样本协方差，如何与这个和联系起来。

> [!note]- Real degrees of freedom use a density · 实数自由度由密度定义
> <!-- bilingual-en:start -->
> With a [[正定矩阵|positive-definite]] scale $\Sigma$, the nonsingular family extends to every real $\nu>p-1$. For a possible positive-definite symmetric value $w$, its density has the form
> $$f(w)=C\,|w|^{(\nu-p-1)/2}
> \exp\!\left[-\tfrac12\operatorname{tr}(\Sigma^{-1}w)\right].$$
> Here $|w|$ is the [[行列式|determinant]], $\operatorname{tr}$ adds the diagonal entries, $\Sigma^{-1}$ is the [[逆矩阵|inverse]], and $C>0$ depends on $p,\nu,\Sigma$ so the density integrates to one over the independent entries of symmetric matrices. The [[Wishart参数化|parameterization card]] makes the scale dependence explicit.
>
> For an integer $\nu\ge p$, this density describes the same law as the construction above. For noninteger $\nu$, such as $\nu=1.5$ when $p=2$, the density defines the law; there are not “1.5 generating vectors.” Conversely, an integer $1\le\nu<p$ still gives a valid outer-product sum, but it is singular and has no such full-dimensional density. The [[Wishart秩与可逆性|rank and invertibility conditions]] distinguish these cases. The displayed density also requires positive-definite scale; it does not cover the singular-scale construction.
> <!-- bilingual-en:end -->
>
> 尺度 $\Sigma$ [[正定矩阵|正定]]时，非奇异分布族可延拓到任意实数 $\nu>p-1$。对正定对称取值 $w$，密度如上。$|w|$ 是[[行列式]]，$\operatorname{tr}$ 表示对角元之和，$\Sigma^{-1}$ 是[[逆矩阵]]；正常数 $C$ 依赖 $p,\nu,\Sigma$，使密度对对称矩阵的独立元素积分为一。[[Wishart参数化]]会把尺度相关部分写得更明确。
>
> 整数 $\nu\ge p$ 时，这个密度与前面的构造描述同一个分布。非整数时，例如 $p=2,\nu=1.5$，由密度定义分布，并不存在“1.5 个生成向量”。反过来，整数 $1\le\nu<p$ 的外积和仍然有效，但矩阵奇异，没有上述全维密度。[[Wishart秩与可逆性]]区分这些情形。这里的密度还要求尺度正定，不覆盖奇异尺度的构造。

## 来源与核验

<!-- bilingual-en:start -->
- [R, stats::rWishart, Details](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/rWishart.html): Gaussian cross-product construction, scale-first notation, $E[W]=\nu\Sigma$, and the scalar chi-square case.
- [Stanford STATS 305C, One sample problem, “Wishart distribution”](https://web.stanford.edu/class/stats305c/lectures/Onesample.html#wishart-distribution): independent centered Gaussian vectors, their outer-product sum, and its sample-covariance connection.
- [Stan Functions Reference, “Wishart distribution”](https://mc-stan.org/docs/functions-reference/covariance_matrix_distributions.html#wishart-distribution): positive-definite support and scale, real domain $\nu>p-1$, and the density kernel. Only this kernel and domain are used here.
- [MIT 6.436J, Lecture 14, Definitions 2–3 and Theorem 1, pp. 3–6](https://ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/ceb423b6a703abd60429367b733f45bf_MIT6_436JF18_lec14.pdf#page=3): Gaussian generating factors and singular covariance; these justify allowing positive-semidefinite scale in the integer construction.
- [[01_Math/04_多元统计分析/04_多元正态分布The Multivariate Normal Distribution.md#1.6. Wishart 分布|Multivariate Statistics §1.6.1]]: course construction and scalar special case. The course writes degrees of freedom first; the same scale convention is used here with the argument order reversed. The two-vector arithmetic is calculated explicitly above.
<!-- bilingual-en:end -->

- R 文档 Details：核对 Gaussian 乘积构造、尺度在前记号、均值和一元卡方特例。
- Stanford 第 “Wishart distribution” 节：核对独立中心 Gaussian 外积和及其与样本协方差的联系。
- Stan 第 “Wishart distribution” 节：核对正定支持与尺度、实数域 $\nu>p-1$ 和密度核；本卡只使用该处的核与参数域。
- MIT 6.436J 第 14 讲第 3–6 页：核对 Gaussian 生成因子与奇异协方差，支持整数构造允许半正定尺度。
- 多元统计课程第 1.6.1 节：核对本课构造与一元特例。课程把自由度写在前面；本卡仅调换参数顺序，尺度含义一致。两个向量的算术在正文直接展开。

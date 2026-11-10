---
aliases:
  - "多元正态马氏平方的卡方律"
  - "Gaussian Mahalanobis chi-square law"
student_os: knowledge-atom
atom_id: STAT-MVN-005
atom_set: multivariate-normal
atom_type: theorem-boundary
status: source-checked
requires:
  - "[[Gaussian密度与奇异支撑]]"
  - "[[Gaussian仿射闭包]]"
  - "[[正定逆平方根]]"
  - "[[马氏距离]]"
part_of:
  - "[[多元正态分布.canvas]]"
related:
  - "[[联合Gaussian诊断]]"
  - "[[单样本Hotelling T²]]"
  - "[[样本协方差Wishart律]]"
  - "[[正态均值协方差独立]]"
  - "[[Wishart抽样假设]]"
---

<!-- bilingual-en:start -->
# The Gaussian Mahalanobis chi-square law

For a Gaussian observation, measuring displacement from the population mean in units of its population covariance gives a precise probability law: the **squared** distance is chi-square. This combines the geometry of [[马氏距离|Mahalanobis distance]] with the Gaussian distribution. The mean and covariance in this result are the true, fixed population parameters.
<!-- bilingual-en:end -->

*Gaussian 马氏平方的卡方律*

对一个 Gaussian 观测，用总体协方差作尺度来衡量它偏离总体均值的程度，会得到一个精确的概率规律：距离的**平方**服从卡方分布。这把[[马氏距离]]的几何含义与 Gaussian 分布联系起来。这里使用的均值和协方差是真实、固定的总体参数。

> [!example]
> <!-- bilingual-en:start -->
> Suppose independent [[正态分布|normal variables]] $X_1\sim N(10,4)$ and $X_2\sim N(20,9)$ have standard deviations $2$ and $3$. Standardize each displacement:
> $$Z_1=\frac{X_1-10}{2},\qquad Z_2=\frac{X_2-20}{3}.$$
> Then $Z_1,Z_2$ are independent standard normals, and the squared Mahalanobis distance is
> $$D^2=\frac{(X_1-10)^2}{4}+\frac{(X_2-20)^2}{9}=Z_1^2+Z_2^2.$$
> By definition, the sum of squares of $p$ independent standard normals has the **[[卡方分布|chi-square distribution]] with $p$ degrees of freedom**, written $\chi_p^2$. Here $D^2\sim\chi_2^2$. An observed point $(12,23)$ is one standard deviation above each mean, so its squared distance is $2$ and its distance is $\sqrt2$.
> <!-- bilingual-en:end -->
>
> 设[[正态分布|正态变量]] $X_1\sim N(10,4)$ 与 $X_2\sim N(20,9)$ 相互独立，标准差分别为 $2$ 和 $3$。先把每个偏差标准化：
> $$Z_1=\frac{X_1-10}{2},\qquad Z_2=\frac{X_2-20}{3}.$$
> 于是 $Z_1,Z_2$ 为独立标准正态，马氏距离的平方为
> $$D^2=\frac{(X_1-10)^2}{4}+\frac{(X_2-20)^2}{9}=Z_1^2+Z_2^2.$$
> 按定义，$p$ 个独立标准正态的平方和服从**自由度为 $p$ 的[[卡方分布]]**，记为 $\chi_p^2$。因此这里 $D^2\sim\chi_2^2$。观测点 $(12,23)$ 的两个坐标都比均值高一个标准差，所以距离平方为 $2$，距离为 $\sqrt2$。

<!-- bilingual-en:start -->
## Removing scale and correlation together

Let $X\sim N_p(\mu,\Sigma)$ be a [[联合Gaussian|jointly Gaussian]] vector with $p\ge1$ real coordinates. Assume its [[协方差矩阵|covariance matrix]] $\Sigma$ is [[正定矩阵|positive definite]]: every nonzero linear combination of the centered coordinates has positive variance. This ensures that $\Sigma^{-1}$ exists.
<!-- bilingual-en:end -->

*同时消除尺度和相关性*

设 $X\sim N_p(\mu,\Sigma)$ 是具有 $p\ge1$ 个实坐标的[[联合Gaussian|联合 Gaussian]] 向量。假设[[协方差矩阵]] $\Sigma$ [[正定矩阵|正定]]，即中心化坐标的每个非零线性组合都有正方差；因此 $\Sigma^{-1}$ 存在。

> [!important]
> <!-- bilingual-en:start -->
> $$D^2=(X-\mu)^T\Sigma^{-1}(X-\mu)\sim\chi_p^2.$$
>
> The matrix, center, and number of coordinates must match the stated Gaussian model. This is a law for a random observation $X$; after observing $X=x$, the computed value is a number.
> <!-- bilingual-en:end -->
>
> $$D^2=(X-\mu)^T\Sigma^{-1}(X-\mu)\sim\chi_p^2.$$
>
> 矩阵、中心和坐标数量必须与所述 Gaussian 模型对应。这描述随机观测 $X$ 的分布；当已经观察到 $X=x$ 时，计算得到的是一个数值。

<!-- bilingual-en:start -->
The [[正定逆平方根|positive-definite inverse square root]] $\Sigma^{-1/2}$ is the symmetric matrix that rescales each covariance eigen-direction by the reciprocal of its standard deviation. Set
$$Z=\Sigma^{-1/2}(X-\mu).$$
By [[Gaussian仿射闭包|Gaussian affine closure]], $Z$ remains Gaussian, with mean $0$ and covariance
$$\Sigma^{-1/2}\Sigma\Sigma^{-1/2}=I_p.$$
Here $I_p$ is the identity matrix. Its diagonal entries give variance $1$ for every coordinate and its off-diagonal entries give zero cross-covariances. [[联合高斯独立判据|Joint Gaussianity converts these zero cross-covariances into independence]]. Finally,
$$D^2=Z^TZ=\sum_{j=1}^p Z_j^2\sim\chi_p^2.$$
This operation is called whitening. The proof explains why the degrees of freedom count independent random directions.
<!-- bilingual-en:end -->

[[正定逆平方根]] $\Sigma^{-1/2}$ 是一个对称矩阵，沿协方差的每个特征方向，按该方向标准差的倒数重新缩放。令
$$Z=\Sigma^{-1/2}(X-\mu).$$
由 [[Gaussian仿射闭包|Gaussian 仿射闭包]]，$Z$ 仍 Gaussian，均值为 $0$，协方差为
$$\Sigma^{-1/2}\Sigma\Sigma^{-1/2}=I_p.$$
这里 $I_p$ 是单位矩阵，对角元说明各坐标方差为 $1$，非对角元说明跨坐标协方差为零。[[联合高斯独立判据|联合 Gaussian 把这些零协方差转成独立性]]。最后，
$$D^2=Z^TZ=\sum_{j=1}^p Z_j^2\sim\chi_p^2.$$
这个操作称为白化。证明也说明，自由度数的是相互独立的随机变化方向。

<!-- bilingual-en:start -->
The law also converts a geometric cutoff into a probability. For $c\ge0$, the ellipsoidal region $(x-\mu)^T\Sigma^{-1}(x-\mu)\le c$ contains the Gaussian observation with probability $P(D^2\le c)=F_{\chi_p^2}(c)$, where $F_{\chi_p^2}$ is the chi-square [[累积分布函数|CDF]], the probability accumulated up to $c$. Choosing its 95th percentile as $c$ gives 95% population coverage under these known-parameter assumptions.
<!-- bilingual-en:end -->

这个分布还把几何阈值转成概率。对 $c\ge0$，椭球区域 $(x-\mu)^T\Sigma^{-1}(x-\mu)\le c$ 包含 Gaussian 观测的概率为 $P(D^2\le c)=F_{\chi_p^2}(c)$，其中 $F_{\chi_p^2}$ 是卡方的[[累积分布函数|CDF]]，表示累积到 $c$ 的概率。取其第 95 百分位作为 $c$，就在这些已知总体参数的条件下得到 95% 的总体覆盖概率。

<!-- bilingual-en:start -->
## A singular covariance changes the number of directions

If $\Sigma$ has rank $r<p$, the Gaussian vector varies only in an $r$-dimensional [[Gaussian密度与奇异支撑|support]], and the ordinary inverse does not exist. For $r\ge1$, replacing it with the [[Moore–Penrose 伪逆|Moore–Penrose pseudoinverse]] gives
$$
(X-\mu)^T\Sigma^+(X-\mu)\sim\chi_r^2.
$$
In covariance eigen-coordinates, the $r$ positive eigenvalues contribute $r$ standardized normal squares; coordinates with zero variance contribute zero. If $r=0$, $X=\mu$ almost surely and the quadratic form is identically zero. One may call this the degenerate $\chi_0^2$ convention, but it is not a nondegenerate chi-square density.

For example, $X=(Z,2Z)^T$ with $Z\sim N(0,1)$ has
$$
\Sigma=\begin{pmatrix}1&2\\2&4\end{pmatrix},\qquad
\Sigma^+=\frac1{25}\begin{pmatrix}1&2\\2&4\end{pmatrix}.
$$
Its quadratic form is $Z^2\sim\chi_1^2$, despite the two displayed coordinates.
<!-- bilingual-en:end -->

*协方差奇异会改变随机方向的数量*

若 $\Sigma$ 的秩为 $r<p$，Gaussian 向量只在 $r$ 维[[Gaussian密度与奇异支撑|支撑]]上变化，普通逆矩阵不存在。对于 $r\ge1$，换用 [[Moore–Penrose 伪逆]]，有
$$
(X-\mu)^T\Sigma^+(X-\mu)\sim\chi_r^2.
$$
在协方差的特征坐标中，$r$ 个正特征值贡献 $r$ 个标准化正态平方，零方差坐标贡献零。若 $r=0$，则 $X=\mu$ 几乎处处成立，二次型恒等于零；可以约定称其为退化的 $\chi_0^2$，但它不是具有非退化密度的卡方分布。

例如，$X=(Z,2Z)^T$、$Z\sim N(0,1)$ 时，
$$
\Sigma=\begin{pmatrix}1&2\\2&4\end{pmatrix},\qquad
\Sigma^+=\frac1{25}\begin{pmatrix}1&2\\2&4\end{pmatrix}.
$$
其二次型就是 $Z^2\sim\chi_1^2$，尽管表面上写了两个坐标。

<!-- bilingual-en:start -->
## Estimating the center or scale changes the problem

Replacing $\mu,\Sigma$ with a [[样本均值向量|sample mean]] $\bar x$ and a [[样本协方差矩阵|sample covariance]] $S$ produces fitted distances
$$d_i^2=(x_i-\bar x)^TS^{-1}(x_i-\bar x),$$
provided $S$ is invertible. They share estimated quantities and fitting constraints, so the known-parameter proof no longer gives independent exact $\chi_p^2$ variables.

A one-dimensional example makes the difference visible. With two distinct observations $x_1,x_2$, use the sample variance with denominator $n-1=1$:
$$\bar x=\frac{x_1+x_2}{2},\qquad S=\frac{(x_1-x_2)^2}{2}.$$
Both fitted squared distances equal $1/2$. They cannot each have a $\chi_1^2$ distribution, which is not constant.

An independent training sample removes dependence between a new observation and the fitted parameters, but does not make those parameters fixed. For example, let a new observation and $m\ge1$ training observations be independent draws from the same $N(\mu,\sigma^2)$ law, with known $\sigma^2>0$. Estimating the mean by the training average gives
$$\frac{(X_{\mathrm{new}}-\bar X_{\mathrm{train}})^2}{\sigma^2}\sim\left(1+\frac1m\right)\chi_1^2.$$
The extra variance $\sigma^2/m$ comes from the estimated mean. Mean inference using [[单样本Hotelling T²|Hotelling’s statistic]] and distance-based diagnostics therefore require their own finite-sample calibration; a chi-square approximation must be identified as an approximation.
<!-- bilingual-en:end -->

*估计中心或尺度会改变问题*

用[[样本均值向量|样本均值]] $\bar x$ 与[[样本协方差矩阵|样本协方差]] $S$ 代替 $\mu,\Sigma$，得到拟合后的距离
$$d_i^2=(x_i-\bar x)^TS^{-1}(x_i-\bar x),$$
前提是 $S$ 可逆。这些距离共享估计量和拟合约束，所以已知参数的证明不再给出相互独立且精确服从 $\chi_p^2$ 的变量。

一个一维例子就能看到差别。若只有两个不同的观测 $x_1,x_2$，样本方差使用分母 $n-1=1$，则
$$\bar x=\frac{x_1+x_2}{2},\qquad S=\frac{(x_1-x_2)^2}{2}.$$
两个拟合距离的平方都等于 $1/2$，显然不可能各自服从并非常数的 $\chi_1^2$ 分布。

独立训练样本能使新观测与估计量独立，却不能把估计量变成固定参数。例如，设新观测和 $m\ge1$ 个训练观测都独立抽自同一个 $N(\mu,\sigma^2)$ 分布，已知 $\sigma^2>0$。用训练平均值估计总体均值，会得到
$$\frac{(X_{\mathrm{new}}-\bar X_{\mathrm{train}})^2}{\sigma^2}\sim\left(1+\frac1m\right)\chi_1^2.$$
额外方差 $\sigma^2/m$ 来自估计均值。因此，[[单样本Hotelling T²|Hotelling 统计量]]的均值推断和基于距离的诊断都需要各自的有限样本校准；若使用卡方近似，就应明确它只是近似。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/04_多元统计分析/04_多元正态分布The Multivariate Normal Distribution.md#1.4. 二次型与相关分布|Multivariate statistics course, §1.4]]: the known-parameter quadratic form and chi-square law. [[01_Math/04_多元统计分析/04_多元正态分布The Multivariate Normal Distribution.md#1.8.2. 多变量正态性检查|§1.8.2]] uses chi-square quantiles as an approximate graphical reference for fitted distances.
- [MIT 6.436J Lecture 14, pp. 3–7](https://ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/ceb423b6a703abd60429367b733f45bf_MIT6_436JF18_lec14.pdf): Gaussian square-root constructions, affine closure, and independence after whitening.
- [MIT 18.443 Lecture 4, p. 3](https://ocw.mit.edu/courses/18-443-statistics-for-applications-fall-2006/da928274a46632ce0d35d2012a1c8e74_lecture4.pdf): spectral coordinates and singular Gaussian support; applying the reciprocal of each positive covariance eigenvalue gives the rank-based extension proved above.
- [NIST/SEMATECH, Chi-Square Distribution](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3666.htm): definition as a sum of squared independent standard normals. The diagonal, rank-one, two-observation, and independent-training examples are direct calculations in this card.
<!-- bilingual-en:end -->

- [[01_Math/04_多元统计分析/04_多元正态分布The Multivariate Normal Distribution.md#1.4. 二次型与相关分布|多元统计课程 §1.4]]：已知总体参数的二次型与卡方律。[[01_Math/04_多元统计分析/04_多元正态分布The Multivariate Normal Distribution.md#1.8.2. 多变量正态性检查|§1.8.2]] 把卡方分位数用于拟合距离的近似图形参照。
- [MIT 6.436J Lecture 14，第 3–7 页](https://ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/ceb423b6a703abd60429367b733f45bf_MIT6_436JF18_lec14.pdf)：Gaussian 平方根构造、仿射闭包及白化后的独立性。
- [MIT 18.443 Lecture 4，第 3 页](https://ocw.mit.edu/courses/18-443-statistics-for-applications-fall-2006/da928274a46632ce0d35d2012a1c8e74_lecture4.pdf)：谱坐标与奇异 Gaussian 支撑；对正协方差特征值取倒数，即得到正文推导的秩版本。
- [NIST/SEMATECH, Chi-Square Distribution](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3666.htm)：独立标准正态平方和的卡方定义。对角、秩一、两个观测和独立训练样本例子均由正文直接计算。

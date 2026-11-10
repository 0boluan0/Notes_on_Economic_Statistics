---
aliases:
  - "Wishart law of sample covariance"
  - "样本协方差的 Wishart 分布"
student_os: knowledge-atom
atom_id: STAT-WISH-002
atom_set: wishart-sample-covariance
atom_type: theorem
status: source-checked
requires:
  - "[[Wishart分布]]"
  - "[[样本协方差矩阵]]"
part_of:
  - "[[Wishart 分布与样本协方差推断.canvas]]"
related:
  - "[[正态均值协方差独立]]"
  - "[[Wishart抽样假设]]"
---

<!-- bilingual-en:start -->
# The Wishart law of sample covariance
<!-- bilingual-en:end -->

*样本协方差的 Wishart 抽样律*

<!-- bilingual-en:start -->
A [[样本协方差矩阵|sample covariance matrix]] changes from sample to sample. Under independent sampling from one multivariate normal population, the **Wishart law** describes that entire random matrix, including its off-diagonal entries. It is the matrix counterpart of the [[卡方分布|chi-square law]] for a sum of independent squared standard normals.
<!-- bilingual-en:end -->

[[样本协方差矩阵]]会随抽到的样本改变。如果各次观测独立来自同一个多元正态总体，**Wishart 律**便描述这整个随机矩阵，包括非对角元素。它对应于标量情形中独立标准正态平方和的[[卡方分布]]。

<!-- bilingual-en:start -->
Take $n\ge2$ observations, each a $p$-coordinate column vector with $p\ge1$, and assume
$$X_1,\ldots,X_n\overset{\mathrm{iid}}{\sim}N_p(\mu,\Sigma),\qquad \Sigma\succ0.$$
Here [[独立同分布|iid]] means independence **between observations** and the same [[联合Gaussian|joint Gaussian law]] for each vector; the coordinates within one vector may be dependent. The fixed population mean is $\mu$ and the positive-definite covariance is $\Sigma$.
<!-- bilingual-en:end -->

取 $n\ge2$ 次观测，每次是 $p\ge1$ 个坐标组成的列向量，并假设上式成立。[[独立同分布|iid]]要求**不同观测之间**独立，每个向量具有相同的[[联合Gaussian|联合 Gaussian 分布]]；一次观测内的各坐标可以相关。固定的总体均值为 $\mu$，正定协方差为 $\Sigma$。

<!-- bilingual-en:start -->
The [[样本均值向量|sample mean]] is $\bar X=n^{-1}\sum_iX_i$. Define the unnormalized centered scatter $C$ and the sample covariance $S$ by
$$C=\sum_{i=1}^n(X_i-\bar X)(X_i-\bar X)^T,\qquad S=\frac{C}{n-1}.$$
Each outer product multiplies every coordinate deviation by every other deviation; $C$ adds these $p\times p$ tables. In the [[Wishart参数化|scale-first convention]], $W_p(\Sigma,m)$ means the law of $\sum_{j=1}^m Z_jZ_j^T$ for independent $Z_j\sim N_p(0,\Sigma)$ when $m$ is a positive integer.
<!-- bilingual-en:end -->

[[样本均值向量]]是 $\bar X=n^{-1}\sum_iX_i$。上式定义未除分母的中心化散布矩阵 $C$，以及样本协方差 $S=C/(n-1)$。每个外积把坐标偏差两两相乘，$C$ 汇总这些 $p\times p$ 表。在[[Wishart参数化|尺度在前的约定]]下，对正整数 $m$，$W_p(\Sigma,m)$ 指独立 $Z_j\sim N_p(0,\Sigma)$ 的外积和 $\sum_{j=1}^mZ_jZ_j^T$ 的分布。

> [!important] The sampling law · 抽样分布
> <!-- bilingual-en:start -->
> $$C=(n-1)S\sim W_p(\Sigma,n-1).$$
> The degrees of freedom are $n-1$, and the scale is the population covariance $\Sigma$. This is an exact finite-sample statement, not a large-sample approximation.
> <!-- bilingual-en:end -->
>
> 自由度是 $n-1$，尺度是总体协方差 $\Sigma$。这是精确的有限样本结论，不是大样本近似。

<!-- bilingual-en:start -->
## Why centering removes one direction
<!-- bilingual-en:end -->

*为什么中心化少一个方向*

<!-- bilingual-en:start -->
The raw residuals $X_i-\bar X$ sum to zero, so they are not independent. The proof instead replaces them with **contrasts**: weighted sums across observations whose weights add to zero. For two observations there is one normalized contrast,
$$D=\frac{X_1-X_2}{\sqrt2}\sim N_p(0,\Sigma),\qquad
C=DD^T.$$
Indeed, the two residuals are $(X_1-X_2)/2$ and its negative; adding their outer products gives the displayed identity. One independent Gaussian contrast produces one Wishart degree of freedom.
<!-- bilingual-en:end -->

原始残差 $X_i-\bar X$ 相加为零，所以并不独立。证明换用**对比**：跨观测取加权和，且权重之和为零。两次观测时只有一个标准化对比 $D=(X_1-X_2)/\sqrt2$，它服从 $N_p(0,\Sigma)$。两个原始残差分别是 $(X_1-X_2)/2$ 及其相反数，外积相加恰好等于 $DD^T$。一个独立 Gaussian 对比产生一个 Wishart 自由度。

> [!example] Two observations can give a singular matrix · 两次观测的矩阵可以奇异
> <!-- bilingual-en:start -->
> If the observed vectors are $x_1=(1,2)^T$ and $x_2=(3,4)^T$, then $\bar x=(2,3)^T$, the deviations are $(-1,-1)^T,(1,1)^T$, and
> $$c=\begin{bmatrix}2&2\\2&2\end{bmatrix}=dd^T,\qquad d=(-\sqrt2,-\sqrt2)^T.$$
> This is one possible scatter matrix. Under the stated Gaussian sampling model, its random counterpart has law $W_2(\Sigma,1)$ and rank one almost surely, even though the population covariance is positive definite.
> <!-- bilingual-en:end -->
>
> 观测均值为 $(2,3)^T$，两个偏差为 $(-1,-1)^T,(1,1)^T$，外积和如上。这里展示一个可能算出的散布矩阵；在给定 Gaussian 抽样模型下，对应随机矩阵服从 $W_2(\Sigma,1)$，几乎必然秩为一，尽管总体协方差正定。

<!-- bilingual-en:start -->
For general $n$, let $X_{\mathrm{data}}$ have rows $X_i^T$ and let the [[中心化矩阵|centering matrix]] be $H=I_n-\mathbf1\mathbf1^T/n$. Its range is the $(n-1)$-dimensional space of zero-sum observation weights. Choose an [[标准正交组|orthonormal basis]] of that space (perpendicular unit vectors spanning it) and collect its vectors as the columns of an $n\times(n-1)$ matrix $G$. Then
$$G^TG=I_{n-1},\qquad GG^T=H,\qquad G^T\mathbf1=0.$$
The rows of $R=G^TX_{\mathrm{data}}$ are the contrasts. They have mean zero, covariance $\Sigma$ each, and zero cross-covariance between distinct rows. Because they are jointly Gaussian, [[联合高斯独立判据|zero cross-covariance makes these rows independent]]. Finally,
$$C=X_{\mathrm{data}}^THX_{\mathrm{data}}=R^TR=\sum_{j=1}^{n-1}R_jR_j^T,$$
where $R_j$ denotes the column version of row $j$. This is exactly the [[Wishart分布|Wishart construction]].
<!-- bilingual-en:end -->

一般地，把 $X_i^T$ 排成数据矩阵的各行，并用[[中心化矩阵]] $H=I_n-\mathbf1\mathbf1^T/n$。其像空间是 $n-1$ 维的观测权重和为零的空间。取一组[[标准正交组|标准正交基]]（张成该空间、两两垂直且长度为一的向量）作为 $n\times(n-1)$ 矩阵 $G$ 的列，就得到上面的三个恒等式。$R=G^TX_{\mathrm{data}}$ 的各行是这些对比：均值为零，各有协方差 $\Sigma$，不同行之间交叉协方差为零。它们仍联合 Gaussian，因此[[联合高斯独立判据|零交叉协方差推出行之间独立]]。最后的恒等式把散布矩阵写成 $n-1$ 个独立 Gaussian 向量的外积和，恰好是[[Wishart分布]]的构造；$R_j$ 在这里指第 $j$ 行转成的列向量。

<!-- bilingual-en:start -->
## Scaling, rank and the meaning of the theorem
<!-- bilingual-en:end -->

*分母、秩与适用范围*

<!-- bilingual-en:start -->
If the empirical covariance instead uses denominator $n$, write $\widetilde S=C/n$. Then $n\widetilde S\sim W_p(\Sigma,n-1)$: changing the divisor does not restore the removed mean direction. Calling $\widetilde S$ the Gaussian covariance MLE additionally requires a full-rank scatter matrix in the positive-definite model; see [[Gaussian协方差MLE边界|the MLE existence boundary]].
<!-- bilingual-en:end -->

若经验协方差改用分母 $n$，记作 $\widetilde S=C/n$，则 $n\widetilde S\sim W_p(\Sigma,n-1)$。改分母并不会把已经移除的均值方向恢复回来。在正定模型内把它称为 Gaussian 协方差 MLE，还要求散布矩阵满秩，见[[Gaussian协方差MLE边界]]。

<!-- bilingual-en:start -->
When $n-1<p$, this is the singular outer-product Wishart law; the matrix identity remains valid although there is no ordinary density on the positive-definite matrices. With $\Sigma\succ0$, $S$ is invertible almost surely exactly when $n-1\ge p$; [[Wishart秩与可逆性|rank and density conventions]] must be checked before using $S^{-1}$. For $p=1$, the statement becomes $(n-1)S/\sigma^2\sim\chi^2_{n-1}$, where $S$ is the scalar sample variance. Equal means and covariances alone do not imply this distribution; see [[Wishart抽样假设|the sampling-assumption boundary]].
<!-- bilingual-en:end -->

当 $n-1<p$ 时，这里指外积构造的奇异 Wishart 律；没有正定矩阵空间上的普通密度，并不影响分布等式成立。在 $\Sigma\succ0$ 下，$S$ 几乎必然可逆当且仅当 $n-1\ge p$；使用 $S^{-1}$ 前必须核对[[Wishart秩与可逆性|秩与密度的约定]]。当 $p=1$ 时，结论变成 $(n-1)S/\sigma^2\sim\chi^2_{n-1}$，其中 $S$ 就是标量样本方差。只有相同均值与协方差并不足以得到这个分布，见[[Wishart抽样假设]]。

## 来源与核验

<!-- bilingual-en:start -->
- [Peter Hoff, Multivariate Analysis notes, PDF pp.67–69](https://www2.stat.duke.edu/~pdh10/Teaching/832/Materials/notes.pdf): the independent Gaussian contrast representation and centered scatter Wishart law. The source calls unnormalized scatter $S$; this card calls it $C$.
- [Stanford STATS305C, One sample problem, Wishart distribution](https://web.stanford.edu/class/stats305c/lectures/Onesample.html#wishart-distribution): outer-product definition, scale, and the $n$ versus $n-1$ covariance divisors.
- [[01_Math/04_多元统计分析/04_多元正态分布The Multivariate Normal Distribution.md#1.6. Wishart 分布|Course §1.6]]: course scope and parameter order; the course puts degrees of freedom before scale. Its [[01_Math/04_多元统计分析/04_多元正态分布The Multivariate Normal Distribution.md#1.5.2. 最大似然估计|§1.5.2]] supplies the covariance-MLE rank condition.
<!-- bilingual-en:end -->

- Hoff PDF 第67–69页核对独立 Gaussian 对比与中心化散布的 Wishart 律；原源的 $S$ 是本卡的未归一化 $C$。
- Stanford 核对外积定义、尺度及两种分母；课程 §1.6 的参数顺序为自由度在前，§1.5.2 核对 MLE 的满秩条件。

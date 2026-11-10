---
aliases:
- Multivariate GLS
- GLS for correlated response rows
- 多响应广义最小二乘
student_os: knowledge-atom
atom_id: STAT-MVR-014
atom_set: multivariate-linear-regression
atom_type: estimation-method
status: source-checked
part_of:
- '[[多元线性回归.canvas|多元线性回归]]'
requires:
- '[[多响应线性回归]]'
- '[[观测行协方差]]'
related:
- '[[共享设计的逐响应OLS]]'
- '[[相关行不容独立推断]]'
- '[[随机效应GLS]]'
---

<!-- bilingual-en:start -->
# Multivariate GLS with correlated rows
<!-- bilingual-en:end -->
*相关观测行下的多响应 GLS*

<!-- bilingual-en:start -->
Suppose two measurements share noise, while a third supplies partly independent information. Giving all three equal influence may waste precision. [[广义最小二乘|Generalized least squares (GLS)]] uses a specified error covariance to change how residuals are measured. In a multiple-response model with the same predictors for every response, a **separable** covariance lets the same row transformation be used for every response column.
<!-- bilingual-en:end -->

假设前两次测量共享噪声，第三次提供相对独立的信息。让三次测量影响完全相同，可能没有充分利用精度。[[广义最小二乘|广义最小二乘（GLS）]]利用指定的误差协方差，改变衡量残差的方式。当每个响应使用相同解释变量，且协方差具有**可分离**结构时，同一行变换可以用于所有响应列。

<!-- bilingual-en:start -->
## What is assumed known?
<!-- bilingual-en:end -->
*哪些东西要固定、已知？*

<!-- bilingual-en:start -->
Use the [[多响应线性回归|model]] $Y=XB+U$. There are $n$ observations and $q\ge1$ responses, so $Y,U$ are $n\times q$. The shared design $X$ is $n\times k$ with $1\le k\le n$ and full column rank, meaning its columns are [[线性无关|linearly independent]]; $B$ is a fixed $k\times q$ coefficient matrix. Work with fixed $X$, or condition the whole model on its realised value. Assume zero conditional error mean and finite conditional second moments, with
<!-- bilingual-en:end -->
$$
E(U\mid X)=0,\qquad
\operatorname{Cov}(U_{ia},U_{jb}\mid X)=\Omega_{ij}\Sigma_{ab}.
$$
<!-- bilingual-en:start -->
Here the [[观测行协方差|row factor]] $\Omega$ is a **known**, fixed, symmetric [[正定矩阵|positive-definite]] $n\times n$ matrix. The response factor $\Sigma$ is a fixed positive-definite $q\times q$ model parameter; it may be unknown. “Fixed” means it is not a weight matrix estimated from the observed responses in this derivation. The scalar entries of $U$ have a separable covariance, not an arbitrary full covariance. Gaussianity is unnecessary for the estimator, its unbiasedness and its covariance formula below.
<!-- bilingual-en:end -->

采用[[多响应线性回归|模型]] $Y=XB+U$：$Y,U$ 是 $n\times q$，共同设计 $X$ 是满列秩的 $n\times k$，即各列[[线性无关]]，$1\le k\le n$，$B$ 是固定的 $k\times q$ 系数矩阵。固定 $X$，或对它的实现值作条件分析。假定零条件均值和有限条件二阶矩，并有上面的可分离协方差。[[观测行协方差|行因子]] $\Omega$ 是**已知、固定、对称正定**的 $n\times n$ 矩阵；响应因子 $\Sigma$ 是固定的 $q\times q$ [[正定矩阵|正定]]模型参数，可以未知。这里“固定”排除了用本次响应数据估出来的随机权重。以下估计式、无偏性和协方差式本身不要求 Gaussian 分布。

<!-- bilingual-en:start -->
## Remove the row covariance before fitting
<!-- bilingual-en:end -->
*先把行方向的协方差变成单位矩阵*

<!-- bilingual-en:start -->
Choose the symmetric [[正定逆平方根|inverse square root]] $W=\Omega^{-1/2}$. It rescales and mixes the rows so that $W\Omega W^T=I_n$. Transform **both** the responses and the design:
<!-- bilingual-en:end -->
$$
Y^*=WY,\qquad X^*=WX,\qquad U^*=WU,
\qquad Y^*=X^*B+U^*.
$$
<!-- bilingual-en:start -->
This is row whitening. The transformed errors have covariance $\operatorname{Cov}(U^*_{ia},U^*_{jb}\mid X)=\delta_{ij}\Sigma_{ab}$, where $\delta_{ij}$ is $1$ when $i=j$ and $0$ otherwise. Different transformed rows are uncorrelated, and each has covariance $\Sigma$; independence additionally needs, for example, a [[矩阵正态误差结构|joint Gaussian error model]]. Responses within a row can still be correlated. An intercept column must also be transformed: $W\mathbf1_n$ is not generally a column of ones.
<!-- bilingual-en:end -->

取对称[[正定逆平方根]] $W=\Omega^{-1/2}$，使 $W\Omega W^T=I_n$，并且**同时**变换响应和设计。这叫行方向白化。变换后 $\operatorname{Cov}(U^*_{ia},U^*_{jb}\mid X)=\delta_{ij}\Sigma_{ab}$，其中 $i=j$ 时 $\delta_{ij}=1$，否则为零。因此各行互不相关且各有协方差 $\Sigma$；独立性还需要例如[[矩阵正态误差结构|联合 Gaussian 误差模型]]。同一行内的响应仍可能相关。截距列也必须变换，$W\mathbf1_n$ 一般不是全为 $1$ 的列。

> [!important] One GLS matrix for all response columns
> <!-- bilingual-en:start -->
> Apply [[共享设计的逐响应OLS|OLS with a shared design]] to $(X^*,Y^*)$. Because $W^TW=\Omega^{-1}$ and $X$ has full column rank, the result is
> <!-- bilingual-en:end -->
> $$
> \hat B_{\mathrm{GLS}}
> =(X^T\Omega^{-1}X)^{-1}X^T\Omega^{-1}Y.
> $$
> <!-- bilingual-en:start -->
> The $k\times n$ matrix multiplying $Y$ is the same for every response. No estimate of $\Sigma$ is needed to calculate these coefficients.
> <!-- bilingual-en:end -->
>
> 对 $(X^*,Y^*)$ 做[[共享设计的逐响应OLS|共同设计下的 OLS]]，利用 $W^TW=\Omega^{-1}$ 即得上式。乘在 $Y$ 左边的 $k\times n$ 矩阵对每列响应相同，计算这些系数不需要先估计 $\Sigma$。

<!-- bilingual-en:start -->
The role of $\Sigma$ is clearer from the full quadratic criterion. For a candidate coefficient matrix $D$, let $R(D)=Y-XD$, and let $\operatorname{tr}$ mean the [[迹与特征值总和|trace]], the sum of diagonal entries. Minimizing
<!-- bilingual-en:end -->
$$
Q(D)=\operatorname{tr}\!\left[\Sigma^{-1}R(D)^T\Omega^{-1}R(D)\right]
$$
<!-- bilingual-en:start -->
leads to $X^T\Omega^{-1}(XD-Y)\Sigma^{-1}=0$. Multiplying on the right by $\Sigma$ removes the response factor and gives the displayed GLS estimate. Under a matrix-normal model, the same criterion gives the likelihood estimate of $B$ **for fixed covariance factors**. Jointly estimating an unrestricted covariance adds a separate existence/rank question; this argument does not establish that joint maximum likelihood always exists.
<!-- bilingual-en:end -->

完整二次准则能更清楚地说明 $\Sigma$ 的角色。令候选系数矩阵为 $D$，$R(D)=Y-XD$，$\operatorname{tr}$ 是对角元之和，即[[迹与特征值总和|迹]]。最小化上面的 $Q(D)$，得到 $X^T\Omega^{-1}(XD-Y)\Sigma^{-1}=0$；右乘 $\Sigma$ 后，响应因子消去，所以系数仍是同一 GLS 式。在矩阵正态模型中，这也给出**固定协方差因子时**的 $B$ 似然估计；若还要联合估计不受限协方差，存在性与秩是另一个问题，不能从这里推出联合极大似然一定存在。

> [!example] A correlated pair and one separate reading
> <!-- bilingual-en:start -->
> Take an intercept-only model, so $X=\mathbf1_3$, with known
> <!-- bilingual-en:end -->
> $$
> \Omega=\begin{pmatrix}1&1/2&0\\1/2&1&0\\0&0&1\end{pmatrix},\qquad
> Y=\begin{pmatrix}1&2\\3&4\\8&1\end{pmatrix}.
> $$
> <!-- bilingual-en:start -->
> The first two rows share positive covariance; the third has zero covariance with them. Now $\Omega^{-1}\mathbf1_3=(2/3,2/3,1)^T$, so the GLS weights are $(2/7,2/7,3/7)$. Apply them to each column:
> <!-- bilingual-en:end -->
> $$
> \hat B_{\mathrm{GLS}}=(32/7,\;15/7),\qquad
> \hat B_{\mathrm{OLS}}=(4,\;7/3).
> $$
> <!-- bilingual-en:start -->
> The separate third row gets more weight here. This is a covariance-based result, not a general rule to delete one row from every correlated pair.
> <!-- bilingual-en:end -->
>
> 这里只有截距，$X=\mathbf1_3$。前两行有正协方差，第三行与它们零协方差。由 $\Omega^{-1}\mathbf1_3=(2/3,2/3,1)^T$，归一化权重为 $(2/7,2/7,3/7)$，分别作用到两个响应列，就得到上述 GLS 值；OLS 则对三行等权。这个例子中，第三行的信息较不重复，因而权重更高；它不是“相关两行总要删一行”的规则。

<!-- bilingual-en:start -->
## Precision, equality with OLS, and estimated weights
<!-- bilingual-en:end -->
*精度、与 OLS 相同的情形，以及估出来的权重*

<!-- bilingual-en:start -->
Let $G=(X^T\Omega^{-1}X)^{-1}$. Since the GLS coefficient map sends $X$ to $I_k$, the estimate is conditionally unbiased. For coefficient indices $r,s\in\{1,\ldots,k\}$ and response indices $a,b\in\{1,\ldots,q\}$, its covariance is
<!-- bilingual-en:end -->
$$
\operatorname{Cov}(\hat B_{ra},\hat B_{sb}\mid X)=G_{rs}\Sigma_{ab}.
$$
<!-- bilingual-en:start -->
Equivalently, stacking complete coefficient columns as in the [[矩阵正态误差结构|column-vec convention]] gives $\operatorname{Cov}(\operatorname{vec}\hat B\mid X)=\Sigma\otimes G$. Under the stated covariance model, GLS is best among linear unbiased estimators, meaning its covariance cannot be reduced in a coefficient direction within that class. In the example, the covariance of the estimated two-response mean is $(3/7)\Sigma$, compared with $(4/9)\Sigma$ for OLS; the difference is $(1/63)\Sigma$. Gaussianity adds an exact Gaussian sampling law, not the unbiasedness or covariance calculation itself.
<!-- bilingual-en:end -->

令 $G=(X^T\Omega^{-1}X)^{-1}$。GLS 的系数映射乘 $X$ 得到 $I_k$，所以估计量条件无偏，逐元素协方差如上，其中 $r,s$ 是 $1$ 到 $k$ 的系数索引，$a,b$ 是 $1$ 到 $q$ 的响应索引。按[[矩阵正态误差结构|逐列 vec 约定]]，也可写成 $\Sigma\otimes G$。在明示协方差模型下，GLS 在线性无偏估计类中最优，即该类中不能再降低某个系数方向的方差。例子中，两响应均值估计的协方差是 $(3/7)\Sigma$，OLS 为 $(4/9)\Sigma$，二者相差 $(1/63)\Sigma$。Gaussian 假设额外给出精确 Gaussian 抽样律，而不是这些二阶矩计算的前提。

<!-- bilingual-en:start -->
If $\Omega=cI_n$ with $c>0$, the scalar cancels and GLS equals OLS. This is sufficient, not necessary. With an intercept-only design and $\Omega=I_n+\mathbf1_n\mathbf1_n^T$, every row sum is the same, so $\Omega^{-1}\mathbf1_n$ is proportional to $\mathbf1_n$ and both methods give the ordinary mean, even though the errors are correlated for $n\ge2$. Their coinciding point estimates do not justify independent-row standard errors. More generally, multiplying any $\Omega$ by a positive scalar leaves the GLS coefficients unchanged; this matches the [[观测行协方差|factor-scale ambiguity]].
<!-- bilingual-en:end -->

若 $\Omega=cI_n$、$c>0$，常数抵消，GLS 等于 OLS。这只是充分条件。仅含截距且 $\Omega=I_n+\mathbf1_n\mathbf1_n^T$ 时，各行和相同，使 $\Omega^{-1}\mathbf1_n$ 与 $\mathbf1_n$ 成比例，两种方法都给出普通平均数；但 $n\ge2$ 时误差行确实相关。点估计相同不意味着可使用独立行标准误。一般地，任意 $\Omega$ 乘正数都不改变 GLS 系数，这也符合[[观测行协方差|因子尺度不唯一]]。

<!-- bilingual-en:start -->
Replacing the known $\Omega$ by an estimate from $Y$ produces [[可行GLS|feasible GLS]]. The coefficient map is then data-dependent; fixed-weight unbiasedness and finite-sample Gaussian/$F$/Wishart calibrations do not automatically carry over. Suitable covariance estimation and asymptotic conditions may justify approximate inference, and some first-stage uncertainty can be asymptotically negligible, but this requires a theorem for that setting. If $X$ is rank deficient, the inverse above fails and coefficient uniqueness must be handled through [[最小二乘解唯一性|identifiability and estimable fitted values]]. If the covariance is not separable, general GLS may couple response columns. A wrong conditional mean is not repaired by any of these covariance choices.
<!-- bilingual-en:end -->

若把已知 $\Omega$ 换成用 $Y$ 估计的矩阵，方法就成为[[可行GLS]]。系数映射此时依赖数据，固定权重下的无偏性及有限样本 Gaussian、$F$、Wishart 校准不能自动继承。合适的协方差估计和渐近条件可能支持近似推断，某些第一阶段不确定性在渐近下也可能可忽略，但需要相应定理。若 $X$ 秩亏，上式逆矩阵不存在，须另行处理[[最小二乘解唯一性|系数可识别性与拟合值]]；若协方差不可分离，一般 GLS 可能把响应列耦合起来。任何协方差选择都不能修复错误的条件均值。

## 来源与核验

<!-- bilingual-en:start -->
- [Shalizi, CMU 36-401, Lectures 24–25, §6–7, PDF pp.24–27](https://www.stat.cmu.edu/~cshalizi/mreg/15/lectures/24/lecture-24--25.pdf#page=24): known-covariance whitening, GLS, the linear-unbiased efficiency statement and covariance/mean-model distinction. The multiple-response result is obtained by applying this to the shared transformed design.
- [SciPy `matrix_normal`, Notes](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.matrix_normal.html): the positive-definite matrix-normal density and column-vector covariance order. The displayed normal equation and cancellation of $\Sigma$ are derived directly for fixed factors.
- [Sawyer, *Multivariate Linear Models*, §1–2, PDF pp.1–3](https://www.math.wustl.edu/~sawyer/handouts/manovas.pdf#page=1): common-design multiple-response Gaussian regression and cancellation of a fixed response covariance; here it is applied after known row whitening.
- [MIT 14.32, *More on FGLS and How to Test for Heteroskedasticity*, PDF pp.1–2](https://ocw.mit.edu/courses/14-32-econometrics-spring-2007/c68f54a5cac9d51f45f5916c123a223f_rec_4_20.pdf#page=1): the distinction between known covariance parameters and residual-based estimated weights, with large-sample rather than automatic finite-sample justification. [[02_Economy/01_Econometrics/08_自相关.md#5.1.1. GLS 原理|The local GLS course section]] supplies the course connection.
<!-- bilingual-en:end -->

- CMU 第24–27页核对白化、已知协方差 GLS、线性无偏类内效率与均值错设边界；本卡把共同变换后的设计用于多个响应。
- SciPy 核对矩阵正态密度及逐列 vec 顺序；Sawyer 第1–3页核对共同设计下固定响应协方差的消去。正规方程与三行算例直接计算。
- MIT 第1–2页区分已知参数与从残差估出的权重；本卡只沿用这个分工及大样本口径，不把未知权重当作已知精确推断。永久课程的 GLS 小节保留课程衔接。

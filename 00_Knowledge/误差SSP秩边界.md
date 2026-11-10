---
aliases:
  - Error SSP rank bound
  - Residual SSCP rank bound
  - high-dimensional response rank boundary
student_os: knowledge-atom
atom_id: STAT-MVR-008
atom_set: multivariate-linear-regression
atom_type: theorem-boundary
status: source-checked
requires:
  - "[[多响应残差与误差SSP]]"
  - "[[矩阵秩]]"
  - "[[秩零度定理]]"
related:
  - "[[Wishart秩与可逆性]]"
  - "[[PCA低秩重构]]"
part_of:
  - "[[多元线性回归.canvas|多元线性回归]]"
---

<!-- bilingual-en:start -->
# The rank of the error SSP
<!-- bilingual-en:end -->

*误差 SSP 的秩边界*

<!-- bilingual-en:start -->
Fitting a model uses some of the directions in which the observations can vary. Residuals occupy only the directions left over. If we then ask those residuals to distinguish too many responses, their square summary matrix cannot be invertible. This card explains that dimension limit, and separates it from the stronger claim that a particular statistical model reaches the limit with probability one.
<!-- bilingual-en:end -->

拟合模型会占用观测能够变动的一部分方向，残差只能落在剩余方向里。若再要求这些残差区分过多响应，它们的方形汇总矩阵就不可能可逆。本卡解释这个维度限制，并把它与更强的“某个统计模型以概率一达到上界”区分开。

<!-- bilingual-en:start -->
## Count the directions left after fitting
<!-- bilingual-en:end -->

*数清拟合后剩余的方向*

<!-- bilingual-en:start -->
Take $n\ge1$ real observations with $q\ge1$ responses, arranged in $Y\in\mathbb R^{n\times q}$. Here $q$ counts responses. Fit each response by [[共享设计的逐响应OLS|unweighted OLS using the same design]] $X\in\mathbb R^{n\times k}$. Let $r=\operatorname{rank}(X)$; [[矩阵秩|rank]] counts independent columns, so redundant predictor columns do not increase $r$. Define the residual degrees of freedom as $\nu=n-r$.
<!-- bilingual-en:end -->

将 $n\ge1$ 条实数观测、$q\ge1$ 个响应排成 $Y\in\mathbb R^{n\times q}$；这里 $q$ 是响应数。每个响应都使用同一个 $X\in\mathbb R^{n\times k}$ 做[[共享设计的逐响应OLS|不加权 OLS]]。令 $r=\operatorname{rank}(X)$；[[矩阵秩|秩]]计算独立列的个数，因此冗余预测列不会增加 $r$。残差自由度定义为 $\nu=n-r$。

<!-- bilingual-en:start -->
Each response column is a vector in the $n$-dimensional observation space. Its fitted vector lies in the column space of $X$, which has dimension $r$. OLS leaves a residual perpendicular to that entire space, in its [[正交补|orthogonal complement]] of dimension $n-r$. If $P_X$ denotes the [[正交投影|orthogonal projector]] onto the column space and $I_n$ denotes the identity matrix, the [[多响应残差与误差SSP|residual and error-SSP matrices]] are
<!-- bilingual-en:end -->

每个响应列都是 $n$ 维观测空间中的一个向量。拟合向量落在 $X$ 的列空间里，这个空间的维数是 $r$。OLS 留下的残差垂直于整个列空间，位于维数为 $n-r$ 的[[正交补]]。若 $P_X$ 表示到列空间的[[正交投影|正交投影矩阵]]，$I_n$ 表示恒等矩阵，则[[多响应残差与误差SSP|残差矩阵与误差 SSP]]为

$$
R=(I_n-P_X)Y\in\mathbb R^{n\times q},
\qquad E_{SSP}=R^TR\in\mathbb R^{q\times q}.
$$

<!-- bilingual-en:start -->
This description remains valid when $X$ has dependent columns: OLS coefficients may be nonunique, but the projected fitted values and residuals are unique. No normality or independent-observation assumption is needed for this fixed-data geometry.
<!-- bilingual-en:end -->

即使 $X$ 有线性相关的列，这种描述仍然成立：OLS 系数可能不唯一，投影拟合值和残差却唯一。这些是固定数据的几何事实，不要求正态性或观测独立。

> [!important] The deterministic bound · 对每份数据成立的上界
> <!-- bilingual-en:start -->
> For every such $X$ and $Y$,
> $$\operatorname{rank}(E_{SSP})=\operatorname{rank}(R)
> \le\min(q,\nu)=\min(q,n-r).$$
> Therefore $q>\nu$ forces the $q\times q$ error SSP to be singular. If $\nu=0$, the fitted space is all of $\mathbb R^n$: $P_X=I_n$, $R=0$, and $E_{SSP}=0$. There is then no residual information for estimating a covariance by division by $\nu$.
> <!-- bilingual-en:end -->
>
> 对每份这样的 $X,Y$，误差 SSP 与残差矩阵同秩，且秩至多为 $\min(q,n-r)$。所以 $q>\nu$ 必使 $q\times q$ 误差 SSP 奇异。$\nu=0$ 时，拟合空间已是整个 $\mathbb R^n$，于是 $P_X=I_n$、$R=0$、$E_{SSP}=0$，没有可用来除以 $\nu$ 估计协方差的残差信息。

<!-- bilingual-en:start -->
Why do $R$ and $R^TR$ have the same rank, although their shapes differ? For any $a\in\mathbb R^q$,
<!-- bilingual-en:end -->

$R$ 和 $R^TR$ 的形状不同，为什么秩相同？对任意 $a\in\mathbb R^q$，

$$a^TR^TRa=\|Ra\|_2^2.$$

<!-- bilingual-en:start -->
Thus $R^TRa=0$ implies $Ra=0$, because a squared [[欧氏范数|Euclidean length]] is zero only for the zero vector. The reverse implication is immediate. The two maps therefore have the same [[零空间|nullspace]] of input coefficients. They both have $q$ input coordinates, so [[秩零度定理|rank–nullity]] gives equal ranks; this is the [[Gram矩阵正定判据|Gram rank identity]]. Finally, the $q$ columns of $R$ all lie in the same $\nu$-dimensional residual space, yielding the bound.
<!-- bilingual-en:end -->

因此，$R^TRa=0$ 推出 $Ra=0$，因为[[欧氏范数|欧氏长度]]平方为零只能对应零向量；反方向直接成立。这两个映射具有相同的输入系数[[零空间]]，输入又都有 $q$ 个坐标，所以[[秩零度定理]]给出同秩，这就是 [[Gram矩阵正定判据|Gram 秩恒等式]]。最后，$R$ 的 $q$ 列全落在同一个 $\nu$ 维残差空间中，从而得到上界。

<!-- bilingual-en:start -->
## Enough dimensions do not force actual full rank
<!-- bilingual-en:end -->

*维度足够，不代表实际数据一定满秩*

<!-- bilingual-en:start -->
For $n=30$, $r=8$ and $q=25$, only $\nu=22$ residual directions remain. Hence the $25\times25$ error SSP has rank at most $22$, whatever the observed values. In contrast, $q\le\nu$ only makes full rank possible; the actual response columns can still repeat the same information.
<!-- bilingual-en:end -->

$n=30,r=8,q=25$ 时，只剩 $\nu=22$ 个残差方向；无论观测值是什么，$25\times25$ 的误差 SSP 都至多秩 $22$。反过来，$q\le\nu$ 只是让满秩成为可能，实际响应列仍可能重复同一信息。

> [!example] Enough rows, but duplicate response information · 行数足够，响应信息仍重复
> <!-- bilingual-en:start -->
> For an intercept-only fit to $Y=\begin{bmatrix}1&2\\2&4\\3&6\end{bmatrix}$, we have $n=3$, $r=1$, $q=2$ and $\nu=2$. The second response is twice the first. Subtracting column means gives
> $$R=\begin{bmatrix}-1&-2\\0&0\\1&2\end{bmatrix},\qquad
> E_{SSP}=\begin{bmatrix}2&4\\4&8\end{bmatrix}.$$
> Although $q=\nu$, the rank is one. The nonzero coefficient vector $a=(2,-1)^T$ gives $Ra=0$: that response combination has no residual variation.
> <!-- bilingual-en:end -->
>
> 对上述三行数据只拟合截距，有 $n=3,r=1,q=2,\nu=2$。第二个响应是第一个的两倍；减去列均值得到所列 $R$ 和 $E_{SSP}$。虽然 $q=\nu$，秩仍只有一。非零系数 $a=(2,-1)^T$ 满足 $Ra=0$，这个响应组合没有残差变动。

<!-- bilingual-en:start -->
## What Gaussian assumptions add
<!-- bilingual-en:end -->

*Gaussian 假设额外保证什么*

<!-- bilingual-en:start -->
Now add a sampling model, with fixed $X$ and correct mean $XB$. Suppose $Y=XB+U$, where the rows of $U$ are iid [[联合Gaussian|Gaussian vectors]] $N_q(0,\Sigma)$ and $\Sigma$ is [[正定矩阵|positive definite]]. With integer $\nu\ge1$, the [[误差SSP抽样律|residual sampling law]] and [[Wishart秩与可逆性|Wishart rank theorem]] give
<!-- bilingual-en:end -->

现在加入抽样模型：固定 $X$，真实均值为 $XB$。假设 $Y=XB+U$，其中 $U$ 的行是 iid [[联合Gaussian|Gaussian 向量]] $N_q(0,\Sigma)$，且 $\Sigma$ [[正定矩阵|正定]]。整数 $\nu\ge1$ 时，[[误差SSP抽样律|残差抽样分布]]与 [[Wishart秩与可逆性|Wishart 秩定理]]给出

$$
E_{SSP}\sim W_q(\Sigma,\nu),\qquad
\operatorname{rank}(E_{SSP})=\min(q,\nu)\quad\text{almost surely}.
$$

<!-- bilingual-en:start -->
Here $W_q(\Sigma,\nu)$ is a [[Wishart分布|Wishart distribution]]: a sum of $\nu$ outer products of independent $N_q(0,\Sigma)$ vectors. To see where these independent vectors come from, choose an $n\times\nu$ matrix $Q$ whose columns are a [[标准正交组|unit-length, mutually perpendicular basis]] of the residual space. Then $Q^TQ=I_\nu$, $I_n-P_X=QQ^T$, and
<!-- bilingual-en:end -->

这里的 $W_q(\Sigma,\nu)$ 是 [[Wishart分布|Wishart 分布]]，即 $\nu$ 个独立 $N_q(0,\Sigma)$ 向量的外积之和。为了看清这些独立向量从哪里来，选取 $n\times\nu$ 矩阵 $Q$，其列是残差空间的一组[[标准正交组|长度为一、互相垂直的基]]。于是 $Q^TQ=I_\nu$、$I_n-P_X=QQ^T$，且

$$Z=Q^TU\in\mathbb R^{\nu\times q},\qquad E_{SSP}=Z^TZ.$$

<!-- bilingual-en:start -->
The rows of $Z$ are Gaussian residual **coordinates**, not the original observation rows of $R$. Orthogonality makes their cross-covariances zero and preserves covariance $\Sigma$ within each row; [[Gaussian仿射闭包|joint Gaussianity under linear transformations]] then allows the [[联合高斯独立判据|Gaussian block-independence criterion]] to make these rows independent. Positive-definite $\Sigma$ supplies a full-dimensional Gaussian density, so new independent rows add directions until the smaller of $q$ and $\nu$ is reached. The Wishart rank card develops that last step. The fitted residual rows $R=QZ$ themselves generally remain dependent.
<!-- bilingual-en:end -->

$Z$ 的行是 Gaussian 残差**坐标**，并不是 $R$ 原来的观测行。正交性使不同行的交叉协方差为零，并使每行仍有协方差 $\Sigma$；[[Gaussian仿射闭包|线性变换保留的联合 Gaussian 性]]进一步允许使用[[联合高斯独立判据|Gaussian 块独立判据]]，得到这些行独立。正定 $\Sigma$ 提供全维 Gaussian 密度，因此独立的新行会继续增加方向，直到达到 $q$ 与 $\nu$ 中较小者；最后这一步由 Wishart 秩卡展开。拟合残差 $R=QZ$ 的原始各行通常仍有依赖。

<!-- bilingual-en:start -->
Consequently, $E_{SSP}$ is positive definite and invertible almost surely when $\nu\ge q$ under this model. “Almost surely” allows exceptional data sets of probability zero; it does not turn the deterministic condition into a guarantee for every imaginable data table. The zero-degree case remains $E_{SSP}=0$, outside the positive-degree Wishart statement above. If $X$ is random, the same argument can be applied conditional on $X$ when the stated error assumptions hold conditional on it.
<!-- bilingual-en:end -->

所以在这个模型下，$\nu\ge q$ 时，$E_{SSP}$ 几乎必然正定、可逆。“几乎必然”容许概率为零的例外数据，并不意味着每一张想象得出的数据表都保证满秩。零自由度时仍是 $E_{SSP}=0$，不属于上面的正自由度 Wishart 陈述。若 $X$ 随机，只要误差条件在给定 $X$ 后成立，就可以条件于 $X$ 使用同样论证。

<!-- bilingual-en:start -->
## An inverse is only the first threshold
<!-- bilingual-en:end -->

*可逆只是第一道门槛*

<!-- bilingual-en:start -->
Even an invertible matrix can be sensitive to small changes: $\operatorname{diag}(1,\varepsilon)$ has inverse $\operatorname{diag}(1,1/\varepsilon)$ for $\varepsilon>0$. A small positive $\varepsilon$ can therefore amplify error greatly. A covariance estimate near the rank threshold can likewise be unstable; [[行列式与条件数边界|conditioning]] and the sampling assumptions still matter after a numerical inverse succeeds.
<!-- bilingual-en:end -->

可逆矩阵也可能对微小变化敏感：$\varepsilon>0$ 时，$\operatorname{diag}(1,\varepsilon)$ 的逆为 $\operatorname{diag}(1,1/\varepsilon)$，很小的正 $\varepsilon$ 会大幅放大误差。接近秩门槛的协方差估计也可能不稳定；成功算出逆以后，[[行列式与条件数边界|条件数]]和抽样假设仍然重要。

<!-- bilingual-en:start -->
When the full-response error SSP is singular, a classical procedure requiring its ordinary inverse is unavailable. A preselected lower-dimensional response contrast, a justified dimension reduction, covariance regularization or a method designed for high-dimensional data may address a different, well-defined problem. Each needs its own target and calibration; inserting a pseudoinverse does not by itself preserve the original MANOVA test's reference distribution.
<!-- bilingual-en:end -->

全响应误差 SSP 奇异时，要求普通逆的经典程序无法直接使用。事先选定较低维响应对比、有依据的降维、协方差正则化或专门的高维方法，都可能解决重新明确的问题。但每种做法需要自己的对象与校准；仅把普通逆换成广义逆，不会自动保留原 MANOVA 检验的参考分布。

## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.1sum.pdf#page=2|MIT 18.06SC, Session 2.1, pp. 1–2]]: orthogonal-complement dimensions and $N(A^TA)=N(A)$, hence equal Gram ranks.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.2sum.pdf#page=2|MIT 18.06SC, Session 2.2, p. 2]]: OLS fitted values are projections and residuals lie in the left nullspace of the design. The argument above uses the column space even when the design has redundant columns.
- [MIT 6.436J, Lecture 14, pp. 3–6](https://ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/ceb423b6a703abd60429367b733f45bf_MIT6_436JF18_lec14.pdf#page=3): Gaussian affine transformations, covariance and independence, and the density under positive-definite covariance. Together with the displayed orthogonal-coordinate construction, these verify the Gaussian strengthening of the deterministic bound.
- [Sawyer, *Multivariate Linear Models*, (3.2b), p. 3](https://www.math.wustl.edu/~sawyer/handouts/manovas.pdf#page=3): the error SSCP is the response-by-response product $R^TR$. The examples and zero-degree boundary follow directly from the displayed matrices and projection identities.
- [R official `summary.manova` documentation, Details](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/summary.manova.html): residual-correlation rank checks and the risk of inaccurate results with highly correlated responses.
<!-- bilingual-en:end -->

- MIT 第 2.1 节摘要第 1–2 页：核对正交补维数、Gram 零空间与秩恒等式。
- MIT 第 2.2 节摘要第 2 页：核对 OLS 投影与残差落在设计矩阵左零空间；正文以列空间表述，覆盖冗余设计列。
- MIT 概率论第 14 讲第 3–6 页：核对 Gaussian 仿射变换、协方差与独立性、正定协方差下的密度；结合正文正交坐标构造，核验概率一达到上界的结论。
- Sawyer 讲义第 3 页式 (3.2b)：核对响应×响应的 $R^TR$ 形式；正文直接推导例子与零自由度边界。
- R 官方 `summary.manova` 的 Details：核对残差相关矩阵的秩检查，以及高度相关响应影响准确性的提醒。

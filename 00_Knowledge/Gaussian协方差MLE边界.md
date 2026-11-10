---
aliases:
  - "Existence boundary for the Gaussian covariance MLE"
  - "Singular Gaussian covariance likelihood"
student_os: knowledge-atom
atom_id: STAT-MVD-021
atom_set: multivariate-data-foundations
atom_type: boundary
status: source-checked
requires:
  - "[[样本协方差矩阵]]"
  - "[[Gaussian密度与奇异支撑]]"
related:
  - "[[样本协方差无偏分母]]"
  - "[[样本协方差秩边界]]"
part_of:
  - "[[多元数据与协方差.canvas]]"
---

<!-- bilingual-en:start -->
# When does the Gaussian covariance MLE exist?
<!-- bilingual-en:end -->

*Gaussian 协方差 MLE 的存在条件*

<!-- bilingual-en:start -->
Fitting a Gaussian distribution means choosing its center and its spread to describe the observations. [[极大似然估计|Maximum likelihood estimation (MLE)]] chooses the allowed parameters that give the observed data the largest joint density. A formula for a candidate estimate is not enough: the candidate must belong to the allowed model and actually attain the maximum. For an unrestricted Gaussian covariance, the crucial question is whether the observed points spread in every direction.
<!-- bilingual-en:end -->

拟合 Gaussian 分布，就是选择它的中心与散布来描述观测。[[极大似然估计|极大似然估计（MLE）]]从允许的参数中，选出使已观测数据的联合密度最大的参数。写出一个候选估计公式还不够：候选值必须属于允许的模型，并且真正达到最大值。对不加结构约束的 Gaussian 协方差，关键是观测点是否在每个方向上都有变化。

<!-- bilingual-en:start -->
For example, the three points $(-1,0),(0,1),(1,0)$ spread across a plane. Move the middle point to $(0,0)$ and all three lie on a line. Both datasets have three observations and two coordinates, but only the first has a covariance MLE within the model that requires positive spread in every direction. We will see why making the fitted Gaussian ever thinner around the second dataset keeps increasing its likelihood.
<!-- bilingual-en:end -->

例如，三个点 $(-1,0),(0,1),(1,0)$ 在平面内展开；把中间那个点移到 $(0,0)$，三个点就落在一条直线上。两组数据都有三次观测、两个坐标，但在要求每个方向都有正散布的模型里，只有第一组存在协方差 MLE。下面会看到：对第二组，不断把拟合 Gaussian 沿直线的垂直方向压薄，似然就会不断增大。

<!-- bilingual-en:start -->
## The model and the observed scatter
<!-- bilingual-en:end -->

*先明确模型与观测散布*

<!-- bilingual-en:start -->
Assume $n\ge1$ [[独立同分布|independent, identically distributed]] random observation vectors $X_1,\ldots,X_n$, each with $p\ge1$ real coordinates, follow the same [[联合Gaussian|multivariate Gaussian distribution]] $N_p(\mu,\Sigma)$. Both the common mean $\mu\in\mathbb R^p$ and the real symmetric [[协方差矩阵|covariance matrix]] $\Sigma$ are unknown. The parameter space allows every $\mu$ and every [[正定矩阵|positive-definite]] $\Sigma$: $a^T\Sigma a>0$ for each nonzero $a$. Thus the model requires positive variance in every nontrivial linear direction and has the usual [[Gaussian密度与奇异支撑|full-dimensional Gaussian density]]. No diagonal, sparsity or factor-model constraints are imposed.
<!-- bilingual-en:end -->

假设 $n\ge1$ 个[[独立同分布]]的随机观测向量 $X_1,\ldots,X_n$，各有 $p\ge1$ 个实坐标，服从同一个[[联合Gaussian|多元 Gaussian 分布]] $N_p(\mu,\Sigma)$。共同均值 $\mu\in\mathbb R^p$ 与实对称[[协方差矩阵]] $\Sigma$ 都未知。参数空间允许任意 $\mu$ 以及任意[[正定矩阵|正定]]的 $\Sigma$：每个非零 $a$ 都满足 $a^T\Sigma a>0$。这要求每个非平凡线性方向都有正方差，因而具有通常的[[Gaussian密度与奇异支撑|全维 Gaussian 密度]]；这里没有附加对角、稀疏或因子模型等结构约束。

<!-- bilingual-en:start -->
After observing complete, finite numerical vectors $x_1,\ldots,x_n$, hold them fixed. Their [[样本均值向量|sample mean]] is $\bar x=n^{-1}\sum_i x_i$. The **centered scatter matrix** $C$ adds the outer products of the residuals $x_i-\bar x$; the transpose $T$ turns each column residual into a row:
<!-- bilingual-en:end -->

观察到完整、有限的数值向量 $x_1,\ldots,x_n$ 后，把这些数据固定。它们的[[样本均值向量]]为 $\bar x=n^{-1}\sum_i x_i$。**中心化散布矩阵** $C$ 将各残差 $x_i-\bar x$ 的外积相加；转置符号 $T$ 将列残差变成行：

$$
C=\sum_{i=1}^n(x_i-\bar x)(x_i-\bar x)^T.
$$

<!-- bilingual-en:start -->
For $n\ge2$, the usual observed [[样本协方差矩阵|sample covariance matrix]] is $s=C/(n-1)$. Here $C$ is the undivided sum. In any direction $a$, $a^TCa=\sum_i[a^T(x_i-\bar x)]^2$, so $C$ is always [[半正定矩阵|positive semidefinite]]. It is positive definite exactly when no nonzero linear combination of the coordinates is constant across all observed points.
<!-- bilingual-en:end -->

当 $n\ge2$ 时，通常的观测[[样本协方差矩阵]]为 $s=C/(n-1)$；这里 $C$ 是尚未除分母的总和。沿任意方向 $a$，$a^TCa=\sum_i[a^T(x_i-\bar x)]^2$，所以 $C$ 总是[[半正定矩阵|半正定]]。它正定，当且仅当不存在一个非零坐标线性组合，在所有观测点上都取相同值。

> [!important] Existence is a property of this sample · 先检查这组样本
> <!-- bilingual-en:start -->
> In the stated model, a finite global maximum of the likelihood exists **if and only if** $C\succ0$. Then the maximizing pair is unique:
> $$\widehat\mu=\bar x,\qquad \widehat\Sigma_{\mathrm{MLE}}=\frac Cn.$$
> If $C$ is singular, the likelihood is unbounded above on the positive-definite parameter space. The matrix $C/n$ can still be calculated, but it is outside that parameter space and is not its MLE.
> <!-- bilingual-en:end -->
>
> 在上述模型中，似然存在有限全局最大值，**当且仅当** $C\succ0$；这时唯一的最大化参数对如上。若 $C$ 奇异，似然在正定参数空间中向上无界。虽然仍能计算 $C/n$，但它不属于这个参数空间，也不是其中的 MLE。

<!-- bilingual-en:start -->
## Why the mean and the divisor appear
<!-- bilingual-en:end -->

*均值与分母为何这样出现*

<!-- bilingual-en:start -->
The [[似然函数|likelihood]] is the product of the $n$ Gaussian densities, viewed as a function of $\mu,\Sigma$ while the observations remain fixed. It compares density values, not the probability of observing an exact continuous-data point. Taking the natural logarithm preserves the maximizer. For each fixed positive-definite $\Sigma$, the identity
<!-- bilingual-en:end -->

[[似然函数]]是 $n$ 个 Gaussian 密度的乘积：数据固定，变量是 $\mu,\Sigma$。它比较的是密度值，不是精确连续观测点的概率。取自然对数不改变最大化参数。对每个固定正定 $\Sigma$，有恒等式

$$
\sum_i(x_i-\mu)(x_i-\mu)^T
=C+n(\bar x-\mu)(\bar x-\mu)^T.
$$

<!-- bilingual-en:start -->
The cross terms vanish because the centered residuals sum to zero. Relative to choosing $\mu=\bar x$, any other mean subtracts $\tfrac n2(\bar x-\mu)^T\Sigma^{-1}(\bar x-\mu)>0$ from the log-likelihood. Thus $\bar x$ is its unique best mean for each fixed covariance. This statement holds even when no jointly maximizing pair exists.
<!-- bilingual-en:end -->

因为中心化残差之和为零，展开后的交叉项消失。相对于取 $\mu=\bar x$，其他均值会使对数似然减少严格为正的 $\tfrac n2(\bar x-\mu)^T\Sigma^{-1}(\bar x-\mu)$。因此，每个固定协方差下的唯一最优均值都是 $\bar x$；即使不存在联合最大化参数对，这句话仍然成立。

<!-- bilingual-en:start -->
Substitute that mean to obtain the **profile log-likelihood**: the value after optimizing the mean for each covariance. Write $K$ for a constant that does not depend on $\Sigma$, $\det\Sigma$ for its [[行列式|determinant]], and $\operatorname{tr}$ for the sum of diagonal entries of a matrix. Then
<!-- bilingual-en:end -->

代入这个均值，得到**轮廓对数似然**：对每个协方差先优化均值后得到的值。令 $K$ 为与 $\Sigma$ 无关的常数，$\det\Sigma$ 为其[[行列式]]，$\operatorname{tr}$ 表示矩阵对角元素之和，则

$$
\ell_{\mathrm{prof}}(\Sigma)
=K-\frac n2\log\det\Sigma
-\frac12\operatorname{tr}(\Sigma^{-1}C).
$$

<!-- bilingual-en:start -->
The determinant term rewards concentrating the fitted density. The trace term equals the sum of covariance-adjusted squared residual lengths, so it penalizes a covariance that is too small along directions where the data vary. If $C\succ0$, balancing these effects gives $C/n$, and the argument below verifies that it is the unique global maximum.
<!-- bilingual-en:end -->

行列式项鼓励拟合密度集中；迹项等于经过协方差调整后的残差长度平方之和，因此会惩罚在数据实际变化的方向上取过小的协方差。若 $C\succ0$，两者的平衡点为 $C/n$；下面的论证还确认它是唯一全局最大值。

> [!note]- Why this is a global maximum · 为什么不只是一个驻点
> <!-- bilingual-en:start -->
> Put $A=C/n\succ0$. For any candidate $\Sigma\succ0$, form $B=A^{-1/2}\Sigma A^{-1/2}$ using the [[正定逆平方根|positive-definite inverse square root]]. This measures the candidate covariance in coordinates where $A$ becomes the identity. By the [[实对称矩阵谱定理|spectral theorem]], $B$ has positive eigenvalues $\lambda_1,\ldots,\lambda_p$. Determinant and trace identities give
> $$\ell_{\mathrm{prof}}(\Sigma)-\ell_{\mathrm{prof}}(A)
> =-\frac n2\sum_{j=1}^p\left(\log\lambda_j+\frac1{\lambda_j}-1\right).$$
> The scalar function $g(t)=\log t+1/t-1$ has [[导数|derivative]] $(t-1)/t^2$ for $t>0$. It decreases until $1$, increases afterwards, and equals zero only at its minimum $t=1$. Every summand is therefore nonnegative. Equality of the two log-likelihoods requires all $\lambda_j=1$, hence $B=I_p$ and $\Sigma=A$. This proves global optimality and uniqueness.
> <!-- bilingual-en:end -->
>
> 令 $A=C/n\succ0$。对任意候选 $\Sigma\succ0$，用[[正定逆平方根]]构造 $B=A^{-1/2}\Sigma A^{-1/2}$，相当于在把 $A$ 变成单位矩阵的坐标中衡量候选协方差。由[[实对称矩阵谱定理]]，$B$ 的特征值 $\lambda_1,\ldots,\lambda_p$ 均为正，行列式与迹的恒等式给出上述差值。标量函数 $g(t)=\log t+1/t-1$ 的[[导数]]为 $(t-1)/t^2$：它在 $1$ 之前下降，之后上升，仅在最小点 $t=1$ 取零。因此每一项都非负；两个对数似然相等要求所有特征值为 $1$，即 $B=I_p$、$\Sigma=A$，从而证明全局最优和唯一性。

<!-- bilingual-en:start -->
## Why singular scatter makes the likelihood unbounded
<!-- bilingual-en:end -->

## 奇异时为什么没有有限最大值

> [!example] Keep the points fixed and squeeze one direction · 固定数据，压薄一个方向
> <!-- bilingual-en:start -->
> For $(-1,0),(0,1),(1,0)$, the mean is $(0,1/3)^T$ and
> $$C=\operatorname{diag}(2,2/3),\qquad
> \widehat\Sigma_{\mathrm{MLE}}=\operatorname{diag}(2/3,2/9).$$
> Here $\operatorname{diag}(a,b)$ means diagonal entries $a,b$ and zero off-diagonal entries. Both fitted variances are positive.
>
> For $(-1,0),(0,0),(1,0)$, however, $\bar x=0$ and $C=\operatorname{diag}(2,0)$. Take the valid positive-definite candidate $\Sigma_\varepsilon=\operatorname{diag}(2/3,\varepsilon)$ for any $\varepsilon>0$. Then
> $$\det\Sigma_\varepsilon=(2/3)\varepsilon,\qquad
> \operatorname{tr}(\Sigma_\varepsilon^{-1}C)=3,\qquad
> \ell_{\mathrm{prof}}(\Sigma_\varepsilon)=K_0-\frac32\log\varepsilon\longrightarrow+\infty.$$
> The new constant $K_0$ is independent of $\varepsilon$. Narrowing the second coordinate creates no residual penalty because every observed second coordinate equals the fitted mean. Every positive $\varepsilon$ can be improved upon by taking a smaller one; setting it to zero would leave the model.
> <!-- bilingual-en:end -->
>
> 第一组点 $(-1,0),(0,1),(1,0)$ 的均值是 $(0,1/3)^T$，散布与 MLE 如上。$\operatorname{diag}(a,b)$ 表示对角为 $a,b$、非对角为零；两个拟合方差都严格为正。
>
> 第二组点 $(-1,0),(0,0),(1,0)$ 则有 $\bar x=0$、$C=\operatorname{diag}(2,0)$。对任何 $\varepsilon>0$，$\Sigma_\varepsilon=\operatorname{diag}(2/3,\varepsilon)$ 仍是允许的正定候选，迹项恒为 $3$，对数似然却按上述公式趋于正无穷，$K_0$ 不依赖于 $\varepsilon$。第二坐标的观测都等于拟合均值，因此压薄这个方向不产生残差惩罚。任何正 $\varepsilon$ 都能通过进一步缩小来改善；取零又会离开模型。

<!-- bilingual-en:start -->
The same mechanism works for every singular $C$, even when no original coordinate is constant. Let $r=\operatorname{rank}(C)<p$. An [[实对称矩阵谱定理|orthogonal eigenbasis]] writes $C=Q\operatorname{diag}(c_1,\ldots,c_r,0,\ldots,0)Q^T$, with $c_j>0$ and $Q^TQ=I_p$. Set the candidate variances to $c_j/n$ on these $r$ directions and to $\varepsilon$ on the remaining $p-r$ directions. Its trace term is $nr$ and its determinant is $\bigl(\prod_{j=1}^r c_j/n\bigr)\varepsilon^{p-r}$. Hence its log-likelihood tends to $+\infty$ as $\varepsilon\downarrow0$. For $r=0$, use an empty product of $1$ and a zero trace: all observations coincide, and the same conclusion holds. In particular, this covers $n=1$.
<!-- bilingual-en:end -->

任意奇异 $C$ 都有同样机制，不要求某个原坐标本身恒定。令 $r=\operatorname{rank}(C)<p$；选择[[实对称矩阵谱定理|正交特征基]]，写成 $C=Q\operatorname{diag}(c_1,\ldots,c_r,0,\ldots,0)Q^T$，其中 $c_j>0$、$Q^TQ=I_p$。在前 $r$ 个方向取候选方差 $c_j/n$，剩余 $p-r$ 个方向取 $\varepsilon$，就得到恒定的迹项 $nr$，以及行列式 $\bigl(\prod_{j=1}^r c_j/n\bigr)\varepsilon^{p-r}$。所以 $\varepsilon\downarrow0$ 时对数似然趋于正无穷。$r=0$ 时把空乘积记为 $1$、迹项取零：所有观测重合，结论仍成立，特别包括 $n=1$。

<!-- bilingual-en:start -->
## Sample size, probability one, and different estimation questions
<!-- bilingual-en:end -->

*样本量、概率一与不同估计问题*

<!-- bilingual-en:start -->
The [[样本协方差秩边界|centered-scatter rank bound]] is $\operatorname{rank}(C)\le\min(p,n-1)$. Therefore $n>p$ is necessary for existence in this unknown-mean model. It is not sufficient for a fixed sample: the collinear example has $n=3>2=p$. Full rank means that the observed points fill all $p$ affine directions, rather than lying in a lower-dimensional affine subspace. Exact duplicate variables, constant variables and other exact affine relationships can prevent this even with many observations.
<!-- bilingual-en:end -->

[[样本协方差秩边界|中心化散布的秩上界]]为 $\operatorname{rank}(C)\le\min(p,n-1)$。所以，$n>p$ 是这个未知均值模型中存在 MLE 的必要条件，却不是固定样本的充分条件：刚才共线的例子就有 $n=3>2=p$。满秩要求观测点张成全部 $p$ 个仿射方向，而不是落在更低维的仿射子空间中。重复变量、常数变量以及其他精确仿射关系，都可能使很多观测仍然不足。

<!-- bilingual-en:start -->
Under the stated iid Gaussian sampling model with a true positive-definite covariance, the [[Wishart秩与可逆性|rank is maximal almost surely]]. Consequently, when $n>p$ a unique MLE exists **with probability one**. Geometrically, a new independent observation has probability zero of landing in the proper affine subspace spanned by the previous observations; after $p+1$ observations the affine span fills the space almost surely. This probabilistic statement permits exceptional numerical datasets and does not replace checking an observed or rounded table.
<!-- bilingual-en:end -->

在上述 iid Gaussian 抽样模型、真实协方差正定的条件下，[[Wishart秩与可逆性|秩几乎必然达到最大可能值]]。因此 $n>p$ 时，唯一 MLE **以概率一存在**。几何上，新独立观测落入此前观测张成的真仿射子空间的概率为零；到第 $p+1$ 个观测时，仿射张成空间几乎必然填满整个空间。这个概率陈述允许例外数值数据存在，不能代替对已观测或舍入数据表的检查。

> [!warning] Keep the estimation problem explicit · 不同问题不能混用结论
> <!-- bilingual-en:start -->
> The divisor $n$ comes from maximizing this likelihood. The divisor $n-1$ in $s=C/(n-1)$ instead provides [[样本协方差无偏分母|unbiasedness under iid finite-second-moment sampling]]; Gaussian sampling meets those moment conditions. Changing the divisor cannot repair deficient rank. Likewise, adding a positive diagonal term may produce an invertible estimate, but that modified estimate is not the maximizer of the unrestricted likelihood studied here. A regularized or constrained method needs its own stated objective or parameter restrictions.
> <!-- bilingual-en:end -->
>
> 分母 $n$ 来自最大化这里的似然；$s=C/(n-1)$ 中的 $n-1$ 则用于[[样本协方差无偏分母|iid、有限二阶矩抽样下的无偏性]]，Gaussian 抽样满足这些矩条件。更换分母不能补回缺失的秩。类似地，加一个正对角项可能得到可逆估计，但修改后的矩阵并不是这里无约束似然的最大化参数；正则化或受约束方法须另行说明目标函数或参数限制。

<!-- bilingual-en:start -->
Finally, if the population mean is known in advance, the residuals are taken around that known mean, not around an estimated sample mean. That is a different likelihood problem; the unknown-mean $n-1$ rank restriction and $n>p$ threshold above must not simply be transferred to it. Allowing singular Gaussian distributions is also a change of model and reference measure, as explained in [[Gaussian密度与奇异支撑|Gaussian densities and singular support]]. A singular matrix cannot just be inserted into the determinant-and-inverse density used here.
<!-- bilingual-en:end -->

最后，如果总体均值预先已知，残差应相对那个已知均值计算，而不是相对估计的样本均值。这是另一个似然问题，不能直接移用上面未知均值情形的 $n-1$ 秩限制与 $n>p$ 门槛。允许奇异 Gaussian 分布，也意味着改变模型与参考测度，见[[Gaussian密度与奇异支撑]]；不能直接把奇异矩阵代入这里含行列式与普通逆的密度。

## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/04_多元统计分析/04_多元正态分布The Multivariate Normal Distribution.md#1.5.2. 最大似然估计|Multivariate Statistics §1.5.2]]: the course's full-rank existence condition and distinction between likelihood and unbiased divisors.
- [Hoff, Multivariate Statistical Analysis, March 31, 2026, pp. 68–71 and 81](https://www2.stat.duke.edu/~pdh10/Teaching/832/Materials/notes.pdf#page=68): centered residuals, the positive-definite model, profile likelihood and the $n$ divisor. The singular case is proved explicitly above rather than inferred from the displayed candidate formula.
- [Stanford STATS305C, “Maximizing likelihood”](https://web.stanford.edu/class/stats305c/lectures/Onesample.html#maximizing-likelihood): the profiled log-determinant/trace expression. Its displayed candidate is used here with the required full-rank condition.
- [Zwiernik, Uhler and Richards, Maximum Likelihood Estimation for Linear Gaussian Covariance Models, §3.2, arXiv PDF pp. 11–12](https://arxiv.org/pdf/1408.5604#page=11): the eigenvalue argument for unique global optimality when the empirical covariance is positive definite. The centered-data sample-size threshold is established separately above from rank and general position.
<!-- bilingual-en:end -->

- 多元统计课程第 1.5.2 节：核对满秩存在条件，以及似然分母与无偏分母的区别。
- Hoff 2026 年 3 月 31 日版第 68–71、81 页：核对中心化残差、正定模型、轮廓似然与 $n$ 分母；奇异情形由正文显式证明，不从候选公式直接推断。
- Stanford 第 “Maximizing likelihood” 节：核对轮廓似然中的 log-determinant 与 trace 表达式；其候选公式在本卡明确附有满秩条件。
- Zwiernik、Uhler 与 Richards 第 3.2 节、arXiv PDF 第 11–12 页：核对经验协方差正定时的特征值全局唯一最优论证。中心化数据的样本量门槛由正文的秩与一般位置论证另行确定。

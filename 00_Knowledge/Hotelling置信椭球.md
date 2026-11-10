---
aliases:
  - "Hotelling confidence ellipsoid"
  - "均值向量的 Hotelling 置信椭球"
  - "所有线性组合的 Hotelling 同时区间"
student_os: knowledge-atom
atom_id: STAT-HOT-004
atom_set: hotelling-mean-inference
atom_type: equivalence-theorem
status: source-checked
requires:
  - "[[单样本Hotelling T²]]"
  - "[[Hotelling投影极值]]"
part_of:
  - "[[Hotelling T² 与多元均值推断.canvas]]"
contrasts_with:
  - "[[Bonferroni对比区间]]"
related:
  - "[[正定二次型的椭球]]"
---

<!-- bilingual-en:start -->
# Hotelling confidence ellipsoid
<!-- bilingual-en:end -->

*均值向量的 Hotelling 置信椭球*

<!-- bilingual-en:start -->
A hypothesis test checks one proposed mean vector. A **confidence region** collects the mean vectors that the same test would not reject. With several measurements, this region must preserve their joint uncertainty: a vector can be incompatible with the data even when each of its coordinates looks individually plausible. The Hotelling region is an ellipsoid centered at the sample mean.
<!-- bilingual-en:end -->

假设检验检查一个指定的均值向量；**置信区域**则收集同一种检验不会拒绝的全部候选均值向量。多个指标的区域需要保留共同不确定性：每个坐标单看都似乎合理，并不保证它们组合起来仍与数据相容。Hotelling 置信区域是以样本均值为中心的椭球。

<!-- bilingual-en:start -->
Assume $X_1,\ldots,X_n\overset{\mathrm{iid}}{\sim}N_p(\mu,\Sigma)$ with $\Sigma\succ0$ and $n>p\ge1$, as in the [[单样本Hotelling T²|one-sample Hotelling theorem]]. Use the [[样本均值向量|sample mean]] $\bar X$ and [[样本协方差矩阵|sample covariance]] $S=(n-1)^{-1}\sum_i(X_i-\bar X)(X_i-\bar X)^T$. Choose $0<\alpha<1$ and put
$$c_\alpha=\frac{p(n-1)}{n-p}F_{p,n-p}(1-\alpha),$$
where $F_{p,n-p}(1-\alpha)$ is the $(1-\alpha)$ [[总体分位数|quantile]] of the [[F分布|F distribution]] with degrees of freedom $p,n-p$.
<!-- bilingual-en:end -->

沿用[[单样本Hotelling T²|单样本定理]]的条件：iid 多元正态、总体协方差正定、$n>p\ge1$。使用[[样本均值向量]] $\bar X$ 与分母为 $n-1$ 的[[样本协方差矩阵]] $S$。取 $0<\alpha<1$，用上式定义临界值 $c_\alpha$；其中的 F 量是自由度为 $p,n-p$ 的[[F分布]]的 $(1-\alpha)$ [[总体分位数|分位数]]。

> [!important] Keep every mean vector not rejected · 保留全部未拒绝的均值向量
> <!-- bilingual-en:start -->
> For a candidate vector $m\in\mathbb R^p$, define
> $$\mathcal C_{1-\alpha}=\left\{m:n(\bar X-m)^TS^{-1}(\bar X-m)\le c_\alpha\right\}.$$
> Under the stated model, $P_{\mu,\Sigma}\{\mu\in\mathcal C_{1-\alpha}\}=1-\alpha$. The candidate $m$ varies across the region; the true $\mu$ is fixed. A target $\mu_0$ lies outside this region exactly when the corresponding level-$\alpha$ Hotelling test rejects it.
> <!-- bilingual-en:end -->
>
> 让候选向量 $m$ 变化，把不超过同一阈值的全部候选收集成 $\mathcal C_{1-\alpha}$。在所列模型下，它覆盖真实均值 $\mu$ 的概率恰为 $1-\alpha$。区域中的候选 $m$ 可变，真实参数 $\mu$ 是固定的。$\mu_0$ 在区域外，当且仅当对应的水平 $\alpha$ Hotelling 检验拒绝它。

<!-- bilingual-en:start -->
The [[置信区间覆盖率|coverage statement]] refers to repeated sampling: both $\bar X$ and $S$, and hence the region, change across samples. It does not assign posterior probability $1-\alpha$ to the fixed parameter after this particular region has been observed.
<!-- bilingual-en:end -->

[[置信区间覆盖率|覆盖率]]描述重复抽样程序：$\bar X$、$S$ 以及区域都会随样本变化。它不是在已经得到当前区域后，又给固定参数赋予 $1-\alpha$ 的后验概率。

<!-- bilingual-en:start -->
## Project one region into intervals
<!-- bilingual-en:end -->

*把同一个区域投影成区间*

<!-- bilingual-en:start -->
A vector $a$ chooses a linear combination $a^T\mu$: $a=e_j$ (1 in position $j$, zero elsewhere) picks coordinate $j$, while $a=(1,-1)^T$ compares two means. Projecting the whole region onto that quantity gives
$$a^T\mu\in a^T\bar X\ \pm\sqrt{\frac{c_\alpha}{n}\,a^TSa}.$$
Here “$\pm$” denotes the interval between the minus and plus endpoints. The factor $a^TSa$ is the estimated variance of one projected observation, so division by $n$ gives its mean's estimated variance.
<!-- bilingual-en:end -->

向量 $a$ 选择线性组合 $a^T\mu$：$a=e_j$（只有第 $j$ 位为 1，其余为零）选出第 $j$ 个均值，$a=(1,-1)^T$ 比较两个均值之差。把整个区域投影到该量，得到上式区间。$\pm$ 表示分别减、加根号项所给出的两个端点；$a^TSa$ 是一个投影观测的估计方差，除以 $n$ 才是投影均值的估计方差。

<!-- bilingual-en:start -->
These intervals hold **simultaneously for every $a\in\mathbb R^p$** with probability $1-\alpha$. To see why, set $\delta=\mu-\bar X$. The [[Hotelling投影极值|maximum-projection identity]] gives
$$n\delta^TS^{-1}\delta\le c_\alpha
\quad\Longleftrightarrow\quad
|a^T\delta|\le\sqrt{\frac{c_\alpha}{n}a^TSa}\ \text{for every }a.$$
One joint event therefore covers an infinite family of directions. The endpoints are genuine projections of the ellipsoid: for $a\ne0$, they are attained by displacements $\delta=\pm\sqrt{c_\alpha/n}\,Sa/\sqrt{a^TSa}$.
<!-- bilingual-en:end -->

这些区间以 $1-\alpha$ 的概率**对所有 $a\in\mathbb R^p$ 同时成立**。令 $\delta=\mu-\bar X$，[[Hotelling投影极值|最大投影恒等式]]说明“真实均值在椭球里”等价于“所有方向都满足相应区间界”。因此无限多个方向由同一个联合事件覆盖。端点确实来自椭球上的点：$a\ne0$ 时，取所示的 $\delta=\pm\sqrt{c_\alpha/n}\,Sa/\sqrt{a^TSa}$ 就达到上下端点。

<!-- bilingual-en:start -->
Consequently a direction may be chosen after inspecting the data while retaining this simultaneous guarantee. That does not make an ordinary unadjusted $t$ interval valid for a selected direction. If only a small, prespecified finite family matters, [[Bonferroni对比区间|Bonferroni intervals]] use a different guarantee and may be narrower. Coordinate intervals alone are only a subset of all directions: their joint coverage is at least $1-\alpha$, not generally exactly $1-\alpha$.
<!-- bilingual-en:end -->

因此，即使看过数据后再挑方向，这套同时覆盖保证仍保留；这不意味着普通未调整的 $t$ 区间也能用于事后挑选的方向。若只关心少量预先指定的比较，[[Bonferroni对比区间]]采用另一种保证，可能更短。单看坐标区间，只取了全部方向的一部分；它们的联合覆盖率至少为 $1-\alpha$，一般并非恰好等于 $1-\alpha$。

<!-- bilingual-en:start -->
## Read the shape and the coordinate shadows
<!-- bilingual-en:end -->

*怎样读形状与坐标投影*

<!-- bilingual-en:start -->
The [[正定二次型的椭球|positive-definite quadratic form]] makes a bounded ellipsoid. If $S q_j=\lambda_jq_j$ with unit [[特征对|eigenvector]] $q_j$ (a direction that $S$ simply scales by $\lambda_j$), its semiaxis in that direction has length $\sqrt{c_\alpha\lambda_j/n}$. Larger sample variation makes that direction longer; off-diagonal covariance can rotate the axes. The region includes its interior, not only its boundary surface.
<!-- bilingual-en:end -->

[[正定二次型的椭球|正定二次型]]给出有界椭球。如果单位[[特征对|特征向量]] $q_j$ 满足 $Sq_j=\lambda_jq_j$，该方向的半轴长为 $\sqrt{c_\alpha\lambda_j/n}$。样本变动越大的方向越长，非对角协方差可以使主轴旋转。置信区域包含内部，不只是边界曲面。

> [!example] A vector can fail while its coordinates fit · 坐标都落入区间，向量仍可能在区域外
> <!-- bilingual-en:start -->
> Take $n=10$, $p=2$, $\bar x=(1,1)^T$ and $S=\operatorname{diag}(1,4)$, where $\operatorname{diag}(1,4)$ has diagonal entries 1 and 4 and zero off-diagonal entries. At 95% confidence, $c_{0.05}\approx10.033$. The region is
> $$10\left[(m_1-1)^2+\frac{(m_2-1)^2}{4}\right]\le10.033.$$
> Its coordinate projections are approximately $[-0.002,2.002]$ and $[-1.003,3.003]$. Both contain zero, yet $m=(0,0)^T$ gives 12.5 on the left, outside the region. The coordinate intervals form a rectangle enclosing the ellipse; points near that rectangle's corners need not belong to the ellipse.
> <!-- bilingual-en:end -->
>
> 取上述样本摘要，95% 区域由所列不等式给出。两个坐标投影约为 $[-0.002,2.002]$ 和 $[-1.003,3.003]$，都包含零；但候选向量 $(0,0)^T$ 使左边等于 $12.5$，已经在椭球外。两个坐标区间构成包住椭圆的矩形，靠近矩形角落的点未必属于椭圆。这里 $\operatorname{diag}(1,4)$ 表示对角元为 1、4，非对角元为零的矩阵。

<!-- bilingual-en:start -->
Thus exclusion by any coordinate interval from this same ellipsoid implies global rejection, but the converse fails. For $p=1$, $F_{1,n-1}(1-\alpha)=[t_{n-1}(1-\alpha/2)]^2$ and the region reduces to the usual two-sided $t$ interval; the symbols in parentheses here denote quantiles.
<!-- bilingual-en:end -->

所以，由同一椭球投影得到的任一坐标区间排除目标值，都会推出全局拒绝；反方向不成立。$p=1$ 时，利用 $F_{1,n-1}(1-\alpha)=[t_{n-1}(1-\alpha/2)]^2$，区域退化为普通双侧 $t$ 区间；这里括号中的概率表示分位数。

> [!warning] This region is for a mean parameter · 这是均值参数的区域
> <!-- bilingual-en:start -->
> It is not a prediction region for a future individual observation or a tolerance region intended to contain a chosen fraction of the population. Those targets need different variance and calibration. If $S$ is singular, this ordinary inverse is undefined; replacing it with a pseudoinverse does not automatically yield a bounded region with the same coverage. The exact guarantee also relies on the stated Gaussian sampling model.
> <!-- bilingual-en:end -->
>
> 它不是未来单个观测的预测区域，也不是希望包住某一比例总体观测的容忍区域；这些目标需要不同的方差和校准。若 $S$ 奇异，普通逆没有定义；换成伪逆并不会自动得到具有相同覆盖率的有界区域。精确覆盖保证还依赖所列 Gaussian 抽样模型。

## 来源与核验

<!-- bilingual-en:start -->
- [Penn State STAT 505, Lesson 7, §7.1.4](https://online.stat.psu.edu/stat505/Lesson07): inversion of the F-calibrated statistic, joint mean region and simultaneous intervals for all linear combinations.
- [Peter Hoff, Multivariate Analysis notes, PDF p.166](https://www2.stat.duke.edu/~pdh10/Teaching/832/Materials/notes.pdf): confidence regions obtained by retaining non-rejected parameter values and their repeated-sampling coverage.
- [[01_Math/04_多元统计分析/05_ 总体平均向量的推论.md#1.5. 置信区域与同时置信区间|Course §1.5]]: region, coordinate and arbitrary-direction intervals, and the one-way implication from coordinate exclusion to global rejection.
- [[Hotelling投影极值|Maximum-projection identity]] and [[正定二次型的椭球|positive-definite ellipsoid geometry]]: direct checks of equivalence, endpoints and semiaxes. The worked example and coordinate-coverage distinction are derived from the same event.
<!-- bilingual-en:end -->

- Penn State §7.1.4 核对 F 统计量反演、均值联合区域与全部线性组合的同时区间。
- Hoff PDF 第166页核对检验反演与重复抽样覆盖率。
- 课程 §1.5 核对区域、坐标区间与任意方向区间，以及坐标排除到全局拒绝的单向关系。
- 最大投影恒等式与正定椭球几何支持正文对等价事件、端点和半轴的直接计算；算例与坐标覆盖率的区别也由同一事件推出。

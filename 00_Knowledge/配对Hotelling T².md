---
aliases:
  - Paired Hotelling T-squared test
  - 配对 Hotelling T²
  - 配对均值向量比较
student_os: knowledge-atom
atom_id: STAT-HOT-006
atom_set: hotelling-mean-inference
atom_type: design-reduction
status: source-checked
requires:
  - "[[单样本Hotelling T²]]"
  - "[[样本协方差矩阵]]"
part_of:
  - "[[Hotelling T² 与多元均值推断.canvas]]"
related:
  - "[[线性约束Hotelling T²]]"
  - "[[pooled Hotelling T²]]"
---

<!-- bilingual-en:start -->
# Paired Hotelling T-squared test
<!-- bilingual-en:end -->

*配对 Hotelling T² 检验*

<!-- bilingual-en:start -->
A paired study compares two measurements that belong together: before and after measurements on the same person, two laboratories measuring the same specimen, or two deliberately matched units. If each measurement has several coordinates, the **paired Hotelling test** asks whether the population mean change vector equals a proposed target. The crucial first step is to subtract within each pair. The resulting difference vectors form one sample for the [[单样本Hotelling T²|one-sample Hotelling test]].
<!-- bilingual-en:end -->

配对研究比较有对应关系的两次测量，例如同一个人的前后测量、两个实验室对同一份样本的测量，或经过设计匹配的两个单位。若每次测量都有多个坐标，**配对 Hotelling 检验**就用来判断总体平均变化向量是否等于给定目标。关键的第一步是在每一对内部相减；得到的差向量组成一份样本，交给[[单样本Hotelling T²|单样本 Hotelling 检验]]。

<!-- bilingual-en:start -->
## One pair becomes one difference vector

For pair $i$, let $X_i,Y_i\in\mathbb R^p$ contain the same $p\ge1$ variables, in the same order and corresponding units. Choose and report a subtraction direction, for example after minus before:
$$D_i=X_i-Y_i.$$
The [[均值向量|mean difference vector]] is $\delta=E[D_i]$. If the original means exist, this equals $E[X_i]-E[Y_i]$. The common target $\delta_0=0$ expresses no mean change in any coordinate; a nonzero fixed $\delta_0$ can represent a specified change. We test $H_0:\delta=\delta_0$ against a departure in at least one coordinate.

Here $n$ counts **pairs**, not the $2n$ original measurements. For $n\ge2$ complete pairs, compute the [[样本均值向量|sample mean]] and [[样本协方差矩阵|sample covariance]] of the differences:
<!-- bilingual-en:end -->

*一对观测变成一个差向量*

对第 $i$ 对，$X_i,Y_i\in\mathbb R^p$ 包含相同的 $p\ge1$ 个变量，排列顺序和对应单位一致。先选定并报告减法方向，例如后测减前测：
$$D_i=X_i-Y_i.$$
[[均值向量|均值差向量]]为 $\delta=E[D_i]$。如果原始均值存在，它等于 $E[X_i]-E[Y_i]$。常用目标 $\delta_0=0$ 表示所有坐标的平均变化都为零；固定的非零 $\delta_0$ 可以表示指定变化。检验 $H_0:\delta=\delta_0$，备择是假设至少一个坐标偏离该目标。

这里 $n$ 数的是**对数**，而不是 $2n$ 次原始测量。对 $n\ge2$ 个完整配对，计算差值的[[样本均值向量]]与[[样本协方差矩阵]]：

$$
\bar D=\frac1n\sum_{i=1}^nD_i,\qquad
S_D=\frac1{n-1}\sum_{i=1}^n(D_i-\bar D)(D_i-\bar D)^T.
$$

<!-- bilingual-en:start -->
$S_D$ describes variation among individual difference vectors. The estimated covariance of their average is $S_D/n$, so a discrepancy $\bar D-\delta_0$ is measured relative to $S_D/n$, not to the two original covariance matrices separately.
<!-- bilingual-en:end -->

$S_D$ 描述各个差向量之间的波动；平均差的协方差估计是 $S_D/n$。所以，$\bar D-\delta_0$ 的偏离程度应相对于 $S_D/n$ 衡量，而不是分别拿两个原始协方差矩阵作尺度。

> [!important] The exact model is a model for the differences · 精确模型作用于差向量
> <!-- bilingual-en:start -->
> Require $D_1,\ldots,D_n$ to be [[独立同分布|iid]] [[联合Gaussian|jointly Gaussian]] $N_p(\delta,\Sigma_D)$, with [[正定矩阵|positive-definite]] covariance $\Sigma_D$ and $n>p$. Then $S_D$ is invertible with probability one. Under $H_0$,
> <!-- bilingual-en:end -->
>
> 要求 $D_1,\ldots,D_n$ [[独立同分布|iid]] 于[[联合Gaussian|联合 Gaussian]] 分布 $N_p(\delta,\Sigma_D)$，其中协方差 $\Sigma_D$ [[正定矩阵|正定]]且 $n>p$。此时 $S_D$ 以概率一可逆。在 $H_0$ 下，
>
> $$
> T_D^2=n(\bar D-\delta_0)^TS_D^{-1}(\bar D-\delta_0),\qquad
> \boxed{\frac{n-p}{p(n-1)}T_D^2\sim F_{p,n-p}.}
> $$
>
> <!-- bilingual-en:start -->
> The [[F分布|F degrees of freedom]] are $p,n-p$, in that order. At a significance level $0<\alpha<1$ fixed before inspecting the results, reject when the scaled statistic exceeds the cumulative $1-\alpha$ F quantile. When $p=1$, $T_D^2$ is the square of the ordinary paired t statistic, with $n-1$ degrees of freedom for the [[Student t分布|t distribution]].
> <!-- bilingual-en:end -->
>
> [[F分布|F 自由度]]依次是 $p,n-p$。在查看结果前确定显著性水平 $0<\alpha<1$，缩放后的统计量超过累积概率为 $1-\alpha$ 的 F 分位数时拒绝原假设。若 $p=1$，$T_D^2$ 就是普通配对 t 统计量的平方，对应 [[Student t分布|t 分布]]的自由度为 $n-1$。

<!-- bilingual-en:start -->
This is the one-sample theorem with $D_i$ as its observations; no second sample-size factor or pooled covariance is introduced. A jointly Gaussian pair $(X_i,Y_i)$ has a Gaussian difference by [[Gaussian仿射闭包|affine closure]]. But joint normality of the whole pair is only sufficient: the exact test needs normality of $D_i$, and does not require $X_i$ and $Y_i$ to be independent or separately normal. Conversely, normal marginal measurements alone do not guarantee a normal difference.
<!-- bilingual-en:end -->

这就是把 $D_i$ 当作观测的单样本定理，不会再引入第二个样本量因子或合并协方差。若整对 $(X_i,Y_i)$ 联合 Gaussian，由[[Gaussian仿射闭包]]可知差值 Gaussian；但整对的联合正态只是充分条件。精确检验需要的是 $D_i$ 正态，不要求 $X_i,Y_i$ 相互独立，也不要求它们分别正态。反过来，两个边际测量各自正态，也不能保证差向量正态。

<!-- bilingual-en:start -->
## Why within-pair dependence matters

When the original coordinates have finite second moments, define $\Sigma_X=\operatorname{Cov}(X_i)$, $\Sigma_Y=\operatorname{Cov}(Y_i)$ and the cross-covariance matrix
$$\Sigma_{XY}=E[(X_i-E[X_i])(Y_i-E[Y_i])^T],\qquad \Sigma_{YX}=\Sigma_{XY}^T.$$
Expanding the centered product for $D_i$ gives the [[协方差仿射变换|covariance transformation]]
<!-- bilingual-en:end -->

*为什么不能忽略对内相关性*

如果原始各坐标的二阶矩有限，定义 $\Sigma_X=\operatorname{Cov}(X_i)$、$\Sigma_Y=\operatorname{Cov}(Y_i)$，以及交叉协方差矩阵
$$\Sigma_{XY}=E[(X_i-E[X_i])(Y_i-E[Y_i])^T],\qquad \Sigma_{YX}=\Sigma_{XY}^T.$$
展开 $D_i$ 的中心化乘积，得到[[协方差仿射变换|协方差变换公式]]

$$\Sigma_D=\Sigma_X+\Sigma_Y-\Sigma_{XY}-\Sigma_{YX}.$$

<!-- bilingual-en:start -->
The cross terms carry the within-pair relationship. In one coordinate, variances $9$ and $9$ with covariance $8$ give difference variance $9+9-2(8)=2$, not $18$. Positive covariance can make a difference more precise; negative covariance can increase its variance. The [[和的方差协方差项|variance expansion]] explains the sign, so pairing does not promise a gain in every situation. Computing $S_D$ directly from the differences automatically retains the observed cross terms.

Dependence within a pair and dependence across pairs are separate questions. Shared shocks across subjects can make the $D_i$ dependent, invalidating the displayed exact calibration. The required sampling independence belongs to the difference vectors, not to every measurement inside each vector.
<!-- bilingual-en:end -->

交叉项记录对内关系。若一个坐标的两次测量方差均为 $9$、协方差为 $8$，差值方差是 $9+9-2(8)=2$，不是 $18$。正协方差可能使差值更精确，负协方差则可能增大差值波动；[[和的方差协方差项|方差展开]]解释了这个符号，因此配对并不保证在所有情况下提高精度。直接用差值计算 $S_D$，就已经保留了样本中的交叉项。

对内依赖和对间依赖是两回事。跨对象的共同冲击可能使 $D_i$ 相互依赖，从而破坏上面的精确校准。抽样独立性要求作用于各个差向量，而不是要求向量内部的每次测量都独立。

> [!example] Four pairs and two measurements · 四对观测与两个指标
> <!-- bilingual-en:start -->
> The following small arithmetic example uses after minus before. It illustrates the calculation; the table alone cannot establish the Gaussian sampling assumptions.
> <!-- bilingual-en:end -->
>
> 下面的小例子采用后测减前测，用于演示计算；仅凭这张表不能确立 Gaussian 抽样假设。
>
> | Pair / 配对 | Before $Y_i$ / 前测 | After $X_i$ / 后测 | Difference $D_i$ / 差值 |
> | --- | --- | --- | --- |
> | 1 | $(10,20)$ | $(10,20)$ | $(0,0)$ |
> | 2 | $(14,18)$ | $(16,18)$ | $(2,0)$ |
> | 3 | $(9,25)$ | $(9,27)$ | $(0,2)$ |
> | 4 | $(20,11)$ | $(22,13)$ | $(2,2)$ |
>
> <!-- bilingual-en:start -->
> Here $n=4,p=2$, $\bar D=(1,1)^T$, and the centered differences are $(-1,-1),(1,-1),(-1,1),(1,1)$. Their outer products sum to $\begin{pmatrix}4&0\\0&4\end{pmatrix}$, so
> <!-- bilingual-en:end -->
>
> 此处 $n=4,p=2$，$\bar D=(1,1)^T$，中心化差值为 $(-1,-1),(1,-1),(-1,1),(1,1)$。它们的外积之和为 $\begin{pmatrix}4&0\\0&4\end{pmatrix}$，因此
>
> $$
> S_D=\begin{pmatrix}4/3&0\\0&4/3\end{pmatrix},\qquad
> T_D^2=4(3/4+3/4)=6\quad(\delta_0=0).
> $$
>
> <!-- bilingual-en:start -->
> The F statistic is $\frac{4-2}{2(4-1)}6=2$, with degrees of freedom $2,2$. Its 95th percentile is $19$, so this result would not reject at $\alpha=0.05$ under the exact model. Four pairs provide little information for a two-dimensional test; a nonzero observed average does not itself establish a population change.
> <!-- bilingual-en:end -->
>
> F 统计量为 $\frac{4-2}{2(4-1)}6=2$，自由度为 $2,2$；其 95% 分位数为 $19$，所以在精确模型下，$\alpha=0.05$ 时不拒绝原假设。四对观测对二维检验提供的信息很少，样本平均变化非零本身并不能确立总体变化。

<!-- bilingual-en:start -->
## Direction, intervals and the meaning of a pair

Reversing the subtraction replaces $D_i$ with $-D_i$, $\bar D$ with $-\bar D$, and leaves $S_D$ unchanged. For a zero target the statistic and p-value are unchanged. For a nonzero target, also replace $\delta_0$ by $-\delta_0$ to preserve the same question. An interval $[L,U]$ for one directional mean difference becomes $[-U,-L]$ for its reverse.

The [[Hotelling置信椭球|Hotelling confidence ellipsoid]] applies directly to $\delta$. Under the same model, define $c_\alpha=p(n-1)F_{p,n-p}(1-\alpha)/(n-p)$ using the cumulative F quantile. The intervals
$$a^T\bar D\ \pm\ \sqrt{\frac{c_\alpha}{n}\,a^TS_Da}$$
cover $a^T\delta$ simultaneously for every nonzero $a\in\mathbb R^p$, with joint coverage $1-\alpha$. This describes population mean differences, not individual future differences. For just the coordinate means, the corresponding subset of these intervals has joint coverage at least $1-\alpha$.
<!-- bilingual-en:end -->

*方向、区间与配对的含义*

反过来相减，会把 $D_i$ 变成 $-D_i$、$\bar D$ 变成 $-\bar D$，而 $S_D$ 不变。目标为零时，统计量和 p 值都不变；若目标非零，也必须把 $\delta_0$ 改为 $-\delta_0$，才能保持同一个检验问题。某一方向均值差的区间 $[L,U]$，反向后成为 $[-U,-L]$。

[[Hotelling置信椭球]]可直接用于 $\delta$。在相同模型下，用累积 F 分位数定义 $c_\alpha=p(n-1)F_{p,n-p}(1-\alpha)/(n-p)$，则区间
$$a^T\bar D\ \pm\ \sqrt{\frac{c_\alpha}{n}\,a^TS_Da}$$
以联合覆盖率 $1-\alpha$ 同时覆盖所有非零 $a\in\mathbb R^p$ 对应的 $a^T\delta$。它描述的是总体均值差，不是未来单个对象的差值。若只取各坐标均值对应的区间，这个子族的联合覆盖率至少为 $1-\alpha$。

> [!warning] Preserve the sampling model when forming differences · 构造差值时保留抽样模型
> <!-- bilingual-en:start -->
> A meaningful pairing comes from the study design or a justified sampling construction. Sorting two observed groups and matching similar values can create dependent, non-Gaussian differences; equal row numbers alone do not establish the model. A fixed, data-independent one-to-one pairing of two mutually independent iid Gaussian samples of equal size gives iid Gaussian differences with covariance $\Sigma_X+\Sigma_Y$; the exact test applies when that sum is positive definite and $n>p$. It creates no within-pair correlation advantage, and it need not be the most efficient analysis of those groups.
>
> Complete-case analysis does not automatically repair missing pairs. The retained differences must still have the assumed distribution and independence, and their mean must correspond to the intended population. Selection depending on the unobserved changes can alter both. If $S_D$ is singular, the classical inverse and F calibration do not apply; substituting a pseudoinverse requires a separately justified method. Finally, a before–after mean difference is not by itself a causal treatment effect.
> <!-- bilingual-en:end -->
>
> 有意义的配对来自研究设计或有依据的抽样构造。对两组已观测数据先排序、再匹配相近值，可能产生相互依赖且不正态的差值；行号相同本身也不能确立模型。不过，两份彼此独立、各自 iid Gaussian 且大小相同的样本，若采用预先固定、不依赖数据的一一匹配，得到的差值确实 iid Gaussian，协方差为 $\Sigma_X+\Sigma_Y$；当这个和正定且 $n>p$ 时，精确检验适用。这种配对不会凭空带来对内相关性的优势，也未必是分析两组数据的最高效方法。
>
> 只保留完整配对不会自动解决缺失问题：保留下来的差值仍需符合所假设的分布与独立性，均值也必须对应目标总体；依赖未观测变化的选择可能同时改变这些条件。若 $S_D$ 奇异，经典逆矩阵和 F 校准不可用，换成伪逆需要另有依据的方法。最后，前后均值变化本身也不等于处理的因果效应。

## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/04_多元统计分析/06_比较多个均值向量comparisons of multivariate mean vectors.md#1.2. 配对样本均值向量比较|Multivariate statistics course, Chapter 6 §1.2]]: paired observations, difference-vector notation, the $n-1$ covariance divisor and simultaneous mean intervals.
- [Penn State STAT 505, Lesson 7, §§7.1.6–7.1.10](https://online.stat.psu.edu/stat505/Lesson07): matched sampling units, assumptions on the differences, paired Hotelling statistic, exact F transformation and intervals. The four-pair example above is calculated directly rather than borrowed from the course's spouse dataset.
- [MIT 6.436J Lecture 14, pp. 3–6, Theorem 1(c)](https://ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/ceb423b6a703abd60429367b733f45bf_MIT6_436JF18_lec14.pdf): joint Gaussian linear transformations and the cross-covariance definition. The difference-covariance expansion, sign reversal and fixed independent-pairing case follow directly.
- [NIST/SEMATECH, F distribution](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3665.htm): F density and cumulative distribution. For $F_{2,2}$ its CDF is $x/(1+x)$ for $x\ge0$, giving the example's 95th percentile $19$.
<!-- bilingual-en:end -->

- 多元统计课程第 6 章 §1.2 核对配对观测、差向量记号、协方差分母 $n-1$ 与均值同时区间。
- Penn State §7.1.6–§7.1.10 核对匹配抽样单位、差值假设、配对 Hotelling 统计量、精确 F 变换和区间。正文四对观测的例子由所列数据直接计算，并未借用该课程的配偶数据。
- MIT 第 3–6 页定理 1(c) 核对联合 Gaussian 线性变换和交叉协方差定义。差值协方差展开、反向相减及固定独立配对特例均由此直接推导。
- NIST 核对 F 密度与累计分布；$F_{2,2}$ 在 $x\ge0$ 时的 CDF 为 $x/(1+x)$，从而例子的 95% 分位数为 $19$。

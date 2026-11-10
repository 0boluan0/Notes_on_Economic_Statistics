---
aliases:
  - "Maximum-projection interpretation of Hotelling T-squared"
  - "Hotelling T² 的最大投影解释"
  - "Hotelling 与逐坐标 t 检验"
student_os: knowledge-atom
atom_id: STAT-HOT-002
atom_set: hotelling-mean-inference
atom_type: equivalence-interpretation
status: source-checked
requires:
  - "[[单样本Hotelling T²]]"
  - "[[正定与半正定可逆性]]"
part_of:
  - "[[Hotelling T² 与多元均值推断.canvas]]"
implies:
  - "[[Hotelling置信椭球]]"
related:
  - "[[Bonferroni对比区间]]"
---

<!-- bilingual-en:start -->
# Hotelling T-squared as the largest projected departure
<!-- bilingual-en:end -->

*Hotelling T² 的最大投影解释*

<!-- bilingual-en:start -->
A multivariate mean can look unusual in a **combination** of measurements even when no single measurement stands out. For example, two instruments may usually rise and fall together. A sample in which one rises and the other falls can then produce an unusually large difference. [[单样本Hotelling T²|Hotelling's $T^2$]] searches all linear combinations and measures the largest squared standardized mean departure.
<!-- bilingual-en:end -->

多元均值的异常可能出现在测量值的**组合**上，而不突出地出现在任何单项测量上。例如，两个仪器的结果平时总是一起升降；如果一个升、另一个降，它们的差可能异常。[[单样本Hotelling T²|Hotelling T²]]衡量所有线性组合中最大的标准化均值偏离平方。

<!-- bilingual-en:start -->
Let $d=\bar X-\mu_0$ be the sample-mean departure and let $S\succ0$ be the [[样本协方差矩阵|sample covariance]], using divisor $n-1$ for $n\ge2$ observations. A nonzero vector $a=(a_1,\ldots,a_p)^T$ chooses the scalar measurement $a^TX_i=\sum_j a_jX_{ij}$. Its sample mean departs from its target by $a^Td$, and its sample variance is $a^TSa$. The estimated [[标准误含义|standard error]], which measures sampling variation of its mean, is $\sqrt{a^TSa/n}$, so define
$$t^2(a)=\frac{n(a^Td)^2}{a^TSa}.$$
For example, $a=(1,-1)^T$ compares the two measurements by subtraction. Multiplying $a$ by any nonzero constant changes neither $t^2(a)$ nor the direction being tested.
<!-- bilingual-en:end -->

记 $d=\bar X-\mu_0$，令 $S\succ0$ 是由 $n\ge2$ 次观测、分母 $n-1$ 算出的[[样本协方差矩阵]]。非零向量 $a$ 选择标量测量 $a^TX_i=\sum_j a_jX_{ij}$：它的样本均值偏离目标 $a^Td$，样本方差为 $a^TSa$，均值的估计[[标准误含义|标准误]]（衡量均值的抽样变动）为 $\sqrt{a^TSa/n}$。上式就是这个投影的标准化偏离平方。例如 $a=(1,-1)^T$ 比较两项之差；把 $a$ 乘以非零常数，不改变该比值。

> [!important] One statistic searches every direction · 一个统计量概括全部方向
> <!-- bilingual-en:start -->
> For any $d\in\mathbb R^p$ and real symmetric $S\succ0$,
> $$\max_{a\ne0}t^2(a)=n d^TS^{-1}d=T^2.$$
> If $d\ne0$, the maximizing directions are $a=cS^{-1}d$ with $c\ne0$. If $d=0$, every nonzero direction gives zero. This is an algebraic identity: Gaussian sampling is needed for the exact reference distribution, not for the identity itself.
> <!-- bilingual-en:end -->
>
> 对任意 $d$ 与实对称正定 $S$，全部非零投影中的最大值就是 $T^2$。$d\ne0$ 时，最大方向为 $S^{-1}d$ 的任意非零倍；$d=0$ 时，所有非零方向都给出零。这是代数恒等式，本身不需要 Gaussian 假设；精确参考分布才需要相应抽样条件。

<!-- bilingual-en:start -->
## Why the maximum has this form
<!-- bilingual-en:end -->

*最大值为什么是这个二次型*

<!-- bilingual-en:start -->
The [[半正定主平方根|symmetric square root]] $S^{1/2}$ and its [[正定逆平方根|inverse]] $S^{-1/2}$ rescale the two vectors so that
$$a^Td=(S^{1/2}a)^T(S^{-1/2}d).$$
Apply the [[Cauchy–Schwarz不等式|Cauchy–Schwarz inequality]], which bounds a dot product by the product of the lengths:
$$(a^Td)^2\le(a^TSa)(d^TS^{-1}d).$$
Dividing by $a^TSa>0$ gives $t^2(a)\le T^2$. When $d\ne0$, equality requires $S^{1/2}a$ to be a nonzero multiple of $S^{-1/2}d$, which is exactly $a=cS^{-1}d$. Thus the bound is attained, not merely approached.
<!-- bilingual-en:end -->

用[[半正定主平方根|对称平方根]] $S^{1/2}$ 及其[[正定逆平方根|逆]] $S^{-1/2}$ 重写点积，再用[[Cauchy–Schwarz不等式]]，就得到上式。除以正数 $a^TSa$，便有 $t^2(a)\le T^2$。$d\ne0$ 时，等号要求两个变换后的向量成非零比例，即 $a=cS^{-1}d$；因此最大值确实能取到。

> [!example] A quiet difference can be very informative · 平时很稳定的差值得注意
> <!-- bilingual-en:start -->
> Let $n=10$, $d=(0.25,-0.25)^T$, and
> $$S=\begin{bmatrix}1&0.9\\0.9&1\end{bmatrix}.$$
> Each coordinate alone has $t^2=10(0.25)^2=0.625$. For $a=(1,-1)^T$, however, $a^Td=0.5$ while $a^TSa=1+1-2(0.9)=0.2$. Hence $t^2(a)=10(0.5)^2/0.2=12.5$. This is the maximum because $S^{-1}d=(2.5,-2.5)^T$. The large score reflects the small variation of the difference, not large coordinate departures.
> <!-- bilingual-en:end -->
>
> 每个坐标单独的 $t^2$ 都只有 $0.625$。但差值方向的偏离为 $0.5$，方差仅为 $0.2$，于是 $t^2(1,-1)=12.5$。$S^{-1}d=(2.5,-2.5)^T$，说明这个方向恰好达到最大值。较大的统计量来自差值平时很稳定，而非各坐标偏离很大。

<!-- bilingual-en:start -->
## Searching changes the reference distribution
<!-- bilingual-en:end -->

*搜索方向会改变参考分布*

<!-- bilingual-en:start -->
Under iid $N_p(\mu_0,\Sigma)$ sampling with $\Sigma\succ0$, the statistic for one **fixed, prespecified** nonzero $a$ has a $t_{n-1}^2=F_{1,n-1}$ law. Choosing $a$ after examining the data to maximize the score is a different operation. When $n>p$, its maximum instead uses the [[单样本Hotelling T²|Hotelling F calibration]],
$$\frac{n-p}{p(n-1)}\max_{a\ne0}t^2(a)\sim F_{p,n-p}.$$
In the example, the maximum is compared with the 5% Hotelling threshold $c_{0.05}\approx10.033$, not with an ordinary single-direction $t^2$ threshold. It exceeds that joint threshold.
<!-- bilingual-en:end -->

在 iid $N_p(\mu_0,\Sigma)$、$\Sigma\succ0$ 的模型下，一个**事先固定**的非零 $a$ 对应 $t_{n-1}^2=F_{1,n-1}$ 分布。看过数据后寻找最大方向，则是另一项操作。$n>p$ 时，最大值要使用[[单样本Hotelling T²|Hotelling 的 F 校准]]。例子中的最大值应与 5% 联合阈值 $c_{0.05}\approx10.033$ 比较，而不是套单一方向的普通 $t^2$ 阈值；它超过联合阈值。

<!-- bilingual-en:start -->
A global rejection provides evidence against the target vector through at least one direction. It neither establishes every coordinate's significance nor identifies a unique scientifically important effect. Coordinate tests only examine $e_1,\ldots,e_p$, where $e_j$ has a 1 in position $j$ and zeros elsewhere; testing each at level $\alpha$ without adjustment generally fails to control the probability of any false rejection. [[Bonferroni对比区间|Bonferroni intervals]] address a prespecified finite collection. [[Hotelling置信椭球|Hotelling simultaneous intervals]] instead cover every direction through one joint event, so selecting a direction afterward does not destroy that simultaneous coverage guarantee.
<!-- bilingual-en:end -->

全局拒绝意味着至少某个方向提供了反对目标向量的证据；不表示每个坐标都显著，也不自动确定唯一具有实际意义的效应。逐坐标检验只检查 $e_1,\ldots,e_p$，其中 $e_j$ 只有第 $j$ 位为 1，其余为零；每项都用水平 $\alpha$ 而不调整，一般无法控制“至少一次误拒绝”的概率。[[Bonferroni对比区间]]针对预先指定的有限比较组；[[Hotelling置信椭球|Hotelling 同时区间]]则通过同一联合事件覆盖所有方向，因此事后选择某个方向不会破坏这套同时覆盖保证。

<!-- bilingual-en:start -->
If $S$ is singular, some nonzero directions have $a^TSa=0$ and this ordinary-inverse statement no longer applies. A generalized inverse needs its own domain restrictions and distributional justification; it cannot inherit the displayed F law merely from a formal substitution.
<!-- bilingual-en:end -->

若 $S$ 奇异，某些非零方向满足 $a^TSa=0$，普通逆版本的恒等式就失去前提。广义逆需要另行明确允许的方向与分布依据，不能形式上替换以后直接继承上述 F 律。

## 来源与核验

<!-- bilingual-en:start -->
- [Stanford STATS305C, Hotelling T2, “Largest t statistic interpretation”](https://web.stanford.edu/class/stats305c/lectures/Hotelling_T2.html#largest-t-statistic-interpretation): optimization over directions and the square-root proof. The source uses the two-sample scale; this card substitutes the one-sample variance $S/n$ and proves the identity directly.
- [Penn State STAT 505, Lesson 7, §§7.1.2–7.1.4](https://online.stat.psu.edu/stat505/Lesson07): coordinatewise multiplicity, joint calibration and simultaneous interpretation.
- [[01_Math/04_多元统计分析/05_ 总体平均向量的推论.md#1.2.2. 从一元 t 检验到多元 T²|Course §1.2.2]] and [[01_Math/04_多元统计分析/05_ 总体平均向量的推论.md#1.5.3. 线性组合的同时置信区间|§1.5.3]]: one-sample notation and all-direction inference. The example and equality cases are computed above.
<!-- bilingual-en:end -->

- Stanford 核对全方向优化与平方根证明；原源是两样本尺度，本卡换成单样本的 $S/n$ 并直接证明。
- Penn State 核对逐坐标多重性、联合校准与同时区间；课程 §§1.2.2、1.5.3 核对单样本记号及全方向推断。数值与等号条件由正文直接计算。

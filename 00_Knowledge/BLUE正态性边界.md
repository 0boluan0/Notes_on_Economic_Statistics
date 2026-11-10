---
aliases:
  - Normality and OLS inference
  - OLS 正态性边界
  - OLS BLUE 无需正态性
student_os: knowledge-atom
atom_id: ECON-OLS-013
atom_type: distinction
status: source-checked
part_of:
  - "[[OLS 线性回归.canvas|OLS 线性回归]]"
  - "[[回归推断.canvas|回归推断]]"
requires:
  - "[[经典 Gauss–Markov 定理]]"
---

<!-- bilingual-en:start -->
# Normality, BLUE, and OLS inference
<!-- bilingual-en:end -->

*正态性、BLUE 与 OLS 推断*

<!-- bilingual-en:start -->
Normality answers a different question from BLUE. **BLUE (best linear unbiased estimator)** means that ordinary least squares has the smallest variance among the specified linear unbiased rules. A **normal sampling distribution** describes the entire pattern of its estimates across repeated samples. Knowing which rule has the smallest variance does not determine that pattern.
<!-- bilingual-en:end -->

正态性与 BLUE 回答不同的问题。**BLUE（最佳线性无偏估计量）**表示普通最小二乘在指定的线性无偏规则中方差最小；**正态抽样分布**描述重复抽样时估计值的完整分布形状。知道哪条规则方差最小，并不能确定这个形状。

<!-- bilingual-en:start -->
## An average can be BLUE and discrete
<!-- bilingual-en:end -->

*平均数可以既是 BLUE，又只有几个可能值*

<!-- bilingual-en:start -->
Measure one unknown quantity $\beta$ twice: $Y_1=\beta+u_1$ and $Y_2=\beta+u_2$. Let the two errors be independent, each taking $-1$ or $1$ with probability $1/2$. Both have mean zero and variance one. Fitting a common constant by [[普通最小二乘|ordinary least squares]] means minimising the sum of squared discrepancies; it gives the average $\hat\beta=(Y_1+Y_2)/2$.
<!-- bilingual-en:end -->

对同一个未知量 $\beta$ 测量两次：$Y_1=\beta+u_1$、$Y_2=\beta+u_2$。两个误差相互独立，各以 $1/2$ 的概率取 $-1$ 或 $1$，所以均值为零、方差为一。用[[普通最小二乘]]拟合一个共同常数，就是最小化两次读数与这个常数的差的平方和，得到平均数 $\hat\beta=(Y_1+Y_2)/2$。

$$
P(\hat\beta=\beta-1)=\frac14,\qquad
P(\hat\beta=\beta)=\frac12,\qquad
P(\hat\beta=\beta+1)=\frac14.
$$

<!-- bilingual-en:start -->
The four error pairs are equally likely. Two pairs have opposite signs and average to zero; the other two average to $-1$ or $1$. This [[抽样分布|sampling distribution]] has three possible values, so it is not a nondegenerate normal distribution. Nevertheless, [[经典 Gauss–Markov 定理|the classical Gauss–Markov theorem]] makes this average BLUE: among rules $aY_1+(1-a)Y_2$, the variance is $1/2+2(a-1/2)^2$, minimised at $a=1/2$.
<!-- bilingual-en:end -->

四种误差组合等可能，其中两种符号相反、平均为零，另两种平均为 $-1$ 或 $1$。这个[[抽样分布]]只有三个可能值，并非非退化正态分布。但根据[[经典 Gauss–Markov 定理]]，平均数仍是 BLUE：在 $aY_1+(1-a)Y_2$ 这些规则中，方差为 $1/2+2(a-1/2)^2$，在 $a=1/2$ 处最小。

<!-- bilingual-en:start -->
## Three distinct roles
<!-- bilingual-en:end -->

## 三件事不要混在一起

<!-- bilingual-en:start -->
For a real regression model $y=X\beta+u$, the $n\times k$ design matrix $X$ contains one row per observation and $k$ regression terms, including an intercept if present. The unknown vector $\beta$ contains the coefficients, and $u$ contains the model errors. The [[回归残差|fitted residuals]] $\hat u=y-X\hat\beta$ use estimated coefficients and are different from those errors.
<!-- bilingual-en:end -->

在实数回归模型 $y=X\beta+u$ 中，$n\times k$ 设计矩阵 $X$ 每行对应一个观测，包含 $k$ 个回归项；若有截距，也计入 $k$。未知向量 $\beta$ 包含各系数，$u$ 包含模型误差。[[回归残差|拟合残差]] $\hat u=y-X\hat\beta$ 使用估计出的系数，与模型误差不是同一个量。

<!-- bilingual-en:start -->
**Computing the fit** is an algebraic problem. [[满列秩与OLS唯一性|Full column rank]] means the columns of $X$ are linearly independent and gives the unique formula $\hat\beta=(X^TX)^{-1}X^Ty$. No error distribution is needed to calculate it.
<!-- bilingual-en:end -->

**计算拟合**是代数问题。[[满列秩与OLS唯一性|满列秩]]表示 $X$ 的各列线性无关，由此得到唯一公式 $\hat\beta=(X^TX)^{-1}X^Ty$。计算它不需要指定误差分布。

<!-- bilingual-en:start -->
**Establishing BLUE** adds a statistical model. With $X$ fixed, assume $E(u\mid X)=0$ and $\operatorname{Cov}(u\mid X)=\sigma^2I_n$, where $0<\sigma^2<\infty$ and $I_n$ is the identity matrix. Every error then has the same finite conditional variance, and distinct errors have zero conditional covariance. These are moment conditions, not a normality assumption. They give the Gauss–Markov comparison and the covariance $\sigma^2(X^TX)^{-1}$.
<!-- bilingual-en:end -->

**证明 BLUE**需要统计模型。在固定 $X$ 后，假设 $E(u\mid X)=0$、$\operatorname{Cov}(u\mid X)=\sigma^2I_n$，其中 $0<\sigma^2<\infty$，$I_n$ 是单位矩阵。于是每个误差的条件方差相同且有限，不同误差的条件协方差为零。这些是矩条件，并非正态性假设；它们给出 Gauss–Markov 比较与协方差 $\sigma^2(X^TX)^{-1}$。

<!-- bilingual-en:start -->
**Obtaining exact classical reference distributions** uses an additional distributional model. The standard Gaussian route is $u\mid X\sim N_n(0,\sigma^2I_n)$: the entire error vector is [[联合Gaussian|jointly normal]] with the stated mean and covariance. If $X$ is random, all these conditional statements apply to almost every realised full-rank design; an unconditional normal distribution for $\hat\beta$ does not follow automatically, because its conditional covariance depends on $X$.
<!-- bilingual-en:end -->

**得到经典检验的精确参考分布**还需要分布模型。标准的 Gaussian 路线是 $u\mid X\sim N_n(0,\sigma^2I_n)$：整个误差向量[[联合Gaussian|联合正态]]，且均值、协方差如式所示。若 $X$ 随机，这些条件表述适用于几乎每个实现出的满列秩设计；不能自动把 $\hat\beta$ 的无条件分布也叫正态，因为其条件协方差会随 $X$ 改变。

<!-- bilingual-en:start -->
## What the Gaussian assumption supplies
<!-- bilingual-en:end -->

*Gaussian 假设具体提供什么？*

<!-- bilingual-en:start -->
A linear transformation of a Gaussian vector is Gaussian. Thus the OLS formula gives, conditional on $X$,
<!-- bilingual-en:end -->

Gaussian 向量的线性变换仍是 Gaussian，因此由 OLS 公式，在给定 $X$ 后有

$$
\hat\beta\mid X\sim N_k\!\left(\beta,\sigma^2(X^TX)^{-1}\right).
$$

<!-- bilingual-en:start -->
When $\sigma^2$ is unknown, estimating it from residuals requires **$n>k$**. Define the residual sum of squares $RSS=\hat u^T\hat u$ and $s^2=RSS/(n-k)$, with $s=\sqrt{s^2}$, as explained in [[OLS经典协方差估计|classical OLS covariance estimation]]. Under this Gaussian model,
<!-- bilingual-en:end -->

若 $\sigma^2$ 未知，用残差估计它要求 **$n>k$**。定义残差平方和 $RSS=\hat u^T\hat u$、$s^2=RSS/(n-k)$，并令 $s=\sqrt{s^2}$，详细由来见[[OLS经典协方差估计]]。在上述 Gaussian 模型下，

$$
\frac{RSS}{\sigma^2}\mid X\sim\chi^2_{n-k},\qquad
\hat\beta\ \text{and}\ RSS\ \text{are independent conditional on }X.
$$

<!-- bilingual-en:start -->
The [[卡方分布|chi-square distribution]] appears because the residual has $n-k$ remaining orthogonal directions, whose scaled Gaussian coordinates are independent standard normals. Its squared length adds their squares. The fitted and residual parts have zero cross-covariance; their joint Gaussianity upgrades that zero covariance to independence. Without a distributional condition, [[零协方差不推独立|zero covariance alone would not do so]].
<!-- bilingual-en:end -->

[[卡方分布]]出现，是因为残差剩下 $n-k$ 个相互正交的方向，缩放后的 Gaussian 坐标是独立标准正态量；平方长度就是这些坐标的平方和。拟合部分与残差部分的交叉协方差为零，联合 Gaussian 性进一步把零协方差升级为独立；没有相应的分布条件，[[零协方差不推独立|单凭零协方差做不到这一步]]。

> [!important] Exactness comes from the whole construction · 精确性来自整套构造
> <!-- bilingual-en:start -->
> Fix a nonzero coefficient-weight vector $a\in\mathbb R^k$ and a number $c$. Under the null hypothesis $a^T\beta=c$, the [[回归t检验|classical regression statistic]] satisfies
> <!-- bilingual-en:end -->
>
> 固定一个非零系数权重向量 $a\in\mathbb R^k$ 和数值 $c$。在原假设 $a^T\beta=c$ 下，[[回归t检验|经典回归统计量]]满足
>
> $$T=\frac{a^T\hat\beta-c}{s\sqrt{a^T(X^TX)^{-1}a}}\ \Big|\ X\sim t_{n-k}.$$
>
> <!-- bilingual-en:start -->
> The numerator, divided by its true standard deviation, is standard normal. The estimated denominator contributes the square root of an independent chi-square variable scaled by $1/(n-k)$. This is the construction of the [[Student t分布|Student t distribution]]. For $q$ linearly independent coefficient restrictions, $1\le q\le k$, the corresponding [[嵌套模型F检验|classical joint test]] has an exact [[F分布|$F_{q,n-k}$ distribution]] under the null. Its two degrees of freedom count the restrictions and residual directions respectively.
> <!-- bilingual-en:end -->
>
> 分子除以真实标准差后是标准正态量，估计出的分母则带入一个独立卡方量除以 $n-k$ 后的平方根。这正是 [[Student t分布]]的构造。对 $q$ 个线性无关的系数限制，$1\le q\le k$，相应的[[嵌套模型F检验|经典联合检验]]在原假设下具有精确的 [[F分布|$F_{q,n-k}$ 分布]]，两个自由度分别数限制个数与残差方向数。

<!-- bilingual-en:start -->
These are sufficient conditions for this familiar exact theory, not a claim that every possible exact test must assume Gaussian errors. Nor does replacing $s$ with any other standard-error formula preserve the same finite-sample distribution. When $n=k$, full-rank OLS still exists, but it fits every outcome exactly and leaves zero residual degrees of freedom for estimating $\sigma^2$ this way.
<!-- bilingual-en:end -->

这些条件足以建立这套熟悉的精确理论，并不是说一切精确检验都必须假设 Gaussian 误差。把 $s$ 换成任意其他标准误公式，也不会自动保留同一有限样本分布。$n=k$ 时满秩 OLS 仍存在，但它会精确拟合所有观测，不再留下残差自由度来按此方法估计 $\sigma^2$。

> [!warning] Revisit the two-point errors · 回到两点误差例子
> <!-- bilingual-en:start -->
> In the opening example, $n=2$, $k=1$, and the two observations coincide whenever their errors have the same sign. This happens with probability $1/2$ and gives $RSS=s^2=0$, even though the true error variance is one. The usual ratio is then undefined. BLUE and an unbiased estimate of error variance do not by themselves produce a continuous $t_1$ reference distribution.
> <!-- bilingual-en:end -->
>
> 开篇例子中 $n=2$、$k=1$；两个误差同号时，两次观测完全相同。这一事件的概率为 $1/2$，此时 $RSS=s^2=0$，但真实误差方差仍是一，常规比值无法定义。因此，BLUE 加上无偏的误差方差估计，本身不足以给出连续的 $t_1$ 参考分布。

<!-- bilingual-en:start -->
## Continue with the relevant inference question
<!-- bilingual-en:end -->

## 继续

<!-- bilingual-en:start -->
With nonnormal errors, suitably standardised OLS estimates can still approach a normal distribution as the sample grows, under appropriate moment, design, and sampling or dependence conditions. This is an approximation, not a consequence of BLUE alone. [[标准误口径匹配|The covariance estimator must match those conditions]]. A normal-looking [[残差图边界|residual plot]] does not establish zero conditional mean, homoskedasticity, independence, or a correct regression mean. Use [[回归t检验|the regression t-test card]] to distinguish exact calibration from large-sample calibration.
<!-- bilingual-en:end -->

即使误差不正态，只要合适的矩条件、设计条件及抽样或依赖条件成立，适当标准化的 OLS 估计量仍可能随样本增长而趋近正态。但这是近似理论，不是单凭 BLUE 就能得到的结论，[[标准误口径匹配|协方差估计也必须匹配相应条件]]。看起来像正态的[[残差图边界|残差图]]不能证明零条件均值、同方差、独立性或回归均值设定正确；精确校准与大样本校准的使用区别见[[回归t检验]]。

## 来源与核验

<!-- bilingual-en:start -->
- Bruce E. Hansen, [*Econometrics*, §§4.5, 4.7, 4.13; PDF pp. 57–58, 60–61, 68–70](https://users.ssc.wisc.edu/~xshi28/Econometrics_Hansen.pdf#page=57): the linear-unbiased comparison, residual-variance correction, conditional Gaussian coefficient law, independence, chi-square law and exact t ratio. §§5.4 and 5.8 (PDF pp. 77–79, 86) distinguish asymptotic normality from exact distributions. The text sometimes leaves conditioning implicit; it is kept explicit here.
- [Penn State STAT 501, Lesson 6, §6.2, General Linear F-Test](https://online.stat.psu.edu/stat501/Lesson06#the-general-linear-f-test): the joint-test construction and the two residual-degree-of-freedom counts; the Gaussian independence argument above explains its exact calibration.
- [[02_Economy/01_Econometrics/03_多元线性回归.md#4.1. 极大似然估计|The permanent course section on normal-error likelihood]] and [[02_Economy/01_Econometrics/03_多元线性回归.md#6.1. JB 正态性检验|the JB diagnostic section]] locate normality in the course. They are course context, not a proof that normal-looking residuals validate all regression assumptions. The two-point error example is enumerated directly above.
<!-- bilingual-en:end -->

- Hansen《Econometrics》第 4.5、4.7、4.13 节（PDF pp. 57–58、60–61、68–70）核对线性无偏比较、残差方差修正、条件 Gaussian 系数分布、独立性、卡方分布与精确 t 比值；第 5.4、5.8 节（PDF pp. 77–79、86）区分渐近正态与精确分布。正文始终明确条件于 $X$。
- Penn State STAT 501 第 6 课第 6.2 节核对联合检验的构造与两个自由度；正文结合 Gaussian 独立性解释精确校准。
- [[02_Economy/01_Econometrics/03_多元线性回归.md#4.1. 极大似然估计|永久课程的正态误差似然部分]]与[[02_Economy/01_Econometrics/03_多元线性回归.md#6.1. JB 正态性检验|JB 诊断部分]]提供课程位置，不把残差正态外观当作全部假设成立的证明。两点误差例子由正文直接枚举核验。

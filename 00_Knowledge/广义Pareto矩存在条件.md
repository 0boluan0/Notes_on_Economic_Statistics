---
aliases:
  - 广义Pareto分布的正阶矩有限当且仅当形状参数小于该阶数的倒数
  - GPD矩存在条件
student_os: knowledge-atom
atom_id: RM-EVT-009
atom_type: theorem
status: source-checked
part_of:
  - "[[极值理论 EVT 与尾部风险.canvas]]"
requires:
  - "[[广义Pareto分布]]"
  - "[[矩存在性的使用边界]]"
related:
  - "[[极大值吸引域]]"
leads_to:
  - "[[POT尾部ES]]"
---

# 广义Pareto分布的正阶矩有限当且仅当形状参数小于该阶数的倒数
<!-- bilingual-en:start -->
*A positive-order GPD moment is finite exactly when its shape is smaller than the reciprocal of the moment order*
<!-- bilingual-en:end -->

设 $Y$ 精确服从[[广义Pareto分布]] $G_{\xi,\beta}$、$\beta>0$。对任意实数 $k>0$，有 $E[Y^k]<\infty$ 当且仅当 $\xi<1/k$。因此均值有限要求 $\xi<1$，二阶矩及方差有限要求 $\xi<1/2$；$\xi\le0$ 时所有正阶矩都有限。
<!-- bilingual-en:start -->
Let $Y$ have an exact [[广义Pareto分布|GPD]] law $G_{\xi,\beta}$ with $\beta>0$. For every real $k>0$, $E[Y^k]<\infty$ exactly when $\xi<1/k$. A finite mean requires $\xi<1$, and a finite second moment and variance require $\xi<1/2$. All positive-order moments are finite for $\xi\le0$.
<!-- bilingual-en:end -->

理由可从非负随机变量的尾积分看出：
<!-- bilingual-en:start -->
The nonnegative-variable tail integral gives the reason:
<!-- bilingual-en:end -->

$$
E[Y^k]=k\int_0^\infty y^{k-1}P(Y>y)\,dy.
$$

当 $\xi>0$，GPD 的生存函数在无穷远与常数乘 $y^{-1/\xi}$ 同阶，故积分的尾部与 $\int^\infty y^{k-1-1/\xi}\,dy$ 同敛散，仅在 $k<1/\xi$ 时收敛；等号时出现对数发散。$\xi=0$ 时指数尾使全部正阶矩有限；$\xi<0$ 时支持有界，结论也成立。
<!-- bilingual-en:start -->
For $\xi>0$, GPD survival is asymptotic to a positive constant times $y^{-1/\xi}$. Its moment integral therefore has the same tail convergence as $\int^\infty y^{k-1-1/\xi}\,dy$, which is finite only for $k<1/\xi$; equality gives logarithmic divergence. Exponential tails at $\xi=0$ and bounded support at $\xi<0$ give every positive-order moment.
<!-- bilingual-en:end -->

在相应存在条件下，直接积分得到：
<!-- bilingual-en:start -->
Direct integration, under the respective existence conditions, gives:
<!-- bilingual-en:end -->

$$
E[Y]=\frac{\beta}{1-\xi}\quad(\xi<1),\qquad
\operatorname{Var}(Y)=\frac{\beta^2}{(1-\xi)^2(1-2\xi)}\quad(\xi<1/2).
$$

例如 $\beta=2,\xi=1/4$ 时，均值为 $8/3$、方差为 $128/9$。若只将形状改为 $3/4$，均值为 8，但方差不再有限；若形状为 1，非负变量的均值为 $+\infty$。不能把这些参数代入有限矩公式后，以负数或分母为零的结果作为有效风险度量。
<!-- bilingual-en:start -->
With $\beta=2,\xi=1/4$, the mean is $8/3$ and the variance is $128/9$. Changing only the shape to $3/4$ gives mean 8 but no finite variance. At shape 1, the nonnegative variable has mean $+\infty$. Substituting inadmissible shapes into finite-moment formulas does not turn negative values or division by zero into valid risk measures.
<!-- bilingual-en:end -->

本结论针对精确 GPD，包括明确选定的拟合 GPD 模型；它不是仅凭某个形状估计就证实真实损失分布矩存在与否。也不能把“GPD 在临界阶 $k=1/\xi$ 发散”直接搬给任意同[[极大值吸引域]]的母分布：母分布的临界矩还取决于其更细的尾部行为。应用 [[POT尾部ES]] 前应明确是在使用拟合模型及其 $\xi<1$ 条件。
<!-- bilingual-en:start -->
This result concerns an exact GPD, including an explicitly chosen fitted GPD model; a shape estimate alone does not establish moment existence for the true loss distribution. Nor does GPD divergence at $k=1/\xi$ automatically transfer to every parent in the same [[极大值吸引域|maximum domain of attraction]]: critical moments depend on finer parent-tail behaviour. Applying [[POT尾部ES|POT expected shortfall]] requires making the fitted model and its $\xi<1$ condition explicit.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- Martin Haugh，[Extreme Value Theory，PDF 第 19、21、31 页](https://www.columbia.edu/~mh2078/QRM/EVT_MasterSlides.pdf#page=31)：核对 GPD 生存函数、均值与 GPD 的临界阶矩发散；第 11 页对一般 Fréchet 吸引域仅给出严格高于尾指数的发散结论。本文用尾积分独立补全任意 $k>0$ 的充要条件、二阶矩及方差公式，并复算数值例。
- [[矩存在性的使用边界]]：复用“先核验有限矩再用公式”的一般判断，不重复定义期望或方差。
<!-- bilingual-en:start -->
- Haugh's notes, PDF pp. 19, 21, and 31, support GPD survival, the mean, and divergence at the critical GPD moment order. Page 11 states divergence only strictly above the tail index for a general Fréchet-domain parent. The tail integral independently completes the criterion for all real $k>0$, the second moment and variance, and the numerical checks.
- [[矩存在性的使用边界|Moment-existence boundaries]] supplies the general requirement to check finiteness before using a formula, without duplicating definitions of expectation or variance.
<!-- bilingual-en:end -->

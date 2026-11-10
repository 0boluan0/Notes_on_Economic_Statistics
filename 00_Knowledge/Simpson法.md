---
aliases:
  - Composite Simpson's one-third rule
  - Simpson's rule
  - 复合Simpson法
student_os: knowledge-atom
atom_id: CALC-DEFINT-034
atom_type: method
status: source-checked
requires:
  - "[[定积分]]"
related:
  - "[[Simpson组合恒等式]]"
  - "[[Simpson误差界]]"
part_of:
  - "[[定积分与应用.canvas]]"
---
<!-- bilingual-en:start -->
# Simpson's rule
<!-- bilingual-en:end -->
*Simpson 法*

<!-- bilingual-en:start -->
Simpson's rule estimates a [[定积分|definite integral]] from a table of function values. Over two adjacent equal-width intervals, it fits a polynomial of degree at most two through the left, middle and right sampled points, then integrates that polynomial. Repeating this on successive pairs gives the **composite Simpson rule**. The calculation uses sampled heights and widths; it does not require finding an antiderivative.
<!-- bilingual-en:end -->

Simpson 法用一组函数值来估计[[定积分]]。每次取相邻的两个等宽小区间，用一个次数不超过二的多项式经过左端、中点、右端三个采样点，再对这个多项式积分。把相邻小区间依次配对并相加，就是**复合 Simpson 法**。计算只用采样高度与宽度，不需要先求出原函数。

> [!example] Three heights give a first estimate · 用三个高度得到第一次数值估计
> <!-- bilingual-en:start -->
> To estimate $\int_1^2 dx/x$, split $[1,2]$ into two intervals of width $h=1/2$. The inputs and heights are
> <!-- bilingual-en:end -->
>
> 为估计 $\int_1^2 dx/x$，把 $[1,2]$ 分成两个宽度为 $h=1/2$ 的小区间。输入与高度为
>
> | $x$ | $1$ | $3/2$ | $2$ |
> |---|---:|---:|---:|
> | $f(x)=1/x$ | $1$ | $2/3$ | $1/2$ |
>
> <!-- bilingual-en:start -->
> Simpson's rule weights these heights by $1,4,1$, then multiplies by $h/3$:
> <!-- bilingual-en:end -->
>
> Simpson 法给三个高度分别乘上 $1,4,1$，再整体乘以 $h/3$：
>
> $$S_2=\frac{1/2}{3}\left(1+4\cdot\frac23+\frac12\right)
> =\frac{25}{36}\approx0.694444.$$
>
> <!-- bilingual-en:start -->
> The subscript $2$ counts the two small intervals, not the three sample points. The exact integral is [[自然对数积分表示|$\ln2$]], approximately $0.693147$, so this estimate is close but not exact. We next explain where the weights come from.
> <!-- bilingual-en:end -->
>
> 下标 $2$ 数的是两个小区间，不是三个采样点。由[[自然对数积分表示]]，准确积分为 $\ln2\approx0.693147$，因此这次估计很接近，却不是精确值。下面解释这些权重从何而来。

<!-- bilingual-en:start -->
## Why the middle height receives weight four
<!-- bilingual-en:end -->
*为什么中点的权重是四*

<!-- bilingual-en:start -->
For one pair, place the origin at its midpoint: write $u=x-m$, where $m$ is that midpoint. The three input positions become $-h,0,h$. A quadratic interpolant is a polynomial that takes the same three sampled values as $f$; write it as $q(u)=A+Bu+Cu^2$. Here $A$ is the middle height, while $B$ and $C$ adjust the slope and curvature to fit the two other heights. A straight line or a constant is allowed when the corresponding coefficients vanish.
<!-- bilingual-en:end -->

先只看一对小区间，把其中点移到坐标原点：令 $u=x-m$，其中 $m$ 是中点。三个输入位置便成为 $-h,0,h$。所谓二次插值多项式，就是在这三个位置与 $f$ 取相同值的多项式，写为 $q(u)=A+Bu+Cu^2$。其中 $A$ 是中点高度，$B,C$ 调整斜率和弯曲程度，以配合另外两个高度；相应系数为零时，也允许退化成直线或常数。

$$\int_{-h}^{h}q(u)\,du=2hA+\frac{2h^3}{3}C
=\frac h3\bigl[q(-h)+4q(0)+q(h)\bigr].$$

<!-- bilingual-en:start -->
The $Bu$ term integrates to zero over this symmetric interval. Expanding the bracket on the right gives the same expression, so the rule integrates the fitted polynomial exactly. It approximates $f$ only because $f$ need not agree with that polynomial between the sampled points.
<!-- bilingual-en:end -->

$Bu$ 在对称区间上的积分为零。展开右边的括号，也得到相同的表达式，因此这个规则对拟合出的多项式积分是精确的。对原函数 $f$ 只是近似，是因为在三个采样点之间，$f$ 未必与这个多项式重合。

<!-- bilingual-en:start -->
## Join the pairs across the whole interval
<!-- bilingual-en:end -->
*把每一对小区间拼起来*

<!-- bilingual-en:start -->
Let $f$ be a real-valued function on a finite interval $[a,b]$ with $a<b$. Choose an **even integer $N\ge2$**. Divide $[a,b]$ into $N$ equal small intervals of width $h=(b-a)/N$, with nodes $x_i=a+ih$ for $i=0,1,\ldots,N$. There are $N+1$ nodes and $N/2$ pairs, often called panels. For each panel $[x_{2j-2},x_{2j}]$, use its three heights and add its Simpson value to the total.
<!-- bilingual-en:end -->

设 $f$ 是有限区间 $[a,b]$ 上的实值函数，其中 $a<b$。取一个**偶数 $N\ge2$**。把 $[a,b]$ 等分成 $N$ 个小区间，宽度为 $h=(b-a)/N$，节点为 $x_i=a+ih$，其中 $i=0,1,\ldots,N$。一共有 $N+1$ 个节点、$N/2$ 对小区间；每一对也称一个面板。对每个面板 $[x_{2j-2},x_{2j}]$ 使用其三个高度，再把各面板的 Simpson 值相加。

> [!important] Composite Simpson's one-third rule · 复合 Simpson 三分之一法
>
> $$\boxed{S_N=\frac h3\left[f(x_0)
> +4\sum_{j=1}^{N/2}f(x_{2j-1})
> +2\sum_{j=1}^{N/2-1}f(x_{2j})+f(x_N)\right].}$$
>
> <!-- bilingual-en:start -->
> The summation symbol $\sum$ means add all terms with the indicated indices; when $N=2$, the second sum has no terms and is zero. Odd-indexed nodes are panel midpoints and receive weight $4$. Interior even-indexed nodes are shared endpoints of two panels, so their weights $1+1$ become $2$. The two outer endpoints receive weight $1$.
> <!-- bilingual-en:end -->
>
> 求和符号 $\sum$ 表示把指定下标的项相加；当 $N=2$ 时，第二个求和没有任何项，按零处理。奇数下标的节点是面板中点，权重为 $4$；内部偶数下标的节点是两个面板共用的端点，两个权重 $1+1$ 合成 $2$；最外侧两个端点的权重都是 $1$。

<!-- bilingual-en:start -->
For six small intervals, the seven weights are $1,4,2,4,2,4,1$. They sum to $18=3N$, so a constant height $f=1$ gives $(h/3)(3N)=Nh=b-a$, exactly the width of the interval. This checks both the endpoint weights and the factor $h/3$.
<!-- bilingual-en:end -->

六个小区间对应七个权重 $1,4,2,4,2,4,1$，总和为 $18=3N$。因此，常高度 $f=1$ 给出 $(h/3)(3N)=Nh=b-a$，恰好是区间宽度。这个检查同时核对了端点权重和前面的 $h/3$。

<!-- bilingual-en:start -->
Some books count panels instead: their $n$ panels mean our $N=2n$ small intervals, with $h=(b-a)/(2n)$. The [[Simpson组合恒等式|trapezoidal–midpoint identity]] uses this distinction: coarse-grid values $T_n$ and $M_n$ combine to give the finer-grid value $S_{2n}=(T_n+2M_n)/3$.
<!-- bilingual-en:end -->

有些教材改数面板：教材中的 $n$ 个面板，就是这里 $N=2n$ 个小区间，此时 $h=(b-a)/(2n)$。[[Simpson组合恒等式]]也使用这一区分：粗网格的梯形值 $T_n$ 和中点值 $M_n$，组合得到更细网格的 $S_{2n}=(T_n+2M_n)/3$。

<!-- bilingual-en:start -->
## Exact for cubics, approximate for general functions
<!-- bilingual-en:end -->
*对三次多项式精确，对一般函数近似*

<!-- bilingual-en:start -->
The rule is exact for every polynomial of degree at most three. The quadratic part was checked above; after centring a panel, the additional cubic term $Du^3$ has both zero integral and zero weighted sum at $-h,0,h$. Thus its contribution also agrees exactly. Adding panels preserves this exactness.
<!-- bilingual-en:end -->

这个规则对次数不超过三的每个多项式都精确。二次部分已经核对；把面板居中以后，额外的三次项 $Du^3$ 不仅积分为零，在 $-h,0,h$ 处的加权和也为零，因此它的贡献也精确匹配。各面板相加后仍保持这一性质。

> [!warning] A computable number is not yet an accuracy guarantee · 算得出一个数，不等于已有精度保证
> <!-- bilingual-en:start -->
> The displayed weights require equally spaced nodes and an even number of small intervals. Computing them needs finite real values at those nodes; using the result to approximate an ordinary integral also presupposes that the integral exists. [[连续函数可积|Continuity on the closed interval]] is one sufficient condition. A quantitative guarantee such as an error bound proportional to $h^4$ needs additional smoothness and a derivative bound; see [[Simpson误差界|the Simpson error bound]]. Cubic exactness alone does not promise a chosen accuracy for a general function.
> <!-- bilingual-en:end -->
>
> 上面的固定权重要求节点等距、小区间数为偶数。计算加权和需要各节点处取有限实数值；用它近似普通积分，还预先要求该积分存在。[[连续函数可积|闭区间上的连续性]]是一个充分条件。若要得到与 $h^4$ 成比例的定量误差保证，还需额外的光滑性和导数界，见 [[Simpson误差界]]。对三次多项式精确，并不能单独保证对一般函数达到某个指定精度。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Ses63d_Problems.pdf#page=1|MIT 18.01SC Session 63d, pp.1–2]]: paired equal intervals, quadratic fitting and composite weights. The centred-polynomial calculation above derives the coefficients directly.
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/03_Definite_Integrals/Ses64b_Lecture_Notes.pdf#page=1|Session 64b, p.1]] and [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/03_Definite_Integrals/Ses64c_Lecture_Notes.pdf#page=1|Session 64c, p.1]]: the reciprocal example, cubic exactness and the constant-height check. Accuracy conditions are stated separately above.
- [OpenStax Calculus 2 §3.6, Simpson's Rule](https://openstax.org/books/calculus-volume-2/pages/3-6-numerical-integration): even subdivisions, the one-third formula and the connection to midpoint and trapezoidal values.
<!-- bilingual-en:end -->

- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Ses63d_Problems.pdf#page=1|MIT 18.01SC Session 63d，第1–2页]]：等宽小区间配对、二次拟合和复合权重；正文用居中的多项式直接推导系数。
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/03_Definite_Integrals/Ses64b_Lecture_Notes.pdf#page=1|Session 64b，第1页]]与 [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/03_Definite_Integrals/Ses64c_Lecture_Notes.pdf#page=1|Session 64c，第1页]]：倒数函数算例、三次精确性和常高度检查；精度所需条件在正文单独说明。
- [OpenStax《Calculus 2》第3.6节 Simpson 法](https://openstax.org/books/calculus-volume-2/pages/3-6-numerical-integration)：偶数细分、三分之一公式及其与中点值和梯形值的联系。

---
aliases:
  - Composite Simpson error bound
student_os: knowledge-atom
atom_id: CALC-DEFINT-038
atom_type: theorem
status: source-checked
requires:
  - "[[Simpson法]]"
related:
  - "[[梯形积分误差界]]"
  - "[[中点积分误差界]]"
part_of:
  - "[[定积分与应用.canvas]]"
---
<!-- bilingual-en:start -->
# Simpson error bound
<!-- bilingual-en:end -->
*Simpson 误差界*

<!-- bilingual-en:start -->
An error bound tells us how far a [[Simpson法|Simpson estimate]] can lie from the true integral, using information about the function between the sampled points. For this rule, a bound on the **fourth derivative** controls the error. The fourth derivative $f^{(4)}$ means applying differentiation four times, as in [[高阶导数|higher derivatives]]; it is not the fourth power of $f$.
<!-- bilingual-en:end -->

误差界利用采样点之间的函数信息，告诉我们 [[Simpson法|Simpson 估计值]]与真实积分最多能差多少。这个规则由**四阶导数**的界来控制误差。四阶导数 $f^{(4)}$ 是[[高阶导数]]中连续求导四次的结果，不是 $f$ 的四次方。

> [!example] An error that reaches the bound · 恰好达到上界的误差
> <!-- bilingual-en:start -->
> For $f(x)=x^4$ on $[0,1]$, two small intervals give $h=1/2$ and
> <!-- bilingual-en:end -->
>
> 在 $[0,1]$ 上对 $f(x)=x^4$ 使用两个小区间，此时 $h=1/2$，
>
> $$S_2=\frac16\left(0+4\cdot\frac1{16}+1\right)=\frac5{24},
> \qquad I=\int_0^1x^4\,dx=\frac15.$$
>
> <!-- bilingual-en:start -->
> Define the signed error as $E_2=I-S_2=-1/120$. Its negative sign says Simpson's value is too large; the absolute error is the positive distance $|E_2|=1/120$. Since $f^{(4)}=24$, the bound below gives exactly this distance. The bound is therefore not merely a description of the error's order: its constant matters too.
> <!-- bilingual-en:end -->
>
> 把带符号误差定义为 $E_2=I-S_2=-1/120$。负号说明 Simpson 值偏大；绝对误差则是正的距离 $|E_2|=1/120$。由于 $f^{(4)}=24$，下面的界恰好给出这个距离。因此，误差界不只是描述误差的阶数，其中的常数也有意义。

<!-- bilingual-en:start -->
## What must be known over the whole interval
<!-- bilingual-en:end -->
*需要掌握整个区间上的什么信息*

<!-- bilingual-en:start -->
Let $a<b$ be finite real endpoints. Suppose the real-valued function $f$ belongs to $C^4[a,b]$: its derivatives through order four exist in the interior and extend continuously to the closed interval, with $f$ itself continuous there. This is a sufficient smoothness assumption. Let $K_4\ge0$ be a known finite bound satisfying $|f^{(4)}(x)|\le K_4$ for every $x\in[a,b]$. It may be larger than the smallest possible bound; a larger choice gives a more conservative guarantee.
<!-- bilingual-en:end -->

设 $a<b$ 为有限实端点。假设实值函数 $f\in C^4[a,b]$：直至四阶的导数在内部存在，并能连续延拓到闭区间，且 $f$ 本身也在闭区间上连续。这是一组充分的光滑性条件。取一个已知有限数 $K_4\ge0$，使每个 $x\in[a,b]$ 都满足 $|f^{(4)}(x)|\le K_4$。它可以比最小可用界更大；选得越大，保证越保守。

<!-- bilingual-en:start -->
Use an even integer $N\ge2$ of equal small intervals, let $h=(b-a)/N$, and let $S_N$ be the composite Simpson value. Write $I=\int_a^b f(x)\,dx$ for the true integral and $E_N=I-S_N$ for the signed error.
<!-- bilingual-en:end -->

使用偶数 $N\ge2$ 个等宽小区间，令 $h=(b-a)/N$，以 $S_N$ 表示复合 Simpson 值。记真实积分为 $I=\int_a^b f(x)\,dx$，带符号误差为 $E_N=I-S_N$。

> [!important] Absolute error guarantee · 绝对误差保证
>
> $$\boxed{|E_N|\le\frac{K_4(b-a)h^4}{180}
> =\frac{K_4(b-a)^5}{180N^4}.}$$
>
> <!-- bilingual-en:start -->
> The bound on the derivative must hold throughout the interval, not just at the sampled nodes. A bound on $f''$ is a different piece of information and cannot be substituted for $K_4$ here. In the quartic example, $K_4=24$, $b-a=1$ and $N=2$, giving $24/(180\cdot16)=1/120$.
> <!-- bilingual-en:end -->
>
> 导数界必须在整个区间成立，不能只检查采样节点。二阶导数 $f''$ 的界是另一类信息，不能在这里替代 $K_4$。在开头的四次多项式例子中，$K_4=24$、$b-a=1$、$N=2$，代入得到 $24/(180\cdot16)=1/120$。

<!-- bilingual-en:start -->
## Add the errors from pairs of intervals
<!-- bilingual-en:end -->

## 每两段的余项怎样累加

<!-- bilingual-en:start -->
Set $x_i=a+ih$. Each panel contains two small intervals and has total width $2h$. The single-panel Simpson remainder theorem, under the stated smoothness assumption, gives a point $\xi_j$ strictly inside panel $j$ such that
<!-- bilingual-en:end -->

令 $x_i=a+ih$。每个面板由两个小区间组成，总宽度为 $2h$。在上述光滑性条件下，单面板的 Simpson 余项定理保证：对第 $j$ 个面板，内部存在一点 $\xi_j$，使

$$\int_{x_{2j-2}}^{x_{2j}} f(x)\,dx
-\frac h3\bigl[f(x_{2j-2})+4f(x_{2j-1})+f(x_{2j})\bigr]
=-\frac{(2h)^5}{2880}f^{(4)}(\xi_j)
=-\frac{h^5}{90}f^{(4)}(\xi_j).$$

<!-- bilingual-en:start -->
The unknown point $\xi_j$ is supplied by the theorem; it is not an extra node we must locate. Its fourth derivative is still bounded by $K_4$, so the absolute error on one panel is at most $K_4h^5/90$. There are $N/2$ panels. Summing their signed errors and using the fact that the absolute value of a sum is at most the sum of absolute values gives
<!-- bilingual-en:end -->

未知点 $\xi_j$ 由定理保证存在，不是需要额外找出的采样节点。该点的四阶导数仍被 $K_4$ 控制，所以一个面板的绝对误差至多为 $K_4h^5/90$。总共有 $N/2$ 个面板。先把带符号误差相加，再利用“和的绝对值不超过绝对值之和”，得到

$$|E_N|\le\frac N2\frac{K_4h^5}{90}
=\frac{K_4(Nh)h^4}{180}
=\frac{K_4(b-a)h^4}{180}.$$

<!-- bilingual-en:start -->
This also explains the error's sign when the fourth derivative has one sign: if $f^{(4)}\ge0$ throughout, every panel error is nonpositive and $S_N\ge I$; if $f^{(4)}\le0$, then $S_N\le I$. If it changes sign, panel errors may cancel. When $f$ has degree at most three, $f^{(4)}=0$, giving zero error, as the method's direct polynomial calculation already showed.
<!-- bilingual-en:end -->

这也解释了四阶导数同号时的误差方向：若处处 $f^{(4)}\ge0$，每个面板误差都非正，因此 $S_N\ge I$；若处处 $f^{(4)}\le0$，则 $S_N\le I$。如果四阶导数变号，不同面板的误差可能抵消。当 $f$ 的次数不超过三时，$f^{(4)}=0$，误差为零，与方法卡中直接核算多项式得到的结论一致。

<!-- bilingual-en:start -->
## Choose a subdivision count for a tolerance
<!-- bilingual-en:end -->
*根据允许误差选择细分数*

<!-- bilingual-en:start -->
If a required absolute error is $\varepsilon>0$, choose an even integer $N$ satisfying
<!-- bilingual-en:end -->

若要求绝对误差不超过 $\varepsilon>0$，可选择满足下式的偶数 $N$：

$$N\ge\max\left\{2,\left(\frac{K_4(b-a)^5}{180\varepsilon}\right)^{1/4}\right\}.$$

<!-- bilingual-en:start -->
For a threshold of $7.2$, choose $N=8$; for $8.1$, choose $N=10$. If the threshold is already an admissible even integer, keep it. For the quartic example with $\varepsilon=1/1000$, the threshold is $(400/3)^{1/4}\approx3.40$, so $N=4$ suffices; its bound is $1/1920<1/1000$.
<!-- bilingual-en:end -->

阈值为 $7.2$ 时取 $N=8$；为 $8.1$ 时取 $N=10$；如果阈值已经是合格的偶整数，就直接使用它。对开头的四次多项式，若要求 $\varepsilon=1/1000$，阈值为 $(400/3)^{1/4}\approx3.40$，因此取 $N=4$ 即可；对应的界为 $1/1920<1/1000$。

<!-- bilingual-en:start -->
Keeping the same function, interval and bound $K_4$, doubling $N$ divides this certified upper bound by $16$. It does not force the actual error to be divided by exactly $16$, because cancellations and the derivative values involved can change. If a book counts $n$ panels rather than small intervals, substitute $N=2n$ into the bound: the denominator becomes $180(2n)^4$.
<!-- bilingual-en:end -->

保持函数、区间和导数界 $K_4$ 不变，把 $N$ 加倍，就把这个有保证的上界除以 $16$。实际误差未必恰好也除以 $16$，因为误差抵消程度和涉及的导数值都可能改变。如果教材用 $n$ 数面板而不是小区间，应代入 $N=2n$，此时分母为 $180(2n)^4$。

> [!warning] What this guarantee covers · 这个保证控制什么
> <!-- bilingual-en:start -->
> The theorem bounds the integration error when exact function values are inserted into the Simpson formula. Rounded or noisy measurements and floating-point arithmetic add other errors. A smooth-looking plot, or two nearby numerical estimates that happen to agree, does not by itself establish the required fourth-derivative bound. If the stated conditions fail, this theorem supplies no guarantee; the numerical value may still be useful, but needs another justification.
> <!-- bilingual-en:end -->
>
> 定理控制的是把准确函数值代入 Simpson 公式后产生的积分近似误差。舍入或有噪声的测量值、浮点运算，还会带来其他误差。图像看起来平滑，或两次数值估计碰巧接近，都不能单独证明所需的四阶导数界。如果上述条件不满足，这个定理就不给出保证；算出的数仍可能有用，但需要其他依据。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [OpenStax Calculus 2 §3.6, Error Bound for Simpson's Rule](https://openstax.org/books/calculus-volume-2/pages/3-6-numerical-integration): the whole-interval fourth-derivative bound, even subdivisions and the constant $1/180$.
- [UBC Math 405, Topic 4b: Composite Quadrature, pp.1–2](https://www.math.ubc.ca/~cbm/math405/2016/04b_composite_quad.pdf): the signed single-panel remainder and its composite sum. Its notation uses $2n$ small intervals; this card calls that count $N$. The displayed quartic and tolerance examples are computed directly above.
<!-- bilingual-en:end -->

- [OpenStax《Calculus 2》第3.6节 Simpson 误差界](https://openstax.org/books/calculus-volume-2/pages/3-6-numerical-integration)：整个区间上的四阶导数界、偶数细分及常数 $1/180$。
- [UBC Math 405，Topic 4b: Composite Quadrature，第1–2页](https://www.math.ubc.ca/~cbm/math405/2016/04b_composite_quad.pdf)：带符号的单面板余项及其复合累加。原文用 $2n$ 表示小区间数，本卡把这个数量记为 $N$；正文的四次多项式和容差例子直接计算核验。

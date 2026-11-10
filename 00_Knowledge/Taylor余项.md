---
aliases:
  - Lagrange remainder
  - Taylor error bound
student_os: knowledge-atom
atom_id: CALC-APP-004
atom_set: derivative-applications
atom_type: theorem
status: source-checked
part_of:
  - "[[导数的应用.canvas]]"
  - "[[无穷级数与幂级数.canvas]]"
requires:
  - "[[Taylor多项式]]"
  - "[[拉格朗日中值定理]]"
leads_to:
  - "[[线性近似]]"
  - "[[二次近似]]"
  - "[[Taylor级数等于函数]]"
---
<!-- bilingual-en:start -->
# Taylor remainder
<!-- bilingual-en:end -->
*Taylor 余项*

<!-- bilingual-en:start -->
A Taylor approximation leaves something out. The **remainder** is exactly the difference between the true function value and the polynomial estimate. For example, $P_{2,0}(x)=1+x+x^2/2$ gives the estimate $1.105$ for $e^{0.1}$. The remainder asks how far that estimate is from the true value. A theorem about the next derivative lets us bound the difference without first knowing the true value to many decimal places.
<!-- bilingual-en:end -->

Taylor 近似省略了一部分信息。**余项**就是函数真值减去多项式估计值的精确差。例如，$P_{2,0}(x)=1+x+x^2/2$ 给出 $e^{0.1}$ 的估计值 $1.105$；余项要回答它离真值究竟多远。利用下一阶导数的定理，可以在不知道真值很多位小数的情况下，先给出误差上界。

<!-- bilingual-en:start -->
Let $a$ be the expansion centre, $n\ge0$ a finite integer, and $P_{n,a}$ the [[Taylor多项式|Taylor polynomial]] matching the value and derivatives through order $n$ at $a$. Define the signed remainder by
<!-- bilingual-en:end -->

设 $a$ 为展开中心，$n\ge0$ 是有限整数，$P_{n,a}$ 为在 $a$ 匹配函数值及直到 $n$ 阶导数的[[Taylor多项式]]。定义带符号的余项

$$R_{n,a}(x)=f(x)-P_{n,a}(x).$$

<!-- bilingual-en:start -->
This definition makes $f(x)=P_{n,a}(x)+R_{n,a}(x)$ an exact identity wherever the terms exist. It does not yet say that the remainder is small. Positive remainder means the estimate is too low; negative remainder means it is too high. Its absolute value is the absolute approximation error.
<!-- bilingual-en:end -->

只要各项有定义，就有精确恒等式 $f(x)=P_{n,a}(x)+R_{n,a}(x)$，但定义本身还没有说余项很小。余项为正，表示估计偏低；为负，表示估计偏高；其绝对值就是绝对近似误差。

<!-- bilingual-en:start -->
## The derivative must be controlled between the two inputs
<!-- bilingual-en:end -->
*导数条件要覆盖两个输入之间*

<!-- bilingual-en:start -->
Take $x\ne a$, and consider the closed interval joining these two finite real inputs. Assume that $f,f',\ldots,f^{(n)}$ are continuous on this interval, and that $f^{(n+1)}$ exists at every interior point. Here [[高阶导数|$f^{(k)}$]] means repeated differentiation, with $f^{(0)}=f$. At an endpoint, the derivatives through order $n$ are interpreted by their continuous one-sided extensions from inside the interval; no values outside it are required. For $n=0$, the assumptions mean simply that $f$ is continuous on the closed interval and differentiable inside.
<!-- bilingual-en:end -->

取 $x\ne a$，考虑连接这两个有限实输入的闭区间。假设 $f,f',\ldots,f^{(n)}$ 在整个闭区间上连续，而且 $f^{(n+1)}$ 在每个内点存在。这里[[高阶导数|$f^{(k)}$]]表示逐次求导，并记 $f^{(0)}=f$。端点处直到 $n$ 阶的导数，按从区间内部连续延伸得到的单侧值理解，不要求区间外还有定义。若 $n=0$，条件就是 $f$ 在闭区间连续、内部可导。

> [!important] Lagrange's remainder · Lagrange 余项
> <!-- bilingual-en:start -->
> Under these assumptions, some input $\xi$ strictly between $a$ and $x$ satisfies
> <!-- bilingual-en:end -->
>
> 在这些条件下，存在严格位于 $a$ 与 $x$ 之间的输入 $\xi$，使
>
> $$R_{n,a}(x)=\frac{f^{(n+1)}(\xi)}{(n+1)!}(x-a)^{n+1}.$$
>
> <!-- bilingual-en:start -->
> The symbol $\xi$ names an intermediate input, generally unknown and dependent on the centre, target and order. The factorial $(n+1)!$ is the product $1\cdot2\cdots(n+1)$. The conclusion is an exact formula, not merely an approximation.
> <!-- bilingual-en:end -->
>
> $\xi$ 是一个中间输入，通常不知道其具体位置，并且可能随中心、目标与阶数改变。阶乘 $(n+1)!$ 是乘积 $1\cdot2\cdots(n+1)$。结论是一条精确公式，不只是近似式。

<!-- bilingual-en:start -->
An easier sufficient condition is that $f$ has continuous derivatives through order $n+1$ on an open interval containing the entire segment; this is written $f\in C^{n+1}$ there. The theorem above does not require continuity of the last derivative. If $x=a$, there is no point strictly between the inputs: simply use $R_{n,a}(a)=0$. For distinct inputs and $n=0$, the displayed formula reduces to the [[拉格朗日中值定理|mean value theorem]], $f(x)-f(a)=f'(\xi)(x-a)$.
<!-- bilingual-en:end -->

一个更容易检查的充分条件是：$f$ 在包含整段的开区间上，直到 $n+1$ 阶的导数都连续，记为 $f\in C^{n+1}$。上面较弱的定理并不要求最后一阶导数连续。若 $x=a$，两点之间没有严格内点，直接使用 $R_{n,a}(a)=0$ 即可。两点不同且 $n=0$ 时，公式正好是[[拉格朗日中值定理]]：$f(x)-f(a)=f'(\xi)(x-a)$。

<!-- bilingual-en:start -->
## Why an intermediate derivative gives the exact error
<!-- bilingual-en:end -->
*为什么中间点的导数能给出精确误差*

<!-- bilingual-en:start -->
Fix $x\ne a$ and choose the number $K=R_{n,a}(x)/(x-a)^{n+1}$. Subtract a correction designed to make the error vanish at the target:
<!-- bilingual-en:end -->

固定 $x\ne a$，令 $K=R_{n,a}(x)/(x-a)^{n+1}$。减去一项专门设计的修正，使误差在目标点也归零：

$$G(t)=f(t)-P_{n,a}(t)-K(t-a)^{n+1}.$$

<!-- bilingual-en:start -->
The definition of $K$ gives $G(x)=0$. Derivative matching at $a$ gives $G(a)=G'(a)=\cdots=G^{(n)}(a)=0$, because the added power and its first $n$ derivatives vanish there. The [[罗尔定理|Rolle theorem]] says that a continuous function with equal endpoint values and an interior derivative has a zero derivative between them.
<!-- bilingual-en:end -->

由 $K$ 的定义，有 $G(x)=0$。由中心处的导数匹配，有 $G(a)=G'(a)=\cdots=G^{(n)}(a)=0$，因为新增幂项及其前 $n$ 阶导数在那里都为零。[[罗尔定理]]说明：函数在闭段连续、内部可导且两端值相等，就在两端之间有一个零导数点。

<!-- bilingual-en:start -->
Apply it first to $G$ between $a$ and $x$ to obtain a zero of $G'$. If $n\ge1$, apply it to $G'$ between that zero and $a$, where $G'(a)=0$, to obtain a zero of $G''$. Continue in this way. After $n+1$ applications, there is a $\xi$ strictly between the original inputs with $G^{(n+1)}(\xi)=0$. Each application has the required endpoint continuity and interior derivative by the stated hypotheses. Since the polynomial has no derivative of order $n+1$ remaining,
<!-- bilingual-en:end -->

先在 $a,x$ 之间对 $G$ 应用罗尔定理，得到 $G'$ 的一个零点。当 $n\ge1$ 时，再在这个零点与 $a$ 之间对 $G'$ 应用罗尔定理，因为 $G'(a)=0$，便得到 $G''$ 的零点。如此继续，应用 $n+1$ 次后，在原来两点之间得到 $G^{(n+1)}(\xi)=0$。每一步所需的端点连续性和内部可导性，都由假设保证。由于多项式求到 $n+1$ 阶已经为零，

$$0=G^{(n+1)}(\xi)=f^{(n+1)}(\xi)-(n+1)!K.$$

<!-- bilingual-en:start -->
Solving for $K$ and substituting its definition proves the remainder formula. This argument works on either side of the centre; the power $(x-a)^{n+1}$ retains its sign.
<!-- bilingual-en:end -->

解出 $K$，再代回它的定义，就得到了余项公式。目标在中心左侧或右侧，论证都成立；幂 $(x-a)^{n+1}$ 保留相应的符号。

<!-- bilingual-en:start -->
## Turn the unknown point into a usable bound
<!-- bilingual-en:end -->
*把未知位置换成可用的上界*

<!-- bilingual-en:start -->
If a finite number $M\ge0$ satisfies $|f^{(n+1)}(t)|\le M$ at every point strictly between $a$ and $x$, it also controls the unknown $\xi$. Taking absolute values gives
<!-- bilingual-en:end -->

如果存在有限数 $M\ge0$，使严格位于 $a,x$ 之间的每一点都满足 $|f^{(n+1)}(t)|\le M$，这个上界也就控制了未知的 $\xi$。取绝对值得到

$$|R_{n,a}(x)|\le\frac{M|x-a|^{n+1}}{(n+1)!}.$$

<!-- bilingual-en:start -->
When the last derivative is continuous on the whole closed segment, it is bounded there by the [[极值定理|extreme value theorem]]. To use the displayed estimate numerically, we still need an actual suitable value of $M$. A small derivative at $a$ alone does not control values between $a$ and $x$.
<!-- bilingual-en:end -->

若最后一阶导数在整个闭段连续，[[极值定理]]保证它在那里有界。但要把公式用于数值估计，仍需给出一个实际可用的 $M$。仅知道 $a$ 处的导数很小，不能控制 $a$ 与 $x$ 之间的值。

> [!example] Certify the exponential estimate · 为指数近似给出保证
> <!-- bilingual-en:start -->
> For $f(x)=e^x$, $a=0$ and $n=2$, the third derivative is $e^t$. If $|x|\le0.1$, the connecting segment lies in $[-0.1,0.1]$, where $e^t\le e^{0.1}$. Therefore
> <!-- bilingual-en:end -->
>
> 对 $f(x)=e^x$，取 $a=0,n=2$，三阶导数为 $e^t$。若 $|x|\le0.1$，连接两点的线段都在 $[-0.1,0.1]$ 内，其中 $e^t\le e^{0.1}$。因此
>
> $$\left|e^x-\left(1+x+\frac{x^2}{2}\right)\right|
> \le e^{0.1}\frac{|x|^3}{6}
> \le\frac{e^{0.1}}{6000}<1.85\times10^{-4}.$$
>
> <!-- bilingual-en:start -->
> In particular, $1.105$ approximates $e^{0.1}$ within that absolute error. The zero-input case holds directly with error zero. The bound is uniform over the stated range because the same $M=e^{0.1}$ works for every connecting segment.
> <!-- bilingual-en:end -->
>
> 特别地，$1.105$ 对 $e^{0.1}$ 的绝对误差不超过这个界。输入为零时，直接知道误差为零。这个保证同时适用于所述范围内的全部输入，因为同一个 $M=e^{0.1}$ 控制了每一条连接线段。

<!-- bilingual-en:start -->
A polynomial example makes the equality visible. For $f(x)=x^3+2x+1$ expanded at $a=1$, the quadratic polynomial is $P_{2,1}(x)=4+5(x-1)+3(x-1)^2$. Direct expansion gives $R_{2,1}(x)=(x-1)^3$. Since $f^{(3)}=6$ everywhere, the formula gives exactly $6(x-1)^3/3!$, whichever intermediate point is used. At $x=3$, the true value is $34$, the estimate is $26$, and the remainder is $8$.
<!-- bilingual-en:end -->

一个多项式例子能直接看见等号。对 $f(x)=x^3+2x+1$ 在 $a=1$ 展开，二次多项式为 $P_{2,1}(x)=4+5(x-1)+3(x-1)^2$。直接展开得到 $R_{2,1}(x)=(x-1)^3$。由于处处 $f^{(3)}=6$，无论中间点选在哪里，公式都给出精确的 $6(x-1)^3/3!$。在 $x=3$ 处，真值为 $34$，估计为 $26$，余项为 $8$。

<!-- bilingual-en:start -->
## A finite-order estimate is not an infinite-series guarantee
<!-- bilingual-en:end -->
*有限阶估计不等于无穷级数保证*

<!-- bilingual-en:start -->
Increasing the order does not automatically reduce the error at a fixed target. For $f(x)=x/(1-x)$ about zero, the first two Taylor polynomials are $x$ and $x+x^2$. At $x=-2$, the function value is $-2/3$, while the estimates are $-2$ and $2$: the absolute errors grow from $4/3$ to $8/3$. The function is smooth on the whole connecting segment; smoothness alone does not make a distant target suitable for these truncations.
<!-- bilingual-en:end -->

提高阶数，不会自动减小某个固定目标处的误差。对 $f(x)=x/(1-x)$ 在零点展开，前两个 Taylor 多项式是 $x$ 与 $x+x^2$。在 $x=-2$ 处，真值为 $-2/3$，估计值却分别是 $-2$ 与 $2$，绝对误差从 $4/3$ 增至 $8/3$。函数在整个连接段上光滑；光滑本身不能保证较远目标适合这些截断。

<!-- bilingual-en:start -->
To conclude that a [[Taylor级数等于函数|Taylor series equals its function]] at a fixed $x$, the remainders must tend to zero as $n\to\infty$. The derivative bounds generally change with the order: write them as $M_n$. A sufficient estimate is $M_n|x-a|^{n+1}/(n+1)!\to0$, with all its factors considered together. Seeing a factorial in the denominator, or knowing that every derivative exists, is not enough.
<!-- bilingual-en:end -->

要断言固定 $x$ 处的[[Taylor级数等于函数]]，必须证明 $n\to\infty$ 时余项趋于零。导数上界一般会随阶数改变，应写作 $M_n$。充分的估计是 $M_n|x-a|^{n+1}/(n+1)!\to0$，需要把所有因素一起考察。只看见分母有阶乘，或只知道每阶导数都存在，都还不够。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [Lebl, Basic Analysis I, Theorem 4.3.2 and proof, PDF pp.155–157](https://ocw.mit.edu/courses/18-100a-real-analysis-fall-2020/mit18_100af20_basic_analysis.pdf#page=155): endpoint continuity, the interior next derivative, the repeated-zero proof and the limits of increasing degree.
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise034_Solutions.pdf#page=1|MIT 18.01SC Exercise 34 solutions, PDF pp.1–2]]: the mean-value connection, the cubic test case and derivative-bound interpretation. The course solution illustrates the formula rather than proving the general theorem.
- [[01_Math/01_calculus/05_Infinite_Series_and_Improper_Integrals#Taylor 多项式与余项|Permanent course note, Taylor polynomials and remainder]]: the whole-segment bound, the equal-input case and the exponential estimate. Numerical values and both polynomial comparisons are checked directly here.
<!-- bilingual-en:end -->

- Lebl《Basic Analysis I》定理4.3.2及证明，PDF155–157页：端点连续性、内部下一阶导数、重复零点证明，以及提高次数的局限。
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise034_Solutions.pdf#page=1|MIT 18.01SC Exercise 34 题解，PDF1–2页]]：与中值定理的联系、立方函数验算及导数上界的用途。课程题解展示公式，并未证明一般定理。
- [[01_Math/01_calculus/05_Infinite_Series_and_Improper_Integrals#Taylor 多项式与余项|永久课程笔记“Taylor 多项式与余项”]]：整段上界、输入相同的情形与指数误差估计。数值及两个多项式比较均在此直接核算。

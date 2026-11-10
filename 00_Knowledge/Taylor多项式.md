---
aliases:
  - Taylor polynomial
  - Maclaurin polynomial
student_os: knowledge-atom
atom_id: CALC-APP-003
atom_set: derivative-applications
atom_type: definition
status: source-checked
part_of:
  - "[[导数的应用.canvas]]"
  - "[[无穷级数与幂级数.canvas]]"
requires:
  - "[[高阶导数]]"
related:
  - "[[线性近似]]"
  - "[[二次近似]]"
leads_to:
  - "[[Taylor余项]]"
  - "[[Taylor级数]]"
---
<!-- bilingual-en:start -->
# Taylor polynomial
<!-- bilingual-en:end -->
*Taylor 多项式*

<!-- bilingual-en:start -->
A **Taylor polynomial** builds a polynomial model of a function from information at one chosen input. It first matches the function's value, then its slope, then how the slope changes, and continues for as many derivative orders as we choose. A polynomial uses only a finite number of powers, so its values are usually easier to calculate than those of the original function.
<!-- bilingual-en:end -->

**Taylor 多项式**利用函数在一个选定输入处的信息，构造一个多项式模型。先匹配函数值，再匹配斜率、斜率的变化率，继续匹配到选定的导数阶数。多项式只含有限个幂次，因此通常比原函数更容易计算。

<!-- bilingual-en:start -->
For example, suppose we want to model $f(x)=e^x$ near $0$. The [[指数函数求导|exponential derivative]] is again $e^x$, so $f(0)=f'(0)=f''(0)=1$. Start with $P(x)=c_0+c_1x+c_2x^2$, where the three $c$'s are unknown constants. Matching the value at zero gives $c_0=1$; matching the first derivative gives $c_1=1$; matching the second derivative gives $2c_2=1$. Therefore
<!-- bilingual-en:end -->

例如，要在 $0$ 附近给 $f(x)=e^x$ 建立模型。由[[指数函数求导]]，它求导后仍为 $e^x$，所以 $f(0)=f'(0)=f''(0)=1$。先设 $P(x)=c_0+c_1x+c_2x^2$，三个 $c$ 是待定常数。匹配零点函数值，得到 $c_0=1$；匹配一阶导数，得到 $c_1=1$；匹配二阶导数，得到 $2c_2=1$。于是

$$P(x)=1+x+\frac{x^2}{2},\qquad P(0.1)=1.105.$$

<!-- bilingual-en:start -->
The factor $1/2$ is forced by differentiating $x^2$ twice. The calculation gives an estimate for $e^{0.1}$; to certify its error, we can use the separate [[Taylor余项|Taylor remainder theorem]]. Matching information at the centre is not itself a numerical error bound at a specified distance.
<!-- bilingual-en:end -->

这里的 $1/2$ 来自对 $x^2$ 求两次导所产生的因子 $2$。算式给出了 $e^{0.1}$ 的估计值；要保证误差有多小，可以使用独立的[[Taylor余项]]定理。在中心匹配信息，本身不是对指定距离处的数值误差作保证。

<!-- bilingual-en:start -->
## Why factorials appear
<!-- bilingual-en:end -->
*为什么系数要除以阶乘*

<!-- bilingual-en:start -->
Let $a$ be the **expansion centre**, and write $h=x-a$ for the signed distance from it. The notation $f^{(k)}$ means the [[高阶导数|$k$th derivative]], obtained by differentiating $k$ times; $f^{(0)}=f$ means no differentiation. Choose a finite integer $n\ge0$. For $n\ge1$, assume the ordinary derivatives through order $n$ at $a$ exist, with the lower derivatives defined near $a$ as required by successive differentiation. For $n=0$, only $f(a)$ is needed to form the constant polynomial.
<!-- bilingual-en:end -->

设 $a$ 是**展开中心**，$h=x-a$ 是离开中心的带符号距离。$f^{(k)}$ 表示连续求导 $k$ 次得到的[[高阶导数|$k$ 阶导数]]；$f^{(0)}=f$ 表示不求导。选定有限整数 $n\ge0$。当 $n\ge1$ 时，假设 $a$ 处直到 $n$ 阶的普通导数存在，并且低阶导数在附近有定义，以便逐次求导。若 $n=0$，构造常数多项式只需知道 $f(a)$。

<!-- bilingual-en:start -->
Write a candidate of degree at most $n$ as $P(a+h)=c_0+c_1h+\cdots+c_nh^n$. After $k$ differentiations, every term of lower degree has disappeared. Every term of higher degree still contains a power of $h$, so it vanishes at $h=0$. The $k$th term alone contributes $k!c_k$, where $k!=1\cdot2\cdots k$ for $k\ge1$, and $0!=1$. Thus matching the $k$th derivative requires $c_k=f^{(k)}(a)/k!$.
<!-- bilingual-en:end -->

把次数不超过 $n$ 的候选式写成 $P(a+h)=c_0+c_1h+\cdots+c_nh^n$。求 $k$ 次导后，所有低于 $k$ 次的项已经消失；所有高于 $k$ 次的项仍含 $h$ 的正次幂，在 $h=0$ 时为零。只有第 $k$ 项留下 $k!c_k$。这里 $k!=1\cdot2\cdots k$（$k\ge1$），并规定 $0!=1$。因此，匹配第 $k$ 阶导数就必须令 $c_k=f^{(k)}(a)/k!$。

> [!important] One polynomial matches all the chosen derivatives · 一个多项式同时匹配所选各阶
> <!-- bilingual-en:start -->
> The **Taylor polynomial of order $n$ at $a$** is
> <!-- bilingual-en:end -->
>
> **在 $a$ 处的 $n$ 阶 Taylor 多项式**为
>
> $$P_{n,a}(x)=\sum_{k=0}^{n}\frac{f^{(k)}(a)}{k!}(x-a)^k.$$
>
> <!-- bilingual-en:start -->
> The summation adds the terms for $k=0,1,\ldots,n$. This polynomial satisfies $P_{n,a}^{(k)}(a)=f^{(k)}(a)$ for each such $k$. The coefficient argument determines every $c_k$, so it also proves uniqueness among polynomials of degree at most $n$.
> <!-- bilingual-en:end -->
>
> 求和号表示把 $k=0,1,\ldots,n$ 的各项相加。这个多项式对每个这样的 $k$ 都满足 $P_{n,a}^{(k)}(a)=f^{(k)}(a)$。前面的系数论证确定了全部 $c_k$，因此也证明：在次数不超过 $n$ 的多项式中，它是唯一的。

<!-- bilingual-en:start -->
The order $n$ is a limit on the degree, not a promise that the highest coefficient is nonzero. For $f(x)=x$ at $a=0$, the order-three Taylor polynomial is still $x$. At the centre, $P_{n,a}(a)=f(a)$ exactly. For $n=0$, the polynomial is $f(a)$; for $n=1$ it is the [[线性近似|linear approximation]]; for $n=2$ it is the [[二次近似|quadratic approximation]]. The powers of $x-a$ keep the centre visible. When $a=0$, the same object is also called a **Maclaurin polynomial**.
<!-- bilingual-en:end -->

阶数 $n$ 限制的是次数上界，不是承诺最高次系数一定非零。例如 $f(x)=x$ 在 $a=0$ 的三阶 Taylor 多项式仍然是 $x$。在中心处总有精确等式 $P_{n,a}(a)=f(a)$。$n=0$ 时，它是常数 $f(a)$；$n=1$ 时是[[线性近似]]；$n=2$ 时是[[二次近似]]。用 $x-a$ 的幂，能明确保留展开中心。当 $a=0$ 时，也称为 **Maclaurin 多项式**。

> [!example] Add the cubic information · 再加入三阶信息
> <!-- bilingual-en:start -->
> For $e^x$ at zero, the third derivative also equals $1$. Since differentiating $cx^3$ three times gives $6c$, the extra coefficient is $1/6$:
> <!-- bilingual-en:end -->
>
> $e^x$ 在零点的三阶导数也为 $1$。对 $cx^3$ 连续求三次导得到 $6c$，所以新增系数是 $1/6$：
>
> $$P_{3,0}(x)=1+x+\frac{x^2}{2}+\frac{x^3}{6}.$$
>
> <!-- bilingual-en:start -->
> This is a finite calculation with four terms. It does not, by itself, prove an infinite-series identity or give the error at a chosen $x$.
> <!-- bilingual-en:end -->
>
> 这只是含四项的有限计算。它本身没有证明无穷级数恒等式，也没有给出指定 $x$ 处的误差大小。

<!-- bilingual-en:start -->
## What matching at one point leaves open
<!-- bilingual-en:end -->
*一点匹配之后，还有什么没有确定*

<!-- bilingual-en:start -->
Functions can share these derivatives yet differ away from the centre. For instance, $P_{3,0}(x)+Cx^4$ has the same first three derivatives and value at zero as $P_{3,0}$, for every real $C$. At $x=0.1$ their difference is $10^{-4}C$, which has no bound independent of $C$. The omitted information matters for a numerical guarantee.
<!-- bilingual-en:end -->

不同函数可以具有这些相同的导数，却在离开中心后不同。例如，对任意实数 $C$，$P_{3,0}(x)+Cx^4$ 与 $P_{3,0}$ 在零点的函数值和前三阶导数都相同；但在 $x=0.1$ 处，两者相差 $10^{-4}C$，不存在与 $C$ 无关的统一上界。数值保证需要被省略部分的信息。

<!-- bilingual-en:start -->
Letting the order run without end creates a [[Taylor级数|Taylor series]], an infinite object with a separate convergence question. Even having derivatives of every order does not ensure that this series represents the function nearby; the [[光滑不代表解析|smooth nonanalytic counterexample]] explains that failure. Keep three questions distinct: which polynomial matches the derivatives, how large its finite-order [[Taylor余项|remainder]] is, and whether the infinite series equals the function.
<!-- bilingual-en:end -->

让阶数无限延伸，会得到[[Taylor级数]]，这是一个需要另行讨论收敛性的无限对象。即使每一阶导数都存在，也不能保证级数在附近表示原函数；[[光滑不代表解析]]给出了失败的反例。因此要分别问：哪个多项式匹配这些导数？指定有限阶的[[Taylor余项]]有多大？无穷级数是否等于原函数？

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [Lebl, Basic Analysis I, §4.3, PDF pp.155–157, Definition 4.3.1](https://ocw.mit.edu/courses/18-100a-real-analysis-fall-2020/mit18_100af20_basic_analysis.pdf#page=155): the finite polynomial, derivative matching and distinction from an infinite-series representation.
- [OpenStax Calculus Volume 2, §6.3, Taylor/Maclaurin polynomial definition](https://openstax.org/books/calculus-volume-2/pages/6-3-taylor-and-maclaurin-series): the zero-centre name and the finite exponential examples.
- [[Ses98b_Lecture_Notes.pdf#page=1|MIT 18.01SC Session 98b, PDF p.1]]: extraction of coefficients and factorials. The finite coefficient calculation is used here; equality with an infinite Taylor series requires additional justification.
- [[01_Math/01_calculus/02_Applications_of_Differentiation#26d：一般 $n$ 次匹配的来源|Permanent course note, §26d coefficient matching]]: the course's finite matching argument. The displayed polynomial examples are verified by direct differentiation.
<!-- bilingual-en:end -->

- Lebl《Basic Analysis I》第4.3节，PDF155–157页，定义4.3.1：有限多项式、导数匹配，以及它与无穷级数表示的区别。
- [OpenStax Calculus Volume 2 第6.3节，Taylor/Maclaurin 多项式定义](https://openstax.org/books/calculus-volume-2/pages/6-3-taylor-and-maclaurin-series)：零中心的名称及有限指数多项式例子。
- [[Ses98b_Lecture_Notes.pdf#page=1|MIT 18.01SC Session 98b，PDF1页]]：系数读取及阶乘来源。这里采用有限系数计算；函数等于无穷 Taylor 级数仍需额外论证。
- [[01_Math/01_calculus/02_Applications_of_Differentiation#26d：一般 $n$ 次匹配的来源|永久课程笔记“26d：一般 n 次匹配的来源”]]：课程的有限阶匹配论证。所列多项式例子均由直接求导核验。

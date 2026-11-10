---
aliases:
  - Taylor series
  - Maclaurin series
student_os: knowledge-atom
atom_id: CALC-SER-023
atom_type: definition
status: source-checked
part_of:
  - "[[无穷级数与幂级数.canvas]]"
requires:
  - "[[幂级数]]"
  - "[[Taylor多项式]]"
leads_to:
  - "[[Taylor级数等于函数]]"
  - "[[光滑不代表解析]]"
---

<!-- bilingual-en:start -->
# Taylor series
<!-- bilingual-en:end -->
*Taylor 级数*

<!-- bilingual-en:start -->
A Taylor series packages a function's value, slope and all its higher derivatives at one chosen point into an infinite list of polynomial terms. Each extra term matches one more derivative at that point. The list is a candidate for representing the function; whether its partial sums actually approach the function is a further question.
<!-- bilingual-en:end -->

Taylor 级数把函数在某个选定点的函数值、斜率以及所有更高阶导数，组织成无限多个多项式项。每增加一项，就能在该点多匹配一阶导数。这份列表给出候选表示；部分和是否真的趋近原函数，还要另外判断。

<!-- bilingual-en:start -->
Start with $f(x)=e^x$ at the centre $0$. The [[指数函数求导|exponential derivative rule]] says every derivative is again $e^x$, so the value of each at zero is $1$. The first few [[Taylor多项式|Taylor polynomials]] are
<!-- bilingual-en:end -->

先看 $f(x)=e^x$，中心取 $0$。[[指数函数求导]]说明每阶导数仍是 $e^x$，因此在零点都取值 $1$。前几个 [[Taylor多项式]]为

$$P_{0,0}(x)=1,\quad P_{1,0}(x)=1+x,\quad
P_{2,0}(x)=1+x+\frac{x^2}{2},\quad
P_{3,0}(x)=1+x+\frac{x^2}{2}+\frac{x^3}{6}.$$

<!-- bilingual-en:start -->
The subscript $N,0$ records the order $N$ and centre $0$. Why divide the quadratic term by $2$? Its second derivative is then $1$, matching $f''(0)$. The third derivative of $x^3/6$ is also $1$. Continuing gives the candidate $1+x+x^2/2!+x^3/3!+\cdots$. The derivatives at zero are all $1$; the coefficients are $1/n!$.
<!-- bilingual-en:end -->

下标 $N,0$ 记录阶数 $N$ 与中心 $0$。二次项为什么除以 $2$？这样它的二阶导数才是 $1$，与 $f''(0)$ 匹配。$x^3/6$ 的三阶导数也正好是 $1$。继续下去就得到候选式 $1+x+x^2/2!+x^3/3!+\cdots$。零点处各阶导数都是 $1$，而各项系数是 $1/n!$。

<!-- bilingual-en:start -->
## The coefficients and the centre
<!-- bilingual-en:end -->
*系数与展开中心*

<!-- bilingual-en:start -->
Let a real function $f$ be defined near a real point $a$, with finite derivatives of every order at $a$. The notation [[高阶导数|$f^{(n)}$]] means differentiating $n$ times, not raising $f$ to a power; set $f^{(0)}=f$. For a positive integer $n$, $n!=1\cdot2\cdots n$, and $0!=1$. The **Taylor series of $f$ at $a$** is the [[幂级数|power series]]
<!-- bilingual-en:end -->

设实值函数 $f$ 在实数点 $a$ 附近有定义，并且在 $a$ 的每阶导数都是有限实数。[[高阶导数|$f^{(n)}$]]表示连续求导 $n$ 次，不是 $f$ 的 $n$ 次方；约定 $f^{(0)}=f$。正整数 $n$ 的阶乘为 $n!=1\cdot2\cdots n$，且 $0!=1$。**$f$ 在 $a$ 的 Taylor 级数**是[[幂级数]]

$$
\sum_{n=0}^{\infty}\frac{f^{(n)}(a)}{n!}(x-a)^n
=f(a)+f'(a)(x-a)+\frac{f''(a)}{2!}(x-a)^2+\cdots.
$$

<!-- bilingual-en:start -->
The input $x$ varies, while the centre $a$ and coefficients $c_n=f^{(n)}(a)/n!$ stay fixed. The term with $n=0$ is the constant $f(a)$, also when $x=a$; the zeroth-power factor is interpreted as $1$. The special case $a=0$ is called the **Maclaurin series**. A different centre uses derivatives evaluated at that new centre: for $e^x$ at $a=2$, the candidate is $e^2\sum_{n=0}^{\infty}(x-2)^n/n!$.
<!-- bilingual-en:end -->

变化的是输入 $x$，中心 $a$ 和系数 $c_n=f^{(n)}(a)/n!$ 保持固定。$n=0$ 的项就是常数 $f(a)$，包括 $x=a$ 的情形；这里零次幂因子按 $1$ 理解。中心为 $0$ 的特殊情形称为 **Maclaurin 级数**。换中心时，导数也要在新中心取值：例如 $e^x$ 在 $a=2$ 的候选式为 $e^2\sum_{n=0}^{\infty}(x-2)^n/n!$。

<!-- bilingual-en:start -->
For a nonnegative integer $N$, stop after the term with index $N$. The resulting partial sum is exactly the Taylor polynomial
<!-- bilingual-en:end -->

对非负整数 $N$，在第 $N$ 项后停止，得到的部分和恰好是 Taylor 多项式

$$P_{N,a}(x)=\sum_{n=0}^{N}\frac{f^{(n)}(a)}{n!}(x-a)^n.$$

<!-- bilingual-en:start -->
There are $N+1$ indexed terms, though some can be zero, so the degree is at most $N$. By the definition of an [[无穷级数|infinite series]], its sum at a fixed $x$ exists only when these finite partial sums have a finite limit. At $x=a$, every partial sum is $f(a)$, so equality there is automatic. Knowing derivatives only through a finite order gives a Taylor polynomial, not the full infinite series.
<!-- bilingual-en:end -->

下标从 $0$ 到 $N$，共有 $N+1$ 项，但其中有些可以为零，所以次数至多为 $N$。按[[无穷级数]]的定义，只有这些有限部分和在固定 $x$ 处具有有限极限，才能说级数在那里有和。中心 $x=a$ 处每个部分和都等于 $f(a)$，所以中心处相等是自动成立的。若只知道有限阶导数，就只能据此构造 Taylor 多项式，不能确定整个无穷级数。

<!-- bilingual-en:start -->
## Why a local power-series representation has no other coefficients
<!-- bilingual-en:end -->
*局部幂级数表示为什么只能用这些系数*

<!-- bilingual-en:start -->
Suppose we already know $f(x)=\sum_{n=0}^{\infty}c_n(x-a)^n$ on some open interval around $a$. Its radius is positive, so [[幂级数逐项求导|termwise differentiation]] is valid in a smaller open interval around the centre and can be repeated any finite number of times. After $k$ differentiations, terms of degree below $k$ vanish. Terms of degree above $k$ still contain a positive power of $x-a$ and vanish when evaluated at $a$. Only the degree-$k$ term remains:
<!-- bilingual-en:end -->

假设已经知道，在 $a$ 周围某个开区间上有 $f(x)=\sum_{n=0}^{\infty}c_n(x-a)^n$。此时半径为正，因而可以在中心附近较小的开区间内[[幂级数逐项求导]]，并重复任意有限次。求导 $k$ 次后，次数低于 $k$ 的项已消失；次数高于 $k$ 的项仍含 $x-a$ 的正次幂，在 $a$ 处取值为零。只留下次数恰好为 $k$ 的项：

$$f^{(k)}(a)=k!c_k,\qquad c_k=\frac{f^{(k)}(a)}{k!}.$$

<!-- bilingual-en:start -->
Thus a power-series representation on a neighbourhood, if it exists, is uniquely forced by the derivatives. This argument uses equality on an interval, not merely at the single point $a$. For example, $1$ and $1+x$ both equal $1$ at zero but have different coefficients.
<!-- bilingual-en:end -->

因此，在邻域内成立的幂级数表示若存在，其系数就由导数唯一确定。这个论证使用的是整个区间上的相等，不是仅在 $a$ 这一个点相等。例如 $1$ 与 $1+x$ 在零点都取值 $1$，系数却不同。

> [!warning] Coefficients do not certify representation / 系数没有自动保证表示成立
> <!-- bilingual-en:start -->
> The definition supplies the terms. The [[幂级数收敛半径|radius of convergence]] asks where their partial sums converge. [[Taylor级数等于函数|Vanishing Taylor remainders]] decide whether the limit equals the original function. Even a zero series can converge everywhere while missing a smooth function away from its centre; see [[光滑不代表解析|the smooth nonanalytic example]].
> <!-- bilingual-en:end -->
>
> 定义给出各项；[[幂级数收敛半径]]回答部分和在哪里收敛；[[Taylor级数等于函数|Taylor 余项趋零]]决定其极限是否等于原函数。即使是处处收敛的零级数，也可能在中心之外无法表示原来的光滑函数，见[[光滑不代表解析]]。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [[Ses98b_Lecture_Notes.pdf#page=1|MIT 18.01SC Session 98b, p.1]] and [[Ses98c_Lecture_Notes.pdf#page=1|Session 98c, p.1]]: extracting coefficients by derivatives and the exponential coefficients.
- [OpenStax Calculus2 §6.3, Definition and Theorem6.6](https://openstax.org/books/calculus-volume-2/pages/6-3-taylor-and-maclaurin-series): general centre, Maclaurin terminology, partial sums and uniqueness of an existing local representation.
- [Lebl, Basic Analysis I v6.3, §4.3, PDF171–173 and Corollary6.2.13, PDF242](https://www.jirka.org/ra/realanal.pdf#page=171): finite Taylor polynomials, the separate infinite-series question, and the termwise differentiation used in the uniqueness argument.
<!-- bilingual-en:end -->

- MIT 18.01SC Session98b、98c，各第1页：由导数取出系数及指数函数系数。
- OpenStax §6.3 定义、定理6.6：一般中心、Maclaurin 名称、部分和及已有局部表示的唯一性。
- Lebl《Basic Analysis I》v6.3，第4.3节，PDF171–173页及推论6.2.13、PDF242页：有限 Taylor 多项式、独立的无穷级数问题，以及唯一性论证使用的逐项求导。

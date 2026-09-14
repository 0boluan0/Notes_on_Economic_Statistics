---
aliases:
  - "Taylor级数是以函数在展开中心的各阶导数除以阶乘为系数的幂级数"
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

# Taylor级数是以函数在展开中心的各阶导数除以阶乘为系数的幂级数
<!-- bilingual-en:start -->
*A Taylor series has coefficients given by derivatives at its centre divided by factorials*
<!-- bilingual-en:end -->

若 $f$ 在 $a$ 的各阶导数都存在，它在 $a$ 的 Taylor 级数定义为
$$
\sum_{n=0}^{\infty}\frac{f^{(n)}(a)}{n!}(x-a)^n.
$$
中心 $a=0$ 时也称 Maclaurin 级数。它的第 $N$ 个部分和就是已有的 [[Taylor多项式|$N$ 次 Taylor 多项式]] $P_{N,a}(x)$。
<!-- bilingual-en:start -->
When all derivatives at $a$ exist, the displayed series is the Taylor series of $f$ at $a$. A series centred at zero is also called a Maclaurin series. Its $N$th partial sum is the [[Taylor多项式|Taylor polynomial]] $P_{N,a}(x)$.
<!-- bilingual-en:end -->

系数中的阶乘来自导数匹配：对 $c_n(x-a)^n$ 求 $n$ 次导并在 $a$ 取值，留下 $n!c_n$。例如 $f(x)=e^x$ 在 $0$ 的所有阶导数都是 $1$，所以候选级数为 $\sum x^n/n!$。
<!-- bilingual-en:start -->
The factorial comes from matching derivatives: differentiating $c_n(x-a)^n$ exactly $n$ times and evaluating at $a$ gives $n!c_n$. For $e^x$ at zero, every derivative equals one, giving the candidate series $\sum x^n/n!$.
<!-- bilingual-en:end -->

这个定义给出系数，但还没有证明级数在某个 $x$ 收敛，更没有证明其和等于 $f(x)$。后一个问题由 [[Taylor级数等于函数|余项趋零条件]] 解决；[[光滑不代表解析]] 提供失败的具体例子。
<!-- bilingual-en:start -->
The definition determines the coefficients. It does not yet establish convergence at a given $x$ or equality with $f(x)$. Equality requires the [[Taylor级数等于函数|remainder to tend to zero]]; a [[光滑不代表解析|smooth nonanalytic function]] shows why this matters.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [[Ses98b_Lecture_Notes.pdf#page=1|MIT 18.01SC Session 98b，第 1 页]]：核对系数的逐阶导数推导；[[Ses98c_Lecture_Notes.pdf#page=1|Session 98c，第 1 页]] 核对指数函数系数。
- [OpenStax Calculus Volume 2 §6.3，Definition](https://openstax.org/books/calculus-volume-2/pages/6-3-taylor-and-maclaurin-series)：核对一般中心、Maclaurin 名称及部分和。MIT 98b 开头的等号须结合余项条件理解。
<!-- bilingual-en:start -->
- MIT Sessions 98b and 98c support coefficient extraction and the exponential example. OpenStax §6.3 verifies the definition and partial-sum interpretation; equality with the function requires an additional remainder argument.
<!-- bilingual-en:end -->

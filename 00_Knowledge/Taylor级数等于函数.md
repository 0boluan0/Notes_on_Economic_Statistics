---
aliases:
  - "Taylor级数在一点等于原函数当且仅当该点的Taylor余项随阶数趋于零"
  - Convergence of a Taylor series to its function
student_os: knowledge-atom
atom_id: CALC-SER-024
atom_type: theorem
status: source-checked
part_of:
  - "[[无穷级数与幂级数.canvas]]"
requires:
  - "[[Taylor级数]]"
  - "[[Taylor余项]]"
related:
  - "[[光滑不代表解析]]"
leads_to:
  - "[[指数函数幂级数]]"
  - "[[正弦函数幂级数]]"
---

# Taylor级数在一点等于原函数当且仅当该点的Taylor余项随阶数趋于零
<!-- bilingual-en:start -->
*A Taylor series equals its function at a point exactly when the remainders there tend to zero*
<!-- bilingual-en:end -->

设 $f$ 在包含中心 $a$ 的区间内无限可微，并令 $R_{N,a}(x)=f(x)-P_{N,a}(x)$。对固定的 $x$，有
$$
f(x)=\sum_{n=0}^{\infty}\frac{f^{(n)}(a)}{n!}(x-a)^n
\quad\Longleftrightarrow\quad
\lim_{N\to\infty}R_{N,a}(x)=0.
$$
因为 Taylor 级数的部分和正是 $P_{N,a}$，这就是部分和收敛到目标函数的精确定义。
<!-- bilingual-en:start -->
For a smooth function on an interval containing $a$, define $R_{N,a}=f-P_{N,a}$. At a fixed $x$, the displayed equivalence follows because the Taylor polynomials are exactly the series' partial sums.
<!-- bilingual-en:end -->

实用证明可复用 [[Taylor余项]]：若连接 $a$ 与 $x$ 的整段上有 $|f^{(N+1)}(t)|\le M_N$，且
$$
\frac{M_N|x-a|^{N+1}}{(N+1)!}\longrightarrow0,
$$
就能推出等式。$M_N$ 可以随阶数改变，必须一起带入极限；分母有阶乘本身并不足够。要证明整个区间上逐点相等，须让论证覆盖区间里的每个固定 $x$，无需在这里额外要求一致收敛。
<!-- bilingual-en:start -->
A [[Taylor余项|Taylor remainder bound]] proves equality if $M_N|x-a|^{N+1}/(N+1)!$ tends to zero, where $M_N$ bounds the relevant derivative throughout the connecting interval. The bound may depend on $N$, so the factorial alone proves nothing. Equality throughout an interval requires the argument at every fixed point.
<!-- bilingual-en:end -->

“Taylor 级数自身收敛”和“它收敛到原函数”是两个判断；[[光滑不代表解析]] 中零级数处处收敛，却在中心之外不等于构造它的函数。
<!-- bilingual-en:start -->
Convergence of the series and convergence to its source function are separate questions. The [[光滑不代表解析|smooth nonanalytic example]] has an everywhere-convergent zero Taylor series that misses the function away from its centre.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [OpenStax Calculus Volume 2 §6.3，Theorem 6.8](https://openstax.org/books/calculus-volume-2/pages/6-3-taylor-and-maclaurin-series)：核对余项趋零的充要条件。
- [[Taylor余项]]：复用已分立的 Lagrange 余项，不另建重复定理；本卡核对的是上界随 $N$ 的极限及区间覆盖要求。
<!-- bilingual-en:start -->
- OpenStax Theorem 6.8 verifies the equivalence. The existing remainder atom supplies the finite-degree bound; this atom checks its limiting use and the required interval coverage.
<!-- bilingual-en:end -->

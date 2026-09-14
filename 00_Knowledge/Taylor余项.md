---
aliases:
  - "$C^{n+1}$ 函数的 Taylor 误差可由中间点的 $(n+1)$ 阶导数写成 Lagrange 余项并用区间上界控制"
  - "For a C n+1 function the Taylor error has a Lagrange remainder controlled by an interval bound on the n+1 derivative"
  - Lagrange remainder
  - Taylor error bound
student_os: knowledge-atom
atom_id: CALC-APP-004
atom_set: derivative-applications
atom_type: theorem
status: source-checked
mastery_state: unassessed
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

# $C^{n+1}$ 函数的 Taylor 误差可由中间点的 $(n+1)$ 阶导数写成 Lagrange 余项并用区间上界控制
<!-- bilingual-en:start -->
*For a $C^{n+1}$ function the Taylor error has a Lagrange remainder controlled by an interval bound on the $(n+1)$st derivative*
<!-- bilingual-en:end -->

> [!summary] 原子定理
> 设 $f\in C^{n+1}$ 于包含 $a$ 与 $x$ 的区间。则存在位于两点之间的 $\xi$，使
> $$
> f(x)-P_{n,a}(x)
> =\frac{f^{(n+1)}(\xi)}{(n+1)!}(x-a)^{n+1}.
> $$
> 若整段区间上 $|f^{(n+1)}(t)|\le M$，便有
> $$
> |f(x)-P_{n,a}(x)|
> \le \frac{M|x-a|^{n+1}}{(n+1)!}.
> $$
> <!-- bilingual-en:start -->
> If $f$ is $C^{n+1}$ on an interval containing $a$ and $x$, then the Taylor error equals $f^{(n+1)}(\xi)(x-a)^{n+1}/(n+1)!$ for some intermediate $\xi$. Bounding that derivative by $M$ over the entire connecting interval gives the corresponding absolute error bound.
> <!-- bilingual-en:end -->

这里的 $\xi$ 通常未知；定理的用途不是找出它，而是用整段上的导数上界消掉未知位置。例如 $P_2(x)=1+x+x^2/2$ 近似 $e^x$。当 $|x|\le0.1$ 时，连接 $0$ 与 $x$ 的区间上 $e^t\le e^{0.1}$，所以
$$
|e^x-P_2(x)|
\le e^{0.1}\frac{|x|^3}{6}
<1.85\times10^{-4}.
$$
<!-- bilingual-en:start -->
The intermediate point is usually unknown. The theorem becomes practical by replacing its derivative with a bound valid throughout the interval. For $e^x$ on $|x|\le0.1$, the quadratic remainder is below $e^{0.1}|x|^3/6$, hence below $1.85\times10^{-4}$.
<!-- bilingual-en:end -->

三个条件不能偷换：导数阶数要够；$M$ 必须控制连接展开中心与目标点的整段，而不是只在中心成立；结论控制的是指定次数的余项。次数增加是否改善给定点的误差，要看上界与距离共同怎样变化，不能只看阶数。
<!-- bilingual-en:start -->
Three details matter: enough derivatives must exist, $M$ must hold on the whole interval rather than only at the centre, and the bound concerns the remainder after a specified degree. Raising the degree improves a particular estimate only when the derivative bound and distance make the new remainder smaller.
<!-- bilingual-en:end -->

> [!question]- 自检
> 为什么知道 $f^{(n+1)}(a)$ 很小，还不足以给 $x$ 处的 Lagrange 余项界？
>
> **答案：** 余项中的导数在未知中间点 $\xi$ 取值；必须控制从 $a$ 到 $x$ 的整个区间。
> <!-- bilingual-en:start -->
> **Check:** Why does a small value of $f^{(n+1)}(a)$ not suffice to bound the Lagrange remainder at $x$?
>
> **Answer:** The remainder uses the derivative at an unknown intermediate point $\xi$. A bound must cover the entire interval connecting $a$ to $x$.
> <!-- bilingual-en:end -->

## 来源与核验

- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise034_Solutions.pdf#page=1|MIT 18.01SC Exercise 34 solution]]：核对 Lagrange 余项形式、未知中间点及以高阶导数上界估计误差的用途。
- [[01_Math/01_calculus/05_Infinite_Series_and_Improper_Integrals#Taylor 多项式与余项|课程记录 Taylor 多项式与余项]]：交叉核对区间上界与 $e^x$ 误差示例。

---
aliases:
  - "四阶导数绝对值有界时复合Simpson法的绝对误差不超过区间长度乘步长四次方乘该界的一百八十分之一"
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

# 四阶导数绝对值有界时复合Simpson法的绝对误差不超过区间长度乘步长四次方乘该界的一百八十分之一
<!-- bilingual-en:start -->
*With a bound on the fourth derivative, composite Simpson error is at most one one-hundred-eightieth of that bound times interval length times the fourth power of step size*
<!-- bilingual-en:end -->

设 $a<b$、$f\in C^4([a,b])$，并在整个区间上有 $|f^{(4)}(x)|\le K_4$。令 $I=\int_a^b f(x)\,dx$，用偶数 $N\ge2$ 个等宽小区间、$h=(b-a)/N$ 计算 [[Simpson法]] 的 $S_N$，则
<!-- bilingual-en:start -->
Let $a<b$, $f\in C^4([a,b])$, and $|f^{(4)}(x)|\le K_4$ throughout the interval. Set $I=\int_a^b f(x)\,dx$, and compute the [[Simpson法|composite Simpson value]] $S_N$ with an even number $N\ge2$ of equal subintervals and $h=(b-a)/N$. Then
<!-- bilingual-en:end -->

$$
\boxed{|I-S_N|\le\frac{K_4(b-a)h^4}{180}
=\frac{K_4(b-a)^5}{180N^4}.}
$$

$C^4$ 表示直到四阶的导数连续，是本卡采用的充分条件。$K_4$ 必须是整个区间上的有效界；图像看似光滑、只有二阶导数有界，或有限次取样表现良好，都不足以直接建立这个四阶误差保证。
<!-- bilingual-en:start -->
$C^4$ requires continuous derivatives through order four and is the sufficient condition used here. $K_4$ must bound the full interval. Visual smoothness, a second-derivative bound alone, or favourable finite samples do not establish this fourth-order guarantee.
<!-- bilingual-en:end -->

## 每两段的余项怎样累加
<!-- bilingual-en:start -->
*How the remainder on each pair of subintervals accumulates*
<!-- bilingual-en:end -->

在每个宽度为 $2h$ 的片段 $[x_{2j-2},x_{2j}]$ 上，Simpson 余项定理给出某个内部点 $\xi_j$，使
<!-- bilingual-en:start -->
On each panel $[x_{2j-2},x_{2j}]$ of width $2h$, the Simpson remainder theorem gives an interior point $\xi_j$ such that
<!-- bilingual-en:end -->

$$
\int_{x_{2j-2}}^{x_{2j}}f(x)\,dx
-\frac h3[f(x_{2j-2})+4f(x_{2j-1})+f(x_{2j})]
=-\frac{(2h)^5}{2880}f^{(4)}(\xi_j)
=-\frac{h^5}{90}f^{(4)}(\xi_j).
$$

共有 $N/2$ 个这样的片段；对各段余项取绝对值再累加，得到 $\tfrac N2\cdot K_4h^5/90=K_4(b-a)h^4/180$。三次及以下多项式的四阶导数为零，因此误差为零；对一般函数，$N$ 加倍使这个上界变为 $1/16$，不保证实际误差恰好按同一比例变化。
<!-- bilingual-en:start -->
There are $N/2$ panels. Taking absolute values and summing gives $\tfrac N2\cdot K_4h^5/90=K_4(b-a)h^4/180$. For polynomials of degree at most three, the fourth derivative vanishes, so the error is zero. For a general function, doubling $N$ divides this bound by $16$, without forcing the actual error to follow that exact ratio.
<!-- bilingual-en:end -->

例如 $f(x)=x^4$、$[0,1]$、$N=2$，$K_4=24$，界为 $24/(180\cdot2^4)=1/120$。直接算得 $S_2=\tfrac16[0+4(1/2)^4+1]=5/24$、$I=1/5$，所以 $|I-S_2|=1/120$，恰达到界。
<!-- bilingual-en:start -->
For $f(x)=x^4$ on $[0,1]$ with $N=2$, take $K_4=24$, giving $24/(180\cdot2^4)=1/120$. Directly, $S_2=\tfrac16[0+4(1/2)^4+1]=5/24$ and $I=1/5$, so $|I-S_2|=1/120$, attaining the bound.
<!-- bilingual-en:end -->

要保证绝对误差不超过 $\varepsilon>0$，先求 $r=[K_4(b-a)^5/(180\varepsilon)]^{1/4}$，再选 $N\ge\max(2,r)$ 的偶整数。这里只控制精确函数值下的求积截断误差；取样数据误差和浮点舍入误差不由 $K_4$ 公式控制。
<!-- bilingual-en:start -->
For absolute tolerance $\varepsilon>0$, compute $r=[K_4(b-a)^5/(180\varepsilon)]^{1/4}$ and choose an even integer $N\ge\max(2,r)$. This controls quadrature truncation error with exact function values; sample-data error and floating-point roundoff are not controlled by the $K_4$ formula.
<!-- bilingual-en:end -->

> [!question]- 自检
> 若计算得到必须满足 $N\ge7.2$，最少取多少段？若 $N\ge8.1$ 呢？
>
> **答案：** 分别取 $8$ 段和 $10$ 段；既要向上取整满足误差要求，也要保持偶数段。
> <!-- bilingual-en:start -->
> If the bound requires $N\ge7.2$, what is the smallest valid choice? What if it requires $N\ge8.1$? **Answer:** Choose $8$ and $10$, respectively. Round upward enough to meet the bound and retain an even number of subintervals.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [OpenStax《Calculus Volume 2》§3.6，Error Bound for Simpson's Rule，PDF 第 333 页／书页 325](https://d3bxy9euw4e147.cloudfront.net/oscms-prodcms/media/documents/CalculusVolume2-OP_esPpXTB.pdf#page=333)：核对四阶导数条件、常数 $1/180$ 与 $N^{-4}$；本卡采用明确的充分条件 $C^4([a,b])$。
<!-- bilingual-en:start -->
- [OpenStax, *Calculus Volume 2*, §3.6, Error Bound for Simpson's Rule, PDF p. 333 / printed p. 325](https://d3bxy9euw4e147.cloudfront.net/oscms-prodcms/media/documents/CalculusVolume2-OP_esPpXTB.pdf#page=333) supports fourth-derivative control, $1/180$, and $N^{-4}$. This card uses the explicit sufficient condition $C^4([a,b])$.
<!-- bilingual-en:end -->
- [UBC Math 405，Topic 4b: Composite Quadrature，PDF 第 1–2 页](https://www.math.ubc.ca/~cbm/math405/2016/04b_composite_quad.pdf#page=1)：核对宽 $2h$ 片段的余项 $-(2h)^5f^{(4)}(\xi)/2880$、负号和复合求和。本卡统一用 $N$ 表示小区间总数；四次例子已直接重算。
<!-- bilingual-en:start -->
- [UBC Math 405, Topic 4b: Composite Quadrature, PDF pp. 1–2](https://www.math.ubc.ca/~cbm/math405/2016/04b_composite_quad.pdf#page=1) supports the panel remainder $-(2h)^5f^{(4)}(\xi)/2880$, its sign, and composite summation. This card uses $N$ consistently for the total number of small subintervals. The quartic example was recalculated.
<!-- bilingual-en:end -->

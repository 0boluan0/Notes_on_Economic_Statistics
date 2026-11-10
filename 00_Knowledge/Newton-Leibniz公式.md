---
aliases:
  - Newton–Leibniz formula
  - Antiderivative evaluation of a definite integral
student_os: knowledge-atom
atom_id: CALC-DEFINT-013
atom_type: theorem
status: source-checked
part_of:
  - "[[定积分与应用.canvas]]"
requires:
  - "[[原函数]]"
  - "[[积分求导定理]]"
  - "[[原函数相差常数]]"
leads_to:
  - "[[定积分换元]]"
  - "[[分部积分]]"
---

<!-- bilingual-en:start -->
# Newton–Leibniz formula
<!-- bilingual-en:end -->
*Newton–Leibniz 公式*

<!-- bilingual-en:start -->
The **Newton–Leibniz formula** turns a definite integral into two function evaluations. If $f$ describes a rate and $F$ is a function whose derivative is $f$, the total signed accumulation from $a$ to $b$ is the change $F(b)-F(a)$. For example, a rate $f(x)=2x$ has antiderivative $F(x)=x^2$, so its integral from $1$ to $3$ is $9-1=8$.
<!-- bilingual-en:end -->

**Newton–Leibniz 公式**把一个定积分化成两次函数值计算。如果 $f$ 表示变化率，而 $F$ 求导后得到 $f$，那么从 $a$ 到 $b$ 的带符号总累积，就是变化量 $F(b)-F(a)$。例如，变化率 $f(x)=2x$ 有原函数 $F(x)=x^2$，所以从 $1$ 到 $3$ 的积分为 $9-1=8$。

<!-- bilingual-en:start -->
The [[定积分|definite integral]] is defined independently through limiting sums. An [[原函数|antiderivative]] is defined through differentiation. The formula is a theorem connecting those two definitions; it does not redefine an integral as an endpoint difference without conditions.
<!-- bilingual-en:end -->

[[定积分]]通过求和极限独立定义，[[原函数]]则通过求导定义。这个公式是连接两种定义的定理，不是无条件地把积分重新定义成端点差。

> [!important] Evaluate an antiderivative at the endpoints · 原函数上限值减下限值
> <!-- bilingual-en:start -->
> Let $a<b$ be finite real numbers. Suppose the real function $f$ is continuous on $[a,b]$, and $F$ is continuous on $[a,b]$ with $F'(x)=f(x)$ for every $x\in(a,b)$. Then
> <!-- bilingual-en:end -->
>
> 设 $a<b$ 是有限实数，实函数 $f$ 在 $[a,b]$ 连续；$F$ 在 $[a,b]$ 连续，并在每个 $x\in(a,b)$ 满足 $F'(x)=f(x)$。则
>
> $$\int_a^b f(x)\,dx=[F(x)]_a^b=F(b)-F(a).$$
>
> <!-- bilingual-en:start -->
> The bracket $[F(x)]_a^b$ means “upper endpoint value minus lower endpoint value.” No derivative is required at the endpoints. Any additive constant in $F$ cancels: $(F(b)+C)-(F(a)+C)=F(b)-F(a)$. The resulting definite integral is one number, with no extra $+C$.
> <!-- bilingual-en:end -->
>
> 方括号 $[F(x)]_a^b$ 表示“上限值减下限值”。不要求端点有导数。$F$ 中任何加法常数都会抵消：$(F(b)+C)-(F(a)+C)=F(b)-F(a)$。所得定积分是一个数，不再额外附加 $+C$。

<!-- bilingual-en:start -->
MIT 18.01SC calls this **FTC I**. Its FTC II is the [[积分求导定理|differentiation of an accumulation integral]]. One direction differentiates an integral-defined function; the other evaluates an integral using an antiderivative. State the direction explicitly when comparing course numbering.
<!-- bilingual-en:end -->

MIT 18.01SC 把它称为 **FTC I**，把[[积分求导定理|累积积分的求导]]称为 FTC II。一个方向对积分定义的函数求导；另一个方向借助原函数求出积分。比较课程编号时，最好把所说方向明确写出。

<!-- bilingual-en:start -->
## Why the endpoint difference is the integral
<!-- bilingual-en:end -->
*为什么端点差恰好就是积分*

<!-- bilingual-en:start -->
Since [[连续函数可积|a continuous function on a finite closed interval is integrable]], define the [[累积函数|accumulation function]] $G(x)=\int_a^x f(t)\,dt$. It is continuous on $[a,b]$, has $G(a)=0$, and the accumulation theorem gives $G'=f$ in the interior. Thus $F$ and $G$ have the same derivative on $(a,b)$.
<!-- bilingual-en:end -->

因为[[连续函数可积|有限闭区间上的连续函数可积]]，可以定义[[累积函数]] $G(x)=\int_a^x f(t)\,dt$。它在 $[a,b]$ 连续，满足 $G(a)=0$，而积分求导定理给出内部的 $G'=f$。因此，$F,G$ 在 $(a,b)$ 有相同导数。

<!-- bilingual-en:start -->
The [[原函数相差常数|constant-difference theorem]] gives $F(x)-G(x)=C$ throughout the interior, with one constant $C$. Continuity of both functions extends that equality to $a$ and $b$. Subtracting the two endpoint equations eliminates $C$:
<!-- bilingual-en:end -->

由[[原函数相差常数]]，内部处处有 $F(x)-G(x)=C$，其中是同一个常数 $C$。两函数在端点连续，使等式延伸到 $a,b$。把两个端点的等式相减，消去 $C$：

$$F(b)-F(a)=G(b)-G(a)=\int_a^b f(t)\,dt.$$

> [!example] Keep both minus signs · 保留两个负号
> <!-- bilingual-en:start -->
> Since $(-\cos x)'=\sin x$ and both functions are continuous on $[0,\pi]$,
> <!-- bilingual-en:end -->
>
> 因为 $(-\cos x)'=\sin x$，且相应函数在 $[0,\pi]$ 连续，所以
>
> $$\int_0^\pi\sin x\,dx=[-\cos x]_0^\pi
> =-\cos\pi-(-\cos0)=1-(-1)=2.$$
>
> <!-- bilingual-en:start -->
> The negative sign in the antiderivative and the subtraction of the lower endpoint have different roles. This integral is also geometric area because the integrand is nonnegative throughout the interval.
> <!-- bilingual-en:end -->
>
> 原函数自身的负号，与“减去下限值”的负号，各有不同作用。由于这一段被积函数非负，该积分也等于几何面积。

<!-- bilingual-en:start -->
## A more general version assumes an antiderivative already exists
<!-- bilingual-en:end -->
*更一般的版本先假设原函数已经存在*

<!-- bilingual-en:start -->
Continuity of $f$ is sufficient but not required for the endpoint formula. It is enough that $f$ is **Riemann integrable on $[a,b]$**, while $F$ remains continuous on $[a,b]$ and differentiable inside with $F'=f$ there. These conditions do not promise that every integrable $f$ has such an $F$; the [[积分求导定理|step-function counterexample]] shows why that promise would be false. Values assigned to $f$ at the two endpoints do not affect its integral or the interior derivative condition.
<!-- bilingual-en:end -->

$f$ 连续是充分条件，但不是端点公式必需的条件。只要 $f$ **在 $[a,b]$ 上黎曼可积**，而 $F$ 仍在闭区间连续、内部可导且满足 $F'=f$，结论依然成立。这些条件没有承诺每个可积的 $f$ 都存在这样的 $F$；[[积分求导定理|阶梯函数反例]]说明这种承诺会出错。给 $f$ 两个端点赋什么值，不影响积分，也不影响内部导数条件。

> [!note]- Proof using sampled sums · 用采样和证明
> <!-- bilingual-en:start -->
> Choose a partition $a=x_0<\cdots<x_n=b$, and let $\Delta x_i=x_i-x_{i-1}$. On each piece the [[拉格朗日中值定理|mean value theorem]] gives an interior point $\xi_i$ with $F(x_i)-F(x_{i-1})=F'(\xi_i)\Delta x_i=f(\xi_i)\Delta x_i$. Summing makes the intermediate endpoint values cancel:
> <!-- bilingual-en:end -->
>
> 选取分割 $a=x_0<\cdots<x_n=b$，令 $\Delta x_i=x_i-x_{i-1}$。每一段都可用[[拉格朗日中值定理]]，得到内部点 $\xi_i$，使 $F(x_i)-F(x_{i-1})=F'(\xi_i)\Delta x_i=f(\xi_i)\Delta x_i$。求和后，中间端点值全部抵消：
>
> $$F(b)-F(a)=\sum_{i=1}^n f(\xi_i)\Delta x_i.$$
>
> <!-- bilingual-en:start -->
> The right side is a [[黎曼和|Riemann sum]]. Let the largest piece width tend to zero. Riemann integrability ensures that every such selection of sample points has the same integral limit. The left side never changes, so it must equal that limit. This proof needs no continuity of $f$, but it does need all the stated conditions on $F$ and integrability of $f$.
> <!-- bilingual-en:end -->
>
> 右侧是一个[[黎曼和]]。让最大段宽趋于零，黎曼可积性保证这种采样方式同样趋向积分。左侧始终不变，因此必定等于该极限。这个证明不要求 $f$ 连续，但仍然需要上述关于 $F$ 的所有条件与 $f$ 的可积性。

<!-- bilingual-en:start -->
Endpoint continuity of $F$ cannot simply be discarded. On $[0,1]$, let $f(x)=1$, and let $F(x)=x$ for $x<1$ but $F(1)=99$. The interior derivative is still $1$, yet $F(1)-F(0)=99$ whereas the integral is $1$. The endpoint jump invalidates the formula's hypotheses.
<!-- bilingual-en:end -->

$F$ 的端点连续性不能随便删去。例如在 $[0,1]$ 上令 $f(x)=1$，并令 $x<1$ 时 $F(x)=x$，但 $F(1)=99$。内部导数仍是 $1$，可端点差是 $99$，积分却是 $1$；端点跳跃破坏了公式的假设。

> [!warning] An algebraic antiderivative cannot bridge a singularity · 形式原函数不能跨越奇点
> <!-- bilingual-en:start -->
> Although $(-1/x)'=1/x^2$ away from zero, substituting $-1$ and $1$ into $-1/x$ would give $-2$, which is not $\int_{-1}^1x^{-2}\,dx$. The integrand is undefined and unbounded at zero, and the proposed antiderivative is not continuous on the full interval. As an [[反常积分|improper integral]], each side must be treated separately: $\int_\varepsilon^1x^{-2}\,dx=1/\varepsilon-1\to+\infty$ as $\varepsilon\downarrow0$. The analogous left-hand integral also diverges. Endpoints alone cannot cancel that failure.
> <!-- bilingual-en:end -->
>
> 虽然离开零点时有 $(-1/x)'=1/x^2$，把 $-1,1$ 代入 $-1/x$ 却会得到 $-2$，这不是 $\int_{-1}^1x^{-2}\,dx$。被积函数在零点无定义且无界，所用原函数也不在整段连续。作为[[反常积分]]，必须分别处理两侧：$\varepsilon\downarrow0$ 时，$\int_\varepsilon^1x^{-2}\,dx=1/\varepsilon-1\to+\infty$，左侧相应积分同样发散。不能只用两个外端点把这个问题抵消掉。

<!-- bilingual-en:start -->
Equal bounds give zero; reversing bounds reverses the sign. These conventions agree with the endpoint formula whenever the whole interval satisfies its hypotheses. For [[定积分换元|definite-integral substitution]] and [[分部积分|integration by parts]], the same interval and domain checks remain part of the calculation.
<!-- bilingual-en:end -->

上下限相同，积分为零；交换上下限，积分变号。只要整段满足假设，这些约定与端点公式一致。使用[[定积分换元]]、[[分部积分]]时，整段区间与定义域检查仍然是计算的一部分。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/03_Definite_Integrals/Ses47b_Lecture_Notes.pdf#page=1|MIT 18.01SC Session 47b, pp. 1–2]]: endpoint evaluation, the sine example and the course's FTC I numbering.
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/03_Definite_Integrals/Ses52b_Lecture_Notes.pdf#page=1|MIT Session 52b, p. 1]]: the accumulation-function/constant-difference proof for continuous $f$.
- [Jiří Lebl, Basic Analysis I, version 6.3, May 2026, §5.3.1, PDF pp. 200–201, Theorem 5.3.1](https://www.jirka.org/ra/realanal.pdf#page=200): Riemann-integrable $f$, closed-interval continuity of $F$, interior differentiability and the mean-value argument on partition pieces. The endpoint-jump and singular-integral checks are derived directly above.
<!-- bilingual-en:end -->

- MIT 47b 第1–2页支持端点计算、正弦例子及课程 FTC I 编号；52b 第1页支持连续 $f$ 情形下累积函数与相差常数的证明。
- Lebl 上述版本第5.3.1节、PDF200–201页定理5.3.1，支持黎曼可积版本、$F$ 的闭区间连续与内部可导条件，以及分段使用中值定理的证明。端点跳跃与奇点积分已在正文直接核算。

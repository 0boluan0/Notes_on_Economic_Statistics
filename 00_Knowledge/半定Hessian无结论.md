---
aliases:
  - "Singular semidefinite Hessian test"
  - "Inconclusive semidefinite Hessian test"
student_os: knowledge-atom
atom_id: LA-SPD-029
atom_set: symmetric-positive-definite
atom_type: diagnostic-boundary
status: source-checked
requires:
  - "[[半正定矩阵]]"
  - "[[多元Taylor近似]]"
related:
  - "[[Hessian 局部极小判据]]"
  - "[[Hessian 局部极大判据]]"
  - "[[Hessian 鞍点判据]]"
  - "[[严格最优不推系统非奇异]]"
part_of:
  - "[[对称矩阵与正定二次型.canvas]]"
  - "[[多元微分.canvas]]"
  - "[[多元优化.canvas]]"
---

<!-- bilingual-en:start -->
# When the semidefinite Hessian test is inconclusive
<!-- bilingual-en:end -->

*奇异半定 Hessian 的判别边界*

<!-- bilingual-en:start -->
A Hessian can capture upward curvature in some directions while saying nothing decisive about others. At a stationary point, a **singular semidefinite Hessian** leaves at least one direction with zero quadratic change. Terms beyond that quadratic approximation can determine what happens there. Consequently the Hessian alone may not distinguish a minimum from a saddle, or a strict minimum from a non-strict one.
<!-- bilingual-en:end -->

Hessian 可能抓住某些方向向上的二阶弯曲，却无法决定其他方向的行为。在驻点处，**奇异半定 Hessian** 至少留下一个二次变化为零的方向；二次近似以外的项可能决定那里发生什么。因此，仅凭 Hessian 未必能区分极小点与鞍点，也未必能区分严格和非严格极小。

<!-- bilingual-en:start -->
The simplest comparison keeps almost everything the same:
<!-- bilingual-en:end -->

最直接的比较，是只改变一个二次近似看不到的项：

$$f_+(x,y)=x^2+y^4,\qquad f_-(x,y)=x^2-y^4.$$

<!-- bilingual-en:start -->
At the origin both functions have value $0$, gradient $0$, and the same [[Hessian矩阵|Hessian]]:
<!-- bilingual-en:end -->

在原点，两函数的值和梯度都为 $0$，而且具有相同的[[Hessian矩阵]]：

$$H_{f_+}(0,0)=H_{f_-}(0,0)=\begin{pmatrix}2&0\\0&0\end{pmatrix}.$$

> [!example] Same matrix, different answer · 同一矩阵，两种结论
> <!-- bilingual-en:start -->
> For $f_+$, both terms are nonnegative and their sum is zero only at the origin, so the origin is a strict minimum. For $f_-$, values are positive at $(t,0)$ and negative at $(0,t)$ for every $t\ne0$, so the origin is a saddle. The Hessian misses the sign of $y^4$ because its second derivative is zero at $y=0$.
> <!-- bilingual-en:end -->
>
> 对 $f_+$，两项非负且只在原点同时为零，所以原点是严格极小。对 $f_-$，任意 $t\ne0$ 时，$(t,0)$ 处为正、$(0,t)$ 处为负，所以原点是鞍点。Hessian 看不到 $y^4$ 的正负号，因为它在 $y=0$ 处的二阶导为零。

<!-- bilingual-en:start -->
## Why the word singular matters
<!-- bilingual-en:end -->

*为什么要明确“奇异半定”*

<!-- bilingual-en:start -->
Work with a real-valued $f$ that is $C^2$ near an interior point $a$ of an [[开集|open subset]] of $\mathbb R^n$, for finite integer $n\ge1$. Thus the partial derivatives through order two are continuous nearby and $H=H_f(a)$ is symmetric. Assume $a$ is a [[驻点|stationary point]], meaning $\nabla f(a)=0$.
<!-- bilingual-en:end -->

设 $f$ 为实值函数，在 $\mathbb R^n$ 的某个[[开集]]中、内点 $a$ 附近为 $C^2$，$n\ge1$ 为有限整数。也就是说，直到二阶的偏导都在附近连续，因此 $H=H_f(a)$ 对称。还要求 $a$ 是[[驻点]]，即 $\nabla f(a)=0$。

<!-- bilingual-en:start -->
A [[半正定矩阵|positive-semidefinite matrix]] has $h^THh\ge0$ for every real column vector $h$; here $T$ is [[矩阵转置|transpose]]. This inclusive definition also contains [[正定矩阵|positive-definite matrices]], whose quadratic form is strictly positive for every $h\ne0$. The undecided case is positive semidefinite **but not positive definite**, equivalently singular positive semidefinite. For a real symmetric semidefinite matrix, a nonzero zero-quadratic direction lies in its [[零空间|nullspace]], so $Hu=0$. The [[半负定矩阵|negative-semidefinite]] case reverses all signs.
<!-- bilingual-en:end -->

[[半正定矩阵]]要求对每个实列向量 $h$ 都有 $h^THh\ge0$，其中 $T$ 是[[矩阵转置|转置]]。这个宽义定义包含[[正定矩阵]]；正定要求每个 $h\ne0$ 的二次型值都严格为正。无定论的情形是半正定**但非正定**，等价地说，是奇异半正定。对实对称半定矩阵，非零的零二次型方向属于[[零空间]]，即满足 $Hu=0$。[[半负定矩阵]]对应把所有符号反转。

<!-- bilingual-en:start -->
## The quadratic term disappears on a null direction
<!-- bilingual-en:end -->

*零方向为什么留下缺口*

<!-- bilingual-en:start -->
At a stationary point, the [[多元Taylor近似|Taylor expansion]] is
<!-- bilingual-en:end -->

在驻点处，[[多元Taylor近似|Taylor 展开]]为

$$f(a+h)-f(a)=\frac12h^THh+r(h),\qquad \frac{r(h)}{\|h\|_2^2}\to0.$$

<!-- bilingual-en:start -->
Here $\|h\|_2$ is [[欧氏范数|Euclidean length]], and the ratio condition applies as nonzero $h$ tends to zero from all directions. Along a fixed null direction $u\ne0$, set $h=tu$. Then the quadratic term is zero and only $r(tu)$ remains. Being small relative to $t^2$ does not specify its sign: $t^4$, $-t^4$, and $0$ all satisfy that smallness condition. This explanation requires no assumption that third or higher derivatives exist.
<!-- bilingual-en:end -->

这里 $\|h\|_2$ 是[[欧氏范数|欧氏长度]]，比值条件要求非零 $h$ 从各种方向趋近零时都成立。沿固定零空间方向 $u\ne0$ 取 $h=tu$，二次项为零，只剩 $r(tu)$。相对 $t^2$ 很小并不能决定正负号：$t^4$、$-t^4$ 与 $0$ 都满足这个小量条件。这套解释并不要求三阶或更高阶导数存在。

<!-- bilingual-en:start -->
The function $f_0(x,y)=x^2$ has the same gradient and Hessian at the origin as $f_+$ and $f_-$. It has a minimum there, but not a strict one: every nearby point $(0,y)$ has the same value. Thus the same nonzero singular Hessian allows a strict minimum, a non-strict minimum, or a saddle.
<!-- bilingual-en:end -->

$f_0(x,y)=x^2$ 在原点也具有与 $f_+,f_-$ 相同的梯度和 Hessian。原点是极小点，却不是严格极小，因为附近每个 $(0,y)$ 都有相同的值。因此，同一个非零奇异 Hessian 可以对应严格极小、非严格极小或者鞍点。

<!-- bilingual-en:start -->
## Inconclusive does not mean no information at all
<!-- bilingual-en:end -->

*无定论不等于完全没有信息*

<!-- bilingual-en:start -->
The particular Hessian $\operatorname{diag}(2,0)$ cannot occur at a $C^2$ interior local maximum. Along its positive direction, the Taylor argument still guarantees nearby larger values. More generally, a nonzero positive-semidefinite Hessian at a stationary point rules out a local maximum, although it need not decide minimum versus saddle. A nonzero negative-semidefinite Hessian similarly rules out a local minimum.
<!-- bilingual-en:end -->

特定的 Hessian $\operatorname{diag}(2,0)$ 不可能出现在 $C^2$ 内点局部极大处：沿正号方向，Taylor 论证仍保证附近有更大的值。一般地，驻点处非零半正定 Hessian 可以排除局部极大，但未必区分极小与鞍点；非零半负定 Hessian 则可以排除局部极小。

<!-- bilingual-en:start -->
If the Hessian is the zero matrix, even that exclusion disappears. At the origin, $x^4+y^4$ has a strict minimum, $-x^4-y^4$ a strict maximum, and $x^4-y^4$ a saddle. All three have zero gradient and zero Hessian. Negating the earlier $f_+,f_-,f_0$ examples gives the corresponding nonzero negative-semidefinite cases.
<!-- bilingual-en:end -->

若 Hessian 是零矩阵，连这种排除能力也会消失。在原点，$x^4+y^4$ 有严格极小，$-x^4-y^4$ 有严格极大，$x^4-y^4$ 则是鞍点；三者的梯度与 Hessian 全为零。把先前的 $f_+,f_-,f_0$ 整体取负，则得到非零半负定的对应情形。

<!-- bilingual-en:start -->
## Return to the original function
<!-- bilingual-en:end -->

*回到原函数继续判断*

<!-- bilingual-en:start -->
The decisive tests have strict signs: [[Hessian 局部极小判据|positive definite gives a strict local minimum]], [[Hessian 局部极大判据|negative definite gives a strict local maximum]], and [[Hessian 鞍点判据|indefinite gives a saddle]], under their stationary $C^2$ interior assumptions. A singular semidefinite matrix does not complete that classification.
<!-- bilingual-en:end -->

能完成分类的判据具有严格符号：在相应的 $C^2$ 内点驻点假设下，[[Hessian 局部极小判据|正定给出严格局部极小]]，[[Hessian 局部极大判据|负定给出严格局部极大]]，[[Hessian 鞍点判据|不定给出鞍点]]。奇异半定矩阵不能完成这一分类。

<!-- bilingual-en:start -->
Once the whole Hessian is known to be singular semidefinite, calculating more of its minors cannot recover information absent from that matrix. Inspect the original function, factor it, bound its sign on a whole neighborhood, or obtain additional remainder information. To prove a minimum, a bound must cover all nearby points, not only a few chosen directions. To disprove one, a single sequence of nearby points with smaller values is enough. The [[01_SOFP Lecture 1 - 二次型、Taylor 展开与凹凸性#6.3 半定无结论不是“主子式算得还不够多”|course discussion of the same-Hessian examples]] provides the classroom context.
<!-- bilingual-en:end -->

一旦整张 Hessian 已确认是奇异半定，再计算它的其他主子式也无法补回矩阵没有记录的信息。应查看原函数，尝试因式分解、在整个邻域上控制符号，或取得更多余项信息。证明极小需要覆盖所有附近点，不能只检查几条选定方向；否定极小，只需找到一列趋近该点且函数值更低的点。[[01_SOFP Lecture 1 - 二次型、Taylor 展开与凹凸性#6.3 半定无结论不是“主子式算得还不够多”|课程中关于同 Hessian 例子的讨论]]保留课堂背景。

> [!question]- What did the matrix miss? · 矩阵漏掉了什么？
> <!-- bilingual-en:start -->
> Why does the positive-semidefinite Hessian of $x^2-y^4$ not prevent a saddle? Its $y$ direction has zero quadratic change. The negative term first appears at order four, while the positive $x$ direction is already visible at order two.
> <!-- bilingual-en:end -->
>
> 为什么 $x^2-y^4$ 的半正定 Hessian 没有排除鞍点？因为 $y$ 方向的二次变化为零，负项到四阶才出现；正号的 $x$ 方向则在二阶已经可见。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/08_EC400_MathsCamp/2026_Course_Materials/02_SOFP/Lectures/EC400 Slides Lecture 2.pdf#page=10|LSE EC400 SOFP Lecture 2, slides 10–14]], and [[01_Math/08_EC400_MathsCamp/2026_Course_Materials/02_SOFP/Lectures/EC400 Lecture Notes SOFP.pdf#page=21|SOFP lecture notes, pp.21–22]]: interior stationarity, definite and semidefinite conditions, the indefinite example, and local versus global conclusions. The sign proofs here keep the Taylor remainder explicitly.
- [MIT 18.S096, Second Derivatives, §12.1, PDF pp.1–2 / print pp.87–88](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/mit18_s096iap23_lec12.pdf#page=2): Hessian quadratic term and the second-order remainder. The linked [[多元Taylor近似|Taylor card]] gives the stated $C^2$ assumptions and the remainder argument.
- [OpenStax, Calculus Volume 3, §4.7, saddle definition and theorem 4.17](https://openstax.org/books/calculus-volume-3/pages/4-7-maxima-minima-problems): the two-variable meaning of a stationary saddle and the inconclusive degenerate second-derivative test. The examples and inequalities in this card are verified directly.
<!-- bilingual-en:end -->

- LSE EC400 SOFP Lecture 2，第10–14张幻灯片，以及 SOFP 讲义第21–22页：内点驻点、定号与半定条件、不定例子，以及局部和全局结论的区别。正文保留 Taylor 余项，完整证明符号判断。
- MIT 18.S096《Second Derivatives》§12.1，PDF 第1–2页／印本87–88页：Hessian 二次项及二阶余项；所链接的 Taylor 卡给出本文使用的 $C^2$ 假设和余项论证。
- OpenStax《Calculus Volume 3》§4.7 的鞍点定义与定理4.17：二元驻点鞍点的含义及退化二阶检验的无定论边界。本文例子与不等式均直接核验。

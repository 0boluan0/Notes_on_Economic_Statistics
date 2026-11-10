---
student_os: knowledge-atom
atom_id: OPT-TOOLS-016
aliases:
- Strict concavity and negative-definite Hessians
- 严格凹的 Hessian 充分条件
status: source-checked
---

<!-- bilingual-en:start -->
# Strict concavity and negative-definite Hessians
<!-- bilingual-en:end -->
*严格凹与负定 Hessian*

<!-- bilingual-en:start -->
**Strict concavity** means that mixing two different inputs gives an output strictly above the corresponding mixture of their outputs. An everywhere negative-definite Hessian is one way to guarantee this strict inequality. It is a sufficient curvature test; some strictly concave functions have a direction with zero second derivative at a point.
<!-- bilingual-en:end -->

**严格凹性**指把两个不同输入混合后，所得输出严格高于两个输出的对应混合。Hessian 处处负定是一种保证这条严格不等式的办法。它是充分的曲率检验；有些严格凹函数仍会在某点的某个方向上具有零二阶导数。

<!-- bilingual-en:start -->
Recall that a [[凸集|convex domain]] contains every segment joining its points. In the [[凹凸性|chord definition]], strict concavity requires, for every $x\ne y$ in the domain and every $0<\theta<1$,
<!-- bilingual-en:end -->

先回顾，[[凸集|凸域]]包含连接其中任意两点的线段。在[[凹凸性|弦的定义]]中，严格凹性要求域内每对 $x\ne y$ 和每个 $0<\theta<1$ 都满足

$$f((1-\theta)x+\theta y)>(1-\theta)f(x)+\theta f(y).$$

<!-- bilingual-en:start -->
The distinct inputs and interior weight matter: equal inputs or a weight of zero or one give equality, even for a strictly concave function.
<!-- bilingual-en:end -->

输入不同、权重在内部这两个条件很重要：即使函数严格凹，输入相同或权重为零、一时也会取等。

<!-- bilingual-en:start -->
## The sufficient curvature condition
<!-- bilingual-en:end -->
*充分的曲率条件*

<!-- bilingual-en:start -->
Let $U\subseteq\mathbb R^n$ be nonempty, [[开集|open]] and convex, with finite integer $n\ge1$. Assume $f:U\to\mathbb R$ is $C^2$, meaning that all partial derivatives through order two exist and are continuous. Its [[Hessian矩阵|Hessian]] $H_f(x)$ is then symmetric. [[负定矩阵|Negative definiteness]] means $v^TH_f(x)v<0$ for every real nonzero column vector $v$; this is the sign of a scalar quadratic value, not a requirement that all matrix entries be negative. Here $T$ denotes [[矩阵转置|transpose]].
<!-- bilingual-en:end -->

设 $U\subseteq\mathbb R^n$ 非空、为[[开集]]且为凸集，$n\ge1$ 是有限整数。假设 $f:U\to\mathbb R$ 为 $C^2$，即直到二阶的全部偏导都存在且连续。它的[[Hessian矩阵]] $H_f(x)$ 因而对称。[[负定矩阵|负定]]指每个非零实列向量 $v$ 都使 $v^TH_f(x)v<0$；检查的是整个标量二次型值，不是要求矩阵全部元素为负。这里 $T$ 表示[[矩阵转置|转置]]。

> [!important] A sufficient global test · 全域充分判据
> <!-- bilingual-en:start -->
> If $H_f(x)$ is negative definite at **every** $x\in U$, then $f$ is strictly concave on $U$. No stationary point is required for this shape statement, and no single negative constant bounding curvature throughout $U$ is assumed.
> <!-- bilingual-en:end -->
>
> 若 $H_f(x)$ 在**每个** $x\in U$ 都负定，则 $f$ 在 $U$ 上严格凹。这条形状结论不要求存在驻点，也不假设用同一个负常数控制全域曲率。

在开凸域上，$f\in C^2$ 且 $H_f(x)\prec0$ 处处成立，则每条非平凡线段限制的二阶导数严格负，因此 $f$ 严格凹。

<!-- bilingual-en:start -->
To prove it, choose distinct $x,y\in U$ and put $d=y-x\ne0$. Along the entire segment, let $g(t)=f(x+td)$ for $0\le t\le1$. Convexity keeps the segment in the domain; openness and $C^2$ provide the needed derivatives. The [[Hessian方向二阶导数|directional Hessian identity]] gives
<!-- bilingual-en:end -->

证明时，任取不同的 $x,y\in U$，记 $d=y-x\ne0$。沿整条线段令 $g(t)=f(x+td)$，其中 $0\le t\le1$。凸性使线段留在域内；开放性与 $C^2$ 保证所需导数存在。[[Hessian方向二阶导数]]给出

$$g''(t)=d^TH_f(x+td)d<0.$$

<!-- bilingual-en:start -->
By the [[拉格朗日中值定理|mean value theorem]] applied to $g'$, this makes $g'$ strictly decreasing. For $0<\theta<1$, applying the theorem to $g$ on $[0,\theta]$ and $[\theta,1]$ supplies $a\in(0,\theta)$ and $b\in(\theta,1)$ with
<!-- bilingual-en:end -->

对 $g'$ 使用[[拉格朗日中值定理]]，便知 $g'$ 严格递减。对 $0<\theta<1$，再在 $[0,\theta]$ 和 $[\theta,1]$ 上对 $g$ 使用中值定理，得到 $a\in(0,\theta)$、$b\in(\theta,1)$，使

$$\frac{g(\theta)-g(0)}{\theta}=g'(a)
\gt g'(b)=\frac{g(1)-g(\theta)}{1-\theta}.$$

<!-- bilingual-en:start -->
Both denominators are positive. Multiplying and collecting terms gives $g(\theta)>(1-\theta)g(0)+\theta g(1)$, exactly the required strict chord inequality. This works for every distinct pair of inputs, so proves strict concavity on the whole domain.
<!-- bilingual-en:end -->

两个分母都为正。交叉相乘并整理，得到 $g(\theta)>(1-\theta)g(0)+\theta g(1)$，正是所需严格弦不等式。这个论证适用于每一对不同输入，因此证明了整个定义域上的严格凹性。

> [!example] A quadratic with mixed coordinates · 含混合坐标项的二次函数
> <!-- bilingual-en:start -->
> On $\mathbb R^2$, take $f(x_1,x_2)=-x_1^2-2x_2^2+x_1x_2$. Its constant Hessian satisfies
> <!-- bilingual-en:end -->
>
> 在 $\mathbb R^2$ 上取 $f(x_1,x_2)=-x_1^2-2x_2^2+x_1x_2$。它的 Hessian 为常矩阵，满足
>
> $$H_f=\begin{pmatrix}-2&1\\1&-4\end{pmatrix},\qquad
> v^TH_fv=-2\left(v_1-\frac{v_2}{2}\right)^2-\frac72v_2^2.$$
>
> <!-- bilingual-en:start -->
> If this value is zero, both squares vanish, forcing $v_2=0$ and then $v_1=0$. It is therefore strictly negative for every nonzero $v$, proving strict concavity. For instance, the inputs $(0,0)$ and $(2,0)$ have outputs $0$ and $-4$, while their midpoint $(1,0)$ has output $-1>-2$, strictly above the average output.
> <!-- bilingual-en:end -->
>
> 若二次型值为零，则两个平方都为零，先有 $v_2=0$，再有 $v_1=0$。所以每个非零 $v$ 都使它严格为负，严格凹性得证。例如，输入 $(0,0)$ 和 $(2,0)$ 的输出分别为 $0,-4$，中点 $(1,0)$ 的输出为 $-1>-2$，严格高于输出的平均值。

<!-- bilingual-en:start -->
## Why the converse fails
<!-- bilingual-en:end -->
*为什么逆命题不成立*

<!-- bilingual-en:start -->
Strictly negative directional second derivatives on every nontrivial segment imply strict concavity. The function $-x^4$ disproves the converse: its derivative is strictly decreasing but its second derivative vanishes at zero. Ordinary concavity uses the [[Hessian凹凸性判据|semidefinite criterion]], and [[严格凹不保证解存在|existence remains a separate issue]].
<!-- bilingual-en:end -->

反向失败：$f(x)=-x^4$ 严格凹，因其导数 $-4x^3$ 严格递减，但 $f''(0)=0$。严格凹是任意不同端点的严格弦不等式，不是二阶导数处处严格负的另一种名称。普通凹性用[[Hessian凹凸性判据]]；最优解的取得性由[[严格凹不保证解存在]]区分。

<!-- bilingual-en:start -->
Here is the full strictness check. If $u<v$, then $v^3-u^3=(v-u)(u^2+uv+v^2)>0$: the first factor is positive, and the second equals $(u+v/2)^2+3v^2/4$, which could be zero only if $u=v=0$. Thus $f'(u)=-4u^3>-4v^3=f'(v)$. For any $x<z<y$, the mean value theorem gives one derivative value for the secant from $x$ to $z$ and a later, strictly smaller derivative value for the secant from $z$ to $y$:
<!-- bilingual-en:end -->

完整的严格性检验如下。若 $u<v$，则 $v^3-u^3=(v-u)(u^2+uv+v^2)>0$：第一因子为正，第二因子等于 $(u+v/2)^2+3v^2/4$，只有 $u=v=0$ 才可能为零。因此 $f'(u)=-4u^3>-4v^3=f'(v)$。对任意 $x<z<y$，中值定理把从 $x$ 到 $z$ 的割线斜率表示为一个导数值，把从 $z$ 到 $y$ 的割线斜率表示为更靠后的、严格较小的导数值：

$$\frac{f(z)-f(x)}{z-x}>\frac{f(y)-f(z)}{y-z}.$$

<!-- bilingual-en:start -->
Multiplication by the positive denominators gives the strict chord inequality at $z=(1-\theta)x+\theta y$. Every interior point of the segment has this form, including segments crossing zero. Hence $-x^4$ is strictly concave on $\mathbb R$, even though $f''(x)=-12x^2$ is zero at $x=0$.
<!-- bilingual-en:end -->

乘以正分母，便得到 $z=(1-\theta)x+\theta y$ 处的严格弦不等式。线段内的每个点都有这种形式，包括跨过零点的线段。因此 $-x^4$ 在 $\mathbb R$ 上严格凹，尽管 $f''(x)=-12x^2$ 在 $x=0$ 为零。

<!-- bilingual-en:start -->
## Keep local tests and existence separate
<!-- bilingual-en:end -->
*区分局部检验与解的存在*

<!-- bilingual-en:start -->
An everywhere negative-semidefinite Hessian guarantees ordinary concavity, but it need not guarantee strictness: a constant function has Hessian zero and equality on every chord. Checking a negative-definite Hessian only at one stationary point instead invokes the [[Hessian 局部极大判据|strict local maximum test]], with its own neighborhood assumptions. For example, $q(x)=-x^2+x^4$ has $q'(0)=0$ and $q''(0)=-2$, but $q''(1)=10$; it is not concave on all of $\mathbb R$.
<!-- bilingual-en:end -->

Hessian 处处半负定保证普通凹性，却不一定保证严格性：常函数的 Hessian 为零，每条弦上都取等。仅在一个驻点检查 Hessian 负定，使用的是[[Hessian 局部极大判据|严格局部极大判据]]，并有它自己的邻域假设。例如，$q(x)=-x^2+x^4$ 满足 $q'(0)=0$、$q''(0)=-2$，但 $q''(1)=10$；它并不在整个 $\mathbb R$ 上凹。

<!-- bilingual-en:start -->
Strict concavity ensures that there is at most one maximizer on the convex domain: if two distinct points both attained the maximum, their midpoint would have an even larger value. It does not supply a point attaining the supremum. The separate [[严格凹不保证解存在|existence boundary]] explains that remaining issue. Changing signs gives the corresponding statements for strict convexity and positive-definite Hessians.
<!-- bilingual-en:end -->

严格凹性保证凸域上至多有一个最大点：若两个不同点都取得最大值，其中点便会取得更大的值。但它并不提供一个实际达到上确界的点；剩余问题见[[严格凹不保证解存在]]。把函数变号，便得到严格凸性与正定 Hessian 的对应结论。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- The slide supplies the ordinary Hessian criterion; the strict sufficient condition follows along segments, and the quartic directly disproves necessity.
- The linked course exposition retains the detailed reasoning, assumptions, examples, and context.
- [Boyd and Vandenberghe, Convex Optimization, §§3.1.1 and 3.1.4, PDF pp.81 and 85](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=85): strict chord inequalities, positive-definite sufficiency, and the quartic counterexample to necessity. Applying these statements to $-f$ gives the concave version; the proof and displayed calculations above are supplied in full.
- [[EC400 Slides Lecture 1.pdf#page=49|LSE EC400 SOFP, PDF p.49, slide 47]] and [[01_SOFP Lecture 1 - 二次型、Taylor 展开与凹凸性#7.4 二阶判据：一点的曲率与处处的曲率|the permanent course exposition, §7.4]] provide the ordinary criterion and the course quadratic example.
<!-- bilingual-en:end -->

- [[EC400 Slides Lecture 1.pdf#page=49|EC400 SOFP 原材料，PDF p.49]]：课件提供普通凹性的 Hessian 判据；负定的充分性沿线段推导，严格凹不要求处处负定由 −x⁴ 及其导数直接验证。
- [[01_SOFP Lecture 1 - 二次型、Taylor 展开与凹凸性#7.4 二阶判据：一点的曲率与处处的曲率|SOFP Lecture 1 对应讲解]]：保留完整推导、条件、例子及课程语境。
- [Boyd 与 Vandenberghe《Convex Optimization》§§3.1.1、3.1.4，PDF81、85页](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=85)：严格弦不等式、正定的充分性以及否定必要性的四次反例。对 $-f$ 应用即可得到凹的版本；正文完整写出证明及所列计算。

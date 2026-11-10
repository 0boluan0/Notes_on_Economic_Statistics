---
aliases:
  - "Hessian saddle-point test"
  - "Indefinite Hessian saddle criterion"
student_os: knowledge-atom
atom_id: CALC-MV-011
atom_set: multivariable-differentiation
atom_type: diagnostic-rule
status: source-checked
requires:
  - "[[Hessian矩阵]]"
  - "[[多元Taylor近似]]"
related:
  - "[[Hessian 局部极小判据]]"
  - "[[Hessian 局部极大判据]]"
  - "[[半定Hessian无结论]]"
part_of:
  - "[[多元微分.canvas|多元微分]]"
---

<!-- bilingual-en:start -->
# The Hessian saddle-point test
<!-- bilingual-en:end -->

*Hessian 鞍点判据*

<!-- bilingual-en:start -->
A **saddle point** is a stationary point with both higher and lower function values arbitrarily nearby. Imagine standing on a mountain pass: one path goes uphill and another goes downhill. The Hessian test detects this situation when the function bends upward in one direction and downward in another.
<!-- bilingual-en:end -->

**鞍点**是一种驻点：无论把周围范围缩得多小，都能找到比它函数值更高和更低的位置。可以想象站在山口，一条路向上，另一条路向下。Hessian 判据通过“一种方向向上弯、另一种方向向下弯”识别这种情形。

<!-- bilingual-en:start -->
For $f(x,y)=x^2-y^2$, the origin has value $0$. Moving to $(t,0)$ gives $t^2>0$ and moving to $(0,t)$ gives $-t^2<0$ for every $t\ne0$. The origin is therefore neither a local minimum nor a local maximum. The general test explains why this pattern survives small higher-order corrections.
<!-- bilingual-en:end -->

例如 $f(x,y)=x^2-y^2$ 在原点的值为 $0$。只要 $t\ne0$，移动到 $(t,0)$ 就得到 $t^2>0$，移动到 $(0,t)$ 就得到 $-t^2<0$。所以原点既不是局部极小，也不是局部极大。一般判据解释的是：为什么加入足够小的高阶误差后，这种正负变化仍然存在。

<!-- bilingual-en:start -->
Let $f:U\to\mathbb R$, where $U\subseteq\mathbb R^n$ is [[开集|open]] and $n\ge2$ is a finite integer. Assume that all partial derivatives through order two exist and are continuous near $a\in U$; this is **$C^2$ near $a$**. The [[全微分#实值函数、梯度与全微分|gradient]] $\nabla f(a)$ is the column of first partial derivatives. At a [[驻点|stationary point]] this column is zero, so the first-order change vanishes. Write $H=H_f(a)$ for the [[Hessian矩阵|Hessian matrix]] of second partials; it is [[Hessian对称条件|symmetric under these assumptions]].
<!-- bilingual-en:end -->

设 $f:U\to\mathbb R$，$U\subseteq\mathbb R^n$ 为[[开集]]，$n\ge2$ 是有限整数。假设直到二阶的偏导都在 $a\in U$ 附近存在且连续，这就是 **$f$ 在 $a$ 附近为 $C^2$**。[[全微分#实值函数、梯度与全微分|梯度]] $\nabla f(a)$ 是一阶偏导组成的列向量；在[[驻点]]处，这一列为零，因而一阶变化消失。记二阶偏导组成的[[Hessian矩阵]]为 $H=H_f(a)$；在这些假设下，它[[Hessian对称条件|是对称矩阵]]。

> [!important] Two opposite quadratic signs · 两个相反的二次项符号
> <!-- bilingual-en:start -->
> If $H$ is [[不定二次型|indefinite]], there are nonzero real column vectors $u,v$ with opposite signs of its [[二次型|quadratic form]]. Then $a$ is a saddle point.
> <!-- bilingual-en:end -->
>
> 若 $H$ [[不定二次型|不定]]，就存在两个非零实列向量 $u,v$，使其[[二次型]]具有相反符号。此时 $a$ 是鞍点。
>
> $$u^THu>0,\qquad v^THv<0.$$

<!-- bilingual-en:start -->
Here $T$ denotes [[矩阵转置|transpose]]. These signs describe the quadratic form, not the signs of individual matrix entries. By the [[实对称矩阵谱定理|spectral theorem]], a real symmetric matrix is indefinite exactly when it has at least one positive and at least one negative eigenvalue. The corresponding eigenvectors provide suitable $u$ and $v$. Neither vector has to be a unit vector.
<!-- bilingual-en:end -->

这里 $T$ 表示[[矩阵转置|转置]]。正负号指二次型的值，并不是逐项查看矩阵元素的符号。由[[实对称矩阵谱定理|谱定理]]，实对称矩阵不定，恰好对应至少一个正特征值和至少一个负特征值；相应特征向量就可作为 $u,v$。它们不必是单位向量。

<!-- bilingual-en:start -->
## Why the remainder cannot reverse the two signs
<!-- bilingual-en:end -->

## Taylor 展开为什么锁定两个相反符号

<!-- bilingual-en:start -->
At the stationary point, the linear term in the [[多元Taylor近似|second-order Taylor approximation]] is zero. For a small displacement $h$,
<!-- bilingual-en:end -->

在驻点处，[[多元Taylor近似|二阶 Taylor 近似]]中的一次项为零。对足够小的位移 $h$，

$$f(a+h)-f(a)=\frac12h^THh+r(h),\qquad \frac{r(h)}{\|h\|_2^2}\longrightarrow0\quad(h\to0,\ h\ne0).$$

<!-- bilingual-en:start -->
The norm $\|h\|_2$ is [[欧氏范数|Euclidean length]]. Fix the positive direction $u$ and set $h=tu$. Since $u$ is fixed and nonzero, $r(tu)/t^2\to0$. Put $q=u^THu>0$. For all sufficiently small nonzero $t$, the remainder obeys $|r(tu)|\le qt^2/4$. Hence
<!-- bilingual-en:end -->

$\|h\|_2$ 表示[[欧氏范数|欧氏长度]]。固定正号方向 $u$，令 $h=tu$。因为 $u$ 固定且非零，有 $r(tu)/t^2\to0$。记 $q=u^THu>0$；对所有足够小的非零 $t$，余项满足 $|r(tu)|\le qt^2/4$。因此

$$f(a+tu)-f(a)\ge\frac{q}{2}t^2-\frac{q}{4}t^2=\frac{q}{4}t^2>0.$$

<!-- bilingual-en:start -->
For the negative direction, put $p=-v^THv>0$. Its remainder can similarly be bounded by $pt^2/4$, giving $f(a+tv)-f(a)\le-pt^2/4<0$. Choose $|t|$ below both thresholds and small enough that $a+tu$ and $a+tv$ lie in any prescribed neighborhood of $a$. Such a choice is possible because $U$ is open and the two directions are fixed. Thus higher and lower values occur arbitrarily close; neither local extremum is possible.
<!-- bilingual-en:end -->

对负号方向，记 $p=-v^THv>0$。同样可把对应余项控制在 $pt^2/4$ 以内，于是 $f(a+tv)-f(a)\le-pt^2/4<0$。把 $|t|$ 取到两个阈值以下，并进一步缩小，使 $a+tu$、$a+tv$ 落在任意指定的邻域内即可。开域与两条固定方向保证这样的选择可行。因此高于和低于 $f(a)$ 的值会任意接近 $a$，排除了两类局部极值。

<!-- bilingual-en:start -->
## The basic example, now read as a matrix test
<!-- bilingual-en:end -->

## 最小例子

<!-- bilingual-en:start -->
For $f(x,y)=x^2-y^2$, compute the gradient and Hessian:
<!-- bilingual-en:end -->

对 $f(x,y)=x^2-y^2$，计算梯度与 Hessian：

$$\nabla f(x,y)=\begin{pmatrix}2x\\-2y\end{pmatrix},\qquad H_f(x,y)=\begin{pmatrix}2&0\\0&-2\end{pmatrix}.$$

<!-- bilingual-en:start -->
At $(0,0)$ the gradient is zero. Choose $u=(1,0)^T$ and $v=(0,1)^T$: their quadratic values are $2$ and $-2$. Taylor has no remainder for this quadratic function, so its predicted changes are exactly $t^2$ and $-t^2$, as in the opening example.
<!-- bilingual-en:end -->

在 $(0,0)$ 处梯度为零。取 $u=(1,0)^T$、$v=(0,1)^T$，两个二次型值分别为 $2$ 与 $-2$。这个二次函数的 Taylor 余项为零，因此预测变化恰好就是开头的 $t^2$ 与 $-t^2$。

<!-- bilingual-en:start -->
## What the test does and does not decide
<!-- bilingual-en:end -->

## 判据边界

<!-- bilingual-en:start -->
Indefiniteness is a sufficient condition for a saddle at a $C^2$ stationary point, not a necessary one. For example, $x^2-y^4$ has a saddle at the origin although its Hessian there is $\operatorname{diag}(2,0)$; the missing sign first appears beyond the quadratic term. See [[半定Hessian无结论|the singular semidefinite boundary]] for a comparison with the same Hessian at a minimum.
<!-- bilingual-en:end -->

不定性是在 $C^2$ 驻点处推出鞍点的充分条件，不是必要条件。例如 $x^2-y^4$ 在原点也是鞍点，但其 Hessian 为 $\operatorname{diag}(2,0)$；缺少的负变化在二次项之后才出现。[[半定Hessian无结论]]把它与具有同一 Hessian 的极小点作了比较。

<!-- bilingual-en:start -->
Likewise, “not positive definite” does not mean “indefinite”: a negative-definite Hessian at a stationary point gives a [[Hessian 局部极大判据|strict local maximum]]. Positive definiteness gives a [[Hessian 局部极小判据|strict local minimum]]. Check stationarity before using this classification. The result here concerns unconstrained interior behavior: restricting $x^2-y^2$ to the line $y=0$ turns its origin into a minimum along that constraint.
<!-- bilingual-en:end -->

同样，“不是正定”不等于“不定”：驻点处负定 Hessian 给出[[Hessian 局部极大判据|严格局部极大]]，正定给出[[Hessian 局部极小判据|严格局部极小]]。使用这套分类前要先检查驻点条件。本卡讨论无约束内点行为；若把 $x^2-y^2$ 限制在 $y=0$ 上，原点沿该约束就成了极小点。

> [!question]- Why are two eigenvalue signs enough? · 为什么两个特征值符号就够？
> <!-- bilingual-en:start -->
> At a $C^2$ stationary point, a positive and a negative Hessian eigenvalue give two fixed directions. Their nonzero quadratic coefficients dominate the remainder for sufficiently small steps. Only these two directions are needed to disprove both local extrema; no claim about the function far away is required.
> <!-- bilingual-en:end -->
>
> 在 $C^2$ 驻点处，Hessian 的一个正特征值和一个负特征值提供两条固定方向。足够小步长下，严格非零的二次项系数压过余项。这两条方向已经足以排除两类局部极值，不需要对远处的函数行为作任何判断。

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

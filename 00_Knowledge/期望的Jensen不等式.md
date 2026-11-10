---
student_os: knowledge-atom
atom_id: PROB-CONC-012
aliases:
  - Jensen inequality
  - Jensen's inequality
  - Jensen不等式
  - 期望的詹森不等式
atom_type: theorem
status: source-checked
---

<!-- bilingual-en:start -->
# Jensen's inequality for expectations
<!-- bilingual-en:end -->

*期望的 Jensen 不等式*

<!-- bilingual-en:start -->
There are two ways to combine an uncertain number with a function: average the number first and then apply the function, or apply the function to each possible value and then average. **Jensen's inequality** compares these two answers. For a convex function, transforming the average gives the smaller answer, or an equal one.
<!-- bilingual-en:end -->

把一个不确定的数与函数结合，有两种顺序：先平均数值，再代入函数；或者先把每个可能值代入函数，再平均结果。**Jensen 不等式**比较的就是这两个答案。对于凸函数，先平均再变换得到的答案较小，或者两者相等。

> [!example] Square before or after averaging · 平方放在平均之前还是之后
> <!-- bilingual-en:start -->
> Let the [[随机变量|random variable]] $X$ be $1$ or $3$, each with probability $1/2$. Its [[期望|expectation]], the probability-weighted average, is $E[X]=2$. For the function $\varphi(x)=x^2$, the two orders give
> <!-- bilingual-en:end -->
>
> 设[[随机变量]] $X$ 以各半概率取 $1$ 或 $3$。它的[[期望]]是按概率加权的平均值 $E[X]=2$。对函数 $\varphi(x)=x^2$，两种顺序给出
>
> $$
> \varphi(E[X])=2^2=4,
> \qquad E[\varphi(X)]=\tfrac12\,1^2+\tfrac12\,3^2=5.
> $$
>
> <!-- bilingual-en:start -->
> The upper possibility $3$ contributes more after squaring than the lower possibility $1$ removes. Averaging first hides that spread. The inequality $4\le5$ is one instance of Jensen's rule.
> <!-- bilingual-en:end -->
>
> 平方之后，较高的可能值 $3$ 增加的贡献，比较低的可能值 $1$ 减少的贡献更大；先平均会抹去这种分散。$4\le5$ 就是 Jensen 规则的一个实例。

<!-- bilingual-en:start -->
## Convexity determines the direction
<!-- bilingual-en:end -->

*凸性决定不等式的方向*

<!-- bilingual-en:start -->
A function has [[凹凸性|convexity]] when its value at a weighted average of two inputs is no greater than the same weighted average of its values. Geometrically, the graph lies below the chord joining any two graph points. This definition allows corners: differentiability is not required. Jensen extends that two-input comparison to a probability-weighted average over a random variable's whole distribution.
<!-- bilingual-en:end -->

函数的[[凹凸性|凸性]]是指：把两个输入加权平均后代入函数，结果不大于两个函数值的同权重平均。几何上，函数图像不高于连接任意两个图像点的弦。定义允许尖角，不要求可导。Jensen 把这个双输入比较推广到随机变量整个分布上的概率加权平均。

> [!important] Finite expectation version · 有限期望版本
> <!-- bilingual-en:start -->
> Let $X$ be a real random variable with $E|X|<\infty$. Let $\varphi:\mathbb R\to\mathbb R$ be a finite-valued convex function, and also require $E|\varphi(X)|<\infty$. Then
> <!-- bilingual-en:end -->
>
> $$\boxed{\varphi(E[X])\le E[\varphi(X)].}$$
>
> <!-- bilingual-en:start -->
> Here $E[X]$ is a finite number; $\varphi(X)$ means applying the same function to each outcome. The two integrability assumptions make both sides finite. For a concave function with the same assumptions, the inequality reverses, because its negative is convex.
> <!-- bilingual-en:end -->
>
> 设实随机变量 $X$ 满足 $E|X|<\infty$，$\varphi:\mathbb R\to\mathbb R$ 是有限值凸函数，并要求 $E|\varphi(X)|<\infty$，就有上式。$E[X]$ 是一个有限数，$\varphi(X)$ 表示逐个结果应用同一个函数；两项可积性条件使两边都是有限值。对满足相同条件的凹函数，不等号反向，因为它的相反数是凸函数。

<!-- bilingual-en:start -->
For discrete values $x_i$ with probabilities $p_i$, the formula reads $\varphi(\sum_i p_i x_i)\le\sum_i p_i\varphi(x_i)$. The weights are nonnegative and sum to one. There is only one random variable here: no independence assumption is involved.
<!-- bilingual-en:end -->

若离散取值 $x_i$ 的概率为 $p_i$，公式就是 $\varphi(\sum_i p_i x_i)\le\sum_i p_i\varphi(x_i)$。权重非负且总和为一。这里讨论的只有一个随机变量，不涉及独立假设。

<!-- bilingual-en:start -->
## A line underneath the curve explains the result
<!-- bilingual-en:end -->

*用曲线下方的一条直线理解证明*

<!-- bilingual-en:start -->
Write $m=E[X]$. A finite convex function on the real line has a **supporting line** at $m$: an affine function $\ell(x)=\varphi(m)+a(x-m)$ that touches the graph at $m$ and stays below it everywhere. The real number $a$ is its slope. If $\varphi$ is differentiable there, its tangent supplies this line; at a corner, an appropriate slope between the one-sided slopes works.
<!-- bilingual-en:end -->

记 $m=E[X]$。实数轴上的有限值凸函数在 $m$ 处有一条**支撑直线**：仿射函数 $\ell(x)=\varphi(m)+a(x-m)$ 在 $m$ 处接触图像，并始终位于图像下方，实数 $a$ 是它的斜率。若该处可导，可以用切线；若有尖角，可以在两侧斜率之间选合适的斜率。

<!-- bilingual-en:start -->
Since $\varphi(X)\ge\ell(X)$ for every input, taking expectations preserves that order. [[期望线性性|Linearity of expectation]] then makes the average displacement $X-m$ vanish:
<!-- bilingual-en:end -->

由于每个输入都满足 $\varphi(X)\ge\ell(X)$，取期望会保留大小关系；再用[[期望线性性]]，平均偏移 $X-m$ 正好消失：

$$
E[\varphi(X)]\ge E[\ell(X)]
=\varphi(m)+a(E[X]-m)=\varphi(m).
$$

<!-- bilingual-en:start -->
For an affine function itself, the inequality is always equality. For the square function, the gap $E[X^2]-(E[X])^2$ is exactly the [[方差|variance]] when $E[X^2]<\infty$; it is zero precisely when $X$ is constant almost surely. Convexity gives a weak inequality, not a promise that every example has a strictly positive gap.
<!-- bilingual-en:end -->

如果函数本身就是仿射函数，不等式始终取等号。对于平方函数，在 $E[X^2]<\infty$ 时，差值 $E[X^2]-(E[X])^2$ 正是[[方差]]；它恰在 $X$ 几乎必然为常数时为零。凸性给出的是允许等号的不等式，不保证每个例子都有严格正的差距。

> [!warning] Check the transform as well as the input · 原变量与变换后的变量都要检查
> <!-- bilingual-en:start -->
> $E|X|<\infty$ does not imply $E[X^2]<\infty$, or integrability after every convex transform. The finite theorem above requires both moment checks; see [[矩存在性的使用边界|moment-existence conditions]]. It also uses a function defined on all of $\mathbb R$. For a function defined only on a smaller interval, first check its domain for the random inputs and their mean; an undefined function value cannot enter the formula.
> <!-- bilingual-en:end -->
>
> $E|X|<\infty$ 不推出 $E[X^2]<\infty$，也不保证任意凸变换后仍可积。上面的有限版本要求检查这两项条件，见[[矩存在性的使用边界]]。它还使用定义在整个 $\mathbb R$ 上的函数；函数若只定义在更小的区间上，应先检查随机输入及其均值是否位于定义域，不能把未定义的函数值代入公式。

<!-- bilingual-en:start -->
The broader distinction is that [[期望与非线性变换不可交换|expectation and nonlinear transformation generally do not commute]]. Jensen adds a useful direction when curvature is known. If the averaging must use specified information rather than the whole population, use [[条件Jensen不等式|conditional Jensen's inequality]].
<!-- bilingual-en:end -->

更一般的区别是[[期望与非线性变换不可交换|期望与非线性变换通常不能交换]]；知道凹凸性后，Jensen 进一步给出比较方向。若平均时需要使用指定信息，而不是对全体结果一起平均，则使用[[条件Jensen不等式]]。

## 来源与核验

<!-- bilingual-en:start -->
- [Durrett, *Probability: Theory and Examples*, 5th ed., Theorem 1.5.1, printed p. 25 / PDF p. 33; §1.6, printed p. 28](https://math.duke.edu/~rtd/PTE/PTE5_011119.pdf#page=33): convexity, integrability of the input and transform, probability-measure Jensen, and the supporting-line proof. Section 1.6 identifies the probability integral with expectation. The two-point calculation, concave sign reversal and square-gap identity are derived directly above.
<!-- bilingual-en:end -->

- Durrett 第五版定理 1.5.1（印刷第 25 页／PDF 第 33 页）核对凸性、原变量与变换的可积性、概率测度版本及支撑直线证明；第 1.6 节把概率积分解释为期望。两点例子、凹函数反向和平方差值在正文直接推算。

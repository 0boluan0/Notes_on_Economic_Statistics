---
aliases:
  - Multivariable Taylor approximation
  - Second-order multivariable Taylor expansion
student_os: knowledge-atom
atom_id: CALC-MV-009
atom_set: multivariable-differentiation
atom_type: approximation-theorem
status: source-checked
requires:
  - "[[全微分]]"
  - "[[Hessian矩阵]]"
leads_to:
  - "[[Hessian 局部极小判据]]"
  - "[[Hessian 局部极大判据]]"
  - "[[Hessian 鞍点判据]]"
  - "[[半定Hessian无结论]]"
part_of:
  - "[[多元微分.canvas|多元微分]]"
---

<!-- bilingual-en:start -->
# Multivariable Taylor approximation
<!-- bilingual-en:end -->

*多元 Taylor 近似*

<!-- bilingual-en:start -->
A **multivariable Taylor approximation** predicts a function value after several inputs change together. It starts from a known value, adds the linear response, then corrects for how the rates themselves change. The second-order formula includes interactions between inputs: changing two coordinates together can have an effect that separate coordinate changes miss.
<!-- bilingual-en:end -->

**多元 Taylor 近似**预测多个输入一起改变后的函数值。它从已知值出发，加入线性响应，再根据变化率本身的改变作修正。二阶公式还包含输入之间的相互作用：两个坐标一起变化，可能出现分别改变各坐标时看不到的效果。

<!-- bilingual-en:start -->
The [[全微分|total differential]] supplies one linear approximation valid for all small input vectors. Its scalar-valued version uses the [[全微分#实值函数、梯度与全微分|gradient]] $\nabla f$, a column of first partials. The [[Hessian矩阵|Hessian]] $H_f=\nabla^2f$ collects second partials and supplies the quadratic correction. We use real column vectors, and $T$ denotes [[矩阵转置|transpose]].
<!-- bilingual-en:end -->

[[全微分]]给出对所有小输入向量都适用的同一个线性近似。实值函数用[[全微分#实值函数、梯度与全微分|梯度]] $\nabla f$ 表示它，梯度是一阶偏导组成的列向量；[[Hessian矩阵]] $H_f=\nabla^2f$ 收集二阶偏导，提供二次修正。这里使用实列向量，上标 $T$ 表示[[矩阵转置|转置]]。

> [!important] A local quadratic model · 局部二次模型
> <!-- bilingual-en:start -->
> Let $f:U\to\mathbb R$, with $U\subseteq\mathbb R^n$ [[开集|open]], $n\ge1$ an integer, and $a\in U$. Assume $f$ is $C^2$ on a neighborhood of $a$: all partial derivatives through order two exist and are continuous there. For a small displacement $h$,
>
> $$f(a+h)=f(a)+\nabla f(a)^Th+\frac12h^TH_f(a)h+R_2(h),\qquad R_2(h)=o(\|h\|_2^2).$$
>
> The vector $h$ is the input change, not the new input $a+h$. Every derivative in the polynomial is evaluated at the fixed centre $a$.
> <!-- bilingual-en:end -->
>
> 设 $f:U\to\mathbb R$，其中 $U\subseteq\mathbb R^n$ 为[[开集]]，$n\ge1$ 为整数，$a\in U$。假设 $f$ 在 $a$ 的某个邻域内为 $C^2$，即直到二阶的偏导都存在且连续。对小位移 $h$，上式给出函数值的二阶展开。$h$ 是输入的改变量，不是新输入 $a+h$；多项式中每个导数都在固定中心 $a$ 取值。

<!-- bilingual-en:start -->
Here $\|h\|_2$ is [[欧氏范数|Euclidean length]]. The small-$o$ statement means $|R_2(h)|/\|h\|_2^2\to0$ for nonzero vectors approaching zero from **any direction**, including changing directions. Equivalently, for every $\varepsilon>0$, all sufficiently small $h$ satisfy $|R_2(h)|\le\varepsilon\|h\|_2^2$. At $h=0$, the formula is exact and $R_2(0)=0$; no quotient is needed.
<!-- bilingual-en:end -->

$\|h\|_2$ 是[[欧氏范数|欧氏长度]]。小 $o$ 表示非零向量从**任意方向**趋于零时，$|R_2(h)|/\|h\|_2^2\to0$，其中方向也可以随位移改变。等价地，给定任意 $\varepsilon>0$，所有充分小的 $h$ 都满足 $|R_2(h)|\le\varepsilon\|h\|_2^2$。在 $h=0$ 时，等式精确且 $R_2(0)=0$，不必组成比值。

<!-- bilingual-en:start -->
## Separate the two accuracy scales
<!-- bilingual-en:end -->

## 一阶与二阶近似回答不同精度的问题

$$f(a+h)-f(a)=\nabla f(a)^Th+o(\|h\|_2).$$

<!-- bilingual-en:start -->
The first-order statement makes the error small relative to displacement length. Adding the Hessian term gives an error small relative to its square. This is a local limiting guarantee, not an assertion that a quadratic estimate must beat a linear estimate at every chosen finite step. At a stationary point, where the gradient is zero, the quadratic term is the next candidate for the leading change; if it also vanishes, higher-order behavior can matter.
<!-- bilingual-en:end -->

一阶结论让误差相对于位移长度变小；加入 Hessian 项后，剩余误差相对于位移长度的平方变小。这是局部极限保证，不是说在每个指定的有限步长上，二次估计都必然胜过一次估计。驻点的梯度为零，二次项便成为下一项候选主导变化；若它也为零，更高阶行为可能起决定作用。

<!-- bilingual-en:start -->
## Compute the value, the two corrections and the error
<!-- bilingual-en:end -->

## 完整例子：线性项、二次项和真实误差

<!-- bilingual-en:start -->
Take $f(x,y)=x^2y$, centre $a=(1,2)^T$, and displacement $h=(0.01,-0.02)^T$. The base value is $f(a)=2$, and
<!-- bilingual-en:end -->

取 $f(x,y)=x^2y$，中心 $a=(1,2)^T$，位移 $h=(0.01,-0.02)^T$。基准值为 $f(a)=2$，并且

$$\nabla f(a)=\begin{pmatrix}4\\1\end{pmatrix},\qquad H_f(a)=\begin{pmatrix}4&2\\2&0\end{pmatrix}.$$

$$\nabla f(a)^Th=0.02,\qquad \frac12h^TH_f(a)h=-0.0002.$$

<!-- bilingual-en:start -->
The quadratic model predicts the increment $0.0198$, hence the value $2.0198$. Direct evaluation gives $f(1.01,1.98)=2.019798$, so the actual increment is $0.019798$ and the signed error is $-0.000002$. Expanding for a general $h=(h_1,h_2)^T$ shows exactly where it comes from:
<!-- bilingual-en:end -->

二次模型预测增量为 $0.0198$，所以预测函数值为 $2.0198$。直接计算得到 $f(1.01,1.98)=2.019798$，真实增量为 $0.019798$，带符号误差为 $-0.000002$。对一般 $h=(h_1,h_2)^T$ 展开，就能看清误差的来源：

$$(1+h_1)^2(2+h_2)=2+4h_1+h_2+2h_1^2+2h_1h_2+h_1^2h_2.$$

<!-- bilingual-en:start -->
The first two corrections include all terms through degree two, leaving $R_2(h)=h_1^2h_2$. Since $|h_1|,|h_2|\le\|h\|_2$, we have $|R_2(h)|\le\|h\|_2^3$, so the ratio to $\|h\|_2^2$ tends to zero. This cubic bound follows from this particular polynomial; it is stronger than the general $C^2$ conclusion.
<!-- bilingual-en:end -->

前两项修正包含所有次数不超过二的项，留下 $R_2(h)=h_1^2h_2$。因为 $|h_1|,|h_2|\le\|h\|_2$，有 $|R_2(h)|\le\|h\|_2^3$，所以它除以 $\|h\|_2^2$ 后趋于零。这个三次界来自此处的具体多项式，比一般的 $C^2$ 结论更强。

<!-- bilingual-en:start -->
## The entire segment shrinks, even when the parameter equals one
<!-- bilingual-en:end -->

## 沿线段推导时为什么可以取参数一

<!-- bilingual-en:start -->
Fix a sufficiently small $h$ so the full segment $a+th$, $0\le t\le1$, lies in the $C^2$ neighborhood. Define $g_h(t)=f(a+th)$. A different displacement gives a different one-variable function. The [[Hessian方向二阶导数|line-derivative formula]] yields
<!-- bilingual-en:end -->

固定充分小的 $h$，使整条线段 $a+th$（$0\le t\le1$）都留在 $C^2$ 邻域内。定义 $g_h(t)=f(a+th)$；换一个位移，就得到另一个一元函数。由[[Hessian方向二阶导数|沿线导数公式]]，

$$g_h'(0)=\nabla f(a)^Th,\qquad g_h''(t)=h^TH_f(a+th)h.$$

<!-- bilingual-en:start -->
Because $g_h$ is $C^2$ on the segment, [[分部积分|integration by parts]] and the [[Newton-Leibniz公式|fundamental theorem of calculus]] give the exact identity
<!-- bilingual-en:end -->

因为 $g_h$ 在这段上为 $C^2$，由[[分部积分]]和[[Newton-Leibniz公式]]，得到精确恒等式

$$\begin{aligned}\int_0^1(1-t)g_h''(t)\,dt&=[(1-t)g_h'(t)]_0^1+\int_0^1g_h'(t)\,dt\\&=-g_h'(0)+g_h(1)-g_h(0).\end{aligned}$$

<!-- bilingual-en:start -->
Add and subtract $H_f(a)$ in the integral. Since $\int_0^1(1-t)\,dt=1/2$, this gives both the coefficient one half and the remaining error:
<!-- bilingual-en:end -->

在积分中加减 $H_f(a)$。由于 $\int_0^1(1-t)\,dt=1/2$，这同时解释了二分之一的系数，并得到余项：

$$R_2(h)=\int_0^1(1-t)h^T[H_f(a+th)-H_f(a)]h\,dt.$$

<!-- bilingual-en:start -->
To bound it, the matrix [[谱范数|induced Euclidean norm]] satisfies $\|Bh\|_2\le\|B\|_2\|h\|_2$. Combined with [[Cauchy–Schwarz不等式|Cauchy–Schwarz]], this gives $|h^TBh|\le\|B\|_2\|h\|_2^2$. Therefore, for $h\ne0$,
<!-- bilingual-en:end -->

为了估计它，矩阵的[[谱范数|诱导二范数]]满足 $\|Bh\|_2\le\|B\|_2\|h\|_2$。结合[[Cauchy–Schwarz不等式]]，得到 $|h^TBh|\le\|B\|_2\|h\|_2^2$。因此对 $h\ne0$，

$$\frac{|R_2(h)|}{\|h\|_2^2}\le\frac12\sup_{0\le t\le1}\|H_f(a+th)-H_f(a)\|_2\longrightarrow0.$$

<!-- bilingual-en:start -->
The [[上确界|supremum]] takes one bound over the whole segment. Hessian continuity at $a$ makes it tend to zero: every point satisfies $\|th\|_2\le\|h\|_2$, so one small neighborhood works for every $t$ and every direction of $h$. Thus evaluating $g_h$ at $t=1$ is legitimate. The shrinking quantity is $h$, not the number one. Looking at each fixed line separately, with a different error threshold for each direction, would not establish this uniform conclusion.
<!-- bilingual-en:end -->

[[上确界]]给出整条线段上的统一上界。Hessian 在 $a$ 处连续，使它趋于零：每个点都满足 $\|th\|_2\le\|h\|_2$，所以同一个小邻域同时控制所有 $t$ 和所有 $h$ 的方向。因此可以在 $t=1$ 评价 $g_h$；真正缩小的是 $h$，不是数字一。若只分别检查每条固定直线，允许每个方向使用不同的误差阈值，就还没有证明这里的统一结论。

> [!note] What would justify a cubic bound? · 什么条件能保证三次误差界？
> <!-- bilingual-en:start -->
> If a finite $L\ge0$ satisfies $\|H_f(a+th)-H_f(a)\|_2\le Lt\|h\|_2$ on every sufficiently short segment, the same integral proof gives
>
> $$|R_2(h)|\le L\|h\|_2^3\int_0^1t(1-t)\,dt=\frac L6\|h\|_2^3.$$
>
> For example, continuous third partials on a neighborhood supply such a local bound after shrinking to a closed ball inside it. Mere $C^2$ continuity does not supply a cubic rate: $f(x)=|x|^{5/2}$ is $C^2$ on $\mathbb R$: for $x\ne0$, $f^{\prime\prime}(x)=\tfrac{15}{4}\sqrt{|x|}$, extending continuously with $f^{\prime\prime}(0)=0$. Its quadratic Taylor polynomial at zero is zero, but $|f(x)|/|x|^3=|x|^{-1/2}$ is unbounded. An actual numerical error certificate still needs a known bound over the whole chosen segment.
> <!-- bilingual-en:end -->
>
> 如果某个有限 $L\ge0$ 在每条足够短的线段上都满足 $\|H_f(a+th)-H_f(a)\|_2\le Lt\|h\|_2$，同一个积分证明就给出上述 $L\|h\|_2^3/6$ 的界。例如，邻域中的三阶偏导连续，缩到内部闭球后即可得到这种局部界。单纯 $C^2$ 连续并不保证三次速率：$f(x)=|x|^{5/2}$ 在 $\mathbb R$ 上为 $C^2$：对 $x\ne0$，$f^{\prime\prime}(x)=\tfrac{15}{4}\sqrt{|x|}$，并连续延伸为 $f^{\prime\prime}(0)=0$。它在零点的二次 Taylor 多项式为零，但 $|f(x)|/|x|^3=|x|^{-1/2}$ 无界。要为具体数值误差提供保证，仍需掌握整条所选线段上的实际界。

<!-- bilingual-en:start -->
## Use the quadratic term for local classification
<!-- bilingual-en:end -->

## 从 Taylor 到局部极值分类

<!-- bilingual-en:start -->
At an interior stationary point, a [[正定矩阵|positive-definite]] Hessian leads to the [[Hessian 局部极小判据|strict local-minimum criterion]], a negative-definite Hessian to the [[Hessian 局部极大判据|strict local-maximum criterion]], and an indefinite Hessian to the [[Hessian 鞍点判据|saddle criterion]]. These results use the quadratic term together with the uniform remainder, not just the vanishing gradient. If the Hessian is semidefinite but not definite, its quadratic form vanishes in some nonzero directions; [[半定Hessian无结论|second-order information alone does not complete the classification]].
<!-- bilingual-en:end -->

在内部驻点处，[[正定矩阵|正定]] Hessian 导向[[Hessian 局部极小判据|严格局部极小判据]]，负定 Hessian 导向[[Hessian 局部极大判据|严格局部极大判据]]，不定 Hessian 导向[[Hessian 鞍点判据|鞍点判据]]。这些结论同时使用二次项与统一余项，不能只从梯度为零得出。若 Hessian 半定但非定，其二次型在某些非零方向上为零，[[半定Hessian无结论|仅凭二阶信息便无法完成分类]]。

<!-- bilingual-en:start -->
All these conclusions concern a neighborhood of $a$. Information at one point cannot rule out larger or smaller values far away. Global conclusions require additional arguments, such as suitable [[凹凸性|convexity or concavity]] on the full domain, or direct comparison.
<!-- bilingual-en:end -->

这些结论都只涉及 $a$ 的邻域。一点处的信息不能排除远处还有更大或更小的值。全局结论需要额外论证，例如整个定义域上适用的[[凹凸性|凸性或凹性]]，或者直接比较。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [MIT 18.S096, Second Derivatives, Bilinear Maps, and Hessian Matrices, §12.1, PDF pp.1–2 / print 87–88](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/mit18_s096iap23_lec12.pdf#page=1): the Hessian quadratic correction and second-order small-$o$ form. This card gives the full integral-remainder proof with stated $C^2$ hypotheses.
- [Lebl, Basic Analysis II, §8.6, PDF pp.60–63](https://www.jirka.org/ra/realanal2.pdf#page=60): continuous second partials and the need for a neighborhood estimate rather than separate line tests; exercise 8.6.7 points to the uniform issue.
- [[01_Math/08_EC400_MathsCamp/2026_Course_Materials/01_Revision_Mathematics/Notes/By_Topic/Revision Maths Notes 7, Working with Multivariate Calculus.pdf#page=13|LSE EC400 Notes 7, §7.9, PDF pp.13–15 / print 71–73]]: the course quadratic formula and line restriction.
- [[EC400 Slides Lecture 1.pdf#page=41|EC400 SOFP Lecture 1, PDF p.41]] and [[01_SOFP Lecture 1 - 二次型、Taylor 展开与凹凸性#5.4 为什么沿直线可以在参数一处取值|permanent course exposition §5.4]]: the quadratic model, integral remainder and small-displacement interpretation. The exact polynomial error and the $C^2$ noncubic example are computed directly.
<!-- bilingual-en:end -->

- MIT 18.S096 §12.1，PDF1–2页／印本87–88页：Hessian 二次修正及二阶小 $o$ 形式。本文明确 $C^2$ 前提，并完整推导积分余项。
- Lebl《Basic Analysis II》§8.6，PDF60–63页：连续二阶偏导，以及邻域估计不能被各条直线分别检验替代；习题8.6.7特别指出统一控制问题。
- LSE EC400 Notes 7 §7.9，PDF13–15页／印本71–73页：课程二次公式与沿线限制。
- [[EC400 Slides Lecture 1.pdf#page=41|EC400 SOFP Lecture 1 原材料]]与[[01_SOFP Lecture 1 - 二次型、Taylor 展开与凹凸性#5.4 为什么沿直线可以在参数一处取值|永久课程展开 §5.4]]：二次模型、积分余项及小位移解释；多项式精确误差与 $C^2$ 非三次误差例另行直接核算。

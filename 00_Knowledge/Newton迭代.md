---
aliases:
  - "Newton iteration"
  - "Newton–Raphson method"
  - "Newton法"
  - "牛顿法"
student_os: knowledge-atom
atom_id: CS-NR-004
atom_set: numerical-root-finding
atom_type: algorithm-definition
status: needs-review
requires:
  - "[[导数]]"
related:
  - "[[线性近似]]"
leads_to:
  - "[[Newton局部收敛]]"
  - "[[Newton失效边界]]"
  - "[[混合求根]]"
part_of:
  - "[[数值求根.canvas|数值求根]]"
  - "[[导数的应用.canvas]]"
---

<!-- bilingual-en:start -->
# Newton iteration
<!-- bilingual-en:end -->
*Newton 迭代*

<!-- bilingual-en:start -->
A **root** of $f$ is an input where $f(x)=0$. Newton iteration tries to find one by repeatedly replacing the curve near the current guess with its tangent line, solving that easier linear equation, and trying again from the resulting input. For example, finding the positive root of $f(x)=x^2-2$ means finding $\sqrt2$. Each new input is a candidate, not yet a certified root.
<!-- bilingual-en:end -->

函数的**根**是使 $f(x)=0$ 的输入。Newton 迭代每轮先把当前猜测附近的曲线换成切线，求这条直线的零点，再从新输入重复这个过程。例如，求 $f(x)=x^2-2$ 的正根就是求 $\sqrt2$。每次产生的是候选点，还不是已经得到证明的根。

<!-- bilingual-en:start -->
The [[导数|derivative]] $f'(x_k)$ is the tangent slope; $f(x_k)$ is the current height above or below zero. The index $k=0,1,2,\ldots$ counts iterations, so $x_0$ is the starting guess. A [[线性近似|linear approximation]] uses these two pieces of information near $x_k$. The mathematical statements here use exact function values; machine-evaluated zeros need the separate checks in [[数值迭代停止条件|stopping criteria]].
<!-- bilingual-en:end -->

[[导数]] $f'(x_k)$ 给出切线斜率，$f(x_k)$ 是当前函数值距离零的有向高度。下标 $k=0,1,2,\ldots$ 数迭代轮次，所以 $x_0$ 是初始猜测。[[线性近似]]把这两项信息用于 $x_k$ 附近。这里的数学推导使用精确函数值；机器计算得到的零仍须按[[数值迭代停止条件]]核验。

> [!abstract] The central idea · 核心思路
> <!-- bilingual-en:start -->
> Newton iteration uses the tangent at the current point $x_k$ as a local linear model and takes the tangent's zero as the next root approximation: $x_{k+1}=x_k-f(x_k)/f'(x_k)$. If $f(x_k)=0$, return the current root before dividing; otherwise the update can be computed only when $f'(x_k)\ne0$.
> <!-- bilingual-en:end -->
>
> Newton 迭代在当前点 $x_k$ 用原函数的切线作局部线性模型，并把这条切线的零点作为下一次求根近似：$x_{k+1}=x_k-f(x_k)/f'(x_k)$。若 $f(x_k)=0$，应先直接返回当前根；否则只有在 $f'(x_k)\ne0$ 时才能计算这一步。

<!-- bilingual-en:start -->
## Solve the local tangent model
<!-- bilingual-en:end -->

## 更新式来自一阶局部模型

<!-- bilingual-en:start -->
Write a first-order Taylor approximation near $x_k$. Here $s$ is the proposed change in the input:
<!-- bilingual-en:end -->

在 $x_k$ 附近写一阶 Taylor 近似

$$
f(x_k+s)\approx f(x_k)+f'(x_k)s.
$$

<!-- bilingual-en:start -->
Set the right-hand side to zero and solve for the step:
<!-- bilingual-en:end -->

令右侧等于 0，得到

$$
s_k=-\frac{f(x_k)}{f'(x_k)},
\qquad
x_{k+1}=x_k-\frac{f(x_k)}{f'(x_k)},
$$

<!-- bilingual-en:start -->
This requires $f'(x_k)\ne0$. The equation solves for a zero of the tangent, not of the original function; it is a useful correction only when the local model is sufficiently accurate.
<!-- bilingual-en:end -->

前提是 $f'(x_k)\ne0$。这一步求的是切线的零点，不是原函数零点；只有当局部线性模型足够好时，它才是有用的修正。

<!-- bilingual-en:start -->
Check $f(x_k)=0$ first. If the current input is already a root, neither the quotient nor a further step is needed. This matters especially at a multiple root where the derivative also vanishes. Only at a nonroot must the denominator be nonzero before another candidate is generated.
<!-- bilingual-en:end -->

实际顺序应先检查 $f(x_k)=0$。若当前点已经是根，就不需要、也不应该再计算 $f(x_k)/f'(x_k)$；这对导数同样为 0 的重根尤其重要。只有在当前点尚不是根时，才要求分母非零并生成下一候选。

<!-- bilingual-en:start -->
For $x^2-2=0$, the update becomes
<!-- bilingual-en:end -->

例如求 $x^2-2=0$ 时，

$$
x_{k+1}=\frac12\left(x_k+\frac{2}{x_k}\right).
$$

<!-- bilingual-en:start -->
Starting at $x_0=1.5$ gives $x_1\approx1.4166667$ and $x_2\approx1.4142157$. This illustrates local speed, but is not a convergence proof for every starting guess.
<!-- bilingual-en:end -->

从 $x_0=1.5$ 出发，$x_1\approx1.4166667$，$x_2\approx1.4142157$。这个例子展示了局部速度，却不能替所有初值提供收敛证明。

> [!example] Follow the first two steps · 手算两步
> <!-- bilingual-en:start -->
> At $x_0=3/2$, the height is $f(x_0)=1/4$ and the slope is $f'(x_0)=3$. The step is $s_0=-(1/4)/3=-1/12$, giving $x_1=17/12$. Recompute at that input: $f(17/12)=1/144$ and $f'(17/12)=17/6$. Thus $s_1=-1/408$ and $x_2=577/408$. Function-value units divided by slope units give input units. Reusing the old derivative would define a different iteration.
> <!-- bilingual-en:end -->
>
> 在 $x_0=3/2$ 处，函数高度为 $f(x_0)=1/4$，斜率为 $f'(x_0)=3$。所以步长 $s_0=-(1/4)/3=-1/12$，得到 $x_1=17/12$。在新输入处重新计算：$f(17/12)=1/144$、$f'(17/12)=17/6$，于是 $s_1=-1/408$，$x_2=577/408$。函数值单位除以斜率单位，得到的正是输入增量单位。若一直复用旧导数，就已经换成另一种迭代。

<!-- bilingual-en:start -->
## What the update does and what needs another argument
<!-- bilingual-en:end -->

## 收敛与失效由独立原子判断

<!-- bilingual-en:start -->
This card owns only the tangent update. [[Newton局部收敛|Newton local convergence]] states the conditions under which the update becomes locally quadratic, while [[Newton失效边界|Newton failure boundaries]] separately diagnose zero derivatives, large steps, cycles, competing basins, and domain escape. [[数值迭代停止条件|Numerical stopping criteria]] governs residual, iteration-limit, and non-finite exits; when a sign-changing bracket is also available, [[混合求根|hybrid root finding]] can try Newton candidates without discarding that bracket.
<!-- bilingual-en:end -->

本卡只定义切线更新。它何时具有局部二次速度，由 [[Newton局部收敛]] 给出条件；零导数、巨大步长、循环、其他吸引域和越界等机制，由 [[Newton失效边界]] 单独诊断。实现层面的残差、最大迭代和非有限值出口统一见 [[数值迭代停止条件]]；若还能先找到异号区间，[[混合求根]] 可以在不丢掉括区间的前提下尝试 Newton 候选。

> [!question-] Try the distinction · 检验理解
> <!-- bilingual-en:start -->
> What does the Newton update answer, and what does it leave open?
>
> **Answer:** It constructs the tangent zero $x_{k+1}$ from $x_k$. It does not by itself prove convergence, identify which root will be reached, or bound the error. Those questions need a local theorem, a failure diagnosis, or a bracket certificate.
> <!-- bilingual-en:end -->
>
> Newton 更新式本身回答了什么，又没有回答什么？
>
> **答案：** 它回答怎样从 $x_k$ 计算切线零点 $x_{k+1}$；它不独自回答序列是否收敛、收敛到哪个根或误差多大。这些问题分别需要局部定理、失败诊断或括区间证书。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- MIT 18.330, Chapter 4: Nonlinear equations, PDF pp. 2–4: tangent construction and the distinction between a defined update and convergence.
<!-- bilingual-en:end -->

- MIT 18.330, [*Introduction to Numerical Analysis, Chapter 4: Nonlinear equations*](https://ocw.mit.edu/courses/18-330-introduction-to-numerical-analysis-spring-2012/5b325bfa56a599794c7196de926844b0_MIT18_330S12_Chapter4.pdf)：核对切线更新推导，以及“更新式有定义”与“序列收敛”必须分开的边界。

<!-- bilingual-en:start -->
- The linked MIT 6.100L Lecture 6 slides, pp. 30–35, provide the derivative-based square-root classroom example.
<!-- bilingual-en:end -->

- [[03_Computer_Science/03_MIT 6.100L/MIT 6.100L-slides/mit6_100l_lec06.pdf|MIT 6.100L Lecture 6 slides]]，slides 30–35：核对从导数产生下一猜测以及平方根迭代的课程语境。

<!-- bilingual-en:start -->
- SciPy’s linked official newton documentation specifies the initial-guess/derivative interface and warns that step-based stopping alone does not guarantee a root.
<!-- bilingual-en:end -->

- SciPy, [`newton`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton.html)：核对初值与导数接口、步长停止不保证已找到根，以及有括区间时更安全方法的官方边界。

<!-- bilingual-en:start -->
- The linked calculus course section supplies the tangent geometry; the local convergence and failure cards retain their separate responsibilities.
<!-- bilingual-en:end -->

- [[01_Math/01_calculus/02_Applications_of_Differentiation.md#33a：迭代公式的推导|微积分课程页]]：复用切线近似的几何推导；本卡只拥有更新式，局部收敛条件与失败机制分别由 [[Newton局部收敛]]、[[Newton失效边界]] 拥有。

<!-- bilingual-en:start -->
The MIT 6.100L slide’s “better approximation” wording is not used as an unconditional guarantee. MIT 18.330 PDF p. 2 was checked as an image: its derivative condition is nonzero, although text extraction can lose the inequality sign.
<!-- bilingual-en:end -->

MIT 6.100L 幻灯片的“更好近似”措辞不作为无条件保证。MIT 18.330 PDF 第 2 页已核对原图：导数条件是不等于零，文本提取可能丢失不等号。

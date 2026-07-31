---
aliases:
  - MIT 18.01SC Applications of Differentiation
  - MIT 18.01SC 导数的应用
  - Applications of Differentiation
tags:
  - math/calculus
  - course/mit-ocw
  - calculus/applications-of-differentiation
source: https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/pages/unit-2-applications-of-differentiation/
---

# Applications of Differentiation

> [!abstract] 本章主线
> 第一章回答“怎样求导”，本章回答“导数知道以后能做什么”。局部的一阶、二阶导数分别给出直线与抛物线近似；导数的符号把局部信息拼成整张图；约束、相关变化率和 Newton 法把现实问题变成可计算方程；平均值定理（Mean Value Theorem, MVT）再严格说明为什么局部斜率能够控制整体变化。最后，反导数、换元和可分离微分方程为积分单元搭桥。
> <!-- bilingual-en:start -->
> Chapter 1 answers "How do we differentiate?" This chapter asks, "What can we do once we know the derivatives?" The first and second derivatives give local linear and quadratic approximations; the signs of derivatives assemble local information into a global graph; optimization constraints, related rates, and Newton's method turn real problems into computable equations; and the Mean Value Theorem (MVT) rigorously explains why local slopes can control global change. Finally, differentials, antiderivatives, substitution, and separable differential equations build a bridge to integration.
> <!-- bilingual-en:end -->

- 官方课程：[MIT OCW 18.01SC — Unit 2](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/pages/unit-2-applications-of-differentiation/)
- 教师：David Jerison；学期：Fall 2010
- 官方顺序：Part A（Session 23–28）→ Problem Set 3 → Part B（Session 29–33）→ Problem Set 4 → Part C（Session 34–40）→ Problem Set 5 → Exam 2（Session 41–42）。
- 本地 SesXXa/b/c 表示同一 Session 中依次播放的片段；正文按字母顺序整合。所有 PDF 仍在原位，链接均给到内容起始页。
- 记号：若 $x$ 是自变量，$\Delta x$ 表示有限改变量；$dx$ 表示在线性化中自由指定的输入微分；$C$ 表示任意常数。近似式后的“$x\approx a$”是适用条件，不是等式的一部分。
<!-- bilingual-en:start -->
- Official course: [MIT OCW 18.01SC — Unit 2](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/pages/unit-2-applications-of-differentiation/)
- Teacher: David Jerison; Semester: Fall 2010
- Official order: Part A (Session 23-28) → Problem Set 3 → Part B (Session 29-33) → Problem Set 4 → Part C (Session 34-40) → Problem Set 5 → Exam 2 (Session 41-42).
- A local SesXXa/b/c represents the fragments that play sequentially within the same Session; the body is alphabetically integrated.  All PDFs are still in place and links are given to the content start page.
- Notation: if $x$ is the independent variable, $\Delta x$ denotes a finite change; $dx$ denotes a freely chosen input differential in a linearization; and $C$ denotes an arbitrary constant. The condition "$x\approx a$" following an approximation states its range of validity; it is not part of the equation.
<!-- bilingual-en:end -->

## 学习目标
<!-- bilingual-en:start -->
*Learning objectives*
<!-- bilingual-en:end -->

学完本章，应当能够：
<!-- bilingual-en:start -->
After completing this chapter, you should be able to:
<!-- bilingual-en:end -->

1. 构造线性化 $L_a(x)$ 与二次近似 $Q_a(x)$，并解释匹配了哪些导数、误差为什么随离基点的距离增长。
2. 由定义域、端点、间断、$f'$ 和 $f''$ 的符号，画出定性正确且标注完整的函数图像。
3. 把优化与相关变化率文字题翻译成变量、约束和目标，并在临界点之外检查端点、间断与单位。
4. 推导并执行 Newton 迭代，识别水平切线、错误根、振荡和初值太远等失败模式。
5. 准确陈述 MVT 的全部假设，用它证明单调性、常函数结论和不等式。
6. 理解微分、反导数、换元和分离变量之间的知识链，并能检查初值、平衡解与最大定义区间。
<!-- bilingual-en:start -->
1. Construct a linearization $L_a(x)$ and a quadratic approximation $Q_a(x)$, explain which derivatives they match, and explain why error grows with distance from the base point.
2. Use the domain, endpoints, discontinuities, and the signs of $f'$ and $f''$ to sketch a qualitatively correct, fully labeled graph.
3. Translate optimization and related-rates problems into variables, constraints, and objective functions, and check endpoints, discontinuities, and units in addition to critical points.
4. Derive and carry out Newton iteration, and recognize failures caused by a horizontal tangent, convergence to the wrong root, oscillation, or a poor initial guess.
5. State every hypothesis of the MVT precisely, and use the theorem to prove monotonicity, constancy, and inequalities.
6. Understand how differentials, antiderivatives, substitution, and separation of variables connect, and check initial conditions, equilibrium solutions, and maximal intervals of existence.
<!-- bilingual-en:end -->

## 课程目录

### Part A：Approximation and Curve Sketching

1. [[#Session 23：Linear Approximation|线性近似]]
2. [[#Session 24：Examples of Linear Approximation|线性近似应用与相对误差]]
3. [[#Session 25：Introduction to Quadratic Approximation|二次近似的构造]]
4. [[#Session 26：Using Quadratic Approximations|组合二次近似]]
5. [[#Session 27：Sketching Graphs I — Polynomials and Rational Functions|多项式与有理函数作图]]
6. [[#Session 28：Sketching Graphs II — General Strategies|一般作图流程]]
7. [[#Problem Set 3|Problem Set 3]]

### Part B：Optimization, Related Rates and Newton’s Method

1. [[#Session 29：Optimization Problems|极值问题]]
2. [[#Session 30：Optimization Problems II|约束优化]]
3. [[#Session 31：Related Rates|相关变化率]]
4. [[#Session 32：Ring on a String|绳上圆环]]
5. [[#Session 33：Newton’s Method|Newton 法]]
6. [[#Problem Set 4|Problem Set 4]]

### Part C：Mean Value Theorem, Antiderivatives and Differential Equations

1. [[#Session 34：Introduction to the Mean Value Theorem|平均值定理]]
2. [[#Session 35：Using the Mean Value Theorem|平均值定理的应用]]
3. [[#Session 36：Differentials|微分]]
4. [[#Session 37：Antiderivatives|反导数]]
5. [[#Session 38：Integration by Substitution|换元积分]]
6. [[#Session 39：Introduction to Differential Equations|微分方程入门]]
7. [[#Session 40：Separation of Variables|分离变量]]
8. [[#Problem Set 5|Problem Set 5]]
9. [[#Session 41：Review for Exam 2|Exam 2 复习]]
10. [[#Session 42：Materials for Exam 2|Exam 2 完整题解]]

---

## Part A：Approximation and Curve Sketching

## Session 23：Linear Approximation

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题：**曲线为什么能在很小范围内用直线代替？怎样把“切线很像曲线”写成可计算的公式？
<!-- bilingual-en:start -->
**Question:** Why can a curve be replaced by a straight line over a very small interval? How can we turn the idea that “the tangent line resembles the curve” into a formula we can calculate with?
<!-- bilingual-en:end -->

**前置：**导数定义、切线方程，以及 $\sin,\cos,e^x,\ln x,(1+x)^r$ 的导数。
<!-- bilingual-en:start -->
**Prerequisites:** The definition of the derivative, the tangent-line equation, and the derivatives of $\sin x$, $\cos x$, $e^x$, $\ln x$, and $(1+x)^r$.
<!-- bilingual-en:end -->

### 23a–23c：从切线方程到线性化
<!-- bilingual-en:start -->
*23a-23c: From tangent equation to linearization*
<!-- bilingual-en:end -->

[[导数的应用#线性与二次近似|线性近似]]从切线开始。设 $f$ 在 $a$ 可导，过 $(a,f(a))$、斜率为 $f'(a)$ 的切线是
<!-- bilingual-en:start -->
[[导数的应用#线性与二次近似|Linear approximation]] begins with the tangent line. Let $f$ be differentiable at $a$. The tangent line through $(a,f(a))$ with slope $f'(a)$ is
<!-- bilingual-en:end -->

$$
L_a(x)=f(a)+f'(a)(x-a).
$$

当 $x$ 接近 $a$ 时，以 $L_a(x)$ 代替 $f(x)$：
<!-- bilingual-en:start -->
When $x$ is close to $a$, use $L_a(x)$ to approximate $f(x)$:
<!-- bilingual-en:end -->

$$
\boxed{f(x)\approx f(a)+f'(a)(x-a)}.
$$

这不是猜测。令 $h=x-a$，可导定义等价于
<!-- bilingual-en:start -->
This is not a guess. Let $h=x-a$; differentiability is equivalent to
<!-- bilingual-en:end -->

$$
\lim_{h\to0}\frac{f(a+h)-f(a)-f'(a)h}{h}=0.
$$

若把余项记作
<!-- bilingual-en:start -->
Define the remainder by
<!-- bilingual-en:end -->

$$
R_1(h)=f(a+h)-f(a)-f'(a)h,
$$

上式就是 $R_1(h)=o(h)$：误差相对于 $h$ 更快趋于 $0$。因此“局部线性”精确地表示为
<!-- bilingual-en:start -->
The limit says that $R_1(h)=o(h)$: relative to $h$, the error tends to zero. Thus local linearity can be stated precisely as
<!-- bilingual-en:end -->

$$
f(a+h)=f(a)+f'(a)h+o(h).
$$

> [!note] 几何直觉
> 放大一条光滑曲线，弯曲部分逐渐看成直线。放大倍率使横向误差按 $h$ 缩放，而余项比 $h$ 更小，所以在极限中消失。
> <!-- bilingual-en:start -->
> As you zoom in on a smooth curve, it increasingly resembles a straight line. Horizontal displacement scales like $h$, while the remainder is of smaller order than $h$, so the error disappears at that scale in the limit.
> <!-- bilingual-en:end -->

**讲义例题：$\ln x$ 在 $a=1$ 的线性化。**
<!-- bilingual-en:start -->
**Example from the lecture notes: linearizing $\ln x$ at $a=1$.**
<!-- bilingual-en:end -->

$$
f(1)=0,\qquad f'(1)=1,
$$

所以
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

$$
\ln x\approx x-1\qquad(x\approx1).
$$

例如 $\ln1.02\approx0.02$。离 $1$ 越远，不能仅凭该式保证准确；需要单独判断[[导数的应用#线性与二次近似|近似误差]]。
<!-- bilingual-en:start -->
For example, $\ln 1.02\approx0.02$. Farther from $1$, this formula alone gives no guarantee of accuracy; the [[导数的应用#线性与二次近似|approximation error]] must be assessed separately.
<!-- bilingual-en:end -->

### 23d–23e：必须掌握的基准近似
<!-- bilingual-en:start -->
*23d–23e: Essential benchmark approximations*
<!-- bilingual-en:end -->

把 $a=0$ 代入一般式，逐项计算 $f(0)$ 与 $f'(0)$：
<!-- bilingual-en:start -->
Set $a=0$ in the general formula and compute $f(0)$ and $f'(0)$ for each function:
<!-- bilingual-en:end -->

$$
\begin{aligned}
\sin x&\approx x,\\
\cos x&\approx1,\\
e^x&\approx1+x,\\
\ln(1+x)&\approx x,\\
(1+x)^r&\approx1+rx,
\end{aligned}
\qquad x\approx0.
$$

最后两式需要先把基点平移到 $0$。例如 $\ln u$ 在 $u=1$ 附近，令 $u=1+x$，便回到 $\ln(1+x)\approx x$。
<!-- bilingual-en:start -->
For the last two formulas, first shift the base point to $0$. For example, near $u=1$, set $u=1+x$ to recover $\ln(1+x)\approx x$.
<!-- bilingual-en:end -->

![[98_attachment/MIT18.01SC/unit02-linear-quadratic.png|760]]

### 配套练习：比较 $\sin x$ 的近似
<!-- bilingual-en:start -->
*Supporting Activity: Comparing $\sin x$ Approximations*
<!-- bilingual-en:end -->

线性化为 $L(x)=x$，故
<!-- bilingual-en:start -->
The linearization is $L(x)=x$, so
<!-- bilingual-en:end -->

$$
\sin0.01\approx0.01,\quad \sin0.1\approx0.1,\quad \sin1\approx1.
$$

计算器真值约为 $0.0099998333,0.0998334166,0.8414709848$。前两项靠近基点，误差很小；$x=1$ 已不够接近 $0$。三角函数必须用**弧度**，因为 $(\sin x)'_{x=0}=1$ 只在弧度制成立。
<!-- bilingual-en:start -->
The calculator values are approximately $0.0099998333$, $0.0998334166$, and $0.8414709848$. The first two inputs are close to the base point, so the errors are small; $x=1$ is no longer sufficiently close to $0$. Angles must be measured in **radians**, because $(\sin x)'_{x=0}=1$ holds only in radian measure.
<!-- bilingual-en:end -->

### 易错点与边界
<!-- bilingual-en:start -->
*Common mistakes and limits of the method*
<!-- bilingual-en:end -->

- “$\approx$”不能在任意 $x$ 使用；必须注明基点。
- 线性化匹配函数值和一阶导数，不匹配二阶导数。
- 把 $\cos x\approx1$ 误解为余弦恒等于 $1$，会丢掉全部弯曲信息。
- 相乘时要丢掉二次及更高次项，才是“乘积的线性部分”。
<!-- bilingual-en:start -->
- The symbol $\approx$ is not valid for arbitrary $x$; always state the base point or the regime in which the approximation applies.
- A linearization matches the function value and first derivative, but not the second derivative.
- Treating $\cos x\approx1$ as though cosine were identically $1$ discards all curvature information.
- When multiplying linear approximations, discard quadratic and higher-order terms to obtain the linear part of the product.
<!-- bilingual-en:end -->

> [!question]- 三道自检题与答案
> 1. $\sqrt{x}$ 在 $a=4$ 的线性化？
> $L(x)=2+\frac14(x-4)$。
>
> 2. 用它估计 $\sqrt{4.1}$。
> $2+\frac14(0.1)=2.025$。
>
> 3. 为什么不能用 $\ln(1+x)\approx x$ 估计 $\ln101$？
> 此时 $x=100$ 不接近 $0$；应另选靠近 $101$ 且函数值已知的基点。
> <!-- bilingual-en:start -->
> 1. Linearization of $\sqrt{x}$ at $a=4$?
> $L(x)=2+\frac14(x-4)$.
> 2. Use it to estimate $\sqrt{4.1}$.
> $2+\frac14(0.1)=2.025$.
> 3. Why can't $\ln101$ be estimated using $\ln(1+x)\approx x$?
> Here $x=100$ is nowhere near the base point $0$. Instead, choose a base point near $101$ at which the function value is known.
> <!-- bilingual-en:end -->

### 本地材料

- [[Ses23a_Lecture_Notes.pdf#page=1|23a Introduction to Linear Approximation]]
- [[Ses23b_Lecture_Notes.pdf#page=1|23b Linear Approximation to ln x at x=1]]
- [[Ses23c_Lecture_Notes.pdf#page=1|23c Definition of the Derivative]]
- [[Ses23d_Lecture_Notes.pdf#page=1|23d Sine, Cosine and Exponential]]
- [[Ses23e_Lecture_Notes.pdf#page=1|23e ln(1+x) and (1+x)^r]]
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise023_Problems.pdf#page=1|Exercise 23 原题]] · [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise023_Solutions.pdf#page=1|Exercise 23 解答]]

**知识链小结：**可导 $\Rightarrow$ 局部线性；切线公式不仅画线，也把难函数值转成四则运算。
<!-- bilingual-en:start -->
**Knowledge-chain summary:** Differentiability $\Rightarrow$ local linearity. The tangent-line formula does more than draw a line: it turns a difficult function evaluation into elementary arithmetic.
<!-- bilingual-en:end -->

## Session 24：Examples of Linear Approximation

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题：**怎样组合多个已知近似？误差应当用绝对量还是相对量衡量？
<!-- bilingual-en:start -->
**Question:** How can several known approximations be combined? Should error be measured in absolute or relative terms?
<!-- bilingual-en:end -->

**前置：**Session 23 的五个基准近似、乘积法则和链式法则。
<!-- bilingual-en:start -->
**Prerequisites:** The five benchmark approximations from Session 23, the product rule, and the chain rule.
<!-- bilingual-en:end -->

### 24a–24c：复杂函数先拆后近似
<!-- bilingual-en:start -->
*24a–24c: Decompose a complicated function before approximating it*
<!-- bilingual-en:end -->

讲义计算
<!-- bilingual-en:start -->
The lecture notes consider
<!-- bilingual-en:end -->

$$
f(x)=\frac{e^{-3x}}{\sqrt{1+x}}
=e^{-3x}(1+x)^{-1/2},\qquad x\approx0.
$$

分别线性化：
<!-- bilingual-en:start -->
Linearize each factor:
<!-- bilingual-en:end -->

$$
e^{-3x}\approx1-3x,\qquad (1+x)^{-1/2}\approx1-\frac12x.
$$

相乘：
<!-- bilingual-en:start -->
Multiply the approximations:
<!-- bilingual-en:end -->

$$
(1-3x)\left(1-\frac12x\right)
=1-\frac72x+\frac32x^2.
$$

线性近似只保留到一次项，因此
<!-- bilingual-en:start -->
A linear approximation retains only terms through first order, so
<!-- bilingual-en:end -->

$$
\boxed{f(x)\approx1-\frac72x}.
$$

为什么这与直接求导必然一致？若
<!-- bilingual-en:start -->
Why must this agree with direct differentiation? Suppose
<!-- bilingual-en:end -->

$$
f(x)\approx f_0+f_1h,\qquad g(x)\approx g_0+g_1h,\qquad h=x-a,
$$

则乘积的常数、一次部分为
<!-- bilingual-en:start -->
then the constant and first-order terms of the product are
<!-- bilingual-en:end -->

$$
f_0g_0+(f_1g_0+f_0g_1)h,
$$

而括号内恰是 $(fg)'(a)$。被丢掉的 $f_1g_1h^2$ 是二阶小量。
<!-- bilingual-en:start -->
The coefficient in parentheses is exactly $(fg)'(a)$. The discarded term $f_1g_1h^2$ is second order in $h$.
<!-- bilingual-en:end -->

### 24d：GPS 时间膨胀例子
<!-- bilingual-en:start -->
*24d: GPS time-dilation example*
<!-- bilingual-en:end -->

特殊相对论给出
<!-- bilingual-en:start -->
Special relativity gives
<!-- bilingual-en:end -->

$$
T_m=\frac{T}{\sqrt{1-v^2/c^2}}.
$$

令 $u=v^2/c^2$。当 $v\ll c$ 时 $u\approx0$，由 $(1-u)^{-1/2}\approx1+\frac12u$，
<!-- bilingual-en:start -->
Let $u=v^2/c^2$. When $v\ll c$, we have $u\approx0$, so $(1-u)^{-1/2}\approx1+\frac12u$ gives
<!-- bilingual-en:end -->

$$
T_m\approx T\left(1+\frac12\frac{v^2}{c^2}\right).
$$

近似带来的时间差是
<!-- bilingual-en:start -->
The resulting time difference is approximately
<!-- bilingual-en:end -->

$$
\Delta T=T_m-T\approx \frac{T}{2}\frac{v^2}{c^2}.
$$

这里 $v/c$ 无量纲，所以括号内可与 $1$ 相加；这也是应用题的单位自检。
<!-- bilingual-en:start -->
Here $v/c$ is dimensionless, so its square can legitimately be added to $1$. This provides a useful dimensional check on the calculation.
<!-- bilingual-en:end -->

### 24e：相对误差
<!-- bilingual-en:start -->
*24e: Relative error*
<!-- bilingual-en:end -->

绝对误差是 $|\widetilde y-y|$；相对误差是
<!-- bilingual-en:start -->
The absolute error is $|\widetilde y-y|$ and the relative error is
<!-- bilingual-en:end -->

$$
\frac{|\widetilde y-y|}{|y|}\qquad(y\ne0).
$$

相对误差回答“错了真实量的百分之几”，比跨尺度比较绝对误差更合理。对于上例，
<!-- bilingual-en:start -->
Relative error answers, “What fraction of the true value is the error?” It is therefore more meaningful than absolute error when comparing quantities on different scales. In the example above,
<!-- bilingual-en:end -->

$$
\frac{\Delta T}{T}\approx\frac12\frac{v^2}{c^2}.
$$

### 配套练习：乘积近似证明
<!-- bilingual-en:start -->
*Supporting exercise: proving the product approximation*
<!-- bilingual-en:end -->

在 $a$ 附近令 $h=x-a$：
<!-- bilingual-en:start -->
In the vicinity of $a$, let $h=x-a$:
<!-- bilingual-en:end -->

$$
\begin{aligned}
L_fL_g
&=[f(a)+f'(a)h][g(a)+g'(a)h]\\
&=f(a)g(a)+[f'(a)g(a)+f(a)g'(a)]h+f'(a)g'(a)h^2.
\end{aligned}
$$

乘积 $fg$ 的线性化是前两项；两者只差二次项。因此“先线性化再相乘并截到一次”与“先相乘再线性化”等价。
<!-- bilingual-en:start -->
The first two terms form the linearization of $fg$; the only discrepancy is quadratic. Thus “linearize each factor, multiply, and truncate after first order” agrees with “multiply first and then linearize.”
<!-- bilingual-en:end -->

### 易错点与边界
<!-- bilingual-en:start -->
*Common mistakes and limits of the method*
<!-- bilingual-en:end -->

- 组合近似时，必须统一在同一个小参数上展开。
- 线性近似不能保留 $x^2$ 项；保留会伪装成二次近似，却没有完整匹配二阶导数。
- 相对误差在真实值为 $0$ 时无定义。
- 物理式中先检查无量纲比值和单位，再代数运算。
<!-- bilingual-en:start -->
- When combining approximations, expand everything in the same small parameter.
- A linear approximation must not retain an $x^2$ term. Keeping one would imitate a quadratic approximation without actually matching the full second-derivative information.
- Relative error is undefined when the true value is $0$.
- In a physical formula, check dimensionless ratios and units before doing the algebra.
<!-- bilingual-en:end -->

> [!question]- 三道自检题与答案
> 1. 线性化 $(1+2x)^3e^{-x}$。
> $(1+6x)(1-x)\approx1+5x$。
>
> 2. 若真值 $1000$、估计 $1002$，绝对与相对误差？
> $2$ 与 $0.002=0.2\%$。
>
> 3. 为什么 $x^2$ 比 $x$ 小必须附带条件？
> 只有 $|x|<1$ 时 $|x^2|<|x|$；近似还要求 $x$ 足够接近展开点。
> <!-- bilingual-en:start -->
> 1. Linearize $(1+2x)^3e^{-x}$.
> $(1+6x)(1-x)\approx1+5x$.
> 2. If the true value is $1000$ and the estimate is $1002$, what are the absolute and relative errors?
> $2$ and $0.002=0.2\%$.
> 3. Why does the claim that $x^2$ is smaller than $x$ require a condition?
> $|x^2|<|x|$ only for $|x|<1$; the approximation also requires $x$ to be close enough to the point of expansion.
> <!-- bilingual-en:end -->

### 本地材料

- [[Ses24a_Lecture_Notes.pdf#page=1|24a Curves are Hard, Lines are Easy]]
- [[Ses24b_Lecture_Notes.pdf#page=1|24b Complicated Exponential]]
- [[Ses24c_Lecture_Notes.pdf#page=1|24c Direct Formula Check]]
- [[Ses24d_Lecture_Notes.pdf#page=1|24d GPS Time Dilation]]
- [[Ses24e_Lecture_Notes.pdf#page=1|24e Relative Error]]
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise024_Problems.pdf#page=1|Exercise 24 原题]] · [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise024_Solutions.pdf#page=1|Exercise 24 解答]]

**知识链小结：**局部线性可像代数式一样组合，但必须按阶截断；应用中用相对误差说明结果是否“足够准”。
<!-- bilingual-en:start -->
**Knowledge-chain summary:** Local linear approximations can be combined algebraically, provided the result is truncated at the correct order. In applications, relative error tells us whether the approximation is accurate enough for the scale of the problem.
<!-- bilingual-en:end -->

## Session 25：Introduction to Quadratic Approximation

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题：**切线没记录[[导数的应用#从导数读图像|凹凸性]]，怎样加入最少的新信息来改进精度？为什么二次项系数是 $f''(a)/2$？
<!-- bilingual-en:start -->
**Question:** A tangent line does not capture [[导数的应用#从导数读图像|concavity]]. What is the smallest amount of additional information needed to improve the approximation, and why is the quadratic coefficient $f''(a)/2$?
<!-- bilingual-en:end -->

**前置：**线性化、二阶导数及凹凸性。
<!-- bilingual-en:start -->
**Prerequisites:** Linearization, second derivatives, and concavity.
<!-- bilingual-en:end -->

### 25a：二次近似公式
<!-- bilingual-en:start -->
*25a: Quadratic approximation formula*
<!-- bilingual-en:end -->

若 $f$ 在 $a$ 有二阶导数，[[导数的应用#线性与二次近似|二次近似]]定义为
<!-- bilingual-en:start -->
If $f$ has a second derivative at $a$, its [[导数的应用#线性与二次近似|quadratic approximation]] is defined by
<!-- bilingual-en:end -->

$$
\boxed{
Q_a(x)=f(a)+f'(a)(x-a)+\frac{f''(a)}{2}(x-a)^2
}.
$$

$Q_a$ 满足三个匹配条件：
<!-- bilingual-en:start -->
$Q_a$ meets three matching conditions:
<!-- bilingual-en:end -->

$$
Q_a(a)=f(a),\qquad Q_a'(a)=f'(a),\qquad Q_a''(a)=f''(a).
$$

### 25b：系数 $1/2$ 的逐步推导
<!-- bilingual-en:start -->
*25b: Step-by-step derivation of the coefficient $1/2$*
<!-- bilingual-en:end -->

设在 $a=0$ 附近寻找 $Q(x)=A+Bx+Cx^2$。
<!-- bilingual-en:start -->
Near $a=0$, seek an approximation of the form $Q(x)=A+Bx+Cx^2$.
<!-- bilingual-en:end -->

$$
Q(0)=A,\quad Q'(x)=B+2Cx,\quad Q''(x)=2C.
$$

要求 $Q(0)=f(0),Q'(0)=f'(0),Q''(0)=f''(0)$，依次得到
<!-- bilingual-en:start -->
Requiring $Q(0)=f(0)$, $Q'(0)=f'(0)$, and $Q''(0)=f''(0)$ gives
<!-- bilingual-en:end -->

$$
A=f(0),\qquad B=f'(0),\qquad C=\frac{f''(0)}2.
$$

因此 $1/2$ 不是约定，而是二次项求两次导后产生因子 $2$ 的必然补偿。一般基点只需把 $x$ 换成 $x-a$。
<!-- bilingual-en:start -->
Thus the factor $1/2$ is not a convention: differentiating the quadratic term twice produces a factor of $2$, which must be cancelled. For a general base point, simply replace $x$ by $x-a$.
<!-- bilingual-en:end -->

### 25c：基本二次近似库
<!-- bilingual-en:start -->
*25c: Library of Basic Quadratic Approximations*
<!-- bilingual-en:end -->

$$
\begin{aligned}
\sin x&\approx x,\\
\cos x&\approx1-\frac{x^2}{2},\\
e^x&\approx1+x+\frac{x^2}{2},\\
\ln(1+x)&\approx x-\frac{x^2}{2},\\
(1+x)^r&\approx1+rx+\frac{r(r-1)}2x^2,
\end{aligned}
\qquad x\approx0.
$$

例如
<!-- bilingual-en:start -->
For example
<!-- bilingual-en:end -->

$$
\ln1.1=\ln(1+0.1)\approx0.1-\frac{0.1^2}{2}=0.095,
$$

比线性估计 $0.1$ 更接近真值约 $0.09531$。
<!-- bilingual-en:start -->
This is closer to the true value, about $0.09531$, than the linear estimate $0.1$ is.
<!-- bilingual-en:end -->

### 配套练习：$e^x$ 的精度
<!-- bilingual-en:start -->
*Complementary Exercise: Precision of $e^x$*
<!-- bilingual-en:end -->

二次近似 $Q(x)=1+x+x^2/2$ 给出
<!-- bilingual-en:start -->
The quadratic approximation $Q(x)=1+x+x^2/2$ gives
<!-- bilingual-en:end -->

$$
e^{0.01}\approx1.01005,\quad e^{0.1}\approx1.105,\quad e^1\approx2.5.
$$

真值分别约 $1.010050167,1.105170918,2.718281828$。仍然是越靠近 $0$ 越好；提高次数并不允许无视展开点。
<!-- bilingual-en:start -->
The true values are approximately $1.010050167$, $1.105170918$, and $2.718281828$. Accuracy still improves as the input approaches $0$; increasing the degree does not make the expansion point irrelevant.
<!-- bilingual-en:end -->

### 易错点与边界
<!-- bilingual-en:start -->
*Common mistakes and limits of the method*
<!-- bilingual-en:end -->

- “二次近似”允许二次项系数为 $0$，例如 $\sin x$ 在 $0$ 的二次近似仍是 $x$。
- 二阶可导只保证公式可构造；精确误差界需要更高阶信息，后续由 Taylor 余项给出。
- $f''(a)>0$ 表示最佳拟合抛物线开口向上，不代表函数在所有地方凸。
<!-- bilingual-en:start -->
- A “quadratic approximation” may have a zero quadratic coefficient. For example, the quadratic approximation to $\sin x$ at $0$ is still $x$.
- Having a second derivative is enough to construct the formula, but a precise error bound requires higher-order information, later supplied by the Taylor remainder.
- $f''(a)>0$ means that the best-fitting local parabola opens upward; it does not imply that the function is concave up everywhere.
<!-- bilingual-en:end -->

> [!question]- 三道自检题与答案
> 1. $\sqrt{1+x}$ 在 $0$ 的二次近似？
> $1+\frac12x-\frac18x^2$。
>
> 2. 为什么 $\cos x$ 的线性化看不出弯曲？
> $\cos'(0)=0$，线性化是水平线；$\cos''(0)=-1$ 才记录向下弯曲。
>
> 3. $Q_a$ 与 $f$ 在 $a$ 处共匹配几项信息？
> 函数值、一阶导、二阶导，共三项。
> <!-- bilingual-en:start -->
> 1. What is the quadratic approximation to $\sqrt{1+x}$ at $0$?
> $1+\frac12x-\frac18x^2$.
> 2. Why is $\cos x$'s linearization blind to bending?
> $\cos'(0)=0$, the linearization is horizontal; $\cos''(0)=-1$ records the downward bend.
> 3. Which pieces of information do $Q_a$ and $f$ match at $a$?
> The function value, first derivative, and second derivative: three pieces in total.
> <!-- bilingual-en:end -->

### 本地材料

- [[Ses25a_Lecture_Notes.pdf#page=1|25a Formula for Quadratic Approximation]]
- [[Ses25b_Lecture_Notes.pdf#page=1|25b Explaining the Formula]]
- [[Ses25c_Lecture_Notes.pdf#page=1|25c Basic Quadratic Approximations]]
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise025_Problems.pdf#page=1|Exercise 25 原题]] · [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise025_Solutions.pdf#page=1|Exercise 25 解答]]

**知识链小结：**线性化匹配位置与斜率；二次近似再匹配曲率，是用最小复杂度换取更高局部精度。
<!-- bilingual-en:start -->
**Knowledge-chain summary:** Linearization matches position and slope; quadratic approximation also matches curvature. It gains greater local accuracy with the smallest possible increase in complexity.
<!-- bilingual-en:end -->

## Session 26：Using Quadratic Approximations

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题：**复杂函数怎样不做两次繁琐求导就得到二次近似？“丢掉高阶项”怎样系统执行？
<!-- bilingual-en:start -->
**Question:** How can we obtain a quadratic approximation to a complicated function without differentiating it twice at length? How can higher-order terms be discarded systematically?
<!-- bilingual-en:end -->

**前置：**Session 25 的基本二次近似库、按次数展开乘积。
<!-- bilingual-en:start -->
**Prerequisites:** The basic quadratic approximations from Session 25 and multiplication of truncated expansions by degree.
<!-- bilingual-en:end -->

### 26a–26b：近似与收敛速度
<!-- bilingual-en:start -->
*26a-26b: Approximation and Convergence Rate*
<!-- bilingual-en:end -->

令
<!-- bilingual-en:start -->
Consider the sequence
<!-- bilingual-en:end -->

$$
a_k=\left(1+\frac1k\right)^k.
$$

线性近似 $\ln(1+x)\approx x$ 给出
<!-- bilingual-en:start -->
The linear approximation $\ln(1+x)\approx x$ gives
<!-- bilingual-en:end -->

$$
\ln a_k=k\ln\left(1+\frac1k\right)\approx1,
$$

解释 $a_k\to e$。若还要看误差速度，必须保留二次项：
<!-- bilingual-en:start -->
This explains why $a_k\to e$. To determine the rate of convergence, retain the quadratic term:
<!-- bilingual-en:end -->

$$
\ln a_k
\approx k\left(\frac1k-\frac1{2k^2}\right)
=1-\frac1{2k}.
$$

所以 $\ln a_k$ 与 $1$ 的主误差约为 $-1/(2k)$；线性近似能给极限，二次近似进一步给收敛速度。
<!-- bilingual-en:start -->
So the main error between $\ln a_k$ and $1$ is about $-1/(2k)$. The linear approximation can give the limit and the quadratic approximation can give the convergence rate.
<!-- bilingual-en:end -->

### 26c：复杂函数的二次展开
<!-- bilingual-en:start -->
*26c: Quadratic Expansion of Complex Functions*
<!-- bilingual-en:end -->

再次处理
<!-- bilingual-en:start -->
Now reconsider
<!-- bilingual-en:end -->

$$
\frac{e^{-3x}}{\sqrt{1+x}}.
$$

$$
e^{-3x}\approx1-3x+\frac92x^2,
$$

而 $r=-1/2$ 时
<!-- bilingual-en:start -->
while at $r=-1/2$
<!-- bilingual-en:end -->

$$
(1+x)^{-1/2}\approx1-\frac12x+\frac38x^2.
$$

相乘只收集总次数不超过 $2$ 的项：
<!-- bilingual-en:start -->
On multiplication, retain only terms of total degree at most $2$:
<!-- bilingual-en:end -->

$$
\begin{aligned}
\left(1-3x+\frac92x^2\right)
\left(1-\frac12x+\frac38x^2\right)
&\approx1-\frac72x+
\left(\frac38+\frac32+\frac92\right)x^2\\
&=1-\frac72x+\frac{51}{8}x^2.
\end{aligned}
$$

一次部分与 Session 24 完全相同，这是不同阶近似的一致性检查。
<!-- bilingual-en:start -->
The first-order part is exactly the same as in Session 24, providing a consistency check across approximation orders.
<!-- bilingual-en:end -->

### 26d：一般 $n$ 次匹配的来源
<!-- bilingual-en:start -->
*26d: Why the general $n$th-order coefficients have their form*
<!-- bilingual-en:end -->

若
<!-- bilingual-en:start -->
Suppose
<!-- bilingual-en:end -->

$$
P_n(x)=a_0+a_1x+\cdots+a_nx^n,
$$

则在 $0$ 取第 $k$ 阶导，低于 $k$ 次项消失，高于 $k$ 次项仍含 $x$ 而在 $0$ 消失，仅剩
<!-- bilingual-en:start -->
When we take the $k$th derivative and evaluate it at $0$, terms of degree below $k$ have vanished under differentiation, while terms of degree above $k$ still contain a factor of $x$ and vanish at $0$. Only one term remains:
<!-- bilingual-en:end -->

$$
P_n^{(k)}(0)=k!a_k.
$$

令其等于 $f^{(k)}(0)$，
<!-- bilingual-en:start -->
Setting this equal to $f^{(k)}(0)$ gives
<!-- bilingual-en:end -->

$$
a_k=\frac{f^{(k)}(0)}{k!}.
$$

于是
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

$$
P_n(x)=\sum_{k=0}^n\frac{f^{(k)}(0)}{k!}x^k.
$$

这已是 Taylor 多项式的骨架；本章先用到二次，Unit 5 再研究无穷级数与余项。
<!-- bilingual-en:start -->
This is already the structure of the Taylor polynomial. This chapter uses only the quadratic case; Unit 5 later develops infinite series and remainder terms.
<!-- bilingual-en:end -->

### 易错点与边界
<!-- bilingual-en:start -->
*Common mistakes and limits of the method*
<!-- bilingual-en:end -->

- 相乘时不是分别“保留二次”后把所有项都留下，而是最终总次数仅保留到 $2$。
- $e^{-3x}$ 的二次项是 $(-3x)^2/2=9x^2/2$，符号为正。
- 近似等式不能在中途当精确恒等式消去可能与误差同阶的量。
<!-- bilingual-en:start -->
- When multiplying truncated expansions, do not keep every product merely because each factor was truncated at degree $2$; keep only terms whose final total degree is at most $2$.
- The quadratic term of $e^{-3x}$ is $(-3x)^2/2=9x^2/2$ with a positive sign.
- Do not manipulate an approximation as though it were an exact identity when the cancelled quantity may be of the same order as the approximation error.
<!-- bilingual-en:end -->

> [!question]- 三道自检题与答案
> 1. 求 $e^x(1+x)^{-1}$ 的二次近似。
> $(1+x+x^2/2)(1-x+x^2)\approx1+x^2/2$。
>
> 2. 为什么三次系数分母是 $3!$？
> $d^3(a_3x^3)/dx^3=3!a_3$。
>
> 3. 二次近似能否自动给出严格误差上界？
> 不能；还需控制区间上的三阶导数等余项信息。
> <!-- bilingual-en:start -->
> 1. Find the quadratic approximation of $e^x(1+x)^{-1}$.
> $(1+x+x^2/2)(1-x+x^2)\approx1+x^2/2$.
> 2. Why is the denominator of the cubic coefficient $3!$?
> $d^3(a_3x^3)/dx^3=3!a_3$.
> 3. Does a quadratic approximation automatically provide a rigorous upper bound on the error?
> No. One also needs remainder information, such as a bound on the third derivative over the interval.
> <!-- bilingual-en:end -->

### 本地材料

- [[Ses26a_Lecture_Notes.pdf#page=1|26a Quadratic Approximation Library]]
- [[Ses26b_Lecture_Notes.pdf#page=1|26b Approximation of ln e]]
- [[Ses26c_Lecture_Notes.pdf#page=1|26c Complicated Quadratic Approximation]]
- [[Ses26d_Lecture_Notes.pdf#page=1|26d Deriving ln(1+x) and (1+x)^r]]
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise026_Problems.pdf#page=1|Exercise 26 原题]] · [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise026_Solutions.pdf#page=1|Exercise 26 解答]]

**知识链小结：**组合近似的核心是“统一小参数、按总次数截断”；阶数越高，能读取的局部变化层次越多。
<!-- bilingual-en:start -->
**Knowledge-chain summary:** To combine approximations, express them in one small parameter and truncate by total degree. Higher-order approximations capture progressively finer levels of local change.
<!-- bilingual-en:end -->

## Session 27：Sketching Graphs I — Polynomials and Rational Functions

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题：**不用逐点计算，怎样由 $f'$、$f''$ 画出全局形状？为何间断点和无穷远行为与临界点同样重要？
<!-- bilingual-en:start -->
**Question:** How can $f'$ and $f''$ determine a graph's global shape without point-by-point calculation? Why are discontinuities and end behavior just as important as critical points?
<!-- bilingual-en:end -->

**前置：**极限、定义域、一阶/二阶导数与奇偶性。
<!-- bilingual-en:start -->
**Prerequisites:** Limits, domains, first and second derivatives, and parity.
<!-- bilingual-en:end -->

### 27a：两条基本原则
<!-- bilingual-en:start -->
*27a: Two basic principles*
<!-- bilingual-en:end -->

在一个区间上：
<!-- bilingual-en:start -->
On an interval:
<!-- bilingual-en:end -->

$$
f'(x)>0\Rightarrow f\text{ 递增},\qquad
f'(x)<0\Rightarrow f\text{ 递减}.
$$

再对 $f'$ 应用同样思想：
<!-- bilingual-en:start -->
Then apply the same idea to $f'$:
<!-- bilingual-en:end -->

$$
f''(x)>0\Rightarrow f'\text{ 递增，图像凹向上};
$$

$$
f''(x)<0\Rightarrow f'\text{ 递减，图像凹向下}.
$$

严格证明将在 Session 34 用 MVT 完成。
<!-- bilingual-en:start -->
A rigorous proof using the MVT appears in Session 34.
<!-- bilingual-en:end -->

### 27b：多项式例 $f(x)=3x-x^3$
<!-- bilingual-en:start -->
*27b: Polynomial Example $f(x)=3x-x^3$*
<!-- bilingual-en:end -->

一阶导数
<!-- bilingual-en:start -->
first derivative
<!-- bilingual-en:end -->

$$
f'(x)=3-3x^2=3(1-x)(1+x).
$$

符号表：
<!-- bilingual-en:start -->
Sign chart:
<!-- bilingual-en:end -->

| 区间 | $(-\infty,-1)$ | $(-1,1)$ | $(1,\infty)$ |
|---|---:|---:|---:|
| $f'$ | $-$ | $+$ | $-$ |
| $f$ | 递减 | 递增 | 递减 |

[[导数的应用#从导数读图像|临界点]]（critical point）是定义域内 $f'(x)=0$ 或 $f'$ 不存在而 $f$ 存在的候选位置。此处 $x=\pm1$，函数点为 $(-1,-2),(1,2)$。故前者局部最小，后者局部最大。
<!-- bilingual-en:start -->
A [[导数的应用#从导数读图像|critical point]] is a point in the domain where $f'(x)=0$, or where $f'$ does not exist while $f$ does. Here $x=\pm1$, giving the points $(-1,-2)$ and $(1,2)$. The former is a local minimum and the latter a local maximum.
<!-- bilingual-en:end -->

$$
f''(x)=-6x.
$$

$x<0$ 时凹向上，$x>0$ 时凹向下；$x=0$ 两侧凹凸改变，所以 $(0,0)$ 是拐点。最高次项 $-x^3$ 控制两端：
<!-- bilingual-en:start -->
The graph is concave up for $x<0$ and concave down for $x>0$. Concavity changes at $x=0$, so $(0,0)$ is an inflection point. The leading term $-x^3$ controls both end behaviors:
<!-- bilingual-en:end -->

$$
x\to\infty:f(x)\to-\infty,\qquad
x\to-\infty:f(x)\to\infty.
$$

![[98_attachment/MIT18.01SC/unit02-curve-sign-chart.png|760]]

### 27c：有理函数例 $f(x)=\dfrac{x+1}{x+2}$
<!-- bilingual-en:start -->
*27c: Rational Function Example $f(x)=\dfrac{x+1}{x+2}$*
<!-- bilingual-en:end -->

先改写
<!-- bilingual-en:start -->
First rewrite the function as
<!-- bilingual-en:end -->

$$
f(x)=1-\frac1{x+2}.
$$

因此定义域排除 $x=-2$，竖直渐近线为 $x=-2$，水平渐近线为 $y=1$。并且
<!-- bilingual-en:start -->
Thus the domain excludes $x=-2$, the vertical asymptote is $x=-2$, and the horizontal asymptote is $y=1$. Moreover,
<!-- bilingual-en:end -->

$$
f'(x)=\frac1{(x+2)^2}>0,\qquad x\ne-2.
$$

它分别在 $(-\infty,-2)$ 与 $(-2,\infty)$ 递增，但**不能**跨越间断点说在整个定义域递增。二阶导
<!-- bilingual-en:start -->
It is increasing on $(-\infty,-2)$ and on $(-2,\infty)$ separately, but it is **not** increasing across the discontinuity when the entire domain is considered. The second derivative is
<!-- bilingual-en:end -->

$$
f''(x)=-\frac2{(x+2)^3},
$$

左支凹向上、右支凹向下。$x=-2$ 不是拐点，因为原函数在那里不连续。
<!-- bilingual-en:start -->
The left branch is concave up and the right branch is concave down. Nevertheless, $x=-2$ is not an inflection point because the function is discontinuous there.
<!-- bilingual-en:end -->

### 易错点与边界
<!-- bilingual-en:start -->
*Common mistakes and limits of the method*
<!-- bilingual-en:end -->

- 只解 $f'=0$ 会漏掉端点、间断点及 $f'$ 不存在但 $f$ 存在的尖点。
- $f''(c)=0$ 只是拐点候选；必须检查两侧凹凸是否改变且函数在 $c$ 连续。
- 两个分离区间上都递增，不等于跨间断点递增。
- 有理函数的无穷远极限与竖直渐近线必须先由原式/改写式判断。
<!-- bilingual-en:start -->
- Solving only $f'=0$ misses endpoints, discontinuities, and cusps where $f$ exists but $f'$ does not.
- $f''(c)=0$ identifies only a candidate inflection point; you must verify that concavity changes across $c$ and that the function is continuous there.
- A function can be increasing on each of two separate intervals without being increasing across the discontinuity between them.
- Determine a rational function's end behavior and vertical asymptotes from the original or algebraically rewritten formula before sketching.
<!-- bilingual-en:end -->

> [!question]- 三道自检题与答案
> 1. $x^4$ 的 $x=0$ 是拐点吗？
> 不是；$f''=12x^2$ 两侧均非负，凹凸不改变。
>
> 2. $|x|$ 的 $x=0$ 是否应在作图时检查？
> 是；$f$ 存在而 $f'$ 不存在，是尖点和极小值。
>
> 3. $\frac1x$ 能否说在整个定义域递减？
> 按全局定义不能；它分别在两支递减，但例如 $-1<1$ 而 $f(-1)<f(1)$。
> <!-- bilingual-en:start -->
> 1. Is $x=0$ an inflection point of $x^4$?
> No. Since $f''=12x^2$ is nonnegative on both sides, the concavity does not change.
>
> 2. Should $x=0$ be checked when sketching $|x|$?
> Yes. The function exists there but its derivative does not; the point is both a cusp and a minimum.
>
> 3. Is $1/x$ decreasing on its entire domain?
> Not under the global definition of a decreasing function. It decreases separately on each branch, but, for example, $-1<1$ while $f(-1)<f(1)$.
> <!-- bilingual-en:end -->

### 本地材料

- [[Ses27a_Lecture_Notes.pdf#page=1|27a Introduction to Curve Sketching]]
- [[Ses27b_Lecture_Notes.pdf#page=1|27b Polynomial Example]]
- [[Ses27c_Lecture_Notes.pdf#page=1|27c Rational Function Example]]

**知识链小结：**作图不是“求几个导数零点”，而是把定义域、间断、端点、无穷远、单调和凹凸拼成一张相互校验的图。
<!-- bilingual-en:start -->
**Knowledge chain summary:** Graph sketching is not merely "finding a few zeros of derivatives"; it combines the domain, discontinuities, endpoints, end behavior, monotonicity, and concavity into one picture.
<!-- bilingual-en:end -->

## Session 28：Sketching Graphs II — General Strategies

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题：**面对任意函数，怎样用一套不会漏项的顺序完成作图？
<!-- bilingual-en:start -->
**Question:** Given an arbitrary function, what reliable sequence of steps produces a graph without omitting essential features?
<!-- bilingual-en:end -->

**前置：**Session 27 的单调、凹凸、渐近线与临界点。
<!-- bilingual-en:start -->
**Prerequisites:** Monotonicity, concavity, asymptotes, and critical points from Session 27.
<!-- bilingual-en:end -->

### 28a：[[导数的应用#从导数读图像|曲线描绘]]五步法
<!-- bilingual-en:start -->
*28a: A five-step method for [[导数的应用#从导数读图像|curve sketching]]*
<!-- bilingual-en:end -->

1. **先做预备代数：**定义域、对称性、截距、易算点。
2. **查边界：**端点、间断点的单侧极限、$x\to\pm\infty$，标出渐近线。
3. **查一阶信息：**求 $f'$，列全部临界候选，做符号表，算必要的临界值。
4. **查二阶信息：**求 $f''$，找凹凸区间及真正拐点。
5. **合成并复核：**图像必须同时满足函数值、极限、单调、凹凸；若冲突，回查代数。
<!-- bilingual-en:start -->
1. **Do the preliminary algebra first:** Find the domain, symmetry, intercepts, and any easy reference points.
2. **Check boundaries:** Evaluate endpoints, one-sided limits at discontinuities, and behavior as $x\to\pm\infty$; mark any asymptotes.
3. **Check first-order information:** Find $f'$, list every critical-point candidate, construct a sign chart, and evaluate the function at the necessary points.
4. **Check second-order information:** Find $f''$, determine intervals of concavity, and identify genuine inflection points.
5. **Assemble and verify:** The graph must simultaneously agree with function values, limits, monotonicity, and concavity. If these conflict, revisit the algebra.
<!-- bilingual-en:end -->

### 28b：完整例题 $f(x)=\dfrac{x}{\ln x}$
<!-- bilingual-en:start -->
*28b: Full Example $f(x)=\dfrac{x}{\ln x}$*
<!-- bilingual-en:end -->

**定义域：**$x>0$ 且 $x\ne1$。
<!-- bilingual-en:start -->
**Domain:** $x>0$ and $x\ne1$.
<!-- bilingual-en:end -->

**边界：**
<!-- bilingual-en:start -->
**Boundary behavior:**
<!-- bilingual-en:end -->

$$
\lim_{x\to0^+}\frac{x}{\ln x}=0^-,
\quad
\lim_{x\to1^-}\frac{x}{\ln x}=-\infty,
\quad
\lim_{x\to1^+}\frac{x}{\ln x}=+\infty,
\quad
\lim_{x\to\infty}\frac{x}{\ln x}=+\infty.
$$

因此 $x=1$ 是竖直渐近线。
<!-- bilingual-en:start -->
Therefore, $x=1$ is a vertical asymptote.
<!-- bilingual-en:end -->

**一阶导：**
<!-- bilingual-en:start -->
**First derivative:**
<!-- bilingual-en:end -->

$$
f'(x)=\frac{\ln x-1}{(\ln x)^2}.
$$

分母正，符号由 $\ln x-1$ 决定。因此在 $(0,1)$、$(1,e)$ 递减，在 $(e,\infty)$ 递增；唯一普通临界点 $x=e$，且 $f(e)=e$，是右支最小值。
<!-- bilingual-en:start -->
The denominator is positive, and the sign is determined by $\ln x-1$.  Therefore, it decreases at $(0,1)$ and $(1,e)$, and increases at $(e,\infty)$. The only ordinary critical point is $x=e$, and $f(e)=e$ is the minimum value of the right branch.
<!-- bilingual-en:end -->

**二阶导：**
<!-- bilingual-en:start -->
**Second derivative:**
<!-- bilingual-en:end -->

$$
f''(x)=\frac{2-\ln x}{x(\ln x)^3}.
$$

因为 $x>0$，符号表给出：
<!-- bilingual-en:start -->
Because $x>0$, the sign chart gives
<!-- bilingual-en:end -->

$$
(0,1):f''<0,\qquad
(1,e^2):f''>0,\qquad
(e^2,\infty):f''<0.
$$

在 $x=e^2$ 连续且凹凸改变，拐点为 $(e^2,e^2/2)$；$x=1$ 虽两侧符号改变，却是间断点而非拐点。
<!-- bilingual-en:start -->
At $x=e^2$, the function is continuous and its concavity changes, so $(e^2,e^2/2)$ is an inflection point. Although the sign of $f''$ changes across $x=1$, that point is a discontinuity rather than an inflection point.
<!-- bilingual-en:end -->

### 配套练习：由特征反造三次函数
<!-- bilingual-en:start -->
*Supporting exercise: constructing a cubic from prescribed features*
<!-- bilingual-en:end -->

若要求拐点在 $x=2$，且 $x>2$ 凹向下，可先选
<!-- bilingual-en:start -->
To place an inflection point at $x=2$ with the graph concave down for $x>2$, begin by choosing
<!-- bilingual-en:end -->

$$
f''(x)=2-x.
$$

积分得
<!-- bilingual-en:start -->
Integrating gives
<!-- bilingual-en:end -->

$$
f'(x)=2x-\frac{x^2}{2}+C.
$$

要求临界点在 $1,3$，代 $x=1$ 得 $C=-3/2$，并自动满足 $f'(3)=0$。再次积分：
<!-- bilingual-en:start -->
To make $x=1$ and $x=3$ critical points, substitute $x=1$ to obtain $C=-3/2$; this choice automatically gives $f'(3)=0$. Integrating once more,
<!-- bilingual-en:end -->

$$
f(x)=x^2-\frac{x^3}{6}-\frac32x+D.
$$

$D$ 只做竖直平移，不影响单调与凹凸，所以答案不唯一。原题 PDF 的第一条凹凸描述与解答首页有文字方向差异；上述推导遵循原题“$x<2$ 凹向上、$x>2$ 凹向下”。
<!-- bilingual-en:start -->
$D$ produces only a vertical shift and does not affect monotonicity or concavity, so the answer is not unique. The original PDF and the first page of its solution describe the concavity in opposite directions; the derivation here follows the problem statement: "$x<2$ concave up, $x>2$ concave down."
<!-- bilingual-en:end -->

### 易错点与边界
<!-- bilingual-en:start -->
*Common mistakes and limits of the method*
<!-- bilingual-en:end -->

- 把 $x\to0^+$ 的极限点 $(0,0)$ 画成函数实际包含的点；本例 $x=0$ 不在定义域。
- 写 $f(\infty)$ 只是非正式缩写，正式论证必须写极限。
- 复杂二阶导若只为定性图，可在一阶信息已足够时省略；但若题目要求拐点则不能省。
<!-- bilingual-en:start -->
- Do not draw the limiting point $(0,0)$ as though it belonged to the graph. In this example, $x=0$ is outside the domain.
- Writing $f(\infty)$ is only informal shorthand; a formal argument must use a limit.
- If only a qualitative sketch is required and first-order information is sufficient, a complicated second derivative may be unnecessary. It cannot be omitted when the problem explicitly asks for inflection points.
<!-- bilingual-en:end -->

> [!question]- 三道自检题与答案
> 1. 作图的第一步为何不是求导？
> 定义域和间断会决定导数符号表应分成哪些区间，也可能直接给出最显著特征。
>
> 2. 本例为何 $x=e$ 是右支绝对最小？
> 右支从 $+\infty$ 递减到 $e$，再递增到 $+\infty$。
>
> 3. 加常数 $D$ 会改变什么？
> 只竖直平移；不改变 $f'$、$f''$、临界点横坐标和凹凸。
> <!-- bilingual-en:start -->
> 1. Why is differentiation not the first step in graph sketching?
> The domain and discontinuities determine how the sign chart must be partitioned, and they may reveal the graph's most prominent features immediately.
> 2. In this example, why is $x=e$ the absolute minimum of the right branch?
> The right branch decreases from $+\infty$ to $e$, and then increases to $+\infty$.
> 3. What does the addition of the constant $D$ change?
> It produces only a vertical translation; it does not change $f'$, $f''$, the $x$-coordinates of critical points, or concavity.
> <!-- bilingual-en:end -->

### 本地材料

- [[Ses28a_Lecture_Notes.pdf#page=1|28a General Strategy]]
- [[Ses28b_Lecture_Notes.pdf#page=1|28b Detailed Example x/ln x]]
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise028_Problems.pdf#page=1|Exercise 28 原题]] · [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise028_Solutions.pdf#page=1|Exercise 28 解答]]

**知识链小结：**稳定的作图流程先划分定义域，再用 $f'$ 决定方向、$f''$ 修饰形状，最后用极限封住各段的两端。
<!-- bilingual-en:start -->
**Knowledge chain summary:** A reliable graph-sketching workflow first partitions the domain, then uses $f'$ to determine direction, $f''$ to determine shape, and limits to close off both ends of each interval.
<!-- bilingual-en:end -->

## Problem Set 3

> [!info] 官方指定范围与材料
> 2A：1, 3, 6, 11, 12a, 12d, 12e；2B：2a, 2e, 2h, 4, 6a, 6b, 7a, 7b。
> [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/PSet02_Problems.pdf#page=1|Applications of Differentiation 原题]] · [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/PSet02_Solutions.pdf#page=1|官方解答]]

### 2A：Approximation

<!-- bilingual-en:start -->
*2A: Approximation*
<!-- bilingual-en:end -->
> [!example]- 2A-1：$\sqrt{a+bx}$ 在 $0$ 的线性化
> 已知 $a>0$。令 $f(x)=\sqrt{a+bx}$，
> $$
> f(0)=\sqrt a,\qquad f'(x)=\frac{b}{2\sqrt{a+bx}},\qquad f'(0)=\frac{b}{2\sqrt a}.
> $$
> 因此
> $$
> \boxed{L(x)=\sqrt a+\frac{b}{2\sqrt a}x}.
> $$
> 另一条检查路径：
> $$
> \sqrt{a+bx}=\sqrt a\left(1+\frac{b}{a}x\right)^{1/2}
> \approx\sqrt a\left(1+\frac{b}{2a}x\right).
> $$
> 条件 $a>0$ 保证展开点附近有实值且导数有限。
> <!-- bilingual-en:start -->
> Assume $a>0$ and let $f(x)=\sqrt{a+bx}$. Then
> $$
> f(0)=\sqrt a,\qquad f'(x)=\frac{b}{2\sqrt{a+bx}},\qquad f'(0)=\frac{b}{2\sqrt a}.
> $$
> Therefore,
> $$
> \boxed{L(x)=\sqrt a+\frac{b}{2\sqrt a}x}.
> $$
> A second way to check the result is
> $$
> \sqrt{a+bx}=\sqrt a\left(1+\frac{b}{a}x\right)^{1/2}
> \approx\sqrt a\left(1+\frac{b}{2a}x\right).
> $$
> The condition $a>0$ ensures that the function is real-valued and has a finite derivative near the expansion point.
> <!-- bilingual-en:end -->

> [!example]- 2A-3：$\dfrac{(1+x)^{3/2}}{1+2x}$ 的线性化
> 用基本式
> $$
> (1+x)^{3/2}\approx1+\frac32x,\qquad
> (1+2x)^{-1}\approx1-2x.
> $$
> 相乘并去掉 $x^2$：
> $$
> \boxed{\frac{(1+x)^{3/2}}{1+2x}\approx1-\frac12x}.
> $$
> 直接求导在 $0$ 得 $f(0)=1,f'(0)=-1/2$，结果一致。
> <!-- bilingual-en:start -->
> Use the standard approximations
> $$
> (1+x)^{3/2}\approx1+\frac32x,\qquad
> (1+2x)^{-1}\approx1-2x.
> $$
> Multiply and discard terms of order $x^2$ and higher:
> $$
> \boxed{\frac{(1+x)^{3/2}}{1+2x}\approx1-\frac12x}.
> $$
> Direct differentiation gives $f(0)=1$ and $f'(0)=-1/2$, confirming the result.
> <!-- bilingual-en:end -->

> [!example]- 2A-6：$\tan\theta$ 在 $0$ 的二次近似
> $$
> f(0)=0,\quad f'(\theta)=\sec^2\theta,\ f'(0)=1,
> $$
> $$
> f''(\theta)=2\sec^2\theta\tan\theta,\ f''(0)=0.
> $$
> 所以
> $$
> \boxed{\tan\theta\approx\theta}.
> $$
> 二次项恰为零；第一个未记录的非零修正其实是三次项。
> <!-- bilingual-en:start -->
> $$
> f(0)=0,\quad f'(\theta)=\sec^2\theta,\ f'(0)=1,
> $$
> $$
> f''(\theta)=2\sec^2\theta\tan\theta,\ f''(0)=0.
> $$
> Hence,
> $$
> \boxed{\tan\theta\approx\theta}.
> $$
> The quadratic coefficient is exactly zero; the first omitted nonzero correction is cubic.
> <!-- bilingual-en:end -->

> [!example]- 2A-11：理想气体的二次近似
> 由 $pv^k=C$，
> $$
> p=Cv^{-k}.
> $$
> 写 $v=v_0+\Delta v=v_0(1+u)$，其中 $u=\Delta v/v_0$：
> $$
> p=\frac{C}{v_0^k}(1+u)^{-k}.
> $$
> 应用 $(1+u)^r$ 的二次式，取 $r=-k$：
> $$
> \boxed{
> p\approx\frac{C}{v_0^k}
> \left[
> 1-k\frac{\Delta v}{v_0}
> +\frac{k(k+1)}2\left(\frac{\Delta v}{v_0}\right)^2
> \right]}.
> $$
> 小参数必须是无量纲相对改变量 $\Delta v/v_0$。
> <!-- bilingual-en:start -->
> From $pv^k=C$,
> $$
> p=Cv^{-k}.
> $$
> Write $v=v_0+\Delta v=v_0(1+u)$, where $u=\Delta v/v_0$:
> $$
> p=\frac{C}{v_0^k}(1+u)^{-k}.
> $$
> Apply the quadratic approximation for $(1+u)^r$ with $r=-k$:
> $$
> \boxed{
> p\approx\frac{C}{v_0^k}
> \left[
> 1-k\frac{\Delta v}{v_0}
> +\frac{k(k+1)}2\left(\frac{\Delta v}{v_0}\right)^2
> \right]}.
> $$
> The small parameter must be the dimensionless relative change $\Delta v/v_0$.
> <!-- bilingual-en:end -->

> [!example]- 2A-12a：$\dfrac{e^x}{1-x}$ 的二次近似
> $$
> e^x\approx1+x+\frac{x^2}{2},\qquad
> \frac1{1-x}\approx1+x+x^2.
> $$
> 乘积截至二次：
> $$
> \boxed{\frac{e^x}{1-x}\approx1+2x+\frac52x^2}.
> $$
> <!-- bilingual-en:start -->
> $$
> e^x\approx1+x+\frac{x^2}{2},\qquad
> \frac1{1-x}\approx1+x+x^2.
> $$
> Multiplying and retaining terms through degree two gives
> $$
> \boxed{\frac{e^x}{1-x}\approx1+2x+\frac52x^2}.
> $$
> <!-- bilingual-en:end -->

> [!example]- 2A-12d：$\ln(\cos x)$ 的二次近似
> 先用 $\cos x\approx1-x^2/2$。令 $u=-x^2/2$，则
> $$
> \ln(\cos x)\approx\ln(1+u)\approx u=-\frac{x^2}{2}.
> $$
> 因 $u^2$ 已是四次，不进入二次近似。答案：
> $$
> \boxed{\ln(\cos x)\approx-\frac{x^2}{2}}.
> $$
> <!-- bilingual-en:start -->
> First use $\cos x\approx1-x^2/2$. Let $u=-x^2/2$; then
> $$
> \ln(\cos x)\approx\ln(1+u)\approx u=-\frac{x^2}{2}.
> $$
> Since $u^2$ is already fourth order, it does not contribute to the quadratic approximation. Thus,
> $$
> \boxed{\ln(\cos x)\approx-\frac{x^2}{2}}.
> $$
> <!-- bilingual-en:end -->

> [!example]- 2A-12e：$x\ln x$ 在 $x=1$ 的二次近似
> 令 $h=x-1$，则 $x=1+h$：
> $$
> x\ln x=(1+h)\ln(1+h)
> \approx(1+h)\left(h-\frac{h^2}{2}\right)
> \approx h+\frac{h^2}{2}.
> $$
> 因此
> $$
> \boxed{x\ln x\approx(x-1)+\frac12(x-1)^2}.
> $$
> <!-- bilingual-en:start -->
> Let $h=x-1$, so $x=1+h$:
> $$
> x\ln x=(1+h)\ln(1+h)
> \approx(1+h)\left(h-\frac{h^2}{2}\right)
> \approx h+\frac{h^2}{2}.
> $$
> Therefore,
> $$
> \boxed{x\ln x\approx(x-1)+\frac12(x-1)^2}.
> $$
> <!-- bilingual-en:end -->

### 2B：Curve Sketching

<!-- bilingual-en:start -->
*2B: Curve sketching*
<!-- bilingual-en:end -->
> [!example]- 2B-2a、2B-2e、2B-2h：拐点
> **(a)** $f=x^3-3x+1$，$f''=6x$，在 $0$ 两侧变号，拐点 $\boxed{(0,1)}$。
> **(e)** $f=x/(x+4)$，$f''=-8/(x+4)^3$。符号在 $-4$ 两侧改变，但 $f$ 在 $-4$ 不连续，所以 $\boxed{\text{无拐点}}$。
> **(h)** $f=e^{-x^2}$，
> $$
> f''=(4x^2-2)e^{-x^2}.
> $$
> 指数因子恒正，故在 $x=\pm1/\sqrt2$ 变号，拐点为
> $$
> \boxed{\left(\pm\frac1{\sqrt2},e^{-1/2}\right)}.
> $$
> <!-- bilingual-en:start -->
> **(a)** For $f=x^3-3x+1$, $f''=6x$ changes sign across $0$, so the inflection point is $\boxed{(0,1)}$.
> **(e)** For $f=x/(x+4)$, $f''=-8/(x+4)^3$. Its sign changes across $-4$, but $f$ is discontinuous there, so $\boxed{\text{there is no inflection point}}$.
> **(h)** For $f=e^{-x^2}$,
> $$
> f''=(4x^2-2)e^{-x^2}.
> $$
> The exponential factor is always positive, so $f''$ changes sign at $x=\pm1/\sqrt2$. The inflection points are
> $$
> \boxed{\left(\pm\frac1{\sqrt2},e^{-1/2}\right)}.
> $$
> <!-- bilingual-en:end -->

> [!example]- 2B-4：由文字描述作图并判断绝对极值
> $f$ 在 $[0,10]$ 连续，零点为 $4,7,9$；在 $(0,5),(8,10)$ 递增，在 $(5,8)$ 递减。
> 因而 $x=5$ 是局部最大，$x=8$ 是局部最小。符号还可确定：$f(4)=0$ 且递增，所以 $f(5)>0$；从 $5$ 递减并在 $7$ 过零，故 $f(8)<0$；之后递增并在 $9$ 过零。
> 绝对最大只能在 $\boxed{x=5\text{ 或 }x=10}$；绝对最小只能在 $\boxed{x=0\text{ 或 }x=8}$。题目未给端点高度，不能再唯一决定。
> <!-- bilingual-en:start -->
> The function $f$ is continuous on $[0,10]$, has zeros at $4,7,9$, increases on $(0,5)$ and $(8,10)$, and decreases on $(5,8)$.
> Hence $x=5$ is a local maximum and $x=8$ is a local minimum. The signs can also be inferred: because $f(4)=0$ and the function is increasing, $f(5)>0$; it then decreases through zero at $7$, so $f(8)<0$; afterward it increases through zero at $9$.
> The absolute maximum can occur only at $\boxed{x=5\text{ or }x=10}$, and the absolute minimum only at $\boxed{x=0\text{ or }x=8}$. Since the endpoint values are not given, the answer cannot be narrowed further.
> <!-- bilingual-en:end -->

> [!example]- 2B-6a、2B-6b：指定极值位置构造三次函数
> 希望 $x=-1,1$ 是导数的零点，最简单选
> $$
> f'(x)=3(x+1)(x-1)=3x^2-3.
> $$
> 积分得
> $$
> \boxed{f(x)=x^3-3x+C}.
> $$
> 取 $C=0$。符号为 $+\,-\,+$，所以 $(-1,2)$ 是局部最大、$(1,-2)$ 是局部最小；端点 $f(-3)=-18,f(3)=18$，且函数为奇函数，据此完成 $[-3,3]$ 图像。
> <!-- bilingual-en:start -->
> To make $x=-1$ and $x=1$ zeros of the derivative, choose the simplest suitable derivative:
> $$
> f'(x)=3(x+1)(x-1)=3x^2-3.
> $$
> Integrating gives
> $$
> \boxed{f(x)=x^3-3x+C}.
> $$
> Take $C=0$. The sign pattern of $f'$ is $+\,-\,+$, so $(-1,2)$ is a local maximum and $(1,-2)$ is a local minimum. Also, $f(-3)=-18$, $f(3)=18$, and $f$ is odd; these facts determine the sketch on $[-3,3]$.
> <!-- bilingual-en:end -->

> [!example]- 2B-7a、2B-7b：递增函数在可导点的导数
> 若 $f$ 递增且在 $a$ 可导。对 $h>0$，$f(a+h)-f(a)\ge0$，故差商 $\ge0$；对 $h<0$，分子 $\le0$、分母 $<0$，差商仍 $\ge0$。两侧极限相等且存在，故
> $$
> \boxed{f'(a)\ge0}.
> $$
> 不能加强成 $f'(a)>0$，因为一串正数的极限可以是 $0$。反例 $f(x)=x^3$ 严格递增，但 $f'(0)=0$。
> <!-- bilingual-en:start -->
> Suppose $f$ is increasing and differentiable at $a$. If $h>0$, then $f(a+h)-f(a)\ge0$, so the difference quotient is nonnegative. If $h<0$, the numerator is nonpositive and the denominator is negative, so the quotient is again nonnegative. Since the two one-sided limits exist and agree,
> $$
> \boxed{f'(a)\ge0}.
> $$
> This cannot be strengthened to $f'(a)>0$, because a sequence of positive numbers may converge to $0$. For example, $f(x)=x^3$ is strictly increasing but $f'(0)=0$.
> <!-- bilingual-en:end -->

### Problem Set 3 错误检查
<!-- bilingual-en:start -->
*Problem Set 3 Bugcheck*
<!-- bilingual-en:end -->

- 拐点必须是函数图像上的点，不能只写 $x$ 值而忽略连续性。
- 2A-12a 的本地文本抽取容易把分母排成 $1-2x$；由官方答案 $1+2x+\frac52x^2$ 可确认原式是 $1-x$。
- 构造函数时加任意常数不改变导数信息，因此答案通常不唯一。
<!-- bilingual-en:start -->
- An inflection point must lie on the graph of the function; giving only an $x$-value while ignoring continuity is insufficient.
- Local text extraction for 2A-12a can misread the denominator as $1-2x$. The official answer $1+2x+\frac52x^2$ confirms that the original denominator is $1-x$.
- Adding an arbitrary constant when constructing a function does not change its derivative information, so the answer is usually not unique.
<!-- bilingual-en:end -->

**本组小结：**PS3 把“按阶展开”和“按导数符号读图”连在一起：近似研究一个点附近，作图研究由这些局部信息拼成的全局结构。
<!-- bilingual-en:start -->
** Summary of this group:**PS3 Joins "expansion by order" and "reading by derivative symbols": Approximately studies the vicinity of a point, and plots the global structure composed of these local information.
<!-- bilingual-en:end -->

---

## Part B：Optimization, Related Rates and Newton’s Method

## Session 29：Optimization Problems

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题：**怎样不画出整张精确图，就找到函数的最大/最小值？为什么只检查 $f'=0$ 可能得到“最坏答案”？
<!-- bilingual-en:start -->
**Question:** How can we find a function's maximum or minimum without drawing its exact graph, and why is checking only $f'=0$ a worst-case mistake?
<!-- bilingual-en:end -->

**前置：**临界点、端点、间断点和一阶导数符号。
<!-- bilingual-en:start -->
**Prerequisites:** Critical points, endpoints, discontinuities, and the sign of the first derivative.
<!-- bilingual-en:end -->

### 29a：极值候选清单
<!-- bilingual-en:start -->
*29a: Extreme candidate list*
<!-- bilingual-en:end -->

[[导数的应用#优化：把目标变成一阶条件|导数优化]]先列候选。若 $f$ 在闭区间 $[a,b]$ 连续，极值定理保证绝对最大、最小存在。它们只能出现在：
<!-- bilingual-en:start -->
[[导数的应用#优化：把目标变成一阶条件|derivative optimization]] first.  If $f$ is $[a,b]$ continuous in the closed interval, the extremum theorem guarantees absolute maximum and minimum existence.  They can only appear in:
<!-- bilingual-en:end -->

1. 内点且 $f'(x)=0$；
2. 内点且 $f'$ 不存在；
3. 端点 $a,b$。
<!-- bilingual-en:start -->
1. interior point and $f'(x)=0$;
2. Inner point and $f'$ does not exist;
3. Endpoint $a,b$.
<!-- bilingual-en:end -->

若定义域不是闭区间，还必须检查开端点的单侧极限、无穷远和间断点附近。求出候选只是第一步，最后要**比较目标函数值**。
<!-- bilingual-en:start -->
If the domain is not a closed interval, also check one-sided limits at open endpoints, behavior at infinity, and behavior near discontinuities. Finding candidates is only the first step; the final step is to **compare the objective-function values**.
<!-- bilingual-en:end -->

### 29b：一根铁丝围两个正方形
<!-- bilingual-en:start -->
*29b: A wire encloses two squares*
<!-- bilingual-en:end -->

总长为 $1$，一段长 $x$，另一段长 $1-x$，所以
<!-- bilingual-en:start -->
The total length is $1$, one length is $x$, and the other length is $1-x$, so
<!-- bilingual-en:end -->

$$
0\le x\le1.
$$

两正方形边长分别为 $x/4,(1-x)/4$，总面积
<!-- bilingual-en:start -->
The length of the two squares is $x/4,(1-x)/4$, and the total area of the two squares is respectively
<!-- bilingual-en:end -->

$$
A(x)=\frac{x^2}{16}+\frac{(1-x)^2}{16}.
$$

求导：
<!-- bilingual-en:start -->
Differentiate:
<!-- bilingual-en:end -->

$$
A'(x)=\frac{2x-1}{8}.
$$

唯一内点候选 $x=1/2$，此时
<!-- bilingual-en:start -->
$x=1/2$, which is the only inside point candidate
<!-- bilingual-en:end -->

$$
A(1/2)=\frac1{32}.
$$

端点：
<!-- bilingual-en:start -->
Endpoint:
<!-- bilingual-en:end -->

$$
A(0)=A(1)=\frac1{16}.
$$

所以等分铁丝给出**最小**总面积 $1/32$；把全部铁丝给一个正方形给出最大面积 $1/16$。如果只算临界点，就会把题目所求最大值误判成最小值。
<!-- bilingual-en:start -->
Thus splitting the wire equally gives the **minimum** total area, $1/32$, while assigning the entire wire to one square gives the maximum area, $1/16$. If we examine only the critical point, we mistake the problem's minimum for its maximum.
<!-- bilingual-en:end -->

### 极大/极小、极值点与极值
<!-- bilingual-en:start -->
*Maximum/Minimum, Extreme Point and Extreme Value*
<!-- bilingual-en:end -->

- “最大值是多少？”答函数值，例如 $1/16$。
- “在哪里达到？”答自变量，例如 $x=0$ 或 $1$。
- 图像上的完整信息是点 $(x,A(x))$。
- 局部极值只比较附近；绝对极值比较整个定义域。
<!-- bilingual-en:start -->
- "What is the maximum value?" Answer function value, for example, $1/16$.
- "Where do you get it?" Answer arguments, such as $x=0$ or $1$.
- The complete graphical information is the point $(x,A(x))$.
- Local extrema compare nearby points; absolute extrema compare the entire domain.
<!-- bilingual-en:end -->

### 易错点与边界
<!-- bilingual-en:start -->
*Fault-prone points and boundaries*
<!-- bilingual-en:end -->

- 没有把几何限制翻译为定义域。
- 找到临界点就停止，不比较端点。
- 把“最大面积”和“最大面积发生的位置”混为一谈。
- 对开放定义域写出并不存在的“端点取值”，应使用极限。
<!-- bilingual-en:start -->
- Failing to translate geometric constraints into a feasible domain.
- Stop when a critical point is found, without comparing endpoints.
- Confuse "maximum area" with "location where maximum area occurs".
- Write an "endpoint value" for an open domain that does not exist, and use limits.
<!-- bilingual-en:end -->

> [!question]- 三道自检题与答案
> 1. 连续函数在闭区间为何一定要查端点？
> 绝对极值可以在端点达到，而端点不要求导数为零。
>
> 2. 若 $f'$ 在内点不存在，该点是否自动为极值？
> 不是，只是候选；例如 $x^{1/3}$ 在 $0$ 导数无穷但仍递增。
>
> 3. 本例 $A''=1/4>0$ 能说明什么？
> 内点 $x=1/2$ 是局部最小；最大值仍需由端点比较确定。
> <!-- bilingual-en:start -->
> 1. Why do continuous functions have to look up endpoints in a closed interval?
> Absolute extremes can be reached at endpoints that do not require a derivative of zero.
> 2. If the $f'$ internal point does not exist, is it automatically extremum?
> No. It is only a candidate; for example, $x^{1/3}$ has an infinite slope at $0$ but remains increasing through that point.
> 3. What does this $A''=1/4>0$ tell you?
> The interior point $x=1/2$ is a local minimum, and the maximum value is still determined by the comparison of endpoints.
> <!-- bilingual-en:end -->

### 本地材料

- [[Ses29a_Lecture_Notes.pdf#page=1|29a Introduction to Maxima and Minima]]
- [[Ses29b_Lecture_Notes.pdf#page=1|29b Maximum Area of Two Squares]]

**知识链小结：**优化的可靠流程是“建模并定域 → 列全部候选 → 比较目标值 → 用单位和图形复核”。
<!-- bilingual-en:start -->
** Knowledge Chain Summary: The reliable process for ** optimization is "Modeling and Localizing → Column All Candidates → Comparing Target Values → Unit and Graph Reviews."
<!-- bilingual-en:end -->

## Session 30：Optimization Problems II

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题：**目标函数含多个变量并受约束时，怎样降为一个自由变量？隐式求导何时更短？
<!-- bilingual-en:start -->
**Question:** When a constrained objective function contains several variables, how can it be reduced to one free variable? When is implicit differentiation shorter?
<!-- bilingual-en:end -->

**前置：**Session 29 的候选比较、隐函数求导。
<!-- bilingual-en:start -->
**Prerequisites:** Candidate comparison from Session 29 and implicit differentiation.
<!-- bilingual-en:end -->

### 30a：无盖方盒的最小表面积
<!-- bilingual-en:start -->
*30a: Minimum surface area of uncapped box*
<!-- bilingual-en:end -->

设正方形底边为 $x>0$、高为 $y>0$，固定体积 $V$：
<!-- bilingual-en:start -->
Suppose the square bottom edge is $x>0$, the height is $y>0$, the fixed volume is $V$:
<!-- bilingual-en:end -->

$$
V=x^2y.
$$

无盖盒表面积
<!-- bilingual-en:start -->
lidless box surface area
<!-- bilingual-en:end -->

$$
A=x^2+4xy.
$$

由约束 $y=V/x^2$ 消去 $y$：
<!-- bilingual-en:start -->
The constraint $y=V/x^2$ eliminates $y$:
<!-- bilingual-en:end -->

$$
A(x)=x^2+\frac{4V}{x}.
$$

$$
A'(x)=2x-\frac{4V}{x^2}.
$$

令 $A'=0$：
<!-- bilingual-en:start -->
Let $A'=0$:
<!-- bilingual-en:end -->

$$
x^3=2V,\qquad x=(2V)^{1/3}.
$$

相应
<!-- bilingual-en:start -->
corresponding
<!-- bilingual-en:end -->

$$
y=\frac{V}{x^2}=2^{-2/3}V^{1/3},
\qquad\boxed{\frac{x}{y}=2}.
$$

边界检查：
<!-- bilingual-en:start -->
Boundary checking:
<!-- bilingual-en:end -->

$$
\lim_{x\to0^+}A(x)=\infty,\qquad
\lim_{x\to\infty}A(x)=\infty.
$$

且只有一个临界点，所以它给出全局最小。比例 $x/y=2$ 是无量纲结论，比含 $V$ 的尺寸式更能说明“最佳形状”。
<!-- bilingual-en:start -->
And there is only one critical point, so it gives the global minimum.  The scale $x/y=2$ is a dimensionless conclusion that better describes the "best shape" than the dimensional formula containing $V$.
<!-- bilingual-en:end -->

### 30b：用隐式求导直接得到比例
<!-- bilingual-en:start -->
*30b: The ratio is obtained directly by implicit derivation*
<!-- bilingual-en:end -->

保持 $V$ 不变，对约束求导：
<!-- bilingual-en:start -->
Keeping $V$ unchanged, the constraint derivation:
<!-- bilingual-en:end -->

$$
0=\frac{dV}{dx}=2xy+x^2y'
\quad\Rightarrow\quad
y'=-\frac{2y}{x}.
$$

对面积求导：
<!-- bilingual-en:start -->
Derive area:
<!-- bilingual-en:end -->

$$
\frac{dA}{dx}=2x+4y+4xy'
=2x-4y.
$$

临界条件直接给
<!-- bilingual-en:start -->
critical condition directly to
<!-- bilingual-en:end -->

$$
\boxed{x=2y}.
$$

此法更快地产生比例，但没有自动完成“这是最小而非最大”的边界论证，仍要补查。
<!-- bilingual-en:start -->
This method produces proportions more quickly, but does not automatically complete the "This is the smallest, not the largest" boundary argument and still needs to be reexamined.
<!-- bilingual-en:end -->

### 配套练习：体积 $1000\ \mathrm{cm^3}$ 的封闭圆罐
<!-- bilingual-en:start -->
*Complementary Exercise: Closed Cylinder of $1000\ \mathrm{cm^3}$ Volume*
<!-- bilingual-en:end -->

圆罐表面积与体积为
<!-- bilingual-en:start -->
The surface area and volume of the circular tank are
<!-- bilingual-en:end -->

$$
S=2\pi r^2+2\pi rh,\qquad
1000=\pi r^2h.
$$

代入 $h=1000/(\pi r^2)$：
<!-- bilingual-en:start -->
Introduce $h=1000/(\pi r^2)$:
<!-- bilingual-en:end -->

$$
S(r)=2\pi r^2+\frac{2000}{r}.
$$

$$
S'(r)=4\pi r-\frac{2000}{r^2}=0
\Rightarrow r^3=\frac{500}{\pi}.
$$

故
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

$$
\boxed{r=\left(\frac{500}{\pi}\right)^{1/3}\approx5.42\ \mathrm{cm}},
\qquad
\boxed{h=2r\approx10.84\ \mathrm{cm}}.
$$

$r\to0^+$ 或 $r\to\infty$ 时 $S\to\infty$，故这是全局最小。
<!-- bilingual-en:start -->
$S\to\infty$ when $r\to0^+$ or $r\to\infty$, so this is the global minimum.
<!-- bilingual-en:end -->

### 易错点与边界
<!-- bilingual-en:start -->
*Fault-prone points and boundaries*
<!-- bilingual-en:end -->

- 约束方程和目标函数角色不同：一个消元，一个优化。
- 把固定量 $V$ 在求导时当变量。
- 隐式法得到临界比例后漏掉边界检查。
- 尺寸带单位，比例无单位；答案应按题目要求给全。
<!-- bilingual-en:start -->
- Different roles for constraint equations and objective functions: one elimination, one optimization.
- Make the fixed quantity $V$ a variable when deriving.
- Implicit method missing boundary check after getting critical ratio.
- Dimensions are in units and scale is in units; answers should be given in full as required by the topic.
<!-- bilingual-en:end -->

> [!question]- 三道自检题与答案
> 1. 为什么无盖方盒只有两个自由尺寸却最终是一元问题？
> 固定体积约束把 $x,y$ 关联，只剩一个自由度。
>
> 2. 若圆罐无顶，最优比例还会是 $h=2r$ 吗？
> 不会；目标表面积改变，必须重新求导。
>
> 3. $S''>0$ 是否可以替代全部边界检查？
> 若能证明定义域上处处 $S''>0$，临界点唯一且为全局最小；但仍应说明定义域和极端形状。
> <!-- bilingual-en:start -->
> 1. Why is it that an uncapped box has only two free dimensions and is ultimately a unitary problem?
> The fixed volume constraint associates the $x,y$ with only one degree of freedom remaining.
> 2. If the pot is uncapped, will the optimal ratio be $h=2r$?
> No; the target surface area changes and must be rederived.
> 3. Can $S''>0$ override all bounds checking?
> If the $S''>0$ at the domain can be proved, the critical point is unique and global minimum; however, the domain and extreme shape should be described.
> <!-- bilingual-en:end -->

### 本地材料

- [[Ses30a_Lecture_Notes.pdf#page=1|30a Open Box Optimization]]
- [[Ses30b_Lecture_Notes.pdf#page=1|30b Implicit Differentiation and Min/Max]]
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise030_Problems.pdf#page=1|Exercise 30 原题]] · [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise030_Solutions.pdf#page=1|Exercise 30 解答]]

**知识链小结：**约束把多个变量压缩成一个自由度；显式消元利于边界分析，隐式求导利于快速得到最优比例。
<!-- bilingual-en:start -->
**Knowledge chain summary:** A constraint compresses several variables to one degree of freedom. Explicit elimination is useful for boundary analysis, while implicit differentiation can reveal the optimal ratio quickly.
<!-- bilingual-en:end -->

## Session 31：Related Rates

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题：**[[导数的应用#相关变化率与弹性|相关变化率]]中，多个随时间变化的量由一个几何方程联系时，怎样从已知速率求未知速率？
<!-- bilingual-en:start -->
**Question:** In a [[导数的应用#相关变化率与弹性|related-rates]] problem, several time-varying quantities are connected by a geometric equation. How can a known rate be used to find an unknown one?
<!-- bilingual-en:end -->

**前置：**链式法则、隐函数求导、相似三角形、常见面积与体积公式。
<!-- bilingual-en:start -->
**Prerequisites:** The chain rule, implicit differentiation, similar triangles, and standard area and volume formulas.
<!-- bilingual-en:end -->

### 31a–31b：路边雷达
<!-- bilingual-en:start -->
*31a-31b: Roadside radar*
<!-- bilingual-en:end -->

警车距道路垂直距离 $30$ ft；车与警车直线距离 $D=50$ ft 时，雷达测得
<!-- bilingual-en:start -->
The distance between the police car and the road is $30$ ft, and the distance between the car and the police car is $D=50$ ft
<!-- bilingual-en:end -->

$$
\frac{dD}{dt}=-80\ \mathrm{ft/s}.
$$

设车沿道路到垂足的有向距离为 $x$。几何关系
<!-- bilingual-en:start -->
The distance from the vehicle to the foot is $x$.  geometric relation
<!-- bilingual-en:end -->

$$
x^2+30^2=D^2.
$$

此刻 $x=40$。**先求导、后代数值：**
<!-- bilingual-en:start -->
$x=40$.  ** Derive First, Descendent Value: **
<!-- bilingual-en:end -->

$$
2x\frac{dx}{dt}=2D\frac{dD}{dt}.
$$

因此
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

$$
\frac{dx}{dt}
=\frac{D}{x}\frac{dD}{dt}
=\frac{50}{40}(-80)
=-100\ \mathrm{ft/s}.
$$

负号表示朝垂足方向运动；速率大小 $100\ \mathrm{ft/s}\approx68.2\ \mathrm{mph}$，超过 $65\ \mathrm{mph}$。
<!-- bilingual-en:start -->
A negative sign indicates a vertical motion; the rate size is $100\ \mathrm{ft/s}\approx68.2\ \mathrm{mph}$, exceeding $65\ \mathrm{mph}$.
<!-- bilingual-en:end -->

> [!warning] 为何不能先代 $D=50$？
> $D$ 是随时间变化的函数，只在所问瞬间取值 $50$。若求导前替成常数，错误得到 $D'=0$，抹掉了已知速率。
> <!-- bilingual-en:start -->
> $D$ is a function that changes with time and takes the value of $50$ only at the instant being asked.  If the derivative is replaced by a constant, the error is $D'=0$, which erases the known rate.
> <!-- bilingual-en:end -->

### 31c：锥形水箱
<!-- bilingual-en:start -->
*31c: Tapered tank*
<!-- bilingual-en:end -->

水箱高 $10$ ft、顶半径 $4$ ft；水深 $h$、水面半径 $r$。相似三角形：
<!-- bilingual-en:start -->
The height of the water tank is $10$ ft, the top radius is $4$ ft, the water depth is $h$, and the water surface radius is $r$.  Similar triangle:
<!-- bilingual-en:end -->

$$
\frac{r}{h}=\frac4{10}
\Rightarrow r=\frac25h.
$$

水体积
<!-- bilingual-en:start -->
volume of water
<!-- bilingual-en:end -->

$$
V=\frac13\pi r^2h
=\frac{4\pi}{75}h^3.
$$

求时间导数：
<!-- bilingual-en:start -->
To calculate the time derivative:
<!-- bilingual-en:end -->

$$
\frac{dV}{dt}
=\frac{4\pi}{25}h^2\frac{dh}{dt}.
$$

若 $dV/dt=2\ \mathrm{ft^3/min}$ 且 $h=5$ ft，
<!-- bilingual-en:start -->
If $dV/dt=2\ \mathrm{ft^3/min}$ and $h=5$ ft,
<!-- bilingual-en:end -->

$$
2=4\pi\frac{dh}{dt}
\Rightarrow
\boxed{\frac{dh}{dt}=\frac1{2\pi}\ \mathrm{ft/min}}.
$$

![[98_attachment/MIT18.01SC/unit02-related-rates-cone.png|720]]

### 一般流程
<!-- bilingual-en:start -->
*General Process*
<!-- bilingual-en:end -->

1. 画图，标出**会变**与**不变**的量。
2. 用几何/物理关系写一个在所有时刻成立的方程。
3. 对时间 $t$ 求导，给每个变量补链式因子。
4. 再代入所问瞬间的数值，解未知速率。
5. 检查符号、单位和量级。
<!-- bilingual-en:start -->
1. Paint to indicate the amount by which ** changes ** and ** does not change **.
2. Writing an equation which is established at all times by using the geometric/physical relation.
3. Derive the time $t$ and supply the chain factor for each variable.
4. The unknown rate is solved by substituting the values of the instant in question.
5. Check symbols, units and scales.
<!-- bilingual-en:end -->

### 配套练习：半球顶粮仓的隐式优化
<!-- bilingual-en:start -->
*Supporting Exercise: Implicit Optimization of Hemisphere Top Grain Warehouse*
<!-- bilingual-en:end -->

设柱高 $h$、半径 $r$，体积
<!-- bilingual-en:start -->
The height of the column is $h$, the radius is $r$, and the volume is 100%
<!-- bilingual-en:end -->

$$
V=\pi r^2h+\frac23\pi r^3.
$$

固定 $V$ 对 $r$ 求导：
<!-- bilingual-en:start -->
Fixed $V$ to $r$ derivative:
<!-- bilingual-en:end -->

$$
0=2\pi rh+\pi r^2h'+2\pi r^2
\Rightarrow h'=-2\frac{h+r}{r}.
$$

有地板时 $S=2\pi rh+3\pi r^2$：
<!-- bilingual-en:start -->
$S=2\pi rh+3\pi r^2$ with flooring:
<!-- bilingual-en:end -->

$$
S'=2\pi h+2\pi rh'+6\pi r=2\pi(r-h).
$$

临界条件 $h=r$，结合体积得
<!-- bilingual-en:start -->
The critical condition is $h=r$, and the binding volume is
<!-- bilingual-en:end -->

$$
\boxed{r=h=\left(\frac{3V}{5\pi}\right)^{1/3}},
$$

并由极端形状比较确认最小。无地板时 $S=2\pi rh+2\pi r^2$，同法得 $S'=-2\pi h<0$，最小出现在边界 $h=0$，即只有半球。该练习再次说明：导数零点之外必须检查边界。
<!-- bilingual-en:start -->
The minimum is confirmed by the extreme shape comparison.  $S=2\pi rh+2\pi r^2$ without floor, $S'=-2\pi h<0$, $h=0$, i.e. only hemisphere.  This exercise again shows that you must check the boundary outside of the zero derivative.
<!-- bilingual-en:end -->

### 易错点与边界
<!-- bilingual-en:start -->
*Fault-prone points and boundaries*
<!-- bilingual-en:end -->

- 把“正在减小”的已知速率写成正数。
- 对 $x(t)^2$ 求导漏写 $2xx'$。
- 相似三角形中的 $r,h$ 必须来自水体小锥，而不是混用水箱固定尺寸。
- 速率的单位比原变量多一个“每时间”。
<!-- bilingual-en:start -->
- Write the known rate of "decreasing" as a positive number.
- Leaves $2xx'$ to $x(t)^2$.
- The $r,h$ in a similar triangle must come from a water cone, not a mixing tank fixed dimension.
- The rate has one more "per time" per unit than the original variable.
<!-- bilingual-en:end -->

> [!question]- 三道自检题与答案
> 1. $A=\pi r^2$，若 $r'=3$，则 $A'$？
> $A'=2\pi rr'=6\pi r$；没有当前 $r$ 就不能给数值。
>
> 2. 相关变化率为何本质上是链式法则？
> 几何量都是 $t$ 的复合函数，例如 $dV(h(t))/dt=(dV/dh)h'$。
>
> 3. 何时可以在求导前代常数？
> 只有题意保证该量在整个过程中恒定，例如警车到道路的垂距 $30$。
> <!-- bilingual-en:start -->
> 1. $A=\pi r^2$, if $r'=3$, then $A'$?
> $A'=2\pi rr'=6\pi r$; cannot give a numeric value without the current $r$.
> 2. Why is the relevant rate of change inherently chained?
> Geometric quantities are all composite functions of $t$, such as $dV(h(t))/dt=(dV/dh)h'$.
> 3. When can the derivative precedent constants be obtained?
> Only the question ensures that the amount is constant throughout the process, such as the vertical distance from the police car to the road $30$.
> <!-- bilingual-en:end -->

### 本地材料

- [[Ses31a_Lecture_Notes.pdf#page=1|31a Radar Setup]]
- [[Ses31b_Lecture_Notes.pdf#page=1|31b Radar Calculation]]
- [[Ses31c_Lecture_Notes.pdf#page=1|31c Conical Tank]]
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise031_Problems.pdf#page=1|Exercise 31 原题]] · [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise031_Solutions.pdf#page=1|Exercise 31 解答]]

**知识链小结：**相关变化率先保留变量之间的动态关系，再在最后冻结到某个瞬间；“先求导后代值”是核心纪律。
<!-- bilingual-en:start -->
**Knowledge chain summary:** In related-rates problems, preserve the dynamic relationship among the variables first and substitute the values for a particular instant only at the end. "Differentiate first, substitute later" is the essential discipline.
<!-- bilingual-en:end -->

## Session 32：Ring on a String

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题：**一个圆环在固定长度的绳上自由滑动，最低点为何满足两侧绳与竖直方向夹角相等？
<!-- bilingual-en:start -->
**Question:**A ring slides freely on a fixed length rope. Why does the lowest point meet the condition that the angles between the two sides of the rope are equal to the vertical direction?
<!-- bilingual-en:end -->

**前置：**距离公式、隐式求导、约束极小值。
<!-- bilingual-en:start -->
**Prerequisites:** The distance formula, implicit differentiation, and constrained minimization.
<!-- bilingual-en:end -->

### 建模：固定长度产生椭圆约束
<!-- bilingual-en:start -->
*Modeling: Fixed Length Generates Ellipse Constraints*
<!-- bilingual-en:end -->

绳端固定在 $(0,0)$ 与 $(a,b)$，圆环位于 $(x,y)$。总长 $L$ 固定：
<!-- bilingual-en:start -->
The rope ends are fixed at $(0,0)$ and $(a,b)$, and the ring is at $(x,y)$.  Total length $L$ fixed:
<!-- bilingual-en:end -->

$$
\sqrt{x^2+y^2}+\sqrt{(x-a)^2+(y-b)^2}=L.
$$

所有可行点构成以两端为焦点的椭圆。重力势能与高度 $y$ 成正比，因此稳定位置是约束曲线的最低点；若最低点位于光滑内点，则
<!-- bilingual-en:start -->
All feasible points form an ellipse whose foci are the fixed endpoints. Gravitational potential energy is proportional to height $y$, so the stable position is the lowest point of the constraint curve. If that lowest point is a smooth interior point, then
<!-- bilingual-en:end -->

$$
y'(x)=0.
$$

### 隐式求导与等角条件
<!-- bilingual-en:start -->
*Implicit differentiation and the equal-angle condition*
<!-- bilingual-en:end -->

对约束求 $x$ 导数：
<!-- bilingual-en:start -->
Differentiating the constraint with respect to $x$ gives
<!-- bilingual-en:end -->

$$
\frac{x+yy'}{\sqrt{x^2+y^2}}
+
\frac{x-a+(y-b)y'}{\sqrt{(x-a)^2+(y-b)^2}}
=0.
$$

在最低点 $y'=0$：
<!-- bilingual-en:start -->
At the lowest point $y'=0$:
<!-- bilingual-en:end -->

$$
\frac{x}{\sqrt{x^2+y^2}}
=
\frac{a-x}{\sqrt{(x-a)^2+(y-b)^2}}.
$$

两边分别是左右绳段与竖直线夹角 $\alpha,\beta$ 的正弦，所以
<!-- bilingual-en:start -->
The two sides are respectively the sine of the angle $\alpha,\beta$ between the left and right rope segments and the vertical line, so
<!-- bilingual-en:end -->

$$
\sin\alpha=\sin\beta.
$$

实际几何中两角均为锐角，故
<!-- bilingual-en:start -->
In real geometry, both corners are acute, so
<!-- bilingual-en:end -->

$$
\boxed{\alpha=\beta}.
$$

这同时是力学平衡条件：两侧张力的水平分量抵消；也是椭圆反射性质“入射角等于反射角”的微积分来源。
<!-- bilingual-en:start -->
This is also the condition for mechanical equilibrium: the horizontal components of the two tensions cancel. It is likewise the calculus source of the ellipse's reflection property, "angle of incidence equals angle of reflection."
<!-- bilingual-en:end -->

### 若继续求坐标
<!-- bilingual-en:start -->
*If we continue to find coordinates*
<!-- bilingual-en:end -->

由水平投影相加可得
<!-- bilingual-en:start -->
Available by Horizontal Projection Add
<!-- bilingual-en:end -->

$$
\sin\alpha=\frac{a}{L},
\qquad
L\cos\alpha=\sqrt{L^2-a^2}.
$$

竖直投影关系给
<!-- bilingual-en:start -->
vertical projection relation to
<!-- bilingual-en:end -->

$$
y=\frac12\left(b-\sqrt{L^2-a^2}\right).
$$

再由相似三角形可求
<!-- bilingual-en:start -->
then be solved by a similar triangle
<!-- bilingual-en:end -->

$$
x=\frac a2\left(1-\frac{b}{\sqrt{L^2-a^2}}\right).
$$

存在真实松弛位置还要求长度足以连接两端，且所得点属于相应椭圆下支。
<!-- bilingual-en:start -->
The presence of a true slack position also requires that the length be sufficient to connect the two ends and that the resulting points belong to the corresponding lower ellipse branch.
<!-- bilingual-en:end -->

### 易错点与边界
<!-- bilingual-en:start -->
*Fault-prone points and boundaries*
<!-- bilingual-en:end -->

- $y'=0$ 只适用于光滑内点；若绳长退化或最低点落在端部，要另查边界。
- 平方距离后求导常会引入不属于原椭圆的点；保留根式更忠实。
- $\sin\alpha=\sin\beta$ 一般还可能有补角；本题由角的几何范围排除。
<!-- bilingual-en:start -->
- $y'=0$ is only available for smooth interior points; if the length of the rope is degenerated or the lowest point is at the end, check the boundary separately.
- Squaring a distance equation can introduce points that do not satisfy the original ellipse; retaining the radicals preserves the original constraint more faithfully.
- $\sin\alpha=\sin\beta$ generally also has a supplementary angle; this problem is excluded by the geometry of the angle.
<!-- bilingual-en:end -->

> [!question]- 三道自检题与答案
> 1. $b=0$ 时公式给出什么？
> $x=a/2$，最低点在两端的水平中点下方。
>
> 2. 为什么最小化的是 $y$ 而不是绳长？
> 绳长已是约束常数；重力势能随高度 $y$ 增加。
>
> 3. 椭圆知识是否是求等角条件的前提？
> 不是；距离约束与隐式求导已经足够。
> <!-- bilingual-en:start -->
> 1. What does the formula give when $b=0$?
> $x=a/2$, the lowest point is below the horizontal midpoint at both ends.
> 2. Why is the $y$ minimized instead of the length of the rope?
> The length of the rope is already a constraint constant and the gravitational potential energy increases with the height $y$.
> 3. Is the knowledge of ellipse the premise of finding the equiangular condition?
> No. The distance constraint and implicit differentiation are sufficient.
> <!-- bilingual-en:end -->

### 本地材料

- [[Ses32a_Lecture_Notes.pdf#page=1|32a Ring on a String]]

**知识链小结：**约束极值把物理平衡、椭圆几何与隐式求导汇到同一个条件 $y'=0$。
<!-- bilingual-en:start -->
**Knowledge chain summary:** A constrained extremum brings physical equilibrium, ellipse geometry, and implicit differentiation together in the single condition $y'=0$.
<!-- bilingual-en:end -->

## Session 33：Newton’s Method

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题：**没有代数公式可解 $f(x)=0$ 时，怎样用切线快速逼近根？什么时候会失败？
<!-- bilingual-en:start -->
**Problem:**When there is no algebraic formula to solve $f(x)=0$, how to use tangent to approach root quickly?  When will it fail?
<!-- bilingual-en:end -->

**前置：**线性化、切线方程和迭代序列。
<!-- bilingual-en:start -->
**Leading: **Linearization, tangent equations, and iteration sequences.
<!-- bilingual-en:end -->

### 33a：迭代公式的推导
<!-- bilingual-en:start -->
*33a: Iterative Formula Derivation*
<!-- bilingual-en:end -->

[[数值求根：二分法与 Newton–Raphson#Newton–Raphson|Newton 法]]从已有近似 $x_n$ 出发，在它附近用切线
<!-- bilingual-en:start -->
Starting from the existing approximation $x_n$, [[数值求根：二分法与 Newton–Raphson#Newton–Raphson|Newton method]] uses a tangent near it
<!-- bilingual-en:end -->

$$
L_n(x)=f(x_n)+f'(x_n)(x-x_n)
$$

代替 $f$。让切线而非原曲线取零：
<!-- bilingual-en:start -->
Instead of $f$.  Zero the tangent instead of the original curve:
<!-- bilingual-en:end -->

$$
0=f(x_n)+f'(x_n)(x_{n+1}-x_n).
$$

若 $f'(x_n)\ne0$，
<!-- bilingual-en:start -->
If $f'(x_n)\ne0$,
<!-- bilingual-en:end -->

$$
\boxed{x_{n+1}=x_n-\frac{f(x_n)}{f'(x_n)}}.
$$

![[98_attachment/MIT18.01SC/unit02-newton-method.png|740]]

**例：求 $\sqrt5$。**令 $f(x)=x^2-5$：
<!-- bilingual-en:start -->
**Example: Find $\sqrt5$.** Let $f(x)=x^2-5$:
<!-- bilingual-en:end -->

$$
x_{n+1}=x_n-\frac{x_n^2-5}{2x_n}
=\frac12\left(x_n+\frac5{x_n}\right).
$$

取 $x_0=2$：
<!-- bilingual-en:start -->
Take $x_0=2$:
<!-- bilingual-en:end -->

$$
x_1=\frac94=2.25,\qquad
x_2=\frac{161}{72}\approx2.236111,
$$

已经非常接近 $\sqrt5\approx2.236068$。
<!-- bilingual-en:start -->
We're close to $\sqrt5\approx2.236068$.
<!-- bilingual-en:end -->

### 33b：为何通常非常快
<!-- bilingual-en:start -->
*33b: Why it's usually very fast*
<!-- bilingual-en:end -->

设真根为 $r$，误差 $e_n=x_n-r$。若 $f'(r)\ne0$ 且二阶导在附近受控，Taylor 展开给
<!-- bilingual-en:start -->
The real root is $r$, and the error is $e_n=x_n-r$.  If $f'(r)\ne0$ and the second derivative is controlled nearby, Taylor expands to
<!-- bilingual-en:end -->

$$
e_{n+1}\approx
\frac{f''(r)}{2f'(r)}e_n^2.
$$

误差大致平方，所以当误差小于 $1$ 后，有效数字常近似翻倍。这叫二次收敛（quadratic convergence），但它是局部结论。
<!-- bilingual-en:start -->
The error is about square, so when the error is less than $1$, the effective number is often approximately doubled.  This is called quadratic convergence, but it is a local conclusion.
<!-- bilingual-en:end -->

### 33c：四类失败
<!-- bilingual-en:start -->
*33c: Four types of failure*
<!-- bilingual-en:end -->

1. $f'(x_n)=0$：切线水平，没有有限的横轴交点。
2. 初值落在另一个根的吸引域，收敛到“错误的根”。
3. 曲率过大或初值太远，切线零点反而更远。
4. 迭代进入周期，例如两点来回跳动。
<!-- bilingual-en:start -->
1. If $f'(x_n)=0$, the tangent is horizontal and has no finite $x$-intercept.
2. The initial value may lie in the basin of attraction of another root and converge to the “wrong” root.
3. If the curvature is too large or the initial value too far away, the tangent's zero may be even farther from the desired root.
4. The iteration may enter a cycle, such as oscillating between two points.
<!-- bilingual-en:end -->

因此每步应检查 $|f(x_n)|$ 是否变小、$f'(x_n)$ 是否接近 $0$，不能只机械按键。
<!-- bilingual-en:start -->
Therefore, at each step check whether $|f(x_n)|$ is decreasing and whether $f'(x_n)$ is close to $0$; do not merely press the calculator keys mechanically.
<!-- bilingual-en:end -->

### 配套练习：$f(x)=x^3$ 永远不会有限步到根
<!-- bilingual-en:start -->
*Complementary Exercise: $f(x)=x^3$ will never be limited to the root*
<!-- bilingual-en:end -->

若 $x_n\ne0$，
<!-- bilingual-en:start -->
If $x_n\ne0$,
<!-- bilingual-en:end -->

$$
x_{n+1}
=x_n-\frac{x_n^3}{3x_n^2}
=\frac23x_n.
$$

归纳得
<!-- bilingual-en:start -->
induces
<!-- bilingual-en:end -->

$$
x_n=\left(\frac23\right)^nx_0.
$$

若 $x_0\ne0$，任何有限 $n$ 都有 $x_n\ne0$，但 $x_n\to0$。这里根是重根且 $f'(0)=0$，只呈线性收敛，而不是通常的二次收敛。
<!-- bilingual-en:start -->
If $x_0\ne0$, any limited $n$ has $x_n\ne0$, but $x_n\to0$.  Here the root is multiple and $f'(0)=0$, only linear convergence, not the usual quadratic convergence.
<!-- bilingual-en:end -->

### 易错点与边界
<!-- bilingual-en:start -->
*Fault-prone points and boundaries*
<!-- bilingual-en:end -->

- 分母必须是 $f'(x_n)$，不能写成 $f'(x_{n+1})$。
- “连续若干位不变”是停止准则之一，但更可靠的是同时检查残差 $|f(x_n)|$。
- 重根处 $f'(r)=0$，普通快速收敛理论不适用。
<!-- bilingual-en:start -->
- The denominator must be $f'(x_n)$ and cannot be written as $f'(x_{n+1})$.
- "Continuous bits unchanged" is one of the stop criteria, but it is more reliable to check the residual $|f(x_n)|$ at the same time.
- $f'(r)=0$ at multiple roots. The ordinary fast convergence theory is not applicable.
<!-- bilingual-en:end -->

> [!question]- 三道自检题与答案
> 1. 用 Newton 法求 $1/a$ 可选什么方程？
> 例如 $f(x)=1/x-a$ 或 $f(x)=1-ax$；后者本身线性，一步到达。
>
> 2. 若 $f(x_n)=0$ 会怎样？
> 已找到精确根，公式给 $x_{n+1}=x_n$（只要 $f'(x_n)\ne0$）。
>
> 3. 为何初值应尽量靠近目标根？
> 线性化只在局部可信，靠近后才进入误差平方的收敛区。
> <!-- bilingual-en:start -->
> 1. Using Newton method to solve $1/a$.
> For example, $f(x)=1/x-a$ or $f(x)=1-ax$; the latter itself is linear and arrives in one step.
> 2. What if $f(x_n)=0$?
> Found exact root, formula for $x_{n+1}=x_n$ (only $f'(x_n)\ne0$).
> 3. Why should initial values be as close to the target root as possible?
> The linearization is only locally credible and enters the convergence region of the square error when it is close.
> <!-- bilingual-en:end -->

### 本地材料

- [[Ses33a_Lecture_Notes.pdf#page=1|33a Newton’s Method]]
- [[Ses33b_Lecture_Notes.pdf#page=1|33b Accuracy]]
- [[Ses33c_Lecture_Notes.pdf#page=1|33c What Could Go Wrong?]]
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise033_Problems.pdf#page=1|Exercise 33 原题]] · [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise033_Solutions.pdf#page=1|Exercise 33 解答]]

**知识链小结：**Newton 法反复把非线性方程局部线性化；速度来自误差平方，风险也来自切线只代表局部。
<!-- bilingual-en:start -->
** Knowledge Chain Summary: **Newton method repeatedly linearizes the non-linear equation locally; speed from the square of error, risk also from the tangent only represents the local.
<!-- bilingual-en:end -->

## Problem Set 4

> [!info] 官方指定范围与材料
> 2C：1, 2, 4, 10, 13；2E：2, 3, 5, 7；2F：1。
> [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/PSet02_Problems.pdf#page=5|原题 2C 起始页]] · [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/PSet02_Solutions.pdf#page=10|官方解答 2C 起始页]]

### 2C：Max-min Problems

<!-- bilingual-en:start -->
*2C: Maximum and minimum problems*
<!-- bilingual-en:end -->
> [!example]- 2C-1：$12\times12$ 纸板折无盖盒
> 四角各剪边长 $x$，盒高 $x$、底边 $12-2x$，定义域 $0\le x\le6$。
> $$
> V(x)=x(12-2x)^2.
> $$
> $$
> V'=(12-2x)(12-6x).
> $$
> 候选 $x=2,6$，再查 $x=0$。端点体积均 $0$；$x=2$ 时
> $$
> V=2\cdot8^2=128.
> $$
> 故剪去 $\boxed{2\ \mathrm{in}}$ 的角，最大体积 $\boxed{128\ \mathrm{in^3}}$。
> <!-- bilingual-en:start -->
> Cut a square of side $x$ from each corner. The box has height $x$, base side $12-2x$, and feasible domain $0\le x\le6$.
> $$
> V(x)=x(12-2x)^2.
> $$
> $$
> V'=(12-2x)(12-6x).
> $$
> The candidates are $x=2$ and $x=6$, together with the endpoint $x=0$. Both endpoint volumes are zero, while at $x=2$,
> $$
> V=2\cdot8^2=128.
> $$
> Therefore, cut out $\boxed{2\ \mathrm{in}}$ squares; the maximum volume is $\boxed{128\ \mathrm{in^3}}$.
> <!-- bilingual-en:end -->

> [!example]- 2C-2：三面围栏的谷仓
> 设两条垂直谷仓的边为 $x$，平行边为 $y$。面积约束 $xy=20000$，围栏长度
> $$
> L=2x+y=2x+\frac{20000}{x}.
> $$
> $$
> L'=2-\frac{20000}{x^2}=0
> \Rightarrow x=100,\quad y=200.
> $$
> 两端极限均为 $\infty$，故最短围栏
> $$
> \boxed{L=400\ \mathrm{ft}}.
> $$
> <!-- bilingual-en:start -->
> Let the two sides perpendicular to the barn be $x$ and the parallel side be $y$. The area constraint is $xy=20000$, so the fence length is
> $$
> L=2x+y=2x+\frac{20000}{x}.
> $$
> $$
> L'=2-\frac{20000}{x^2}=0
> \Rightarrow x=100,\quad y=200.
> $$
> Since $L\to\infty$ at both ends of the feasible interval, the minimum fence length is
> $$
> \boxed{L=400\ \mathrm{ft}}.
> $$
> <!-- bilingual-en:end -->

> [!example]- 2C-4：邮政箱最大体积
> 令垂直于长度的正方形截面边长 $x$、长度 $y$。长度加周长限制：
> $$
> y+4x=108.
> $$
> $$
> V=x^2y=x^2(108-4x).
> $$
> $$
> V'=216x-12x^2=12x(18-x).
> $$
> 非退化候选 $x=18$ in，$y=36$ in。与退化端点及“长度不短于截面边”边界比较后为最大：
> $$
> \boxed{18\times18\times36\ \mathrm{in}},
> \qquad
> V=11664\ \mathrm{in^3}=\boxed{6.75\ \mathrm{ft^3}}.
> $$
> <!-- bilingual-en:start -->
> Let $x$ be the side length of the square cross-section perpendicular to the box's length, and let the length be $y$. The length-plus-girth restriction is
> $$
> y+4x=108.
> $$
> $$
> V=x^2y=x^2(108-4x).
> $$
> $$
> V'=216x-12x^2=12x(18-x).
> $$
> The nondegenerate candidate is $x=18$ in and $y=36$ in. Comparison with the degenerate endpoints and with the boundary imposed by “length not shorter than a cross-sectional side” confirms the maximum:
> $$
> \boxed{18\times18\times36\ \mathrm{in}},
> \qquad
> V=11664\ \mathrm{in^3}=\boxed{6.75\ \mathrm{ft^3}}.
> $$
> <!-- bilingual-en:end -->

> [!example]- 2C-10：最短时间与 Snell 定律
> 设沿岸跑动的水平位移为 $x$。跑、游路程分别为
> $$
> \sqrt{100^2+x^2},\qquad
> \sqrt{100^2+(a-x)^2}.
> $$
> 速度分别 $5,2$ m/s，总时间
> $$
> T(x)=\frac{\sqrt{100^2+x^2}}5
> +\frac{\sqrt{100^2+(a-x)^2}}2.
> $$
> 求导并用
> $$
> \sin\alpha=\frac{x}{\sqrt{100^2+x^2}},
> \quad
> \sin\beta=\frac{a-x}{\sqrt{100^2+(a-x)^2}},
> $$
> 得
> $$
> T'=\frac{\sin\alpha}{5}-\frac{\sin\beta}{2}.
> $$
> 最短时间的内点条件：
> $$
> \boxed{\frac{\sin\alpha}{\sin\beta}=\frac52}.
> $$
> 极端 $|x|\to\infty$ 时 $T\to\infty$，且目标为凸函数，所以该临界点给全局最小。
> <!-- bilingual-en:start -->
> Let $x$ be the horizontal displacement covered by running along the shore. The running and swimming distances are
> $$
> \sqrt{100^2+x^2},\qquad
> \sqrt{100^2+(a-x)^2}.
> $$
> The respective speeds are $5$ and $2$ m/s, so total time is
> $$
> T(x)=\frac{\sqrt{100^2+x^2}}5
> +\frac{\sqrt{100^2+(a-x)^2}}2.
> $$
> Differentiate and use
> $$
> \sin\alpha=\frac{x}{\sqrt{100^2+x^2}},
> \quad
> \sin\beta=\frac{a-x}{\sqrt{100^2+(a-x)^2}},
> $$
> to obtain
> $$
> T'=\frac{\sin\alpha}{5}-\frac{\sin\beta}{2}.
> $$
> Thus an interior minimum must satisfy
> $$
> \boxed{\frac{\sin\alpha}{\sin\beta}=\frac52}.
> $$
> Since $T\to\infty$ as $|x|\to\infty$ and the objective is convex, this critical point is the global minimum.
> <!-- bilingual-en:end -->

> [!example]- 2C-13：收益与利润
> **(a)** 票价 $p$ 美元时乘客数
> $$
> n(p)=100+\frac25(200-p)=180-\frac25p.
> $$
> 收益
> $$
> R=p\left(180-\frac25p\right),\qquad
> R'=180-\frac45p.
> $$
> 抛物线开口向下，故最优票价
> $$
> \boxed{p=225\ \text{美元}}.
> $$
> **(b)** 需求为 $x=10^5(10-p/2)$，单位成本为 $10-x/10^5$ 分。利润
> $$
> P=x\left[p-\left(10-\frac{x}{10^5}\right)\right].
> $$
> 代入需求后
> $$
> P(p)=\frac{10^5}{4}p(20-p).
> $$
> $P$ 开口向下，顶点在 $\boxed{p=10\text{ 分/kWh}}$；此时 $x=5\times10^5$，在产能范围内。
> <!-- bilingual-en:start -->
> **(a)** At a fare of $p$ dollars, the number of passengers is
> $$
> n(p)=100+\frac25(200-p)=180-\frac25p.
> $$
> Revenue is
> $$
> R=p\left(180-\frac25p\right),\qquad
> R'=180-\frac45p.
> $$
> Since the parabola opens downward, the optimal fare is
> $$
> \boxed{p=225\ \text{dollars}}.
> $$
> **(b)** Demand is $x=10^5(10-p/2)$ and unit cost is $10-x/10^5$ cents. Profit is
> $$
> P=x\left[p-\left(10-\frac{x}{10^5}\right)\right].
> $$
> Substituting the demand equation gives
> $$
> P(p)=\frac{10^5}{4}p(20-p).
> $$
> This parabola opens downward and has its vertex at $\boxed{p=10\text{ cents/kWh}}$. At that price, $x=5\times10^5$, which lies within capacity.
> <!-- bilingual-en:end -->

### 2E：Related Rates

<!-- bilingual-en:start -->
*2E: Related rates*
<!-- bilingual-en:end -->
> [!example]- 2E-2：旋转灯塔光斑
> 灯塔距海岸 $4$ mi。令光束与海岸夹角 $\theta$，光斑离垂足距离 $x$：
> $$
> x=4\cot\theta.
> $$
> 每分钟 $3$ 转，所以 $|\theta'|=6\pi$ rad/min。
> $$
> x'=-4\csc^2\theta\,\theta'.
> $$
> 当 $\theta=60^\circ$，$\csc^2\theta=4/3$，速率大小
> $$
> \boxed{|x'|=32\pi\ \mathrm{mi/min}}.
> $$
> 正负取决于旋转方向；题目问“多快”通常报大小。
> <!-- bilingual-en:start -->
> The lighthouse is $4$ mi from the shore. Let $\theta$ be the angle between the beam and the shore, and let $x$ be the distance from the foot of the perpendicular to the light spot:
> $$
> x=4\cot\theta.
> $$
> The beam makes $3$ revolutions per minute, so $|\theta'|=6\pi$ rad/min.
> $$
> x'=-4\csc^2\theta\,\theta'.
> $$
> When $\theta=60^\circ$, $\csc^2\theta=4/3$, so the speed is
> $$
> \boxed{|x'|=32\pi\ \mathrm{mi/min}}.
> $$
> The sign depends on the direction of rotation; when the question asks “how fast,” report the magnitude.
> <!-- bilingual-en:end -->

> [!example]- 2E-3：两船距离
> 第二船过交点后行 $10$ mi，用时 $1/3$ h；第一船早 $1/6$ h 经过，所以此刻在北方 $15$ mi。令东西、南北距离为 $x,y$，两者均以 $30$ mph 增长：
> $$
> z^2=x^2+y^2,\qquad
> zz'=xx'+yy'.
> $$
> 此刻 $z=\sqrt{10^2+15^2}=5\sqrt{13}$：
> $$
> \boxed{z'=\frac{10(30)+15(30)}{5\sqrt{13}}
> =\frac{150}{\sqrt{13}}\ \mathrm{mph}\approx41.6\ \mathrm{mph}}.
> $$
> <!-- bilingual-en:start -->
> The second ship has traveled $10$ mi since passing the intersection, taking $1/3$ h. The first ship passed the intersection $1/6$ h earlier, so it is now $15$ mi north. Let $x$ and $y$ be the east–west and north–south distances; both increase at $30$ mph:
> $$
> z^2=x^2+y^2,\qquad
> zz'=xx'+yy'.
> $$
> At this instant, $z=\sqrt{10^2+15^2}=5\sqrt{13}$, so
> $$
> \boxed{z'=\frac{10(30)+15(30)}{5\sqrt{13}}
> =\frac{150}{\sqrt{13}}\ \mathrm{mph}\approx41.6\ \mathrm{mph}}.
> $$
> <!-- bilingual-en:end -->

> [!example]- 2E-5：滑轮与绳长
> 人离滑轮正下方水平距离 $x$，手比滑轮低 $10$ ft，斜绳长
> $$
> s=\sqrt{x^2+100}.
> $$
> 另一端竖直绳长为 $z$，总长固定：$s+z=L$。重物以 $4$ ft/s 上升，故 $z'=-4$，从而 $s'=4$。
> $$
> ss'=xx'.
> $$
> 当 $x=20$，$s=10\sqrt5$：
> $$
> \boxed{x'=\frac{ss'}x=2\sqrt5\ \mathrm{ft/s}}.
> $$
> <!-- bilingual-en:start -->
> Let $x$ be the person's horizontal distance from the point directly below the pulley. The person's hand is $10$ ft below the pulley, so the slanted rope length is
> $$
> s=\sqrt{x^2+100}.
> $$
> Let $z$ be the vertical length at the other end. Since total length is fixed, $s+z=L$. The load rises at $4$ ft/s, so $z'=-4$ and therefore $s'=4$.
> $$
> ss'=xx'.
> $$
> When $x=20$, $s=10\sqrt5$, hence
> $$
> \boxed{x'=\frac{ss'}x=2\sqrt5\ \mathrm{ft/s}}.
> $$
> <!-- bilingual-en:end -->

> [!example]- 2E-7：梯形水槽
> 水深 $h$ 时，下底宽 $1/2$ m；两侧 $45^\circ$，故上宽 $1/2+2h$。截面积
> $$
> A=\frac{(1/2)+(1/2+2h)}2h
> =\frac h2+h^2.
> $$
> 槽长 $4$ m：
> $$
> V=4A=2h+4h^2.
> $$
> $$
> V'=(2+8h)h'.
> $$
> 在 $V'=1,h=1/2$ 时：
> $$
> \boxed{h'=\frac16\ \mathrm{m/s}}.
> $$
> <!-- bilingual-en:start -->
> At water depth $h$, the bottom width is $1/2$ m. Because both sides are at $45^\circ$, the top width is $1/2+2h$. The cross-sectional area is
> $$
> A=\frac{(1/2)+(1/2+2h)}2h
> =\frac h2+h^2.
> $$
> For a trough $4$ m long,
> $$
> V=4A=2h+4h^2.
> $$
> Therefore,
> $$
> V'=(2+8h)h'.
> $$
> When $V'=1$ and $h=1/2$,
> $$
> \boxed{h'=\frac16\ \mathrm{m/s}}.
> $$
> <!-- bilingual-en:end -->

### 2F：Locating Zeros; Newton’s Method

<!-- bilingual-en:start -->
*2F: Locating zeros; Newton's method*
<!-- bilingual-en:end -->
> [!example]- 2F-1：解 $\cos x=x$
> 令 $f(x)=\cos x-x$。任何根满足 $x=\cos x\in[-1,1]$。在 $[-1,1]$，
> $$
> f'(x)=-\sin x-1<0,
> $$
> 所以至多一根；又 $f(0)=1>0,f(1)=\cos1-1<0$，连续性给出唯一根位于 $(0,1)$。
> Newton 公式要保留正确符号：
> $$
> x_{n+1}
> =x_n-\frac{\cos x_n-x_n}{-\sin x_n-1}
> =x_n+\frac{\cos x_n-x_n}{\sin x_n+1}.
> $$
> 取 $x_1=1$：
> $$
> x_2=0.750363868,\quad
> x_3=0.739112891,\quad
> x_4=0.739085133.
> $$
> 三位小数为 $\boxed{0.739}$。固定点迭代 $z_{n+1}=\cos z_n$ 约需 $53$ 步才稳定到九位，而 Newton 法约三次更新。
> **勘误：**本地官方解答排版把等价公式中间的符号写成了减号；与所列数值迭代不相容。上式由标准 Newton 公式直接化简并已数值回验。
> <!-- bilingual-en:start -->
> Let $f(x)=\cos x-x$. Any root satisfies $x=\cos x\in[-1,1]$. On $[-1,1]$,
> $$
> f'(x)=-\sin x-1<0,
> $$
> so there is at most one root. Since $f(0)=1>0$, $f(1)=\cos1-1<0$, and $f$ is continuous, there is exactly one root in $(0,1)$.
> Keep the signs straight in Newton's formula:
> $$
> x_{n+1}
> =x_n-\frac{\cos x_n-x_n}{-\sin x_n-1}
> =x_n+\frac{\cos x_n-x_n}{\sin x_n+1}.
> $$
> Starting with $x_1=1$ gives
> $$
> x_2=0.750363868,\quad
> x_3=0.739112891,\quad
> x_4=0.739085133.
> $$
> To three decimal places, the root is $\boxed{0.739}$. Fixed-point iteration $z_{n+1}=\cos z_n$ takes about $53$ steps to stabilize to nine decimal places, whereas Newton's method needs only about three updates.
> **Correction:** The typesetting in the local official solution places a minus sign in the equivalent formula, which is inconsistent with its own numerical iterates. The formula above follows directly from the standard Newton update and has been checked numerically.
> <!-- bilingual-en:end -->

### Problem Set 4 错误检查
<!-- bilingual-en:start -->
*Problem Set 4 Error Checking*
<!-- bilingual-en:end -->

- 优化题必须把物理/几何可行域写进定义域。
- 相关变化率的正负来自坐标方向；若只问 speed，应报告绝对值并解释方向。
- Newton 法每步同时检查分母与残差；迭代数值不能替代唯一性证明。
<!-- bilingual-en:start -->
- The optimization problem must write the physical/geometric feasible domain into the domain.
- The sign of the relevant rate of change comes from the coordinate direction; if only speed is asked, the absolute value should be reported and the direction explained.
- The Newton method checks both the denominator and the residual at each step; iterative values cannot substitute for the proof of uniqueness.
<!-- bilingual-en:end -->

**本组小结：**PS4 的共同结构是“变量受关系约束”：优化对自由变量求极值，相关变化率对时间求导，Newton 法则用切线约束产生下一次猜测。
<!-- bilingual-en:start -->
**The common structure of **PS4 is "variable is constrained by the relation": the optimization is to find the extremum of the free variable, the relevant rate of change is to derive the derivative of time, and Newton's rule is to use tangent constraint to generate the next guess.
<!-- bilingual-en:end -->

---

## Part C：Mean Value Theorem, Antiderivatives and Differential Equations

## Session 34：Introduction to the Mean Value Theorem

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题：**平均变化率与某个瞬时变化率为何必然相等？证明单调性时究竟需要哪些假设？
<!-- bilingual-en:start -->
**Question:** Why must an average rate of change equal an instantaneous rate somewhere in the interval? What hypotheses are needed to prove monotonicity?
<!-- bilingual-en:end -->

**前置：**连续、可导、极值定理和切线斜率。
<!-- bilingual-en:start -->
**Prerequisites:** Continuity, differentiability, the Extreme Value Theorem, and tangent slopes.
<!-- bilingual-en:end -->

### 34a：定理陈述与几何意义
<!-- bilingual-en:start -->
*34a: Theorem Statement and Geometric Meaning*
<!-- bilingual-en:end -->

[[导数的应用|拉格朗日中值定理]]要求 $f$ 满足：
<!-- bilingual-en:start -->
[[导数的应用|Lagrange's Mean Value Theorem]] requires $f$ to:
<!-- bilingual-en:end -->

1. 在闭区间 $[a,b]$ 上连续；
2. 在开区间 $(a,b)$ 上可导；
3. $a<b$；
<!-- bilingual-en:start -->
1. Continuous in the closed interval $[a,b]$;
2. Differentiable on the open interval $(a,b)$;
3. $a<b$;
<!-- bilingual-en:end -->

则存在至少一个 $c\in(a,b)$，使
<!-- bilingual-en:start -->
then there is at least one $c\in(a,b)$ that causes
<!-- bilingual-en:end -->

$$
\boxed{
f'(c)=\frac{f(b)-f(a)}{b-a}
}.
$$

右边是端点割线斜率，左边是内部某点切线斜率。定理只保证 $c$ **存在**，不保证唯一，也不提供直接算法。
<!-- bilingual-en:start -->
The right-hand side is the slope of the secant through the endpoints, while the left-hand side is the tangent slope at an interior point. The theorem guarantees only that at least one such $c$ **exists**; it neither guarantees uniqueness nor provides a direct algorithm for finding it.
<!-- bilingual-en:end -->

![[98_attachment/MIT18.01SC/unit02-mean-value-theorem.png|740]]

### 从 Fermat 引理到 Rolle 定理
<!-- bilingual-en:start -->
*From Fermat Lemma to Rolle Theorem*
<!-- bilingual-en:end -->

**Fermat 引理：**若 $f$ 在内点 $c$ 可导且在 $c$ 取局部极大或极小，则 $f'(c)=0$。
<!-- bilingual-en:start -->
**Fermat's lemma:** If $f$ is differentiable at an interior point $c$ and has a local maximum or minimum there, then $f'(c)=0$.
<!-- bilingual-en:end -->

以局部极大为例。足够小的 $h>0$ 有
<!-- bilingual-en:start -->
Take local maxima.  $h>0$ small enough to have
<!-- bilingual-en:end -->

$$
\frac{f(c+h)-f(c)}h\le0;
$$

足够小的 $h<0$ 时分子仍 $\le0$、分母为负，故
<!-- bilingual-en:start -->
When $h<0$ is sufficiently small, the numerator is still $\le0$ while the denominator is negative, so
<!-- bilingual-en:end -->

$$
\frac{f(c+h)-f(c)}h\ge0.
$$

若两侧极限共同存在，只能同时等于 $0$。
<!-- bilingual-en:start -->
If both side limits exist, they can only be equal to $0$.
<!-- bilingual-en:end -->

**[[导数的应用|罗尔定理]]：**若 $f$ 在 $[a,b]$ 连续、在 $(a,b)$ 可导且 $f(a)=f(b)$，则存在 $c\in(a,b)$ 使 $f'(c)=0$。
<!-- bilingual-en:start -->
**[[导数的应用|Rolle's theorem]]:** If $f$ is continuous on $[a,b]$, differentiable on $(a,b)$, and $f(a)=f(b)$, then some $c\in(a,b)$ satisfies $f'(c)=0$.
<!-- bilingual-en:end -->

证明：连续函数由极值定理在闭区间达到最大、最小。若二者相同，$f$ 为常函数，任取内点即可。若不同，因为两端函数值相等，至少一个非平凡极值必须在内点取得；Fermat 引理给 $f'(c)=0$。
<!-- bilingual-en:start -->
It is proved that the continuous function reaches the maximum and minimum in the closed interval by the extremum theorem.  If the two are the same, $f$ is a constant function, choose any interior point.  If not, because the end function values are equal, at least one non-trivial extreme value must be obtained at an interior point; the Fermat lemma is given to $f'(c)=0$.
<!-- bilingual-en:end -->

### [[导数的应用|平均值定理证明]]
<!-- bilingual-en:start -->
*[[导数的应用|proof of mean value theorem]]*
<!-- bilingual-en:end -->

令割线斜率
<!-- bilingual-en:start -->
Make Secant Slope
<!-- bilingual-en:end -->

$$
m=\frac{f(b)-f(a)}{b-a},
$$

构造“减去割线倾斜部分”的辅助函数
<!-- bilingual-en:start -->
Construct an auxiliary function by subtracting the linear part of the secant:
<!-- bilingual-en:end -->

$$
g(x)=f(x)-m(x-a).
$$

$g$ 继承连续与可导性，且
<!-- bilingual-en:start -->
$g$ inherits continuity and differentiability, and
<!-- bilingual-en:end -->

$$
g(a)=f(a),
$$

$$
g(b)=f(b)-m(b-a)=f(b)-[f(b)-f(a)]=f(a).
$$

所以 $g(a)=g(b)$。由 Rolle 定理，某个 $c\in(a,b)$ 满足
<!-- bilingual-en:start -->
So, $g(a)=g(b)$.  By Rolle's theorem, a certain $c\in(a,b)$ satisfies
<!-- bilingual-en:end -->

$$
0=g'(c)=f'(c)-m.
$$

因此 $f'(c)=m$，证毕。
<!-- bilingual-en:start -->
Therefore $f'(c)=m$, completing the proof.
<!-- bilingual-en:end -->

### 34b：三个重要推论
<!-- bilingual-en:start -->
*34b: Three important consequences*
<!-- bilingual-en:end -->

在区间 $I$ 上，若对所有内点：
<!-- bilingual-en:start -->
On an interval $I$, if the following conditions hold at every interior point, then
<!-- bilingual-en:end -->

$$
f'>0\Rightarrow f\text{ 严格递增},
$$

$$
f'<0\Rightarrow f\text{ 严格递减},
$$

$$
f'=0\Rightarrow f\text{ 为常数}.
$$

证明以递增为例：任取 $x_1<x_2$，MVT 给
<!-- bilingual-en:start -->
It is proved that, as an example, $x_1<x_2$, MVT
<!-- bilingual-en:end -->

$$
f(x_2)-f(x_1)=f'(c)(x_2-x_1)>0.
$$

这里把一个点处的局部导数信息通过 MVT 连接到了任意两个点之间。
<!-- bilingual-en:start -->
The MVT turns local derivative information into a statement comparing any two points in the interval.
<!-- bilingual-en:end -->

### 配套练习：Taylor 余项的模式
<!-- bilingual-en:start -->
*Complementary Exercise: Modes for Taylor Remainder*
<!-- bilingual-en:end -->

$n$ 次 Taylor 多项式
<!-- bilingual-en:start -->
$n$ degree Taylor polynomials
<!-- bilingual-en:end -->

$$
P_n(b)=\sum_{k=0}^n\frac{f^{(k)}(a)}{k!}(b-a)^k.
$$

Taylor 定理将 MVT 的结构推广为：在足够光滑的条件下，存在 $c$ 位于 $a,b$ 之间，使
<!-- bilingual-en:start -->
Taylor's theorem generalizes the structure of MVT to the following conclusion: under smooth enough conditions, there exists a $c$ between $a,b$, so that
<!-- bilingual-en:end -->

$$
\boxed{
f(b)-P_n(b)=\frac{f^{(n+1)}(c)}{(n+1)!}(b-a)^{n+1}
}.
$$

练习取 $f(x)=x^3+2x+1,a=1,b=3,n=2$。有
<!-- bilingual-en:start -->
Practice getting $f(x)=x^3+2x+1,a=1,b=3,n=2$. Yes
<!-- bilingual-en:end -->

$$
P_2(x)=4+5(x-1)+3(x-1)^2,
$$

$$
f(3)=34,\qquad P_2(3)=26,
$$

误差为 $8$。因 $f^{(3)}=6$，
<!-- bilingual-en:start -->
The error is $8$.  Because $f^{(3)}=6$,
<!-- bilingual-en:end -->

$$
\frac{f^{(3)}(c)}{3!}(3-1)^3
=\frac66\cdot8=8,
$$

与实际误差完全一致。本课只验证模式，严格的一般证明留到 Taylor 单元。
<!-- bilingual-en:start -->
The results are in good agreement with the actual ones.  This lesson validates only the schema, leaving the strict generic proof in the Taylor module.
<!-- bilingual-en:end -->

### 易错点与边界
<!-- bilingual-en:start -->
*Fault-prone points and boundaries*
<!-- bilingual-en:end -->

- 闭区间要求连续，开区间要求可导；端点不要求双侧导数。
- $|x|$ 在跨越 $0$ 的区间不可直接用 MVT，因为内点 $0$ 不可导。
- 跳跃函数破坏连续性，平均速度可以没有对应的瞬时速度。
- MVT 的 $c$ 可能不止一个。
<!-- bilingual-en:start -->
- Continuity is required on the closed interval, while differentiability is required only on the open interval; two-sided derivatives are not required at the endpoints.
- $|x|$ cannot use MVT directly in an interval spanning $0$ because the interior point $0$ is not differentiable.
- The jump function destroys continuity and the average velocity may not have a corresponding instantaneous velocity.
- MVT may have more than one $c$.
<!-- bilingual-en:end -->

> [!question]- 三道自检题与答案
> 1. $f(x)=x^2$ 在 $[0,2]$ 的 MVT 点？
> 割线斜率 $2$，$f'(c)=2c=2$，故 $c=1$。
>
> 2. 若 $f(a)=f(b)$，MVT 化为什么？
> Rolle 定理：某个内点 $f'(c)=0$。
>
> 3. 定理能否由端点值直接求出 $c$？
> 一般不能；还需函数具体形式，且可能多解。
> <!-- bilingual-en:start -->
> 1. $f(x)=x^2$ at $[0,2]$'s MVT?
> The secant slope is $2$, so $f'(c)=2c=2$ and hence $c=1$.
> 2. If $f(a)=f(b)$, why MVT?
> Rolle theorem: some interior point $f'(c)=0$.
> 3. Can the theorem get $c$ directly from the endpoint values?
> In general, it is not; it also need function concrete form and may have many solutions.
> <!-- bilingual-en:end -->

### 本地材料

- [[Ses34a_Lecture_Notes.pdf#page=1|34a Mean Value Theorem]]
- [[Ses34b_Lecture_Notes.pdf#page=1|34b Consequences]]
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise034_Problems.pdf#page=1|Exercise 34 原题]] · [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise034_Solutions.pdf#page=1|Exercise 34 解答]]

**知识链小结：**MVT 用 Rolle 定理把“端点割线”转成“内部水平切线”；它是从局部导数推断全局行为的逻辑桥梁。
<!-- bilingual-en:start -->
**Knowledge chain summary:** The MVT uses Rolle's theorem to turn an endpoint secant into an interior horizontal tangent for an auxiliary function. It is the logical bridge from local derivative information to global behavior.
<!-- bilingual-en:end -->

## Session 35：Using the Mean Value Theorem

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题：**MVT 与线性近似有何不同？怎样把导数上下界变成函数差值和不等式？
<!-- bilingual-en:start -->
** Question: What is the difference between **MVT and linear approximation?  How to change the upper and lower bounds of the derivative into the difference and inequality of the function?
<!-- bilingual-en:end -->

**前置：**Session 34 的定理及单调性推论。
<!-- bilingual-en:start -->
**Preamble:**Theorem and monotonic inference of Session 34.
<!-- bilingual-en:end -->

### 35a：精确存在式与近似式
<!-- bilingual-en:start -->
*35a: Exact Existence and Approximation*
<!-- bilingual-en:end -->

线性近似在 $a$ 附近写成
<!-- bilingual-en:start -->
The linear approximation is written near $a$
<!-- bilingual-en:end -->

$$
f(b)\approx f(a)+f'(a)(b-a).
$$

MVT 则精确地说
<!-- bilingual-en:start -->
MVT is more precise
<!-- bilingual-en:end -->

$$
f(b)=f(a)+f'(c)(b-a)
$$

其中 $c$ 位于 $a,b$ 之间，却通常未知。若 $f'$ 在短区间上变化很小，$f'(c)\approx f'(a)$，前者便解释了后者为何可靠。
<!-- bilingual-en:start -->
where $c$ is between $a,b$ and is usually unknown.  If the $f'$ varies only slightly over a short interval, the $f'(c)\approx f'(a)$ explains why the latter is reliable.
<!-- bilingual-en:end -->

若 $m\le f'(x)\le M$ 在 $[a,b]$ 上成立，则
<!-- bilingual-en:start -->
If $m\le f'(x)\le M$ is true on $[a,b]$, then
<!-- bilingual-en:end -->

$$
\boxed{
m(b-a)\le f(b)-f(a)\le M(b-a)
}
\qquad(a<b).
$$

这叫导数界的“有限增量形式”。
<!-- bilingual-en:start -->
This is called the "finite increment form" of the derivative bound.
<!-- bilingual-en:end -->

### 35b：用单调性证明指数不等式
<!-- bilingual-en:start -->
*35b: Proving exponential inequalities with monotonicity*
<!-- bilingual-en:end -->

证明 $e^x>1+x$（$x>0$）。令
<!-- bilingual-en:start -->
Attestation $e^x>1+x$ ($x>0$).
<!-- bilingual-en:end -->

$$
F(x)=e^x-(1+x).
$$

$F(0)=0$，且
<!-- bilingual-en:start -->
$F(0)=0$, and
<!-- bilingual-en:end -->

$$
F'(x)=e^x-1>0\qquad(x>0).
$$

所以 $F$ 在 $(0,\infty)$ 递增，$F(x)>F(0)=0$。
<!-- bilingual-en:start -->
So $F$ in $(0,\infty)$ increments $F(x)>F(0)=0$.
<!-- bilingual-en:end -->

进一步证明
<!-- bilingual-en:start -->
further proof
<!-- bilingual-en:end -->

$$
e^x>1+x+\frac{x^2}{2}\qquad(x>0).
$$

令
<!-- bilingual-en:start -->
order
<!-- bilingual-en:end -->

$$
G(x)=e^x-\left(1+x+\frac{x^2}{2}\right).
$$

$G(0)=0$ 且
<!-- bilingual-en:start -->
$G(0)=0$ and
<!-- bilingual-en:end -->

$$
G'(x)=e^x-(1+x)>0
$$

正是上一结论，所以 $G(x)>0$。这展示了“前一个不等式成为下一个不等式的导数判据”。
<!-- bilingual-en:start -->
That's exactly the conclusion, $G(x)>0$.  This shows that "the former inequality becomes the derivative criterion of the next inequality".
<!-- bilingual-en:end -->

### 配套练习：正弦差的 Lipschitz 界
<!-- bilingual-en:start -->
*Complementary Exercise: Lipschitz Boundary for Sine Difference*
<!-- bilingual-en:end -->

对任意 $a\ne b$，MVT 给某个 $c$：
<!-- bilingual-en:start -->
For any $a\ne b$, MVT gives a $c$:
<!-- bilingual-en:end -->

$$
\sin b-\sin a=\cos c\,(b-a).
$$

取绝对值并用 $|\cos c|\le1$：
<!-- bilingual-en:start -->
Take the absolute value and use $|\cos c|\le1$:
<!-- bilingual-en:end -->

$$
\boxed{|\sin b-\sin a|\le|b-a|}.
$$

这说明正弦函数的输出变化绝不会超过输入变化；“导数绝对值不超过 $1$”控制了全局变化。
<!-- bilingual-en:start -->
This shows that the output of sine function will never exceed the input change, and the "absolute value of the derivative does not exceed $1$" controls the global change.
<!-- bilingual-en:end -->

### 易错点与边界
<!-- bilingual-en:start -->
*Fault-prone points and boundaries*
<!-- bilingual-en:end -->

- 当 $b<a$ 时直接乘不等式会翻转方向；用绝对值形式可避免。
- 要证明 $F(x)>0$，需同时给基准值和单调方向，仅算 $F'>0$ 不够。
- $f'$ 的最大/最小若不存在，可使用任意已知上下界，不必真的求极值。
<!-- bilingual-en:start -->
- When $b<a$, the direct multiplication inequality flips the direction; it can be avoided in absolute terms.
- To prove a $F(x)>0$, both the base value and the monotone direction need to be given; $F'>0$ alone is not sufficient.
- Maximum/Minimum of $f'$ If not present, use any known upper and lower bounds without actually having to perform an extremum.
<!-- bilingual-en:end -->

> [!question]- 三道自检题与答案
> 1. 若 $|f'|\le K$，可以推出什么？
> $|f(b)-f(a)|\le K|b-a|$。
>
> 2. 为什么这能给线性近似的粗误差控制？
> 对余项函数再用 MVT，可把误差变成导数差的界乘区间长度。
>
> 3. 证明 $\ln(1+x)<x$（$x>0$）。
> 令 $H=x-\ln(1+x)$；$H(0)=0$，$H'=x/(1+x)>0$。
> <!-- bilingual-en:start -->
> 1. If $|f'|\le K$, what can be launched?
> $|f(b)-f(a)|\le K|b-a|$.
> 2. Why does this give a coarse error control for linear approximations?
> By using MVT again for the remainder function, the error can be transformed into the interval length of the derivative difference.
> 3. Proof $\ln(1+x)<x$ ($x>0$).
> $H=x-\ln(1+x)$; $H(0)=0$, $H'=x/(1+x)>0$.
> <!-- bilingual-en:end -->

### 本地材料

- [[Ses35a_Lecture_Notes.pdf#page=1|35a MVT and Linear Approximation]]
- [[Ses35b_Lecture_Notes.pdf#page=1|35b MVT and Inequalities]]
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise035_Problems.pdf#page=1|Exercise 35 原题]] · [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise035_Solutions.pdf#page=1|Exercise 35 解答]]

**知识链小结：**MVT 把导数的界积分式地累积成函数差的界；线性近似可看作区间很短时把未知 $f'(c)$ 换成已知 $f'(a)$。
<!-- bilingual-en:start -->
** Knowledge chain summary: **MVT integrally cumulates the bound of the derivative into the bound of the difference of functions; linear approximation can be seen as the interval is very short, the unknown $f'(c)$ is replaced by the known $f'(a)$.
<!-- bilingual-en:end -->

## Session 36：Differentials

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题：**$dy=f'(x)\,dx$ 是什么对象？它和真实改变量 $\Delta y$ 有何差别？
<!-- bilingual-en:start -->
** Question: What is **$dy=f'(x)\,dx$?  How is it different from the real variable $\Delta y$?
<!-- bilingual-en:end -->

**前置：**线性近似、Leibniz 记号与链式法则。
<!-- bilingual-en:start -->
** Preamble: ** Linear approximation, Leibniz notation and chain rule.
<!-- bilingual-en:end -->

### 36a：定义
<!-- bilingual-en:start -->
*36a: Definition*
<!-- bilingual-en:end -->

对可导函数 $y=f(x)$，在给定基点 $x$ 处，把 $dx$ 视为可自由选取的小输入，定义输出微分；它是线性近似的增量形式
<!-- bilingual-en:start -->
For a differentiable function $y=f(x)$, regard $dx$ as a freely chosen small input at the base point $x$. The output differential is defined by the following incremental form of the linear approximation:
<!-- bilingual-en:end -->

$$
\boxed{dy=f'(x)\,dx}.
$$

$dy$ 是 $dx$ 的线性函数。真实改变量是
<!-- bilingual-en:start -->
$dy$ is a linear function of $dx$.  The true change is
<!-- bilingual-en:end -->

$$
\Delta y=f(x+\Delta x)-f(x).
$$

取 $dx=\Delta x$ 时，
<!-- bilingual-en:start -->
When you take $dx=\Delta x$,
<!-- bilingual-en:end -->

$$
\Delta y=dy+o(dx),
$$

故 $dy$ 是真实变化的一阶主部，而不是与 $\Delta y$ 永远相等。
<!-- bilingual-en:start -->
Therefore, $dy$ is the first-order principal part of real change, not always equal to $\Delta y$.
<!-- bilingual-en:end -->

### 36b：估计 $\sqrt[3]{64.1}$
<!-- bilingual-en:start -->
*36b: Estimated $\sqrt[3]{64.1}$*
<!-- bilingual-en:end -->

令 $y=x^{1/3}$，基点 $x=64$：
<!-- bilingual-en:start -->
Let $y=x^{1/3}$, base point $x=64$:
<!-- bilingual-en:end -->

$$
y_0=4,\qquad
dy=\frac13x^{-2/3}dx.
$$

在 $x=64$，
<!-- bilingual-en:start -->
In $x=64$,
<!-- bilingual-en:end -->

$$
dy=\frac1{48}dx.
$$

取 $dx=0.1$：
<!-- bilingual-en:start -->
Take $dx=0.1$:
<!-- bilingual-en:end -->

$$
\sqrt[3]{64.1}\approx y_0+dy
=4+\frac{0.1}{48}
\approx4.002083.
$$

单位也随导数传播：若 $x$ 的单位为体积，$dy/dx$ 是“长度/体积”，乘 $dx$ 后 $dy$ 恢复长度单位。
<!-- bilingual-en:start -->
Units also propagate with the derivative: If the unit of $x$ is volume, $dy/dx$ is "length/volume", and $dy$ restores the unit of length after multiplying by $dx$.
<!-- bilingual-en:end -->

### 微分形式的链式法则
<!-- bilingual-en:start -->
*chain rule of differential form*
<!-- bilingual-en:end -->

若 $y=f(u),u=g(x)$，
<!-- bilingual-en:start -->
If $y=f(u),u=g(x)$,
<!-- bilingual-en:end -->

$$
dy=f'(u)\,du,\qquad du=g'(x)\,dx.
$$

代入即
<!-- bilingual-en:start -->
immediately
<!-- bilingual-en:end -->

$$
dy=f'(g(x))g'(x)\,dx.
$$

Leibniz 记号看似约分，背后的严格依据仍是复合函数链式法则。
<!-- bilingual-en:start -->
The Leibniz notation appears to be reductive, and the strict basis behind it is still the chain rule of composite functions.
<!-- bilingual-en:end -->

### 配套练习：固定点的吸引性
<!-- bilingual-en:start -->
*Companion exercise: Attracting fixed points*
<!-- bilingual-en:end -->

若 $P(x_0)=x_0$ 且 $|P'(x_0)|<1$，线性化给
<!-- bilingual-en:start -->
If $P(x_0)=x_0$ and $|P'(x_0)|<1$, linearize to
<!-- bilingual-en:end -->

$$
P(x_0+dx)-x_0\approx P'(x_0)dx,
$$

所以一次迭代把小偏差缩小。若再假设 $P'$ 在 $x_0$ 连续，可取 $q<1$ 与一个邻域，使该邻域内 $|P'|\le q$。MVT 给
<!-- bilingual-en:start -->
Thus one iteration contracts a small deviation. If $P'$ is also continuous at $x_0$, we can choose $q<1$ and a neighborhood on which $|P'|\le q$. The MVT then gives
<!-- bilingual-en:end -->

$$
|P(x)-x_0|
=|P(x)-P(x_0)|
\le q|x-x_0|.
$$

迭代后误差至多 $q^n$ 倍，严格得到吸引性。
<!-- bilingual-en:start -->
After $n$ iterations the error is at most $q^n$ times its initial size, which rigorously establishes local attraction.
<!-- bilingual-en:end -->

对 $P(x)=ax(b-x)$，固定点解
<!-- bilingual-en:start -->
For $P(x)=ax(b-x)$, the fixed points solve
<!-- bilingual-en:end -->

$$
ax(b-x)=x
\Rightarrow
x_0=0,\quad x_1=\frac{ab-1}{a}.
$$

$$
P'(x)=ab-2ax.
$$

$P'(0)=ab>1$，所以 $0$ 不吸引；$P'(x_1)=2-ab$，第二固定点吸引当且仅当
<!-- bilingual-en:start -->
Since $P'(0)=ab>1$, $0$ is repelling. Also, $P'(x_1)=2-ab$, so the second fixed point is attracting if and only if
<!-- bilingual-en:end -->

$$
|2-ab|<1
\Longleftrightarrow
\boxed{1<ab<3}.
$$

### 易错点与边界
<!-- bilingual-en:start -->
*Fault-prone points and boundaries*
<!-- bilingual-en:end -->

- $dx$ 与 $\Delta x$ 可以取同一个数，但概念不同：前者进入线性映射，后者进入原函数真实差值。
- 不能把微分的“像分数运算”当作无需链式法则的纯代数。
- $|P'(x_0)|<1$ 的严格吸引结论需要邻域控制；单点线性式只是直觉。
<!-- bilingual-en:start -->
- $dx$ and $\Delta x$ may have the same numerical value, but they play different roles: $dx$ is the input to a linear approximation, whereas $\Delta x$ produces the function's actual change.
- The fraction-like appearance of differentials is not permission to manipulate them as pure algebra without the chain rule.
- $|P'(x_0)|<1$'s strict attraction conclusion requires neighborhood control; a single point of linearity is just intuition.
<!-- bilingual-en:end -->

> [!question]- 三道自检题与答案
> 1. $y=x^2$ 在 $x=3,dx=0.01$ 时 $dy$？
> $dy=2x\,dx=0.06$。
>
> 2. 对同题真实 $\Delta y$？
> $(3.01)^2-9=0.0601$，与 $dy$ 差 $0.0001=(dx)^2$。
>
> 3. 固定点导数为 $-1/2$ 有何直觉？
> 偏差每步约缩半并交替换侧。
> <!-- bilingual-en:start -->
> 1. $y=x^2$ at $x=3,dx=0.01$ $dy$?
> $dy=2x\,dx=0.06$.
> 2. True $\Delta y$?
> $(3.01)^2-9=0.0601$, $0.0001=(dx)^2$ less than $dy$.
> 3. What is the intuition of the fixed-point derivative $-1/2$?
> The deviation is reduced by about half in each step and the replacement side is interchanged.
> <!-- bilingual-en:end -->

### 本地材料

- [[Ses36a_Lecture_Notes.pdf#page=1|36a Differentials]]
- [[Ses36b_Lecture_Notes.pdf#page=1|36b Differentials and Linear Approximation]]
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise036_Problems.pdf#page=1|Exercise 36 原题]] · [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise036_Solutions.pdf#page=1|Exercise 36 解答]]

**知识链小结：**微分把导数从“一个比值”改写成“输入小变化到输出一阶变化的线性映射”，为换元与微分方程准备语言。
<!-- bilingual-en:start -->
**Knowledge chain summary:** Differentials recast the derivative from a ratio into a linear map from a small input change to the first-order output change, providing the language needed for substitution and differential equations.
<!-- bilingual-en:end -->

## Session 37：Antiderivatives

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题：**已知变化率 $f$，怎样恢复原函数？为什么答案必然带常数？
<!-- bilingual-en:start -->
**Problem:** Given a rate of change $f$, how do we recover an antiderivative? Why must the answer include an arbitrary constant?
<!-- bilingual-en:end -->

**前置：**求导公式、MVT 的常函数推论。
<!-- bilingual-en:start -->
**Preamble:** Derivative formula, constant function inference for MVT.
<!-- bilingual-en:end -->

### 37a：定义与不定积分
<!-- bilingual-en:start -->
*37a: Definition and indefinite integral*
<!-- bilingual-en:end -->

若
<!-- bilingual-en:start -->
if
<!-- bilingual-en:end -->

$$
F'(x)=f(x),
$$

则称 $F$ 是 $f$ 的[[定积分与微积分基本定理#两个基本定理怎样把导数与积分接起来|反导数]]（antiderivative）。记号
<!-- bilingual-en:start -->
$F$ is $f$'s [[定积分与微积分基本定理#两个基本定理怎样把导数与积分接起来|antiderivative]] (antiderivative).  Token
<!-- bilingual-en:end -->

$$
\int f(x)\,dx=F(x)+C
$$

叫[[定积分与微积分基本定理#积分应用的建模顺序|积分]]（这里是不定积分，indefinite integral）。积分号表示“寻找全部反导数”，$dx$ 指明积分变量。
<!-- bilingual-en:start -->
Call it [[定积分与微积分基本定理#积分应用的建模顺序|Integral]]. (Here's indefinite integral.)  The integral sign indicates "Find all antiderivatives" and $dx$ indicates the integral variable.
<!-- bilingual-en:end -->

例如
<!-- bilingual-en:start -->
For example
<!-- bilingual-en:end -->

$$
\int\sin x\,dx=-\cos x+C.
$$

### 37b–37c：基本公式
<!-- bilingual-en:start -->
*37b-37c: Base Formula*
<!-- bilingual-en:end -->

从求导公式反读：
<!-- bilingual-en:start -->
Reverse Reading from Derivative Formula:
<!-- bilingual-en:end -->

$$
\boxed{\int x^n\,dx=\frac{x^{n+1}}{n+1}+C\quad(n\ne-1)}
$$

特殊指数 $n=-1$：
<!-- bilingual-en:start -->
Special exponent $n=-1$:
<!-- bilingual-en:end -->

$$
\boxed{\int\frac{dx}{x}=\ln|x|+C}
$$

绝对值确保在 $x<0$ 的区间也有
<!-- bilingual-en:start -->
Absolute values ensure that they also exist in the $x<0$ range
<!-- bilingual-en:end -->

$$
\frac{d}{dx}\ln|x|=\frac1x.
$$

其他常用式：
<!-- bilingual-en:start -->
Other common usage:
<!-- bilingual-en:end -->

$$
\int\sec^2x\,dx=\tan x+C,
$$

$$
\int\frac{dx}{\sqrt{1-x^2}}=\arcsin x+C,
$$

$$
\int\frac{dx}{1+x^2}=\arctan x+C.
$$

每个公式都受原函数定义域限制；例如 $\ln|x|$ 的两个定义域分支不能用一个连续常数跨过 $0$。
<!-- bilingual-en:start -->
Each antiderivative formula is restricted to its domain; for example, the two branches of the domain of $\ln|x|$ cannot be joined across $0$ by a single constant.
<!-- bilingual-en:end -->

### 37d：唯一性证明
<!-- bilingual-en:start -->
*37d: proof of uniqueness*
<!-- bilingual-en:end -->

若在同一连通区间上
<!-- bilingual-en:start -->
If they are on the same connectivity interval
<!-- bilingual-en:end -->

$$
F'=f,\qquad G'=f,
$$

则
<!-- bilingual-en:start -->
then
<!-- bilingual-en:end -->

$$
(F-G)'=0.
$$

由 MVT 的推论，$F-G$ 在该区间为常数：
<!-- bilingual-en:start -->
According to MVT's inference, $F-G$ is a constant in this interval:
<!-- bilingual-en:end -->

$$
\boxed{F(x)=G(x)+C}.
$$

这证明了“$+C$”已经穷尽所有可能，而不是随手附加。
<!-- bilingual-en:start -->
This proves that "$+C$" has been exhausted, not appended.
<!-- bilingual-en:end -->

### 配套练习：从求导法则反读积分法则
<!-- bilingual-en:start -->
*Supporting Exercise: Reverse Reading Integral Rule from Derivation Rule*
<!-- bilingual-en:end -->

和法则给线性性：
<!-- bilingual-en:start -->
The sum rule gives linearity:
<!-- bilingual-en:end -->

$$
\int(f+g)\,dx=\int f\,dx+\int g\,dx.
$$

乘积法则
<!-- bilingual-en:start -->
product rule
<!-- bilingual-en:end -->

$$
(FG)'=F'G+FG'
$$

反读得
<!-- bilingual-en:start -->
read inversely
<!-- bilingual-en:end -->

$$
\int(F'G+FG')\,dx=FG+C.
$$

移项得到后续的分部积分雏形：
<!-- bilingual-en:start -->
Shift term gets the embryonic form of the following partial integral:
<!-- bilingual-en:end -->

$$
\int F\,G'\,dx=FG-\int F'G\,dx.
$$

### 易错点与边界
<!-- bilingual-en:start -->
*Fault-prone points and boundaries*
<!-- bilingual-en:end -->

- $\int x^{-1}dx$ 不能套幂公式，因为分母 $n+1=0$。
- 不定积分答案漏写 $C$。
- 两个看似不同的答案可能只差常数；用恒等式或相减求导检查。
- 反导数通常应在一个连通区间内讨论。
<!-- bilingual-en:start -->
- $\int x^{-1}dx$ cannot overlay the power formula because the denominator $n+1=0$.
- Indefinite integral answer omits $C$.
- Two seemingly different answers may differ only by a constant; check with identity or subtraction.
- The antiderivative should normally be discussed in a connected interval.
<!-- bilingual-en:end -->

> [!question]- 三道自检题与答案
> 1. $\int3x^2dx$？
> $x^3+C$。
>
> 2. 为何 $\frac12\sin^2x$ 与 $-\frac12\cos^2x$ 都可作为 $\sin x\cos x$ 的反导数？
> 两者导数相同，且相差常数 $1/2$。
>
> 3. 怎样最快检查不定积分？
> 对答案求导，确认回到原被积函数，并检查定义域。
> <!-- bilingual-en:start -->
> 1. $\int3x^2dx$?
> $x^3+C$.
> 2. Why are $\frac12\sin^2x$ and $-\frac12\cos^2x$ antiderivatives of $\sin x\cos x$?
> The derivatives of the two are the same, and the difference constant is $1/2$.
> 3. What's the quickest way to check indefinite points?
> Derive the answer, confirm a return to the original product function, and examine the domain.
> <!-- bilingual-en:end -->

### 本地材料

- [[Ses37a_Lecture_Notes.pdf#page=1|37a Introduction to Antiderivatives]]
- [[Ses37b_Lecture_Notes.pdf#page=1|37b Antiderivative of x^a]]
- [[Ses37c_Lecture_Notes.pdf#page=1|37c Basic Antiderivatives]]
- [[Ses37d_Lecture_Notes.pdf#page=1|37d Unique up to a Constant]]
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise037_Problems.pdf#page=1|Exercise 37 原题]] · [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise037_Solutions.pdf#page=1|Exercise 37 解答]]

**知识链小结：**反导数逆转求导；MVT 保证这种逆转只丢失一个常数，这正是后续由变化率重建函数的基础。
<!-- bilingual-en:start -->
**Knowledge chain summary:** Antidifferentiation reverses differentiation. The MVT shows that this reversal loses only an additive constant, which is the basis for reconstructing a function from its rate of change.
<!-- bilingual-en:end -->

## Session 38：Integration by Substitution

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题：**复合函数求导的链式法则怎样反向使用？选换元变量时应看什么？
<!-- bilingual-en:start -->
**Question:** How can the chain rule for differentiating a composite function be used in reverse, and what should we look for when choosing a substitution?
<!-- bilingual-en:end -->

**前置：**微分、基本反导数与链式法则。
<!-- bilingual-en:start -->
**Preamble: **Differential, Basic Antiderivative, and Chain Rule.
<!-- bilingual-en:end -->

### 38a：[[积分方法#积分方法选择树|换元积分]]的来源
<!-- bilingual-en:start -->
*38a: Where [[积分方法#积分方法选择树|integration by substitution]] comes from*
<!-- bilingual-en:end -->

链式法则：
<!-- bilingual-en:start -->
Chain rule:
<!-- bilingual-en:end -->

$$
\frac{d}{dx}F(g(x))=F'(g(x))g'(x).
$$

若 $F'=f$，反读为
<!-- bilingual-en:start -->
If $F'=f$, read back as
<!-- bilingual-en:end -->

$$
\boxed{\int f(g(x))g'(x)\,dx=F(g(x))+C}.
$$

写 $u=g(x),du=g'(x)dx$，就得到
<!-- bilingual-en:start -->
Write $u=g(x),du=g'(x)dx$ and you'll get
<!-- bilingual-en:end -->

$$
\int f(u)\,du.
$$

这不是凭符号约掉 $dx$，而是链式法则的结构匹配。
<!-- bilingual-en:start -->
This is not a formal cancellation of $dx$; it is a structural use of the chain rule.
<!-- bilingual-en:end -->

**讲义例题：**
<!-- bilingual-en:start -->
** Handout Example: **
<!-- bilingual-en:end -->

$$
\int x^3(x^4+2)^5\,dx.
$$

令 $u=x^4+2$，$du=4x^3dx$：
<!-- bilingual-en:start -->
Let $u=x^4+2$, $du=4x^3dx$:
<!-- bilingual-en:end -->

$$
\begin{aligned}
\int x^3(x^4+2)^5dx
&=\frac14\int u^5du\\
&=\frac{u^6}{24}+C\\
&=\boxed{\frac{(x^4+2)^6}{24}+C}.
\end{aligned}
$$

### 38b–38c：“高级猜测”与常数修正
<!-- bilingual-en:start -->
*38b-38c: "Advanced Guess" and Constant Correction*
<!-- bilingual-en:end -->

对
<!-- bilingual-en:start -->
Yes
<!-- bilingual-en:end -->

$$
\int\frac{x}{\sqrt{1+x^2}}dx,
$$

看到内层 $1+x^2$ 的导数含 $x$，猜结果形如 $\sqrt{1+x^2}$。验证：
<!-- bilingual-en:start -->
The derivative of the inner $1+x^2$ is found to contain $x$, and the guessing result is $\sqrt{1+x^2}$.  Validation:
<!-- bilingual-en:end -->

$$
\frac{d}{dx}\sqrt{1+x^2}
=\frac{x}{\sqrt{1+x^2}}.
$$

所以答案直接得到。对 $\int e^{6x}dx$，猜 $e^{6x}$ 后求导多出因子 $6$，于是
<!-- bilingual-en:start -->
So the answer is straight.  For $\int e^{6x}dx$, the derivative of $e^{6x}$ is calculated to get the additional factor $6$, so
<!-- bilingual-en:end -->

$$
\int e^{6x}dx=\frac16e^{6x}+C.
$$

“猜测”必须由求导验证；它是熟练后的模式识别，不是省略证明。
<!-- bilingual-en:start -->
A "guess" must be checked by differentiation. It is practiced pattern recognition, not a substitute for verification.
<!-- bilingual-en:end -->

### 配套练习：同一积分的两种换元
<!-- bilingual-en:start -->
*Companion exercise: Two substitutions for the same integral*
<!-- bilingual-en:end -->

$$
I=\int\tan x\,\sec^2x\,dx.
$$

令 $u=\tan x,du=\sec^2x\,dx$：
<!-- bilingual-en:start -->
Let $u=\tan x,du=\sec^2x\,dx$:
<!-- bilingual-en:end -->

$$
I=\frac12\tan^2x+C.
$$

令 $v=\sec x,dv=\sec x\tan x\,dx$，把原式写成 $\sec x(\sec x\tan x\,dx)$：
<!-- bilingual-en:start -->
Let $v=\sec x,dv=\sec x\tan x\,dx$ write the original $\sec x(\sec x\tan x\,dx)$:
<!-- bilingual-en:end -->

$$
I=\frac12\sec^2x+\widetilde C.
$$

由 $\sec^2x-\tan^2x=1$，两答案相差 $1/2$，因此等价。
<!-- bilingual-en:start -->
By $\sec^2x-\tan^2x=1$, the two answers differ by $1/2$, so they are equivalent.
<!-- bilingual-en:end -->

### 易错点与边界
<!-- bilingual-en:start -->
*Fault-prone points and boundaries*
<!-- bilingual-en:end -->

- 只替换内层表达式却没替换对应微分因子。
- 换元完成后答案仍留变量 $u$。
- 常数因子漏乘倒数；最稳妥的校验是对最终答案求导。
- $\ln$ 型答案应根据定义域写绝对值。
<!-- bilingual-en:start -->
- Replace only the inner expression without replacing the corresponding differential factor.
- Failing to substitute back for $u$ after completing the integration.
- Forgetting the reciprocal of a constant factor; the safest check is to differentiate the final answer.
- Logarithmic answers should use absolute values where required by the domain.
<!-- bilingual-en:end -->

> [!question]- 三道自检题与答案
> 1. $\int2x\cos(x^2)dx$？
> $\sin(x^2)+C$。
>
> 2. $\int x e^{x^2}dx$？
> $\frac12e^{x^2}+C$。
>
> 3. 为什么令 $u=x$ 对任何积分都“合法”却没帮助？
> 它没有简化结构；好换元要把复合内层及其导数同时吸收。
> <!-- bilingual-en:start -->
> 1. $\int2x\cos(x^2)dx$?
> $\sin(x^2)+C$.
> 2. $\int x e^{x^2}dx$?
> $\frac12e^{x^2}+C$.
> 3. Why is it not helpful to make $u=x$ "legal" for any points?
> It does not have a simplified structure; the good substitution element must absorb the composite inner layer and its derivative simultaneously.
> <!-- bilingual-en:end -->

### 本地材料

- [[Ses38a_Lecture_Notes.pdf#page=1|38a Substitution Example]]
- [[Ses38b_Lecture_Notes.pdf#page=1|38b Advanced Guessing]]
- [[Ses38c_Lecture_Notes.pdf#page=1|38c More Examples]]
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise038_Problems.pdf#page=1|Exercise 38 原题]] · [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise038_Solutions.pdf#page=1|Exercise 38 解答]]

**知识链小结：**换元积分就是反向链式法则；选元的目标是让“内层函数 + 它的微分”一起变成一个基本积分。
<!-- bilingual-en:start -->
**Knowledge chain summary:** Integration by substitution is the chain rule in reverse. The goal is to absorb both an inner function and its differential into a standard integral.
<!-- bilingual-en:end -->

## Session 39：Introduction to Differential Equations

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题：**方程若包含未知函数及其导数，什么叫“解”？怎样从变化规律恢复一族函数？
<!-- bilingual-en:start -->
**Question:** When an equation contains an unknown function and its derivatives, what counts as a solution, and how can a family of functions be recovered from a law of change?
<!-- bilingual-en:end -->

**前置：**反导数、微分与换元。
<!-- bilingual-en:start -->
**Prerequisites:** Antiderivatives, differentials, and substitution.
<!-- bilingual-en:end -->

### 39a：微分方程与通解
<!-- bilingual-en:start -->
*39a: Differential Equations and General Solutions*
<!-- bilingual-en:end -->

[[导数的应用#微分方程与增长率|微分方程]]（differential equation）把未知函数与导数联系起来。最简单的
<!-- bilingual-en:start -->
A [[导数的应用#微分方程与增长率|differential equation]] relates an unknown function to one or more of its derivatives. In the simplest case,
<!-- bilingual-en:end -->

$$
\frac{dy}{dx}=f(x)
$$

的通解是
<!-- bilingual-en:start -->
the general solution is
<!-- bilingual-en:end -->

$$
y=\int f(x)\,dx.
$$

更有内容的讲义例题：
<!-- bilingual-en:start -->
A more substantial example from the notes is
<!-- bilingual-en:end -->

$$
\frac{dy}{dx}+xy=0
\quad\Longleftrightarrow\quad
\frac{dy}{dx}=-xy.
$$

先假设 $y\ne0$，分离变量：
<!-- bilingual-en:start -->
First assume $y\ne0$ and separate the variables:
<!-- bilingual-en:end -->

$$
\frac{dy}{y}=-x\,dx.
$$

积分：
<!-- bilingual-en:start -->
Integrating gives
<!-- bilingual-en:end -->

$$
\ln|y|=-\frac{x^2}{2}+C.
$$

指数化并把符号吸收到任意常数 $A$：
<!-- bilingual-en:start -->
Exponentiation and absorption of the sign to an arbitrary constant $A$:
<!-- bilingual-en:end -->

$$
\boxed{y=Ae^{-x^2/2}}.
$$

代回检查：
<!-- bilingual-en:start -->
Substitution check:
<!-- bilingual-en:end -->

$$
y'=-xAe^{-x^2/2}=-xy.
$$

$A=0$ 也成立，补回了分离过程中除以 $y$ 可能丢失的零解。若给初值 $y(0)=3$，则 $A=3$，唯一选出 $y=3e^{-x^2/2}$。
<!-- bilingual-en:start -->
$A=0$ is also valid, restoring the zero solution that may have been lost when we divided by $y$. If the initial condition is $y(0)=3$, then $A=3$, selecting the unique solution $y=3e^{-x^2/2}$.
<!-- bilingual-en:end -->

### 39b：一般分离变量框架
<!-- bilingual-en:start -->
*39b: General separation of variables framework*
<!-- bilingual-en:end -->

若
<!-- bilingual-en:start -->
if
<!-- bilingual-en:end -->

$$
\frac{dy}{dx}=f(x)g(y),
$$

先另列 $g(y)=0$ 的常数平衡解。对 $g(y)\ne0$：
<!-- bilingual-en:start -->
The constant equilibrium solutions of $g(y)=0$ are listed first.  For $g(y)\ne0$:
<!-- bilingual-en:end -->

$$
\frac{dy}{g(y)}=f(x)\,dx.
$$

若 $H'(y)=1/g(y)$、$F'(x)=f(x)$，则
<!-- bilingual-en:start -->
If $H'(y)=1/g(y)$, $F'(x)=f(x)$
<!-- bilingual-en:end -->

$$
\boxed{H(y)=F(x)+C}.
$$

这可能已经是合格的隐式通解；能方便求逆时再写成 $y=H^{-1}(F(x)+C)$。
<!-- bilingual-en:start -->
This may already be a valid implicit general solution. Only when inversion is convenient do we write $y=H^{-1}(F(x)+C)$.
<!-- bilingual-en:end -->

![[98_attachment/MIT18.01SC/unit02-separation-of-variables.png|780]]

### 配套练习：代回检查候选解
<!-- bilingual-en:start -->
*Complementary Exercise: Check Back Candidate Solution*
<!-- bilingual-en:end -->

**(a)** $y=e^x/3$：

$$
y''=\frac13e^x,
\quad
4y''-y=\frac43e^x-\frac13e^x=e^x.
$$

故满足 $4y''-y=e^x$。
<!-- bilingual-en:start -->
Therefore, $4y''-y=e^x$ is satisfied.
<!-- bilingual-en:end -->

**(b)** $y=1/x$（$x\ne0$）：

$$
y'=-x^{-2},\qquad y''=2x^{-3}.
$$

$$
x^2y''+3xy'+y
=2x^{-1}-3x^{-1}+x^{-1}=0.
$$

代回不仅查代数，也暴露定义域 $x\ne0$。
<!-- bilingual-en:start -->
Substitution checks the algebra and also exposes the domain restriction $x\ne0$.
<!-- bilingual-en:end -->

### 易错点与边界
<!-- bilingual-en:start -->
*Fault-prone points and boundaries*
<!-- bilingual-en:end -->

- 通解是一族函数；初值才选择常数。
- 两边积分只需一个任意常数，因为 $C_2-C_1$ 仍是任意常数。
- 除以 $g(y)$ 会丢掉 $g(y)=0$ 的平衡解。
- 隐式解不必强行解出 $y$；强行开根号还可能漏分支。
<!-- bilingual-en:start -->
- General solutions are a family of functions; constants are selected for initial values.
- The two-sided integration requires only one arbitrary constant, because $C_2-C_1$ is still an arbitrary constant.
- Dividing by $g(y)$ will lose the equilibrium solution for $g(y)=0$.
- An implicit solution need not be solved explicitly for $y$; forcing a square root can also lose a branch.
<!-- bilingual-en:end -->

> [!question]- 三道自检题与答案
> 1. 如何确认候选函数是解？
> 求出方程所需各阶导数，代回原方程，在其定义区间逐点成立。
>
> 2. 为什么 $A$ 可以为负而 $e^C$ 只能为正？
> 从 $|y|=e^Ce^{-x^2/2}$ 拆正负支，并把符号并入 $A$。
>
> 3. 初值 $y(0)=0$ 选出哪条解？
> $A=0$，即 $y\equiv0$。
> <!-- bilingual-en:start -->
> 1. How to confirm the candidate function is the solution?
> The required derivatives of the equation are obtained and replaced by the original equation. The equation is established point by point in its defined interval.
> 2. Why can $A$ be negative and $e^C$ only positive?
> Remove the positive and negative branches from $|y|=e^Ce^{-x^2/2}$ and incorporate the symbols into $A$.
> 3. Which solution is selected by the initial $y(0)=0$?
> $A=0$, or $y\equiv0$.
> <!-- bilingual-en:end -->

### 本地材料

- [[Ses39a_Lecture_Notes.pdf#page=1|39a Introduction to ODEs]]
- [[Ses39b_Lecture_Notes.pdf#page=1|39b Separation of Variables]]
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise039_Problems.pdf#page=1|Exercise 39 原题]] · [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise039_Solutions.pdf#page=1|Exercise 39 解答]]

**知识链小结：**微分方程给变化规律，积分恢复函数族；完整答案由通解、平衡解、初值与定义区间共同组成。
<!-- bilingual-en:start -->
**Knowledge chain summary:** A differential equation specifies a law of change, and integration recovers a family of functions. A complete answer accounts for the general solution, equilibrium solutions, initial conditions, and the interval of definition.
<!-- bilingual-en:end -->

## Session 40：Separation of Variables

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题：**怎样从几何斜率条件建立并求解微分方程？分离变量时有哪些隐藏的奇点？
<!-- bilingual-en:start -->
**Question:** How can a differential equation be built and solved from a geometric condition on slopes, and what singularities can be hidden by separation of variables?
<!-- bilingual-en:end -->

**前置：**Session 39 的一般框架、$\int dx/x$ 与隐式曲线。
<!-- bilingual-en:start -->
**Preamble: **Session 39's general frame, $\int dx/x$, and implicit curves.
<!-- bilingual-en:end -->

### 40a：最简单的[[导数的应用#微分方程与增长率|分离变量法]]
<!-- bilingual-en:start -->
*40a: Easiest [[导数的应用#微分方程与增长率|separation of variables method]]*
<!-- bilingual-en:end -->

$$
\frac{dy}{dx}=f(x)
\quad\Rightarrow\quad
dy=f(x)dx
\quad\Rightarrow\quad
y=\int f(x)dx.
$$

这说明普通求反导数是分离变量法的特殊情形。
<!-- bilingual-en:start -->
This shows that ordinary antidifferentiation is a special case of separation of variables.
<!-- bilingual-en:end -->

### 40b：切线斜率是径向斜率的两倍
<!-- bilingual-en:start -->
*40b: Tangent slope is twice radial slope*
<!-- bilingual-en:end -->

点 $(x,y)$ 到原点的射线斜率为 $y/x$。条件给
<!-- bilingual-en:start -->
Point $(x,y)$ to the origin of the slope of the ray is $y/x$.  Condition to
<!-- bilingual-en:end -->

$$
\frac{dy}{dx}=\frac{2y}{x},
\qquad x\ne0.
$$

对 $y\ne0$：
<!-- bilingual-en:start -->
For $y\ne0$:
<!-- bilingual-en:end -->

$$
\frac{dy}{y}=2\frac{dx}{x}.
$$

$$
\ln|y|=2\ln|x|+C.
$$

指数化：
<!-- bilingual-en:start -->
Indexation:
<!-- bilingual-en:end -->

$$
|y|=e^Cx^2.
$$

合并正、负与零解：
<!-- bilingual-en:start -->
Combine positive, negative, and zero solutions:
<!-- bilingual-en:end -->

$$
\boxed{y=Ax^2}.
$$

代回 $y'=2Ax=2y/x$ 对 $x\ne0$ 成立。注意原方程在 $x=0$ 未规定斜率，因此穿过 $x=0$ 时可能把左右不同参数的抛物线拼接；这是奇点导致的非唯一性。
<!-- bilingual-en:start -->
Replacing $y'=2Ax=2y/x$ is true for $x\ne0$.  Note that the original equation does not prescribe a slope in $x=0$, so it is possible to stitch parabolas of different parameters of left and right when passing through $x=0$; this is non-uniqueness caused by singularities.
<!-- bilingual-en:end -->

### 40c：与抛物线正交的轨线
<!-- bilingual-en:start -->
*40c: Trace orthogonal to parabola*
<!-- bilingual-en:end -->

抛物线族 $y=ax^2$ 在点 $(x,y)$ 的斜率为 $2y/x$。正交曲线斜率取负倒数：
<!-- bilingual-en:start -->
The slope of the parabola family $y=ax^2$ at point $(x,y)$ is $2y/x$.  Negative reciprocal of slope of orthogonal curve:
<!-- bilingual-en:end -->

$$
\frac{dy}{dx}=-\frac{x}{2y}.
$$

分离并积分：
<!-- bilingual-en:start -->
Separate and Integrate:
<!-- bilingual-en:end -->

$$
2y\,dy=-x\,dx
\Rightarrow
y^2=-\frac{x^2}{2}+C.
$$

所以
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

$$
\boxed{y^2+\frac{x^2}{2}=C}.
$$

$C>0$ 时是一族椭圆。写成显式分支
<!-- bilingual-en:start -->
$C>0$ is a family of ellipses.  Write as Explicit Branch
<!-- bilingual-en:end -->

$$
y=\pm\sqrt{C-\frac{x^2}{2}}
$$

可看出顶、底半支分别是函数；在 $y=0$ 处有竖直切线，原斜率式分母也为零。
<!-- bilingual-en:start -->
It can be seen that the top and bottom half-branches are functions respectively, and there is a vertical tangent line at $y=0$, and the original slope denominator is also zero.
<!-- bilingual-en:end -->

### 配套练习：指数与受限增长
<!-- bilingual-en:start -->
*Complementary Exercise: Index and Limited Growth*
<!-- bilingual-en:end -->

**指数增长**
<!-- bilingual-en:start -->
** exponential growth **
<!-- bilingual-en:end -->

$$
\frac{dy}{dx}=ry
\Rightarrow
\frac{dy}{y}=r\,dx
\Rightarrow
\boxed{y=Ae^{rx}}.
$$

**Logistic 型受限增长**
<!-- bilingual-en:start -->
**Logistic-type restricted growth **
<!-- bilingual-en:end -->

$$
\frac{dy}{dx}=ry(s-y),\qquad s>0.
$$

先记录平衡解 $y=0,s$。对其他解：
<!-- bilingual-en:start -->
Record the equilibrium solution $y=0,s$ first.  For other solutions:
<!-- bilingual-en:end -->

$$
\frac{dy}{y(s-y)}=r\,dx.
$$

利用
<!-- bilingual-en:start -->
utilization
<!-- bilingual-en:end -->

$$
\frac1{y(s-y)}=\frac1s\left(\frac1y+\frac1{s-y}\right),
$$

积分：
<!-- bilingual-en:start -->
Credits:
<!-- bilingual-en:end -->

$$
\frac1s\ln\left|\frac{y}{s-y}\right|=rx+C.
$$

令指数常数为 $A$：
<!-- bilingual-en:start -->
Make the exponential constant $A$:
<!-- bilingual-en:end -->

$$
\frac{y}{s-y}=Ae^{srx}.
$$

解出
<!-- bilingual-en:start -->
dissolve
<!-- bilingual-en:end -->

$$
\boxed{y=\frac{sAe^{srx}}{1+Ae^{srx}}}.
$$

对典型初值 $0<y_0<s$ 有 $A>0$，且 $x\to\infty$ 时 $y\to s$，与 $0<y<s$ 时 $y'>0$ 的符号分析相符。
<!-- bilingual-en:start -->
For the typical initial value of $0<y_0<s$ is $A>0$, and $y\to s$ when $x\to\infty$, which is consistent with the sign analysis of $y'>0$ when $0<y<s$.
<!-- bilingual-en:end -->

### 易错点与边界
<!-- bilingual-en:start -->
*Fault-prone points and boundaries*
<!-- bilingual-en:end -->

- 负倒数斜率只适用于有限非零斜率；水平/竖直需单独解释。
- 分离前列平衡解，分离后检查分母为零的位置。
- 显式开根会产生多个分支；一个隐式曲线未必是全局函数。
- 求出公式后要确定由初值连通得到的最大区间，不能跨越分母零点。
<!-- bilingual-en:start -->
- The negative-reciprocal slope rule applies only to finite, nonzero slopes; horizontal and vertical tangents need separate treatment.
- List equilibrium solutions before separating variables, and after separation check where any denominator vanishes.
- Solving explicitly by taking roots can create multiple branches; an implicit curve need not be a global function.
- After deriving a formula, identify the maximal interval connected to the initial condition; it cannot cross a zero of the denominator.
<!-- bilingual-en:end -->

> [!question]- 三道自检题与答案
> 1. $y'=ky$ 的零解是否包含在 $Ae^{kx}$ 中？
> 包含，取 $A=0$。
>
> 2. Logistic 方程在 $y>s$ 时变化方向？
> 若 $r>0$，$y(s-y)<0$，所以 $y$ 下降趋向 $s$。
>
> 3. 正交曲线方程为何可保留隐式？
> 椭圆整体不是单值函数，隐式式同时表示上下分支且更自然。
> <!-- bilingual-en:start -->
> 1. Is the zero solution of $y'=ky$ included in $Ae^{kx}$?
> Contains, take $A=0$.
> 2. The direction of the Logistic equation at $y>s$?
> If $r>0$, $y(s-y)<0$, so $y$ down towards $s$.
> 3. Why can orthogonal curve equations remain implicit?
> The ellipse as a whole is not a single-valued function, implicitly representing both the upper and lower branches and more naturally.
> <!-- bilingual-en:end -->

### 本地材料

- [[Ses40a_Lecture_Notes.pdf#page=1|40a dy/dx=f(x)]]
- [[Ses40b_Lecture_Notes.pdf#page=1|40b Differential Equations and Slope I]]
- [[Ses40c_Lecture_Notes.pdf#page=1|40c Differential Equations and Slope II]]
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise040_Problems.pdf#page=1|Exercise 40 原题]] · [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise040_Solutions.pdf#page=1|Exercise 40 解答]]

**知识链小结：**分离变量把斜率关系转为两个反导数的等式；平衡解、奇点和分支决定公式真正代表哪些曲线。
<!-- bilingual-en:start -->
**Knowledge chain summary:**Separation of variables transforms the slope relation into an equation of two antiderivatives; equilibrium solutions, singularities, and branches determine which curves the formula really represents.
<!-- bilingual-en:end -->

## Problem Set 5

> [!info] 官方指定范围与材料
> 2G：1b, 2b, 5, 6；3A：1d, 1e, 2a, 2c, 2e, 2g, 2i, 2k, 3a, 3c, 3e, 3g；3F：1c, 1d, 2a, 2e, 4b, 4c, 4d, 8b。
> 2G 使用 [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/PSet02_Problems.pdf#page=12|Applications of Differentiation 原题]] 与 [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/PSet02_Solutions.pdf#page=28|官方解答]]；3A、3F 使用 [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/PSet03_Problems.pdf#page=1|Integration 原题]] 与内容较完整的修订文件 [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/PSet03_Solutions_2.pdf#page=1|Integration 修订解答]]。

### 2G：Mean Value Theorem

<!-- bilingual-en:start -->
*2G: Mean Value Theorem*
<!-- bilingual-en:end -->
> [!example]- 2G-1b：求 MVT 保证的点
> 对 $f(x)=\ln x$、区间 $[1,2]$，割线斜率
> $$
> \frac{f(2)-f(1)}{2-1}=\ln2.
> $$
> 因 $f'(c)=1/c$，MVT 条件为
> $$
> \frac1c=\ln2.
> $$
> 所以
> $$
> \boxed{c=\frac1{\ln2}}.
> $$
> 又 $1/2<\ln2<1$，故 $1<c<2$，确在开区间内。
> <!-- bilingual-en:start -->
> For $f(x)=\ln x$ on $[1,2]$, the secant slope is
> $$
> \frac{f(2)-f(1)}{2-1}=\ln2.
> $$
> Since $f'(c)=1/c$, the MVT condition is
> $$
> \frac1c=\ln2.
> $$
> Therefore,
> $$
> \boxed{c=\frac1{\ln2}}.
> $$
> Because $1/2<\ln2<1$, we have $1<c<2$, so the point does lie in the open interval.
> <!-- bilingual-en:end -->

> [!example]- 2G-2b：证明 $\sqrt{1+x}<1+x/2$（$x>0$）
> 对 $f(t)=\sqrt{1+t}$ 在 $[0,x]$ 用 MVT，存在 $c\in(0,x)$：
> $$
> \sqrt{1+x}-1=f'(c)x=\frac{x}{2\sqrt{1+c}}.
> $$
> 因 $c>0$，$\sqrt{1+c}>1$，所以
> $$
> \frac{x}{2\sqrt{1+c}}<\frac x2.
> $$
> 两边加 $1$：
> $$
> \boxed{\sqrt{1+x}<1+\frac x2}.
> $$
> 严格不等号来自 $x>0$；$x=0$ 时是等号。
> <!-- bilingual-en:start -->
> Apply the MVT to $f(t)=\sqrt{1+t}$ on $[0,x]$. There is some $c\in(0,x)$ such that
> $$
> \sqrt{1+x}-1=f'(c)x=\frac{x}{2\sqrt{1+c}}.
> $$
> Since $c>0$, $\sqrt{1+c}>1$, and therefore
> $$
> \frac{x}{2\sqrt{1+c}}<\frac x2.
> $$
> Adding $1$ to both sides gives
> $$
> \boxed{\sqrt{1+x}<1+\frac x2}.
> $$
> The inequality is strict because $x>0$; equality holds at $x=0$.
> <!-- bilingual-en:end -->

> [!example]- 2G-5：三个零点推出某处二阶导为零
> 已知 $f''$ 在含 $a<b<c$ 的区间存在，且 $f(a)=f(b)=f(c)=0$。
> 在 $[a,b]$ 对 $f$ 用 Rolle 定理，存在 $q_1\in(a,b)$ 使 $f'(q_1)=0$；在 $[b,c]$ 再用一次，存在 $q_2\in(b,c)$ 使 $f'(q_2)=0$。
> 对 $f'$ 在 $[q_1,q_2]$ 用 Rolle 定理，存在 $p\in(q_1,q_2)\subset(a,c)$：
> $$
> \boxed{f''(p)=0}.
> $$
> 每次应用都要说明相应函数连续、内点可导；$f''$ 存在保证 $f'$ 可导，从而连续。
> <!-- bilingual-en:start -->
> Suppose $f''$ exists on an interval containing $a<b<c$ and $f(a)=f(b)=f(c)=0$.
> Applying Rolle's theorem to $f$ on $[a,b]$ gives $q_1\in(a,b)$ with $f'(q_1)=0$. Applying it again on $[b,c]$ gives $q_2\in(b,c)$ with $f'(q_2)=0$.
> Now apply Rolle's theorem to $f'$ on $[q_1,q_2]$. There is some $p\in(q_1,q_2)\subset(a,c)$ such that
> $$
> \boxed{f''(p)=0}.
> $$
> Each application requires continuity on the relevant closed interval and differentiability in its interior. The existence of $f''$ ensures that $f'$ is differentiable and hence continuous.
> <!-- bilingual-en:end -->

> [!example]- 2G-6：用 MVT 证明单调与常函数
> 任取 $a\le x_1<x_2\le b$。MVT 给某个 $c\in(x_1,x_2)$：
> $$
> f(x_2)-f(x_1)=f'(c)(x_2-x_1).
> $$
> **(a)** 若区间内 $f'>0$，右边为正，故 $f(x_2)>f(x_1)$，所以 $f$ 严格递增。
> **(b)** 若区间内 $f'=0$，右边为零，任意两点函数值相同，所以 $f$ 为常函数。
> 结论依赖 $f$ 在闭子区间连续、开子区间可导。
> <!-- bilingual-en:start -->
> Choose any $a\le x_1<x_2\le b$. The MVT gives some $c\in(x_1,x_2)$ such that
> $$
> f(x_2)-f(x_1)=f'(c)(x_2-x_1).
> $$
> **(a)** If $f'>0$ throughout the interval, the right-hand side is positive, so $f(x_2)>f(x_1)$ and $f$ is strictly increasing.
> **(b)** If $f'=0$ throughout the interval, the right-hand side is zero. Every two points therefore have the same function value, so $f$ is constant.
> These conclusions require $f$ to be continuous on each closed subinterval and differentiable in its interior.
> <!-- bilingual-en:end -->

### 3A：Differentials and Indefinite Integration

<!-- bilingual-en:start -->
*3A: Differentials and indefinite integration*
<!-- bilingual-en:end -->
> [!example]- 3A-1d、3A-1e：计算微分
> **(d)** $f(x)=e^{3x}\sin x$。由乘积与链式法则：
> $$
> \boxed{df=e^{3x}(3\sin x+\cos x)\,dx}.
> $$
> **(e)** 若 $\sqrt x+\sqrt y=1$，在 $x>0,y>0$ 的光滑部分：
> $$
> \frac{dx}{2\sqrt x}+\frac{dy}{2\sqrt y}=0.
> $$
> 所以
> $$
> \boxed{dy=-\sqrt{\frac yx}\,dx}.
> $$
> 又 $\sqrt y=1-\sqrt x$，也可写
> $$
> \boxed{dy=\left(1-\frac1{\sqrt x}\right)dx}.
> $$
> <!-- bilingual-en:start -->
> **(d)** For $f(x)=e^{3x}\sin x$, the product and chain rules give
> $$
> \boxed{df=e^{3x}(3\sin x+\cos x)\,dx}.
> $$
> **(e)** If $\sqrt x+\sqrt y=1$, then on the smooth portion where $x>0$ and $y>0$,
> $$
> \frac{dx}{2\sqrt x}+\frac{dy}{2\sqrt y}=0.
> $$
> Hence,
> $$
> \boxed{dy=-\sqrt{\frac yx}\,dx}.
> $$
> Since $\sqrt y=1-\sqrt x$, this can also be written as
> $$
> \boxed{dy=\left(1-\frac1{\sqrt x}\right)dx}.
> $$
> <!-- bilingual-en:end -->

> [!example]- 3A-2a：逐项反导
> $$
> \begin{aligned}
> \int(2x^4+3x^2+x+8)\,dx
> &=\frac25x^5+x^3+\frac12x^2+8x+C.
> \end{aligned}
> $$
> 每项求导都恢复对应被积项。
> <!-- bilingual-en:start -->
> $$
> \begin{aligned}
> \int(2x^4+3x^2+x+8)\,dx
> &=\frac25x^5+x^3+\frac12x^2+8x+C.
> \end{aligned}
> $$
> Differentiating each term recovers the corresponding term of the integrand.
> <!-- bilingual-en:end -->

> [!example]- 3A-2c：$\int\sqrt{8+9x}\,dx$
> 令 $u=8+9x$，$du=9dx$：
> $$
> \int\sqrt{8+9x}\,dx
> =\frac19\int u^{1/2}du
> =\boxed{\frac2{27}(8+9x)^{3/2}+C}.
> $$
> <!-- bilingual-en:start -->
> Let $u=8+9x$, so $du=9dx$:
> $$
> \int\sqrt{8+9x}\,dx
> =\frac19\int u^{1/2}du
> =\boxed{\frac2{27}(8+9x)^{3/2}+C}.
> $$
> <!-- bilingual-en:end -->

> [!example]- 3A-2e：$\int\dfrac{x}{\sqrt{8-2x^2}}\,dx$（勘误）
> 按**原题清晰排版**，根号在分母。令 $u=8-2x^2$，$du=-4x\,dx$：
> $$
> \int\frac{x}{\sqrt{8-2x^2}}dx
> =-\frac14\int u^{-1/2}du
> =\boxed{-\frac12\sqrt{8-2x^2}+C}.
> $$
> 求导回验：
> $$
> \frac d{dx}\left[-\frac12(8-2x^2)^{1/2}\right]
> =\frac{x}{\sqrt{8-2x^2}}.
> $$
> **本地修订解答仍有排版错误：**它在中途把 $u^{-1/2}$ 误写为 $u^{1/2}$，给出 $-\frac16(8-2x^2)^{3/2}$；后者求导为 $x\sqrt{8-2x^2}$，对应另一道题，不能通过本题回验。
> <!-- bilingual-en:start -->
> In the clearly typeset original problem, the radical is in the denominator. Let $u=8-2x^2$, so $du=-4x\,dx$:
> $$
> \int\frac{x}{\sqrt{8-2x^2}}dx
> =-\frac14\int u^{-1/2}du
> =\boxed{-\frac12\sqrt{8-2x^2}+C}.
> $$
> Differentiate to check:
> $$
> \frac d{dx}\left[-\frac12(8-2x^2)^{1/2}\right]
> =\frac{x}{\sqrt{8-2x^2}}.
> $$
> **The locally revised solution still contains a typesetting error:** it changes $u^{-1/2}$ to $u^{1/2}$ midway and gives $-\frac16(8-2x^2)^{3/2}$. Differentiating that expression yields $x\sqrt{8-2x^2}$, which belongs to a different integrand and fails this check.
> <!-- bilingual-en:end -->

> [!example]- 3A-2g、3A-2i、3A-2k：结构换元与代数整理
> **(g)**
> $$
> \int7x^4e^{x^5}dx,\quad u=x^5,\ du=5x^4dx
> $$
> $$
> \boxed{\int7x^4e^{x^5}dx=\frac75e^{x^5}+C}.
> $$
> **(i)**
> $$
> \boxed{\int\frac{dx}{3x+2}=\frac13\ln|3x+2|+C}.
> $$
> **(k)** 先写
> $$
> \frac{x}{x+5}=1-\frac5{x+5},
> $$
> 所以
> $$
> \boxed{\int\frac{x}{x+5}dx=x-5\ln|x+5|+C}.
> $$
> 绝对值保证公式在分母不为零的任一连通区间成立。
> <!-- bilingual-en:start -->
> **(g)**
> $$
> \int7x^4e^{x^5}dx,\quad u=x^5,\ du=5x^4dx
> $$
> $$
> \boxed{\int7x^4e^{x^5}dx=\frac75e^{x^5}+C}.
> $$
> **(i)**
> $$
> \boxed{\int\frac{dx}{3x+2}=\frac13\ln|3x+2|+C}.
> $$
> **(k)** First write
> $$
> \frac{x}{x+5}=1-\frac5{x+5},
> $$
> so
> $$
> \boxed{\int\frac{x}{x+5}dx=x-5\ln|x+5|+C}.
> $$
> The absolute value makes the formula valid on any connected interval where the denominator is nonzero.
> <!-- bilingual-en:end -->

> [!example]- 3A-3a、3A-3c、3A-3e、3A-3g：三角反导
> **(a)**
> $$
> \boxed{\int\sin(5x)dx=-\frac15\cos(5x)+C}.
> $$
> **(c)** 令 $u=\cos x,du=-\sin xdx$：
> $$
> \boxed{\int\cos^2x\sin xdx=-\frac13\cos^3x+C}.
> $$
> **(e)** 令 $u=x/5,dx=5du$：
> $$
> \boxed{\int\sec^2(x/5)dx=5\tan(x/5)+C}.
> $$
> **(g)** 把 $\sec^9x\tan x$ 写成 $\sec^8x(\sec x\tan x)$，令 $u=\sec x$：
> $$
> \boxed{\int\sec^9x\tan xdx=\frac19\sec^9x+C}.
> $$
> <!-- bilingual-en:start -->
> **(a)**
> $$
> \boxed{\int\sin(5x)dx=-\frac15\cos(5x)+C}.
> $$
> **(c)** Let $u=\cos x$, so $du=-\sin xdx$:
> $$
> \boxed{\int\cos^2x\sin xdx=-\frac13\cos^3x+C}.
> $$
> **(e)** Let $u=x/5$, so $dx=5du$:
> $$
> \boxed{\int\sec^2(x/5)dx=5\tan(x/5)+C}.
> $$
> **(g)** Write $\sec^9x\tan x$ as $\sec^8x(\sec x\tan x)$ and let $u=\sec x$:
> $$
> \boxed{\int\sec^9x\tan xdx=\frac19\sec^9x+C}.
> $$
> <!-- bilingual-en:end -->

### 3F：Differential Equations — Separation of Variables

<!-- bilingual-en:start -->
*3F: Differential equations—separation of variables*
<!-- bilingual-en:end -->
> [!example]- 3F-1c：$y'=3/\sqrt y$
> 方程要求 $y>0$。分离：
> $$
> y^{1/2}dy=3dx.
> $$
> $$
> \frac23y^{3/2}=3x+C.
> $$
> 整理常数：
> $$
> \boxed{y=\left(\frac92x+A\right)^{2/3}},
> $$
> 取满足括号为正的区间，并代回原式。括号为零处原方程的 $1/\sqrt y$ 无定义，不能跨过。
> <!-- bilingual-en:start -->
> The equation requires $y>0$. Separate the variables:
> $$
> y^{1/2}dy=3dx.
> $$
> $$
> \frac23y^{3/2}=3x+C.
> $$
> After absorbing constants,
> $$
> \boxed{y=\left(\frac92x+A\right)^{2/3}}.
> $$
> Restrict the solution to an interval on which the expression in parentheses is positive, and substitute back into the original equation. At a zero of the parentheses, $1/\sqrt y$ is undefined, so the solution cannot cross that point.
> <!-- bilingual-en:end -->

> [!example]- 3F-1d：$y'=xy^2$
> 先记录平衡解 $\boxed{y\equiv0}$。对 $y\ne0$：
> $$
> y^{-2}dy=x\,dx.
> $$
> $$
> -\frac1y=\frac{x^2}{2}+C.
> $$
> 所以非零解
> $$
> \boxed{y=-\frac1{x^2/2+C}}.
> $$
> 最大定义区间不能穿过分母为零的点。
> <!-- bilingual-en:start -->
> First record the equilibrium solution $\boxed{y\equiv0}$. For $y\ne0$,
> $$
> y^{-2}dy=x\,dx.
> $$
> $$
> -\frac1y=\frac{x^2}{2}+C.
> $$
> Therefore, the nonzero solutions are
> $$
> \boxed{y=-\frac1{x^2/2+C}}.
> $$
> A maximal interval of definition cannot cross a point where the denominator is zero.
> <!-- bilingual-en:end -->

> [!example]- 3F-2a：$y'=4xy,\ y(1)=3$，求 $y(3)$
> $y=0$ 不符合初值。分离并积分：
> $$
> \frac{dy}{y}=4x\,dx
> \Rightarrow\ln y=2x^2+C.
> $$
> 代 $y(1)=3$：
> $$
> C=\ln3-2.
> $$
> 因此
> $$
> y(x)=3e^{2x^2-2},
> \qquad
> \boxed{y(3)=3e^{16}}.
> $$
> <!-- bilingual-en:start -->
> The equilibrium solution $y=0$ does not satisfy the initial condition. Separate and integrate:
> $$
> \frac{dy}{y}=4x\,dx
> \Rightarrow\ln y=2x^2+C.
> $$
> Substituting $y(1)=3$ gives
> $$
> C=\ln3-2.
> $$
> Hence,
> $$
> y(x)=3e^{2x^2-2},
> \qquad
> \boxed{y(3)=3e^{16}}.
> $$
> <!-- bilingual-en:end -->

> [!example]- 3F-2e：$y'=e^y,\ y(3)=0$
> $$
> e^{-y}dy=dx.
> $$
> $$
> -e^{-y}=x+C.
> $$
> 初值给 $-1=3+C$，故 $C=-4$：
> $$
> e^{-y}=4-x,
> \qquad
> \boxed{y=-\ln(4-x)}.
> $$
> 因 $4-x>0$，含初值 $x=3$ 的最大定义区间是
> $$
> \boxed{(-\infty,4)}.
> $$
> 并且 $\boxed{y(0)=-\ln4}$。
> <!-- bilingual-en:start -->
> $$
> e^{-y}dy=dx.
> $$
> $$
> -e^{-y}=x+C.
> $$
> The initial condition gives $-1=3+C$, so $C=-4$:
> $$
> e^{-y}=4-x,
> \qquad
> \boxed{y=-\ln(4-x)}.
> $$
> Because $4-x>0$, the maximal interval containing the initial point $x=3$ is
> $$
> \boxed{(-\infty,4)}.
> $$
> Also, $\boxed{y(0)=-\ln4}$.
> <!-- bilingual-en:end -->

> [!example]- 3F-4b、3F-4c、3F-4d：Newton 冷却定律
> 方程
> $$
> \frac{dT}{dt}=k(T_e-T),\qquad k>0
> $$
> 令温差 $U=T-T_e$，则 $U'=-kU$：
> $$
> U=(T_0-T_e)e^{-kt}.
> $$
> 所以
> $$
> \boxed{T(t)=T_e+(T_0-T_e)e^{-kt}}.
> $$
> 因 $k>0$，$e^{-kt}\to0$，故
> $$
> \boxed{T(t)\to T_e}.
> $$
> 对锭块数据 $T_0=680^\circ,T_e=40^\circ,T(8)=200^\circ$：
> $$
> 160=640e^{-8k}\Rightarrow e^{-8k}=\frac14.
> $$
> 降到 $50^\circ$ 时：
> $$
> 10=640e^{-kt}\Rightarrow e^{-kt}=\frac1{64}=\left(\frac14\right)^3.
> $$
> 因此 $t/8=3$，
> $$
> \boxed{t=24\ \mathrm{h}}.
> $$
> <!-- bilingual-en:start -->
> The equation is
> $$
> \frac{dT}{dt}=k(T_e-T),\qquad k>0.
> $$
> Set the temperature difference $U=T-T_e$. Then $U'=-kU$, so
> $$
> U=(T_0-T_e)e^{-kt}.
> $$
> Therefore,
> $$
> \boxed{T(t)=T_e+(T_0-T_e)e^{-kt}}.
> $$
> Since $k>0$, $e^{-kt}\to0$, and hence
> $$
> \boxed{T(t)\to T_e}.
> $$
> For the ingot data $T_0=680^\circ$, $T_e=40^\circ$, and $T(8)=200^\circ$,
> $$
> 160=640e^{-8k}\Rightarrow e^{-8k}=\frac14.
> $$
> When the ingot has cooled to $50^\circ$,
> $$
> 10=640e^{-kt}\Rightarrow e^{-kt}=\frac1{64}=\left(\frac14\right)^3.
> $$
> Thus $t/8=3$, so
> $$
> \boxed{t=24\ \mathrm{h}}.
> $$
> <!-- bilingual-en:end -->

> [!example]- 3F-8b：切点平分第一象限内的切线段
> 设切线与两坐标轴截点为 $(2x,0),(0,2y)$，其中心正好是 $P=(x,y)$。切线斜率
> $$
> \frac{0-2y}{2x-0}=-\frac yx.
> $$
> 因曲线切线斜率就是 $y'$：
> $$
> \frac{dy}{dx}=-\frac yx.
> $$
> 分离：
> $$
> \frac{dy}{y}=-\frac{dx}{x}
> \Rightarrow
> \ln y=-\ln x+C.
> $$
> 第一象限 $x,y>0$，故
> $$
> \boxed{y=\frac{C}{x},\quad C>0}.
> $$
> 代回可见每条矩形双曲线都满足平分性质。
> <!-- bilingual-en:start -->
> Let the tangent meet the coordinate axes at $(2x,0)$ and $(0,2y)$, whose midpoint is exactly $P=(x,y)$. The slope of the tangent is
> $$
> \frac{0-2y}{2x-0}=-\frac yx.
> $$
> Since the tangent slope to the curve is $y'$,
> $$
> \frac{dy}{dx}=-\frac yx.
> $$
> Separate the variables:
> $$
> \frac{dy}{y}=-\frac{dx}{x}
> \Rightarrow
> \ln y=-\ln x+C.
> $$
> In the first quadrant, $x,y>0$, so
> $$
> \boxed{y=\frac{C}{x},\quad C>0}.
> $$
> Substitution confirms that every rectangular hyperbola in this family has the required bisection property.
> <!-- bilingual-en:end -->

### Problem Set 5 错误检查
<!-- bilingual-en:start -->
*Problem Set 5 Bugcheck*
<!-- bilingual-en:end -->

- MVT/Rolle 每次使用都要重新核对连续与可导区间。
- 反导答案用求导回验；这一步识别出了 3A-2e 的官方排版错误。
- 分离变量前列 $g(y)=0$ 的平衡解，解出后再定最大区间。
- 初值问题的常数、符号和定义域是答案的一部分，不是可省略的附注。
<!-- bilingual-en:start -->
- Whenever MVT or Rolle's theorem is used, recheck continuity on the closed interval and differentiability on its interior.
- The anti-missile answer is a derivative test; this step identifies an official typographical error in 3A-2e.
- Separate the equilibrium solution of the $g(y)=0$ in front of the variable, and then set the maximum interval.
- The constants, symbols, and domains of the initial value question are part of the answer, not an omitted note.
<!-- bilingual-en:end -->

**本组小结：**PS5 把 MVT 的严谨推理、微分记号、反向链式法则和微分方程连成一条完整链：导数控制函数，反导数重建函数。
<!-- bilingual-en:start -->
**This group of summaries:**PS5 connects the rigorous reasoning of MVT, the differential notation, the inverse chain rule and the differential equation into a complete chain: the derivative control function, the inverse derivative reconstruction function.
<!-- bilingual-en:end -->

---

## Exam 2

## Session 41：Review for Exam 2

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题：**如何在考试前把本章看似分散的方法压缩成少数稳定流程？
<!-- bilingual-en:start -->
** Question: ** How do I compress this chapter's seemingly decentralized approach into a few stable processes prior to the exam?
<!-- bilingual-en:end -->

**前置：**Session 23–40 全部内容，尤其是定义域与边界检查。
<!-- bilingual-en:start -->
** Preamble: **Session 23-40 All, especially domain and boundary checking.
<!-- bilingual-en:end -->

### 41a：六类题的最短检查表
<!-- bilingual-en:start -->
*41a: Minimum checklist for six categories of questions*
<!-- bilingual-en:end -->

1. **线性/二次近似**
   - 先写基点 $a$；
   - 计算 $f(a),f'(a),f''(a)$；
   - 二次项必须有 $1/2$；
   - 组合展开按总次数截断。
2. **曲线作图**
   - 定义域、间断、端点/无穷远；
   - $f'$ 符号与临界点；
   - $f''$ 符号与真正拐点；
   - 截距和对称性复核。
3. **优化**
   - 画图与选变量；
   - 写约束、目标和可行域；
   - 降为一个自由变量；
   - 比较临界点、端点、间断。
4. **相关变化率**
   - 写所有时刻成立的关系；
   - 对 $t$ 求导；
   - 最后代瞬时值；
   - 报方向、单位和速率大小。
5. **MVT**
   - 先写连续/可导假设；
   - 再写 $f(b)-f(a)=f'(c)(b-a)$；
   - 用导数符号或界完成结论。
6. **反导数与微分方程**
   - 结果求导回验；
   - 分离前查平衡解；
   - 初值定常数；
   - 写最大定义区间。
<!-- bilingual-en:start -->
1. **Linear and quadratic approximation**
   - State the base point $a$ first.
   - Compute $f(a)$, $f'(a)$, and $f''(a)$.
   - Include the factor $1/2$ in the quadratic term.
   - In products of expansions, truncate by total degree.
2. **Curve sketching**
   - Determine the domain, discontinuities, endpoints, and behavior at infinity.
   - Use the sign of $f'$ and the critical points.
   - Use the sign of $f''$ and identify genuine inflection points.
   - Check intercepts and symmetry.
3. **Optimization**
   - Draw a diagram and choose variables.
   - Write the constraint, objective, and feasible domain.
   - Reduce the problem to one free variable.
   - Compare critical points, endpoints, and discontinuities.
4. **Related rates**
   - Write a relation that holds at every time.
   - Differentiate with respect to $t$.
   - Substitute the instantaneous values only afterward.
   - Report direction, units, and speed or magnitude.
5. **MVT**
   - State the continuity and differentiability assumptions first.
   - Then write $f(b)-f(a)=f'(c)(b-a)$.
   - Finish the argument using the sign or bounds of the derivative.
6. **Antiderivatives and differential equations**
   - Differentiate the result to check it.
   - Look for equilibrium solutions before separation.
   - Use the initial condition to determine the constant.
   - State the maximal interval of definition.
<!-- bilingual-en:end -->

### 贯穿全章的一个问题
<!-- bilingual-en:start -->
*a question that runs through the chapter*
<!-- bilingual-en:end -->

本章所有方法都在回答：
<!-- bilingual-en:start -->
All methods in this chapter are answering:
<!-- bilingual-en:end -->

> 已知函数的导数，能对函数本身说什么？
> <!-- bilingual-en:start -->
> What can be said about the function itself when the derivative of the function is known?
> <!-- bilingual-en:end -->

- $f'(a)$ 给局部直线；
- $f''(a)$ 给局部曲率；
- $f'$ 的符号给单调；
- MVT 把局部斜率变成整体差值；
- 反导数从全部斜率恢复函数；
- 微分方程从变化规律选出函数族。
<!-- bilingual-en:start -->
- $f'(a)$ to local lines;
- $f''(a)$ gives local curvature;
- $f'$'s sign is monotonous;
- MVT turns local slopes into global differences;
- The inverse derivative recovers the function from the total slope;
- Differential equations select a family of functions from the law of change.
<!-- bilingual-en:end -->

### 易错点与考试策略
<!-- bilingual-en:start -->
*False Points and Examination Strategies*
<!-- bilingual-en:end -->

- 先写结构再代数，避免在长计算中忘记定义域和所求量。
- 图像题用符号表而非凭外观；优化题的“端点”可能是 $0^+$ 或 $\infty$。
- 若答案维度不对、Newton 残差变大、反导求导不回原式，应立即停下回查。
<!-- bilingual-en:start -->
- Write the structure before doing the algebra, so a long calculation does not obscure the domain or the requested quantity.
- In graphing problems, use a sign chart rather than visual guesswork. In optimization, a boundary may be approached as $0^+$ or $\infty$ rather than attained as an ordinary endpoint.
- If the dimensions are wrong, a Newton residual grows, or differentiating an antiderivative fails to recover the integrand, stop and check the work immediately.
<!-- bilingual-en:end -->

> [!question]- 三道自检题与答案
> 1. 哪类题最常因漏端点丢分？
> 优化和绝对极值题。
>
> 2. 哪类题必须“先求导后代值”？
> 相关变化率题。
>
> 3. 哪两类题最适合用求导回验？
> 不定积分和微分方程。
> <!-- bilingual-en:start -->
> 1. Which problems most often lose marks because endpoints were omitted?
> Optimization and absolute-extremum problems.
>
> 2. In which problems must you “differentiate first, then substitute values”?
> Related-rates problems.
>
> 3. Which two types of problem are especially well suited to checking by differentiation?
> Indefinite integrals and differential equations.
> <!-- bilingual-en:end -->

### 本地材料

- [[Ses41a_Lecture_Notes.pdf#page=1|41a Review for Test 2]]

**知识链小结：**考试复习不是再记一遍公式，而是把每类题的输入、主步骤、边界检查和回验固定成流程。
<!-- bilingual-en:start -->
** Knowledge Chain Summary:**Examination review is not a one-time formula, but the input of each type of questions, the main steps, boundary check and feedback fixed into a process.
<!-- bilingual-en:end -->

## Session 42：Materials for Exam 2

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题：**能否在一套综合题中，同时正确使用近似、相关变化率、作图、优化、Newton 法和 MVT？
<!-- bilingual-en:start -->
**Question:** Can approximation, related rates, graph sketching, optimization, Newton's method, and the MVT all be used correctly within one integrated set of problems?
<!-- bilingual-en:end -->

**材料说明：**官网 Session 42 是 Exam 2 材料页；本地没有 Ses42 讲义，以考试原题与官方解答组成。本节对官方简答补足推导，并明确两处可见笔误。
<!-- bilingual-en:start -->
**Materials:** Session 42 on the official site is the Exam 2 materials page. There is no separate Ses42 handout in the local archive, so this section uses the exam paper and official solutions. It supplies derivations omitted from the brief official answers and identifies two apparent typographical errors.
<!-- bilingual-en:end -->

### Exam 2 第 1 题：二次近似与 $\ln1.2$
<!-- bilingual-en:start -->
*Exam 2 Question 1: Quadratic Approximation and $\ln1.2$*
<!-- bilingual-en:end -->

> [!example] 题目
> (a) 写出二阶可导函数 $f$ 在 $x=a$ 的一般二次近似。
> (b) 用它估计 $\ln1.2$。
> <!-- bilingual-en:start -->
> (a) Write the general quadratic approximation to a twice-differentiable function $f$ about $x=a$.
> (b) Use it to estimate $\ln1.2$.
> <!-- bilingual-en:end -->

**(a)** 令二次多项式在 $a$ 匹配 $f,f',f''$：
<!-- bilingual-en:start -->
**(a)**Let the quadratic polynomial match $f,f',f''$ at $a$:
<!-- bilingual-en:end -->

$$
\boxed{
Q_a(x)=f(a)+f'(a)(x-a)+\frac{f''(a)}2(x-a)^2
}.
$$

**(b)** 取 $f(x)=\ln x,a=1$：
<!-- bilingual-en:start -->
** (b) **Take $f(x)=\ln x,a=1$:
<!-- bilingual-en:end -->

$$
f(1)=0,\qquad f'(1)=1,\qquad f''(1)=-1.
$$

代 $x=1.2$：
<!-- bilingual-en:start -->
With $x=1.2$:
<!-- bilingual-en:end -->

$$
\begin{aligned}
\ln1.2
&\approx0+1(0.2)-\frac12(0.2)^2\\
&=0.2-0.02\\
&=\boxed{0.18}.
\end{aligned}
$$

真值约 $0.18232$，误差约 $0.00232$；二次近似比线性值 $0.2$ 更准。
<!-- bilingual-en:start -->
The true value is approximately $0.18232$, giving an error of about $0.00232$; the quadratic approximation is more accurate than the linear estimate $0.2$.
<!-- bilingual-en:end -->

> [!warning] 常见错误
> 忘记 $f''(a)/2$ 的 $1/2$；把展开点错取为 $0$，而 $\ln0$ 无定义。
> <!-- bilingual-en:start -->
> Forget $f''(a)/2$'s $1/2$; misplace the expansion point as $0$, and $\ln0$ is undefined.
> <!-- bilingual-en:end -->

### Exam 2 第 2 题：圆锥盐堆
<!-- bilingual-en:start -->
*Exam 2 Question 2: Conical salt stack*
<!-- bilingual-en:end -->

> [!example] 题目
> 盐以 $30\ \mathrm{ft^3/min}$ 落下形成圆锥，圆锥高度始终等于底面直径。高为 $10$ ft 时，求高度增长率。
> <!-- bilingual-en:start -->
> Salt falls at $30\ \mathrm{ft^3/min}$ to form a cone, which is always the same height as the bottom diameter.  When the height is $10$ ft, the height growth rate is calculated.
> <!-- bilingual-en:end -->

设半径 $r$、高 $h$。条件 $h=2r$，即 $r=h/2$。体积
<!-- bilingual-en:start -->
Set a radius of $r$ and a height of $h$.  The condition $h=2r$, or $r=h/2$.  volume
<!-- bilingual-en:end -->

$$
V=\frac13\pi r^2h
=\frac13\pi\left(\frac h2\right)^2h
=\frac{\pi}{12}h^3.
$$

对时间求导：
<!-- bilingual-en:start -->
Differentiate with respect to time:
<!-- bilingual-en:end -->

$$
\frac{dV}{dt}
=\frac{\pi}{4}h^2\frac{dh}{dt}.
$$

代入 $dV/dt=30,h=10$：
<!-- bilingual-en:start -->
Introduce $dV/dt=30,h=10$:
<!-- bilingual-en:end -->

$$
30=25\pi\frac{dh}{dt}.
$$

$$
\boxed{\frac{dh}{dt}=\frac6{5\pi}\ \mathrm{ft/min}}.
$$

> [!warning] 常见错误
> 把“高度等于直径”写成 $h=r$；或在求导前把 $h=10$ 当成恒定值。
> <!-- bilingual-en:start -->
> Write "height equals diameter" as $h=r$; or consider $h=10$ as a constant value before deriving.
> <!-- bilingual-en:end -->

### Exam 2 第 3 题：$f(x)=x-3x^{1/3}$ 作图
<!-- bilingual-en:start -->
*Exam 2 Question 3: Sketching $f(x)=x-3x^{1/3}$*
<!-- bilingual-en:end -->

> [!example] 题目
> 标出局部极值、增减区间和渐近线；拐点可辅助作图。
> <!-- bilingual-en:start -->
> Mark the local extrema, intervals of increase and decrease, and any asymptotes; inflection points may be used to refine the sketch.
> <!-- bilingual-en:end -->

定义域为全体实数，且 $f$ 为奇函数。零点：
<!-- bilingual-en:start -->
The domain is all real numbers, and $f$ is odd. Its zeros satisfy
<!-- bilingual-en:end -->

$$
x-3x^{1/3}=x^{1/3}(x^{2/3}-3)=0,
$$

所以
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

$$
\boxed{x=-3\sqrt3,\ 0,\ 3\sqrt3}.
$$

对 $x\ne0$：
<!-- bilingual-en:start -->
For $x\ne0$:
<!-- bilingual-en:end -->

$$
f'(x)=1-x^{-2/3}=1-\frac1{|x|^{2/3}}.
$$

临界候选为 $x=-1,0,1$；$0$ 处函数存在但导数无穷。符号表：
<!-- bilingual-en:start -->
The critical-point candidates are $x=-1,0,1$. The function is defined at $0$, but its derivative is infinite there. The sign chart is
<!-- bilingual-en:end -->

| 区间 | $(-\infty,-1)$ | $(-1,0)$ | $(0,1)$ | $(1,\infty)$ |
|---|---:|---:|---:|---:|
| $f'$ | $+$ | $-$ | $-$ | $+$ |

因此：
<!-- bilingual-en:start -->
Therefore:
<!-- bilingual-en:end -->

$$
\boxed{\text{递增于 }(-\infty,-1)\text{ 与 }(1,\infty)}
$$

$$
\boxed{\text{递减于 }(-1,0)\text{ 与 }(0,1)}
$$

（也可在单调意义上合写为 $(-1,1)$，但导数在 $0$ 不存在。）
<!-- bilingual-en:start -->
(It can also be written as $(-1,1)$ in the monotonic sense, but the derivative does not exist at $0$.)
<!-- bilingual-en:end -->

$$
f(-1)=2,\qquad f(1)=-2.
$$

故局部最大 $\boxed{(-1,2)}$，局部最小 $\boxed{(1,-2)}$。$x=0$ 有向下的竖直切线，凹凸在其两侧改变。无竖直、水平或斜渐近线；尽管 $f(x)/x\to1$，但 $f(x)-x=-3x^{1/3}$ 不趋于 $0$。两端 $f(x)\to\pm\infty$。
<!-- bilingual-en:start -->
Thus the local maximum is $\boxed{(-1,2)}$ and the local minimum is $\boxed{(1,-2)}$. At $x=0$ the graph has a downward-oriented vertical tangent, and its concavity changes across the origin. There are no vertical, horizontal, or oblique asymptotes: although $f(x)/x\to1$, the difference $f(x)-x=-3x^{1/3}$ does not tend to $0$. At the two ends, $f(x)\to\pm\infty$.
<!-- bilingual-en:end -->

> [!warning] 官方勘误
> 本地官方解答文字把局部最大点写成了 $(1,2)$；由函数值与奇对称性可确认应为 $\boxed{(-1,2)}$。
> <!-- bilingual-en:start -->
> The local official answer is written as $(1,2)$; the value of the function and the odd symmetry confirm that it should be $\boxed{(-1,2)}$.
> <!-- bilingual-en:end -->

### Exam 2 第 4 题：圆柱加半球顶储罐
<!-- bilingual-en:start -->
*Exam 2 Question 4: Cylindrical plus hemispherical top tanks*
<!-- bilingual-en:end -->

> [!example] 题目
> 圆柱有底、上接半球顶，固定体积 $V$；求耗金属最少的尺寸。
> <!-- bilingual-en:start -->
> The cylinder has a bottom and an upper hemispherical top, and the fixed volume is $V$; and the dimension which consumes the least metal is calculated.
> <!-- bilingual-en:end -->

设共同半径 $r>0$，圆柱部分高 $h\ge0$。体积：
<!-- bilingual-en:start -->
A common radius of $r>0$ is set, and the height of the cylindrical part is $h\ge0$.  Volume:
<!-- bilingual-en:end -->

$$
V=\pi r^2h+\frac23\pi r^3.
$$

外表面积包含圆柱底、圆柱侧面、半球曲面：
<!-- bilingual-en:start -->
The outer surface area includes the bottom of the cylinder, the side of the cylinder, the hemispherical surface:
<!-- bilingual-en:end -->

$$
S=\pi r^2+2\pi rh+2\pi r^2
=3\pi r^2+2\pi rh.
$$

由体积约束
<!-- bilingual-en:start -->
Constrained by Volume
<!-- bilingual-en:end -->

$$
h=\frac{V}{\pi r^2}-\frac23r.
$$

代入：
<!-- bilingual-en:start -->
Substitute:
<!-- bilingual-en:end -->

$$
S(r)=3\pi r^2+2\pi r
\left(\frac{V}{\pi r^2}-\frac23r\right)
=\frac53\pi r^2+\frac{2V}{r}.
$$

$$
S'(r)=\frac{10}{3}\pi r-\frac{2V}{r^2}.
$$

令 $S'=0$：
<!-- bilingual-en:start -->
Let $S'=0$:
<!-- bilingual-en:end -->

$$
\frac{10}{3}\pi r^3=2V
\Rightarrow
\boxed{r=\left(\frac{3V}{5\pi}\right)^{1/3}}.
$$

由 $V=\frac53\pi r^3$ 回代：
<!-- bilingual-en:start -->
Generated by $V=\frac53\pi r^3$:
<!-- bilingual-en:end -->

$$
h=\frac{5r}{3}-\frac{2r}{3}=r.
$$

所以
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

$$
\boxed{h=r=\left(\frac{3V}{5\pi}\right)^{1/3}}.
$$

并且
<!-- bilingual-en:start -->
and
<!-- bilingual-en:end -->

$$
S''(r)=\frac{10}{3}\pi+\frac{4V}{r^3}>0,
$$

故 $S$ 严格凸，唯一可行临界点为全局最小；它也满足 $h>0$。
<!-- bilingual-en:start -->
Therefore, $S$ is strictly convex and the only feasible critical point is the global minimum, which also satisfies $h>0$.
<!-- bilingual-en:end -->

> [!warning] 常见错误
> 漏算底面积或把半球曲面写成 $4\pi r^2$；求出 $r$ 后忘记回答 $h$。
> <!-- bilingual-en:start -->
> We can omit the bottom area or write the hemispherical surface as $4\pi r^2$; we can't answer $h$ when we get $r$.
> <!-- bilingual-en:end -->

### Exam 2 第 5 题：Newton 法为什么失败
<!-- bilingual-en:start -->
*Exam 2 Question 5: Why Newton failed*
<!-- bilingual-en:end -->

> [!example] 题目
> $f(x)=x^3-3x+7$，初值 $x_1=2$，解释为何迭代最终失败。
> <!-- bilingual-en:start -->
> $f(x)=x^3-3x+7$, the initial value $x_1=2$, explains why the iteration ultimately fails.
> <!-- bilingual-en:end -->

$$
f'(x)=3x^2-3.
$$

第一步：
<!-- bilingual-en:start -->
Step 1:
<!-- bilingual-en:end -->

$$
x_2
=2-\frac{f(2)}{f'(2)}
=2-\frac{8-6+7}{12-3}
=2-\frac99
=1.
$$

但
<!-- bilingual-en:start -->
but
<!-- bilingual-en:end -->

$$
f'(1)=0,\qquad f(1)=5\ne0.
$$

所以下一步
<!-- bilingual-en:start -->
so next step
<!-- bilingual-en:end -->

$$
x_3=1-\frac{f(1)}{f'(1)}
$$

分母为零而无定义。几何上，$(1,5)$ 处切线水平，不与 $x$ 轴相交，无法产生下一猜测。
<!-- bilingual-en:start -->
The denominator is zero and has no definition.  Geometrically, the tangent at $(1,5)$ is horizontal and does not intersect the $x$ axis to produce the next guess.
<!-- bilingual-en:end -->

$$
\boxed{\text{迭代在到达 }x=1\text{ 后因水平切线停止。}}
$$

> [!warning] 常见错误
> 只说“除以零”而没有展示哪一步到达该点；或误以为 $f'(1)=0$ 表示 $x=1$ 是方程的根。
> <!-- bilingual-en:start -->
> Just say "divide by zero" without showing which step reaches the point; or mistakenly assume that $f'(1)=0$ means $x=1$ is the root of the equation.
> <!-- bilingual-en:end -->

### Exam 2 第 6 题：用 MVT 证明平方根不等式
<!-- bilingual-en:start -->
*Exam 2 Question 6: Proving Square Root Inequalities with MVT*
<!-- bilingual-en:end -->

> [!example] 题目
> 证明当 $x>0$ 时
> $$
> \sqrt{1+x}<1+\frac x2.
> $$
> <!-- bilingual-en:start -->
> Prove that for $x>0$,
> $$
> \sqrt{1+x}<1+\frac x2.
> $$
> <!-- bilingual-en:end -->

令
<!-- bilingual-en:start -->
order
<!-- bilingual-en:end -->

$$
g(x)=1+\frac x2-\sqrt{1+x}.
$$

$g$ 在 $[0,x]$ 连续、在 $(0,x)$ 可导，且
<!-- bilingual-en:start -->
$g$ is continuous at $[0,x]$, differentiable at $(0,x)$, and
<!-- bilingual-en:end -->

$$
g(0)=0.
$$

对任意 $t>0$：
<!-- bilingual-en:start -->
For any $t>0$:
<!-- bilingual-en:end -->

$$
g'(t)=\frac12-\frac1{2\sqrt{1+t}}.
$$

因 $\sqrt{1+t}>1$，
<!-- bilingual-en:start -->
Because $\sqrt{1+t}>1$,
<!-- bilingual-en:end -->

$$
\frac1{2\sqrt{1+t}}<\frac12,
$$

故 $g'(t)>0$。MVT 的单调性推论给 $g(x)>g(0)=0$，即
<!-- bilingual-en:start -->
So, $g'(t)>0$.  The monotonicity of MVT is inferred to $g(x)>g(0)=0$, i.e.
<!-- bilingual-en:end -->

$$
\boxed{\sqrt{1+x}<1+\frac x2}.
$$

也可直接对 $\sqrt{1+t}$ 在 $[0,x]$ 应用 MVT，得到同一证明。
<!-- bilingual-en:start -->
The same proof can also be obtained by applying MVT directly to $\sqrt{1+t}$ in $[0,x]$.
<!-- bilingual-en:end -->

> [!warning] 常见错误
> 用“图上看切线在曲线上方”代替证明；正确逻辑需要由导数符号或凹性定理推出。
> <!-- bilingual-en:start -->
> Instead of proving, "see tangent above curve on graph"; the correct logic needs to be deduced from the derivative sign or the concave theorem.
> <!-- bilingual-en:end -->

### Exam 2 总结与错误诊断
<!-- bilingual-en:start -->
*Exam 2 Summary and Troubleshooting*
<!-- bilingual-en:end -->

| 题号 | 核心规则 | 必做检查 |
|---|---|---|
| 1 | 二次近似 | 基点、$1/2$、误差量级 |
| 2 | 相关变化率 | 相似/比例关系、先求导后代值、单位 |
| 3 | 曲线作图 | $f'$ 不存在点、符号表、零点、渐近线定义 |
| 4 | 约束优化 | 表面积组成、可行域、全局最小证明 |
| 5 | Newton 法 | 实际迭代、分母 $f'$、几何解释 |
| 6 | MVT | 连续/可导假设、基准值、严格不等号 |

### 易错点与边界
<!-- bilingual-en:start -->
*Fault-prone points and boundaries*
<!-- bilingual-en:end -->

- 计算正确但不回答单位、位置或全部尺寸，仍是不完整答案。
- 图像题的 $f'$ 不存在点与优化题的可行边界都必须主动加入候选。
- 使用官方答案时也应回验；本卷第 3 题的极大点文字确有笔误。
<!-- bilingual-en:start -->
- Calculate correctly without answering units, positions, or all dimensions, and remain incomplete.
- The $f'$ dots of the image problem and the feasible boundary of the optimization problem must be actively added candidates.
- Use of official answers should also be checked; the maximalist wording of question 3 of this volume is incorrect.
<!-- bilingual-en:end -->

> [!question]- 三道综合自检与答案
> 1. 第 2 题若盐流入率翻倍，高度增长率如何变化？
> 在同一高度下线性翻倍，因为 $h'$ 与 $V'$ 成正比。
>
> 2. 第 4 题的最优关系中，半球总高是否等于 $r$？
> 不是；$h=r$ 指圆柱部分高度，罐体总高为 $h+r=2r$。
>
> 3. 第 6 题为何是严格小于而非小于等于？
> $x>0$ 时区间内部 $g'(t)>0$，故 $g(x)>g(0)$；只有 $x=0$ 取等。
> <!-- bilingual-en:start -->
> 1. Question 2 How does the high growth rate change if the salt inflow rate doubles?
> Linear doubling at the same height, because $h'$ is proportional to $V'$.
> 2. In Question 4, is the total height of the hemisphere equal to $r$?
> No; $h=r$ refers to the height of the cylindrical portion and the total tank height is $h+r=2r$.
> 3. Why is Question 6 strictly less than, not less than or equal to?
> $x>0$ is $g'(t)>0$ inside the interval, so $g(x)>g(0)$; only $x=0$ is equal.
> <!-- bilingual-en:end -->

### 本地材料

- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/07_Exams/Exam2_Problems.pdf#page=1|Exam 2 原题（题 1–6 各页）]]
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/07_Exams/Exam2_Solutions.pdf#page=1|Exam 2 官方解答]]

**知识链小结：**Exam 2 检查的是同一能力的六种外观：从导数提取可验证的信息，并对近似范围、定义域、约束和定理假设负责。
<!-- bilingual-en:start -->
**Knowledge chain summary:** Exam 2 tests six forms of the same ability: extracting verifiable information from derivatives while keeping track of approximation ranges, domains, constraints, and theorem hypotheses.
<!-- bilingual-en:end -->

---

## 本章总复习：从局部到整体再到逆问题
<!-- bilingual-en:start -->
*Chapter Review: From Local to Global to Inverse Problems*
<!-- bilingual-en:end -->

$$
\begin{array}{c}
f'(a),f''(a)\\
\Downarrow\\
\text{线性/二次局部模型}\\
\Downarrow\\
\text{作图、优化、相关变化率、Newton 法}\\
\Downarrow\ \text{MVT 严格连接}\\
\text{由导数控制整体差值}\\
\Downarrow\\
\text{反导数与微分方程：由变化率恢复函数}
\end{array}
$$

### 最终自检清单
<!-- bilingual-en:start -->
*Final Self-Checklist*
<!-- bilingual-en:end -->

- [ ] 我能解释线性化、二次近似匹配了哪些导数，而不只会套公式。
- [ ] 我画图时总先写定义域，并区分临界点、间断点与真正拐点。
- [ ] 我做优化时会比较端点；做相关变化率时会先求导后代值。
- [ ] 我能从切线方程自行推导 Newton 公式，并识别 $f'=0$ 的失败。
- [ ] 我能完整写出 MVT 的闭区间连续、开区间可导和 $c\in(a,b)$。
- [ ] 我会对不定积分和微分方程解求导回验，并补回平衡解、定义区间。
<!-- bilingual-en:start -->
- [ ] I can explain which derivatives are matched by linearization, quadratic approximation and not just by formulae.
- [ ] When sketching a graph, I always write the domain first and distinguish among critical points, discontinuities, and genuine inflection points.
- [ ] I compare endpoints when doing optimization; I derive descendants when doing related rate of change.
- [ ] I can deduce the Newton formula from the tangent equation and identify the failure of the $f'=0$.
- [ ] I can state the MVT completely: continuity on the closed interval, differentiability on the open interval, and $c\in(a,b)$.
- [ ] I will do derivative tests on the solutions of indefinite integrals and differential equations, and replenish the equilibrium solutions and define the intervals.
<!-- bilingual-en:end -->

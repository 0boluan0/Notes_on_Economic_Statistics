---
aliases:
  - MIT 18.01SC Differentiation
  - MIT 18.01SC 微分
  - Differentiation
tags:
  - math/calculus
  - course/mit-ocw
  - calculus/differentiation
source: https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/pages/1.-differentiation/
---

# Differentiation

> [!abstract] 本章主线
> 微分学把“在一小段区间内平均变化多少”推进为“在某一瞬间怎样变化”。几何上，这个数是切线斜率；物理上，它是瞬时速度；在经济、测量和工程中，它又是边际量与灵敏度。全章都从一个极限出发：
> $$
> f'(x_0)=\lim_{h\to0}\frac{f(x_0+h)-f(x_0)}{h}.
> $$
> 后面的幂、积、商、链式法则，以及三角、指数、对数、反函数的导数，都是在不改变这一定义的前提下，把重复的极限计算压缩成可靠的规则。
> <!-- bilingual-en:start -->
> Differential calculus turns “how much does a quantity change on average over a short interval?” into “how fast is it changing at this instant?” Geometrically, the answer is the slope of a tangent line; physically, it is instantaneous velocity; in economics, measurement, and engineering, it appears as a marginal quantity or sensitivity. The chapter begins with one limit:
> $$
> f'(x_0)=\lim_{h\to0}\frac{f(x_0+h)-f(x_0)}{h}.
> $$
> The power, product, quotient, and chain rules—and the derivatives of trigonometric, exponential, logarithmic, and inverse functions—compress repeated limit calculations into reliable rules without changing this underlying definition.
> <!-- bilingual-en:end -->

- 课程来源：[MIT OpenCourseWare 18.01SC - Unit 1: Differentiation](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/pages/1.-differentiation/)
- 教师：David Jerison；学期：Fall 2010。
- 官方顺序：Part A（Session 1-12）→ Problem Set 1 → Part B（Session 13-20）→ Problem Set 2 → Exam 1（Session 21-22）。
- 本地材料说明：同一 Session 的 `a/b/c...` 是视频片段顺序；笔记正文按这个顺序整合。PDF 不负责导航，但每节末保留精确入口。
<!-- bilingual-en:start -->
- Course source: [MIT OpenCourseWare 18.01SC — Unit 1: Differentiation](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/pages/1.-differentiation/)
- Instructor: David Jerison; term: Fall 2010.
- Official sequence: Part A (Sessions 1–12) → Problem Set 1 → Part B (Sessions 13–20) → Problem Set 2 → Exam 1 (Sessions 21–22).
- Local-material note: the `a/b/c...` suffixes within a session indicate the order of the video clips, and the body of this note follows that order. The PDFs are not the primary navigation layer, but each section retains a precise link to its source.
<!-- bilingual-en:end -->

## 怎样使用这篇笔记
<!-- bilingual-en:start -->
*How to use this note*
<!-- bilingual-en:end -->

1. 先读每节的“问题与前置知识”，明确本节究竟在解决什么。
2. 证明不要只背结论：依次检查目标、构造、每一步依据、使用的假设和边界情形。
3. 代表例题先遮住解答自己做；再用“符号、定义域、单位、图像趋势”四项检查。
4. 每节最后完成三道自检题。答案折叠，适合第二次复习时主动回忆。
5. 题目图形若依赖原印刷页，正文会给出解析描述，同时链接对应 PDF 页。
<!-- bilingual-en:start -->

&nbsp;
**1.** Begin each section with “Questions and Prerequisites” so that you know exactly what problem the section is solving.<br>
**2.** Do not memorize a proof only as a conclusion. Check its goal, construction, justification for each step, assumptions, and boundary cases.<br>
**3.** Hide the solution to each representative problem and attempt it first. Then check the sign, domain, units, and qualitative graph behavior.<br>
**4.** Complete the three self-check questions at the end of each section. Their answers are collapsed so that you can use them for active recall on a second pass.<br>
**5.** When a problem depends on a figure from the printed material, the note provides an analytic description and links to the relevant PDF page.<br>
<!-- bilingual-en:end -->

## 学习目标
<!-- bilingual-en:start -->
*learning objectives*
<!-- bilingual-en:end -->

学完本章后，应当能够：
<!-- bilingual-en:start -->
Upon completion of this chapter, you shall be able to:
<!-- bilingual-en:end -->

1. 从割线极限定义导数，并在几何、运动、单位和误差传播之间切换解释。
2. 区分函数值、极限、连续与可导；识别可去、跳跃、无穷和振荡间断。
3. 从定义证明线性法则、积法则，并熟练使用幂、积、商、链式法则。
4. 在弧度制下证明基本三角极限，进而推导正弦、余弦及其他三角导数。
5. 用隐函数求导处理关系式，并推导反函数、反三角函数的导数。
6. 理解 $e$ 的选择、指数与对数互逆、对数求导、变量幂和双曲函数。
7. 独立完成本章两套指定作业和 Exam 1，并能说明每个步骤使用了什么规则。
<!-- bilingual-en:start -->

&nbsp;
**1.** Define the derivative as a limit of secant slopes and interpret it in terms of geometry, motion, units, and error propagation.<br>
**2.** Distinguish a function value, a limit, continuity, and differentiability; identify removable, jump, infinite, and oscillatory discontinuities.<br>
**3.** Prove the linearity and product rules from the definition, and apply the power, product, quotient, and chain rules fluently.<br>
**4.** Prove the basic trigonometric limits in radians and use them to derive the derivatives of sine, cosine, and the other trigonometric functions.<br>
**5.** Differentiate implicit relations and derive the derivative rules for inverse and inverse-trigonometric functions.<br>
**6.** Understand why the base $e$ is special, how exponential and logarithmic functions are inverses, and how to use logarithmic differentiation, variable powers, and hyperbolic functions.<br>
**7.** Complete both assigned problem sets and Exam 1 independently, explaining the rule used at every step.<br>
<!-- bilingual-en:end -->

## 课程导航

| 官方位置      | 内容                                       |                                                                    |                                                                            |                                                   |                                                                                       |                                                                                  |                                                               |                                                            |                                       |                                        |                                      |                                               |            |
| --------- | ---------------------------------------- | ------------------------------------------------------------------ | -------------------------------------------------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------- | ---------------------------------------------------------- | ------------------------------------- | -------------------------------------- | ------------------------------------ | --------------------------------------------- | ---------- |
| Part A    | [[#Session 1：Introduction to Derivatives | S1 导数简介]] · [[#Session 2：Examples of Derivatives                    | S2 定义计算]] · [[#Session 3：Derivative as Rate of Change                      | S3 变化率]] · [[#Session 4：Limits and Continuity     | S4 极限与连续]] · [[#Session 5：Discontinuity                                               | S5 间断]] · [[#Session 6：Calculating Derivatives                                   | S6 基本规则]] · [[#Session 7：Derivatives of Sine and Cosine       | S7 正余弦导数]] · [[#Session 8：Limits of Sine and Cosine        | S8 三角极限]] · [[#Session 9：Product Rule | S9 积法则]] · [[#Session 10：Quotient Rule | S10 商法则]] · [[#Session 11：Chain Rule | S11 链式法则]] · [[#Session 12：Higher Derivatives | S12 高阶导数]] |
| Part A 练习 | [[#Problem Set 1                         | Problem Set 1]]                                                    |                                                                            |                                                   |                                                                                       |                                                                                  |                                                               |                                                            |                                       |                                        |                                      |                                               |            |
| Part B    | [[#Session 13：Implicit Differentiation   | S13 隐函数与有理幂]] · [[#Session 14：Examples of Implicit Differentiation | S14 隐函数例题]] · [[#Session 15：Implicit Differentiation and Inverse Functions | S15 反函数]] · [[#Session 16：The Derivative of $a^x$ | S16 $a^x$]] · [[#Session 17：The Exponential Function, Its Derivative, and Its Inverse | S17 $e^x$ 与 $\ln x$]] · [[#Session 18：Derivatives of Other Exponential Functions | S18 对数求导]] · [[#Session 19：An Interesting Limit Involving $e$ | S19 关于 $e$ 的极限]] · [[#Session 20：Hyperbolic Trig Functions | S20 双曲函数]]                            |                                        |                                      |                                               |            |
| Part B 练习 | [[#Problem Set 2                         | Problem Set 2]]                                                    |                                                                            |                                                   |                                                                                       |                                                                                  |                                                               |                                                            |                                       |                                        |                                      |                                               |            |
| Exam 1    | [[#Session 21：Review for Exam 1          | S21 综合复习]] · [[#Session 22：Materials for Exam 1                    | S22 完整题解]]                                                                 |                                                   |                                                                                       |                                                                                  |                                                               |                                                            |                                       |                                        |                                      |                                               |            |

---

## Part A：Definition and Basic Rules

## Session 1：Introduction to Derivatives

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

问题 :只给一条曲线和曲线上一点，怎样定义并计算该点的“方向”？
<!-- bilingual-en:start -->
Question: Given a curve and a point on it, how can we define and calculate the curve's direction at that point?
<!-- bilingual-en:end -->

前置知识：两点间直线斜率 $m=(y_2-y_1)/(x_2-x_1)$、点斜式 $y-y_0=m(x-x_0)$、极限的直观含义。
<!-- bilingual-en:start -->
Prerequisites: the slope between two points, $m=(y_2-y_1)/(x_2-x_1)$; point–slope form, $y-y_0=m(x-x_0)$; and the intuitive meaning of a limit.
<!-- bilingual-en:end -->

### 按片段展开：从几何问题到主公式
<!-- bilingual-en:start -->
*Expanding by Pieces: From Geometry Problems to Master Formulas*
<!-- bilingual-en:end -->

**01a - Welcome。** 微分的用途远超画切线：所有“量怎样随另一个量改变”的问题都可能需要导数。课程最终目标之一，是能计算诸如 $e^{x\arctan x}$ 这类多层复合函数的导数。
<!-- bilingual-en:start -->
**01a — Welcome.** Differentiation is useful far beyond drawing tangent lines: any question about how one quantity changes with another may require a derivative. One eventual goal is to differentiate multilevel composite functions such as $e^{x\arctan x}$.
<!-- bilingual-en:end -->

**01b - Geometric interpretation。** 设曲线为 $y=f(x)$，固定
<!-- bilingual-en:start -->
**01b — Geometric interpretation.** Let the curve be $y=f(x)$ and fix the point
<!-- bilingual-en:end -->

$$
P=(x_0,f(x_0)).
$$

经过 $P$ 的切线（tangent line）不是“只接触曲线一次的线”。一条切线可以在别处再次穿过曲线；有些只交一次的线也未必反映局部方向。正确的几何思想是：切线是附近割线（secant line）的极限位置，也是曲线在 $P$ 附近的最佳直线近似。
<!-- bilingual-en:start -->
The tangent line through $P$ is not merely “a line that touches the curve once.” A tangent may cross the curve elsewhere, and a line that intersects the curve only once need not describe its local direction. The right geometric idea is that the tangent is the limiting position of nearby secant lines—and the best linear approximation to the curve near $P$.
<!-- bilingual-en:end -->

**01c - Geometric definition。** 在曲线上再取移动点
<!-- bilingual-en:start -->
**01c — Geometric definition.** Choose a second, moving point on the curve,
<!-- bilingual-en:end -->

$$
Q=(x_0+h,f(x_0+h)),\qquad h\ne0.
$$

$P,Q$ 确定割线。让 $Q$ 沿曲线趋近 $P$，等价于让 $h\to0$。若割线斜率趋向一个唯一有限数，就把这个数定义为切线斜率。
<!-- bilingual-en:start -->
The points $P$ and $Q$ determine a secant line. Letting $Q$ approach $P$ along the curve is equivalent to letting $h\to0$. If the secant slopes approach a unique finite number, that number is defined as the tangent slope.
<!-- bilingual-en:end -->

**01d - Slope as ratio。** 水平改变量和竖直改变量分别是
<!-- bilingual-en:start -->
**01d — Slope as a ratio.** The horizontal and vertical changes are
<!-- bilingual-en:end -->

$$
\Delta x=h,\qquad \Delta f=f(x_0+h)-f(x_0).
$$

割线斜率是
<!-- bilingual-en:start -->
The secant slope is
<!-- bilingual-en:end -->

$$
\frac{\Delta f}{\Delta x}
=\frac{f(x_0+h)-f(x_0)}h.
$$

**01e - Main formula。** 取极限便得到[[导数与求导规则#从差商到导数|导数]]定义；几何上它对应[[导数与求导规则#从差商到导数|导数的几何意义]]：
<!-- bilingual-en:start -->
**01e — Main formula.** Taking the limit gives the definition of the [[导数与求导规则#从差商到导数|derivative]]; geometrically, it is the [[导数与求导规则#从差商到导数|slope of the tangent line]]:
<!-- bilingual-en:end -->

> [!important] 点处导数
> 若下列双侧极限存在且为有限数，则 $f$ 在 $x_0$ 可导（differentiable）：
> $$
> f'(x_0)=\lim_{h\to0}\frac{f(x_0+h)-f(x_0)}h.
> $$
> 对应切线为
> $$
> y-f(x_0)=f'(x_0)(x-x_0).
> $$
> <!-- bilingual-en:start -->
> If the following two-sided limit exists and is finite, then $f$ is differentiable at $x_0$:
> $$
> f'(x_0)=\lim_{h\to0}\frac{f(x_0+h)-f(x_0)}h.
> $$
> The corresponding tangent line is
> $$
> y-f(x_0)=f'(x_0)(x-x_0).
> $$
> <!-- bilingual-en:end -->

![[98_attachment/MIT18.01SC/unit01-secant-tangent.png|900]]

### 代表例题：讲义中的割线实验
<!-- bilingual-en:start -->
*Representative Example: Secant Experiments in Handouts*
<!-- bilingual-en:end -->

课堂练习取
<!-- bilingual-en:start -->
The classroom exercise uses
<!-- bilingual-en:end -->

$$
f(x)=\frac12x^3-x.
$$

在一般点 $x$，步长为 $h$ 的割线斜率为
<!-- bilingual-en:start -->
At a general point $x$, with step size $h$, the secant slope is
<!-- bilingual-en:end -->

$$
\begin{aligned}
\frac{f(x+h)-f(x)}h
&=\frac{\frac12(x+h)^3-(x+h)-\left(\frac12x^3-x\right)}h\\
&=\frac{\frac12(3x^2h+3xh^2+h^3)-h}{h}\\
&=\frac32x^2+\frac32xh+\frac12h^2-1.
\end{aligned}
$$

令 $h\to0$：
<!-- bilingual-en:start -->
Let $h\to0$:
<!-- bilingual-en:end -->

$$
f'(x)=\frac32x^2-1.
$$

例如 $x=-0.75$ 时，切线斜率
<!-- bilingual-en:start -->
For example, when $x=-0.75$, the tangent slope
<!-- bilingual-en:end -->

$$
f'(-0.75)=\frac32(0.75)^2-1=-0.15625.
$$

练习表中 $h=-0.5,-0.25,0.25,0.5$ 得到的割线斜率约为 $0.53,0.16,-0.41,-0.59$；它们并不都很接近 $-0.15625$。这揭示一个重要事实：同样大小的 $h$ 在曲率较大的地方可能不够小。“趋近”不是固定精度，而是可按误差要求继续缩小 $|h|$。
<!-- bilingual-en:start -->
The secant slopes for $h=-0.5,-0.25,0.25,0.5$ in the exercise table are approximately $0.53,0.16,-0.41,-0.59$; they are not all close to $-0.15625$. This reveals an important fact: the same-sized $h$ may not be small enough where the curvature is large. “Approaching” is not a fixed level of accuracy; $|h|$ can be reduced further to meet a desired error tolerance.
<!-- bilingual-en:end -->

### 为什么不能直接令 $h=0$
<!-- bilingual-en:start -->
*Why Can't We Set $h=0$ Directly?*
<!-- bilingual-en:end -->

在原差商中令 $h=0$ 会得到 $0/0$，这不是一个数。正确顺序是：
<!-- bilingual-en:start -->
Setting $h=0$ in the original difference quotient gives $0/0$, which is not a number. The correct order is:
<!-- bilingual-en:end -->

1. 只讨论 $h\ne0$；
2. 用代数变形消去导致 $0/0$ 的公共因子；
3. 再研究 $h\to0$ 时化简后表达式趋向什么。
<!-- bilingual-en:start -->

&nbsp;
**1.** Work with $h\ne0$.<br>
**2.** Use algebra to cancel the common factor responsible for the indeterminate form $0/0$.<br>
**3.** Then determine the limit of the simplified expression as $h\to0$.<br>
<!-- bilingual-en:end -->

极限允许变量任意接近零，但不要求它在计算过程中等于零。
<!-- bilingual-en:start -->
Limits allow a variable to be arbitrarily close to zero, but do not require it to be equal to zero in the calculation.
<!-- bilingual-en:end -->

### 边界情况与易错点
<!-- bilingual-en:start -->
*Boundary Cases and Common Pitfalls*
<!-- bilingual-en:end -->

- 双侧差商极限必须相同。尖角处左右斜率不同，因此不可导。
- 极限趋于 $\pm\infty$ 时可以说有竖直切线，但按本课程“有限导数”约定仍不可导。
- $f'(x_0)$ 是一个数；$f'(x)$ 是随 $x$ 改变的新函数，不要混淆。
- 切线是局部近似，不保证在整个图像上接近曲线。
<!-- bilingual-en:start -->
- The two-sided limits of the difference quotient must agree. At a corner, the left- and right-hand slopes differ, so the function is not differentiable there.
- If the difference quotient tends to $\pm\infty$, the curve may have a vertical tangent, but under this course's finite-derivative convention the function is still not differentiable.
- $f'(x_0)$ is a number, whereas $f'(x)$ is a function of $x$; do not confuse them.
- A tangent line is a local approximation and need not remain close to the curve globally.
<!-- bilingual-en:end -->

### 三道自检
<!-- bilingual-en:start -->
*Three Quick Self-Checks*
<!-- bilingual-en:end -->

1. 写出 $f(x)=x^2$ 在 $x=3$ 的差商，并求切线。
2. 为什么“直线只与曲线相交一点”不能定义切线？
3. 对 $f(x)=\frac12x^3-x$，在 $x=0$ 且 $h=0.25$ 时，割线斜率与切线斜率相差多少？
<!-- bilingual-en:start -->

&nbsp;
**1.** Write the difference quotient for $f(x)=x^2$ at $x=3$, and find the tangent line.<br>
**2.** Why can a tangent line not be defined merely as a line that intersects the curve at one point?<br>
**3.** For $f(x)=\frac12x^3-x$ at $x=0$ with $h=0.25$, what is the difference between the secant slope and the tangent slope?<br>
<!-- bilingual-en:end -->

> [!success]- 自检答案
> 1. $[(3+h)^2-9]/h=6+h\to6$，切线 $y-9=6(x-3)$。
> 2. 切线可能在别处再交曲线；相交次数是全局性质，不能刻画一点附近的方向。割线极限才是局部定义。
> 3. 割线斜率为 $\frac12h^2-1=-0.96875$，切线斜率 $f'(0)=-1$，绝对误差 $0.03125$。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** $[(3+h)^2-9]/h=6+h\to6$, tangent $y-9=6(x-3)$.<br>
> **2.** The tangent line may re-intersect the curve elsewhere; the number of intersections is a global property and cannot characterize the direction near a point.  The secant limit is the local definition.<br>
> **3.** The secant slope is $\frac12h^2-1=-0.96875$, the tangent slope is $f'(0)=-1$, and the absolute difference is $0.03125$.<br>
> <!-- bilingual-en:end -->

### 本地材料

- [[Ses01a_Lecture_Notes.pdf#page=1|01a Welcome to 18.01（p.1）]]
- [[Ses01b_Lecture_Notes.pdf#page=1|01b Geometric Interpretation（p.1）]]
- [[Ses01c_Lecture_Notes.pdf#page=1|01c Geometric Definition（p.1）]]
- [[Ses01d_Lecture_Notes.pdf#page=1|01d Slope as Ratio（p.1）]]
- [[Ses01e_Lecture_Notes.pdf#page=1|01e Main Formula（p.1）]]
- [[Ses01e_lec1ses1ex1_secants.pdf#page=1|课堂练习与答案：Secants and Tangents（pp.1-4）]]
- [[Exercise001_Problems.pdf#page=1|Exercise 001 原题]] · [[Exercise001_Solutions.pdf#page=1|Exercise 001 解答]]

**知识链：**两点斜率 → 割线 → 令第二点逼近第一点 → 切线斜率 → 导数定义。
<!-- bilingual-en:start -->
**Knowledge chain:** slope between two points → secant line → move the second point toward the first → tangent slope → definition of the derivative.
<!-- bilingual-en:end -->

## Session 2：Examples of Derivatives

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题：**怎样把抽象差商真正算出来？幂函数的统一规则从哪里来？
<!-- bilingual-en:start -->
**Question:** How do we evaluate an abstract difference quotient, and where does the general power rule come from?
<!-- bilingual-en:end -->

**前置知识：**导数定义、分式通分、二项式展开、直线截距与三角形面积。
<!-- bilingual-en:start -->
**Prerequisites:** the definition of the derivative, combining fractions over a common denominator, the binomial theorem, line intercepts, and triangle area.
<!-- bilingual-en:end -->

### 02a：由定义求 $f(x)=1/x$
<!-- bilingual-en:start -->
*02a: $f(x)=1/x$ by definition*
<!-- bilingual-en:end -->

固定 $x_0\ne0$。先写差商：
<!-- bilingual-en:start -->
Fix $x_0\ne0$ and begin with the difference quotient:
<!-- bilingual-en:end -->

$$
\begin{aligned}
\frac{f(x_0+h)-f(x_0)}h
&=\frac{\frac1{x_0+h}-\frac1{x_0}}h\\
&=\frac{x_0-(x_0+h)}{h\,x_0(x_0+h)}\\
&=-\frac1{x_0(x_0+h)}.
\end{aligned}
$$

这里第二行用共同分母 $x_0(x_0+h)$，第三行才约去 $h$。于是
<!-- bilingual-en:start -->
The second line uses the common denominator $x_0(x_0+h)$; only in the third line can the factor $h$ be cancelled. Therefore,
<!-- bilingual-en:end -->

$$
\boxed{\left(\frac1x\right)'=-\frac1{x^2}},\qquad x\ne0.
$$

合理性检查：$1/x$ 在定义域两支都随 $x$ 增大而下降，所以导数应为负；当 $|x|$ 很大，图像变平，$-1/x^2\to0$，也与图像一致。
<!-- bilingual-en:start -->
Sanity check: both branches of $1/x$ decrease as $x$ increases, so the derivative should be negative. As $|x|$ grows, the graph flattens and $-1/x^2\to0$, which is consistent with the geometry.
<!-- bilingual-en:end -->

### 02b：双曲线切线围成的三角形
<!-- bilingual-en:start -->
*02b: Triangle Enclosed by a Tangent to the Hyperbola*
<!-- bilingual-en:end -->

**题目。** $y=1/x$ 在第一象限任一点 $P=(x_0,1/x_0)$ 的切线，与两坐标轴围成的三角形面积是多少？
<!-- bilingual-en:start -->
**Problem.** For a point $P=(x_0,1/x_0)$ on $y=1/x$ in the first quadrant, what is the area of the triangle enclosed by the tangent line and the coordinate axes?
<!-- bilingual-en:end -->

导数给出切线斜率 $m=-1/x_0^2$，故
<!-- bilingual-en:start -->
The derivative gives the slope of tangent line $m=-1/x_0^2$, so
<!-- bilingual-en:end -->

$$
y-\frac1{x_0}=-\frac1{x_0^2}(x-x_0).
$$

求 $x$ 截距：令 $y=0$，
<!-- bilingual-en:start -->
To find the $x$-intercept, set $y=0$:
<!-- bilingual-en:end -->

$$
-\frac1{x_0}=-\frac{x-x_0}{x_0^2}
\quad\Longrightarrow\quad x=2x_0.
$$

求 $y$ 截距：令 $x=0$，
<!-- bilingual-en:start -->
To find the $y$-intercept, set $x=0$:
<!-- bilingual-en:end -->

$$
y-\frac1{x_0}=\frac1{x_0}
\quad\Longrightarrow\quad y=\frac2{x_0}.
$$

因此
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

$$
A=\frac12\cdot2x_0\cdot\frac2{x_0}=\boxed{2}.
$$

面积与切点无关。这里微积分只负责求斜率；坐标几何和代数负责余下步骤。变量 $x,y$ 在“曲线上点”“切线上任一点”“坐标轴截距”中扮演不同角色，必须由方程语境辨认。
<!-- bilingual-en:start -->
The area is independent of the point of tangency. Calculus supplies the slope; coordinate geometry and algebra supply the remaining steps. The symbols $x$ and $y$ play different roles for a point on the curve, an arbitrary point on the tangent, and an axis intercept, so their meaning must be read from the equation in context.
<!-- bilingual-en:end -->

### 02c：导数记号
<!-- bilingual-en:start -->
*02c: Derivative notation*
<!-- bilingual-en:end -->

若 $y=f(x)$，常见记号为
<!-- bilingual-en:start -->
If $y=f(x)$, the common notation is
<!-- bilingual-en:end -->

$$
f'(x),\quad y',\quad Df(x),\quad \frac{df}{dx},\quad \frac{dy}{dx},\quad \frac d{dx}f(x).
$$

$f'(x_0)$ 明确表示在 $x_0$ 的值；$dy/dx$ 强调“相对于谁求导”。Leibniz 记号形似分数，但定义上是一个整体运算符；以后链式法则中可以像分数一样帮助记忆，却不能不加条件地任意约分。
<!-- bilingual-en:start -->
$f'(x_0)$ explicitly denotes the value at $x_0$, while $dy/dx$ emphasizes the variable with respect to which the derivative is taken. Leibniz notation looks like a fraction, but by definition it is a single operator. Its fraction-like form can help you remember the chain rule, but it must not be cancelled algebraically without justification.
<!-- bilingual-en:end -->

### 02d：正整数幂法则的完整推导
<!-- bilingual-en:start -->
*02d: A Complete Proof of the Positive-Integer Power Rule*
<!-- bilingual-en:end -->

对正整数 $n$，从定义出发：
<!-- bilingual-en:start -->
For a positive integer $n$, start from the definition:
<!-- bilingual-en:end -->

$$
\frac{(x+h)^n-x^n}{h}.
$$

二项式定理给出
<!-- bilingual-en:start -->
The binomial theorem gives
<!-- bilingual-en:end -->

$$
(x+h)^n=x^n+nx^{n-1}h+\binom n2x^{n-2}h^2+\cdots+h^n.
$$

减去 $x^n$，每一项都有因子 $h$：
<!-- bilingual-en:start -->
After subtracting $x^n$, every term has a factor of $h$:
<!-- bilingual-en:end -->

$$
\frac{(x+h)^n-x^n}{h}
=nx^{n-1}+\binom n2x^{n-2}h+\cdots+h^{n-1}.
$$

当 $h\to0$，除第一项外其余项都至少含一个 $h$，所以趋于零：
<!-- bilingual-en:start -->
As $h\to0$, every term except the first still contains at least one factor of $h$ and therefore tends to zero:
<!-- bilingual-en:end -->

> [!important] 正整数幂法则
> $$
> \boxed{\frac d{dx}x^n=nx^{n-1}},\qquad n=1,2,3,\ldots
> $$

这也解释了课件中的 $O(h^2)$：它代表至少含 $h^2$ 的所有项；除以 $h$ 后成为 $O(h)$，取极限时消失。
<!-- bilingual-en:start -->
This also explains the $O(h^2)$ notation in the lecture notes: it collects terms of order at least $h^2$. After division by $h$, these become $O(h)$ and vanish in the limit.
<!-- bilingual-en:end -->

### 02e：线性近似乘积（超前材料）
<!-- bilingual-en:start -->
*02e: Products of Linear Approximations (Preview Material)*
<!-- bilingual-en:end -->

本地 `Ses02e` 的文件名归入 Session 2，但内容使用了稍后才正式证明的积法则，并预告 Unit 2 的线性近似。若
<!-- bilingual-en:start -->
The local `Ses02e` files are assigned to Session 2, but their content uses the product rule before its formal proof and previews the linear approximation developed in Unit 2. If
<!-- bilingual-en:end -->

$$
f(x)\approx f_0+f'_0\Delta x,\qquad
g(x)\approx g_0+g'_0\Delta x,
$$

则相乘得到
<!-- bilingual-en:start -->
then multiplying the approximations gives
<!-- bilingual-en:end -->

$$
f_0g_0+(f'_0g_0+f_0g'_0)\Delta x+f'_0g'_0(\Delta x)^2.
$$

忽略二阶小量 $(\Delta x)^2$，一次项系数正是积法则。这里应把它理解为直觉预告，不用它反过来证明本节幂法则。
<!-- bilingual-en:start -->
After neglecting the second-order term $(\Delta x)^2$, the coefficient of the linear term is exactly the expression in the product rule. This is only an intuitive preview; it should not be used circularly to prove the power rule in this section.
<!-- bilingual-en:end -->

### 边界情况与易错点
<!-- bilingual-en:start -->
*Boundary Cases and Common Pitfalls*
<!-- bilingual-en:end -->

- $1/x$ 的导数公式只在 $x\ne0$ 有意义；求导不能创造原函数没有的定义点。
- 二项式证明当前只覆盖正整数指数；负整数、有理数、实数指数将在 Session 10、13、18 逐步扩展。
- $O(h^2)$ 不是一个固定常数，而是“量级至多与 $h^2$ 相当”的项集合。
- 直线题要先写点斜式，再分别令 $x=0$、$y=0$ 求截距；不要把切点坐标误当截距。
<!-- bilingual-en:start -->
- The derivative formula for $1/x$ is meaningful only when $x\ne0$; differentiation cannot create domain points at which the function itself is undefined.
- The binomial proof currently covers only positive integer exponents; negative-integer, rational, and real exponents are developed in Sessions 10, 13, and 18.
- $O(h^2)$ is not a fixed constant. It denotes terms whose magnitude is bounded by a constant multiple of $h^2$ near $h=0$.
- In line problems, write the point-slope form first, then set $x=0$ and $y=0$ separately to find the intercepts; do not mistake the point of tangency for an intercept.
<!-- bilingual-en:end -->

### 三道自检
<!-- bilingual-en:start -->
*Three Quick Self-Checks*
<!-- bilingual-en:end -->

1. 用定义求 $f(x)=x^3$ 的导数，不直接引用幂法则。
2. $y=1/x$ 在 $x_0=-2$ 的切线是什么？
3. 为什么二项式展开中只有 $nx^{n-1}h$ 对极限留下贡献？
<!-- bilingual-en:start -->

&nbsp;
**1.** Compute the derivative of $f(x)=x^3$ from the definition, without invoking the power rule.<br>
**2.** What is the tangent line to $y=1/x$ at $x_0=-2$?<br>
**3.** Why is $nx^{n-1}h$ the only term in the binomial expansion that contributes to the limit?<br>
<!-- bilingual-en:end -->

> [!success]- 自检答案
> 1. $[(x+h)^3-x^3]/h=3x^2+3xh+h^2\to3x^2$。
> 2. 点 $(-2,-1/2)$，斜率 $-1/4$，故 $y+1/2=-\frac14(x+2)$，即 $y=-x/4-1$。
> 3. 减去 $x^n$ 并除以 $h$ 后，第一项不再含 $h$；其他项仍至少含一个 $h$，在 $h\to0$ 时趋零。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** $[(x+h)^3-x^3]/h=3x^2+3xh+h^2\to3x^2$.<br>
> **2.** Point $(-2,-1/2)$, slope $-1/4$, so $y+1/2=-\frac14(x+2)$, that is $y=-x/4-1$.<br>
> **3.** After subtracting $x^n$ and dividing by $h$, the first term no longer contains $h$; the other terms still contain at least one $h$ and tend to zero at $h\to0$.<br>
> <!-- bilingual-en:end -->

### 本地材料

- [[Ses02a_Lecture_Notes.pdf#page=1|02a $1/x$（pp.1-2）]]
- [[Ses02b_Lecture_Notes.pdf#page=1|02b A Harder Problem（pp.1-3）]]
- [[Ses02c_Lecture_Notes.pdf#page=1|02c Notations（p.1）]]
- [[Ses02d_MIT18_10SCF10_Ses2d.pdf#page=1|02d Positive Integer Power Rule（pp.1-2）]]
- [[Ses02e_lec9ses2ex1_linearprod.pdf#page=1|02e Product of Linear Approximations（p.1，超前材料）]]
- [[Exercise002_Problems.pdf#page=1|Exercise 002：$|x|$ 的导数]] · [[Exercise002_Solutions.pdf#page=1|答案]]

**知识链：**差商 → 代数消去 $0/0$ → 具体导数 → 二项式结构 → 正整数幂法则。
<!-- bilingual-en:start -->
**Knowledge chain:** difference quotient → algebraically remove the $0/0$ form → compute specific derivatives → identify the binomial pattern → positive-integer power rule.
<!-- bilingual-en:end -->

## Session 3：Derivative as Rate of Change

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题：**切线斜率为什么也能表示速度、电流、温度梯度和测量灵敏度？
<!-- bilingual-en:start -->
**Question:** Why can a tangent slope also represent velocity, electric current, a temperature gradient, or measurement sensitivity?
<!-- bilingual-en:end -->

**前置知识：**差商、导数定义、变量及单位、勾股定理。
<!-- bilingual-en:start -->
**Prerequisites:** difference quotients, the definition of the derivative, variables and units, and the Pythagorean theorem.
<!-- bilingual-en:end -->

### 03a-03b：平均变化率与瞬时变化率
<!-- bilingual-en:start -->
*03a-03b: Average and Instantaneous Rate of Change*
<!-- bilingual-en:end -->

若 $y=f(x)$，在 $x$ 到 $x+h$ 之间，
<!-- bilingual-en:start -->
If $y=f(x)$, between $x$ and $x+h$,
<!-- bilingual-en:end -->

$$
\frac{\Delta y}{\Delta x}=\frac{f(x+h)-f(x)}h
$$

是单位输入变化所对应的平均输出变化。令区间长度 $h\to0$，得到瞬时变化率
<!-- bilingual-en:start -->
is the average change in output per unit change in input. Letting the interval length $h\to0$ gives the instantaneous rate of change
<!-- bilingual-en:end -->

$$
\frac{dy}{dx}=f'(x).
$$

“斜率”与“变化率”是同一个比值的两种语言：在坐标图上看是 rise/run；把横轴解释为时间、距离或产量后，就成为物理或经济量。
<!-- bilingual-en:start -->
Slope and rate of change are two interpretations of the same ratio. On a graph it is rise over run; when the horizontal axis represents time, distance, or output, the ratio acquires a physical or economic meaning.
<!-- bilingual-en:end -->

### 03c：80 米南瓜下落
<!-- bilingual-en:start -->
*03c: An 80-Metre Pumpkin Drop*
<!-- bilingual-en:end -->

课件设南瓜从约 $80$ 米高处静止落下，忽略空气阻力：
<!-- bilingual-en:start -->
The lecture models a pumpkin dropped from rest at a height of about $80$ metres, neglecting air resistance:
<!-- bilingual-en:end -->

$$
h(t)=80-5t^2\quad(\text{m}).
$$

落地时间由 $h(t)=0$ 得
<!-- bilingual-en:start -->
The time of impact is found by solving $h(t)=0$:
<!-- bilingual-en:end -->

$$
80-5t^2=0\Longrightarrow t=4\text{ s}.
$$

全程平均速度为
<!-- bilingual-en:start -->
The average velocity over the entire fall is
<!-- bilingual-en:end -->

$$
\frac{h(4)-h(0)}{4-0}=\frac{0-80}{4}=-20\text{ m/s}.
$$

由幂法则，瞬时速度
<!-- bilingual-en:start -->
By the power rule, the instantaneous velocity is
<!-- bilingual-en:end -->

$$
v(t)=h'(t)=-10t,
$$

所以撞地前
<!-- bilingual-en:start -->
Therefore, immediately before impact,
<!-- bilingual-en:end -->

$$
v(4)=-40\text{ m/s}.
$$

负号表示向下；速率（speed）是速度大小 $|v|=40\text{ m/s}$。平均速度与终点瞬时速度不同，因为下落过程中速度一直在改变。
<!-- bilingual-en:start -->
The negative sign indicates downward motion; the speed is the magnitude $|v|=40\text{ m/s}$. The average velocity differs from the final instantaneous velocity because the velocity changes throughout the fall.
<!-- bilingual-en:end -->

### 单位检查
<!-- bilingual-en:start -->
*Unit Check*
<!-- bilingual-en:end -->

若 $h$ 用米、$t$ 用秒，则 $dh/dt$ 的单位是 m/s；再求导得到加速度
<!-- bilingual-en:start -->
If $h$ is measured in metres and $t$ in seconds, then $dh/dt$ has units of m/s; differentiating once more gives the acceleration
<!-- bilingual-en:end -->

$$
a(t)=v'(t)=h''(t)=-10\text{ m/s}^2.
$$

单位是答案的一部分：
<!-- bilingual-en:start -->
Units are part of the answer:
<!-- bilingual-en:end -->

- 电荷 $q$（库仑）对时间求导 $dq/dt$ 是电流（安培）；
- 温度 $T$ 对位置 $x$ 求导 $dT/dx$ 是温度梯度（度/米）；
- 成本 $C$ 对产量 $q$ 求导 $dC/dq$ 是边际成本（货币/件）。
<!-- bilingual-en:start -->
- If charge $q$ is measured in coulombs, then $dq/dt$ is current in amperes.
- The spatial derivative $dT/dx$ is a temperature gradient, measured in degrees per metre.
- The derivative $dC/dq$ is marginal cost, measured in currency units per unit of output.
<!-- bilingual-en:end -->

### 03d：GPS 灵敏度
<!-- bilingual-en:start -->
*03d: GPS Sensitivity*
<!-- bilingual-en:end -->

简化的平面模型中，卫星高度 $s$ 已知，接收机测得斜距 $h$，水平距离为 $L$：
<!-- bilingual-en:start -->
In the simplified planar model, the satellite altitude $s$ is known, the receiver measures the slant range $h$, and the horizontal distance is $L$:
<!-- bilingual-en:end -->

$$
h^2=s^2+L^2,\qquad L(h)=\sqrt{h^2-s^2}.
$$

对 $h$ 求导：

$$
\frac{dL}{dh}=\frac{h}{\sqrt{h^2-s^2}}=\frac hL.
$$
<!-- bilingual-en:start -->
Differentiate with respect to $h$:
<!-- bilingual-en:end -->

测距有小误差 $\Delta h$ 时，水平误差近似
<!-- bilingual-en:start -->
For a small ranging error $\Delta h$, the horizontal error is approximately
<!-- bilingual-en:end -->

$$
\Delta L\approx\frac{dL}{dh}\Delta h=\frac hL\Delta h.
$$

当接收机几乎在卫星正下方时 $L$ 很小，放大因子 $h/L$ 很大：很小的斜距误差也会造成明显的水平位置误差。这就是导数作为**灵敏度（sensitivity）**的含义。
<!-- bilingual-en:start -->
When the receiver is almost directly below the satellite, $L$ is small and the amplification factor $h/L$ is large: even a small error in slant range can produce a substantial horizontal-position error. This is the meaning of a derivative as **sensitivity**.
<!-- bilingual-en:end -->

> [!note] 近似的逻辑
> $\Delta L/\Delta h$ 是真实有限误差比；$dL/dh$ 是其在 $\Delta h\to0$ 的极限。写 $\Delta L\approx(dL/dh)\Delta h$ 需要误差足够小，并不意味着二者在任意步长下完全相等。

> <!-- bilingual-en:start -->
> $\Delta L/\Delta h$ is the true finite error ratio and $dL/dh$ is the limit of $\Delta h\to0$.  Writing $\Delta L\approx(dL/dh)\Delta h$ requires that the error is small enough, and does not mean that the two are exactly equal at any step.
> <!-- bilingual-en:end -->
### 边界情况与易错点
<!-- bilingual-en:start -->
*Boundary Cases and Common Pitfalls*
<!-- bilingual-en:end -->

- 位置、位移、路程不同。速度可为负，速率不能为负。
- 平均速度用净位移除以时间；总路程必须按改变方向的时刻分段后取绝对值。
- 求导前先说明自变量。$dT/dx$ 与 $dT/dt$ 回答不同问题。
- 灵敏度接近无穷不表示实际误差必定无穷，只说明线性放大因子很大、测量几何很不利。
<!-- bilingual-en:start -->
- Position, displacement, and distance are distinct. Velocity may be negative; speed cannot.
- Average velocity is net displacement divided by elapsed time. To find total distance, split the motion at every change of direction and add the absolute displacements.
- State the independent variable before differentiating: $dT/dx$ and $dT/dt$ answer different questions.
- A sensitivity that becomes arbitrarily large does not mean the actual error must be infinite; it means the linear amplification factor is large and the measurement geometry is poorly conditioned.
<!-- bilingual-en:end -->

### 三道自检
<!-- bilingual-en:start -->
*Three Quick Self-Checks*
<!-- bilingual-en:end -->

1. 对 $s(t)=3t^2-2t$，求 $[1,3]$ 的平均速度和 $t=3$ 的瞬时速度。
2. 南瓜落地时的速度为何是 $-40$ 而不是 $40$？何时应写 $40$？
3. GPS 模型中若 $s=3,h=5$，测距误差约 $0.01$，估计 $L$ 的误差。
<!-- bilingual-en:start -->

&nbsp;
**1.** For $s(t)=3t^2-2t$, find the average velocity on $[1,3]$ and the instantaneous velocity at $t=3$.<br>
**2.** Why is the pumpkin's velocity at impact $-40$ rather than $40$? When should the answer be $40$?<br>
**3.** In the GPS model, if $s=3$, $h=5$, and the ranging error is approximately $0.01$, estimate the error in $L$.<br>
<!-- bilingual-en:end -->

> [!success]- 自检答案
> 1. $[s(3)-s(1)]/2=(21-1)/2=10$；$s'(t)=6t-2$，故 $s'(3)=16$。
> 2. 选向上为正时下落方向为负，所以速度 $-40\text{ m/s}$；若问速率或速度大小，写 $40\text{ m/s}$。
> 3. $L=\sqrt{25-9}=4$，$dL/dh=5/4$，故 $|\Delta L|\approx(5/4)(0.01)=0.0125$。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** $[s(3)-s(1)]/2=(21-1)/2=10$; since $s'(t)=6t-2$, $s'(3)=16$.<br>
> **2.** If upward is positive, downward motion has negative velocity, so the velocity is $-40\text{ m/s}$. If the question asks for speed or the magnitude of velocity, report $40\text{ m/s}$.<br>
> **3.** $L=\sqrt{25-9}=4$ and $dL/dh=5/4$, so $|\Delta L|\approx(5/4)(0.01)=0.0125$.<br>
> <!-- bilingual-en:end -->

### 本地材料

- [[Ses03a_Lecture_Notes.pdf#page=1|03a Introduction to Rates of Change（p.1）]]
- [[Ses03b_Lecture_Notes.pdf#page=1|03b Rates of Change（p.1）]]
- [[Ses03c_Lecture_Notes.pdf#page=1|03c Pumpkin Drop（pp.1-2）]]
- [[Ses03d_Lecture_Notes.pdf#page=1|03d Temperature Gradient and GPS（pp.1-2）]]
- [[Exercise003_Problems.pdf#page=1|Exercise 003：Checking Account Balances]] · [[Exercise003_Solutions.pdf#page=1|答案]]

**知识链：**几何斜率 → 单位输出/单位输入 → 平均变化率 → 区间缩到一点 → 瞬时变化率与灵敏度。
<!-- bilingual-en:start -->
**Knowledge chain:** geometric slope → output units per input unit → average rate of change → shrink the interval to a point → instantaneous rate of change and sensitivity.
<!-- bilingual-en:end -->

## Session 4：Limits and Continuity

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题：**“趋近”究竟依赖函数在目标点的值吗？什么条件保证可以直接代入？
<!-- bilingual-en:start -->
**Question:** Does a limit depend on the function's value at the target point? Under what conditions is direct substitution valid?
<!-- bilingual-en:end -->

**前置知识：**函数图像、单侧趋近、代数化简。
<!-- bilingual-en:start -->
**Prerequisites:** graphs of functions, one-sided limits, and algebraic simplification.
<!-- bilingual-en:end -->

### 04a：极限、容易极限与困难极限
<!-- bilingual-en:start -->
*04a: Limits, Direct-Substitution Cases, and Indeterminate Cases*
<!-- bilingual-en:end -->

记号
<!-- bilingual-en:start -->
Notation
<!-- bilingual-en:end -->

$$
\lim_{x\to a}f(x)=L
$$

表示当 $x$ 取足够靠近但不等于 $a$ 的值时，$f(x)$ 可任意靠近 $L$。极限考察的是**附近行为**，所以 $f(a)$ 可以未定义，也可以与 $L$ 不同。
<!-- bilingual-en:start -->
The statement means that $f(x)$ can be made arbitrarily close to $L$ by taking $x$ sufficiently close to—but not equal to—$a$. A limit describes **nearby behavior**, so $f(a)$ may be undefined or may differ from $L$.
<!-- bilingual-en:end -->

例如
<!-- bilingual-en:start -->
For example
<!-- bilingual-en:end -->

$$
\lim_{x\to3}\frac{x^2+x}{x+1}
=\frac{9+3}{4}=3,
$$

因为分母在 $3$ 附近不为零，函数在此连续，可直接代入。相反，导数差商在 $h=0$ 总得到 $0/0$，必须先化简；涉及除零或无穷远的极限也通常不能直接代入。
<!-- bilingual-en:start -->
Because the denominator is nonzero near $3$, the function is continuous there and direct substitution is valid. By contrast, substituting $h=0$ into a derivative difference quotient always gives $0/0$, so the expression must first be simplified. Limits involving division by zero or behavior at infinity also generally require more analysis than direct substitution.
<!-- bilingual-en:end -->

### 左右极限
<!-- bilingual-en:start -->
*Left- and Right-Hand Limits*
<!-- bilingual-en:end -->

$$
\lim_{x\to a^-}f(x)=L_-,\qquad
\lim_{x\to a^+}f(x)=L_+.
$$

双侧极限存在且等于 $L$，当且仅当左右极限都存在并且
<!-- bilingual-en:start -->
The two-sided limit exists and is equal to $L$ if and only if both the left and right limits exist and
<!-- bilingual-en:end -->

$$
L_-=L_+=L.
$$

课件例子取
<!-- bilingual-en:start -->
The example in the slides uses
<!-- bilingual-en:end -->

$$
f(x)=
\begin{cases}
x+1,&x>0,\\
-x,&x\le0.
\end{cases}
$$

于是
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

$$
\lim_{x\to0^+}f(x)=1,
\qquad
\lim_{x\to0^-}f(x)=0.
$$

左右不同，所以 $\lim_{x\to0}f(x)$ 不存在；虽然 $f(0)=0$，它不能改变右侧附近的行为。
<!-- bilingual-en:start -->
The one-sided limits differ, so $\lim_{x\to0}f(x)$ does not exist. Although $f(0)=0$, that single value cannot change the function's behavior immediately to the right of $0$.
<!-- bilingual-en:end -->

### 04b：连续的三个条件
<!-- bilingual-en:start -->
*04b: The Three Conditions for Continuity*
<!-- bilingual-en:end -->

> [!important] 点连续
> $f$ 在 $x=a$ 具有[[极限与连续#极限与连续|连续性]]，当且仅当：
> 1. $f(a)$ 有定义；
> 2. $\lim_{x\to a}f(x)$ 存在；
> 3. $\lim_{x\to a}f(x)=f(a)$。
> <!-- bilingual-en:start -->
> $f$ is [[极限与连续#极限与连续|continuous]] at $x=a$ if and only if:
> **1.** $f(a)$ is defined;<br>
> **2.** $\lim_{x\to a}f(x)$ exists;<br>
> **3.** $\lim_{x\to a}f(x)=f(a)$.<br>
> <!-- bilingual-en:end -->

等价地说，左右极限和函数值三者相等。连续意味着小输入变化只造成小输出变化，不会突然跳跃；但这不等同于“可导”，因为曲线仍可能有尖角。
<!-- bilingual-en:start -->
Equivalently, the left-hand limit, the right-hand limit, and the function value must all agree. Continuity means that small changes in the input cause only small changes in the output, with no sudden jump. It is not the same as differentiability, however, because a continuous curve may still have a sharp corner.
<!-- bilingual-en:end -->

### 极限定律为何能用
<!-- bilingual-en:start -->
*Why the Limit Laws Work*
<!-- bilingual-en:end -->

若 $\lim f=L$、$\lim g=M$，则在相应极限存在时：
<!-- bilingual-en:start -->
If $\lim f=L$ and $\lim g=M$, then, whenever the displayed operation is defined,
<!-- bilingual-en:end -->

$$
\lim(f+g)=L+M,
\quad
\lim(fg)=LM,
\quad
\lim\frac fg=\frac LM\ (M\ne0).
$$

这些定律将在求导规则证明中拆分复杂差商。分母极限为零时，最后一条不能直接使用。
<!-- bilingual-en:start -->
These laws let us break complicated difference quotients into simpler pieces when proving differentiation rules. If the denominator tends to zero, the quotient law cannot be applied directly.
<!-- bilingual-en:end -->

### 边界情况与易错点
<!-- bilingual-en:start -->
*Boundary Cases and Common Pitfalls*
<!-- bilingual-en:end -->

- “极限不存在”和“极限为无穷”在严格意义上不同；后者描述一种确定的发散方式。
- 端点只要求定义域内部一侧的连续性；例如 $\sqrt{x}$ 在 $0$ 处讨论右连续。
- 直接代入是连续性的结果，不是极限定义本身。
- 图上空心点表示该点未取值；实心点表示函数值，二者不要混作极限。
<!-- bilingual-en:start -->
- “The limit does not exist” and “the function tends to infinity” are distinct statements; the latter describes a specific form of divergence.
- At an endpoint, continuity is required only from within the domain; for example, $\sqrt{x}$ is right-continuous at $0$.
- Direct substitution is a consequence of continuity, not the definition of a limit.
- An open circle marks a value that the graph does not attain at that point, whereas a filled point marks the actual function value; neither should be confused with the limit.
<!-- bilingual-en:end -->

### 三道自检
<!-- bilingual-en:start -->
*Three Quick Self-Checks*
<!-- bilingual-en:end -->

1. 计算 $\lim_{x\to2}(x^2-4)/(x-2)$，并说明为何不能一开始代入。
2. 构造一个 $f(0)=7$ 但 $\lim_{x\to0}f(x)=2$ 的函数。
3. 分段函数 $f(x)=x+a$（$x>1$），$f(x)=x^2$（$x\le1$）在 $1$ 连续时 $a$ 为何值？
<!-- bilingual-en:start -->

&nbsp;
**1.** Compute $\lim_{x\to2}(x^2-4)/(x-2)$ and explain why direct substitution cannot be used at the outset.<br>
**2.** Construct a function with $f(0)=7$ but $\lim_{x\to0}f(x)=2$.<br>
**3.** For the piecewise function $f(x)=x+a$ when $x>1$ and $f(x)=x^2$ when $x\le1$, what value of $a$ makes $f$ continuous at $1$?<br>
<!-- bilingual-en:end -->

> [!success]- 自检答案
> 1. 对 $x\ne2$，原式 $=x+2$，极限为 $4$；直接代入原式是 $0/0$，没有数值。
> 2. 例如 $f(x)=2$（$x\ne0$），$f(0)=7$。
> 3. 左侧及函数值为 $1$，右极限为 $1+a$，故 $a=0$。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** For $x\ne2$, the expression simplifies to $x+2$, so the limit is $4$. Direct substitution into the unsimplified expression gives the indeterminate form $0/0$, not a numerical value.<br>
> **2.** For example, let $f(x)=2$ for $x\ne0$ and define $f(0)=7$.<br>
> **3.** The left-hand limit and the function value are both $1$, while the right-hand limit is $1+a$; therefore $a=0$.<br>
> <!-- bilingual-en:end -->

### 本地材料

- [[Ses04a_Lecture_Notes.pdf#page=1|04a Limits（pp.1-2）]]
- [[Ses04b_Lecture_Notes.pdf#page=1|04b Continuity（p.1）]]
- [[Exercise004_Problems.pdf#page=1|Exercise 004：Continuous but not Smooth]] · [[Exercise004_Solutions.pdf#page=1|答案]]

**知识链：**附近行为 → 左右极限 → 双侧极限 → 极限等于函数值 → 连续。
<!-- bilingual-en:start -->
**Knowledge chain:** nearby behavior → one-sided limits → two-sided limit → equality between the limit and the function value → continuity.
<!-- bilingual-en:end -->

## Session 5：Discontinuity

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题：**连续性会以哪些方式失败？为什么可导一定连续，而连续未必可导？
<!-- bilingual-en:start -->
**Question:** In what ways can continuity fail? Why does differentiability imply continuity, while continuity does not imply differentiability?
<!-- bilingual-en:end -->

**前置知识：**左右极限、点连续定义、导数差商。
<!-- bilingual-en:start -->
**Prerequisites:** one-sided limits, continuity at a point, and the derivative as a difference-quotient limit.
<!-- bilingual-en:end -->

### 05a-05d：四类间断
<!-- bilingual-en:start -->
*05a-05d: Four types of discontinuities*
<!-- bilingual-en:end -->

[[极限与连续#极限与连续|间断]]按连续条件失败的方式分类：
<!-- bilingual-en:start -->
[[极限与连续#极限与连续|Discontinuities]] are classified by which condition for continuity fails:
<!-- bilingual-en:end -->

1. **跳跃间断（jump discontinuity）**：左右极限都存在但不相等。上一节分段函数在 $0$ 即为例子。
2. **可去间断（removable discontinuity）**：左右极限相等且有限，但函数未定义或函数值不等于极限。补上正确函数值即可连续。例如 $(x^2-1)/(x-1)$ 在 $x=1$ 的洞。
3. **无穷间断（infinite discontinuity）**：至少一个单侧极限为 $+\infty$ 或 $-\infty$。例如 $1/x$ 在 $0$；这里有竖直渐近线。
4. **振荡间断（oscillatory discontinuity）**：靠近目标点时无限振荡，没有单侧极限。例如 $\sin(1/x)$ 在 $0$。
<!-- bilingual-en:start -->

&nbsp;
**1.** **Jump discontinuity:** Both one-sided limits exist, but they are unequal. The piecewise function at $0$ in the preceding section is an example.<br>
**2.** **Removable discontinuity:** The one-sided limits agree and are finite, but the function is undefined or has a different value at the point. Defining the function to equal the limit repairs continuity. The hole in $(x^2-1)/(x-1)$ at $x=1$ is an example.<br>
**3.** **Infinite discontinuity:** At least one one-sided limit is $+\infty$ or $-\infty$. The function $1/x$ at $0$ is an example, with a vertical asymptote there.<br>
**4.** **Oscillatory discontinuity:** The function oscillates indefinitely near the target point, so a one-sided limit fails to exist. The function $\sin(1/x)$ at $0$ is an example.<br>
<!-- bilingual-en:end -->

课件还比较 $f(x)=1/x$ 与 $f'(x)=-1/x^2$：原函数为奇函数，导函数为偶函数；$f'(x)<0$ 准确记录两支图像都向右下降，而导函数形状无需像原函数。
<!-- bilingual-en:start -->
The lecture notes also compare $f(x)=1/x$ with $f'(x)=-1/x^2$: $f$ is odd, whereas $f'$ is even. The inequality $f'(x)<0$ correctly records that both branches decrease from left to right, but a derivative's graph need not resemble the original graph.
<!-- bilingual-en:end -->

### 05e：[[导数与求导规则#导数的信息边界|可导蕴含连续]]的逐步证明
<!-- bilingual-en:start -->
*05e: Step-by-step proof that [[导数与求导规则#导数的信息边界|differentiability implies continuity]]*
<!-- bilingual-en:end -->

> [!important] 定理
> 若 $f$ 在 $x_0$ 可导，则 $f$ 在 $x_0$ 连续。
> <!-- bilingual-en:start -->
> If $f$ is differentiable at $x_0$, then $f$ is continuous at $x_0$.
> <!-- bilingual-en:end -->

**目标。** 证明 $\lim_{x\to x_0}[f(x)-f(x_0)]=0$。
<!-- bilingual-en:start -->
**Goal.** Prove that $\lim_{x\to x_0}[f(x)-f(x_0)]=0$.
<!-- bilingual-en:end -->

**构造。** 对 $x\ne x_0$，把函数增量拆成“差商 × 输入增量”：
<!-- bilingual-en:start -->
**Construction.** For $x\ne x_0$, factor the function increment into “difference quotient × input increment”:
<!-- bilingual-en:end -->

$$
f(x)-f(x_0)
=\frac{f(x)-f(x_0)}{x-x_0}(x-x_0).
$$

**取极限。** 可导假设保证第一因子趋于有限数 $f'(x_0)$；第二因子趋于 $0$：
<!-- bilingual-en:start -->
**Take the limit.** Differentiability guarantees that the first factor tends to the finite value $f'(x_0)$, while the second factor tends to $0$:
<!-- bilingual-en:end -->

$$
\begin{aligned}
\lim_{x\to x_0}[f(x)-f(x_0)]
&=\left(\lim_{x\to x_0}\frac{f(x)-f(x_0)}{x-x_0}\right)
\left(\lim_{x\to x_0}(x-x_0)\right)\\
&=f'(x_0)\cdot0=0.
\end{aligned}
$$

因此 $\lim_{x\to x_0}f(x)=f(x_0)$，即连续。
<!-- bilingual-en:start -->
Therefore, $\lim_{x\to x_0}f(x)=f(x_0)$, so $f$ is continuous at $x_0$.
<!-- bilingual-en:end -->

**边界条件。** 证明依赖导数为有限数；若差商趋于无穷，就不能写成“有限数乘零”。又因为极限过程始终取 $x\ne x_0$，中间除以 $x-x_0$ 合法。
<!-- bilingual-en:start -->
**Boundary condition.** The proof relies on the derivative being finite. If the difference quotient diverges, it cannot be treated as a finite number multiplied by zero. Because the limiting process always has $x\ne x_0$, division by $x-x_0$ in the intermediate step is valid.
<!-- bilingual-en:end -->

### 逆命题为什么错：$|x|$
<!-- bilingual-en:start -->
*Why the Converse Fails: $|x|$*
<!-- bilingual-en:end -->

$f(x)=|x|$ 在 $0$ 连续，但
<!-- bilingual-en:start -->
$f(x)=|x|$ is continuous at $0$, but
<!-- bilingual-en:end -->

$$
\frac{|h|-|0|}{h}=\frac{|h|}{h}
=\begin{cases}1,&h>0,\\-1,&h<0.\end{cases}
$$

左右差商极限分别为 $1,-1$，所以不可导。这说明：
<!-- bilingual-en:start -->
The right- and left-hand limits of the difference quotient are $1$ and $-1$, respectively, so the function is not differentiable at $0$. Thus:
<!-- bilingual-en:end -->

$$
\text{可导}\Longrightarrow\text{连续},
\qquad
\text{连续}\centernot\Longrightarrow\text{可导}.
$$

### 边界情况与易错点
<!-- bilingual-en:start -->
*Boundary Cases and Common Pitfalls*
<!-- bilingual-en:end -->

- 可去间断处若重新定义函数值，可修复连续性；跳跃、无穷、振荡间断不能只改一个点修复。
- 分段函数要可导，先匹配函数值，再匹配左右导数；只匹配斜率不够。
- $\infty$ 不是普通实数，不能在代数式中任意做 $\infty-\infty$。
- 一条结论的逆命题必须单独证明；不能因“可导蕴含连续”就反向使用。
<!-- bilingual-en:start -->
- At a removable discontinuity, continuity can be repaired by redefining the function value. Jump, infinite, and oscillatory discontinuities cannot be repaired by changing a single point.
- For a piecewise function to be differentiable at a joining point, first match the function values and then match the one-sided derivatives; matching the slopes alone is not enough.
- $\infty$ is not an ordinary real number, so expressions such as $\infty-\infty$ cannot be manipulated algebraically without further analysis.
- The converse of a result must be proved separately; “differentiability implies continuity” cannot simply be used in reverse.
<!-- bilingual-en:end -->

### 三道自检
<!-- bilingual-en:start -->
*Three Quick Self-Checks*
<!-- bilingual-en:end -->

1. 分类 $f(x)=(x^2-4)/(x-2)$ 在 $x=2$ 的间断。
2. 分类 $1/(x-3)^2$ 在 $x=3$ 的间断，并给左右行为。
3. 证明若函数在一点不连续，则它在该点一定不可导。
<!-- bilingual-en:start -->

&nbsp;
**1.** Classify the discontinuity of $f(x)=(x^2-4)/(x-2)$ at $x=2$.<br>
**2.** Classify the discontinuity of $1/(x-3)^2$ at $x=3$ and describe its behavior from both sides.<br>
**3.** Prove that if a function is discontinuous at a point, then it cannot be differentiable there.<br>
<!-- bilingual-en:end -->

> [!success]- 自检答案
> 1. 对 $x\ne2$，$f=x+2$，极限为 $4$，但原式未定义，是可去间断。
> 2. 无穷间断；左右都趋于 $+\infty$。
> 3. 使用“可导蕴含连续”的逆否命题：若不连续，则不可能可导。这不是逆命题，而是逻辑等价的逆否命题。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** For $x\ne2$, $f=x+2$, so the limit is $4$; the original expression is undefined at $2$, making this a removable discontinuity.<br>
> **2.** This is an infinite discontinuity; both one-sided limits tend to $+\infty$.<br>
> **3.** Use the contrapositive of “differentiability implies continuity”: if the function is discontinuous, then it is not differentiable. This is the contrapositive, not the converse.<br>
> <!-- bilingual-en:end -->

### 本地材料

- [[Ses05a_Lecture_Notes.pdf#page=1|05a Jump Discontinuity（pp.1-2）]]
- [[Ses05b_Lecture_Notes.pdf#page=1|05b Removable Discontinuity（p.1）]]
- [[Ses05c_Lecture_Notes.pdf#page=1|05c Infinite Discontinuity（pp.1-2）]]
- [[Ses05d_Lecture_Notes.pdf#page=1|05d Oscillatory Discontinuity（p.1）]]
- [[Ses05e_Lecture_Notes.pdf#page=1|05e Differentiable Implies Continuous（p.1）]]
- [[Exercise005_Problems.pdf#page=1|Exercise 005：Limits and Discontinuity]] · [[Exercise005_Solutions.pdf#page=1|答案]]

**知识链：**连续条件的不同失败方式 → 间断分类 → 差商分解 → 可导必连续 → 用 $|x|$ 否定逆命题。
<!-- bilingual-en:start -->
**Knowledge chain:** different ways the continuity conditions can fail → classification of discontinuities → decomposition of the difference quotient → differentiability implies continuity → $|x|$ disproves the converse.
<!-- bilingual-en:end -->

## Session 6：Calculating Derivatives

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题：**怎样把已知简单导数组合成多项式等新函数的导数？
<!-- bilingual-en:start -->
**Question:** How can known elementary derivatives be combined to differentiate new functions such as polynomials?
<!-- bilingual-en:end -->

**前置知识：**导数定义、极限定律、幂法则、可导蕴含连续。
<!-- bilingual-en:start -->
**Prerequisites:** The definition of the derivative, limit laws, the power rule, and the fact that differentiability implies continuity.
<!-- bilingual-en:end -->

### 06a：两类公式
<!-- bilingual-en:start -->
*06a: Two Formulas*
<!-- bilingual-en:end -->

课件区分：
<!-- bilingual-en:start -->
The slides distinguish between:
<!-- bilingual-en:end -->

- **特定函数公式**：例如 $(x^n)'=nx^{n-1}$；
- **一般组合规则**：例如 $(u+v)'=u'+v'$、$(cu)'=cu'$。
<!-- bilingual-en:start -->
- **Specific formulas for functions:** for example, $(x^n)'=nx^{n-1}$;
- **General combination rules**: for example, $(u+v)'=u'+v'$, $(cu)'=cu'$.
<!-- bilingual-en:end -->

有了二者，就能把多项式逐项求导。
<!-- bilingual-en:start -->
Together, these formulas let us differentiate a polynomial term by term.
<!-- bilingual-en:end -->

### 常数、常数倍与和法则
<!-- bilingual-en:start -->
*Constants, Constant Multiples, and Sum Rules*
<!-- bilingual-en:end -->

常数函数 $f(x)=C$ 的差商恒为零：
<!-- bilingual-en:start -->
The difference quotient of the constant function $f(x)=C$ is always zero:
<!-- bilingual-en:end -->

$$
\frac{C-C}{h}=0\quad\Longrightarrow\quad C'=0.
$$

若 $c$ 为常数：
<!-- bilingual-en:start -->
If $c$ is a constant:
<!-- bilingual-en:end -->

$$
\begin{aligned}
(cu)'(x)
&=\lim_{h\to0}\frac{cu(x+h)-cu(x)}h\\
&=c\lim_{h\to0}\frac{u(x+h)-u(x)}h=cu'(x).
\end{aligned}
$$

### 06b：和法则的完整证明
<!-- bilingual-en:start -->
*06b: A Complete Proof of the Sum Rule*
<!-- bilingual-en:end -->

假设 $u,v$ 在 $x$ 可导。由定义：
<!-- bilingual-en:start -->
Suppose that $u$ and $v$ are differentiable at $x$. By definition:
<!-- bilingual-en:end -->

$$
\begin{aligned}
(u+v)'(x)
&=\lim_{h\to0}\frac{[u(x+h)+v(x+h)]-[u(x)+v(x)]}{h}\\
&=\lim_{h\to0}\left[
\frac{u(x+h)-u(x)}h+
\frac{v(x+h)-v(x)}h
\right].
\end{aligned}
$$

两个差商极限分别存在，所以可以使用极限的和法则：
<!-- bilingual-en:start -->
The two difference-quotient limits exist separately, so the sum law for limits applies:
<!-- bilingual-en:end -->

$$
\boxed{(u+v)'=u'+v'}.
$$

同理 $(u-v)'=u'-v'$。推广到有限多项之和，可以逐项求导。
<!-- bilingual-en:start -->
Similarly, $(u-v)'=u'-v'$. The rule extends to any finite sum, allowing a polynomial to be differentiated term by term.
<!-- bilingual-en:end -->

### 例：多项式
<!-- bilingual-en:start -->
*Example: Polynomial*
<!-- bilingual-en:end -->

$$
\begin{aligned}
\frac d{dx}(4x^5-3x^2+7x-9)
&=4(5x^4)-3(2x)+7(1)-0\\
&=20x^4-6x+7.
\end{aligned}
$$

每一项同时使用了常数倍法则与幂法则。
<!-- bilingual-en:start -->
Each term uses both the constant-times rule and the power rule.
<!-- bilingual-en:end -->

### 易错点与适用条件
<!-- bilingual-en:start -->
*Boundary Cases and Common Pitfalls*
<!-- bilingual-en:end -->

- $(u+v)'=u'+v'$，但后面将看到 $(uv)'\ne u'v'$。
- “常数”指相对于当前自变量不变。例如对 $x$ 求导时 $a$ 可为常数；若 $a=a(x)$ 就不能用常数倍法则。
- 极限和法则要求各极限存在；不能把两个各自发散的量随意拆开后相消。
- 幂法则对不同指数的适用范围要按课程进度区分，不要提前把未证明范围当作已证。
<!-- bilingual-en:start -->
- $(u+v)'=u'+v'$, but you will see $(uv)'\ne u'v'$ later.
- “Constant” means constant with respect to the current independent variable. For example, $a$ may be treated as constant when differentiating with respect to $x$; if $a=a(x)$, the constant-multiple rule does not apply.
- The sum law for limits requires the component limits to exist; two divergent expressions cannot be split apart and cancelled without justification.
- The domain and proof of the power rule depend on the type of exponent. Do not treat cases not yet established in the course as already proved.
<!-- bilingual-en:end -->

### 三道自检
<!-- bilingual-en:start -->
*Three Quick Self-Checks*
<!-- bilingual-en:end -->

1. 由定义证明 $(u-v)'=u'-v'$。
2. 求 $D(2x^7-\pi x^2+e^2)$；此处 $e^2$ 是什么？
3. 若 $u'(2)=3,v'(2)=-5$，求 $(4u-2v)'(2)$。
<!-- bilingual-en:start -->

&nbsp;
**1.** Prove $(u-v)'=u'-v'$ from the definition.<br>
**2.** Find $D(2x^7-\pi x^2+e^2)$; what role does $e^2$ play here?<br>
**3.** If $u'(2)=3$ and $v'(2)=-5$, find $(4u-2v)'(2)$.<br>
<!-- bilingual-en:end -->

> [!success]- 自检答案
> 1. 将差商拆为 $[u(x+h)-u(x)]/h-[v(x+h)-v(x)]/h$，分别取极限。
> 2. $14x^6-2\pi x$；$e^2$ 是常数，导数为零。
> 3. $4u'(2)-2v'(2)=12+10=22$。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** The difference quotient is split into $[u(x+h)-u(x)]/h-[v(x+h)-v(x)]/h$ and the limits are taken respectively.<br>
> **2.** $14x^6-2\pi x$; $e^2$ is a constant and the derivative is zero.<br>
> **3.** $4u'(2)-2v'(2)=12+10=22$.<br>
> <!-- bilingual-en:end -->

### 本地材料

- [[Ses06a_Lecture_Notes.pdf#page=1|06a Introduction to Differentiation（p.1）]]
- [[Ses06b_Lecture_Notes.pdf#page=1|06b Derivative of a Sum（p.1）]]
- 本地资料库没有 `Exercise006`；本节自检题用于补足练习入口。
<!-- bilingual-en:start -->
- `Exercise006` is not available in the local archive, so the self-checks in this section provide the missing practice component.
<!-- bilingual-en:end -->

**知识链：** 特定导数公式 + 极限线性 → 常数、常数倍、和差法则 → 多项式逐项求导。
<!-- bilingual-en:start -->
**Knowledge chain:** known derivative formulas plus linearity of limits → constant, constant-multiple, and sum/difference rules → term-by-term differentiation of polynomials.
<!-- bilingual-en:end -->

## Session 7：Derivatives of Sine and Cosine

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题：**如何只用导数定义和三角加法公式推导 $\sin x$、$\cos x$ 的导数？
<!-- bilingual-en:start -->
**Question:** How can the derivatives of $\sin x$ and $\cos x$ be derived using only the definition of the derivative and the angle-addition identities?
<!-- bilingual-en:end -->

**前置知识：**差商、极限线性、
<!-- bilingual-en:start -->
**Prerequisites:** difference quotients and linearity of limits.
<!-- bilingual-en:end -->

$$
\sin(a+b)=\sin a\cos b+\cos a\sin b,
$$

$$
\cos(a+b)=\cos a\cos b-\sin a\sin b.
$$

本节暂时把两个基本极限当作已知；Session 8 再证明它们：
<!-- bilingual-en:start -->
For now, this section takes two fundamental limits as given; Session 8 proves them independently:
<!-- bilingual-en:end -->

$$
\lim_{h\to0}\frac{\sin h}{h}=1,
\qquad
\lim_{h\to0}\frac{\cos h-1}{h}=0.
$$

### 07a：正弦导数的代数推导
<!-- bilingual-en:start -->
*07a: An Algebraic Proof of the Derivative of Sine*
<!-- bilingual-en:end -->

从定义开始：
<!-- bilingual-en:start -->
Start from the definition:
<!-- bilingual-en:end -->

$$
\begin{aligned}
\frac d{dx}\sin x
&=\lim_{h\to0}\frac{\sin(x+h)-\sin x}{h}\\
&=\lim_{h\to0}
\frac{\sin x\cos h+\cos x\sin h-\sin x}{h}.
\end{aligned}
$$

把含 $\sin x$ 和 $\cos x$ 的部分分开：
<!-- bilingual-en:start -->
Separate the terms containing $\sin x$ and $\cos x$:
<!-- bilingual-en:end -->

$$
\begin{aligned}
&=\lim_{h\to0}\left[
\sin x\frac{\cos h-1}{h}
+\cos x\frac{\sin h}{h}
\right]\\
&=\sin x\cdot0+\cos x\cdot1\\
&=\boxed{\cos x}.
\end{aligned}
$$

每一步的依据依次是：角和公式、代数分组、极限线性、两个基本极限。
<!-- bilingual-en:start -->
The steps use, in order, the angle-addition identity, algebraic grouping, linearity of limits, and the two fundamental limits.
<!-- bilingual-en:end -->

### 07b：余弦导数的代数推导
<!-- bilingual-en:start -->
*07b: An Algebraic Proof of the Derivative of Cosine*
<!-- bilingual-en:end -->

$$
\begin{aligned}
\frac d{dx}\cos x
&=\lim_{h\to0}\frac{\cos(x+h)-\cos x}{h}\\
&=\lim_{h\to0}\left[
\cos x\frac{\cos h-1}{h}
-\sin x\frac{\sin h}{h}
\right]\\
&=\cos x\cdot0-\sin x\cdot1\\
&=\boxed{-\sin x}.
\end{aligned}
$$

负号有清楚的图像意义：在 $x=\pi/2$ 附近，余弦穿过零且向下，所以斜率应为 $-1$。
<!-- bilingual-en:start -->
The minus sign has a clear graphical meaning: near $x=\pi/2$, the cosine curve crosses zero while decreasing, so its slope should be $-1$.
<!-- bilingual-en:end -->

### 为什么必须用弧度
<!-- bilingual-en:start -->
*Why Must We Use Radians?*
<!-- bilingual-en:end -->

若角度以度为单位，令 $x_{\rm rad}=\pi x_{\rm deg}/180$，则链式法则会给
<!-- bilingual-en:start -->
If the angle is in degrees and $x_{\rm rad}=\pi x_{\rm deg}/180$, then the chain rule gives
<!-- bilingual-en:end -->

$$
\frac d{dx_{\rm deg}}\sin(x_{\rm deg})
=\frac\pi{180}\cos(x_{\rm deg}).
$$

只有弧度制使单位圆弧长等于角度数值，从而 $\lim_{h\to0}\sin h/h=1$，也只有此时导数公式具有最简形式。
<!-- bilingual-en:start -->
Only in radians does an angle's numerical value equal the corresponding arc length on the unit circle. This makes $\lim_{h\to0}\sin h/h=1$ and gives the derivative formula its simplest form.
<!-- bilingual-en:end -->

### 图像检查
<!-- bilingual-en:start -->
*Graphical Check*
<!-- bilingual-en:end -->

- $\sin x$ 在 $x=\pi/2+k\pi$ 处切线水平，对应 $\cos x=0$。
- $\sin x$ 在 $x=2k\pi$ 处上升最快，导数为 $1$；在 $x=(2k+1)\pi$ 处下降最快，导数为 $-1$。
- $\cos x$ 是偶函数，导数 $-\sin x$ 是奇函数，与“偶函数导数为奇函数”一致。
<!-- bilingual-en:start -->
- $\sin x$ is tangent horizontally at $x=\pi/2+k\pi$, corresponding to $\cos x=0$.
- $\sin x$ rises fastest at $x=2k\pi$, where its derivative is $1$, and falls fastest at $x=(2k+1)\pi$, where its derivative is $-1$.
- $\cos x$ is even and its derivative $-\sin x$ is odd, as expected for the derivative of an even function.
<!-- bilingual-en:end -->

### 边界情况与易错点
<!-- bilingual-en:start -->
*Boundary Cases and Common Pitfalls*
<!-- bilingual-en:end -->

- 不可循环论证：本节推导依赖两个基本极限，Session 8 必须独立证明它们。
- $\cos h-1$ 与 $1-\cos h$ 互为相反数；极限都为零，但中间符号不能漏。
- $\sin^2x$ 表示 $(\sin x)^2$，其导数不是 $\cos^2x$；要等链式法则后算作 $2\sin x\cos x$。
- 角度单位若不是弧度，必须额外乘换算因子。
<!-- bilingual-en:start -->
- Avoid circular reasoning: the derivation here depends on two fundamental limits, which Session 8 must prove independently.
- $\cos h-1$ and $1-\cos h$ are negatives of one another. Their limits are both zero, but the sign matters in the intermediate algebra.
- $\sin^2x$ means $(\sin x)^2$, whose derivative is not $\cos^2x$; once the chain rule is available, its derivative is $2\sin x\cos x$.
- If angles are not measured in radians, an additional conversion factor is required.
<!-- bilingual-en:end -->

### 三道自检
<!-- bilingual-en:start -->
*Three Quick Self-Checks*
<!-- bilingual-en:end -->

1. 只用余弦加法公式和两个基本极限，重写 $(\cos x)'$ 的每一步。
2. 求 $\sin x$ 在 $x=\pi$ 的切线。
3. 不画图，说明 $\sin x$ 在 $x=3\pi/2$ 附近为何切线水平且是局部最低点。
<!-- bilingual-en:start -->

&nbsp;
**1.** Rewrite each step of $(\cos x)'$ using only the cosine addition formula and two fundamental limits.<br>
**2.** Find the tangent line to $\sin x$ at $x=\pi$.<br>
**3.** Without drawing a graph, explain why $\sin x$ has a horizontal tangent and a local minimum at $x=3\pi/2$.<br>
<!-- bilingual-en:end -->

> [!success]- 自检答案
> 1. 正文 07b 的四行；关键分组是 $\cos x(\cos h-1)-\sin x\sin h$。
> 2. 点 $(\pi,0)$，斜率 $\cos\pi=-1$，切线 $y=-(x-\pi)$。
> 3. 导数 $\cos(3\pi/2)=0$；其左右 $\cos x$ 从负变正，所以函数先降后升。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** Follow the four displayed lines in Section 07b; the key grouping is $\cos x(\cos h-1)-\sin x\sin h$.<br>
> **2.** The point is $(\pi,0)$ and the slope is $\cos\pi=-1$, so the tangent line is $y=-(x-\pi)$.<br>
> **3.** We have $\cos(3\pi/2)=0$, and $\cos x$ changes from negative to positive across that point, so the function decreases and then increases.<br>
> <!-- bilingual-en:end -->

### 本地材料

- [[Ses07a_Lecture_Notes.pdf#page=1|07a Derivative of Sine（p.1）]]
- [[Ses07b_Lecture_Notes.pdf#page=1|07b Derivative of Cosine（p.1）]]
- [[Exercise007_Problems.pdf#page=1|Exercise 007：Derivatives of Sine and Cosine]] · [[Exercise007_Solutions.pdf#page=1|答案]]

**知识链：**三角加法公式 → 差商拆成两个基本极限 → 正弦、余弦导数。
<!-- bilingual-en:start -->
**Knowledge chain:** trigonometric addition identities → split the difference quotient into two fundamental limits → derivatives of sine and cosine.
<!-- bilingual-en:end -->

## Session 8：Limits of Sine and Cosine

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题：**上一节使用的两个三角极限为何成立？其几何本质是什么？
<!-- bilingual-en:start -->
**Question:** Why do the two trigonometric limits used in the previous section hold, and what is their geometric basis?
<!-- bilingual-en:end -->

**前置知识：** 单位圆、弧度、三角形面积、夹逼定理、共轭式。
<!-- bilingual-en:start -->
**Prerequisites:** the unit circle, radians, triangle area, the squeeze theorem, and conjugate expressions.
<!-- bilingual-en:end -->

### 08a：$\lim_{\theta\to0}\sin\theta/\theta=1$

课件先给出直觉：单位圆中，弦的竖直投影长为 $\sin\theta$，弧长为 $\theta$；小角度时弧与弦越来越接近。为了使流程可检查，写成[[极限与连续#极限存在的检查顺序|三角极限夹逼证明]]。
<!-- bilingual-en:start -->
The lecture notes begin with the geometric intuition: in the unit circle, the chord's vertical projection has length $\sin\theta$, while the corresponding arc has length $\theta$. For small angles, the chord and arc become increasingly close. The rigorous argument is a [[极限与连续#极限存在的检查顺序|squeeze-theorem proof of the trigonometric limit]].
<!-- bilingual-en:end -->

对 $0<\theta<\pi/2$，单位圆内三角形、扇形、外切三角形面积满足
<!-- bilingual-en:start -->
For $0<\theta<\pi/2$, the areas of the inscribed triangle, circular sector, and circumscribed triangle satisfy
<!-- bilingual-en:end -->

$$
\frac12\sin\theta\cos\theta
<\frac12\theta
<\frac12\tan\theta.
$$

左边是圆内接直角三角形面积，中央是扇形面积，右边是外切三角形面积。分别整理两侧不等式：
<!-- bilingual-en:start -->
The left expression is the area of the inscribed right triangle, the middle expression is the area of the sector, and the right expression is the area of the circumscribed triangle. Rearranging the two inequalities gives
<!-- bilingual-en:end -->

$$
\sin\theta\cos\theta<\theta
\quad\Longrightarrow\quad
\frac{\sin\theta}{\theta}<\frac1{\cos\theta},
$$

以及
<!-- bilingual-en:start -->
and
<!-- bilingual-en:end -->

$$
\theta<\tan\theta=\frac{\sin\theta}{\cos\theta}
\quad\Longrightarrow\quad
\cos\theta<\frac{\sin\theta}{\theta}.
$$

合并得到
<!-- bilingual-en:start -->
Combining them gives
<!-- bilingual-en:end -->

$$
\cos\theta<\frac{\sin\theta}{\theta}<\frac1{\cos\theta}.
$$

当 $\theta\to0^+$，两端 $\cos\theta$ 与 $1/\cos\theta$ 都趋于 $1$，夹逼定理给
<!-- bilingual-en:start -->
As $\theta\to0^+$, both $\cos\theta$ and $1/\cos\theta$ tend to $1$, so the squeeze theorem gives
<!-- bilingual-en:end -->

$$
\lim_{\theta\to0^+}\frac{\sin\theta}{\theta}=1.
$$

又因 $\sin(-\theta)/(-\theta)=\sin\theta/\theta$，该比值是偶函数，左极限相同：
<!-- bilingual-en:start -->
Moreover, $\sin(-\theta)/(-\theta)=\sin\theta/\theta$, so the ratio is even and the left-hand limit is the same:
<!-- bilingual-en:end -->

$$
\boxed{\lim_{\theta\to0}\frac{\sin\theta}{\theta}=1}.
$$

![[98_attachment/MIT18.01SC/unit01-trig-squeeze.png|900]]

### 08b-08c：$\lim_{\theta\to0}(1-\cos\theta)/\theta=0$

课件用“圆弧与弦之间的水平缝隙比弧长缩得更快”解释。代数上可以由第一个极限严格推出：
<!-- bilingual-en:start -->
Geometrically, the horizontal gap between the arc and the chord shrinks faster than the arc length. Algebraically, the desired result follows rigorously from the first limit:
<!-- bilingual-en:end -->

$$
\begin{aligned}
\frac{1-\cos\theta}{\theta}
&=\frac{(1-\cos\theta)(1+\cos\theta)}{\theta(1+\cos\theta)}\\
&=\frac{\sin^2\theta}{\theta(1+\cos\theta)}\\
&=\left(\frac{\sin\theta}{\theta}\right)
\left(\frac{\sin\theta}{1+\cos\theta}\right).
\end{aligned}
$$

第一因子趋于 $1$，第二因子趋于 $0/2=0$，所以
<!-- bilingual-en:start -->
The first factor tends to $1$, while the second tends to $0/2=0$, so
<!-- bilingual-en:end -->

$$
\boxed{\lim_{\theta\to0}\frac{1-\cos\theta}{\theta}=0}.
$$

相应地 $(\cos\theta-1)/\theta$ 也趋于 $0$。这里分子和分母都趋零，但分子是二阶小量：由半角公式 $1-\cos\theta=2\sin^2(\theta/2)$ 可看出其数量级约为 $\theta^2/2$。
<!-- bilingual-en:start -->
Consequently, $(\cos\theta-1)/\theta$ also tends to $0$. Both numerator and denominator tend to zero, but the numerator is second order: the half-angle identity $1-\cos\theta=2\sin^2(\theta/2)$ shows that its leading size is approximately $\theta^2/2$.
<!-- bilingual-en:end -->

### 08d：正弦导数的几何图像
<!-- bilingual-en:start -->
*08d: Geometric Interpretation of the Derivative of Sine*
<!-- bilingual-en:end -->

单位圆上点 $P$ 的角为 $\theta$，邻点 $Q$ 的角为 $\theta+\Delta\theta$。小弧 $PQ$ 的长度为 $\Delta\theta$，而其竖直变化为
<!-- bilingual-en:start -->
Let $P$ have angle $\theta$ on the unit circle and let a nearby point $Q$ have angle $\theta+\Delta\theta$. The short arc $PQ$ has length $\Delta\theta$, and the vertical change is
<!-- bilingual-en:end -->

$$
\Delta y=\sin(\theta+\Delta\theta)-\sin\theta.
$$

当 $\Delta\theta$ 很小时，弦方向接近切线方向；圆的切线垂直于半径，因此切线与竖直方向的夹角对应 $\theta$，竖直分量与弧长之比趋于 $\cos\theta$：
<!-- bilingual-en:start -->
When $\Delta\theta$ is small, the chord direction approaches the tangent direction. Since the tangent to a circle is perpendicular to the radius, its vertical component relative to arc length tends to $\cos\theta$:
<!-- bilingual-en:end -->

$$
\frac{\Delta y}{\Delta\theta}\to\cos\theta.
$$

这是几何直观；严格代数证明仍以 07a 的角和公式与刚证明的极限为准。
<!-- bilingual-en:start -->
This supplies the geometric intuition. The rigorous algebraic proof remains the argument in Section 07a using the angle-addition identity and the limits just established.
<!-- bilingual-en:end -->

### sinc 函数
<!-- bilingual-en:start -->
*The sinc Function*
<!-- bilingual-en:end -->

$$
\operatorname{sinc}(x)=\frac{\sin x}{x},\qquad x\ne0.
$$

极限告诉我们在 $x=0$ 只差一个可去间断。定义 $\operatorname{sinc}(0)=1$ 后连续。它是偶函数，在 $x=k\pi$（非零整数 $k$）取零，振幅受 $1/|x|$ 包络并逐渐衰减；这正是信号处理中常见的振荡形状。
<!-- bilingual-en:start -->
The limit shows that the only defect at $x=0$ is a removable discontinuity. Defining $\operatorname{sinc}(0)=1$ makes the function continuous. It is even, vanishes at $x=k\pi$ for nonzero integers $k$, and has a decaying envelope $1/|x|$—a familiar oscillatory shape in signal processing.
<!-- bilingual-en:end -->

### 边界情况与易错点
<!-- bilingual-en:start -->
*Boundary Cases and Common Pitfalls*
<!-- bilingual-en:end -->

- 面积不等式先只对 $0<\theta<\pi/2$ 写；左侧由偶性补齐。
- 取倒数时必须确认各量为正，并反转不等号。
- $\sin\theta\sim\theta$ 是极限等价，不是对非零 $\theta$ 的恒等式。
- 本证明依赖弧度；角度制下单位圆弧长不等于角度数值。
<!-- bilingual-en:start -->
- The area inequalities are first established for $0<\theta<\pi/2$; evenness supplies the limit from the left.
- Before taking reciprocals, verify that the quantities are positive and reverse the inequality direction.
- $\sin\theta\sim\theta$ is an asymptotic equivalence, not an identity for nonzero $\theta$.
- The proof depends on radian measure; in degrees, an angle's numerical value is not the corresponding arc length on the unit circle.
<!-- bilingual-en:end -->

### 三道自检
<!-- bilingual-en:start -->
*Three Quick Self-Checks*
<!-- bilingual-en:end -->

1. 求 $\lim_{x\to0}\sin(5x)/x$。
2. 求 $\lim_{x\to0}(1-\cos x)/x^2$，可用半角公式。
3. sinc 补定义后为何在 $0$ 连续，但原始分式在 $0$ 不可谈导数？
<!-- bilingual-en:start -->

&nbsp;
**1.** Find $\lim_{x\to0}\sin(5x)/x$.<br>
**2.** Find $\lim_{x\to0}(1-\cos x)/x^2$ using a half-angle identity if helpful.<br>
**3.** Why does the continuously extended sinc function become continuous at $0$, while differentiability at $0$ is not even defined for the original quotient?<br>
<!-- bilingual-en:end -->

> [!success]- 自检答案
> 1. $5\,[\sin(5x)/(5x)]\to5$。
> 2. $2\sin^2(x/2)/x^2=\frac12[\sin(x/2)/(x/2)]^2\to1/2$。
> 3. 原始分式在 $0$ 未定义；先定义 sinc$(0)=1$ 才得到新函数。连续性由极限等于补入值保证，之后才可进一步研究导数。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** $5\,[\sin(5x)/(5x)]\to5$.<br>
> **2.** $2\sin^2(x/2)/x^2=\frac12[\sin(x/2)/(x/2)]^2\to1/2$.<br>
> **3.** The original quotient is undefined at $0$. Defining $\operatorname{sinc}(0)=1$ creates a new extended function; the equality between the limit and the inserted value makes it continuous, after which differentiability can be investigated.<br>
> <!-- bilingual-en:end -->

### 本地材料

- [[Ses08a_Lecture_Notes.pdf#page=1|08a $\sin x/x$（pp.1-2）]]
- [[Ses08b_Lecture_Notes.pdf#page=1|08b $(1-\cos x)/x$（pp.1-2）]]
- [[Ses08c_Lecture_Notes.pdf#page=1|08c Questions and Answers（pp.1-2）]]
- [[Ses08d_Lecture_Notes.pdf#page=1|08d Geometric Proof of $(\sin x)'$（pp.1-2）]]
- [[Exercise008_Problems.pdf#page=1|Exercise 008：The Function sinc]] · [[Exercise008_Solutions.pdf#page=1|答案]]

**知识链：**单位圆面积比较 → 夹逼基本极限 → 共轭式推出余弦极限 → 补全三角导数证明。
<!-- bilingual-en:start -->
**Knowledge chain:** compare areas in the unit circle → obtain the fundamental limit by the squeeze theorem → derive the cosine limit using conjugates → complete the proof of the trigonometric derivative formulas.
<!-- bilingual-en:end -->

## Session 9：Product Rule

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题：**两个都在变化的量相乘，乘积的瞬时变化为何是两项之和？
<!-- bilingual-en:start -->
**Question:** When two varying quantities are multiplied, why does the product's instantaneous rate of change contain two terms?
<!-- bilingual-en:end -->

**前置知识：**导数定义、可导蕴含连续、极限乘法与加法法则。
<!-- bilingual-en:start -->
**Prerequisites:** The definition of the derivative, the fact that differentiability implies continuity, and the product and sum laws for limits.
<!-- bilingual-en:end -->

### 09a-09b：规则与直觉
<!-- bilingual-en:start -->
*09a-09b: Rules and Intuition*
<!-- bilingual-en:end -->

若矩形边长为 $u,v$，小变化为 $\Delta u,\Delta v$，面积增量为
<!-- bilingual-en:start -->
If a rectangle has side lengths $u$ and $v$, and these change by $\Delta u$ and $\Delta v$, then its area changes by
<!-- bilingual-en:end -->

$$
(u+\Delta u)(v+\Delta v)-uv
=v\Delta u+u\Delta v+\Delta u\Delta v.
$$

除以输入变化后，最后一项是两个小量的乘积，取极限时消失；保留下来的正是“一次只改变一个因子”的两部分。
<!-- bilingual-en:start -->
After division by the input change, the final term contains a product of two small increments and vanishes in the limit. The two remaining terms correspond to changing one factor at a time.
<!-- bilingual-en:end -->

> [!important] 积法则
> 若 $u,v$ 在 $x$ 可导，则
> $$
> \boxed{(uv)'=u'v+uv'}.
> $$
> <!-- bilingual-en:start -->
> If $u$ and $v$ are differentiable at $x$, then
> <!-- bilingual-en:end -->

### 09c：由定义证明
<!-- bilingual-en:start -->
*09c: Proof from the Definition*
<!-- bilingual-en:end -->

从积的差商开始，并加减中间项 $u(x+h)v(x)$：
<!-- bilingual-en:start -->
Start from the difference quotient of the product, then add and subtract the intermediate term $u(x+h)v(x)$:
<!-- bilingual-en:end -->

$$
\begin{aligned}
(uv)'(x)
&=\lim_{h\to0}
\frac{u(x+h)v(x+h)-u(x)v(x)}h\\
&=\lim_{h\to0}
\frac{u(x+h)v(x+h)-u(x+h)v(x)}h\\
&\qquad+\lim_{h\to0}
\frac{u(x+h)v(x)-u(x)v(x)}h\\
&=\lim_{h\to0}u(x+h)\frac{v(x+h)-v(x)}h\\
&\qquad+\lim_{h\to0}v(x)\frac{u(x+h)-u(x)}h.
\end{aligned}
$$

由于 $u$ 可导，所以连续，$u(x+h)\to u(x)$。两个差商分别趋于 $v'(x),u'(x)$，故
<!-- bilingual-en:start -->
Because $u$ is differentiable, it is continuous, so $u(x+h)\to u(x)$. The two difference quotients tend to $v'(x)$ and $u'(x)$, respectively.
<!-- bilingual-en:end -->

$$
(uv)'(x)=u(x)v'(x)+v(x)u'(x).
$$

这一步明确使用了“可导蕴含连续”。若不知道 $u(x+h)\to u(x)$，第一项不能直接替换。
<!-- bilingual-en:start -->
This step explicitly uses the fact that differentiability implies continuity. Without $u(x+h)\to u(x)$, the first factor cannot be replaced by its limit directly.
<!-- bilingual-en:end -->

### 代表例题
<!-- bilingual-en:start -->
*Representative Example*
<!-- bilingual-en:end -->

$$
\frac d{dx}(x^n\sin x)
=nx^{n-1}\sin x+x^n\cos x.
$$

三个因子时重复应用：
<!-- bilingual-en:start -->
For three factors, apply the product rule repeatedly:
<!-- bilingual-en:end -->

$$
(uvw)'=u'vw+uv'w+uvw'.
$$

一般而言，有限多个因子求导就是“每一项只对一个因子求导，其余保持原样，再相加”。
<!-- bilingual-en:start -->
In general, for a finite product, differentiate one factor at a time, leave all the others unchanged, and add the resulting terms.
<!-- bilingual-en:end -->

### Exercise 009：拼接多项式
<!-- bilingual-en:start -->
*Exercise 009: Stitched Polynomials*
<!-- bilingual-en:end -->

分段函数在连接点可导必须满足：函数值连续 + 左右导数相等。对
<!-- bilingual-en:start -->
For a piecewise function to be differentiable at the join, the function values must agree and the one-sided derivatives must also agree.
<!-- bilingual-en:end -->

$$
f(x)=
\begin{cases}
ax^2+bx+6,&x\le0,\\
2x^5+3x^4+4x^2+5x+6,&x>0,
\end{cases}
$$

在 $0$ 两侧函数值自动都是 $6$；左导数 $b$，右导数 $5$，故 $b=5$，而 $a$ 任意。这个练习虽归在 Product Rule Session，实质复习了可导拼接。
<!-- bilingual-en:start -->
On both sides of $0$, the function values are automatically $6$; the left-hand derivative is $b$ and the right-hand derivative is $5$, so $b=5$, while $a$ is arbitrary. Although this exercise appears in the Product Rule session, it actually reviews how to join pieces differentiably.
<!-- bilingual-en:end -->

### 边界情况与易错点
<!-- bilingual-en:start -->
*Boundary Cases and Common Pitfalls*
<!-- bilingual-en:end -->

- 最大误区是写 $(uv)'=u'v'$；常数因子即可反驳：若 $u=x,v=x$，错误公式给 $1$，真实导数是 $2x$。
- 加减中间项有两种选择，所得两项顺序可不同，但结论相同。
- 因子超过两个时不要漏掉任何“只求一个因子导数”的项。
- 先化简再求导有时更短，但必须保持定义域；约去因子可能掩盖原函数的洞。
<!-- bilingual-en:start -->
- The most serious error is writing $(uv)'=u'v'$. A simple counterexample is $u=v=x$: the false rule gives $1$, whereas the true derivative is $2x$.
- The intermediate term can be added and subtracted in two equivalent ways, so the two resulting terms may appear in the opposite order.
- With more than two factors, include every term obtained by differentiating exactly one factor.
- Simplifying before differentiating can be shorter, but preserve the original domain: cancelling a factor can hide a hole in the function.
<!-- bilingual-en:end -->

### 三道自检
<!-- bilingual-en:start -->
*Three Quick Self-Checks*
<!-- bilingual-en:end -->

1. 用定义证明时为何要加减 $u(x+h)v(x)$？
2. 求 $D[x^2\cos x]$，并在 $x=0$ 检查。
3. 写出 $(uvwx)'$（最后一个 $x$ 是自变量）的展开式。
<!-- bilingual-en:start -->

&nbsp;
**1.** Why do we add and subtract $u(x+h)v(x)$ in the proof from the definition?<br>
**2.** Find $D[x^2\cos x]$ and check the result at $x=0$.<br>
**3.** Expand $(uvwx)'$, where the final $x$ is the independent variable.<br>
<!-- bilingual-en:end -->

> [!success]- 自检答案
> 1. 它把一个无法识别的乘积增量拆成两个标准差商；所加所减相同，不改变分子。
> 2. $2x\cos x-x^2\sin x$；在 $0$ 为 $0$，与 $x^2\cos x$ 在原点的水平切线一致。
> 3. $u'vwx+uv'wx+uvw'x+uvw$。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** It splits an otherwise unrecognisable product increment into two standard difference quotients. Adding and subtracting the same term leaves the numerator unchanged.<br>
> **2.** $2x\cos x-x^2\sin x$; $0$ at $0$, consistent with the horizontal tangent of $x^2\cos x$ at the origin.<br>
> **3.** $u'vwx+uv'wx+uvw'x+uvw$.<br>
> <!-- bilingual-en:end -->

### 本地材料

- [[Ses09a_Lecture_Notes.pdf#page=1|09a General Derivative Rules（p.1）]]
- [[Ses09b_Lecture_Notes.pdf#page=1|09b Introduction to General Rules（p.1）]]
- [[Ses09c_Lecture_Notes.pdf#page=1|09c Product Formula and Proof（pp.1-2）]]
- [[Exercise009_Problems.pdf#page=1|Exercise 009：Smoothing a Piecewise Polynomial]] · [[Exercise009_Solutions.pdf#page=1|答案]]

**知识链：**乘积增量 → 加减中间项 → 两个差商 + 连续性 → 积法则。
<!-- bilingual-en:start -->
**Knowledge chain:** increment of a product → add and subtract an intermediate term → two difference quotients plus continuity → product rule.
<!-- bilingual-en:end -->

## Session 10：Quotient Rule

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题：**分子、分母都变化时，商的导数如何由各自导数组成？
<!-- bilingual-en:start -->
**Question:** When both numerator and denominator vary, how do their derivatives combine in the quotient rule?
<!-- bilingual-en:end -->

**前置知识：**分式通分、积法则、可导蕴含连续。
<!-- bilingual-en:start -->
**Prerequisites:** Combining rational expressions over a common denominator, the product rule, and the fact that differentiability implies continuity.
<!-- bilingual-en:end -->

### 10a：商法则推导
<!-- bilingual-en:start -->
*10a: Deriving the Quotient Rule*
<!-- bilingual-en:end -->

设 $v(x)\ne0$，且在 $x$ 附近分母也不为零。记
<!-- bilingual-en:start -->
Suppose $v(x)\ne0$, and the denominator is not zero near $x$.
<!-- bilingual-en:end -->

$$
\Delta u=u(x+h)-u(x),\qquad
\Delta v=v(x+h)-v(x).
$$

则 $u(x+h)=u+\Delta u$、$v(x+h)=v+\Delta v$。商的增量为
<!-- bilingual-en:start -->
Then $u(x+h)=u+\Delta u$ and $v(x+h)=v+\Delta v$. The increment in the quotient is
<!-- bilingual-en:end -->

$$
\begin{aligned}
\frac{u+\Delta u}{v+\Delta v}-\frac uv
&=\frac{(u+\Delta u)v-u(v+\Delta v)}{v(v+\Delta v)}\\
&=\frac{v\Delta u-u\Delta v}{v(v+\Delta v)}.
\end{aligned}
$$

再除以 $h$：
<!-- bilingual-en:start -->
Dividing by $h$ gives
<!-- bilingual-en:end -->

$$
\frac{\Delta(u/v)}h
=\frac{v(\Delta u/h)-u(\Delta v/h)}{v(v+\Delta v)}.
$$

令 $h\to0$。因 $v$ 可导所以连续，$\Delta v\to0$；差商趋于导数：
<!-- bilingual-en:start -->
Let $h\to0$. Since $v$ is differentiable, it is continuous and hence $\Delta v\to0$; the two difference quotients tend to the corresponding derivatives:
<!-- bilingual-en:end -->

> [!important] 商法则
> $$
> \boxed{\left(\frac uv\right)'=\frac{u'v-uv'}{v^2}},\qquad v\ne0.
> $$

记忆时读作“下乘上导，减上乘下导，除以下平方”。负号次序与分子原顺序绑定。
<!-- bilingual-en:start -->
As a mnemonic, read it as “bottom times the derivative of the top, minus top times the derivative of the bottom, all over the bottom squared.” The order of the two numerator terms fixes the sign.
<!-- bilingual-en:end -->

### 10b：倒数与负整数幂
<!-- bilingual-en:start -->
*10b: Reciprocals and Negative-Integer Powers*
<!-- bilingual-en:end -->

令 $u=1$：
<!-- bilingual-en:start -->
Let $u=1$:
<!-- bilingual-en:end -->

$$
\left(\frac1v\right)'=-\frac{v'}{v^2}=-v^{-2}v'.
$$

取 $v=x^n$：
<!-- bilingual-en:start -->
Take $v=x^n$:
<!-- bilingual-en:end -->

$$
\frac d{dx}x^{-n}
=-x^{-2n}\cdot nx^{n-1}
=-nx^{-n-1}.
$$

因此幂法则从正整数扩展到负整数：
<!-- bilingual-en:start -->
Thus the power rule extends from positive integers to negative integers:
<!-- bilingual-en:end -->

$$
\frac d{dx}x^m=mx^{m-1},\qquad m\in\mathbb Z, x\ne0\text{（若 }m<0\text{）}.
$$

### 代表例题：正切与正割
<!-- bilingual-en:start -->
*Representative Example: Tangent and Secant*
<!-- bilingual-en:end -->

$$
\begin{aligned}
(\tan x)'
&=\left(\frac{\sin x}{\cos x}\right)'\\
&=\frac{\cos^2x+\sin^2x}{\cos^2x}\\
&=\boxed{\sec^2x},\qquad \cos x\ne0.
\end{aligned}
$$

$$
(\sec x)'=\left(\frac1{\cos x}\right)'
=\frac{\sin x}{\cos^2x}
=\boxed{\sec x\tan x}.
$$

### 边界情况与易错点
<!-- bilingual-en:start -->
*Boundary Cases and Common Pitfalls*
<!-- bilingual-en:end -->

- 结果分母是 $v^2$，并不取消原限制 $v\ne0$。
- 分子次序写反会整体差一个负号；可用简单函数如 $1/x$ 检查。
- 若可改写为 $uv^{-1}$，积法则 + 链式法则常比死记商法则更可靠；这是下一节练习主题。
- 先约分可能改变定义域。例如 $(x^2-1)/(x-1)=x+1$ 只在 $x\ne1$ 相等。
<!-- bilingual-en:start -->
- The resulting denominator is $v^2$ and does not remove the original restriction $v\ne0$.
- Reversing the order of the numerator terms changes the overall sign; check the formula on a simple function such as $1/x$.
- If the quotient can be written as $uv^{-1}$, the product and chain rules are often more reliable than memorizing the quotient rule mechanically; this is the theme of the next exercise.
- Simplifying first can change the apparent domain. For example, $(x^2-1)/(x-1)$ equals $x+1$ only when $x\ne1$.
<!-- bilingual-en:end -->

### 三道自检
<!-- bilingual-en:start -->
*Three Quick Self-Checks*
<!-- bilingual-en:end -->

1. 求 $D[x^2/(x+1)]$，并化简。
2. 用倒数公式重新求 $(1/x)'$。
3. 为什么 $\tan x$ 的导数公式不能在 $x=\pi/2$ 使用？
<!-- bilingual-en:start -->

&nbsp;
**1.** Find $D[x^2/(x+1)]$ and simplify.<br>
**2.** Recalculate $(1/x)'$ with the reciprocal formula.<br>
**3.** Why can't the derivative formula for $\tan x$ be used at $x=\pi/2$?<br>
<!-- bilingual-en:end -->

> [!success]- 自检答案
> 1. $[2x(x+1)-x^2]/(x+1)^2=x(x+2)/(x+1)^2$，$x\ne-1$。
> 2. 取 $v=x,v'=1$，得到 $-1/x^2$。
> 3. $\cos(\pi/2)=0$，原函数 $\tan x$ 未定义；求导公式不能补出原函数的定义点。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** $[2x(x+1)-x^2]/(x+1)^2=x(x+2)/(x+1)^2$,$x\ne-1$.<br>
> **2.** Take $v=x,v'=1$ to get $-1/x^2$.<br>
> **3.** Since $\cos(\pi/2)=0$, the function $\tan x$ is undefined there; its derivative formula cannot extend the function's domain.<br>
> <!-- bilingual-en:end -->

### 本地材料

- [[Ses10a_Lecture_Notes.pdf#page=1|10a Quotient Rule（pp.1-2）]]
- [[Ses10b_Lecture_Notes.pdf#page=1|10b Reciprocals and Negative Powers（p.1）]]
- [[Exercise010_Problems.pdf#page=1|Exercise 010：Quotient Rule Practice]] · [[Exercise010_Solutions.pdf#page=1|答案]]

**知识链：**商的有限增量通分 → 分子分解成 $v\Delta u-u\Delta v$ → 取极限 → 商法则与倒数法则。
<!-- bilingual-en:start -->
**Conceptual chain:** Put the finite increment of a quotient over a common denominator → split the numerator into $v\Delta u-u\Delta v$ → take the limit → obtain the quotient and reciprocal rules.
<!-- bilingual-en:end -->

## Session 11：Chain Rule

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题：**输入经过多层函数转换时，总变化率为何是各层局部变化率的乘积？
<!-- bilingual-en:start -->
**Question:** When an input passes through several nested functions, why is the overall rate of change the product of the local rates at each level?
<!-- bilingual-en:end -->

**前置知识：**函数复合、导数、连续、积法则。
<!-- bilingual-en:start -->
**Prerequisites:** function composition, derivatives, continuity, and the product rule.
<!-- bilingual-en:end -->

### 11a：中间变量与变化率相乘
<!-- bilingual-en:start -->
*11a: Intermediate Variables and Multiplication of Rates*
<!-- bilingual-en:end -->

设
<!-- bilingual-en:start -->
Let
<!-- bilingual-en:end -->

$$
x=g(t),\qquad y=f(x)=f(g(t)).
$$

有限变化时，若 $\Delta x\ne0$，
<!-- bilingual-en:start -->
For finite changes with $\Delta x\ne0$,
<!-- bilingual-en:end -->

$$
\frac{\Delta y}{\Delta t}
=\frac{\Delta y}{\Delta x}\frac{\Delta x}{\Delta t}.
$$

当 $\Delta t\to0$，可导性使 $\Delta x\to0$，两个倍率分别趋于 $dy/dx$ 与 $dx/dt$，于是
<!-- bilingual-en:start -->
As $\Delta t\to0$, differentiability implies $\Delta x\to0$, and the two factors tend to $dy/dx$ and $dx/dt$, respectively.
<!-- bilingual-en:end -->

> [!important] [[导数与求导规则#求导规则为何成立|链式法则]]
> $$
> \boxed{\frac{dy}{dt}=\frac{dy}{dx}\frac{dx}{dt}},
> $$
> 或
> $$
> \boxed{(f\circ g)'(t)=f'(g(t))g'(t)}.
> $$
> <!-- bilingual-en:start -->
> or
> <!-- bilingual-en:end -->

直觉是“每单位 $t$ 产生多少 $x$”乘“每单位 $x$ 产生多少 $y$”。单位也会相消：$(y/x)(x/t)=y/t$。
<!-- bilingual-en:start -->
The intuition is “how much $x$ is produced per unit of $t$” multiplied by “how much $y$ is produced per unit of $x$.” The intermediate units cancel: $(y/x)(x/t)=y/t$.
<!-- bilingual-en:end -->

### 不跳过 $\Delta x=0$ 的证明细节
<!-- bilingual-en:start -->
*Do Not Skip the $\Delta x=0$ Detail in the Proof*
<!-- bilingual-en:end -->

直接约掉 $\Delta x$ 在某些步长上可能遇到 $\Delta x=0$。定义辅助函数
<!-- bilingual-en:start -->
Cancelling $\Delta x$ directly can fail for increments at which $\Delta x=0$. Define the auxiliary function
<!-- bilingual-en:end -->

$$
\phi(u)=
\begin{cases}
\dfrac{f(g(t)+u)-f(g(t))}{u},&u\ne0,\\[6pt]
f'(g(t)),&u=0.
\end{cases}
$$

$f$ 在 $g(t)$ 可导意味着 $\phi(u)\to f'(g(t))$，并且补值后 $\phi$ 在 $0$ 连续。令 $u=g(t+h)-g(t)$，则无论 $u$ 是否为零都有
<!-- bilingual-en:start -->
$f$ being differentiable at $g(t)$ means that $\phi(u)\to f'(g(t))$; after defining its value at $0$, $\phi$ is continuous there. Let $u=g(t+h)-g(t)$. Whether or not $u$ is zero,
<!-- bilingual-en:end -->

$$
f(g(t+h))-f(g(t))=\phi(u)u.
$$

所以
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

$$
\frac{f(g(t+h))-f(g(t))}{h}
=\phi(u)\frac{g(t+h)-g(t)}h.
$$

令 $h\to0$：因 $g$ 可导故连续，$u\to0$；第一因子趋于 $f'(g(t))$，第二因子趋于 $g'(t)$。链式法则得证。
<!-- bilingual-en:start -->
Let $h\to0$. Because $g$ is differentiable, it is continuous, so $u\to0$. The first factor tends to $f'(g(t))$ and the second to $g'(t)$, proving the chain rule.
<!-- bilingual-en:end -->

### 11a 例题：$\sin^{10}t$
<!-- bilingual-en:start -->
*11a Example: $\sin^{10}t$*
<!-- bilingual-en:end -->

令 $x=\sin t$、$y=x^{10}$：
<!-- bilingual-en:start -->
Let $x=\sin t$ and $y=x^{10}$:
<!-- bilingual-en:end -->

$$
\frac{dy}{dt}
=\frac{dy}{dx}\frac{dx}{dt}
=10x^9\cos t
=\boxed{10\sin^9t\cos t}.
$$

### 11b 例题：$\sin(10t)$
<!-- bilingual-en:start -->
*11b Example: $\sin(10t)$*
<!-- bilingual-en:end -->

外函数是 $\sin x$，内函数是 $x=10t$：
<!-- bilingual-en:start -->
The outer function is $\sin x$ and the inner function is $x=10t$:
<!-- bilingual-en:end -->

$$
\frac d{dt}\sin(10t)
=\cos(10t)\cdot10
=\boxed{10\cos(10t)}.
$$

注意 $\sin^{10}t$ 与 $\sin(10t)$ 完全不同：前者是函数值的十次幂，后者是角度放大十倍。
<!-- bilingual-en:start -->
Notice that $\sin^{10}t$ and $\sin(10t)$ mean entirely different things: the former is the tenth power of a function value, while the latter evaluates sine at an angle multiplied by ten.
<!-- bilingual-en:end -->

### 多层复合
<!-- bilingual-en:start -->
*Compositions with Several Layers*
<!-- bilingual-en:end -->

$$
\frac d{dx}\sin\bigl((x^2+1)^3\bigr)
=\cos\bigl((x^2+1)^3\bigr)
\cdot3(x^2+1)^2\cdot2x.
$$

从最外层向内层逐层求导；每经过一层就乘该层内函数的导数，直到自变量。
<!-- bilingual-en:start -->
Differentiate from the outermost layer inward, multiplying by the derivative contributed by each layer until reaching the independent variable.
<!-- bilingual-en:end -->

### 边界情况与易错点
<!-- bilingual-en:start -->
*Boundary Cases and Common Pitfalls*
<!-- bilingual-en:end -->

- 最常见错误是只求外层导数，漏乘内层导数。
- $f'(g(x))$ 表示先把 $g(x)$ 代入 $f'$；不是 $f'(x)g(x)$。
- 链式法则要求外函数在内函数输出处可导、内函数在当前点可导。
- Leibniz 记号提供记忆直觉，但正式依据是复合函数的极限证明。
<!-- bilingual-en:start -->
- The most common error is differentiating only the outer function and omitting the derivative of the inner function.
- $f'(g(x))$ means evaluate $f'$ at $g(x)$; it does not mean $f'(x)g(x)$.
- The chain rule requires the outer function to be differentiable at the inner function's output and the inner function to be differentiable at the current point.
- Leibniz notation provides a useful mnemonic, but the formal justification is the limit proof for a composite function.
<!-- bilingual-en:end -->

### 三道自检
<!-- bilingual-en:start -->
*Three Quick Self-Checks*
<!-- bilingual-en:end -->

1. 求 $D[(3x^2-1)^5]$。
2. 比较 $D[\cos^2x]$ 与 $D[\cos(x^2)]$。
3. 若温度 $T$ 随高度 $z$ 变化，气球高度随时间变化，解释 $dT/dt=(dT/dz)(dz/dt)$ 的单位。
<!-- bilingual-en:start -->

&nbsp;
**1.** Find $D[(3x^2-1)^5]$.<br>
**2.** Compare $D[\cos^2x]$ with $D[\cos(x^2)]$.<br>
**3.** If temperature $T$ varies with altitude $z$, while the balloon's altitude varies with time, explain the units in $dT/dt=(dT/dz)(dz/dt)$.<br>
<!-- bilingual-en:end -->

> [!success]- 自检答案
> 1. $5(3x^2-1)^4\cdot6x=30x(3x^2-1)^4$。
> 2. $D[\cos^2x]=-2\sin x\cos x$；$D[\cos(x^2)]=-2x\sin(x^2)$。
> 3. $(\text{度}/\text{米})(\text{米}/\text{秒})=\text{度}/\text{秒}$，表示气球经历的温度随时间变化率。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** $5(3x^2-1)^4\cdot6x=30x(3x^2-1)^4$.<br>
> **2.** $D[\cos^2x]=-2\sin x\cos x$;$D[\cos(x^2)]=-2x\sin(x^2)$.<br>
> **3.** $(\text{degrees}/\text{metre})(\text{metres}/\text{second})=\text{degrees}/\text{second}$, the rate at which the balloon experiences a change in temperature over time.<br>
> <!-- bilingual-en:end -->

### 本地材料

- [[Ses11a_Lecture_Notes.pdf#page=1|11a Chain Rule and $\sin^{10}t$（pp.1-2）]]
- [[Ses11b_Lecture_Notes.pdf#page=1|11b Example $\sin(10t)$（p.1）]]
- [[Exercise011_Problems.pdf#page=1|Exercise 011：Do We Need the Quotient Rule?]] · [[Exercise011_Solutions.pdf#page=1|答案]]

**知识链：**复合函数 → 引入中间变量 → 局部倍率相乘 → 处理 $\Delta x=0$ 的严格细节 → 多层链式法则。
<!-- bilingual-en:start -->
**Knowledge chain:** composite function → introduce an intermediate variable → multiply local rates of change → handle the rigorous $\Delta x=0$ detail → multi-layer chain rule.
<!-- bilingual-en:end -->

## Session 12：Higher Derivatives

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题：**导数本身继续变化时，怎样表示并解释这种“变化率的变化率”？
<!-- bilingual-en:start -->
**Question:** When a derivative itself changes, how do higher derivatives express and interpret that change?
<!-- bilingual-en:end -->

**前置知识：**基本求导规则、正余弦导数、运动的速度解释。
<!-- bilingual-en:start -->
**Prerequisites:** Basic differentiation rules, the derivatives of sine and cosine, and the interpretation of velocity in motion.
<!-- bilingual-en:end -->

### 12a：定义、记号与解释
<!-- bilingual-en:start -->
*12a: Definition, Notation, and Interpretation*
<!-- bilingual-en:end -->

若 $f'$ 仍可导，则
<!-- bilingual-en:start -->
If $f'$ is also differentiable, then
<!-- bilingual-en:end -->

$$
f''(x)=D^2f(x)=\frac{d^2f}{dx^2}.
$$

继续求导：
<!-- bilingual-en:start -->
Continuing to differentiate gives
<!-- bilingual-en:end -->

$$
f^{(n)}(x)=D^nf(x)=\frac{d^nf}{dx^n}.
$$

这里 $f^{(n)}$ 是[[导数与求导规则#从差商到导数|高阶导数]]，不是 $f$ 的 $n$ 次幂。Leibniz 记号中的 $d^2f/dx^2$ 也是一个整体，不应读成普通分数平方。
<!-- bilingual-en:start -->
Here $f^{(n)}$ denotes the [[导数与求导规则#从差商到导数|$n$th derivative]], not the $n$th power of $f$. Likewise, $d^2f/dx^2$ is a single piece of Leibniz notation and should not be read as the square of an ordinary fraction.
<!-- bilingual-en:end -->

运动中
<!-- bilingual-en:start -->
In motion
<!-- bilingual-en:end -->

$$
s'(t)=v(t),\qquad s''(t)=v'(t)=a(t).
$$

图像上，$f''>0$ 表示斜率 $f'$ 随 $x$ 增大而增加（向上凹）；$f''<0$ 表示斜率减少（向下凹）。这一曲线描绘解释将在 Unit 2 系统使用。
<!-- bilingual-en:start -->
On the graph, $f''>0$ means that the slope $f'$ increases with $x$ (concave up), while $f''<0$ means that the slope decreases (concave down). Unit 2 will use this interpretation systematically when sketching curves.
<!-- bilingual-en:end -->

### 正弦的四阶循环
<!-- bilingual-en:start -->
*The Four-Derivative Cycle of Sine*
<!-- bilingual-en:end -->

$$
\sin x\xrightarrow D\cos x
\xrightarrow D-\sin x
\xrightarrow D-\cos x
\xrightarrow D\sin x.
$$

因此阶数对 $4$ 取余即可。例如 $101\equiv1\pmod4$，
<!-- bilingual-en:start -->
Thus only the derivative order modulo $4$ matters. For example, $101\equiv1\pmod4$, so
<!-- bilingual-en:end -->

$$
\frac{d^{101}}{dx^{101}}\sin x=\cos x.
$$

同时 $\sin x$ 与 $\cos x$ 都满足
<!-- bilingual-en:start -->
Both $\sin x$ and $\cos x$ satisfy
<!-- bilingual-en:end -->

$$
y''=-y,
$$

这是简谐振动微分方程的核心形式。
<!-- bilingual-en:start -->
This is the fundamental differential equation for simple harmonic motion.
<!-- bilingual-en:end -->

### 12b：$D^nx^n=n!$

逐次求导：
<!-- bilingual-en:start -->
Differentiating successively gives
<!-- bilingual-en:end -->

$$
\begin{aligned}
Dx^n&=nx^{n-1},\\
D^2x^n&=n(n-1)x^{n-2},\\
D^3x^n&=n(n-1)(n-2)x^{n-3}.
\end{aligned}
$$

第 $k$ 阶为
<!-- bilingual-en:start -->
The $k$th derivative is
<!-- bilingual-en:end -->

$$
D^kx^n=\frac{n!}{(n-k)!}x^{n-k},\qquad 0\le k\le n.
$$

取 $k=n$：
<!-- bilingual-en:start -->
Take $k=n$:
<!-- bilingual-en:end -->

$$
\boxed{D^nx^n=n!}.
$$

再求一次，常数导数为零：$D^{n+1}x^n=0$；更高阶也全为零。
<!-- bilingual-en:start -->
Differentiating once more gives zero because the derivative of a constant is zero: $D^{n+1}x^n=0$. Every higher derivative is also zero.
<!-- bilingual-en:end -->

### 乘积高阶导数：Leibniz 公式
<!-- bilingual-en:start -->
*Higher Derivatives of a Product: Leibniz's Formula*
<!-- bilingual-en:end -->

重复使用积法则会出现二项式系数：
<!-- bilingual-en:start -->
Repeated use of the product rule produces binomial coefficients:
<!-- bilingual-en:end -->

$$
(uv)^{(n)}=
\sum_{k=0}^n\binom nk u^{(k)}v^{(n-k)}.
$$

例如
<!-- bilingual-en:start -->
For example
<!-- bilingual-en:end -->

$$
(uv)''=u''v+2u'v'+uv'',
$$

中间系数 $2$ 来自两条不同求导路径。
<!-- bilingual-en:start -->
The middle coefficient $2$ comes from two distinct differentiation paths.
<!-- bilingual-en:end -->

### 边界情况与易错点
<!-- bilingual-en:start -->
*Boundary Cases and Common Pitfalls*
<!-- bilingual-en:end -->

- 二阶导数记号是 $d^2y/dx^2$，不是 $(dy/dx)^2$。
- $f''(a)=0$ 不足以断言拐点；还要检查凹向是否改变。
- 高阶求导时积法则产生的系数会累积，不能只对每个因子各求同阶一次。
- 物理中加速度与速度同号表示速率增加，异号表示速率减少；不能只看 $a$ 正负。
<!-- bilingual-en:start -->
- Second derivative notation is $d^2y/dx^2$, not $(dy/dx)^2$.
- $f''(a)=0$ is not enough to establish an inflection point; check whether the concavity actually changes.
- Repeated product rules generate accumulated coefficients in higher derivatives; it is not enough simply to differentiate each factor to the same order once.
- In mechanics, velocity and acceleration with the same sign mean speed is increasing, while opposite signs mean speed is decreasing; the sign of acceleration alone is not enough.
<!-- bilingual-en:end -->

### 三道自检
<!-- bilingual-en:start -->
*Three Quick Self-Checks*
<!-- bilingual-en:end -->

1. 求 $D^6x^4$。
2. 求 $D^{2026}\cos x$。
3. 展开 $(uv)'''$。
<!-- bilingual-en:start -->

&nbsp;
**1.** Find $D^6x^4$.<br>
**2.** Find $D^{2026}\cos x$.<br>
**3.** Expand $(uv)^{(3)}$.<br>
<!-- bilingual-en:end -->

> [!success]- 自检答案
> 1. 四阶后为 $4!=24$，五阶起为零，所以答案 $0$。
> 2. $2026\equiv2\pmod4$，故为 $-\cos x$。
> 3. $u'''v+3u''v'+3u'v''+uv'''$。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** The fourth derivative is $4!=24$, and every derivative from the fifth onward is zero, so the answer is $0$.<br>
> **2.** $2026\equiv2\pmod4$, hence $-\cos x$.<br>
> **3.** $u'''v+3u''v'+3u'v''+uv'''$.<br>
> <!-- bilingual-en:end -->

### 本地材料

- [[Ses12a_Lecture_Notes.pdf#page=1|12a Higher Derivatives and Notation（p.1）]]
- [[Ses12b_Lecture_Notes.pdf#page=1|12b Example $D^nx^n$（p.1）]]
- [[Exercise012_Problems.pdf#page=1|Exercise 012：Repeated Differentiation]] · [[Exercise012_Solutions.pdf#page=1|答案]]

**知识链：**导函数仍是函数 → 重复求导 → 速度/加速度与凹向 → 阶数模式、阶乘和 Leibniz 公式。
<!-- bilingual-en:start -->
**Knowledge chain:** the derivative is itself a function → differentiate repeatedly → velocity, acceleration, and concavity → order patterns, factorials, and the Leibniz formula.
<!-- bilingual-en:end -->

## Problem Set 1

官方在 Part A 后指定同一份 Differentiation 题册中的 1A、1B、1C、1D、1E、1F、1G、1J 选题。下列编号严格按[官网 Problem Set 1](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/pages/1.-differentiation/part-a-definition-and-basic-rules/problem-set-1/)；不是整本题册的所有题。
<!-- bilingual-en:start -->
After Part A, the official course assigns selected problems from Sections 1A, 1B, 1C, 1D, 1E, 1F, 1G, and 1J of the same Differentiation problem booklet. The numbering below follows [Problem Set 1](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/pages/1.-differentiation/part-a-definition-and-basic-rules/problem-set-1/) exactly; it is not a list of every problem in the booklet.
<!-- bilingual-en:end -->

- [[PSet01_Problems.pdf#page=1|Differentiation 原题（pp.1-13）]]
- [[PSet01_Solutions.pdf#page=1|官方解答（pp.1-20）]]

| 节 | 官网指定题 |
|---|---|
| 1A Graphing | 1b, 2b, 3a, 3b, 3e, 6b, 7b |
| 1B Velocity and rates | 1a, 1b, 1c |
| 1C Slope and derivative | 1a, 3a, 3b, 3e, 4a, 4b, 5, 6, 2 |
| 1D Limits and continuity | 1a, 1c, 1d, 1f, 1g, 3a, 3c, 3d, 3e, 6a, 8a |
| 1E Polynomials, products, quotients | 1a, 1c, 2b, 3, 4b, 5a, 5c |
| 1J Trigonometric functions | 1e, 2；随后 1a, 1k, 1m |
| 1F Chain rule, implicit differentiation | 1a, 1b, 2, 6, 7b, 7c |
| 1G Higher derivatives | 1b, 1c, 5a |

> [!example]- 1A：图像、奇偶性与三角变换（原题 p.1）
> **1A-1b。** 配方并描绘 $y=3x^2+6x+2$：
> $$
> y=3(x^2+2x)+2=3(x+1)^2-1.
> $$
> 所以它由 $y=x^2$ 左移 $1$、纵向放大 $3$、下移 $1$ 得到。顶点 $(-1,-1)$，对称轴 $x=-1$，开口向上，$y$ 截距 $2$，零点 $x=-1\pm1/\sqrt3$。
>
> **1A-2b。** 描绘
> $$
> y=\frac2{(x-1)^2}.
> $$
> 由 $2/x^2$ 右移 $1$；定义域 $x\ne1$，恒正，关于直线 $x=1$ 对称，竖直渐近线 $x=1$，水平渐近线 $y=0$。当 $x\to1^\pm$ 时都趋 $+\infty$。
>
> **1A-3a。** $f(x)=(x^3+3x)/(1-x^4)$。分子奇、分母偶，故
> $$f(-x)=-f(x),$$
> 是奇函数；定义域也关于原点对称。
>
> **1A-3b。** $f(x)=\sin^2x$：
> $$f(-x)=(-\sin x)^2=\sin^2x,$$
> 是偶函数。
>
> **1A-3e。** 即使不知道 $J_0$ 是什么，$(-x)^2=x^2$，所以
> $$J_0((-x)^2)=J_0(x^2),$$
> 必为偶函数。
>
> **1A-6b。** 将 $\sin x-\cos x$ 写成 $A\sin(x+c)$。展开右侧：
> $$A\sin(x+c)=A\cos c\sin x+A\sin c\cos x.$$
> 比较系数 $A\cos c=1,A\sin c=-1$，故 $A=\sqrt2,c=-\pi/4$：
> $$\boxed{\sin x-\cos x=\sqrt2\sin(x-\pi/4)}.$$
>
> **1A-7b。**
> $$-4\cos(x+\pi/2)=4\sin x.$$
> 因而振幅 $4$、周期 $2\pi$、以正弦形式看的相位移为 $0$。若坚持余弦形式，负振幅可改为相位再移 $\pi$，但标准振幅取正数。
>
> **检查。** 图像题必须同时标出平移、尺度、关键点和渐近线；“相位角/相位移”的符号取决于写成 $A\sin(k(x-c))$ 还是 $A\sin(kx+\phi)$，需先声明形式。
> <!-- bilingual-en:start -->
> **1A-1b.** Complete the square and sketch $y=3x^2+6x+2$:
> $$
> y=3(x^2+2x)+2=3(x+1)^2-1.
> $$
> Starting from $y=x^2$, shift left by $1$, stretch vertically by a factor of $3$, and shift down by $1$. The vertex is $(-1,-1)$; the axis of symmetry is $x=-1$; the parabola opens upward; the $y$-intercept is $2$; and the zeros are $x=-1\pm1/\sqrt3$.
>
> **1A-2b.** Sketch
> $$
> y=\frac2{(x-1)^2}.
> $$
> This is the graph of $2/x^2$ shifted right by $1$. Its domain is $x\ne1$; it is always positive and symmetric about $x=1$; its vertical asymptote is $x=1$ and its horizontal asymptote is $y=0$. As $x\to1^\pm$, the function tends to $+\infty$.
>
> **1A-3a.** For $f(x)=(x^3+3x)/(1-x^4)$, the numerator is odd and the denominator is even, so
> $$f(-x)=-f(x).$$
> Thus $f$ is odd, and its domain is also symmetric about the origin.
>
> **1A-3b.** For $f(x)=\sin^2x$,
> $$f(-x)=(-\sin x)^2=\sin^2x,$$
> so $f$ is even.
>
> **1A-3e.** Even without knowing the definition of $J_0$, $(-x)^2=x^2$, so
> $$J_0((-x)^2)=J_0(x^2).$$
> Therefore the function is even.
>
> **1A-6b.** Write $\sin x-\cos x$ as $A\sin(x+c)$. Expanding gives
> $$A\sin(x+c)=A\cos c\sin x+A\sin c\cos x.$$
> Matching coefficients yields $A\cos c=1$ and $A\sin c=-1$, so $A=\sqrt2$ and $c=-\pi/4$:
> $$\boxed{\sin x-\cos x=\sqrt2\sin(x-\pi/4)}.$$
>
> **1A-7b.**
> $$-4\cos(x+\pi/2)=4\sin x.$$
> The amplitude is $4$, the period is $2\pi$, and the phase shift in sine form is $0$. If a cosine form is required, absorb the negative sign into an additional phase shift of $\pi$; by convention, amplitude is reported as positive.
>
> **Check.** A graphing answer should state the translations, scale changes, key points, and asymptotes. The sign convention for a phase angle or phase shift depends on whether the expression is written as $A\sin(k(x-c))$ or $A\sin(kx+\phi)$, so state the chosen form first.
> <!-- bilingual-en:end -->

> [!example]- 1B：Green Building 落体（原题 p.2）
> 高度为
> $$h(t)=400-16t^2\quad(\text{ft}).$$
> **(a) 前两秒平均速度：**
> $$
> \frac{h(2)-h(0)}{2}
> =\frac{336-400}{2}=\boxed{-32\text{ ft/s}}.
> $$
> **(b) 最后两秒。** 先由 $h(t)=0$ 得落地时 $t=5$；最后两秒是 $[3,5]$：
> $$
> \frac{h(5)-h(3)}2
> =\frac{0-256}{2}=\boxed{-128\text{ ft/s}}.
> $$
> **(c) 落地瞬时速度。** 直接用定义在 $t=5$：
> $$
> \begin{aligned}
> h'(5)
> &=\lim_{t\to5}\frac{h(t)-h(5)}{t-5}\\
> &=\lim_{t\to5}\frac{400-16t^2}{t-5}\\
> &=\lim_{t\to5}-16(t+5)=-160.
> \end{aligned}
> $$
> 因此速度 $\boxed{-160\text{ ft/s}}$，速率 $160\text{ ft/s}$。负号来自向上为正的坐标选择。
> <!-- bilingual-en:start -->
> The height is
> $$h(t)=400-16t^2\quad(\text{ft}).$$
> **(a) Average velocity during the first two seconds:**
> $$
> \frac{h(2)-h(0)}2
> =\frac{336-400}{2}=\boxed{-32\text{ ft/s}}.
> $$
> **(b) Average velocity during the final two seconds.** Solving $h(t)=0$ gives the impact time $t=5$, so the final two seconds are $[3,5]$:
> $$
> \frac{h(5)-h(3)}2
> =\frac{0-256}{2}=\boxed{-128\text{ ft/s}}.
> $$
> **(c) Instantaneous velocity at impact.** Apply the definition at $t=5$:
> $$
> \begin{aligned}
> h'(5)
> &=\lim_{t\to5}\frac{h(t)-h(5)}{t-5}\\
> &=\lim_{t\to5}\frac{400-16t^2}{t-5}\\
> &=\lim_{t\to5}-16(t+5)=-160.
> \end{aligned}
> $$
> Hence the velocity is $\boxed{-160\text{ ft/s}}$ and the speed is $160\text{ ft/s}$. The negative sign results from choosing upward as the positive direction.
> <!-- bilingual-en:end -->

> [!example]- 1C-1、1C-2：从定义看面积变化和因子（原题 p.3）
> **1C-1a。** 圆盘面积 $A(r)=\pi r^2$：
> $$
> \frac{A(r+h)-A(r)}h
> =\pi\frac{2rh+h^2}{h}=2\pi r+\pi h\to\boxed{2\pi r}.
> $$
> 导数等于圆周长。直觉上，半径增加 $h$ 所添薄环面积约为“周长 $\times h$”。
>
> **1C-2。** $f(x)=(x-a)g(x)$ 且 $g$ 在 $a$ 连续。先算 $f(a)=0$，再由定义：
> $$
> f'(a)=\lim_{x\to a}\frac{(x-a)g(x)-0}{x-a}
> =\lim_{x\to a}g(x)=\boxed{g(a)}.
> $$
> 约分只对 $x\ne a$ 做；最后一步明确使用 $g$ 在 $a$ 连续。
> <!-- bilingual-en:start -->
> **1C-1a.** For the area of a disk, $A(r)=\pi r^2$,
> $$
> \frac{A(r+h)-A(r)}h
> =\pi\frac{2rh+h^2}{h}=2\pi r+\pi h\to\boxed{2\pi r}.
> $$
> The derivative equals the circumference. Intuitively, increasing the radius by $h$ adds a thin ring whose area is approximately “circumference $\times h$.”
>
> **1C-2.** Let $f(x)=(x-a)g(x)$, where $g$ is continuous at $a$. First, $f(a)=0$. Then the definition gives
> $$
> f'(a)=\lim_{x\to a}\frac{(x-a)g(x)-0}{x-a}
> =\lim_{x\to a}g(x)=\boxed{g(a)}.
> $$
> The cancellation is performed only for $x\ne a$; the final equality explicitly uses continuity of $g$ at $a$.
> <!-- bilingual-en:end -->

> [!example]- 1C-3：按定义求导与指定斜率（原题 p.3）
> **(a)** $f(x)=1/(2x+1)$：
> $$
> \begin{aligned}
> \frac{f(x+h)-f(x)}h
> &=\frac1h\left(\frac1{2x+2h+1}-\frac1{2x+1}\right)\\
> &=-\frac2{(2x+2h+1)(2x+1)}
> \to\boxed{-\frac2{(2x+1)^2}}.
> \end{aligned}
> $$
> **(b)** $f(x)=2x^2+5x+4$：
> $$
> \frac{2(x+h)^2+5(x+h)+4-(2x^2+5x+4)}h
> =4x+2h+5\to\boxed{4x+5}.
> $$
> **(e) 指定斜率。** 对 (a)，导数恒负，所以无斜率 $1$ 或 $0$ 的点；令斜率 $-1$：
> $$
> -\frac2{(2x+1)^2}=-1
> \Longrightarrow x=-\frac12\pm\frac{\sqrt2}{2}.
> $$
> 对 (b)，$4x+5=1,-1,0$ 分别得
> $$x=-1,-\frac32,-\frac54.$$
> 题目问“点”时还应代回 $f(x)$；若只问横坐标，上述即可。
> <!-- bilingual-en:start -->
> **(a)** For $f(x)=1/(2x+1)$,
> $$
> \begin{aligned}
> \frac{f(x+h)-f(x)}h
> &=\frac1h\left(\frac1{2x+2h+1}-\frac1{2x+1}\right)\\
> &=-\frac2{(2x+2h+1)(2x+1)}
> \to\boxed{-\frac2{(2x+1)^2}}.
> \end{aligned}
> $$
> **(b)** For $f(x)=2x^2+5x+4$,
> $$
> \frac{2(x+h)^2+5(x+h)+4-(2x^2+5x+4)}h
> =4x+2h+5\to\boxed{4x+5}.
> $$
> **(e) Prescribed slopes.** In part (a), the derivative is always negative, so there is no point with slope $1$ or $0$. Setting the slope equal to $-1$ gives
> $$
> -\frac2{(2x+1)^2}=-1
> \Longrightarrow x=-\frac12\pm\frac{\sqrt2}{2}.
> $$
> In part (b), setting $4x+5$ equal to $1,-1,$ and $0$ gives, respectively,
> $$x=-1,-\frac32,-\frac54.$$
> If the problem asks for points, substitute these $x$-coordinates back into $f$; if it asks only for the horizontal coordinates, these values are sufficient.
> <!-- bilingual-en:end -->

> [!example]- 1C-4、1C-5：切线（原题 p.3）
> **1C-4a。** $f(x)=1/(2x+1)$ 在 $x=1$：$f(1)=1/3,f'(1)=-2/9$，
> $$
> y-\frac13=-\frac29(x-1)
> \quad\Longrightarrow\quad
> \boxed{y=\frac{-2x+5}{9}}.
> $$
> **1C-4b。** $f(x)=2x^2+5x+4$ 在 $x=a$：
> $$
> y-(2a^2+5a+4)=(4a+5)(x-a),
> $$
> 或
> $$\boxed{y=(4a+5)x-2a^2+4}.$$
> **1C-5。** 求过原点且与 $y=1+(x-1)^2$ 相切的直线。设切点横坐标为 $a$，斜率 $m=2(a-1)$。切线
> $$
> y=2(a-1)(x-a)+1+(a-1)^2.
> $$
> 令原点在直线上：
> $$0=-2a(a-1)+1+(a-1)^2=2-a^2,$$
> 故 $a=\pm\sqrt2$。两条切线为
> $$
> \boxed{y=2(\sqrt2-1)x},\qquad
> \boxed{y=-2(\sqrt2+1)x}.
> $$
> <!-- bilingual-en:start -->
> **1C-4a.** For $f(x)=1/(2x+1)$ at $x=1$, $f(1)=1/3$ and $f'(1)=-2/9$, so
> $$
> y-\frac13=-\frac29(x-1)
> \quad\Longrightarrow\quad
> \boxed{y=\frac{-2x+5}{9}}.
> $$
> **1C-4b.** For $f(x)=2x^2+5x+4$ at $x=a$,
> $$
> y-(2a^2+5a+4)=(4a+5)(x-a),
> $$
> or
> $$\boxed{y=(4a+5)x-2a^2+4}.$$
> **1C-5.** Find the lines through the origin tangent to $y=1+(x-1)^2$. Let the tangency point have $x$-coordinate $a$; then the slope is $m=2(a-1)$ and the tangent line is
> $$
> y=2(a-1)(x-a)+1+(a-1)^2.
> $$
> Requiring the line to pass through the origin gives
> $$0=-2a(a-1)+1+(a-1)^2=2-a^2,$$
> so $a=\pm\sqrt2$. The two tangent lines are
> $$
> \boxed{y=2(\sqrt2-1)x},\qquad
> \boxed{y=-2(\sqrt2+1)x}.
> $$
> <!-- bilingual-en:end -->

> [!example]- 1C-6：从函数图像画导函数（原题 p.3）
> 该题的五幅具体图必须查看 [[PSet01_Problems.pdf#page=3|原印刷页 p.3]]。逐图规则如下：
>
> 1. **半圆：**上半圆 $y=\sqrt{R^2-x^2}$ 有 $y'=-x/\sqrt{R^2-x^2}$；中心处为零，两端分别趋 $+\infty,-\infty$，导函数为奇函数。
> 2. **抛物线：**斜率随 $x$ 线性变化；顶点导数为零，左右符号相反。
> 3. **给定奇函数：**导函数应为偶函数；关于 $y$ 轴对称地复制斜率值。
> 4. **给定偶函数：**导函数应为奇函数；若在原点可导，必有 $f'(0)=0$。
> 5. **周期函数：**导函数保持相同周期；原函数的极大/极小点对应导函数零点，上升段导数正、下降段负。
>
> 作图检查顺序：先标不可导点与零点，再判正负，最后估计斜率大小；不要把原函数高度当作导数高度。
> <!-- bilingual-en:start -->
> Consult [[PSet01_Problems.pdf#page=3|the original printed page 3]] for the five graphs. Apply these rules to each one:
>
> **1.** **Semicircle:** For the upper semicircle $y=\sqrt{R^2-x^2}$, $y'=-x/\sqrt{R^2-x^2}$. The derivative is zero at the center, tends to $+\infty$ and $-\infty$ at the two endpoints, respectively, and is an odd function.<br>
> **2.** **Parabola:** The slope varies linearly with $x$; the derivative is zero at the vertex and has opposite signs on the two sides.<br>
> **3.** **Given odd function:** Its derivative should be even; reflect slope values across the $y$-axis.<br>
> **4.** **Given even function:** Its derivative should be odd; if it is differentiable at the origin, then $f'(0)=0$.<br>
> **5.** **Periodic function:** Its derivative has the same period. Local maxima and minima correspond to zeros of the derivative when the function is differentiable; the derivative is positive on increasing intervals and negative on decreasing intervals.<br>
>
> To check the sketch, first mark nondifferentiable points and zeros, then determine the sign, and finally estimate the magnitude. Do not confuse the function's height with the value of its derivative.
> <!-- bilingual-en:end -->

> [!example]- 1D-1：极限（原题 p.4）
> **(a)**
> $$\lim_{x\to0}\frac4{x-1}=\boxed{-4}.$$
> **(c)**
> $$\lim_{x\to-2}\frac{4x^2}{x+2}$$
> 左侧分母负且趋零，故为 $-\infty$；右侧为 $+\infty$。左右不同，双侧极限不存在（undefined）。
> **(d)** 当 $x\to2^+$，$2-x\to0^-$、分子趋 $16$，所以
> $$\lim_{x\to2^+}\frac{4x^2}{2-x}=\boxed{-\infty}.$$
> **(f)**
> $$
> \frac{4x^2}{x-2}=\frac{4x}{1-2/x}\to\boxed{+\infty}.
> $$
> **(g)** 先合并而不是做 $\infty-\infty$：
> $$
> \frac{4x^2}{x-2}-4x
> =\frac{4x^2-4x(x-2)}{x-2}
> =\frac{8x}{x-2}\to\boxed{8}.
> $$
> <!-- bilingual-en:start -->
> **(a)**
> $$\lim_{x\to0}\frac4{x-1}=\boxed{-4}.$$
> **(c)**
> $$\lim_{x\to-2}\frac{4x^2}{x+2}$$
> tends to $-\infty$ from the left and $+\infty$ from the right. Because the one-sided limits differ, the two-sided limit does not exist.
> **(d)** As $x\to2^+$, $2-x\to0^-$ while the numerator tends to $16$, so
> $$\lim_{x\to2^+}\frac{4x^2}{2-x}=\boxed{-\infty}.$$
> **(f)**
> $$
> \frac{4x^2}{x-2}=\frac{4x}{1-2/x}\to\boxed{+\infty}.
> $$
> **(g)** Combine the terms before evaluating rather than treating the expression as $\infty-\infty$:
> $$
> \frac{4x^2}{x-2}-4x
> =\frac{4x^2-4x(x-2)}{x-2}
> =\frac{8x}{x-2}\to\boxed{8}.
> $$
> <!-- bilingual-en:end -->

> [!example]- 1D-3：间断分类（原题 p.4）
> **(a)**
> $$\frac{x-2}{x^2-4}=\frac1{x+2}\quad(x\ne2).$$
> $x=2$ 为可去间断；$x=-2$ 为无穷间断。
>
> **(c)** $x^4/x^3=x$（$x\ne0$），所以 $x=0$ 是可去间断，极限为 $0$。
>
> **(d)**
> $$
> f(x)=\begin{cases}x+a,&x>0,\\a-x,&x<0.
> \end{cases}
> $$
> 左右极限都是 $a$，但 $f(0)$ 未定义，因此 $0$ 为可去间断。
>
> **(e)** 对 (d) 的函数，$f'(x)=1$（$x>0$），$f'(x)=-1$（$x<0$）；在 $0$ 左右极限不同，故导函数在 $0$ 是跳跃间断。
> <!-- bilingual-en:start -->
> **(a)**
> $$\frac{x-2}{x^2-4}=\frac1{x+2}\quad(x\ne2).$$
> The point $x=2$ is a removable discontinuity, while $x=-2$ is an infinite discontinuity.
>
> **(c)** Since $x^4/x^3=x$ for $x\ne0$, $x=0$ is a removable discontinuity and the limit there is $0$.
>
> **(d)**
> $$
> f(x)=\begin{cases}x+a,&x>0,\\a-x,&x<0.
> \end{cases}
> $$
> Both one-sided limits equal $a$, but $f(0)$ is undefined, so $0$ is a removable discontinuity.
>
> **(e)** For the function in part (d), $f'(x)=1$ for $x>0$ and $f'(x)=-1$ for $x<0$. The one-sided limits of the derivative at $0$ differ, so $f'$ has a jump discontinuity at $0$.
> <!-- bilingual-en:end -->

> [!example]- 1D-6a、1D-8a：分段可导（原题 pp.4-5）
> **1D-6a。**
> $$
> f(x)=\begin{cases}x^2+4x+1,&x\ge0,\\ax+b,&x<0.
> \end{cases}
> $$
> 连续要求 $b=1$；左右导数匹配要求 $a=4$。故
> $$\boxed{a=4,b=1}.$$
>
> **1D-8a。**
> $$
> f(x)=\begin{cases}ax+b,&x>0,\\\sin2x,&x\le0.
> \end{cases}
> $$
> 连续要求 $b=0$。左导数为 $2$，右导数为 $a$；题目要求连续但不可导，所以
> $$\boxed{b=0,\ a\ne2}.$$
> 常见错误是把“不可导”写成 $a=2$；那恰好会使函数可导。
> <!-- bilingual-en:start -->
> **1D-6a.**
> $$
> f(x)=\begin{cases}x^2+4x+1,&x\ge0,\\ax+b,&x<0.
> \end{cases}
> $$
> Continuity requires $b=1$, and matching the one-sided derivatives requires $a=4$. Hence
> $$\boxed{a=4,b=1}.$$
>
> **1D-8a.**
> $$
> f(x)=\begin{cases}ax+b,&x>0,\\\sin2x,&x\le0.
> \end{cases}
> $$
> Continuity requires $b=0$. The left derivative is $2$ and the right derivative is $a$. The problem asks for continuity without differentiability, so
> $$\boxed{b=0,\ a\ne2}.$$
> A common mistake is to set $a=2$ for the “not differentiable” condition; that value actually makes the function differentiable.
> <!-- bilingual-en:end -->

> [!example]- 1E：多项式、原函数与商（原题 pp.5-6）
> **1E-1a。**
> $$D(x^{10}+3x^5+2x^3+4)=\boxed{10x^9+15x^4+6x^2}.$$
> **1E-1c。** $D(x/2+\pi)=\boxed{1/2}$。
>
> **1E-2b。** 求 $x^6+5x^5+4x^3$ 的一个原函数，逐项把幂次加一再除以新幂次：
> $$\boxed{\frac{x^7}{7}+\frac56x^6+x^4+C}.$$
> 微分回去正好得到原多项式。
>
> **1E-3。** $y=x^3+x^2-x+2$ 的水平切线满足
> $$y'=3x^2+2x-1=(3x-1)(x+1)=0.$$
> 所以 $x=1/3,-1$；代回得
> $$\boxed{(1/3,49/27),\ (-1,3)}.$$
>
> **1E-4b。** 在 $x=1$ 拼接
> $$ax^2+bx+4\quad\text{与}\quad5x^5+3x^4+7x^2+8x+4.$$
> $$
> 连续条件 $a+b+4=27$，即 $a+b=23$；导数条件 $2a+b=59$。相减得
> $$\boxed{a=36,b=-13}.$$
>
> **1E-5a。**
> $$D\frac{x}{1+x}=\frac{1+x-x}{(1+x)^2}=\boxed{\frac1{(1+x)^2}},\quad x\ne-1.$$
> **1E-5c。**
> $$
> D\frac{x+2}{x^2-1}
> =\frac{x^2-1-2x(x+2)}{(x^2-1)^2}
> =\boxed{-\frac{x^2+4x+1}{(x^2-1)^2}},\quad x\ne\pm1.
> $$
> <!-- bilingual-en:start -->
> **1E-1a.**
> $$D(x^{10}+3x^5+2x^3+4)=\boxed{10x^9+15x^4+6x^2}.$$
> **1E-1c.** $D(x/2+\pi)=\boxed{1/2}$.
>
> **1E-2b.** To find an antiderivative of $x^6+5x^5+4x^3$, increase each exponent by one and divide by the new exponent:
> $$\boxed{\frac{x^7}{7}+\frac56x^6+x^4+C}.$$
> Differentiating this expression recovers the original polynomial exactly.
>
> **1E-3.** A horizontal tangent to $y=x^3+x^2-x+2$ must satisfy
> $$y'=3x^2+2x-1=(3x-1)(x+1)=0.$$
> Hence $x=1/3$ or $x=-1$; substituting back gives
> $$\boxed{(1/3,49/27),\ (-1,3)}.$$
>
> **1E-4b.** Join the two polynomial pieces at $x=1$:
> $$ax^2+bx+4\quad\text{and}\quad5x^5+3x^4+7x^2+8x+4.$$
> Continuity gives $a+b+4=27$, or $a+b=23$; matching derivatives gives $2a+b=59$. Subtracting yields
> $$\boxed{a=36,b=-13}.$$
>
> **1E-5a.**
> $$D\frac{x}{1+x}=\frac{1+x-x}{(1+x)^2}=\boxed{\frac1{(1+x)^2}},\quad x\ne-1.$$
> **1E-5c.**
> $$
> D\frac{x+2}{x^2-1}
> =\frac{x^2-1-2x(x+2)}{(x^2-1)^2}
> =\boxed{-\frac{x^2+4x+1}{(x^2-1)^2}},\quad x\ne\pm1.
> $$
> <!-- bilingual-en:end -->

> [!example]- 1F：链式法则与一般公式（原题 pp.6-7）
> **1F-1a。** 展开法：$(x^2+2)^2=x^4+4x^2+4$，导数 $4x^3+8x$。链式法：
> $$D(x^2+2)^2=2(x^2+2)(2x)=\boxed{4x(x^2+2)}.$$
> 两式相同。
>
> **1F-1b。** 展开一百次不可取；链式法直接给
> $$\boxed{D(x^2+2)^{100}=200x(x^2+2)^{99}}.$$
>
> **1F-2。** 对 $x^{10}(x^2+1)^{10}$ 用积法则：
> $$
> \begin{aligned}
> y'&=10x^9(x^2+1)^{10}+x^{10}\cdot10(x^2+1)^9(2x)\\
> &=\boxed{10x^9(x^2+1)^9(3x^2+1)}.
> \end{aligned}
> $$
>
> **1F-6。** 若 $f$ 偶，则 $f(-x)=f(x)$。两边对 $x$ 求导：
> $$-f'(-x)=f'(x),$$
> 即 $f'$ 为奇函数。若 $g$ 奇，$g(-x)=-g(x)$，求导：
> $$-g'(-x)=-g'(x)\Longrightarrow g'(-x)=g'(x),$$
> 故 $g'$ 为偶函数。链式因子 $d(-x)/dx=-1$ 不能漏。
>
> **1F-7b。**
> $$m=m_0(1-v^2/c^2)^{-1/2}.$$
> 把 $m_0,c$ 视为常数：
> $$
> \frac{dm}{dv}
> =m_0\left(-\frac12\right)(1-v^2/c^2)^{-3/2}\left(-\frac{2v}{c^2}\right)
> =\boxed{\frac{m_0v}{c^2(1-v^2/c^2)^{3/2}}}.
> $$
> 定义域要求 $|v|<c$。
>
> **1F-7c。**
> $$F=mg(1+r^2)^{-3/2}$$
> 中 $m,g$ 为常数，故
> $$
> \frac{dF}{dr}
> =mg\left(-\frac32\right)(1+r^2)^{-5/2}(2r)
> =\boxed{-\frac{3mgr}{(1+r^2)^{5/2}}}.
> $$
> <!-- bilingual-en:start -->
> **1F-1a.** Expanding $(x^2+2)^2=x^4+4x^2+4$ gives derivative $4x^3+8x$. The chain rule gives
> $$D(x^2+2)^2=2(x^2+2)(2x)=\boxed{4x(x^2+2)}.$$
> The two expressions are equal.
>
> **1F-1b.** Expanding a hundredth power is impractical; the chain rule gives directly
> $$\boxed{D(x^2+2)^{100}=200x(x^2+2)^{99}}.$$
>
> **1F-2.** Apply the product rule to $x^{10}(x^2+1)^{10}$:
> $$
> \begin{aligned}
> y'&=10x^9(x^2+1)^{10}+x^{10}\cdot10(x^2+1)^9(2x)\\
> &=\boxed{10x^9(x^2+1)^9(3x^2+1)}.
> \end{aligned}
> $$
>
> **1F-6.** If $f$ is even, then $f(-x)=f(x)$. Differentiating both sides with respect to $x$ gives
> $$-f'(-x)=f'(x),$$
> so $f'$ is odd. If $g$ is odd, then $g(-x)=-g(x)$; differentiating gives
> $$-g'(-x)=-g'(x)\Longrightarrow g'(-x)=g'(x),$$
> so $g'$ is even. The chain-rule factor $d(-x)/dx=-1$ must not be omitted.
>
> **1F-7b.**
> $$m=m_0(1-v^2/c^2)^{-1/2}.$$
> Treating $m_0$ and $c$ as constants,
> $$
> \frac{dm}{dv}
> =m_0\left(-\frac12\right)(1-v^2/c^2)^{-3/2}\left(-\frac{2v}{c^2}\right)
> =\boxed{\frac{m_0v}{c^2(1-v^2/c^2)^{3/2}}}.
> $$
> The domain requires $|v|<c$.
>
> **1F-7c.** In
> $$F=mg(1+r^2)^{-3/2},$$
> $m$ and $g$ are constants, so
> $$
> \frac{dF}{dr}
> =mg\left(-\frac32\right)(1+r^2)^{-5/2}(2r)
> =\boxed{-\frac{3mgr}{(1+r^2)^{5/2}}}.
> $$
> <!-- bilingual-en:end -->

> [!example]- 1J：三角求导与导数型极限（原题 p.11）
> **1J-1e。** 对 $f(x)=\sin x/x$（$x\ne0$）：
> $$
> f'(x)=\frac{x\cos x-\sin x}{x^2}.
> $$
> 规则：商法则；检查：原函数为偶函数，结果为奇函数。
>
> **1J-2。** 因 $\cos(\pi/2)=0$，原极限正是导数定义：
> $$
> \lim_{x\to\pi/2}\frac{\cos x}{x-\pi/2}
> =\left.\frac d{dx}\cos x\right|_{x=\pi/2}
> =-\sin(\pi/2)=\boxed{-1}.
> $$
>
> **1J-1a。**
> $$D\sin(5x^2)=\boxed{10x\cos(5x^2)}.$$
> **1J-1k。**
> $$D\tan^2(3x)=2\tan(3x)\sec^2(3x)\cdot3
> =\boxed{6\tan(3x)\sec^2(3x)}.$$
> **1J-1m。** 三个函数：
> $$
> \begin{aligned}
> D\cos2x&=-2\sin2x=-4\sin x\cos x,\\
> D(\cos^2x-\sin^2x)&=-4\sin x\cos x,\\
> D(2\cos^2x)&=-4\sin x\cos x.
> \end{aligned}
> $$
> 前两个恒等；第三个比它们大 $1$：
> $$\cos2x=\cos^2x-\sin^2x=2\cos^2x-1.$$
> 相同导数只说明函数相差常数，不说明函数完全相等。
> <!-- bilingual-en:start -->
> **1J-1e.** For $f(x)=\sin x/x$ with $x\ne0$,
> $$
> f'(x)=\frac{x\cos x-\sin x}{x^2}.
> $$
> This uses the quotient rule. As a check, the original function is even and the derivative is odd.
>
> **1J-2.** Since $\cos(\pi/2)=0$, the given limit is exactly the derivative of cosine at $\pi/2$:
> $$
> \lim_{x\to\pi/2}\frac{\cos x}{x-\pi/2}
> =\left.\frac d{dx}\cos x\right|_{x=\pi/2}
> =-\sin(\pi/2)=\boxed{-1}.
> $$
>
> **1J-1a.**
> $$D\sin(5x^2)=\boxed{10x\cos(5x^2)}.$$
> **1J-1k.**
> $$D\tan^2(3x)=2\tan(3x)\sec^2(3x)\cdot3
> =\boxed{6\tan(3x)\sec^2(3x)}.$$
> **1J-1m.** For the three functions,
> $$
> \begin{aligned}
> D\cos2x&=-2\sin2x=-4\sin x\cos x,\\
> D(\cos^2x-\sin^2x)&=-4\sin x\cos x,\\
> D(2\cos^2x)&=-4\sin x\cos x.
> \end{aligned}
> $$
> The first two functions are identical, while the third is larger by $1$:
> $$\cos2x=\cos^2x-\sin^2x=2\cos^2x-1.$$
> Equal derivatives show only that functions differ by a constant; they do not prove that the functions themselves are identical.
> <!-- bilingual-en:end -->

> [!example]- 1G：高阶导数（原题 p.7）
> **1G-1b。** $y=x/(x+5)=1-5/(x+5)$：
> $$y'=\frac5{(x+5)^2},\qquad
> \boxed{y''=-\frac{10}{(x+5)^3}}.
> $$
> **1G-1c。** $y=-5/(x+5)$：
> $$y'=\frac5{(x+5)^2},\qquad
> \boxed{y''=-\frac{10}{(x+5)^3}}.
> $$
> 两个原函数相差常数 $1$，所以一阶及更高导数相同。
>
> **1G-5a。** 若 $y=uv$，逐次使用积法则：
> $$
> \begin{aligned}
> y'&=u'v+uv',\\
> y''&=u''v+2u'v'+uv'',\\
> y'''&=u'''v+3u''v'+3u'v''+uv'''.
> \end{aligned}
> $$
> 系数 $1,3,3,1$ 是二项式系数；每一阶都应检查所有导数分配方式。
> <!-- bilingual-en:start -->
> **1G-1b.** For $y=x/(x+5)=1-5/(x+5)$,
> $$y'=\frac5{(x+5)^2},\qquad
> \boxed{y''=-\frac{10}{(x+5)^3}}.
> $$
> **1G-1c.** For $y=-5/(x+5)$,
> $$y'=\frac5{(x+5)^2},\qquad
> \boxed{y''=-\frac{10}{(x+5)^3}}.
> $$
> The two original functions differ by the constant $1$, so their first and all higher derivatives agree.
>
> **1G-5a.** If $y=uv$, repeated use of the product rule gives
> $$
> \begin{aligned}
> y'&=u'v+uv',\\
> y''&=u''v+2u'v'+uv'',\\
> y^{(3)}&=u^{(3)}v+3u''v'+3u'v''+uv^{(3)}.
> \end{aligned}
> $$
> The coefficients $1,3,3,1$ are binomial coefficients; at each order, check that every possible distribution of derivatives has been included.
> <!-- bilingual-en:end -->

> [!warning] Problem Set 1 常见错误
> - 图像变换只写答案而不交代平移/尺度；分段题只配函数值不配左右导数。
> - “极限为 $\infty$”与左右分别为 $\pm\infty$ 混为一谈。
> - 求切线只算斜率，漏掉切点；问总路程却只算净位移。
> - 链式法则漏内层导数；商法则负号次序写反；奇偶性证明漏掉 $d(-x)/dx=-1$。
> <!-- bilingual-en:start -->
> - Stating the transformed graph without explaining its translation and scaling; matching function values in a piecewise problem without matching one-sided derivatives.
> - Confusing a limit that tends to $+\infty$ with one whose two sides tend to opposite infinities.
> - Computing only a tangent slope and omitting the point of tangency; reporting net displacement when total distance is requested.
> - Omitting the inner derivative in the chain rule; reversing the signs in the quotient rule; forgetting $d(-x)/dx=-1$ in a parity proof.
> <!-- bilingual-en:end -->

**Problem Set 1 小结：**这些题把 Part A 的三层能力连在一起：先读图和定义域，再选择规则，最后用极限、奇偶性、单位或函数值回代检查。
<!-- bilingual-en:start -->
**Problem Set 1 summary:** These questions connect three levels of skill in Part A: first read the graph and domain, then choose the appropriate rules, and finally check the result using limits, parity, units, or function values.
<!-- bilingual-en:end -->

---

## Part B：Implicit Differentiation and Inverse Functions

## Session 13：Implicit Differentiation

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题：**当 $y$ 没有方便地单独解出时，怎样求 $dy/dx$？正整数幂法则如何扩展到有理指数？
<!-- bilingual-en:start -->
**Question:** How can we find $dy/dx$ when solving explicitly for $y$ is inconvenient, and how does the positive-integer power rule extend to rational exponents?
<!-- bilingual-en:end -->

**前置知识：**链式法则、整数幂法则、指数运算、局部把 $y$ 看成 $y(x)$。
<!-- bilingual-en:start -->
**Prerequisites:** the chain rule, the integer power rule, exponentiation, and treating $y$ locally as a function $y(x)$.
<!-- bilingual-en:end -->

### 13a：隐函数求导的思想
<!-- bilingual-en:start -->
*13a: Implicit Differentiation*
<!-- bilingual-en:end -->

[[导数与求导规则#Worked example：用隐式求导找圆的切线|隐函数求导]]从关系式
<!-- bilingual-en:start -->
[[导数与求导规则#Worked example：用隐式求导找圆的切线|Implicit differentiation]] starts from a relation such as
<!-- bilingual-en:end -->

$$
F(x,y)=0
$$

可能描述多条分支，甚至不能在整个图形上写成单一函数 $y=f(x)$。但在非竖直切线附近，一小段曲线通常仍可把 $y$ 看成由 $x$ 决定。于是对等式两边关于 $x$ 求导：
<!-- bilingual-en:start -->
Such a relation may describe several branches and may not define a single global function $y=f(x)$. Near a point with a nonvertical tangent, however, a small segment can usually be regarded as having $y$ determined by $x$. We therefore differentiate both sides with respect to $x$:
<!-- bilingual-en:end -->

$$
\frac d{dx}F(x,y(x))=0.
$$

关键规则：每当对含 $y$ 的表达式求导，都要乘 $y'=dy/dx$。例如
<!-- bilingual-en:start -->
Key rule: whenever an expression involving $y$ is differentiated, multiply by $y'=dy/dx$. For example,
<!-- bilingual-en:end -->

$$
\frac d{dx}y^n=ny^{n-1}y'.
$$

这不是“额外规则”，正是外函数 $u^n$ 与内函数 $u=y(x)$ 的链式法则。
<!-- bilingual-en:start -->
This is not an "extra rule", but the chain rule of the outer function $u^n$ and the inner function $u=y(x)$.
<!-- bilingual-en:end -->

### 13b：有理指数幂法则的完整推导
<!-- bilingual-en:start -->
*13b: A Complete Proof of the Rational-Exponent Power Rule*
<!-- bilingual-en:end -->

设
<!-- bilingual-en:start -->
Let
<!-- bilingual-en:end -->

$$
y=x^{m/n},
$$

其中 $m\in\mathbb Z,n\in\mathbb N$。为消去分数指数，两边取 $n$ 次幂：
<!-- bilingual-en:start -->
where $m\in\mathbb Z$ and $n\in\mathbb N$. To eliminate the fractional exponent, raise both sides to the $n$th power:
<!-- bilingual-en:end -->

$$
y^n=x^m.
$$

关于 $x$ 求导：
<!-- bilingual-en:start -->
Differentiate with respect to $x$:
<!-- bilingual-en:end -->

$$
ny^{n-1}y'=mx^{m-1}.
$$

在 $y\ne0$ 处解出：
<!-- bilingual-en:start -->
At points where $y\ne0$, solve for $y'$:
<!-- bilingual-en:end -->

$$
y'=\frac{m}{n}\frac{x^{m-1}}{y^{n-1}}.
$$

代回 $y=x^{m/n}$：
<!-- bilingual-en:start -->
Substitute $y=x^{m/n}$ back into the result:
<!-- bilingual-en:end -->

$$
\begin{aligned}
y'
&=\frac mn x^{m-1-(m/n)(n-1)}\\
&=\frac mn x^{m/n-1}.
\end{aligned}
$$

> [!important] 有理指数幂法则
> 在函数有实数定义且上述表达式有意义的区间内，
> $$
> \boxed{\frac d{dx}x^r=rx^{r-1}},\qquad r\in\mathbb Q.
> $$
> <!-- bilingual-en:start -->
> On any interval where the function is real-valued and the displayed expression is defined,
> <!-- bilingual-en:end -->

### 定义域和零点必须另查
<!-- bilingual-en:start -->
*The Domain and Zeros Must Be Checked Separately*
<!-- bilingual-en:end -->

代数推导中除以了 $y^{n-1}$，所以 $y=0$ 处不能仅凭该步骤断言。还要考虑：
<!-- bilingual-en:start -->
The algebraic derivation divides by $y^{n-1}$, so it says nothing by itself about points where $y=0$. Also consider the following:
<!-- bilingual-en:end -->

- $n$ 为偶数时，实函数 $x^{m/n}$ 通常只在 $x\ge0$ 定义；
- 同一个数可有不同分数表示，如 $x^{2/6}=x^{1/3}$，实数幂的定义要先约分；
- $r<1$ 时公式常在 $x=0$ 发散。例如 $\sqrt{x}$ 在 $0$ 有竖直切线而无有限双侧导数；
- $r>1$ 时零点可能可导，例如 $(x^{3/2})'=(3/2)\sqrt{x}$ 在右端点为零，但这里讨论的是单侧导数。
<!-- bilingual-en:start -->
- If $n$ is even, the real-valued function $x^{m/n}$ is generally defined only for $x\ge0$.
- The same rational number has different fractional representations, such as $2/6=1/3$, so reduce the exponent before defining a real power.
- When $r<1$, the derivative formula often diverges at $x=0$. For example, $\sqrt{x}$ has a vertical tangent at $0$ rather than a finite two-sided derivative.
- When $r>1$, differentiability at zero may be possible. For example, $(x^{3/2})'=(3/2)\sqrt{x}$ is zero at the right endpoint, although only a one-sided derivative is relevant there.
<!-- bilingual-en:end -->

### Exercise 013：隐函数二阶导数
<!-- bilingual-en:start -->
*Exercise 013: The Second Derivative of an Implicit Function*
<!-- bilingual-en:end -->

由
<!-- bilingual-en:start -->
Starting from
<!-- bilingual-en:end -->

$$
x^2+4y^2=1
$$

第一次求导：
<!-- bilingual-en:start -->
Differentiate once:
<!-- bilingual-en:end -->

$$
2x+8yy'=0
\quad\Longrightarrow\quad
y'=-\frac{x}{4y}.
$$

再次求导，使用商法则：
<!-- bilingual-en:start -->
Differentiate again using the quotient rule:
<!-- bilingual-en:end -->

$$
\begin{aligned}
y''
&=-\frac14\frac{y-xy'}{y^2}\\
&=-\frac14\frac{y+x^2/(4y)}{y^2}\\
&=-\frac{4y^2+x^2}{16y^3}.
\end{aligned}
$$

原曲线给 $x^2+4y^2=1$，故
<!-- bilingual-en:start -->
The original relation gives $x^2+4y^2=1$, so
<!-- bilingual-en:end -->

$$
\boxed{y''=-\frac1{16y^3}}.
$$

当 $y=0$ 时第一次导数公式已无定义，对应椭圆左右端点的竖直切线；二阶公式也不适用。
<!-- bilingual-en:start -->
When $y=0$, the first-derivative formula is undefined, corresponding to the vertical tangents at the left and right endpoints of the ellipse. The second-derivative formula is therefore inapplicable there as well.
<!-- bilingual-en:end -->

### 边界情况与易错点
<!-- bilingual-en:start -->
*Boundary Cases and Common Pitfalls*
<!-- bilingual-en:end -->

- 隐式求导不是把 $y$ 当常数；$D(y^4)=4y^3y'$。
- 先求一般公式再代点，能避免过早把变量变成常数。
- 除以含 $x,y$ 的因子后，要记录该因子为零的点并单独分析。
- $F(x,y)=0$ 在某点未必真能局部表示为 $y(x)$；若 $F_y=0$，公式 $y'=-F_x/F_y$ 失效，常对应竖直切线或更复杂奇点。
<!-- bilingual-en:start -->
- Implicit differentiation does not treat $y$ as a constant: $D(y^4)=4y^3y'$.
- Derive a general formula before substituting the point, so variables are not turned into constants prematurely.
- After dividing by a factor involving $x$ or $y$, record the points where that factor is zero and analyse them separately.
- $F(x,y)=0$ may not be locally expressed as $y(x)$ at some point; if $F_y=0$, the formula $y'=-F_x/F_y$ fails, often corresponding to a vertical tangent or more complex singular point.
<!-- bilingual-en:end -->

### 三道自检
<!-- bilingual-en:start -->
*Three Quick Self-Checks*
<!-- bilingual-en:end -->

1. 对 $x^2+y^2=25$ 求 $y'$，并求 $(3,4)$ 处切线。
2. 从 $y=x^{2/3}$ 出发，用 $y^3=x^2$ 推导导数，并讨论 $x=0$。
3. 若 $F(x,y)=0$，形式上为什么 $y'=-F_x/F_y$？
<!-- bilingual-en:start -->

&nbsp;
**1.** Find $y'$ for $x^2+y^2=25$ and the tangent line at $(3,4)$.<br>
**2.** Starting from $y=x^{2/3}$, derive the derivative using $y^3=x^2$ and discuss what happens at $x=0$.<br>
**3.** If $F(x,y)=0$, why does the formal calculation give $y'=-F_x/F_y$?<br>
<!-- bilingual-en:end -->

> [!success]- 自检答案
> 1. $2x+2yy'=0$，$y'=-x/y$；在 $(3,4)$ 斜率 $-3/4$，切线 $y-4=-\frac34(x-3)$。
> 2. $3y^2y'=2x$，$y'=2x/(3y^2)=\frac23x^{-1/3}$（$x\ne0$）。在 $0$ 差商为 $|h|^{2/3}/h$，大小趋无穷且左右符号相反，形成尖点/竖直切线，不可导。
> 3. 链式法则给 $F_x+F_y y'=0$；若 $F_y\ne0$，解得 $y'=-F_x/F_y$。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** $2x+2yy'=0$, $y'=-x/y$; at $(3,4)$ slope $-3/4$, tangent $y-4=-\frac34(x-3)$.<br>
> **2.** $3y^2y'=2x$, so $y'=2x/(3y^2)=\frac23x^{-1/3}$ for $x\ne0$. At $x=0$, the difference quotient is $|h|^{2/3}/h$; its magnitude diverges and its sign differs on the two sides, producing a cusp or vertical tangent, so the function is not differentiable there.<br>
> **3.** The chain rule gives $F_x+F_y y'=0$; if $F_y\ne0$, solving yields $y'=-F_x/F_y$.<br>
> <!-- bilingual-en:end -->

### 本地材料

- [[Ses13a_Lecture_Notes.pdf#page=1|13a Introduction to Implicit Differentiation（p.1）]]
- [[Ses13b_Lecture_Notes.pdf#page=1|13b Rational Exponent Rule（pp.1-3）]]
- [[Exercise013_Problems.pdf#page=1|Exercise 013：Implicit Differentiation and Second Derivative]] · [[Exercise013_Solutions.pdf#page=1|答案]]

**知识链：**把 $y$ 看作 $y(x)$ → 链式法则产生 $y'$ → 解出斜率 → 用同一方法证明有理幂法则。
<!-- bilingual-en:start -->
**Knowledge chain:** treat $y$ as $y(x)$ → the chain rule produces $y'$ → solve for the slope → use the same method to prove the rational-power rule.
<!-- bilingual-en:end -->

## Session 14：Examples of Implicit Differentiation

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题：**隐式法何时比先解出 $y$ 更短？如何处理同时含 $x,y$ 的积？
<!-- bilingual-en:start -->
**Question:** When is implicit differentiation shorter than first solving for $y$, and how should products involving both $x$ and $y$ be handled?
<!-- bilingual-en:end -->

**前置知识：**链式、积法则、分支与定义域。
<!-- bilingual-en:start -->
**Prerequisites:** the chain rule, product rule, branches, and domains.
<!-- bilingual-en:end -->

### 14a：圆的直接法
<!-- bilingual-en:start -->
*14a: The Direct Method for a Circle*
<!-- bilingual-en:end -->

单位圆
<!-- bilingual-en:start -->
The unit circle
<!-- bilingual-en:end -->

$$
x^2+y^2=1
$$

不是全局的 $y=f(x)$，因为同一 $x\in(-1,1)$ 对应两个 $y$。若只取上半圆，
<!-- bilingual-en:start -->
does not define a single global function $y=f(x)$, because each $x\in(-1,1)$ corresponds to two values of $y$. If we restrict attention to the upper semicircle,
<!-- bilingual-en:end -->

$$
y=\sqrt{1-x^2}=(1-x^2)^{1/2}.
$$

链式法则给
<!-- bilingual-en:start -->
the chain rule gives
<!-- bilingual-en:end -->

$$
y'=\frac12(1-x^2)^{-1/2}(-2x)
=-\frac{x}{\sqrt{1-x^2}}
=-\frac xy.
$$

这个直接法必须先选择上支；下支要另算。
<!-- bilingual-en:start -->
This direct method must first select the upper branch; the lower branch must be calculated separately.
<!-- bilingual-en:end -->

### 14b：圆的隐式法
<!-- bilingual-en:start -->
*14b: Implicit Method for Circles*
<!-- bilingual-en:end -->

直接对原关系求导：
<!-- bilingual-en:start -->
Differentiate the original relation directly:
<!-- bilingual-en:end -->

$$
2x+2yy'=0
\quad\Longrightarrow\quad
\boxed{y'=-\frac xy}.
$$

它同时覆盖上下半圆。几何检查：半径向量 $(x,y)$ 与切向量 $(1,y')$ 的点积
<!-- bilingual-en:start -->
This single calculation covers both semicircles. As a geometric check, take the dot product of the radial vector $(x,y)$ with the tangent vector $(1,y')$:
<!-- bilingual-en:end -->

$$
(x,y)\cdot(1,-x/y)=x-x=0,
$$

因此切线确实垂直于半径。$y=0$ 时公式分母为零，对应 $(\pm1,0)$ 的竖直切线。
<!-- bilingual-en:start -->
Thus the tangent is perpendicular to the radius. When $y=0$, the formula's denominator vanishes, corresponding to the vertical tangents at $(\pm1,0)$.
<!-- bilingual-en:end -->

### 14c：课件原例 $y^4+xy^2-2=0$
<!-- bilingual-en:start -->
*14c: Lecture Example $y^4+xy^2-2=0$*
<!-- bilingual-en:end -->

这道题不能误记成别的三次曲线。逐项求导：
<!-- bilingual-en:start -->
Do not confuse this equation with a different cubic curve. Differentiate it term by term:
<!-- bilingual-en:end -->

$$
\frac d{dx}y^4+\frac d{dx}(xy^2)-\frac d{dx}2=0.
$$

第一项用链式法则，第二项同时用积法则和链式法则：
<!-- bilingual-en:start -->
The first term uses the chain rule; the second uses both the product and chain rules:
<!-- bilingual-en:end -->

$$
4y^3y'+\left(y^2+x\cdot2yy'\right)=0.
$$

收集所有含 $y'$ 的项：
<!-- bilingual-en:start -->
Collect all items with $y'$:
<!-- bilingual-en:end -->

$$
(4y^3+2xy)y'=-y^2.
$$

所以
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

$$
\boxed{y'=-\frac{y^2}{4y^3+2xy}}
=-\frac{y}{4y^2+2x}\quad(y\ne0\text{ 时}).
$$

保留未约分形式可更清楚看出从哪一步除以了什么；约分后必须补记 $y=0$ 的排除情况，而原曲线在 $y=0$ 时给 $-2=0$，实际没有这样的点，所以此处约分安全。
<!-- bilingual-en:start -->
Keeping the uncancelled form makes each division step visible. After cancellation, the excluded case $y=0$ must be recorded. Here the original curve would give $-2=0$ when $y=0$, so no such point exists and the cancellation is safe.
<!-- bilingual-en:end -->

### Exercise 014：双曲线分支
<!-- bilingual-en:start -->
*Exercise 014: Branches of a Hyperbola*
<!-- bilingual-en:end -->

$$
y^2-x^2=1.
$$

隐式求导：
<!-- bilingual-en:start -->
Implicit differentiation gives
<!-- bilingual-en:end -->

$$
2yy'-2x=0
\quad\Longrightarrow\quad
\boxed{y'=\frac xy}.
$$

当 $y=-1$ 时曲线上只有 $x=0$，斜率 $0$；当 $x=1$ 时 $y=\pm\sqrt2$，两支斜率分别 $\pm1/\sqrt2$。对上支直接写 $y=\sqrt{x^2+1}$，得到 $y'=x/\sqrt{x^2+1}=x/y$，与隐式法一致。
<!-- bilingual-en:start -->
When $y=-1$, the only point on the curve has $x=0$, and its slope is $0$. When $x=1$, we have $y=\pm\sqrt2$, giving slopes $\pm1/\sqrt2$ on the two branches. On the upper branch, writing $y=\sqrt{x^2+1}$ directly gives $y'=x/\sqrt{x^2+1}=x/y$, in agreement with implicit differentiation.
<!-- bilingual-en:end -->

### 通用工作流
<!-- bilingual-en:start -->
*General Workflow*
<!-- bilingual-en:end -->

1. 写清每一项使用和、积、链式中的哪一条规则；
2. 所有含 $y'$ 的项移到同一侧；
3. 提取 $y'$；
4. 除以前先记录可能为零的因子；
5. 最后才代入指定点并写切线。
<!-- bilingual-en:start -->

&nbsp;
**1.** State which sum, product, or chain rule is used for each term.<br>
**2.** Move every term containing $y'$ to the same side.<br>
**3.** Factor out $y'$.<br>
**4.** Before dividing, record any factor that could be zero.<br>
**5.** Only then substitute the specified point and write the tangent line.<br>
<!-- bilingual-en:end -->

### 边界情况与易错点
<!-- bilingual-en:start -->
*Boundary Cases and Common Pitfalls*
<!-- bilingual-en:end -->

- $D(xy^2)=y^2+2xyy'$，漏掉任一项都错。
- 一个隐式方程可有多个分支；同一个 $x$ 上不同 $y$ 可能给不同斜率。
- $y'=0$ 表示水平切线；公式分母为零且分子非零常表示竖直切线。
- 若分子、分母同时为零，不能直接判断，可能是交叉点、尖点或更高阶接触。
<!-- bilingual-en:start -->
- $D(xy^2)=y^2+2xyy'$; omitting either term is an error.
- An implicit equation may have several branches, and different values of $y$ at the same $x$ may give different slopes.
- $y'=0$ indicates a horizontal tangent. A zero denominator with a nonzero numerator often indicates a vertical tangent.
- If both numerator and denominator vanish, no immediate conclusion is possible; the point may be a crossing, a cusp, or a higher-order contact.
<!-- bilingual-en:end -->

### 三道自检
<!-- bilingual-en:start -->
*Three Quick Self-Checks*
<!-- bilingual-en:end -->

1. 对 $x^3+y^3=6xy$ 求 $y'$。
2. 求 $x^2+xy+y^2=7$ 在 $(1,2)$ 的切线。
3. 对 $y^4+xy^2-2=0$，说明为何不能把 $D(xy^2)$ 写成 $2xyy'$。
<!-- bilingual-en:start -->

&nbsp;
**1.** Find $y'$ for $x^3+y^3=6xy$.<br>
**2.** Find the tangent line to $x^2+xy+y^2=7$ at $(1,2)$.<br>
**3.** For $y^4+xy^2-2=0$, explain why $D(xy^2)$ cannot be written as $2xyy'$.<br>
<!-- bilingual-en:end -->

> [!success]- 自检答案
> 1. $3x^2+3y^2y'=6y+6xy'$，故 $y'=(2y-x^2)/(y^2-2x)$。
> 2. $2x+y+xy'+2yy'=0$，$y'=-(2x+y)/(x+2y)$；在 $(1,2)$ 为 $-4/5$，切线 $y-2=-\frac45(x-1)$。
> 3. $x$ 与 $y^2$ 都随 $x$ 变化，积法则给 $x' y^2+x(y^2)'=y^2+2xyy'$；漏掉 $y^2$ 等于错误地把 $x$ 当常数。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** $3x^2+3y^2y'=6y+6xy'$, hence $y'=(2y-x^2)/(y^2-2x)$.<br>
> **2.** $2x+y+xy'+2yy'=0$, $y'=-(2x+y)/(x+2y)$; $-4/5$ at $(1,2)$, tangent $y-2=-\frac45(x-1)$.<br>
> **3.** Both $x$ and $y^2$ vary with $x$, so the product rule gives $x'y^2+x(y^2)'=y^2+2xyy'$. Omitting the $y^2$ term amounts to treating $x$ as a constant.<br>
> <!-- bilingual-en:end -->

### 本地材料

- [[Ses14a_Lecture_Notes.pdf#page=1|14a Circle - Direct Version（p.1）]]
- [[Ses14b_Lecture_Notes.pdf#page=1|14b Circle - Implicit Version（p.1）]]
- [[Ses14c_Lecture_Notes.pdf#page=1|14c $y^4+xy^2-2=0$（p.1）]]
- [[Exercise014_Problems.pdf#page=1|Exercise 014：Implicit Differentiation and Chain Rule]] · [[Exercise014_Solutions.pdf#page=1|答案]]

**知识链：**显式分支的繁琐 → 直接对关系求导 → 积与链式法则同时出现 → 一次覆盖多分支。
<!-- bilingual-en:start -->
**Knowledge chain:** cumbersome explicit branches → differentiate the relation directly → product and chain rules appear together → one calculation covers multiple branches.
<!-- bilingual-en:end -->

## Session 15：Implicit Differentiation and Inverse Functions

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题：**函数与反函数的斜率为何互为倒数？如何由此推导反正切、反正弦的导数？
<!-- bilingual-en:start -->
**Question:** Why are the slopes of a function and its inverse reciprocal, and how does this yield the derivatives of arctangent and arcsine?
<!-- bilingual-en:end -->

**前置知识：**一一对应、函数复合、隐函数和链式法则、三角恒等式。
<!-- bilingual-en:start -->
**Prerequisites:** one-to-one functions, composition, implicit differentiation, the chain rule, and trigonometric identities.
<!-- bilingual-en:end -->

### 15a：反函数导数定理
<!-- bilingual-en:start -->
*15a: Inverse Derivative Theorem*
<!-- bilingual-en:end -->

若 $g=f^{-1}$，则
<!-- bilingual-en:start -->
If $g=f^{-1}$,
<!-- bilingual-en:end -->

$$
f(g(x))=x.
$$

两边求导：
<!-- bilingual-en:start -->
Differentiate both sides:
<!-- bilingual-en:end -->

$$
f'(g(x))g'(x)=1.
$$

只要 $f'(g(x))\ne0$，
<!-- bilingual-en:start -->
As long as $f'(g(x))\ne0$,
<!-- bilingual-en:end -->

> [!important] [[导数与求导规则#求导规则为何成立|反函数导数]]
> $$
> \boxed{(f^{-1})'(x)=\frac1{f'(f^{-1}(x))}}.
> $$

若写对应点 $y_0=f(x_0)$，则
<!-- bilingual-en:start -->
Writing the corresponding point as $y_0=f(x_0)$ gives
<!-- bilingual-en:end -->

$$
(f^{-1})'(y_0)=\frac1{f'(x_0)}.
$$

图像上交换坐标 $(x_0,y_0)\leftrightarrow(y_0,x_0)$，即关于 $y=x$ 反射；切线的 rise/run 也交换，所以斜率取倒数。
<!-- bilingual-en:start -->
Reflecting a graph across $y=x$ exchanges $(x_0,y_0)$ with $(y_0,x_0)$. The tangent's rise and run are exchanged as well, so its slope is reciprocated.
<!-- bilingual-en:end -->

![[98_attachment/MIT18.01SC/unit01-inverse-reflection.png|900]]

**假设不能省略：**$f$ 必须在相关区间一一对应，反函数才存在；且 $f'(x_0)\ne0$，否则倒数公式分母为零，反函数可能出现竖直切线。
<!-- bilingual-en:start -->
**The assumptions matter:** $f$ must be one-to-one on the relevant interval for an inverse to exist, and $f'(x_0)\ne0$ so that the reciprocal formula has a nonzero denominator. Otherwise the inverse may have a vertical tangent.
<!-- bilingual-en:end -->

### 15b：$\arctan x$

令
<!-- bilingual-en:start -->
Let
<!-- bilingual-en:end -->

$$
y=\arctan x,
$$

按主值范围 $y\in(-\pi/2,\pi/2)$，等价于 $\tan y=x$。求导：
<!-- bilingual-en:start -->
On the principal-value range $y\in(-\pi/2,\pi/2)$, this is equivalent to $\tan y=x$. Differentiate:
<!-- bilingual-en:end -->

$$
\sec^2y\,y'=1
\quad\Longrightarrow\quad
y'=\frac1{\sec^2y}.
$$

用 $\sec^2y=1+\tan^2y=1+x^2$：
<!-- bilingual-en:start -->
With $\sec^2y=1+\tan^2y=1+x^2$:
<!-- bilingual-en:end -->

$$
\boxed{\frac d{dx}\arctan x=\frac1{1+x^2}},\qquad x\in\mathbb R.
$$

导数恒正，与 $\arctan x$ 单调增加一致；当 $|x|\to\infty$，导数趋零，与水平渐近线 $y=\pm\pi/2$ 一致。
<!-- bilingual-en:start -->
The derivative is positive everywhere, consistent with the fact that $\arctan x$ is strictly increasing. As $|x|\to\infty$, the derivative tends to zero, consistent with the horizontal asymptotes $y=\pm\pi/2$.
<!-- bilingual-en:end -->

### 15c：$\arcsin x$

令
<!-- bilingual-en:start -->
Let
<!-- bilingual-en:end -->

$$
y=\arcsin x,
$$

主值范围 $y\in[-\pi/2,\pi/2]$，等价于 $\sin y=x$。求导：
<!-- bilingual-en:start -->
The principal-value range is $y\in[-\pi/2,\pi/2]$, and the equation is equivalent to $\sin y=x$. Differentiate:
<!-- bilingual-en:end -->

$$
\cos y\,y'=1
\quad\Longrightarrow\quad
y'=\frac1{\cos y}.
$$

因主值范围内 $\cos y\ge0$，
<!-- bilingual-en:start -->
Because $\cos y\ge0$ on the principal-value range,
<!-- bilingual-en:end -->

$$
\cos y=\sqrt{1-\sin^2y}=\sqrt{1-x^2}.
$$

所以
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

$$
\boxed{\frac d{dx}\arcsin x=\frac1{\sqrt{1-x^2}}},\qquad |x|<1.
$$

在 $x=\pm1$ 分母为零，反正弦图像有竖直切线；函数在端点连续，但没有有限导数。
<!-- bilingual-en:start -->
At $x=\pm1$ the denominator is zero, so the graph of $\arcsin x$ has vertical tangents. The function remains continuous at the endpoints, but has no finite derivative there.
<!-- bilingual-en:end -->

### Exercise 015：平方根是平方函数的反函数
<!-- bilingual-en:start -->
*Exercise 015: The Square Root as the Inverse of a Squaring Function*
<!-- bilingual-en:end -->

限制 $f(x)=x^2$ 的定义域为 $x>0$，使其一一对应。若 $y=f^{-1}(x)$，则 $y^2=x$。隐式求导：
<!-- bilingual-en:start -->
Restrict the domain of $f(x)=x^2$ to $x>0$ so that it is one-to-one. If $y=f^{-1}(x)$, then $y^2=x$. Implicit differentiation gives
<!-- bilingual-en:end -->

$$
2yy'=1
\quad\Longrightarrow\quad
y'=\frac1{2y}=\boxed{\frac1{2\sqrt x}}.
$$

直接写 $y=\sqrt x=x^{1/2}$ 用有理幂法则得到同一结果。
<!-- bilingual-en:start -->
Writing $y=\sqrt x=x^{1/2}$ directly gives the same result by the rational-exponent power rule.
<!-- bilingual-en:end -->

### 边界情况与易错点
<!-- bilingual-en:start -->
*Boundary Cases and Common Pitfalls*
<!-- bilingual-en:end -->

- $(f^{-1})'(x)$ 不是 $1/f'(x)$；分母应在 $f^{-1}(x)$ 处评价。
- $f^{-1}$ 表示反函数，不表示倒数 $1/f$。
- 推导反三角函数必须声明主值范围，平方根符号才能确定。
- 反函数存在需要一一对应；$x^2$ 在整个实数轴上没有函数意义的全局反函数。
<!-- bilingual-en:start -->
- $(f^{-1})'(x)$ is not $1/f'(x)$; the denominator must be evaluated at $f^{-1}(x)$.
- $f^{-1}$ denotes an inverse function, not the reciprocal $1/f$.
- When deriving an inverse-trigonometric derivative, state the principal-value range so the sign of the square root is determined.
- An inverse function requires a one-to-one restriction; $x^2$ has no global inverse function on the entire real line.
<!-- bilingual-en:end -->

### 三道自检
<!-- bilingual-en:start -->
*Three Quick Self-Checks*
<!-- bilingual-en:end -->

1. 若 $f(2)=5,f'(2)=-3$，求 $(f^{-1})'(5)$。
2. 推导 $(\arccos x)'$，注意主值范围。
3. 求 $D[\arctan(3x)]$。
<!-- bilingual-en:start -->

&nbsp;
**1.** If $f(2)=5$ and $f'(2)=-3$, find $(f^{-1})'(5)$.<br>
**2.** Derive $(\arccos x)'$, taking care with its principal-value range.<br>
**3.** Find $D[\arctan(3x)]$.<br>
<!-- bilingual-en:end -->

> [!success]- 自检答案
> 1. $-1/3$。
> 2. 令 $y=\arccos x\in[0,\pi]$，$\cos y=x$，故 $-\sin y\,y'=1$；$\sin y=\sqrt{1-x^2}\ge0$，所以 $y'=-1/\sqrt{1-x^2}$。
> 3. 链式法则给 $3/(1+9x^2)$。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** $-1/3$.<br>
> **2.** Let $y=\arccos x\in[0,\pi]$, $\cos y=x$, hence $-\sin y\,y'=1$; $\sin y=\sqrt{1-x^2}\ge0$, hence $y'=-1/\sqrt{1-x^2}$.<br>
> **3.** The chain rule gives $3/(1+9x^2)$.<br>
> <!-- bilingual-en:end -->

### 本地材料

- [[Ses15a_Lecture_Notes.pdf#page=1|15a Derivative of the Inverse（pp.1-2）]]
- [[Ses15b_Lecture_Notes.pdf#page=1|15b Derivative of $\arctan x$（pp.1-3）]]
- [[Ses15c_Lecture_Notes.pdf#page=1|15c Derivative of $\arcsin x$（p.1）]]
- [[Exercise015_Problems.pdf#page=1|Exercise 015：Derivative of the Square Root]] · [[Exercise015_Solutions.pdf#page=1|答案]]

**知识链：**反函数复合为恒等函数 → 链式法则 → 对应斜率互为倒数 → 反三角导数与主值范围。
<!-- bilingual-en:start -->
**Knowledge chain:** an inverse composed with its function is the identity → chain rule → corresponding slopes are reciprocals → inverse-trigonometric derivatives and principal-value ranges.
<!-- bilingual-en:end -->

## Session 16：The Derivative of $a^x$

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题：**$a^x$ 的导数为何必定是它自身乘一个只依赖底数的常数？
<!-- bilingual-en:start -->
**Question:** Why must the derivative of $a^x$ equal $a^x$ times a constant that depends only on the base?
<!-- bilingual-en:end -->

**前置知识：**指数律、连续性、导数定义。此节尚未选定自然底数 $e$。
<!-- bilingual-en:start -->
**Prerequisites:** exponent laws, continuity, and the definition of the derivative. The natural base $e$ has not yet been selected.
<!-- bilingual-en:end -->

### 16a-16b：把指数函数定义到实数
<!-- bilingual-en:start -->
*16a–16b: Extending the Exponential Function to Real Exponents*
<!-- bilingual-en:end -->

[[导数与求导规则|指数函数]]取 $a>0$。整数指数由重复相乘与倒数定义；对有理数 $p/q$，定义
<!-- bilingual-en:start -->
For the [[导数与求导规则|exponential function]], take $a>0$. Integer exponents are defined by repeated multiplication and reciprocals; for a rational exponent $p/q$, define
<!-- bilingual-en:end -->

$$
a^{p/q}=\sqrt[q]{a^p}.
$$

指数律
<!-- bilingual-en:start -->
The exponent law
<!-- bilingual-en:end -->

$$
a^{x_1+x_2}=a^{x_1}a^{x_2}
$$

在有理指数上成立。对无理 $x$，用趋近 $x$ 的有理数序列补齐，并要求 $a^x$ 连续。这给出熟悉的连续指数曲线。课件为便于画图先设 $a>1$；$0<a<1$ 时函数递减，$a=1$ 时恒为 $1$。
<!-- bilingual-en:start -->
The identity holds first for rational exponents. For irrational $x$, define $a^x$ by limits of rational sequences approaching $x$ and require continuity. This produces the familiar continuous exponential curve. The slides initially assume $a>1$ for ease of graphing; the function decreases when $0<a<1$ and is identically $1$ when $a=1$.
<!-- bilingual-en:end -->

### 16c：从定义分离出 $x$
<!-- bilingual-en:start -->
*16c: Separating $x$ in the Difference Quotient*
<!-- bilingual-en:end -->

$$
\begin{aligned}
\frac d{dx}a^x
&=\lim_{h\to0}\frac{a^{x+h}-a^x}{h}\\
&=\lim_{h\to0}\frac{a^xa^h-a^x}{h}\\
&=a^x\lim_{h\to0}\frac{a^h-1}{h}.
\end{aligned}
$$

定义只依赖底数的常数
<!-- bilingual-en:start -->
Define the constant depending only on the base:
<!-- bilingual-en:end -->

$$
M(a)=\lim_{h\to0}\frac{a^h-1}{h}.
$$

于是
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

> [!important] 一般结构
> $$
> \boxed{\frac d{dx}a^x=M(a)a^x}.
> $$

这一步已经很强：指数函数在任何点的斜率，等于函数高度乘同一个相对增长常数。
<!-- bilingual-en:start -->
This step is already strong: the slope of an exponential function at any point equals the height of the function multiplied by the same relative growth constant.
<!-- bilingual-en:end -->

### 16d：$M(a)$ 的几何意义
<!-- bilingual-en:start -->
*16d: The Geometric Meaning of $M(a)$*
<!-- bilingual-en:end -->

在 $x=0$，$a^0=1$，所以
<!-- bilingual-en:start -->
At $x=0$, $a^0=1$, so
<!-- bilingual-en:end -->

$$
\left.\frac d{dx}a^x\right|_{x=0}=M(a).
$$

因此 $M(a)$ 是 $y=a^x$ 在 $(0,1)$ 的切线斜率。知道这一点的斜率，就通过 $M(a)a^x$ 知道整条曲线上每一点的斜率。对于 $a>1$，$M(a)>0$；$a=1$ 时 $M(1)=0$；$0<a<1$ 时 $M(a)<0$。
<!-- bilingual-en:start -->
Thus $M(a)$ is the slope of the tangent to $y=a^x$ at $(0,1)$. Once this one slope is known, the formula $M(a)a^x$ gives the slope at every point on the curve. For $a>1$, $M(a)>0$; for $a=1$, $M(1)=0$; and for $0<a<1$, $M(a)<0$.
<!-- bilingual-en:end -->

### Exercise 016：复利
<!-- bilingual-en:start -->
*Exercise 016: Compound Interest*
<!-- bilingual-en:end -->

本金 $P$、名义年利率 $r$，一年复利 $k$ 次：
<!-- bilingual-en:start -->
With principal $P$, nominal annual interest rate $r$, and $k$ compounding periods per year,
<!-- bilingual-en:end -->

$$
A=P\left(1+\frac rk\right)^k.
$$

等效年收益率为
<!-- bilingual-en:start -->
The effective annual return is
<!-- bilingual-en:end -->

$$
\mathrm{APR}_{\rm eff}=\left(1+\frac rk\right)^k-1.
$$

代入：
<!-- bilingual-en:start -->
Substitute:
<!-- bilingual-en:end -->

- $5\%$ 月复利：$5.1162\%$；日复利：$5.1267\%$；
- $10\%$ 月复利：$10.4713\%$；双周复利（$k=26$）：$10.4959\%$；日复利：$10.5156\%$。
<!-- bilingual-en:start -->
- At a nominal rate of $5\%$: monthly compounding gives $5.1162\%$, and daily compounding gives $5.1267\%$.
- At a nominal rate of $10\%$: monthly compounding gives $10.4713\%$, biweekly compounding ($k=26$) gives $10.4959\%$, and daily compounding gives $10.5156\%$.
<!-- bilingual-en:end -->

连续复利的极限将在 Session 19 得到 $e^r-1$。
<!-- bilingual-en:start -->
Session 19 will show that the corresponding continuously compounded return is $e^r-1$.
<!-- bilingual-en:end -->

### 边界情况与易错点
<!-- bilingual-en:start -->
*Boundary Cases and Common Pitfalls*
<!-- bilingual-en:end -->

- 实指数底数要求 $a>0$；负底数不能对所有实指数给出实值连续函数。
- $a$ 是固定底数；若底数也随 $x$ 变成 $x^x$，本节公式不能直接套用。
- $M(a)$ 目前只是极限定义，尚未证明等于 $\ln a$；Session 17 才完成识别。
- 名义利率 $r$ 与等效年收益率不同；复利次数越多，后者通常越大但有有限上界。
<!-- bilingual-en:start -->
- For arbitrary real exponents, the base must satisfy $a>0$; a negative base cannot define a real-valued continuous function for every real exponent.
- Here $a$ is fixed. If the base also varies, as in $x^x$, the formula in this section does not apply directly.
- At this stage, $M(a)$ is defined only by a limit; its identification as $\ln a$ comes in Session 17.
- The nominal rate $r$ differs from the effective annual return. More frequent compounding usually increases the latter, but only up to a finite limit.
<!-- bilingual-en:end -->

### 三道自检
<!-- bilingual-en:start -->
*Three Quick Self-Checks*
<!-- bilingual-en:end -->

1. 从定义证明 $(a^x)'/a^x$ 与 $x$ 无关。
2. $M(1)$ 是多少？与图像如何对应？
3. 若 $M(4)=2M(2)$，可从哪条指数关系直观预期这一点？
<!-- bilingual-en:start -->

&nbsp;
**1.** From the definition, prove that $(a^x)'/a^x$ is independent of $x$.<br>
**2.** What is $M(1)$, and how is it reflected in the graph?<br>
**3.** Which exponent law makes the relation $M(4)=2M(2)$ intuitively plausible?<br>
<!-- bilingual-en:end -->

> [!success]- 自检答案
> 1. 正文 16c 把 $a^x$ 提到极限外，剩余极限只含 $a,h$。
> 2. $M(1)=\lim(1^h-1)/h=0$；$y=1$ 是水平线。
> 3. $4^x=(2^x)^2$；用积法则求导得 $(4^x)'=2\cdot2^x(2^x)'=2M(2)4^x$，故 $M(4)=2M(2)$。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** Section 16c factors $a^x$ outside the limit; the remaining limit depends only on $a$ and $h$.<br>
> **2.** $M(1)=\lim(1^h-1)/h=0$; $y=1$ is the horizontal line.<br>
> **3.** $4^x=(2^x)^2$;$(4^x)'=2\cdot2^x(2^x)'=2M(2)4^x$ is obtained by the product rule, so $M(4)=2M(2)$.<br>
> <!-- bilingual-en:end -->

### 本地材料

- [[Ses16a_Lecture_Notes.pdf#page=1|16a Differentiating Logs and Exponentials（p.1）]]
- [[Ses16b_Lecture_Notes.pdf#page=1|16b Working with Exponents（p.1）]]
- [[Ses16c_Lecture_Notes.pdf#page=1|16c $a^x$ and the Definition（p.1）]]
- [[Ses16d_Lecture_Notes.pdf#page=1|16d Slope of the Tangent to $a^x$（pp.1-2）]]
- [[Exercise016_Problems.pdf#page=1|Exercise 016：Compound Interest]] · [[Exercise016_Solutions.pdf#page=1|答案]]

**知识链：**指数律 → 差商中提出 $a^x$ → 剩余常数 $M(a)$ → 一点斜率控制整条指数曲线。
<!-- bilingual-en:start -->
**Knowledge chain:** exponent law → factor $a^x$ out of the difference quotient → isolate the constant $M(a)$ → the slope at one point controls the entire exponential curve.
<!-- bilingual-en:end -->

## Session 17：The Exponential Function, Its Derivative, and Its Inverse

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题：**能否选择一个底数，使指数函数的导数恰好等于自身？它的反函数为何导数为 $1/x$？
<!-- bilingual-en:start -->
**Question:** Can we choose a base whose exponential function is its own derivative, and why does the inverse function then have derivative $1/x$?
<!-- bilingual-en:end -->

**前置知识：**$M(a)$、反函数导数、指数律。
<!-- bilingual-en:start -->
**Prerequisites:** $M(a)$, the inverse-function derivative, and exponent laws.
<!-- bilingual-en:end -->

### 17a：用斜率定义 $e$
<!-- bilingual-en:start -->
*17a: Defining $e$ by Its Slope*
<!-- bilingual-en:end -->

Session 16 得到
<!-- bilingual-en:start -->
Session 16 established that
<!-- bilingual-en:end -->

$$
(a^x)'=M(a)a^x,
\qquad
M(a)=\lim_{h\to0}\frac{a^h-1}{h}.
$$

定义 $e$ 为使 $M(e)=1$ 的唯一正底数：
<!-- bilingual-en:start -->
Define $e$ as the unique positive base satisfying $M(e)=1$:
<!-- bilingual-en:end -->

$$
\lim_{h\to0}\frac{e^h-1}{h}=1.
$$

于是
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

> [!important] 自然指数函数
> $$
> \boxed{\frac d{dx}e^x=e^x}.
> $$

几何上，$y=e^x$ 在 $(0,1)$ 的切线斜率为 $1$。课件用图形夹出 $2<e<4$：$2^x$ 在 $0$ 的切线比连接 $(0,1),(1,2)$ 的割线平，故 $M(2)<1$；$4^x$ 足够陡，$M(4)>1$。若 $M(a)$ 随 $a$ 连续且严格增加，中间必有唯一底数使斜率为 $1$。这是本课层级的存在唯一性说明；更完整证明需要建立指数函数的连续单调理论。
<!-- bilingual-en:start -->
Geometrically, $y=e^x$ has tangent slope $1$ at $(0,1)$. The lecture notes locate the base between $2$ and $4$: the slope of $2^x$ at $0$ is $M(2)<1$, while the slope of $4^x$ there exceeds the slope of the secant through $(0,1)$ and $(1,4)$, so $M(4)>1$. If $M(a)$ varies continuously and strictly increasingly with $a$, there is a unique intermediate base for which $M(a)=1$. A complete proof requires the continuity and monotonicity theory of exponential functions.
<!-- bilingual-en:end -->

### 17b：[[导数与求导规则|自然对数]]是 $e^x$ 的反函数
<!-- bilingual-en:start -->
*17b: The [[导数与求导规则|Natural Logarithm]] Is the Inverse of $e^x$*
<!-- bilingual-en:end -->

定义
<!-- bilingual-en:start -->
Define
<!-- bilingual-en:end -->

$$
y=e^x\quad\Longleftrightarrow\quad x=\ln y.
$$

因此 $\ln x$ 的定义域是 $x>0$，值域为全体实数；图像是 $e^x$ 关于 $y=x$ 的反射，且
<!-- bilingual-en:start -->
Thus $\ln x$ has domain $x>0$ and range $\mathbb R$; its graph is the reflection of $e^x$ across $y=x$. Moreover,
<!-- bilingual-en:end -->

$$
\ln1=0,
\qquad
\ln e=1,
\qquad
\ln(x_1x_2)=\ln x_1+\ln x_2.
$$

令 $w=\ln x$，则 $e^w=x$。隐式求导：
<!-- bilingual-en:start -->
Let $w=\ln x$, so $e^w=x$. Implicit differentiation gives
<!-- bilingual-en:end -->

$$
e^w\frac{dw}{dx}=1.
$$

又 $e^w=x$，所以
<!-- bilingual-en:start -->
And $e^w=x$, so
<!-- bilingual-en:end -->

> [!important] 自然对数导数
> $$
> \boxed{\frac d{dx}\ln x=\frac1x},\qquad x>0.
> $$

![[98_attachment/MIT18.01SC/unit01-exp-log.png|900]]

### 17c：识别一般底数的 $M(a)$
<!-- bilingual-en:start -->
*17c: Identifying $M(a)$ for a General Base*
<!-- bilingual-en:end -->

因为 $a=e^{\ln a}$，
<!-- bilingual-en:start -->
Because $a=e^{\ln a}$,
<!-- bilingual-en:end -->

$$
a^x=e^{x\ln a}.
$$

链式法则给
<!-- bilingual-en:start -->
the chain rule gives
<!-- bilingual-en:end -->

$$
\frac d{dx}a^x
=e^{x\ln a}\cdot\ln a
=a^x\ln a.
$$

与 $(a^x)'=M(a)a^x$ 比较：
<!-- bilingual-en:start -->
Compare to $(a^x)'=M(a)a^x$:
<!-- bilingual-en:end -->

$$
\boxed{M(a)=\ln a},
$$

从而
<!-- bilingual-en:start -->
Thus,
<!-- bilingual-en:end -->

$$
\boxed{(a^x)'=a^x\ln a},\qquad a>0.
$$

若 $a>1$，$\ln a>0$，指数函数递增；若 $0<a<1$，$\ln a<0$，指数函数递减。
<!-- bilingual-en:start -->
If $a>1$, $\ln a>0$, the exponential function increases; if $0<a<1$, $\ln a<0$, the exponential function decreases.
<!-- bilingual-en:end -->

### 17d：为什么自然对数“自然”
<!-- bilingual-en:start -->
*17d: Why the Natural Logarithm Is “Natural”*
<!-- bilingual-en:end -->

若价格 $p(t)>0$，相对变化率为
<!-- bilingual-en:start -->
If a price satisfies $p(t)>0$, its relative rate of change is
<!-- bilingual-en:end -->

$$
\frac{p'(t)}{p(t)}.
$$

链式法则恰好给
<!-- bilingual-en:start -->
The chain rule gives exactly
<!-- bilingual-en:end -->

$$
\frac d{dt}\ln p(t)=\frac{p'(t)}{p(t)}.
$$

它把乘法增长变成加法，把绝对变化除以当前规模，因此适合比较不同规模资产、人口或浓度的增长。若改用 $\log_{10}$，导数会多出 $1/\ln10$，形式不再直接等于相对增长率。
<!-- bilingual-en:start -->
The logarithm turns multiplicative growth into addition and scales absolute change by the current level, making it suitable for comparing assets, populations, or concentrations of different sizes. With $\log_{10}$ instead, the derivative carries an extra factor $1/\ln10$ and no longer equals the relative growth rate directly.
<!-- bilingual-en:end -->

### Exercise 017：指数与对数方程的检查法
<!-- bilingual-en:start -->
*Exercise 017: Exponential and Logarithmic Equations*
<!-- bilingual-en:end -->

例如
<!-- bilingual-en:start -->
For example
<!-- bilingual-en:end -->

$$
\ln(y+1)+\ln(y-1)=2x+\ln x.
$$

先由真数要求 $y>1,x>0$。合并并指数化：
<!-- bilingual-en:start -->
Positivity of the logarithm arguments requires $y>1$ and $x>0$. Combine the logarithms and exponentiate:
<!-- bilingual-en:end -->

$$
\ln(y^2-1)=\ln(xe^{2x})
\Longrightarrow y^2-1=xe^{2x}.
$$

由 $y>1$ 选正根：
<!-- bilingual-en:start -->
The condition $y>1$ selects the positive root:
<!-- bilingual-en:end -->

$$
\boxed{y=\sqrt{xe^{2x}+1}}.
$$

取对数或指数化都可能引入分支选择，最后必须回到原定义域检查。
<!-- bilingual-en:start -->
Taking logarithms or exponentiating can introduce branch choices, so the result must be checked against the original domain.
<!-- bilingual-en:end -->

### 边界情况与易错点
<!-- bilingual-en:start -->
*Boundary Cases and Common Pitfalls*
<!-- bilingual-en:end -->

- $\ln x$ 只对 $x>0$ 定义；$(\ln|x|)'=1/x$ 才能覆盖 $x<0$ 的区间。
- $\ln(u+v)$ 不能拆成 $\ln u+\ln v$；只有乘积可拆。
- $(e^{u(x)})'=e^u u'$，不能因 $e^x$ 自导就漏掉链式因子。
- $\log$ 的底数在不同学科可能表示 $10$、$e$ 或 $2$；本笔记用 $\ln$ 明确自然对数。
<!-- bilingual-en:start -->
- $\ln x$ is defined only for $x>0$; $(\ln|x|)'=1/x$ extends the derivative formula to intervals where $x<0$.
- $\ln(u+v)$ cannot be split into $\ln u+\ln v$; logarithms split products, not sums.
- $(e^{u(x)})'=e^u u'$; the fact that $e^x$ is its own derivative does not remove the chain-rule factor.
- The base of the $\log$ may represent $10$, $e$, or $2$ in different disciplines; this notebook uses $\ln$ for unambiguous natural logarithms.
<!-- bilingual-en:end -->

### 三道自检
<!-- bilingual-en:start -->
*Three Quick Self-Checks*
<!-- bilingual-en:end -->

1. 求 $D[e^{3x^2}]$。
2. 求 $D[\ln(5x)]$，并解释常数 $5$ 为何消失。
3. 若 $p'/p=0.04$，$D(\ln p)$ 是多少？其单位是什么？
<!-- bilingual-en:start -->

&nbsp;
**1.** Find $D[e^{3x^2}]$.<br>
**2.** Find $D[\ln(5x)]$ and explain why the constant $5$ disappears.<br>
**3.** If $p'/p=0.04$, what is $D(\ln p)$, and what are its units?<br>
<!-- bilingual-en:end -->

> [!success]- 自检答案
> 1. $6xe^{3x^2}$。
> 2. 直接链式为 $5/(5x)=1/x$；或 $\ln(5x)=\ln5+\ln x$，常数导数为零。定义域 $x>0$。
> 3. $0.04$；若自变量是年，则单位为每年。$\ln p$ 无量纲，相对增长率的单位来自时间倒数。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** $6xe^{3x^2}$.<br>
> **2.** Direct use of the chain rule gives $5/(5x)=1/x$. Alternatively, $\ln(5x)=\ln5+\ln x$, and the derivative of the constant is zero. The domain is $x>0$.<br>
> **3.** The value is $0.04$. If time is measured in years, the unit is per year. Although $\ln p$ is dimensionless, its rate of change has units of inverse time.<br>
> <!-- bilingual-en:end -->

### 本地材料

- [[Ses17a_Lecture_Notes.pdf#page=1|17a Definition of $e$（pp.1-3）]]
- [[Ses17b_Lecture_Notes.pdf#page=1|17b Natural Log and Its Derivative（pp.1-2）]]
- [[Ses17c_Lecture_Notes.pdf#page=1|17c Derivative of $a^x$（p.1）]]
- [[Ses17d_Lecture_Notes.pdf#page=1|17d The Most Natural Logarithm（p.1）]]
- [[Exercise017_Problems.pdf#page=1|Exercise 017：Solving Equations with $e$ and $\ln$]] · [[Exercise017_Solutions.pdf#page=1|答案]]

**知识链：**$M(a)$ → 选 $M(e)=1$ → $e^x$ 自导 → 反函数 $\ln x$ → 一般 $a^x$ 与相对变化率。
<!-- bilingual-en:start -->
**Knowledge chain:** $M(a)$ → choose $e$ so that $M(e)=1$ → $e^x$ is its own derivative → inverse function $\ln x$ → general $a^x$ and relative rates of change.
<!-- bilingual-en:end -->

## Session 18：Derivatives of Other Exponential Functions

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题：**怎样系统求一般指数、变量幂和任意实数幂的导数？
<!-- bilingual-en:start -->
**Question:** How can we systematically differentiate general exponentials, variable powers, and arbitrary real powers?
<!-- bilingual-en:end -->

**前置知识：**$e^x,\ln x$ 的导数、积与链式法则、对数性质。
<!-- bilingual-en:start -->
**Prerequisites:** the derivatives of $e^x$ and $\ln x$, the product and chain rules, and logarithm laws.
<!-- bilingual-en:end -->

### 18a：$2^x$ 与 $10^x$
<!-- bilingual-en:start -->
*18a: $2^x$ and $10^x$*
<!-- bilingual-en:end -->

一般公式立即给
<!-- bilingual-en:start -->
The general formula immediately gives
<!-- bilingual-en:end -->

$$
(2^x)'=(\ln2)2^x,
\qquad
(10^x)'=(\ln10)10^x.
$$

即便从人类习惯的底数 $2$ 或 $10$ 出发，自然对数仍自动出现；$e$ 的特殊之处正是 $\ln e=1$。
<!-- bilingual-en:start -->
Even with the familiar bases $2$ and $10$, the natural logarithm appears automatically. The special feature of $e$ is precisely that $\ln e=1$.
<!-- bilingual-en:end -->

### 18b：[[导数与求导规则#求导规则为何成立|对数求导]]的核心公式
<!-- bilingual-en:start -->
*18b: The Core Formula for [[导数与求导规则#求导规则为何成立|Logarithmic Differentiation]]*
<!-- bilingual-en:end -->

若 $u(x)>0$，链式法则给
<!-- bilingual-en:start -->
If $u(x)>0$, the chain rule gives
<!-- bilingual-en:end -->

$$
\boxed{(\ln u)'=\frac{u'}u}.
$$

因此若先容易求 $(\ln u)'$，可反解
<!-- bilingual-en:start -->
Thus, if $(\ln u)'$ is easier to compute first, we can recover $u'$ from
<!-- bilingual-en:end -->

$$
u'=u(\ln u)'.
$$

例如对 $u=a^x$：
<!-- bilingual-en:start -->
For example, for $u=a^x$:
<!-- bilingual-en:end -->

$$
\ln u=x\ln a
\Longrightarrow
\frac{u'}u=\ln a
\Longrightarrow
u'=a^x\ln a.
$$

### 18c：移动底数与移动指数 $x^x$
<!-- bilingual-en:start -->
*18c: A Variable Base and Variable Exponent: $x^x$*
<!-- bilingual-en:end -->

设 $v=x^x$，在实数课程中先取 $x>0$。取自然对数：
<!-- bilingual-en:start -->
For $v=x^x$, restrict first to $x>0$ in the real-valued setting and take natural logarithms:
<!-- bilingual-en:end -->

$$
\ln v=x\ln x.
$$

两边求导，右边使用积法则：
<!-- bilingual-en:start -->
Differentiate both sides, using the product rule on the right:
<!-- bilingual-en:end -->

$$
\frac{v'}v=\ln x+x\frac1x=\ln x+1.
$$

乘回 $v=x^x$：
<!-- bilingual-en:start -->
Multiply through by $v=x^x$:
<!-- bilingual-en:end -->

$$
\boxed{\frac d{dx}x^x=x^x(1+\ln x)},\qquad x>0.
$$

一般地，若 $y=u(x)^{v(x)}$ 且 $u>0$：
<!-- bilingual-en:start -->
More generally, if $y=u(x)^{v(x)}$ with $u>0$, then
<!-- bilingual-en:end -->

$$
\ln y=v\ln u.
$$

求导后得到
<!-- bilingual-en:start -->
differentiation gives
<!-- bilingual-en:end -->

$$
\boxed{y'=u^v\left(v'\ln u+v\frac{u'}u\right)}.
$$

两项分别记录指数变化与底数变化；漏掉任一项都错。
<!-- bilingual-en:start -->
The two terms record changes in the exponent and the base, respectively; neither may be omitted.
<!-- bilingual-en:end -->

### 18d：实数幂法则
<!-- bilingual-en:start -->
*18d: The Real-Exponent Power Rule*
<!-- bilingual-en:end -->

对固定实数 $r$、$x>0$：
<!-- bilingual-en:start -->
For fixed real numbers $r$, $x>0$:
<!-- bilingual-en:end -->

$$
x^r=e^{r\ln x}.
$$

链式法则：
<!-- bilingual-en:start -->
Chain rule:
<!-- bilingual-en:end -->

$$
\frac d{dx}x^r
=e^{r\ln x}\frac r x
=x^r\frac r x
=\boxed{rx^{r-1}}.
$$

至此幂法则从正整数 → 负整数 → 有理数 → 实数完成扩展。对特殊 $r$，定义域有时可延伸到 $x\le0$；但上述 $e^{r\ln x}$ 证明本身只覆盖 $x>0$。
<!-- bilingual-en:start -->
The power rule has now been extended from positive integers to negative integers, rational numbers, and finally real numbers. For special values of $r$, the domain may extend to $x\le0$; however, the proof using $e^{r\ln x}$ itself applies only when $x>0$.
<!-- bilingual-en:end -->

### 对数求导何时特别有用
<!-- bilingual-en:start -->
*When Logarithmic Differentiation Is Especially Useful*
<!-- bilingual-en:end -->

- 多个幂的乘除：$y=(x-1)^3(x+2)^5/x^7$；
- 变量在指数中：$x^{\sin x}$、$(1+x)^{1/x}$；
- 直接反复积法则过长的表达式。
<!-- bilingual-en:start -->
- Products and quotients of several powers: $y=(x-1)^3(x+2)^5/x^7$;
- A variable in the exponent: $x^{\sin x}$ or $(1+x)^{1/x}$;
- Expressions for which repeated direct use of the product rule would be unwieldy.
<!-- bilingual-en:end -->

例如 $y=(x-1)^3(x+2)^5/x^7$（在各因子符号固定且可取对数的区间）：
<!-- bilingual-en:start -->
For example, on an interval where the factors have fixed signs and logarithms are valid, let $y=(x-1)^3(x+2)^5/x^7$:
<!-- bilingual-en:end -->

$$
\frac{y'}y=\frac3{x-1}+\frac5{x+2}-\frac7x,
$$

最后乘回 $y$ 即可。若因子可能为负，可在固定不穿零的区间使用 $\ln|u|$。
<!-- bilingual-en:start -->
Finally, multiply through by $y$. If a factor may be negative, use $\ln|u|$ on a fixed interval that does not cross a zero of $u$.
<!-- bilingual-en:end -->

### 边界情况与易错点
<!-- bilingual-en:start -->
*Boundary Cases and Common Pitfalls*
<!-- bilingual-en:end -->

- 对数求导前先保证表达式正，或在不跨零区间使用绝对值。
- $D(x^x)$ 既不是 $xx^{x-1}$，也不是 $x^x\ln x$；底数与指数都变，必须有两项。
- $x^r=e^{r\ln x}$ 的证明只在 $x>0$；不能无说明地把结论扩到负底数的任意实数幂。
- 求出 $y'/y$ 后不要忘记乘回 $y$。
<!-- bilingual-en:start -->
- Before logarithmic differentiation, ensure that the expression is positive, or use absolute values on an interval that does not cross zero.
- $D(x^x)$ is neither $xx^{x-1}$ nor $x^x\ln x$; both base and exponent vary, so the derivative contains two contributions.
- The proof using $x^r=e^{r\ln x}$ is valid only for $x>0$; it does not justify arbitrary real powers of negative bases.
- Don't forget to multiply back to $y$ after finding $y'/y$.
<!-- bilingual-en:end -->

### 三道自检
<!-- bilingual-en:start -->
*Three Quick Self-Checks*
<!-- bilingual-en:end -->

1. 求 $D[x^{\sin x}]$（$x>0$）。
2. 求 $D[(2x+1)^{x}]$（$2x+1>0$）。
3. 用对数求导求 $D[(x^2+1)^5/x^3]$。
<!-- bilingual-en:start -->

&nbsp;
**1.** Find $D[x^{\sin x}]$ for $x>0$.<br>
**2.** Find $D[(2x+1)^x]$ for $2x+1>0$.<br>
**3.** Use logarithmic differentiation to find $D[(x^2+1)^5/x^3]$.<br>
<!-- bilingual-en:end -->

> [!success]- 自检答案
> 1. $x^{\sin x}[\cos x\ln x+(\sin x)/x]$。
> 2. $(2x+1)^x[\ln(2x+1)+2x/(2x+1)]$。
> 3. 令 $y=(x^2+1)^5x^{-3}$，$y'/y=10x/(x^2+1)-3/x$，故 $y'=y[10x/(x^2+1)-3/x]$；定义域 $x\ne0$。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** $x^{\sin x}[\cos x\ln x+(\sin x)/x]$.<br>
> **2.** $(2x+1)^x[\ln(2x+1)+2x/(2x+1)]$.<br>
> **3.** Let $y=(x^2+1)^5x^{-3}$. Then $y'/y=10x/(x^2+1)-3/x$, so $y'=y[10x/(x^2+1)-3/x]$; the domain is $x\ne0$.<br>
> <!-- bilingual-en:end -->

### 本地材料

- [[Ses18a_Lecture_Notes.pdf#page=1|18a The Functions $10^x$ and $2^x$（p.1）]]
- [[Ses18b_Lecture_Notes.pdf#page=1|18b Logarithmic Differentiation（p.1）]]
- [[Ses18c_Lecture_Notes.pdf#page=1|18c Example $x^x$（p.1）]]
- [[Ses18d_Lecture_Notes.pdf#page=1|18d Real Power Rule（pp.1-2）]]
- 本地资料库没有 `Exercise018`；本节三道自检覆盖变量幂与对数求导。
<!-- bilingual-en:start -->
- There is no `Exercise018` in the local archive, so the three self-checks in this section cover variable powers and logarithmic differentiation.
<!-- bilingual-en:end -->

**知识链：**$\ln$ 把指数移到乘法位置 → 隐式求导 → 变量幂统一公式 → 实数幂法则。
<!-- bilingual-en:start -->
**Knowledge chain:** $\ln$ moves an exponent into a multiplicative position → implicit differentiation → unified formula for variable powers → real-power rule.
<!-- bilingual-en:end -->

## Session 19：An Interesting Limit Involving $e$

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题：**为什么“每次增长很少、次数无限多”的极限会产生 $e$？
<!-- bilingual-en:start -->
**Question:** Why does the limit of increasingly frequent compounding produce $e$?
<!-- bilingual-en:end -->

**前置知识：**对数与指数互逆、导数型极限、无穷极限变量替换。
<!-- bilingual-en:start -->
**Prerequisites:** the inverse relationship between logarithms and exponentials, derivative-form limits, and substitutions in limits at infinity.
<!-- bilingual-en:end -->

### 19a：逐步计算 $\left(1+1/n\right)^n$
<!-- bilingual-en:start -->
*19a: Step-by-step calculation of $\left(1+1/n\right)^n$*
<!-- bilingual-en:end -->

设
<!-- bilingual-en:start -->
Let
<!-- bilingual-en:end -->

$$
a_n=\left(1+\frac1n\right)^n.
$$

直接看是 $1^\infty$ 型，不能把底数极限和指数极限分别代入。先取对数，把移动指数变成乘法：
<!-- bilingual-en:start -->
This has the indeterminate form $1^\infty$, so the limits of the base and exponent cannot be substituted separately. First take logarithms, turning the varying exponent into a multiplicative factor:
<!-- bilingual-en:end -->

$$
\ln a_n=n\ln\left(1+\frac1n\right).
$$

令 $h=1/n$，则 $n\to\infty$ 时 $h\to0^+$：
<!-- bilingual-en:start -->
Let $h=1/n$, then $h\to0^+$ when $n\to\infty$:
<!-- bilingual-en:end -->

$$
\ln a_n=\frac{\ln(1+h)}h
=\frac{\ln(1+h)-\ln1}{h}.
$$

这正是 $\ln x$ 在 $x=1$ 的导数：
<!-- bilingual-en:start -->
This is exactly the derivative of $\ln x$ at $x=1$:
<!-- bilingual-en:end -->

$$
\lim_{h\to0}\frac{\ln(1+h)-\ln1}{h}
=(\ln x)'|_{x=1}=1.
$$

所以 $\ln a_n\to1$。指数函数连续，
<!-- bilingual-en:start -->
Thus $\ln a_n\to1$. By continuity of the exponential function,
<!-- bilingual-en:end -->

$$
a_n=e^{\ln a_n}\to e^1=e.
$$

> [!important] 关于 $e$ 的基本极限
> $$
> \boxed{\lim_{n\to\infty}\left(1+\frac1n\right)^n=e}.
> $$

这也给 $e$ 的数值近似，例如 $n=10000$ 时约为 $2.71815$。
<!-- bilingual-en:start -->
This also gives a numerical approximation of $e$, for example, about $2.71815$ for $n=10000$.
<!-- bilingual-en:end -->

### 19b：为什么不是 $1$
<!-- bilingual-en:start -->
*19b: Why the Limit Is Not $1$*
<!-- bilingual-en:end -->

底数 $1+1/n\to1$，但指数 $n\to\infty$；“非常小的增长”累计“非常多次”，总效应不能只看底数。对数后变成
<!-- bilingual-en:start -->
The base $1+1/n$ tends to $1$, but the exponent $n$ tends to infinity. A very small increase repeated very many times has a cumulative effect that cannot be read from the base alone. After taking logarithms, the expression becomes
<!-- bilingual-en:end -->

$$
n\ln(1+1/n),
$$

其中一个因子趋无穷、一个趋零，乘积极限需要上述导数计算，不能写成 $\infty\cdot0=0$。
<!-- bilingual-en:start -->
One factor tends to infinity and the other to zero. This product limit requires the derivative calculation above; the indeterminate form $\infty\cdot0$ cannot simply be assigned the value zero.
<!-- bilingual-en:end -->

### 推广
<!-- bilingual-en:start -->
*Generalization*
<!-- bilingual-en:end -->

对固定常数 $c,d$，在底数最终为正时：
<!-- bilingual-en:start -->
For fixed constants $c$ and $d$, provided the base is eventually positive,
<!-- bilingual-en:end -->

$$
\left(1+\frac cn\right)^{dn}
=\left[\left(1+\frac cn\right)^{n/c}\right]^{cd}\to e^{cd}.
$$

更稳妥的对数推导是
<!-- bilingual-en:start -->
A more robust logarithmic derivation is
<!-- bilingual-en:end -->

$$
dn\ln(1+c/n)
=cd\frac{\ln(1+h)}h\to cd,
\quad h=c/n.
$$

连续复利正是
<!-- bilingual-en:start -->
Continuous compound interest is exactly
<!-- bilingual-en:end -->

$$
\lim_{n\to\infty}P\left(1+\frac rn\right)^n=Pe^r.
$$

### 边界情况与易错点
<!-- bilingual-en:start -->
*Boundary Cases and Common Pitfalls*
<!-- bilingual-en:end -->

- $1^\infty$ 是未定式标签，不是答案 $1$。
- 取对数后必须在最后用指数函数连续性返回原极限。
- $h=1/n$ 只从正侧趋零；本题足够，因为 $\ln$ 在 $1$ 两侧导数一致。
- 有限 $n$ 的复利值不等于 $e^r$，只是随 $n$ 增大趋近。
<!-- bilingual-en:start -->
- $1^\infty$ labels an indeterminate form; it is not the answer $1$.
- After taking logarithms, use continuity of the exponential function to return to the original limit.
- $h=1/n$ goes to zero only from the positive side; this is sufficient because $\ln$ has the same derivative on both sides of $1$.
- For finite $n$, the compounded value is not equal to $e^r$; it approaches $e^r$ as $n$ grows.
<!-- bilingual-en:end -->

### 三道自检
<!-- bilingual-en:start -->
*Three Quick Self-Checks*
<!-- bilingual-en:end -->

1. 求 $\lim_{n\to\infty}(1+1/n)^{3n}$。
2. 求 $\lim_{n\to\infty}(1+2/n)^{5n}$。
3. 解释 $\lim_{h\to0}(1+h)^{1/h}=e$ 与本节公式的关系。
<!-- bilingual-en:start -->

&nbsp;
**1.** Find $\lim_{n\to\infty}(1+1/n)^{3n}$.<br>
**2.** Find $\lim_{n\to\infty}(1+2/n)^{5n}$.<br>
**3.** Explain how $\lim_{h\to0}(1+h)^{1/h}=e$ is related to the formula in this section.<br>
<!-- bilingual-en:end -->

> [!success]- 自检答案
> 1. $e^3$。
> 2. $e^{10}$。
> 3. 令 $h=1/n$ 得同一序列形式；更一般地取对数，$\ln[(1+h)^{1/h}]=\ln(1+h)/h\to1$，所以原式趋 $e$。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** $e^3$.<br>
> **2.** $e^{10}$.<br>
> **3.** Setting $h=1/n$ gives the same sequence form. More generally, take logarithms: $\ln[(1+h)^{1/h}]=\ln(1+h)/h\to1$, so the original expression tends to $e$.<br>
> <!-- bilingual-en:end -->

### 本地材料

- [[Ses19a_Lecture_Notes.pdf#page=1|19a Another Moving Exponent（pp.1-2）]]
- [[Ses19b_Lecture_Notes.pdf#page=1|19b A Formula for $e$（p.1）]]
- [[Exercise019_Problems.pdf#page=1|Exercise 019：Evaluating an Interesting Limit]] · [[Exercise019_Solutions.pdf#page=1|答案]]
- 资料库另有同内容副本 `Exercise019_Problems_2.pdf` / `Solutions_2.pdf`；正文只链接一份，避免重复导航。
<!-- bilingual-en:start -->
- The archive also contains duplicate files, `Exercise019_Problems_2.pdf` and `Solutions_2.pdf`; the note links only one copy to avoid redundant navigation.
<!-- bilingual-en:end -->

**知识链：**移动指数 → 取对数 → 换元成 $\ln$ 在 $1$ 的导数 → 指数化返回 → 连续复利。
<!-- bilingual-en:start -->
**Knowledge chain:** isolate the moving exponent → take logarithms → substitute to obtain the derivative of $\ln$ at $1$ → exponentiate to return to the original limit → continuous compounding.
<!-- bilingual-en:end -->

## Session 20：Hyperbolic Trig Functions

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题：**由 $e^x,e^{-x}$ 的对称组合产生哪些类似三角函数的结构？
<!-- bilingual-en:start -->
**Question:** What trigonometric-like structure emerges from symmetric combinations of $e^x$ and $e^{-x}$?
<!-- bilingual-en:end -->

**前置知识：**指数求导、积与链式法则、双曲线方程。
<!-- bilingual-en:start -->
**Prerequisites:** exponential derivatives, the product and chain rules, and the hyperbola equation.
<!-- bilingual-en:end -->

### 20a：定义与导数
<!-- bilingual-en:start -->
*20a: Definition and Derivative*
<!-- bilingual-en:end -->

[[导数与求导规则|双曲函数]]中的双曲正弦（hyperbolic sine）和双曲余弦（hyperbolic cosine）定义为：
<!-- bilingual-en:start -->
The [[导数与求导规则|hyperbolic functions]] hyperbolic sine and hyperbolic cosine are defined by
<!-- bilingual-en:end -->

$$
\sinh x=\frac{e^x-e^{-x}}2,
\qquad
\cosh x=\frac{e^x+e^{-x}}2.
$$

注意 $(e^{-x})'=-e^{-x}$。因此
<!-- bilingual-en:start -->
Since $(e^{-x})'=-e^{-x}$,
<!-- bilingual-en:end -->

$$
\begin{aligned}
(\sinh x)'
&=\frac{e^x-(-e^{-x})}{2}=\cosh x,\\
(\cosh x)'
&=\frac{e^x+(-e^{-x})}{2}=\sinh x.
\end{aligned}
$$

与圆三角函数不同，$\cosh$ 求导没有负号。
<!-- bilingual-en:start -->
Unlike the circular trigonometric functions, differentiating $\cosh$ introduces no minus sign.
<!-- bilingual-en:end -->

### 核心恒等式的逐步证明
<!-- bilingual-en:start -->
*A Step-by-Step Proof of the Core Identity*
<!-- bilingual-en:end -->

$$
\begin{aligned}
\cosh^2x-\sinh^2x
&=\frac{(e^x+e^{-x})^2-(e^x-e^{-x})^2}{4}\\
&=\frac{[e^{2x}+2+e^{-2x}]-[e^{2x}-2+e^{-2x}]}4\\
&=\frac44=\boxed{1}.
\end{aligned}
$$

所以点
<!-- bilingual-en:start -->
Therefore, the point
<!-- bilingual-en:end -->

$$
(u,v)=(\cosh x,\sinh x)
$$

满足 $u^2-v^2=1$，位于单位双曲线右支；这就是“hyperbolic”的来源。圆三角对应 $\cos^2x+\sin^2x=1$。
<!-- bilingual-en:start -->
satisfies $u^2-v^2=1$ and lies on the right branch of the unit hyperbola; this is the source of the word “hyperbolic.” The circular-trigonometric counterpart is $\cos^2x+\sin^2x=1$.
<!-- bilingual-en:end -->

![[98_attachment/MIT18.01SC/unit01-hyperbolic.png|900]]

### 由定义推出其他公式
<!-- bilingual-en:start -->
*Deriving Further Formulas from the Definitions*
<!-- bilingual-en:end -->

$$
\tanh x=\frac{\sinh x}{\cosh x}.
$$

商法则与恒等式给
<!-- bilingual-en:start -->
The quotient rule and the preceding identity give
<!-- bilingual-en:end -->

$$
(\tanh x)'
=\frac{\cosh^2x-\sinh^2x}{\cosh^2x}
=\boxed{\operatorname{sech}^2x}.
$$

奇偶性：$\sinh$ 为奇函数，$\cosh$ 为偶函数；这也与导数的奇偶转换吻合。
<!-- bilingual-en:start -->
Parity: $\sinh$ is an odd function and $\cosh$ is an even function; this is also consistent with the parity conversion of derivatives.
<!-- bilingual-en:end -->

### Exercise 020：双曲加法公式
<!-- bilingual-en:start -->
*Exercise 020: Hyperbolic Addition Formula*
<!-- bilingual-en:end -->

从指数定义展开：
<!-- bilingual-en:start -->
Expand from the exponential definitions:
<!-- bilingual-en:end -->

$$
\begin{aligned}
\sinh(x+y)
&=\frac{e^xe^y-e^{-x}e^{-y}}2\\
&=\frac{(e^x-e^{-x})(e^y+e^{-y})+(e^x+e^{-x})(e^y-e^{-y})}{4}\\
&=\boxed{\sinh x\cosh y+\cosh x\sinh y}.
\end{aligned}
$$

同理
<!-- bilingual-en:start -->
Similarly,
<!-- bilingual-en:end -->

$$
\boxed{\cosh(x+y)=\cosh x\cosh y+\sinh x\sinh y}.
$$

第二式中间是加号，而圆三角的 $\cos(x+y)$ 公式中间是减号；根源是双曲恒等式使用差平方。
<!-- bilingual-en:start -->
The second formula has a plus sign where the circular identity for $\cos(x+y)$ has a minus sign. The difference comes from the difference-of-squares identity underlying the hyperbolic functions.
<!-- bilingual-en:end -->

### 边界情况与易错点
<!-- bilingual-en:start -->
*Boundary Cases and Common Pitfalls*
<!-- bilingual-en:end -->

- $\cosh x\ge1$，从不为零，所以 $\tanh x$ 对所有实数定义。
- $\sinh$、$\sin$ 名称相似但导数循环不同；不要凭符号机械搬用。
- $\cosh^{-1}x$ 常表示反双曲余弦，不是 $1/\cosh x$；倒数写作 $\operatorname{sech}x$。
- 双曲角的几何解释与普通圆角不同，本章只需要指数定义和代数恒等式。
<!-- bilingual-en:start -->
- Since $\cosh x\ge1$, it never vanishes, so $\tanh x$ is defined for every real $x$.
- The names $\sinh$ and $\sin$ look similar, but their derivative cycles differ; do not transfer formulas mechanically from one to the other.
- $\cosh^{-1}x$ commonly denotes inverse hyperbolic cosine, not $1/\cosh x$; the reciprocal is written $\operatorname{sech}x$.
- Hyperbolic angles have a different geometric interpretation from ordinary circular angles; this chapter needs only the exponential definitions and algebraic identities.
<!-- bilingual-en:end -->

### 三道自检
<!-- bilingual-en:start -->
*Three Quick Self-Checks*
<!-- bilingual-en:end -->

1. 求 $D[\sinh(3x)]$。
2. 证明 $\cosh x\ge1$。
3. 求 $D[\operatorname{sech}x]$。
<!-- bilingual-en:start -->

&nbsp;
**1.** Find $D[\sinh(3x)]$.<br>
**2.** Prove that $\cosh x\ge1$.<br>
**3.** Find $D[\operatorname{sech}x]$.<br>
<!-- bilingual-en:end -->

> [!success]- 自检答案
> 1. $3\cosh(3x)$。
> 2. $\cosh x=(e^x+e^{-x})/2\ge\sqrt{e^xe^{-x}}=1$（AM-GM），等号仅在 $x=0$。
> 3. sech$\,x=1/\cosh x$，故导数 $-\sinh x/\cosh^2x=-\operatorname{sech}x\tanh x$。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** $3\cosh(3x)$.<br>
> **2.** By AM–GM, $\cosh x=(e^x+e^{-x})/2\ge\sqrt{e^xe^{-x}}=1$, with equality only at $x=0$.<br>
> **3.** sech$\,x=1/\cosh x$, so the derivative $-\sinh x/\cosh^2x=-\operatorname{sech}x\tanh x$.<br>
> <!-- bilingual-en:end -->

### 本地材料

- [[Ses20a_Lecture_Notes.pdf#page=1|20a Derivatives of Hyperbolic Sine and Cosine（p.1）]]
- [[Exercise020_Problems.pdf#page=1|Exercise 020：Hyperbolic Angle Sum Formula]] · [[Exercise020_Solutions.pdf#page=1|答案]]

**知识链：**指数函数的对称/反对称组合 → 双曲正余弦 → 导数互换 → 差平方恒等式与双曲线。
<!-- bilingual-en:start -->
**Knowledge chain:** symmetric and antisymmetric combinations of exponentials → hyperbolic sine and cosine → derivatives interchange → difference-of-squares identity and the hyperbola.
<!-- bilingual-en:end -->

## Problem Set 2

官网在 Part B 后同时指定两本题册：Differentiation 的 1F-1I，以及 Integration Techniques 的 5A。5A 在此只承担反三角与双曲函数练习，不表示课程已经进入积分技巧。
<!-- bilingual-en:start -->
After Part B, the official course assigns problems from two booklets: Sections 1F–1I of Differentiation and Section 5A of Integration Techniques. Here 5A supplies practice on inverse-trigonometric and hyperbolic functions; it does not mean that the course has already moved into techniques of integration.
<!-- bilingual-en:end -->

- [[PSet01_Problems.pdf#page=6|Differentiation 原题（相关页 pp.6-10）]]
- [[PSet01_Solutions.pdf#page=11|Differentiation 官方解答（相关页 pp.11-16）]]
- [[PSet05_Problems.pdf#page=1|Integration Techniques 5A 原题（pp.1-2）]]
- [[PSet05_Solutions.pdf#page=1|Integration Techniques 5A 官方解答（pp.1-3）]]

| 节 | 官网指定题 |
|---|---|
| 1F | 3, 5, 8a, 8c |
| 1G | 4, 5b |
| 5A（第一组） | 1a, 1b, 1c（只求 sin、cos、sec）, 3f, 3g, 3h |
| 1H | 1a, 1b, 2, 3a, 5b |
| 1I | 1c, 1d, 1e, 1f, 1m, 4a |
| 5A（第二组） | 5a, 5b, 5c |

> [!example]- 1F-3、1F-5：有理幂与隐式三角曲线
> **1F-3。** 令 $y=x^{1/n}$，则 $y^n=x$。求导：
> $$ny^{n-1}y'=1.$$
> 所以
> $$
> \boxed{y'=\frac1{ny^{n-1}}=\frac1n x^{1/n-1}}.
> $$
> 除以 $y^{n-1}$ 的步骤排除了 $y=0$，零点需按 $n$ 的奇偶和单侧/双侧定义另查。
>
> **1F-5。** 曲线族
> $$\sin x+\sin y=\frac12.$$
> $$
> 隐式求导：
> $$\cos x+\cos y\,y'=0
> \quad\Longrightarrow\quad
> y'=-\frac{\cos x}{\cos y}.
> $$
> 水平切线要求 $y'=0$，即 $\cos x=0$ 且 $\cos y\ne0$。写 $x=\pi/2+k\pi$。若 $k$ 奇，$\sin x=-1$，会要求 $\sin y=3/2$，无解；若 $k$ 偶，$\sin x=1$，故 $\sin y=-1/2$。于是所有水平切点为
> $$
> \boxed{\left(\frac\pi2+2k\pi,-\frac\pi6+2n\pi\right)},
> $$
> 或
> $$
> \boxed{\left(\frac\pi2+2k\pi,\frac{7\pi}6+2n\pi\right)},
> \qquad k,n\in\mathbb Z.
> $$
> 两类点均有 $\cos y\ne0$，所以确为水平而非奇点。
> <!-- bilingual-en:start -->
> **1F-3.** Let $y=x^{1/n}$, so $y^n=x$. Differentiating gives
> $$ny^{n-1}y'=1,$$
> and therefore
> $$
> \boxed{y'=\frac1{ny^{n-1}}=\frac1n x^{1/n-1}}.
> $$
> Dividing by $y^{n-1}$ excludes $y=0$, so the derivative at zero must be checked separately according to the parity of $n$ and whether a one-sided or two-sided definition is appropriate.
>
> **1F-5.** Consider the family of curves
> $$\sin x+\sin y=\frac12.$$
> Implicit differentiation gives
> $$\cos x+\cos y\,y'=0
> \quad\Longrightarrow\quad
> y'=-\frac{\cos x}{\cos y}.$$
> A horizontal tangent requires $y'=0$, hence $\cos x=0$ and $\cos y\ne0$. Write $x=\pi/2+k\pi$. If $k$ is odd, then $\sin x=-1$, which would require $\sin y=3/2$ and is impossible. If $k$ is even, then $\sin x=1$, so $\sin y=-1/2$. Thus all horizontal-tangency points are
> $$
> \boxed{\left(\frac\pi2+2k\pi,-\frac\pi6+2n\pi\right)}
> $$
> or
> $$
> \boxed{\left(\frac\pi2+2k\pi,\frac{7\pi}6+2n\pi\right)},
> \qquad k,n\in\mathbb Z.
> $$
> Both families have $\cos y\ne0$, so these are genuine horizontal tangencies rather than singular points.
> <!-- bilingual-en:end -->

> [!example]- 1F-8a、1F-8c：关系式中的参数变化
> **1F-8a。** 圆锥体积 $V=\frac13\pi r^2h$ 固定，求 $dr/dh$。关于 $h$ 求导：
> $$
> 0=\frac\pi3\left(2r\frac{dr}{dh}h+r^2\right).
> $$
> 因而
> $$
> \boxed{\frac{dr}{dh}=-\frac r{2h}}.
> $$
> 负号表示体积固定时，高度增加必须伴随半径减小。
>
> **1F-8c。** 余弦定理
> $$c^2=a^2+b^2-2ab\cos\theta,$$
> 把 $c,\theta$ 视为常数，求 $da/db$：
> $$
> 0=2a a'+2b-2\cos\theta(a'b+a).
> $$
> 收集 $a'$：
> $$
> 2(a-b\cos\theta)a'=2(a\cos\theta-b),
> $$
> 所以
> $$
> \boxed{\frac{da}{db}=\frac{a\cos\theta-b}{a-b\cos\theta}}.
> $$
> 若分母为零，不能使用该形式，应检查关系曲线相对于 $b$ 是否有竖直切线。
> <!-- bilingual-en:start -->
> **1F-8a.** The cone volume $V=\frac13\pi r^2h$ is fixed. To find $dr/dh$, differentiate with respect to $h$:
> $$
> 0=\frac\pi3\left(2r\frac{dr}{dh}h+r^2\right).
> $$
> Hence,
> $$
> \boxed{\frac{dr}{dh}=-\frac r{2h}}.
> $$
> The negative sign means that, at fixed volume, an increase in height must be accompanied by a decrease in radius.
>
> **1F-8c.** Start from the law of cosines,
> $$c^2=a^2+b^2-2ab\cos\theta,$$
> and regard $c$ and $\theta$ as constants. Differentiating with respect to $b$ gives
> $$
> 0=2a a'+2b-2\cos\theta(a'b+a).
> $$
> Collecting the $a'$ terms,
> $$
> 2(a-b\cos\theta)a'=2(a\cos\theta-b),
> $$
> so
> $$
> \boxed{\frac{da}{db}=\frac{a\cos\theta-b}{a-b\cos\theta}}.
> $$
> If the denominator is zero, this formula cannot be used; instead, check whether the relation has a vertical tangent when viewed as a curve in $b$ and $a$.
> <!-- bilingual-en:end -->

> [!example]- 1G-4、1G-5b：一般高阶公式
> **1G-4。** $y=(x+1)^{-1}$。每求一次导数，系数再乘下一个负整数，幂次减一：
> $$
> \boxed{y^{(n)}=(-1)^n n!(x+1)^{-n-1}}
> =\boxed{\frac{(-1)^n n!}{(x+1)^{n+1}}}.
> $$
> 可用数学归纳法检查：假设第 $n$ 阶成立，再求导得到 $(-1)^{n+1}(n+1)!(x+1)^{-n-2}$。
>
> **1G-5b。** Leibniz 公式
> $$
> (uv)^{(N)}=\sum_{k=0}^N\binom Nk u^{(k)}v^{(N-k)}.
> $$
> 对 $u=x^p,v=(1+x)^q,N=p+q$，若 $k>p$ 则 $u^{(k)}=0$；若 $N-k>q$，即 $k<p$，则 $v^{(N-k)}=0$。只有 $k=p$ 留下：
> $$
> y^{(p+q)}=\binom{p+q}{p}p!q!=\boxed{(p+q)!}.
> $$
> <!-- bilingual-en:start -->
> **1G-4.** Let $y=(x+1)^{-1}$. Each differentiation multiplies the coefficient by the next negative integer and lowers the power by one:
> $$
> \boxed{y^{(n)}=(-1)^n n!(x+1)^{-n-1}}
> =\boxed{\frac{(-1)^n n!}{(x+1)^{n+1}}}.
> $$
> This can be checked by induction: assuming the formula at order $n$, one more differentiation gives $(-1)^{n+1}(n+1)!(x+1)^{-n-2}$.
>
> **1G-5b.** Leibniz's formula is
> $$
> (uv)^{(N)}=\sum_{k=0}^N\binom Nk u^{(k)}v^{(N-k)}.
> $$
> Take $u=x^p$, $v=(1+x)^q$, and $N=p+q$. If $k>p$, then $u^{(k)}=0$; if $N-k>q$, equivalently $k<p$, then $v^{(N-k)}=0$. Only $k=p$ remains, giving
> $$
> y^{(p+q)}=\binom{p+q}{p}p!q!=\boxed{(p+q)!}.
> $$
> <!-- bilingual-en:end -->

> [!example]- 5A-1：反三角主值与三角形
> **(a)** $\arctan\sqrt3=\boxed{\pi/3}$，因为主值在 $(-\pi/2,\pi/2)$。
>
> **(b)** $\arcsin(\sqrt3/2)=\boxed{\pi/3}$，因为主值在 $[-\pi/2,\pi/2]$。
>
> **(c)** 若 $\theta=\arctan5$，主值在第一象限。取邻边 $1$、对边 $5$，斜边 $\sqrt{26}$：
> $$
> \boxed{\sin\theta=\frac5{\sqrt{26}}},\qquad
> \boxed{\cos\theta=\frac1{\sqrt{26}}},\qquad
> \boxed{\sec\theta=\sqrt{26}}.
> $$
> <!-- bilingual-en:start -->
> **(a)** $\arctan\sqrt3=\boxed{\pi/3}$ because the principal-value range is $(-\pi/2,\pi/2)$.
>
> **(b)** $\arcsin(\sqrt3/2)=\boxed{\pi/3}$ because the principal-value range is $[-\pi/2,\pi/2]$.
>
> **(c)** If $\theta=\arctan5$, its principal value lies in the first quadrant. Use a right triangle with adjacent side $1$, opposite side $5$, and hypotenuse $\sqrt{26}$:
> $$
> \boxed{\sin\theta=\frac5{\sqrt{26}}},\qquad
> \boxed{\cos\theta=\frac1{\sqrt{26}}},\qquad
> \boxed{\sec\theta=\sqrt{26}}.
> $$
> <!-- bilingual-en:end -->

> [!example]- 5A-3f、g、h：复合反三角导数
> **(f)** $y=\arcsin(a/x)$。在课本默认 $a>0,x>a$ 的区间：
> $$
> y'=\frac{-a/x^2}{\sqrt{1-a^2/x^2}}
> =\boxed{-\frac{a}{x\sqrt{x^2-a^2}}}.
> $$
> 若允许 $x<-a$，化简平方根时应保留 $|x|$，写成 $-a/(|x|\sqrt{x^2-a^2})$。
>
> **(g)** 令 $u=x/\sqrt{1-x^2}$。先算
> $$u'=(1-x^2)^{-3/2},\qquad 1+u^2=\frac1{1-x^2}.$$
> 因而
> $$
> \frac d{dx}\arctan u
> =\frac{u'}{1+u^2}
> =\boxed{\frac1{\sqrt{1-x^2}}},\quad |x|<1.
> $$
> 它与 $(\arcsin x)'$ 相同；在主值范围内两函数本来就相等。
>
> **(h)** $y=\arcsin\sqrt{1-x}$，$0<x<1$：
> $$
> y'=\frac{-1/(2\sqrt{1-x})}{\sqrt{1-(1-x)}}
> =\boxed{-\frac1{2\sqrt{x(1-x)}}}.
> $$
> <!-- bilingual-en:start -->
> **(f)** Let $y=\arcsin(a/x)$. On the interval assumed in the textbook, where $a>0$ and $x>a$,
> $$
> y'=\frac{-a/x^2}{\sqrt{1-a^2/x^2}}
> =\boxed{-\frac{a}{x\sqrt{x^2-a^2}}}.
> $$
> If $x<-a$ is also allowed, retain $|x|$ when simplifying the square root; the derivative is $-a/(|x|\sqrt{x^2-a^2})$.
>
> **(g)** Set $u=x/\sqrt{1-x^2}$. First compute
> $$u'=(1-x^2)^{-3/2},\qquad 1+u^2=\frac1{1-x^2}.$$
> Hence,
> $$
> \frac d{dx}\arctan u
> =\frac{u'}{1+u^2}
> =\boxed{\frac1{\sqrt{1-x^2}}},\quad |x|<1.
> $$
> This is the same as $(\arcsin x)'$; on the principal-value interval, the two functions themselves are equal.
>
> **(h)** For $y=\arcsin\sqrt{1-x}$ with $0<x<1$,
> $$
> y'=\frac{-1/(2\sqrt{1-x})}{\sqrt{1-(1-x)}}
> =\boxed{-\frac1{2\sqrt{x(1-x)}}}.
> $$
> <!-- bilingual-en:end -->

> [!example]- 1H-1、1H-2：半衰期与 pH
> **1H-1a。** 放射性衰减 $y=y_0e^{-kt}$。半衰期 $\lambda$ 满足
> $$
> \frac{y_0}{2}=y_0e^{-k\lambda}
> \Longrightarrow -\ln2=-k\lambda,
> $$
> 故
> $$\boxed{\lambda=\frac{\ln2}{k}}.$$
> **(b)** 若 $y(t_1)=y_1$，则
> $$
> y(t_1+\lambda)=y_0e^{-kt_1}e^{-k\lambda}
> =y_1e^{-\ln2}=\boxed{y_1/2}.
> $$
> 所以半衰期与起始观察时刻无关，这是指数衰减的乘法结构。
>
> **1H-2。** $\mathrm{pH}=-\log_{10}[H^+]$。等体积水稀释使浓度减半：
> $$
> \mathrm{pH}_{\rm dil}
> =-\log_{10}\left(\frac{[H^+]_{\rm orig}}2\right)
> =\mathrm{pH}_{\rm orig}+\log_{10}2.
> $$
> 即 pH 约增加
> $$\boxed{0.301}.$$
> <!-- bilingual-en:start -->
> **1H-1(a).** Radioactive decay is modeled by $y=y_0e^{-kt}$. The half-life $\lambda$ satisfies
> $$
> \frac{y_0}{2}=y_0e^{-k\lambda}
> \Longrightarrow -\ln2=-k\lambda,
> $$
> so
> $$\boxed{\lambda=\frac{\ln2}{k}}.$$
> **(b)** If $y(t_1)=y_1$, then
> $$
> y(t_1+\lambda)=y_0e^{-kt_1}e^{-k\lambda}
> =y_1e^{-\ln2}=\boxed{y_1/2}.
> $$
> Thus the half-life does not depend on when observation begins; this follows from the multiplicative structure of exponential decay.
>
> **1H-2.** Since $\mathrm{pH}=-\log_{10}[H^+]$, dilution with an equal volume of water halves the concentration:
> $$
> \mathrm{pH}_{\rm dil}
> =-\log_{10}\left(\frac{[H^+]_{\rm orig}}2\right)
> =\mathrm{pH}_{\rm orig}+\log_{10}2.
> $$
> The pH therefore increases by approximately
> $$\boxed{0.301}.$$
> <!-- bilingual-en:end -->

> [!example]- 1H-3a、1H-5b：解对数和指数方程
> **1H-3a。** 已在 Session 17 提示定义域：$y>1,x>0$。合并对数并指数化：
> $$
> (y+1)(y-1)=xe^{2x}
> \Longrightarrow
> \boxed{y=\sqrt{xe^{2x}+1}}.
> $$
> 负根因 $y>1$ 被排除。
>
> **1H-5b。** 解 $y=e^x+e^{-x}$。令 $u=e^x>0$：
> $$
> u+\frac1u=y
> \Longrightarrow
> u^2-yu+1=0.
> $$
> 所以
> $$
> u=\frac{y\pm\sqrt{y^2-4}}2,
> \qquad
> \boxed{x=\ln\frac{y\pm\sqrt{y^2-4}}2}.
> $$
> 实数解要求 $y\ge2$；两根互为倒数，对应 $x$ 与 $-x$，符合 $e^x+e^{-x}=2\cosh x$ 为偶函数。
> <!-- bilingual-en:start -->
> **1H-3(a).** Session 17 already established the domain restrictions $y>1$ and $x>0$. Combining the logarithms and exponentiating gives
> $$
> (y+1)(y-1)=xe^{2x}
> \Longrightarrow
> \boxed{y=\sqrt{xe^{2x}+1}}.
> $$
> The negative root is excluded because $y>1$.
>
> **1H-5(b).** To solve $y=e^x+e^{-x}$, set $u=e^x>0$:
> $$
> u+\frac1u=y
> \Longrightarrow
> u^2-yu+1=0.
> $$
> Hence,
> $$
> u=\frac{y\pm\sqrt{y^2-4}}2,
> \qquad
> \boxed{x=\ln\frac{y\pm\sqrt{y^2-4}}2}.
> $$
> Real solutions require $y\ge2$. The two roots are reciprocals, corresponding to $x$ and $-x$, as expected because $e^x+e^{-x}=2\cosh x$ is even.
> <!-- bilingual-en:end -->

> [!example]- 1I-1：指数与对数求导
> 逐题标出规则与定义域：
>
> - **(c)** $D(e^{-x^2})=\boxed{-2xe^{-x^2}}$（链式法则）。
> - **(d)** $D(x\ln x-x)=\ln x+1-1=\boxed{\ln x}$，$x>0$（积法则）。
> - **(e)** $D\ln(x^2)=2x/x^2=\boxed{2/x}$，$x\ne0$；注意 $\ln(x^2)$ 在负 $x$ 也有定义。
> - **(f)** $D(\ln x)^2=\boxed{2\ln x/x}$，$x>0$。
> - **(m)**
> $$
> D\frac{1-e^x}{1+e^x}
> =\frac{-e^x(1+e^x)-(1-e^x)e^x}{(1+e^x)^2}
> =\boxed{-\frac{2e^x}{(1+e^x)^2}}.
> $$
>
> **1I-4a。**
> $$
> \lim_{n\to\infty}\left(1+\frac1n\right)^{3n}
> =\left[\lim_{n\to\infty}\left(1+\frac1n\right)^n\right]^3
> =\boxed{e^3}.
> $$
> <!-- bilingual-en:start -->
> The rule and domain for each part are:
>
> - **(c)** $D(e^{-x^2})=\boxed{-2xe^{-x^2}}$ by the chain rule.
> - **(d)** $D(x\ln x-x)=\ln x+1-1=\boxed{\ln x}$ for $x>0$, by the product rule.
> - **(e)** $D\ln(x^2)=2x/x^2=\boxed{2/x}$ for $x\ne0$; note that $\ln(x^2)$ is also defined for negative $x$.
> - **(f)** $D(\ln x)^2=\boxed{2\ln x/x}$ for $x>0$.
> - **(m)**
> $$
> D\frac{1-e^x}{1+e^x}
> =\frac{-e^x(1+e^x)-(1-e^x)e^x}{(1+e^x)^2}
> =\boxed{-\frac{2e^x}{(1+e^x)^2}}.
> $$
>
> **1I-4(a).**
> $$
> \lim_{n\to\infty}\left(1+\frac1n\right)^{3n}
> =\left[\lim_{n\to\infty}\left(1+\frac1n\right)^n\right]^3
> =\boxed{e^3}.
> $$
> <!-- bilingual-en:end -->

> [!example]- 5A-5：$\sinh$ 及其反函数
> **(a) 描图。**
> $$
> y=\sinh x,\quad y'=\cosh x>0,\quad y''=\sinh x.
> $$
> 因 $y'$ 从不为零，无临界点且严格递增；$y''$ 在 $0$ 由负变正，所以 $(0,0)$ 为拐点，斜率 $1$；函数为奇函数；$x\to\pm\infty$ 时分别趋 $\pm\infty$。
>
> **(b)** 因 $\sinh$ 在全实轴严格递增且值域为全实数，可定义反双曲正弦
> $$
> y=\operatorname{arsinh}x
> \Longleftrightarrow
> x=\sinh y.
> $$
> 其定义域、值域均为 $\mathbb R$，图像为 $\sinh$ 关于 $y=x$ 的反射，也是奇函数。
>
> **(c)** 对 $x=\sinh y$ 求导：
> $$
> 1=\cosh y\,y'
> \Longrightarrow y'=\frac1{\cosh y}.
> $$
> 因 $\cosh^2y=1+\sinh^2y=1+x^2$ 且 $\cosh y>0$，
> $$
> \boxed{\frac d{dx}\operatorname{arsinh}x=\frac1{\sqrt{1+x^2}}}.
> $$
> <!-- bilingual-en:start -->
> **(a) Sketching the graph.**
> $$
> y=\sinh x,\quad y'=\cosh x>0,\quad y''=\sinh x.
> $$
> Since $y'$ never vanishes, the function has no critical points and is strictly increasing. Because $y''$ changes from negative to positive at $0$, $(0,0)$ is an inflection point with slope $1$. The function is odd, and it tends to $\pm\infty$ as $x\to\pm\infty$, respectively.
>
> **(b)** Because $\sinh$ is strictly increasing on the whole real line and its range is all of $\mathbb R$, the inverse hyperbolic sine is well defined:
> $$
> y=\operatorname{arsinh}x
> \Longleftrightarrow
> x=\sinh y.
> $$
> Both its domain and range are $\mathbb R$. Its graph is the reflection of the graph of $\sinh$ across $y=x$, and it is also odd.
>
> **(c)** Differentiate $x=\sinh y$:
> $$
> 1=\cosh y\,y'
> \Longrightarrow y'=\frac1{\cosh y}.
> $$
> Since $\cosh^2y=1+\sinh^2y=1+x^2$ and $\cosh y>0$,
> $$
> \boxed{\frac d{dx}\operatorname{arsinh}x=\frac1{\sqrt{1+x^2}}}.
> $$
> <!-- bilingual-en:end -->

> [!warning] Problem Set 2 常见错误
> - 反三角题忽略主值范围；把 $\sqrt{x^2}$ 直接写成 $x$ 而非 $|x|$。
> - 对数方程不先检查真数为正；二次方程两根不回代定义域。
> - 变量幂只对底数或指数的一方求导；求出 $y'/y$ 后忘记乘回 $y$。
> - 隐式题除以可能为零的因子而不记录；水平切线只令分子为零却不检查分母。
> <!-- bilingual-en:start -->
> - Ignoring principal-value ranges in inverse-trigonometric problems, or replacing $\sqrt{x^2}$ with $x$ instead of $|x|$.
> - Solving a logarithmic equation without first requiring every logarithm's argument to be positive, or failing to test both roots of a quadratic against the domain.
> - Differentiating only the base or only the exponent in a variable-power expression, or forgetting to multiply by $y$ after finding $y'/y$.
> - Dividing an implicit equation by a factor that might be zero without recording the lost case, or setting only the numerator of $dy/dx$ to zero for a horizontal tangent without checking the denominator.
> <!-- bilingual-en:end -->

**Problem Set 2 小结：**Part B 的核心不是多记几个公式，而是把链式法则用于“没有显式写出的依赖关系”：$y(x)$、反函数、对数后的变量幂，以及由指数定义的双曲函数。
<!-- bilingual-en:start -->
**Problem Set 2 summary:** The core of Part B is not memorising more formulas, but applying the chain rule to dependencies that are not written explicitly: $y(x)$ in an implicit relation, inverse functions, variable powers after taking logarithms, and hyperbolic functions defined through exponentials.
<!-- bilingual-en:end -->

---

## Exam 1

## Session 21：Review for Exam 1

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题：**面对综合求导、定义证明、切线和分段函数题，怎样快速辨认结构并检查答案？
<!-- bilingual-en:start -->
**Question:** How can we quickly identify the relevant structure and check answers in mixed differentiation, definition-proof, tangent-line, and piecewise-function problems?
<!-- bilingual-en:end -->

**前置知识：**Session 1-20 全部内容。目标不是再添新规则，而是建立调用顺序。
<!-- bilingual-en:start -->
**Prerequisites:** Sessions 1–20. The goal is not to add another rule, but to establish a reliable order of attack.
<!-- bilingual-en:end -->

### 21a：公式总表与隐式流程
<!-- bilingual-en:start -->
*21a: Formula Summary and an Implicit-Differentiation Workflow*
<!-- bilingual-en:end -->

$$
\begin{aligned}
&(u+v)'=u'+v',\qquad(cu)'=cu',\\
&(uv)'=u'v+uv',\\
&\left(\frac uv\right)'=\frac{u'v-uv'}{v^2},\\
&(f(g(x)))'=f'(g(x))g'(x),\\
&(x^r)'=rx^{r-1},\\
&(\sin x)'=\cos x,\quad(\cos x)'=-\sin x,\\
&(\tan x)'=\sec^2x,\quad(\sec x)'=\sec x\tan x,\\
&(e^x)'=e^x,\quad(a^x)'=a^x\ln a,\\
&(\ln x)'=1/x,\\
&(\arcsin x)'=1/\sqrt{1-x^2},\quad
(\arctan x)'=1/(1+x^2).
\end{aligned}
$$

隐式例 $y^3+3xy^2=8$：
<!-- bilingual-en:start -->
Implicit case $y^3+3xy^2=8$:
<!-- bilingual-en:end -->

$$
3y^2y'+3y^2+6xyy'=0,
$$

所以
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

$$
\boxed{y'=-\frac{3y^2}{3y^2+6xy}}.
$$

### 21b：链式法则为何是乘法
<!-- bilingual-en:start -->
*21b: Why the Chain Rule Multiplies Rates*
<!-- bilingual-en:end -->

若 $y=10x+b$，则 $dy/dx=10$；若 $x=5t+a$，则 $dx/dt=5$。代入得到
<!-- bilingual-en:start -->
If $y=10x+b$, then $dy/dx=10$; if $x=5t+a$, then $dx/dt=5$. Substitution gives
<!-- bilingual-en:end -->

$$
y=50t+(10a+b),
$$

故 $dy/dt=50=10\cdot5$。这是“输出相对中间量的倍率 × 中间量相对输入的倍率”。链式法则还可把商改写为积：
<!-- bilingual-en:start -->
Hence $dy/dt=50=10\cdot5$: the rate of output change per unit of the intermediate variable is multiplied by the rate of intermediate-variable change per unit of input. The chain rule also lets us rewrite a quotient as a product:
<!-- bilingual-en:end -->

$$
\left(\frac uv\right)'=(uv^{-1})'
=u'v^{-1}-uv^{-2}v'
=\frac{u'v-uv'}{v^2}.
$$

### 21c-21f：按真实片段顺序的四个例子
<!-- bilingual-en:start -->
*21c–21f: Four Examples in the Actual Slide Sequence*
<!-- bilingual-en:end -->

**21c - $\sec x$：**

$$
(\sec x)'=(\cos x)^{-1}'
=-(\cos x)^{-2}(-\sin x)
=\boxed{\sec x\tan x}.
$$

**21d - $\ln(\sec x)$：**

$$
\frac d{dx}\ln(\sec x)
=\frac{\sec x\tan x}{\sec x}
=\boxed{\tan x}.
$$

**21e - $(x^{10}+8x)^6$：**

$$
\frac d{dx}(x^{10}+8x)^6
=\boxed{6(x^{10}+8x)^5(10x^9+8)}.
$$

没有要求展开时，因式形式更容易检查且信息更清楚。
<!-- bilingual-en:start -->
When expansion is not required, the factored form is easier to check and preserves more structural information.
<!-- bilingual-en:end -->

**21f - $e^{x\arctan x}$：** 课件实际是整个 $x\arctan x$ 位于指数中，不是 $e^x\arctan x$。设 $u=x\arctan x$：
<!-- bilingual-en:start -->
**21f — $e^{x\arctan x}$:** In the slide, the entire expression $x\arctan x$ is in the exponent; the function is not $e^x\arctan x$. Set $u=x\arctan x$:
<!-- bilingual-en:end -->

$$
u'=\arctan x+\frac{x}{1+x^2}.
$$

于是
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

$$
\boxed{
\frac d{dx}e^{x\arctan x}
=e^{x\arctan x}\left(\arctan x+\frac{x}{1+x^2}\right)}.
$$

### 21g：定义、极限、反函数与图像复习
<!-- bilingual-en:start -->
*21g: Review of definitions, limits, inverse functions, and graphs*
<!-- bilingual-en:end -->

考试可能把导数定义伪装成极限。例如
<!-- bilingual-en:start -->
An exam may disguise a derivative definition as a limit. For example,
<!-- bilingual-en:end -->

$$
\lim_{u\to0}\frac{e^u-1}{u}
=\left.\frac d{dx}e^x\right|_{x=0}=1.
$$

看到
<!-- bilingual-en:start -->
Similarly, the expression
<!-- bilingual-en:end -->

$$
\lim_{h\to0}\frac{f(a+h)-f(a)}h
$$

应立即识别为 $f'(a)$。若问从图像判断可导性，要比较左右割线极限：跳跃、尖角、尖点、竖直切线均不具有有限且相同的双侧导数。
<!-- bilingual-en:start -->
This should be recognised immediately as $f'(a)$. To judge differentiability from a graph, compare the one-sided limits of the secant slopes: jumps, corners, cusps, and vertical tangents all fail to produce one common finite two-sided derivative.
<!-- bilingual-en:end -->

### 考场决策顺序
<!-- bilingual-en:start -->
*Exam-Day Decision Sequence*
<!-- bilingual-en:end -->

1. 先写原函数定义域，标出不可求导点。
2. 圈出最外层运算：和、积、商还是复合。
3. 复合函数从外向内逐层；隐函数每个 $y$ 项都带链式因子。
4. 切线题写“点 + 斜率”；运动题先找速度零点再分段算路程。
5. 最后检查奇偶性、符号、单位、简单点和是否漏了定义域。
<!-- bilingual-en:start -->

&nbsp;
**1.** Write down the domain of the function and mark every point at which it is not differentiable.<br>
**2.** Circle the outermost operation: sum, product, quotient, or composite.<br>
**3.** Differentiate a composite function from the outside inward; in implicit differentiation, every differentiated term involving $y$ carries a chain-rule factor $y'$.<br>
**4.** For a tangent line, write “point + slope.” For a motion problem, first find the times at which velocity is zero, then compute distance piecewise.<br>
**5.** Finally, check parity, signs, units, simple test points, and whether any domain restrictions were omitted.<br>
<!-- bilingual-en:end -->

### 三道自检
<!-- bilingual-en:start -->
*Three Quick Self-Checks*
<!-- bilingual-en:end -->

1. 求 $D[e^{x\arctan x}]$，并说明用了哪两层规则。
2. 把 $\lim_{h\to0}[\ln(2+h)-\ln2]/h$ 识别为导数并求值。
3. 分段函数在接点处可导需要哪两个独立条件？
<!-- bilingual-en:start -->

&nbsp;
**1.** Find $D[e^{x\arctan x}]$ and explain which rules are used at the two layers.<br>
**2.** Recognize $\lim_{h\to0}[\ln(2+h)-\ln2]/h$ as a derivative and evaluate.<br>
**3.** What two independent conditions are required for a piecewise function to be differentiable at a joining point?<br>
<!-- bilingual-en:end -->

> [!success]- 自检答案
> 1. 正文 21f；外层用指数链式法则，指数内部用积法则和 $\arctan$ 导数。
> 2. 是 $(\ln x)'|_{x=2}=1/2$。
> 3. 先连续：左右极限和函数值相等；再匹配左右导数。第二条件不能替代第一条件。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** See Section 21f: use the chain rule for the outer exponential, then the product rule and the derivative of $\arctan$ inside the exponent.<br>
> **2.** It is $(\ln x)'|_{x=2}=1/2$.<br>
> **3.** First establish continuity by matching both one-sided limits with the function value; then match the one-sided derivatives. The second condition cannot replace the first.<br>
> <!-- bilingual-en:end -->

### 本地材料

- [[Ses21a_Lecture_Notes.pdf#page=1|21a Differentiation Formulas（pp.1-2）]]
- [[Ses21b_Lecture_Notes.pdf#page=1|21b Chain Rule Revisited（p.1）]]
- [[Ses21c_Lecture_Notes.pdf#page=1|21c Derivative of Secant（p.1）]]
- [[Ses21d_Lecture_Notes.pdf#page=1|21d Derivative of $\ln(\sec x)$（p.1）]]
- [[Ses21e_Lecture_Notes.pdf#page=1|21e Derivative of $(x^{10}+8x)^6$（p.1）]]
- [[Ses21f_Lecture_Notes.pdf#page=1|21f Derivative of $e^{x\arctan x}$（p.1）]]
- [[Ses21g_MIT18_01SCF10_Ses21g.pdf#page=1|21g Exam 1 Review Continued（pp.1-3）]]

**知识链：**公式识别 → 外层结构决策 → 定义与规则双向使用 → 定义域、图像和单位检查。
<!-- bilingual-en:start -->
**Knowledge chain:** recognize the formula → identify the outer structure → move in both directions between definitions and rules → check domains, graphs, and units.
<!-- bilingual-en:end -->

## Session 22：Materials for Exam 1

### 本节问题与作答标准
<!-- bilingual-en:start -->
*Questions and Answer Standards for This Section*
<!-- bilingual-en:end -->

**问题：**怎样把本章的定义、规则、隐式关系和运动解释组合成一套完整考试解答？
<!-- bilingual-en:start -->
**Question:** How do you combine the definitions, rules, implicit relationships, and motion explanations of this chapter into a complete set of exam answers?
<!-- bilingual-en:end -->

**作答标准：**每题都写“已知/目标 → 选择规则 → 逐步计算 → 定义域或几何检查 → 最终答案”。本地 Session 22 没有 `Ses22` 讲义；按官方结构，本节由 Exam 1 原题与官方答案构成。
<!-- bilingual-en:start -->
**Answer standard:** For each question, write “known information and target → selected rule → step-by-step calculation → domain or geometric check → final answer.” There is no local `Ses22` handout; in the official structure, Session 22 consists of the original Exam 1 paper and its official solutions.
<!-- bilingual-en:end -->

- [[Exam1_Problems.pdf#page=1|Exam 1 Problems（pp.1-7）]]
- [[Exam1_Solutions.pdf#page=1|Exam 1 Official Solutions（pp.1-8）]]

### Problem 1：计算导数
<!-- bilingual-en:start -->
*Problem 1: Calculating Derivatives*
<!-- bilingual-en:end -->

#### 1(a) $f(x)=x/(1-x^2)$

**规则。** 商法则；原定义域 $x\ne\pm1$。
<!-- bilingual-en:start -->
**Rule.** Apply the quotient rule and retain the original domain restriction $x\ne\pm1$.
<!-- bilingual-en:end -->

$$
\begin{aligned}
f'(x)
&=\frac{1(1-x^2)-x(-2x)}{(1-x^2)^2}\\
&=\frac{1-x^2+2x^2}{(1-x^2)^2}\\
&=\boxed{\frac{1+x^2}{(1-x^2)^2}}.
\end{aligned}
$$

**检查。** 分子、分母在定义域内均正，因此 $f'>0$；原函数在每个定义域区间单调增加，与图像一致。答案不能把 $x=\pm1$ 补回。
<!-- bilingual-en:start -->
**Check.** The numerator and denominator are both positive throughout the domain, so $f'>0$. The function is increasing on each interval of its domain, consistent with its graph. The answer must not add $x=\pm1$ back into the domain.
<!-- bilingual-en:end -->

#### 1(b) $f(x)=\ln(\cos x)-\frac12\sin^2x$

**规则。** 对数链式法则 + 幂的链式法则：
<!-- bilingual-en:start -->
**Rule.** Apply the logarithmic chain rule and the power-chain rule:
<!-- bilingual-en:end -->

$$
\begin{aligned}
f'(x)
&=\frac{-\sin x}{\cos x}
-\frac12\cdot2\sin x\cos x\\
&=-\tan x-\sin x\cos x\\
&=-\sin x\left(\frac1{\cos x}+\cos x\right)\\
&=\boxed{-\sin x\frac{1+\cos^2x}{\cos x}}.
\end{aligned}
$$

**定义域。** 原函数要求 $\cos x>0$，而不仅是 $\cos x\ne0$；在这些开区间内上述形式合法。$x=0$ 时导数为 $0$，与两个偶函数之差仍为偶函数、其导数在原点为零相符。
<!-- bilingual-en:start -->
**Domain.** The function requires $\cos x>0$, not merely $\cos x\ne0$, so the displayed formulas are valid on those open intervals. At $x=0$ the derivative is $0$, as expected because the difference of two even functions is even and therefore has derivative zero at the origin.
<!-- bilingual-en:end -->

#### 1(c) $f(x)=xe^x$ 的五阶导数
<!-- bilingual-en:start -->
*1(c) Fifth derivative of $f(x)=xe^x$*
<!-- bilingual-en:end -->

先逐阶观察：
<!-- bilingual-en:start -->
First inspect the pattern in the successive derivatives:
<!-- bilingual-en:end -->

$$
\begin{aligned}
f'&=(x+1)e^x,\\
f''&=(x+2)e^x,\\
f^{(3)}&=(x+3)e^x.
\end{aligned}
$$

提出一般命题 $f^{(n)}=(x+n)e^x$。若第 $n$ 阶成立，则
<!-- bilingual-en:start -->
Conjecture that $f^{(n)}=(x+n)e^x$. If the formula holds at order $n$, then
<!-- bilingual-en:end -->

$$
f^{(n+1)}=e^x+(x+n)e^x=(x+n+1)e^x,
$$

所以由归纳法成立。取 $n=5$：
<!-- bilingual-en:start -->
This proves the formula by induction. Taking $n=5$ gives
<!-- bilingual-en:end -->

$$
\boxed{f^{(5)}(x)=(x+5)e^x}.
$$

也可用 Leibniz 公式：$x$ 的二阶及以上导数为零，只留下 $xe^x$ 与 $5e^x$ 两项。
<!-- bilingual-en:start -->
You can also use the Leibniz formula: $x$ has zero second-order and higher derivatives, leaving only $xe^x$ and $5e^x$.
<!-- bilingual-en:end -->

### Problem 2：星形线（astroid）的切线
<!-- bilingual-en:start -->
*Problem 2: Tangent to an Astroid*
<!-- bilingual-en:end -->

曲线
<!-- bilingual-en:start -->
Consider the curve
<!-- bilingual-en:end -->

$$
x^{2/3}+y^{2/3}=4
$$

在 $(-\sqrt{27},1)$ 处。先验点：$-\sqrt{27}=-3^{3/2}$，故
<!-- bilingual-en:start -->
at $(-\sqrt{27},1)$. First verify the point: since $-\sqrt{27}=-3^{3/2}$,
<!-- bilingual-en:end -->

$$
(-3^{3/2})^{2/3}+1^{2/3}=3+1=4.
$$

隐式求导：
<!-- bilingual-en:start -->
Implicit differentiation gives
<!-- bilingual-en:end -->

$$
\frac23x^{-1/3}+\frac23y^{-1/3}y'=0.
$$

解出
<!-- bilingual-en:start -->
Solving for $y'$ gives
<!-- bilingual-en:end -->

$$
y'=-\frac{x^{-1/3}}{y^{-1/3}}.
$$

在指定点，$x^{1/3}=-\sqrt3$，所以 $x^{-1/3}=-1/\sqrt3$，而 $y^{-1/3}=1$：
<!-- bilingual-en:start -->
At the specified point, $x^{1/3}=-\sqrt3$, so $x^{-1/3}=-1/\sqrt3$, and $y^{-1/3}=1$:
<!-- bilingual-en:end -->

$$
m=\frac1{\sqrt3}.
$$

点斜式：
<!-- bilingual-en:start -->
The point-slope form is
<!-- bilingual-en:end -->

$$
y-1=\frac1{\sqrt3}(x+\sqrt{27}).
$$

因 $\sqrt{27}/\sqrt3=3$，
<!-- bilingual-en:start -->
Because $\sqrt{27}/\sqrt3=3$,
<!-- bilingual-en:end -->

$$
\boxed{y=\frac{x}{\sqrt3}+4}.
$$

**检查。** 代切点：$(-\sqrt{27})/\sqrt3+4=-3+4=1$；切线确实过指定点。分数幂在负数处按实立方根理解，不能错误地把 $x^{1/3}$ 取成正根。
<!-- bilingual-en:start -->
**Check.** At the specified point, $(-\sqrt{27})/\sqrt3+4=-3+4=1$, so the tangent line does pass through it. For negative inputs, the fractional power is interpreted using the real cube root; $x^{1/3}$ must not be treated as a nonnegative principal square root.
<!-- bilingual-en:end -->

### Problem 3：前三秒的总路程
<!-- bilingual-en:start -->
*Problem 3: Total Distance During the First Three Seconds*
<!-- bilingual-en:end -->

位置
<!-- bilingual-en:start -->
The position is
<!-- bilingual-en:end -->

$$
y(t)=t^3-3t+3,\qquad t\ge0.
$$

速度
<!-- bilingual-en:start -->
The velocity is
<!-- bilingual-en:end -->

$$
v(t)=y'(t)=3t^2-3=3(t-1)(t+1).
$$

在 $[0,3]$ 内只有 $t=1$ 使速度为零；$0<t<1$ 时 $v<0$，$t>1$ 时 $v>0$，所以粒子在 $t=1$ 改变方向。各关键位置：
<!-- bilingual-en:start -->
Within $[0,3]$, velocity vanishes only at $t=1$. Since $v<0$ for $0<t<1$ and $v>0$ for $t>1$, the particle changes direction at $t=1$. The key positions are
<!-- bilingual-en:end -->

$$
y(0)=3,\qquad y(1)=1,\qquad y(3)=21.
$$

总路程是分段位移绝对值之和：
<!-- bilingual-en:start -->
The total distance is the sum of the absolute values of the piecewise displacements:
<!-- bilingual-en:end -->

$$
|y(1)-y(0)|+|y(3)-y(1)|
=|1-3|+|21-1|
=2+20.
$$

$$
\boxed{\text{总路程}=22\text{ m}}.
$$

**常见错误。** 净位移是 $y(3)-y(0)=18$ 米，不是总路程；必须先用速度零点确定是否改变方向。
<!-- bilingual-en:start -->
**Common error.** The net displacement $y(3)-y(0)=18$ metres is not the total distance. First use the zeros of velocity to determine whether the motion changes direction.
<!-- bilingual-en:end -->

### Problem 4：由定义证明积法则
<!-- bilingual-en:start -->
*Problem 4: Prove the Product Rule from the Definition*
<!-- bilingual-en:end -->

**定理。** 若 $f,g$ 在 $x$ 可导，则
<!-- bilingual-en:start -->
**Theorem.** If $f$ and $g$ are differentiable at $x$, then
<!-- bilingual-en:end -->

$$
(fg)'(x)=f'(x)g(x)+f(x)g'(x).
$$

**证明目标。** 把积的差商变成 $f,g$ 的两个标准差商。
<!-- bilingual-en:start -->
**Proof strategy.** Rewrite the product's difference quotient in terms of the two standard difference quotients for $f$ and $g$.
<!-- bilingual-en:end -->

**构造。** 加减中间项 $f(x)g(x+h)$：
<!-- bilingual-en:start -->
**Construction.** Add and subtract the intermediate term $f(x)g(x+h)$:
<!-- bilingual-en:end -->

$$
\begin{aligned}
(fg)'(x)
&=\lim_{h\to0}\frac{f(x+h)g(x+h)-f(x)g(x)}h\\
&=\lim_{h\to0}\frac{f(x+h)g(x+h)-f(x)g(x+h)}h\\
&\quad+\lim_{h\to0}\frac{f(x)g(x+h)-f(x)g(x)}h\\
&=\lim_{h\to0}left[
\frac{f(x+h)-f(x)}h g(x+h)
+f(x)\frac{g(x+h)-g(x)}h
\right].
\end{aligned}
$$

**逐步依据。** 因 $g$ 可导，所以 $g$ 在 $x$ 连续，$g(x+h)\to g(x)$；两个差商分别趋于 $f'(x),g'(x)$。由极限和、积法则：
<!-- bilingual-en:start -->
**Justification.** Because $g$ is differentiable, it is continuous at $x$, so $g(x+h)\to g(x)$. The two difference quotients tend to $f'(x)$ and $g'(x)$, respectively. The sum and product laws for limits then give
<!-- bilingual-en:end -->

$$
\boxed{(fg)'(x)=f'(x)g(x)+f(x)g'(x)}.
$$

**边界说明。** 中间没有除以 $f$ 或 $g$，所以不要求它们非零。可导性不仅提供差商极限，也通过“可导蕴含连续”提供 $g(x+h)\to g(x)$。
<!-- bilingual-en:start -->
**Boundary conditions.** The proof never divides by $f$ or $g$, so neither function is required to be nonzero. Differentiability supplies both the difference-quotient limits and, through “differentiability implies continuity,” the fact that $g(x+h)\to g(x)$.
<!-- bilingual-en:end -->

### Problem 5：分段函数能否处处可导
<!-- bilingual-en:start -->
*Problem 5: Whether a Piecewise Function Is Differentiable Everywhere*
<!-- bilingual-en:end -->

$$
f(x)=
\begin{cases}
\arctan x,&x\le0,\\
ax^2+bx+c,&0<x<2,\\
x^3-\frac14x^2+5,&x\ge2.
\end{cases}
$$

只需检查连接点 $0,2$。
<!-- bilingual-en:start -->
It is enough to check the joining points $0$ and $2$.
<!-- bilingual-en:end -->

**第一步：连续性。** 在 $0$：
<!-- bilingual-en:start -->
**Step 1: Continuity.** At $x=0$,
<!-- bilingual-en:end -->

$$
c=\arctan0=0.
$$

在 $2$，右段函数值
<!-- bilingual-en:start -->
At $x=2$, the right-hand piece has value
<!-- bilingual-en:end -->

$$
2^3-\frac14(2^2)+5=8-1+5=12.
$$

所以
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

$$
4a+2b+c=12
\quad\Longrightarrow\quad
2a+b=6.
$$

**第二步：$x=2$ 处导数匹配。** 中段左导数 $4a+b$；右段导数
<!-- bilingual-en:start -->
**Step 2: Match derivatives at $x=2$.** The derivative from the middle piece is $4a+b$; the derivative of the right-hand piece is
<!-- bilingual-en:end -->

$$
3x^2-\frac12x
$$

在 $2$ 为 $12-1=11$。故
<!-- bilingual-en:start -->
which equals $12-1=11$ at $x=2$. Therefore,
<!-- bilingual-en:end -->

$$
4a+b=11.
$$

联立 $2a+b=6$：
<!-- bilingual-en:start -->
Combining this with $2a+b=6$ gives
<!-- bilingual-en:end -->

$$
2a=5\Longrightarrow a=\frac52,\qquad b=1.
$$

**第三步：回查 $x=0$ 的导数。** 中段右导数为 $b=1$；左段
<!-- bilingual-en:start -->
**Step 3: Recheck derivatives at $x=0$.** The right-hand derivative from the middle piece is $b=1$; the derivative of the left-hand piece is
<!-- bilingual-en:end -->

$$
(\arctan x)'|_{x=0}=\frac1{1+0^2}=1.
$$

左右相等。因此答案确实存在：
<!-- bilingual-en:start -->
The one-sided derivatives agree, so a solution does exist:
<!-- bilingual-en:end -->

$$
\boxed{a=\frac52,\qquad b=1,\qquad c=0}.
$$

**错误诊断。** 只解 $x=2$ 的两条条件会得到参数，却仍必须回查 $x=0$；本题恰好通过，不代表一般题会自动通过。
<!-- bilingual-en:start -->
**Diagnostic.** The two conditions at $x=2$ determine the parameters, but $x=0$ must still be checked. It happens to pass here; that does not mean an unchecked joining point will pass in general.
<!-- bilingual-en:end -->

### Problem 6：函数方程与导数定义
<!-- bilingual-en:start -->
*Problem 6: Function Equations and Derivative Definitions*
<!-- bilingual-en:end -->

已知对所有实数 $x,y$：
<!-- bilingual-en:start -->
Suppose that, for all real $x$ and $y$,
<!-- bilingual-en:end -->

$$
f(x+y)=f(x)+f(y)+x^2y+xy^2,
$$

且
<!-- bilingual-en:start -->
and
<!-- bilingual-en:end -->

$$
\lim_{x\to0}\frac{f(x)}x=1.
$$

#### 6(a) 求 $f(0)$
<!-- bilingual-en:start -->
*6(a) Find $f(0)$*
<!-- bilingual-en:end -->

令 $x=y=0$：
<!-- bilingual-en:start -->
Let $x=y=0$:
<!-- bilingual-en:end -->

$$
f(0)=2f(0)
\quad\Longrightarrow\quad
\boxed{f(0)=0}.
$$

极限条件也说明 $f(x)=x[f(x)/x]\to0$，与结果相容。
<!-- bilingual-en:start -->
The limit condition also indicates that $f(x)=x[f(x)/x]\to0$ is compatible with the results.
<!-- bilingual-en:end -->

#### 6(b) 求 $f'(0)$
<!-- bilingual-en:start -->
*6(b) Find $f'(0)$*
<!-- bilingual-en:end -->

直接使用定义和 (a)：
<!-- bilingual-en:start -->
Use the definition directly together with part (a):
<!-- bilingual-en:end -->

$$
f'(0)=\lim_{h\to0}\frac{f(h)-f(0)}h
=\lim_{h\to0}\frac{f(h)}h
=\boxed{1}.
$$

#### 6(c) 求 $f'(x)$
<!-- bilingual-en:start -->
*6(c) Find $f'(x)$*
<!-- bilingual-en:end -->

在函数方程中把第二个变量取为 $h$：
<!-- bilingual-en:start -->
In the function equation, take the second variable as $h$:
<!-- bilingual-en:end -->

$$
f(x+h)=f(x)+f(h)+x^2h+xh^2.
$$

移项并除以 $h\ne0$：
<!-- bilingual-en:start -->
Rearrange and divide by $h\ne0$:
<!-- bilingual-en:end -->

$$
\frac{f(x+h)-f(x)}h
=\frac{f(h)}h+x^2+xh.
$$

令 $h\to0$，使用已知极限：
<!-- bilingual-en:start -->
Let $h\to0$ and use the known limit:
<!-- bilingual-en:end -->

$$
\begin{aligned}
f'(x)
&=1+x^2+0\\
&=\boxed{1+x^2}.
\end{aligned}
$$

**进一步检查。** 对 $f'(x)$ 求一个候选原函数 $f(x)=x+x^3/3+C$；由 $f(0)=0$ 得 $C=0$。代回函数方程：
<!-- bilingual-en:start -->
**Further check.** A candidate antiderivative of $f'(x)$ is $f(x)=x+x^3/3+C$; the condition $f(0)=0$ gives $C=0$. Substituting into the functional equation yields
<!-- bilingual-en:end -->

$$
\frac{(x+y)^3-x^3-y^3}{3}=x^2y+xy^2,
$$

恰好成立，验证答案一致。
<!-- bilingual-en:start -->
which holds identically and confirms the result.
<!-- bilingual-en:end -->

### Exam 1 三道收尾自检
<!-- bilingual-en:start -->
*Three Final Self-Checks for Exam 1*
<!-- bilingual-en:end -->

1. Problem 2 中若忘记先验点，会漏掉什么潜在问题？
2. Problem 5 为什么必须先连续、再匹配导数？
3. Problem 6(c) 中哪一步真正使用了额外极限条件？
<!-- bilingual-en:start -->

&nbsp;
**1.** What potential problem would be missed in Problem 2 if the point were not checked first?<br>
**2.** Why must Problem 5 establish continuity before matching derivatives?<br>
**3.** Which step in Problem 6(c) actually uses the extra limit condition?<br>
<!-- bilingual-en:end -->

> [!success]- 自检答案
> 1. 指定点可能根本不在曲线上；此时“该点切线”无意义。验点也检查分数幂的实数解释。
> 2. 可导必连续；即使左右导数形式碰巧相等，函数值若跳跃仍不可导。连续条件还决定参数的一部分。
> 3. 令 $h\to0$ 时把 $f(h)/h$ 替换为 $1$；其余项只用代数和导数定义。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** The specified point might not lie on the curve at all, in which case a tangent “at that point” is meaningless. Checking the point also verifies the real-valued interpretation of the fractional power.<br>
> **2.** Differentiability implies continuity. Even if the two derivative expressions happen to match, a jump in the function values still prevents differentiability. The continuity conditions also determine some of the parameters.<br>
> **3.** The extra condition is used when $h\to0$ and $f(h)/h$ is replaced by $1$; every other step uses only algebra and the definition of the derivative.<br>
> <!-- bilingual-en:end -->

**Session 22 小结：**Exam 1 同时检查计算与论证。高分答案不只给最终式，还会指出定义域、转向点、连续性条件，以及证明中使用“可导蕴含连续”的位置。
<!-- bilingual-en:start -->
**Session 22 summary:** Exam 1 tests both calculation and justification. A strong answer goes beyond the final expression by identifying domains, turning points, continuity conditions, and every place where a proof uses the fact that differentiability implies continuity.
<!-- bilingual-en:end -->

---

## 全章知识链与复习清单
<!-- bilingual-en:start -->
*Chapter Knowledge Chain and Review Checklist*
<!-- bilingual-en:end -->

### 一条主链
<!-- bilingual-en:start -->
*The Main Chain*
<!-- bilingual-en:end -->

$$
\text{割线平均变化率}
\xrightarrow{h\to0}
\text{导数}
\xrightarrow{\text{极限代数}}
\text{求导规则}
\xrightarrow{\text{链式法则}}
\text{隐函数与反函数}
\xrightarrow{\ln/\exp}
\text{变量幂与双曲函数}.
$$

### 必须能独立重建的证明
<!-- bilingual-en:start -->
*Proofs You Must Be Able to Reconstruct Independently*
<!-- bilingual-en:end -->

1. 可导蕴含连续：把函数增量写成“差商 × 输入增量”。
2. 正整数幂法则：二项式展开后除以 $h$。
3. $\sin x/x$ 极限：单位圆面积夹逼，且说明弧度条件。
4. 正余弦导数：角和公式 + 两个基本极限。
5. 积法则：加减中间项 + 可导蕴含连续。
6. 链式法则：局部倍率相乘，并处理 $\Delta x=0$ 的辅助函数。
7. 反函数导数：对 $f(f^{-1}(x))=x$ 求导。
8. $e^x,\ln x$ 导数：$M(e)=1$ 与反函数求导。
9. $\lim(1+1/n)^n=e$：取对数化成 $\ln x$ 在 $1$ 的导数。
<!-- bilingual-en:start -->

&nbsp;
**1.** Differentiability implies continuity: write the function increment as “difference quotient × input increment.”<br>
**2.** The positive-integer power rule: expand with the binomial theorem and divide by $h$.<br>
**3.** The limit of $\sin x/x$: squeeze areas in the unit circle and explain why radians are required.<br>
**4.** The derivatives of sine and cosine: combine the angle-addition identities with the two fundamental limits.<br>
**5.** The product rule: add and subtract an intermediate term, then use the fact that differentiability implies continuity.<br>
**6.** The chain rule: multiply local rates and use the auxiliary function to handle $\Delta x=0$.<br>
**7.** The inverse-function derivative: differentiate $f(f^{-1}(x))=x$.<br>
**8.** The derivatives of $e^x$ and $\ln x$: use $M(e)=1$ and the inverse-function rule.<br>
**9.** $\lim(1+1/n)^n=e$: take logarithms and reduce the limit to the derivative of $\ln x$ at $1$.<br>
<!-- bilingual-en:end -->

### 每次求导的五项检查
<!-- bilingual-en:start -->
*Five Checks for Every Differentiation*
<!-- bilingual-en:end -->

- **定义域：**原函数在哪里有意义？答案不能扩张原定义域。
- **结构：**最外层是和、积、商还是复合？内层导数是否齐全？
- **符号：**递增/递减、奇偶性、简单点斜率是否吻合？
- **单位：**导数单位是否为“输出单位/输入单位”？
- **边界：**除过的因子能否为零？端点、尖角、竖直切线是否需单独讨论？
<!-- bilingual-en:start -->
- **Domain:** Where is the function defined? The derivative formula must not enlarge that domain.
- **Structure:** Is the outermost operation a sum, product, quotient, or composition? Is every inner derivative present?
- **Signs:** Do monotonicity, parity, and slopes at simple points agree with the result?
- **Units:** Does the derivative have units of “output units per input unit”?
- **Boundaries:** Can any factor divided out be zero? Do endpoints, corners, cusps, or vertical tangents require separate treatment?
<!-- bilingual-en:end -->

> [!tip] 一遍看懂后的主动复习
> 合上正文，依次从定义重算 $1/x$、证明积法则、证明三角基本极限、推导反函数公式、求 $D(x^x)$。若不仅能写结论，还能说出每一步的假设和检查方法，本章就真正形成了可迁移的知识链。
> <!-- bilingual-en:start -->
> Close the note and, in order, recompute the derivative of $1/x$ from the definition, prove the product rule, prove the fundamental trigonometric limit, derive the inverse-function formula, and find $D(x^x)$. If you can state not only each conclusion but also the assumptions and checks used at every step, this chapter has become a transferable chain of knowledge.
> <!-- bilingual-en:end -->

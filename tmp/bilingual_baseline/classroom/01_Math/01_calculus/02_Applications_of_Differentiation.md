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

- 官方课程：[MIT OCW 18.01SC — Unit 2](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/pages/unit-2-applications-of-differentiation/)
- 教师：David Jerison；学期：Fall 2010
- 官方顺序：Part A（Session 23–28）→ Problem Set 3 → Part B（Session 29–33）→ Problem Set 4 → Part C（Session 34–40）→ Problem Set 5 → Exam 2（Session 41–42）。
- 本地 SesXXa/b/c 表示同一 Session 中依次播放的片段；正文按字母顺序整合。所有 PDF 仍在原位，链接均给到内容起始页。
- 记号：若 $x$ 是自变量，$\Delta x$ 表示有限改变量；$dx$ 表示在线性化中自由指定的输入微分；$C$ 表示任意常数。近似式后的“$x\approx a$”是适用条件，不是等式的一部分。

## 学习目标

学完本章，应当能够：

1. 构造线性化 $L_a(x)$ 与二次近似 $Q_a(x)$，并解释匹配了哪些导数、误差为什么随离基点的距离增长。
2. 由定义域、端点、间断、$f'$ 和 $f''$ 的符号，画出定性正确且标注完整的函数图像。
3. 把优化与相关变化率文字题翻译成变量、约束和目标，并在临界点之外检查端点、间断与单位。
4. 推导并执行 Newton 迭代，识别水平切线、错误根、振荡和初值太远等失败模式。
5. 准确陈述 MVT 的全部假设，用它证明单调性、常函数结论和不等式。
6. 理解微分、反导数、换元和分离变量之间的知识链，并能检查初值、平衡解与最大定义区间。

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

**问题：**曲线为什么能在很小范围内用直线代替？怎样把“切线很像曲线”写成可计算的公式？

**前置：**导数定义、切线方程，以及 $\sin,\cos,e^x,\ln x,(1+x)^r$ 的导数。

### 23a–23c：从切线方程到线性化

[[Linear Approximation|线性近似]]从切线开始。设 $f$ 在 $a$ 可导，过 $(a,f(a))$、斜率为 $f'(a)$ 的切线是

$$
L_a(x)=f(a)+f'(a)(x-a).
$$

当 $x$ 接近 $a$ 时，以 $L_a(x)$ 代替 $f(x)$：

$$
\boxed{f(x)\approx f(a)+f'(a)(x-a)}.
$$

这不是猜测。令 $h=x-a$，可导定义等价于

$$
\lim_{h\to0}\frac{f(a+h)-f(a)-f'(a)h}{h}=0.
$$

若把余项记作

$$
R_1(h)=f(a+h)-f(a)-f'(a)h,
$$

上式就是 $R_1(h)=o(h)$：误差相对于 $h$ 更快趋于 $0$。因此“局部线性”精确地表示为

$$
f(a+h)=f(a)+f'(a)h+o(h).
$$

> [!note] 几何直觉
> 放大一条光滑曲线，弯曲部分逐渐看成直线。放大倍率使横向误差按 $h$ 缩放，而余项比 $h$ 更小，所以在极限中消失。

**讲义例题：$\ln x$ 在 $a=1$ 的线性化。**

$$
f(1)=0,\qquad f'(1)=1,
$$

所以

$$
\ln x\approx x-1\qquad(x\approx1).
$$

例如 $\ln1.02\approx0.02$。离 $1$ 越远，不能仅凭该式保证准确；需要单独判断[[Approximation Error|近似误差]]。

### 23d–23e：必须掌握的基准近似

把 $a=0$ 代入一般式，逐项计算 $f(0)$ 与 $f'(0)$：

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

![[98_attachment/MIT18.01SC/unit02-linear-quadratic.png|760]]

### 配套练习：比较 $\sin x$ 的近似

线性化为 $L(x)=x$，故

$$
\sin0.01\approx0.01,\quad \sin0.1\approx0.1,\quad \sin1\approx1.
$$

计算器真值约为 $0.0099998333,0.0998334166,0.8414709848$。前两项靠近基点，误差很小；$x=1$ 已不够接近 $0$。三角函数必须用**弧度**，因为 $(\sin x)'_{x=0}=1$ 只在弧度制成立。

### 易错点与边界

- “$\approx$”不能在任意 $x$ 使用；必须注明基点。
- 线性化匹配函数值和一阶导数，不匹配二阶导数。
- 把 $\cos x\approx1$ 误解为余弦恒等于 $1$，会丢掉全部弯曲信息。
- 相乘时要丢掉二次及更高次项，才是“乘积的线性部分”。

> [!question]- 三道自检题与答案
> 1. $\sqrt{x}$ 在 $a=4$ 的线性化？
> $L(x)=2+\frac14(x-4)$。
>
> 2. 用它估计 $\sqrt{4.1}$。
> $2+\frac14(0.1)=2.025$。
>
> 3. 为什么不能用 $\ln(1+x)\approx x$ 估计 $\ln101$？
> 此时 $x=100$ 不接近 $0$；应另选靠近 $101$ 且函数值已知的基点。

### 本地材料

- [[Ses23a_Lecture_Notes.pdf#page=1|23a Introduction to Linear Approximation]]
- [[Ses23b_Lecture_Notes.pdf#page=1|23b Linear Approximation to ln x at x=1]]
- [[Ses23c_Lecture_Notes.pdf#page=1|23c Definition of the Derivative]]
- [[Ses23d_Lecture_Notes.pdf#page=1|23d Sine, Cosine and Exponential]]
- [[Ses23e_Lecture_Notes.pdf#page=1|23e ln(1+x) and (1+x)^r]]
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise023_Problems.pdf#page=1|Exercise 23 原题]] · [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise023_Solutions.pdf#page=1|Exercise 23 解答]]

**知识链小结：**可导 $\Rightarrow$ 局部线性；切线公式不仅画线，也把难函数值转成四则运算。

## Session 24：Examples of Linear Approximation

### 本节问题与前置知识

**问题：**怎样组合多个已知近似？误差应当用绝对量还是相对量衡量？

**前置：**Session 23 的五个基准近似、乘积法则和链式法则。

### 24a–24c：复杂函数先拆后近似

讲义计算

$$
f(x)=\frac{e^{-3x}}{\sqrt{1+x}}
=e^{-3x}(1+x)^{-1/2},\qquad x\approx0.
$$

分别线性化：

$$
e^{-3x}\approx1-3x,\qquad (1+x)^{-1/2}\approx1-\frac12x.
$$

相乘：

$$
(1-3x)\left(1-\frac12x\right)
=1-\frac72x+\frac32x^2.
$$

线性近似只保留到一次项，因此

$$
\boxed{f(x)\approx1-\frac72x}.
$$

为什么这与直接求导必然一致？若

$$
f(x)\approx f_0+f_1h,\qquad g(x)\approx g_0+g_1h,\qquad h=x-a,
$$

则乘积的常数、一次部分为

$$
f_0g_0+(f_1g_0+f_0g_1)h,
$$

而括号内恰是 $(fg)'(a)$。被丢掉的 $f_1g_1h^2$ 是二阶小量。

### 24d：GPS 时间膨胀例子

特殊相对论给出

$$
T_m=\frac{T}{\sqrt{1-v^2/c^2}}.
$$

令 $u=v^2/c^2$。当 $v\ll c$ 时 $u\approx0$，由 $(1-u)^{-1/2}\approx1+\frac12u$，

$$
T_m\approx T\left(1+\frac12\frac{v^2}{c^2}\right).
$$

近似带来的时间差是

$$
\Delta T=T_m-T\approx \frac{T}{2}\frac{v^2}{c^2}.
$$

这里 $v/c$ 无量纲，所以括号内可与 $1$ 相加；这也是应用题的单位自检。

### 24e：相对误差

绝对误差是 $|\widetilde y-y|$；相对误差是

$$
\frac{|\widetilde y-y|}{|y|}\qquad(y\ne0).
$$

相对误差回答“错了真实量的百分之几”，比跨尺度比较绝对误差更合理。对于上例，

$$
\frac{\Delta T}{T}\approx\frac12\frac{v^2}{c^2}.
$$

### 配套练习：乘积近似证明

在 $a$ 附近令 $h=x-a$：

$$
\begin{aligned}
L_fL_g
&=[f(a)+f'(a)h][g(a)+g'(a)h]\\
&=f(a)g(a)+[f'(a)g(a)+f(a)g'(a)]h+f'(a)g'(a)h^2.
\end{aligned}
$$

乘积 $fg$ 的线性化是前两项；两者只差二次项。因此“先线性化再相乘并截到一次”与“先相乘再线性化”等价。

### 易错点与边界

- 组合近似时，必须统一在同一个小参数上展开。
- 线性近似不能保留 $x^2$ 项；保留会伪装成二次近似，却没有完整匹配二阶导数。
- 相对误差在真实值为 $0$ 时无定义。
- 物理式中先检查无量纲比值和单位，再代数运算。

> [!question]- 三道自检题与答案
> 1. 线性化 $(1+2x)^3e^{-x}$。
> $(1+6x)(1-x)\approx1+5x$。
>
> 2. 若真值 $1000$、估计 $1002$，绝对与相对误差？
> $2$ 与 $0.002=0.2\%$。
>
> 3. 为什么 $x^2$ 比 $x$ 小必须附带条件？
> 只有 $|x|<1$ 时 $|x^2|<|x|$；近似还要求 $x$ 足够接近展开点。

### 本地材料

- [[Ses24a_Lecture_Notes.pdf#page=1|24a Curves are Hard, Lines are Easy]]
- [[Ses24b_Lecture_Notes.pdf#page=1|24b Complicated Exponential]]
- [[Ses24c_Lecture_Notes.pdf#page=1|24c Direct Formula Check]]
- [[Ses24d_Lecture_Notes.pdf#page=1|24d GPS Time Dilation]]
- [[Ses24e_Lecture_Notes.pdf#page=1|24e Relative Error]]
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise024_Problems.pdf#page=1|Exercise 24 原题]] · [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise024_Solutions.pdf#page=1|Exercise 24 解答]]

**知识链小结：**局部线性可像代数式一样组合，但必须按阶截断；应用中用相对误差说明结果是否“足够准”。

## Session 25：Introduction to Quadratic Approximation

### 本节问题与前置知识

**问题：**切线没记录[[Concavity|凹凸性]]，怎样加入最少的新信息来改进精度？为什么二次项系数是 $f''(a)/2$？

**前置：**线性化、二阶导数及凹凸性。

### 25a：二次近似公式

若 $f$ 在 $a$ 有二阶导数，[[Quadratic Approximation|二次近似]]定义为

$$
\boxed{
Q_a(x)=f(a)+f'(a)(x-a)+\frac{f''(a)}{2}(x-a)^2
}.
$$

$Q_a$ 满足三个匹配条件：

$$
Q_a(a)=f(a),\qquad Q_a'(a)=f'(a),\qquad Q_a''(a)=f''(a).
$$

### 25b：系数 $1/2$ 的逐步推导

设在 $a=0$ 附近寻找 $Q(x)=A+Bx+Cx^2$。

$$
Q(0)=A,\quad Q'(x)=B+2Cx,\quad Q''(x)=2C.
$$

要求 $Q(0)=f(0),Q'(0)=f'(0),Q''(0)=f''(0)$，依次得到

$$
A=f(0),\qquad B=f'(0),\qquad C=\frac{f''(0)}2.
$$

因此 $1/2$ 不是约定，而是二次项求两次导后产生因子 $2$ 的必然补偿。一般基点只需把 $x$ 换成 $x-a$。

### 25c：基本二次近似库

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

$$
\ln1.1=\ln(1+0.1)\approx0.1-\frac{0.1^2}{2}=0.095,
$$

比线性估计 $0.1$ 更接近真值约 $0.09531$。

### 配套练习：$e^x$ 的精度

二次近似 $Q(x)=1+x+x^2/2$ 给出

$$
e^{0.01}\approx1.01005,\quad e^{0.1}\approx1.105,\quad e^1\approx2.5.
$$

真值分别约 $1.010050167,1.105170918,2.718281828$。仍然是越靠近 $0$ 越好；提高次数并不允许无视展开点。

### 易错点与边界

- “二次近似”允许二次项系数为 $0$，例如 $\sin x$ 在 $0$ 的二次近似仍是 $x$。
- 二阶可导只保证公式可构造；精确误差界需要更高阶信息，后续由 Taylor 余项给出。
- $f''(a)>0$ 表示最佳拟合抛物线开口向上，不代表函数在所有地方凸。

> [!question]- 三道自检题与答案
> 1. $\sqrt{1+x}$ 在 $0$ 的二次近似？
> $1+\frac12x-\frac18x^2$。
>
> 2. 为什么 $\cos x$ 的线性化看不出弯曲？
> $\cos'(0)=0$，线性化是水平线；$\cos''(0)=-1$ 才记录向下弯曲。
>
> 3. $Q_a$ 与 $f$ 在 $a$ 处共匹配几项信息？
> 函数值、一阶导、二阶导，共三项。

### 本地材料

- [[Ses25a_Lecture_Notes.pdf#page=1|25a Formula for Quadratic Approximation]]
- [[Ses25b_Lecture_Notes.pdf#page=1|25b Explaining the Formula]]
- [[Ses25c_Lecture_Notes.pdf#page=1|25c Basic Quadratic Approximations]]
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise025_Problems.pdf#page=1|Exercise 25 原题]] · [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise025_Solutions.pdf#page=1|Exercise 25 解答]]

**知识链小结：**线性化匹配位置与斜率；二次近似再匹配曲率，是用最小复杂度换取更高局部精度。

## Session 26：Using Quadratic Approximations

### 本节问题与前置知识

**问题：**复杂函数怎样不做两次繁琐求导就得到二次近似？“丢掉高阶项”怎样系统执行？

**前置：**Session 25 的基本二次近似库、按次数展开乘积。

### 26a–26b：近似与收敛速度

令

$$
a_k=\left(1+\frac1k\right)^k.
$$

线性近似 $\ln(1+x)\approx x$ 给出

$$
\ln a_k=k\ln\left(1+\frac1k\right)\approx1,
$$

解释 $a_k\to e$。若还要看误差速度，必须保留二次项：

$$
\ln a_k
\approx k\left(\frac1k-\frac1{2k^2}\right)
=1-\frac1{2k}.
$$

所以 $\ln a_k$ 与 $1$ 的主误差约为 $-1/(2k)$；线性近似能给极限，二次近似进一步给收敛速度。

### 26c：复杂函数的二次展开

再次处理

$$
\frac{e^{-3x}}{\sqrt{1+x}}.
$$

$$
e^{-3x}\approx1-3x+\frac92x^2,
$$

而 $r=-1/2$ 时

$$
(1+x)^{-1/2}\approx1-\frac12x+\frac38x^2.
$$

相乘只收集总次数不超过 $2$ 的项：

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

### 26d：一般 $n$ 次匹配的来源

若

$$
P_n(x)=a_0+a_1x+\cdots+a_nx^n,
$$

则在 $0$ 取第 $k$ 阶导，低于 $k$ 次项消失，高于 $k$ 次项仍含 $x$ 而在 $0$ 消失，仅剩

$$
P_n^{(k)}(0)=k!a_k.
$$

令其等于 $f^{(k)}(0)$，

$$
a_k=\frac{f^{(k)}(0)}{k!}.
$$

于是

$$
P_n(x)=\sum_{k=0}^n\frac{f^{(k)}(0)}{k!}x^k.
$$

这已是 Taylor 多项式的骨架；本章先用到二次，Unit 5 再研究无穷级数与余项。

### 易错点与边界

- 相乘时不是分别“保留二次”后把所有项都留下，而是最终总次数仅保留到 $2$。
- $e^{-3x}$ 的二次项是 $(-3x)^2/2=9x^2/2$，符号为正。
- 近似等式不能在中途当精确恒等式消去可能与误差同阶的量。

> [!question]- 三道自检题与答案
> 1. 求 $e^x(1+x)^{-1}$ 的二次近似。
> $(1+x+x^2/2)(1-x+x^2)\approx1+x^2/2$。
>
> 2. 为什么三次系数分母是 $3!$？
> $d^3(a_3x^3)/dx^3=3!a_3$。
>
> 3. 二次近似能否自动给出严格误差上界？
> 不能；还需控制区间上的三阶导数等余项信息。

### 本地材料

- [[Ses26a_Lecture_Notes.pdf#page=1|26a Quadratic Approximation Library]]
- [[Ses26b_Lecture_Notes.pdf#page=1|26b Approximation of ln e]]
- [[Ses26c_Lecture_Notes.pdf#page=1|26c Complicated Quadratic Approximation]]
- [[Ses26d_Lecture_Notes.pdf#page=1|26d Deriving ln(1+x) and (1+x)^r]]
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise026_Problems.pdf#page=1|Exercise 26 原题]] · [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise026_Solutions.pdf#page=1|Exercise 26 解答]]

**知识链小结：**组合近似的核心是“统一小参数、按总次数截断”；阶数越高，能读取的局部变化层次越多。

## Session 27：Sketching Graphs I — Polynomials and Rational Functions

### 本节问题与前置知识

**问题：**不用逐点计算，怎样由 $f'$、$f''$ 画出全局形状？为何间断点和无穷远行为与临界点同样重要？

**前置：**极限、定义域、一阶/二阶导数与奇偶性。

### 27a：两条基本原则

在一个区间上：

$$
f'(x)>0\Rightarrow f\text{ 递增},\qquad
f'(x)<0\Rightarrow f\text{ 递减}.
$$

再对 $f'$ 应用同样思想：

$$
f''(x)>0\Rightarrow f'\text{ 递增，图像凹向上};
$$

$$
f''(x)<0\Rightarrow f'\text{ 递减，图像凹向下}.
$$

严格证明将在 Session 34 用 MVT 完成。

### 27b：多项式例 $f(x)=3x-x^3$

一阶导数

$$
f'(x)=3-3x^2=3(1-x)(1+x).
$$

符号表：

| 区间 | $(-\infty,-1)$ | $(-1,1)$ | $(1,\infty)$ |
|---|---:|---:|---:|
| $f'$ | $-$ | $+$ | $-$ |
| $f$ | 递减 | 递增 | 递减 |

[[Critical Points and Extrema|临界点]]（critical point）是定义域内 $f'(x)=0$ 或 $f'$ 不存在而 $f$ 存在的候选位置。此处 $x=\pm1$，函数点为 $(-1,-2),(1,2)$。故前者局部最小，后者局部最大。

$$
f''(x)=-6x.
$$

$x<0$ 时凹向上，$x>0$ 时凹向下；$x=0$ 两侧凹凸改变，所以 $(0,0)$ 是拐点。最高次项 $-x^3$ 控制两端：

$$
x\to\infty:f(x)\to-\infty,\qquad
x\to-\infty:f(x)\to\infty.
$$

![[98_attachment/MIT18.01SC/unit02-curve-sign-chart.png|760]]

### 27c：有理函数例 $f(x)=\dfrac{x+1}{x+2}$

先改写

$$
f(x)=1-\frac1{x+2}.
$$

因此定义域排除 $x=-2$，竖直渐近线为 $x=-2$，水平渐近线为 $y=1$。并且

$$
f'(x)=\frac1{(x+2)^2}>0,\qquad x\ne-2.
$$

它分别在 $(-\infty,-2)$ 与 $(-2,\infty)$ 递增，但**不能**跨越间断点说在整个定义域递增。二阶导

$$
f''(x)=-\frac2{(x+2)^3},
$$

左支凹向上、右支凹向下。$x=-2$ 不是拐点，因为原函数在那里不连续。

### 易错点与边界

- 只解 $f'=0$ 会漏掉端点、间断点及 $f'$ 不存在但 $f$ 存在的尖点。
- $f''(c)=0$ 只是拐点候选；必须检查两侧凹凸是否改变且函数在 $c$ 连续。
- 两个分离区间上都递增，不等于跨间断点递增。
- 有理函数的无穷远极限与竖直渐近线必须先由原式/改写式判断。

> [!question]- 三道自检题与答案
> 1. $x^4$ 的 $x=0$ 是拐点吗？
> 不是；$f''=12x^2$ 两侧均非负，凹凸不改变。
>
> 2. $|x|$ 的 $x=0$ 是否应在作图时检查？
> 是；$f$ 存在而 $f'$ 不存在，是尖点和极小值。
>
> 3. $\frac1x$ 能否说在整个定义域递减？
> 按全局定义不能；它分别在两支递减，但例如 $-1<1$ 而 $f(-1)<f(1)$。

### 本地材料

- [[Ses27a_Lecture_Notes.pdf#page=1|27a Introduction to Curve Sketching]]
- [[Ses27b_Lecture_Notes.pdf#page=1|27b Polynomial Example]]
- [[Ses27c_Lecture_Notes.pdf#page=1|27c Rational Function Example]]

**知识链小结：**作图不是“求几个导数零点”，而是把定义域、间断、端点、无穷远、单调和凹凸拼成一张相互校验的图。

## Session 28：Sketching Graphs II — General Strategies

### 本节问题与前置知识

**问题：**面对任意函数，怎样用一套不会漏项的顺序完成作图？

**前置：**Session 27 的单调、凹凸、渐近线与临界点。

### 28a：[[Curve Sketching|曲线描绘]]五步法

1. **先做预备代数：**定义域、对称性、截距、易算点。
2. **查边界：**端点、间断点的单侧极限、$x\to\pm\infty$，标出渐近线。
3. **查一阶信息：**求 $f'$，列全部临界候选，做符号表，算必要的临界值。
4. **查二阶信息：**求 $f''$，找凹凸区间及真正拐点。
5. **合成并复核：**图像必须同时满足函数值、极限、单调、凹凸；若冲突，回查代数。

### 28b：完整例题 $f(x)=\dfrac{x}{\ln x}$

**定义域：**$x>0$ 且 $x\ne1$。

**边界：**

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

**一阶导：**

$$
f'(x)=\frac{\ln x-1}{(\ln x)^2}.
$$

分母正，符号由 $\ln x-1$ 决定。因此在 $(0,1)$、$(1,e)$ 递减，在 $(e,\infty)$ 递增；唯一普通临界点 $x=e$，且 $f(e)=e$，是右支最小值。

**二阶导：**

$$
f''(x)=\frac{2-\ln x}{x(\ln x)^3}.
$$

因为 $x>0$，符号表给出：

$$
(0,1):f''<0,\qquad
(1,e^2):f''>0,\qquad
(e^2,\infty):f''<0.
$$

在 $x=e^2$ 连续且凹凸改变，拐点为 $(e^2,e^2/2)$；$x=1$ 虽两侧符号改变，却是间断点而非拐点。

### 配套练习：由特征反造三次函数

若要求拐点在 $x=2$，且 $x>2$ 凹向下，可先选

$$
f''(x)=2-x.
$$

积分得

$$
f'(x)=2x-\frac{x^2}{2}+C.
$$

要求临界点在 $1,3$，代 $x=1$ 得 $C=-3/2$，并自动满足 $f'(3)=0$。再次积分：

$$
f(x)=x^2-\frac{x^3}{6}-\frac32x+D.
$$

$D$ 只做竖直平移，不影响单调与凹凸，所以答案不唯一。原题 PDF 的第一条凹凸描述与解答首页有文字方向差异；上述推导遵循原题“$x<2$ 凹向上、$x>2$ 凹向下”。

### 易错点与边界

- 把 $x\to0^+$ 的极限点 $(0,0)$ 画成函数实际包含的点；本例 $x=0$ 不在定义域。
- 写 $f(\infty)$ 只是非正式缩写，正式论证必须写极限。
- 复杂二阶导若只为定性图，可在一阶信息已足够时省略；但若题目要求拐点则不能省。

> [!question]- 三道自检题与答案
> 1. 作图的第一步为何不是求导？
> 定义域和间断会决定导数符号表应分成哪些区间，也可能直接给出最显著特征。
>
> 2. 本例为何 $x=e$ 是右支绝对最小？
> 右支从 $+\infty$ 递减到 $e$，再递增到 $+\infty$。
>
> 3. 加常数 $D$ 会改变什么？
> 只竖直平移；不改变 $f'$、$f''$、临界点横坐标和凹凸。

### 本地材料

- [[Ses28a_Lecture_Notes.pdf#page=1|28a General Strategy]]
- [[Ses28b_Lecture_Notes.pdf#page=1|28b Detailed Example x/ln x]]
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise028_Problems.pdf#page=1|Exercise 28 原题]] · [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise028_Solutions.pdf#page=1|Exercise 28 解答]]

**知识链小结：**稳定的作图流程先划分定义域，再用 $f'$ 决定方向、$f''$ 修饰形状，最后用极限封住各段的两端。

## Problem Set 3

> [!info] 官方指定范围与材料
> 2A：1, 3, 6, 11, 12a, 12d, 12e；2B：2a, 2e, 2h, 4, 6a, 6b, 7a, 7b。
> [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/PSet02_Problems.pdf#page=1|Applications of Differentiation 原题]] · [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/PSet02_Solutions.pdf#page=1|官方解答]]

### 2A：Approximation

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

> [!example]- 2A-12a：$\dfrac{e^x}{1-x}$ 的二次近似
> $$
> e^x\approx1+x+\frac{x^2}{2},\qquad
> \frac1{1-x}\approx1+x+x^2.
> $$
> 乘积截至二次：
> $$
> \boxed{\frac{e^x}{1-x}\approx1+2x+\frac52x^2}.
> $$

> [!example]- 2A-12d：$\ln(\cos x)$ 的二次近似
> 先用 $\cos x\approx1-x^2/2$。令 $u=-x^2/2$，则
> $$
> \ln(\cos x)\approx\ln(1+u)\approx u=-\frac{x^2}{2}.
> $$
> 因 $u^2$ 已是四次，不进入二次近似。答案：
> $$
> \boxed{\ln(\cos x)\approx-\frac{x^2}{2}}.
> $$

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

### 2B：Curve Sketching

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

> [!example]- 2B-4：由文字描述作图并判断绝对极值
> $f$ 在 $[0,10]$ 连续，零点为 $4,7,9$；在 $(0,5),(8,10)$ 递增，在 $(5,8)$ 递减。
> 因而 $x=5$ 是局部最大，$x=8$ 是局部最小。符号还可确定：$f(4)=0$ 且递增，所以 $f(5)>0$；从 $5$ 递减并在 $7$ 过零，故 $f(8)<0$；之后递增并在 $9$ 过零。
> 绝对最大只能在 $\boxed{x=5\text{ 或 }x=10}$；绝对最小只能在 $\boxed{x=0\text{ 或 }x=8}$。题目未给端点高度，不能再唯一决定。

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

> [!example]- 2B-7a、2B-7b：递增函数在可导点的导数
> 若 $f$ 递增且在 $a$ 可导。对 $h>0$，$f(a+h)-f(a)\ge0$，故差商 $\ge0$；对 $h<0$，分子 $\le0$、分母 $<0$，差商仍 $\ge0$。两侧极限相等且存在，故
> $$
> \boxed{f'(a)\ge0}.
> $$
> 不能加强成 $f'(a)>0$，因为一串正数的极限可以是 $0$。反例 $f(x)=x^3$ 严格递增，但 $f'(0)=0$。

### Problem Set 3 错误检查

- 拐点必须是函数图像上的点，不能只写 $x$ 值而忽略连续性。
- 2A-12a 的本地文本抽取容易把分母排成 $1-2x$；由官方答案 $1+2x+\frac52x^2$ 可确认原式是 $1-x$。
- 构造函数时加任意常数不改变导数信息，因此答案通常不唯一。

**本组小结：**PS3 把“按阶展开”和“按导数符号读图”连在一起：近似研究一个点附近，作图研究由这些局部信息拼成的全局结构。

---

## Part B：Optimization, Related Rates and Newton’s Method

## Session 29：Optimization Problems

### 本节问题与前置知识

**问题：**怎样不画出整张精确图，就找到函数的最大/最小值？为什么只检查 $f'=0$ 可能得到“最坏答案”？

**前置：**临界点、端点、间断点和一阶导数符号。

### 29a：极值候选清单

[[Optimization with Derivatives|导数优化]]先列候选。若 $f$ 在闭区间 $[a,b]$ 连续，极值定理保证绝对最大、最小存在。它们只能出现在：

1. 内点且 $f'(x)=0$；
2. 内点且 $f'$ 不存在；
3. 端点 $a,b$。

若定义域不是闭区间，还必须检查开端点的单侧极限、无穷远和间断点附近。求出候选只是第一步，最后要**比较目标函数值**。

### 29b：一根铁丝围两个正方形

总长为 $1$，一段长 $x$，另一段长 $1-x$，所以

$$
0\le x\le1.
$$

两正方形边长分别为 $x/4,(1-x)/4$，总面积

$$
A(x)=\frac{x^2}{16}+\frac{(1-x)^2}{16}.
$$

求导：

$$
A'(x)=\frac{2x-1}{8}.
$$

唯一内点候选 $x=1/2$，此时

$$
A(1/2)=\frac1{32}.
$$

端点：

$$
A(0)=A(1)=\frac1{16}.
$$

所以等分铁丝给出**最小**总面积 $1/32$；把全部铁丝给一个正方形给出最大面积 $1/16$。如果只算临界点，就会把题目所求最大值误判成最小值。

### 极大/极小、极值点与极值

- “最大值是多少？”答函数值，例如 $1/16$。
- “在哪里达到？”答自变量，例如 $x=0$ 或 $1$。
- 图像上的完整信息是点 $(x,A(x))$。
- 局部极值只比较附近；绝对极值比较整个定义域。

### 易错点与边界

- 没有把几何限制翻译为定义域。
- 找到临界点就停止，不比较端点。
- 把“最大面积”和“最大面积发生的位置”混为一谈。
- 对开放定义域写出并不存在的“端点取值”，应使用极限。

> [!question]- 三道自检题与答案
> 1. 连续函数在闭区间为何一定要查端点？
> 绝对极值可以在端点达到，而端点不要求导数为零。
>
> 2. 若 $f'$ 在内点不存在，该点是否自动为极值？
> 不是，只是候选；例如 $x^{1/3}$ 在 $0$ 导数无穷但仍递增。
>
> 3. 本例 $A''=1/4>0$ 能说明什么？
> 内点 $x=1/2$ 是局部最小；最大值仍需由端点比较确定。

### 本地材料

- [[Ses29a_Lecture_Notes.pdf#page=1|29a Introduction to Maxima and Minima]]
- [[Ses29b_Lecture_Notes.pdf#page=1|29b Maximum Area of Two Squares]]

**知识链小结：**优化的可靠流程是“建模并定域 → 列全部候选 → 比较目标值 → 用单位和图形复核”。

## Session 30：Optimization Problems II

### 本节问题与前置知识

**问题：**目标函数含多个变量并受约束时，怎样降为一个自由变量？隐式求导何时更短？

**前置：**Session 29 的候选比较、隐函数求导。

### 30a：无盖方盒的最小表面积

设正方形底边为 $x>0$、高为 $y>0$，固定体积 $V$：

$$
V=x^2y.
$$

无盖盒表面积

$$
A=x^2+4xy.
$$

由约束 $y=V/x^2$ 消去 $y$：

$$
A(x)=x^2+\frac{4V}{x}.
$$

$$
A'(x)=2x-\frac{4V}{x^2}.
$$

令 $A'=0$：

$$
x^3=2V,\qquad x=(2V)^{1/3}.
$$

相应

$$
y=\frac{V}{x^2}=2^{-2/3}V^{1/3},
\qquad\boxed{\frac{x}{y}=2}.
$$

边界检查：

$$
\lim_{x\to0^+}A(x)=\infty,\qquad
\lim_{x\to\infty}A(x)=\infty.
$$

且只有一个临界点，所以它给出全局最小。比例 $x/y=2$ 是无量纲结论，比含 $V$ 的尺寸式更能说明“最佳形状”。

### 30b：用隐式求导直接得到比例

保持 $V$ 不变，对约束求导：

$$
0=\frac{dV}{dx}=2xy+x^2y'
\quad\Rightarrow\quad
y'=-\frac{2y}{x}.
$$

对面积求导：

$$
\frac{dA}{dx}=2x+4y+4xy'
=2x-4y.
$$

临界条件直接给

$$
\boxed{x=2y}.
$$

此法更快地产生比例，但没有自动完成“这是最小而非最大”的边界论证，仍要补查。

### 配套练习：体积 $1000\ \mathrm{cm^3}$ 的封闭圆罐

圆罐表面积与体积为

$$
S=2\pi r^2+2\pi rh,\qquad
1000=\pi r^2h.
$$

代入 $h=1000/(\pi r^2)$：

$$
S(r)=2\pi r^2+\frac{2000}{r}.
$$

$$
S'(r)=4\pi r-\frac{2000}{r^2}=0
\Rightarrow r^3=\frac{500}{\pi}.
$$

故

$$
\boxed{r=\left(\frac{500}{\pi}\right)^{1/3}\approx5.42\ \mathrm{cm}},
\qquad
\boxed{h=2r\approx10.84\ \mathrm{cm}}.
$$

$r\to0^+$ 或 $r\to\infty$ 时 $S\to\infty$，故这是全局最小。

### 易错点与边界

- 约束方程和目标函数角色不同：一个消元，一个优化。
- 把固定量 $V$ 在求导时当变量。
- 隐式法得到临界比例后漏掉边界检查。
- 尺寸带单位，比例无单位；答案应按题目要求给全。

> [!question]- 三道自检题与答案
> 1. 为什么无盖方盒只有两个自由尺寸却最终是一元问题？
> 固定体积约束把 $x,y$ 关联，只剩一个自由度。
>
> 2. 若圆罐无顶，最优比例还会是 $h=2r$ 吗？
> 不会；目标表面积改变，必须重新求导。
>
> 3. $S''>0$ 是否可以替代全部边界检查？
> 若能证明定义域上处处 $S''>0$，临界点唯一且为全局最小；但仍应说明定义域和极端形状。

### 本地材料

- [[Ses30a_Lecture_Notes.pdf#page=1|30a Open Box Optimization]]
- [[Ses30b_Lecture_Notes.pdf#page=1|30b Implicit Differentiation and Min/Max]]
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise030_Problems.pdf#page=1|Exercise 30 原题]] · [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise030_Solutions.pdf#page=1|Exercise 30 解答]]

**知识链小结：**约束把多个变量压缩成一个自由度；显式消元利于边界分析，隐式求导利于快速得到最优比例。

## Session 31：Related Rates

### 本节问题与前置知识

**问题：**[[Related Rates|相关变化率]]中，多个随时间变化的量由一个几何方程联系时，怎样从已知速率求未知速率？

**前置：**链式法则、隐函数求导、相似三角形、常见面积与体积公式。

### 31a–31b：路边雷达

警车距道路垂直距离 $30$ ft；车与警车直线距离 $D=50$ ft 时，雷达测得

$$
\frac{dD}{dt}=-80\ \mathrm{ft/s}.
$$

设车沿道路到垂足的有向距离为 $x$。几何关系

$$
x^2+30^2=D^2.
$$

此刻 $x=40$。**先求导、后代数值：**

$$
2x\frac{dx}{dt}=2D\frac{dD}{dt}.
$$

因此

$$
\frac{dx}{dt}
=\frac{D}{x}\frac{dD}{dt}
=\frac{50}{40}(-80)
=-100\ \mathrm{ft/s}.
$$

负号表示朝垂足方向运动；速率大小 $100\ \mathrm{ft/s}\approx68.2\ \mathrm{mph}$，超过 $65\ \mathrm{mph}$。

> [!warning] 为何不能先代 $D=50$？
> $D$ 是随时间变化的函数，只在所问瞬间取值 $50$。若求导前替成常数，错误得到 $D'=0$，抹掉了已知速率。

### 31c：锥形水箱

水箱高 $10$ ft、顶半径 $4$ ft；水深 $h$、水面半径 $r$。相似三角形：

$$
\frac{r}{h}=\frac4{10}
\Rightarrow r=\frac25h.
$$

水体积

$$
V=\frac13\pi r^2h
=\frac{4\pi}{75}h^3.
$$

求时间导数：

$$
\frac{dV}{dt}
=\frac{4\pi}{25}h^2\frac{dh}{dt}.
$$

若 $dV/dt=2\ \mathrm{ft^3/min}$ 且 $h=5$ ft，

$$
2=4\pi\frac{dh}{dt}
\Rightarrow
\boxed{\frac{dh}{dt}=\frac1{2\pi}\ \mathrm{ft/min}}.
$$

![[98_attachment/MIT18.01SC/unit02-related-rates-cone.png|720]]

### 一般流程

1. 画图，标出**会变**与**不变**的量。
2. 用几何/物理关系写一个在所有时刻成立的方程。
3. 对时间 $t$ 求导，给每个变量补链式因子。
4. 再代入所问瞬间的数值，解未知速率。
5. 检查符号、单位和量级。

### 配套练习：半球顶粮仓的隐式优化

设柱高 $h$、半径 $r$，体积

$$
V=\pi r^2h+\frac23\pi r^3.
$$

固定 $V$ 对 $r$ 求导：

$$
0=2\pi rh+\pi r^2h'+2\pi r^2
\Rightarrow h'=-2\frac{h+r}{r}.
$$

有地板时 $S=2\pi rh+3\pi r^2$：

$$
S'=2\pi h+2\pi rh'+6\pi r=2\pi(r-h).
$$

临界条件 $h=r$，结合体积得

$$
\boxed{r=h=\left(\frac{3V}{5\pi}\right)^{1/3}},
$$

并由极端形状比较确认最小。无地板时 $S=2\pi rh+2\pi r^2$，同法得 $S'=-2\pi h<0$，最小出现在边界 $h=0$，即只有半球。该练习再次说明：导数零点之外必须检查边界。

### 易错点与边界

- 把“正在减小”的已知速率写成正数。
- 对 $x(t)^2$ 求导漏写 $2xx'$。
- 相似三角形中的 $r,h$ 必须来自水体小锥，而不是混用水箱固定尺寸。
- 速率的单位比原变量多一个“每时间”。

> [!question]- 三道自检题与答案
> 1. $A=\pi r^2$，若 $r'=3$，则 $A'$？
> $A'=2\pi rr'=6\pi r$；没有当前 $r$ 就不能给数值。
>
> 2. 相关变化率为何本质上是链式法则？
> 几何量都是 $t$ 的复合函数，例如 $dV(h(t))/dt=(dV/dh)h'$。
>
> 3. 何时可以在求导前代常数？
> 只有题意保证该量在整个过程中恒定，例如警车到道路的垂距 $30$。

### 本地材料

- [[Ses31a_Lecture_Notes.pdf#page=1|31a Radar Setup]]
- [[Ses31b_Lecture_Notes.pdf#page=1|31b Radar Calculation]]
- [[Ses31c_Lecture_Notes.pdf#page=1|31c Conical Tank]]
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise031_Problems.pdf#page=1|Exercise 31 原题]] · [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise031_Solutions.pdf#page=1|Exercise 31 解答]]

**知识链小结：**相关变化率先保留变量之间的动态关系，再在最后冻结到某个瞬间；“先求导后代值”是核心纪律。

## Session 32：Ring on a String

### 本节问题与前置知识

**问题：**一个圆环在固定长度的绳上自由滑动，最低点为何满足两侧绳与竖直方向夹角相等？

**前置：**距离公式、隐式求导、约束极小值。

### 建模：固定长度产生椭圆约束

绳端固定在 $(0,0)$ 与 $(a,b)$，圆环位于 $(x,y)$。总长 $L$ 固定：

$$
\sqrt{x^2+y^2}+\sqrt{(x-a)^2+(y-b)^2}=L.
$$

所有可行点构成以两端为焦点的椭圆。重力势能与高度 $y$ 成正比，因此稳定位置是约束曲线的最低点；若最低点位于光滑内点，则

$$
y'(x)=0.
$$

### 隐式求导与等角条件

对约束求 $x$ 导数：

$$
\frac{x+yy'}{\sqrt{x^2+y^2}}
+
\frac{x-a+(y-b)y'}{\sqrt{(x-a)^2+(y-b)^2}}
=0.
$$

在最低点 $y'=0$：

$$
\frac{x}{\sqrt{x^2+y^2}}
=
\frac{a-x}{\sqrt{(x-a)^2+(y-b)^2}}.
$$

两边分别是左右绳段与竖直线夹角 $\alpha,\beta$ 的正弦，所以

$$
\sin\alpha=\sin\beta.
$$

实际几何中两角均为锐角，故

$$
\boxed{\alpha=\beta}.
$$

这同时是力学平衡条件：两侧张力的水平分量抵消；也是椭圆反射性质“入射角等于反射角”的微积分来源。

### 若继续求坐标

由水平投影相加可得

$$
\sin\alpha=\frac{a}{L},
\qquad
L\cos\alpha=\sqrt{L^2-a^2}.
$$

竖直投影关系给

$$
y=\frac12\left(b-\sqrt{L^2-a^2}\right).
$$

再由相似三角形可求

$$
x=\frac a2\left(1-\frac{b}{\sqrt{L^2-a^2}}\right).
$$

存在真实松弛位置还要求长度足以连接两端，且所得点属于相应椭圆下支。

### 易错点与边界

- $y'=0$ 只适用于光滑内点；若绳长退化或最低点落在端部，要另查边界。
- 平方距离后求导常会引入不属于原椭圆的点；保留根式更忠实。
- $\sin\alpha=\sin\beta$ 一般还可能有补角；本题由角的几何范围排除。

> [!question]- 三道自检题与答案
> 1. $b=0$ 时公式给出什么？
> $x=a/2$，最低点在两端的水平中点下方。
>
> 2. 为什么最小化的是 $y$ 而不是绳长？
> 绳长已是约束常数；重力势能随高度 $y$ 增加。
>
> 3. 椭圆知识是否是求等角条件的前提？
> 不是；距离约束与隐式求导已经足够。

### 本地材料

- [[Ses32a_Lecture_Notes.pdf#page=1|32a Ring on a String]]

**知识链小结：**约束极值把物理平衡、椭圆几何与隐式求导汇到同一个条件 $y'=0$。

## Session 33：Newton’s Method

### 本节问题与前置知识

**问题：**没有代数公式可解 $f(x)=0$ 时，怎样用切线快速逼近根？什么时候会失败？

**前置：**线性化、切线方程和迭代序列。

### 33a：迭代公式的推导

[[Newton's Method|Newton 法]]从已有近似 $x_n$ 出发，在它附近用切线

$$
L_n(x)=f(x_n)+f'(x_n)(x-x_n)
$$

代替 $f$。让切线而非原曲线取零：

$$
0=f(x_n)+f'(x_n)(x_{n+1}-x_n).
$$

若 $f'(x_n)\ne0$，

$$
\boxed{x_{n+1}=x_n-\frac{f(x_n)}{f'(x_n)}}.
$$

![[98_attachment/MIT18.01SC/unit02-newton-method.png|740]]

**例：求 $\sqrt5$。**令 $f(x)=x^2-5$：

$$
x_{n+1}=x_n-\frac{x_n^2-5}{2x_n}
=\frac12\left(x_n+\frac5{x_n}\right).
$$

取 $x_0=2$：

$$
x_1=\frac94=2.25,\qquad
x_2=\frac{161}{72}\approx2.236111,
$$

已经非常接近 $\sqrt5\approx2.236068$。

### 33b：为何通常非常快

设真根为 $r$，误差 $e_n=x_n-r$。若 $f'(r)\ne0$ 且二阶导在附近受控，Taylor 展开给

$$
e_{n+1}\approx
\frac{f''(r)}{2f'(r)}e_n^2.
$$

误差大致平方，所以当误差小于 $1$ 后，有效数字常近似翻倍。这叫二次收敛（quadratic convergence），但它是局部结论。

### 33c：四类失败

1. $f'(x_n)=0$：切线水平，没有有限的横轴交点。
2. 初值落在另一个根的吸引域，收敛到“错误的根”。
3. 曲率过大或初值太远，切线零点反而更远。
4. 迭代进入周期，例如两点来回跳动。

因此每步应检查 $|f(x_n)|$ 是否变小、$f'(x_n)$ 是否接近 $0$，不能只机械按键。

### 配套练习：$f(x)=x^3$ 永远不会有限步到根

若 $x_n\ne0$，

$$
x_{n+1}
=x_n-\frac{x_n^3}{3x_n^2}
=\frac23x_n.
$$

归纳得

$$
x_n=\left(\frac23\right)^nx_0.
$$

若 $x_0\ne0$，任何有限 $n$ 都有 $x_n\ne0$，但 $x_n\to0$。这里根是重根且 $f'(0)=0$，只呈线性收敛，而不是通常的二次收敛。

### 易错点与边界

- 分母必须是 $f'(x_n)$，不能写成 $f'(x_{n+1})$。
- “连续若干位不变”是停止准则之一，但更可靠的是同时检查残差 $|f(x_n)|$。
- 重根处 $f'(r)=0$，普通快速收敛理论不适用。

> [!question]- 三道自检题与答案
> 1. 用 Newton 法求 $1/a$ 可选什么方程？
> 例如 $f(x)=1/x-a$ 或 $f(x)=1-ax$；后者本身线性，一步到达。
>
> 2. 若 $f(x_n)=0$ 会怎样？
> 已找到精确根，公式给 $x_{n+1}=x_n$（只要 $f'(x_n)\ne0$）。
>
> 3. 为何初值应尽量靠近目标根？
> 线性化只在局部可信，靠近后才进入误差平方的收敛区。

### 本地材料

- [[Ses33a_Lecture_Notes.pdf#page=1|33a Newton’s Method]]
- [[Ses33b_Lecture_Notes.pdf#page=1|33b Accuracy]]
- [[Ses33c_Lecture_Notes.pdf#page=1|33c What Could Go Wrong?]]
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise033_Problems.pdf#page=1|Exercise 33 原题]] · [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise033_Solutions.pdf#page=1|Exercise 33 解答]]

**知识链小结：**Newton 法反复把非线性方程局部线性化；速度来自误差平方，风险也来自切线只代表局部。

## Problem Set 4

> [!info] 官方指定范围与材料
> 2C：1, 2, 4, 10, 13；2E：2, 3, 5, 7；2F：1。
> [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/PSet02_Problems.pdf#page=5|原题 2C 起始页]] · [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/PSet02_Solutions.pdf#page=10|官方解答 2C 起始页]]

### 2C：Max-min Problems

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

### 2E：Related Rates

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

### 2F：Locating Zeros; Newton’s Method

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

### Problem Set 4 错误检查

- 优化题必须把物理/几何可行域写进定义域。
- 相关变化率的正负来自坐标方向；若只问 speed，应报告绝对值并解释方向。
- Newton 法每步同时检查分母与残差；迭代数值不能替代唯一性证明。

**本组小结：**PS4 的共同结构是“变量受关系约束”：优化对自由变量求极值，相关变化率对时间求导，Newton 法则用切线约束产生下一次猜测。

---

## Part C：Mean Value Theorem, Antiderivatives and Differential Equations

## Session 34：Introduction to the Mean Value Theorem

### 本节问题与前置知识

**问题：**平均变化率与某个瞬时变化率为何必然相等？证明单调性时究竟需要哪些假设？

**前置：**连续、可导、极值定理和切线斜率。

### 34a：定理陈述与几何意义

[[Lagrange Mean Value Theorem|拉格朗日中值定理]]要求 $f$ 满足：

1. 在闭区间 $[a,b]$ 上连续；
2. 在开区间 $(a,b)$ 上可导；
3. $a<b$；

则存在至少一个 $c\in(a,b)$，使

$$
\boxed{
f'(c)=\frac{f(b)-f(a)}{b-a}
}.
$$

右边是端点割线斜率，左边是内部某点切线斜率。定理只保证 $c$ **存在**，不保证唯一，也不提供直接算法。

![[98_attachment/MIT18.01SC/unit02-mean-value-theorem.png|740]]

### 从 Fermat 引理到 Rolle 定理

**Fermat 引理：**若 $f$ 在内点 $c$ 可导且在 $c$ 取局部极大或极小，则 $f'(c)=0$。

以局部极大为例。足够小的 $h>0$ 有

$$
\frac{f(c+h)-f(c)}h\le0;
$$

足够小的 $h<0$ 时分子仍 $\le0$、分母为负，故

$$
\frac{f(c+h)-f(c)}h\ge0.
$$

若两侧极限共同存在，只能同时等于 $0$。

**[[Rolle's Theorem|罗尔定理]]：**若 $f$ 在 $[a,b]$ 连续、在 $(a,b)$ 可导且 $f(a)=f(b)$，则存在 $c\in(a,b)$ 使 $f'(c)=0$。

证明：连续函数由极值定理在闭区间达到最大、最小。若二者相同，$f$ 为常函数，任取内点即可。若不同，因为两端函数值相等，至少一个非平凡极值必须在内点取得；Fermat 引理给 $f'(c)=0$。

### [[Mean Value Theorem Proof|平均值定理证明]]

令割线斜率

$$
m=\frac{f(b)-f(a)}{b-a},
$$

构造“减去割线倾斜部分”的辅助函数

$$
g(x)=f(x)-m(x-a).
$$

$g$ 继承连续与可导性，且

$$
g(a)=f(a),
$$

$$
g(b)=f(b)-m(b-a)=f(b)-[f(b)-f(a)]=f(a).
$$

所以 $g(a)=g(b)$。由 Rolle 定理，某个 $c\in(a,b)$ 满足

$$
0=g'(c)=f'(c)-m.
$$

因此 $f'(c)=m$，证毕。

### 34b：三个重要推论

在区间 $I$ 上，若对所有内点：

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

$$
f(x_2)-f(x_1)=f'(c)(x_2-x_1)>0.
$$

这里把一个点处的局部导数信息通过 MVT 连接到了任意两个点之间。

### 配套练习：Taylor 余项的模式

$n$ 次 Taylor 多项式

$$
P_n(b)=\sum_{k=0}^n\frac{f^{(k)}(a)}{k!}(b-a)^k.
$$

Taylor 定理将 MVT 的结构推广为：在足够光滑的条件下，存在 $c$ 位于 $a,b$ 之间，使

$$
\boxed{
f(b)-P_n(b)=\frac{f^{(n+1)}(c)}{(n+1)!}(b-a)^{n+1}
}.
$$

练习取 $f(x)=x^3+2x+1,a=1,b=3,n=2$。有

$$
P_2(x)=4+5(x-1)+3(x-1)^2,
$$

$$
f(3)=34,\qquad P_2(3)=26,
$$

误差为 $8$。因 $f^{(3)}=6$，

$$
\frac{f^{(3)}(c)}{3!}(3-1)^3
=\frac66\cdot8=8,
$$

与实际误差完全一致。本课只验证模式，严格的一般证明留到 Taylor 单元。

### 易错点与边界

- 闭区间要求连续，开区间要求可导；端点不要求双侧导数。
- $|x|$ 在跨越 $0$ 的区间不可直接用 MVT，因为内点 $0$ 不可导。
- 跳跃函数破坏连续性，平均速度可以没有对应的瞬时速度。
- MVT 的 $c$ 可能不止一个。

> [!question]- 三道自检题与答案
> 1. $f(x)=x^2$ 在 $[0,2]$ 的 MVT 点？
> 割线斜率 $2$，$f'(c)=2c=2$，故 $c=1$。
>
> 2. 若 $f(a)=f(b)$，MVT 化为什么？
> Rolle 定理：某个内点 $f'(c)=0$。
>
> 3. 定理能否由端点值直接求出 $c$？
> 一般不能；还需函数具体形式，且可能多解。

### 本地材料

- [[Ses34a_Lecture_Notes.pdf#page=1|34a Mean Value Theorem]]
- [[Ses34b_Lecture_Notes.pdf#page=1|34b Consequences]]
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise034_Problems.pdf#page=1|Exercise 34 原题]] · [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise034_Solutions.pdf#page=1|Exercise 34 解答]]

**知识链小结：**MVT 用 Rolle 定理把“端点割线”转成“内部水平切线”；它是从局部导数推断全局行为的逻辑桥梁。

## Session 35：Using the Mean Value Theorem

### 本节问题与前置知识

**问题：**MVT 与线性近似有何不同？怎样把导数上下界变成函数差值和不等式？

**前置：**Session 34 的定理及单调性推论。

### 35a：精确存在式与近似式

线性近似在 $a$ 附近写成

$$
f(b)\approx f(a)+f'(a)(b-a).
$$

MVT 则精确地说

$$
f(b)=f(a)+f'(c)(b-a)
$$

其中 $c$ 位于 $a,b$ 之间，却通常未知。若 $f'$ 在短区间上变化很小，$f'(c)\approx f'(a)$，前者便解释了后者为何可靠。

若 $m\le f'(x)\le M$ 在 $[a,b]$ 上成立，则

$$
\boxed{
m(b-a)\le f(b)-f(a)\le M(b-a)
}
\qquad(a<b).
$$

这叫导数界的“有限增量形式”。

### 35b：用单调性证明指数不等式

证明 $e^x>1+x$（$x>0$）。令

$$
F(x)=e^x-(1+x).
$$

$F(0)=0$，且

$$
F'(x)=e^x-1>0\qquad(x>0).
$$

所以 $F$ 在 $(0,\infty)$ 递增，$F(x)>F(0)=0$。

进一步证明

$$
e^x>1+x+\frac{x^2}{2}\qquad(x>0).
$$

令

$$
G(x)=e^x-\left(1+x+\frac{x^2}{2}\right).
$$

$G(0)=0$ 且

$$
G'(x)=e^x-(1+x)>0
$$

正是上一结论，所以 $G(x)>0$。这展示了“前一个不等式成为下一个不等式的导数判据”。

### 配套练习：正弦差的 Lipschitz 界

对任意 $a\ne b$，MVT 给某个 $c$：

$$
\sin b-\sin a=\cos c\,(b-a).
$$

取绝对值并用 $|\cos c|\le1$：

$$
\boxed{|\sin b-\sin a|\le|b-a|}.
$$

这说明正弦函数的输出变化绝不会超过输入变化；“导数绝对值不超过 $1$”控制了全局变化。

### 易错点与边界

- 当 $b<a$ 时直接乘不等式会翻转方向；用绝对值形式可避免。
- 要证明 $F(x)>0$，需同时给基准值和单调方向，仅算 $F'>0$ 不够。
- $f'$ 的最大/最小若不存在，可使用任意已知上下界，不必真的求极值。

> [!question]- 三道自检题与答案
> 1. 若 $|f'|\le K$，可以推出什么？
> $|f(b)-f(a)|\le K|b-a|$。
>
> 2. 为什么这能给线性近似的粗误差控制？
> 对余项函数再用 MVT，可把误差变成导数差的界乘区间长度。
>
> 3. 证明 $\ln(1+x)<x$（$x>0$）。
> 令 $H=x-\ln(1+x)$；$H(0)=0$，$H'=x/(1+x)>0$。

### 本地材料

- [[Ses35a_Lecture_Notes.pdf#page=1|35a MVT and Linear Approximation]]
- [[Ses35b_Lecture_Notes.pdf#page=1|35b MVT and Inequalities]]
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise035_Problems.pdf#page=1|Exercise 35 原题]] · [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise035_Solutions.pdf#page=1|Exercise 35 解答]]

**知识链小结：**MVT 把导数的界积分式地累积成函数差的界；线性近似可看作区间很短时把未知 $f'(c)$ 换成已知 $f'(a)$。

## Session 36：Differentials

### 本节问题与前置知识

**问题：**$dy=f'(x)\,dx$ 是什么对象？它和真实改变量 $\Delta y$ 有何差别？

**前置：**线性近似、Leibniz 记号与链式法则。

### 36a：定义

对可导函数 $y=f(x)$，在给定基点 $x$ 处，把 $dx$ 视为可自由选取的小输入，定义输出微分；它是线性近似的增量形式

$$
\boxed{dy=f'(x)\,dx}.
$$

$dy$ 是 $dx$ 的线性函数。真实改变量是

$$
\Delta y=f(x+\Delta x)-f(x).
$$

取 $dx=\Delta x$ 时，

$$
\Delta y=dy+o(dx),
$$

故 $dy$ 是真实变化的一阶主部，而不是与 $\Delta y$ 永远相等。

### 36b：估计 $\sqrt[3]{64.1}$

令 $y=x^{1/3}$，基点 $x=64$：

$$
y_0=4,\qquad
dy=\frac13x^{-2/3}dx.
$$

在 $x=64$，

$$
dy=\frac1{48}dx.
$$

取 $dx=0.1$：

$$
\sqrt[3]{64.1}\approx y_0+dy
=4+\frac{0.1}{48}
\approx4.002083.
$$

单位也随导数传播：若 $x$ 的单位为体积，$dy/dx$ 是“长度/体积”，乘 $dx$ 后 $dy$ 恢复长度单位。

### 微分形式的链式法则

若 $y=f(u),u=g(x)$，

$$
dy=f'(u)\,du,\qquad du=g'(x)\,dx.
$$

代入即

$$
dy=f'(g(x))g'(x)\,dx.
$$

Leibniz 记号看似约分，背后的严格依据仍是复合函数链式法则。

### 配套练习：固定点的吸引性

若 $P(x_0)=x_0$ 且 $|P'(x_0)|<1$，线性化给

$$
P(x_0+dx)-x_0\approx P'(x_0)dx,
$$

所以一次迭代把小偏差缩小。若再假设 $P'$ 在 $x_0$ 连续，可取 $q<1$ 与一个邻域，使该邻域内 $|P'|\le q$。MVT 给

$$
|P(x)-x_0|
=|P(x)-P(x_0)|
\le q|x-x_0|.
$$

迭代后误差至多 $q^n$ 倍，严格得到吸引性。

对 $P(x)=ax(b-x)$，固定点解

$$
ax(b-x)=x
\Rightarrow
x_0=0,\quad x_1=\frac{ab-1}{a}.
$$

$$
P'(x)=ab-2ax.
$$

$P'(0)=ab>1$，所以 $0$ 不吸引；$P'(x_1)=2-ab$，第二固定点吸引当且仅当

$$
|2-ab|<1
\Longleftrightarrow
\boxed{1<ab<3}.
$$

### 易错点与边界

- $dx$ 与 $\Delta x$ 可以取同一个数，但概念不同：前者进入线性映射，后者进入原函数真实差值。
- 不能把微分的“像分数运算”当作无需链式法则的纯代数。
- $|P'(x_0)|<1$ 的严格吸引结论需要邻域控制；单点线性式只是直觉。

> [!question]- 三道自检题与答案
> 1. $y=x^2$ 在 $x=3,dx=0.01$ 时 $dy$？
> $dy=2x\,dx=0.06$。
>
> 2. 对同题真实 $\Delta y$？
> $(3.01)^2-9=0.0601$，与 $dy$ 差 $0.0001=(dx)^2$。
>
> 3. 固定点导数为 $-1/2$ 有何直觉？
> 偏差每步约缩半并交替换侧。

### 本地材料

- [[Ses36a_Lecture_Notes.pdf#page=1|36a Differentials]]
- [[Ses36b_Lecture_Notes.pdf#page=1|36b Differentials and Linear Approximation]]
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise036_Problems.pdf#page=1|Exercise 36 原题]] · [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise036_Solutions.pdf#page=1|Exercise 36 解答]]

**知识链小结：**微分把导数从“一个比值”改写成“输入小变化到输出一阶变化的线性映射”，为换元与微分方程准备语言。

## Session 37：Antiderivatives

### 本节问题与前置知识

**问题：**已知变化率 $f$，怎样恢复原函数？为什么答案必然带常数？

**前置：**求导公式、MVT 的常函数推论。

### 37a：定义与不定积分

若

$$
F'(x)=f(x),
$$

则称 $F$ 是 $f$ 的[[Antiderivative|反导数]]（antiderivative）。记号

$$
\int f(x)\,dx=F(x)+C
$$

叫[[Integral|积分]]（这里是不定积分，indefinite integral）。积分号表示“寻找全部反导数”，$dx$ 指明积分变量。

例如

$$
\int\sin x\,dx=-\cos x+C.
$$

### 37b–37c：基本公式

从求导公式反读：

$$
\boxed{\int x^n\,dx=\frac{x^{n+1}}{n+1}+C\quad(n\ne-1)}
$$

特殊指数 $n=-1$：

$$
\boxed{\int\frac{dx}{x}=\ln|x|+C}
$$

绝对值确保在 $x<0$ 的区间也有

$$
\frac{d}{dx}\ln|x|=\frac1x.
$$

其他常用式：

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

### 37d：唯一性证明

若在同一连通区间上

$$
F'=f,\qquad G'=f,
$$

则

$$
(F-G)'=0.
$$

由 MVT 的推论，$F-G$ 在该区间为常数：

$$
\boxed{F(x)=G(x)+C}.
$$

这证明了“$+C$”已经穷尽所有可能，而不是随手附加。

### 配套练习：从求导法则反读积分法则

和法则给线性性：

$$
\int(f+g)\,dx=\int f\,dx+\int g\,dx.
$$

乘积法则

$$
(FG)'=F'G+FG'
$$

反读得

$$
\int(F'G+FG')\,dx=FG+C.
$$

移项得到后续的分部积分雏形：

$$
\int F\,G'\,dx=FG-\int F'G\,dx.
$$

### 易错点与边界

- $\int x^{-1}dx$ 不能套幂公式，因为分母 $n+1=0$。
- 不定积分答案漏写 $C$。
- 两个看似不同的答案可能只差常数；用恒等式或相减求导检查。
- 反导数通常应在一个连通区间内讨论。

> [!question]- 三道自检题与答案
> 1. $\int3x^2dx$？
> $x^3+C$。
>
> 2. 为何 $\frac12\sin^2x$ 与 $-\frac12\cos^2x$ 都可作为 $\sin x\cos x$ 的反导数？
> 两者导数相同，且相差常数 $1/2$。
>
> 3. 怎样最快检查不定积分？
> 对答案求导，确认回到原被积函数，并检查定义域。

### 本地材料

- [[Ses37a_Lecture_Notes.pdf#page=1|37a Introduction to Antiderivatives]]
- [[Ses37b_Lecture_Notes.pdf#page=1|37b Antiderivative of x^a]]
- [[Ses37c_Lecture_Notes.pdf#page=1|37c Basic Antiderivatives]]
- [[Ses37d_Lecture_Notes.pdf#page=1|37d Unique up to a Constant]]
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise037_Problems.pdf#page=1|Exercise 37 原题]] · [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise037_Solutions.pdf#page=1|Exercise 37 解答]]

**知识链小结：**反导数逆转求导；MVT 保证这种逆转只丢失一个常数，这正是后续由变化率重建函数的基础。

## Session 38：Integration by Substitution

### 本节问题与前置知识

**问题：**复合函数求导的链式法则怎样反向使用？选换元变量时应看什么？

**前置：**微分、基本反导数与链式法则。

### 38a：[[Integration by Substitution|换元积分]]的来源

链式法则：

$$
\frac{d}{dx}F(g(x))=F'(g(x))g'(x).
$$

若 $F'=f$，反读为

$$
\boxed{\int f(g(x))g'(x)\,dx=F(g(x))+C}.
$$

写 $u=g(x),du=g'(x)dx$，就得到

$$
\int f(u)\,du.
$$

这不是凭符号约掉 $dx$，而是链式法则的结构匹配。

**讲义例题：**

$$
\int x^3(x^4+2)^5\,dx.
$$

令 $u=x^4+2$，$du=4x^3dx$：

$$
\begin{aligned}
\int x^3(x^4+2)^5dx
&=\frac14\int u^5du\\
&=\frac{u^6}{24}+C\\
&=\boxed{\frac{(x^4+2)^6}{24}+C}.
\end{aligned}
$$

### 38b–38c：“高级猜测”与常数修正

对

$$
\int\frac{x}{\sqrt{1+x^2}}dx,
$$

看到内层 $1+x^2$ 的导数含 $x$，猜结果形如 $\sqrt{1+x^2}$。验证：

$$
\frac{d}{dx}\sqrt{1+x^2}
=\frac{x}{\sqrt{1+x^2}}.
$$

所以答案直接得到。对 $\int e^{6x}dx$，猜 $e^{6x}$ 后求导多出因子 $6$，于是

$$
\int e^{6x}dx=\frac16e^{6x}+C.
$$

“猜测”必须由求导验证；它是熟练后的模式识别，不是省略证明。

### 配套练习：同一积分的两种换元

$$
I=\int\tan x\,\sec^2x\,dx.
$$

令 $u=\tan x,du=\sec^2x\,dx$：

$$
I=\frac12\tan^2x+C.
$$

令 $v=\sec x,dv=\sec x\tan x\,dx$，把原式写成 $\sec x(\sec x\tan x\,dx)$：

$$
I=\frac12\sec^2x+\widetilde C.
$$

由 $\sec^2x-\tan^2x=1$，两答案相差 $1/2$，因此等价。

### 易错点与边界

- 只替换内层表达式却没替换对应微分因子。
- 换元完成后答案仍留变量 $u$。
- 常数因子漏乘倒数；最稳妥的校验是对最终答案求导。
- $\ln$ 型答案应根据定义域写绝对值。

> [!question]- 三道自检题与答案
> 1. $\int2x\cos(x^2)dx$？
> $\sin(x^2)+C$。
>
> 2. $\int x e^{x^2}dx$？
> $\frac12e^{x^2}+C$。
>
> 3. 为什么令 $u=x$ 对任何积分都“合法”却没帮助？
> 它没有简化结构；好换元要把复合内层及其导数同时吸收。

### 本地材料

- [[Ses38a_Lecture_Notes.pdf#page=1|38a Substitution Example]]
- [[Ses38b_Lecture_Notes.pdf#page=1|38b Advanced Guessing]]
- [[Ses38c_Lecture_Notes.pdf#page=1|38c More Examples]]
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise038_Problems.pdf#page=1|Exercise 38 原题]] · [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise038_Solutions.pdf#page=1|Exercise 38 解答]]

**知识链小结：**换元积分就是反向链式法则；选元的目标是让“内层函数 + 它的微分”一起变成一个基本积分。

## Session 39：Introduction to Differential Equations

### 本节问题与前置知识

**问题：**方程若包含未知函数及其导数，什么叫“解”？怎样从变化规律恢复一族函数？

**前置：**反导数、微分与换元。

### 39a：微分方程与通解

[[Differential Equation|微分方程]]（differential equation）把未知函数与导数联系起来。最简单的

$$
\frac{dy}{dx}=f(x)
$$

的通解是

$$
y=\int f(x)\,dx.
$$

更有内容的讲义例题：

$$
\frac{dy}{dx}+xy=0
\quad\Longleftrightarrow\quad
\frac{dy}{dx}=-xy.
$$

先假设 $y\ne0$，分离变量：

$$
\frac{dy}{y}=-x\,dx.
$$

积分：

$$
\ln|y|=-\frac{x^2}{2}+C.
$$

指数化并把符号吸收到任意常数 $A$：

$$
\boxed{y=Ae^{-x^2/2}}.
$$

代回检查：

$$
y'=-xAe^{-x^2/2}=-xy.
$$

$A=0$ 也成立，补回了分离过程中除以 $y$ 可能丢失的零解。若给初值 $y(0)=3$，则 $A=3$，唯一选出 $y=3e^{-x^2/2}$。

### 39b：一般分离变量框架

若

$$
\frac{dy}{dx}=f(x)g(y),
$$

先另列 $g(y)=0$ 的常数平衡解。对 $g(y)\ne0$：

$$
\frac{dy}{g(y)}=f(x)\,dx.
$$

若 $H'(y)=1/g(y)$、$F'(x)=f(x)$，则

$$
\boxed{H(y)=F(x)+C}.
$$

这可能已经是合格的隐式通解；能方便求逆时再写成 $y=H^{-1}(F(x)+C)$。

![[98_attachment/MIT18.01SC/unit02-separation-of-variables.png|780]]

### 配套练习：代回检查候选解

**(a)** $y=e^x/3$：

$$
y''=\frac13e^x,
\quad
4y''-y=\frac43e^x-\frac13e^x=e^x.
$$

故满足 $4y''-y=e^x$。

**(b)** $y=1/x$（$x\ne0$）：

$$
y'=-x^{-2},\qquad y''=2x^{-3}.
$$

$$
x^2y''+3xy'+y
=2x^{-1}-3x^{-1}+x^{-1}=0.
$$

代回不仅查代数，也暴露定义域 $x\ne0$。

### 易错点与边界

- 通解是一族函数；初值才选择常数。
- 两边积分只需一个任意常数，因为 $C_2-C_1$ 仍是任意常数。
- 除以 $g(y)$ 会丢掉 $g(y)=0$ 的平衡解。
- 隐式解不必强行解出 $y$；强行开根号还可能漏分支。

> [!question]- 三道自检题与答案
> 1. 如何确认候选函数是解？
> 求出方程所需各阶导数，代回原方程，在其定义区间逐点成立。
>
> 2. 为什么 $A$ 可以为负而 $e^C$ 只能为正？
> 从 $|y|=e^Ce^{-x^2/2}$ 拆正负支，并把符号并入 $A$。
>
> 3. 初值 $y(0)=0$ 选出哪条解？
> $A=0$，即 $y\equiv0$。

### 本地材料

- [[Ses39a_Lecture_Notes.pdf#page=1|39a Introduction to ODEs]]
- [[Ses39b_Lecture_Notes.pdf#page=1|39b Separation of Variables]]
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise039_Problems.pdf#page=1|Exercise 39 原题]] · [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise039_Solutions.pdf#page=1|Exercise 39 解答]]

**知识链小结：**微分方程给变化规律，积分恢复函数族；完整答案由通解、平衡解、初值与定义区间共同组成。

## Session 40：Separation of Variables

### 本节问题与前置知识

**问题：**怎样从几何斜率条件建立并求解微分方程？分离变量时有哪些隐藏的奇点？

**前置：**Session 39 的一般框架、$\int dx/x$ 与隐式曲线。

### 40a：最简单的[[Separation of Variables|分离变量法]]

$$
\frac{dy}{dx}=f(x)
\quad\Rightarrow\quad
dy=f(x)dx
\quad\Rightarrow\quad
y=\int f(x)dx.
$$

这说明普通求反导数是分离变量法的特殊情形。

### 40b：切线斜率是径向斜率的两倍

点 $(x,y)$ 到原点的射线斜率为 $y/x$。条件给

$$
\frac{dy}{dx}=\frac{2y}{x},
\qquad x\ne0.
$$

对 $y\ne0$：

$$
\frac{dy}{y}=2\frac{dx}{x}.
$$

$$
\ln|y|=2\ln|x|+C.
$$

指数化：

$$
|y|=e^Cx^2.
$$

合并正、负与零解：

$$
\boxed{y=Ax^2}.
$$

代回 $y'=2Ax=2y/x$ 对 $x\ne0$ 成立。注意原方程在 $x=0$ 未规定斜率，因此穿过 $x=0$ 时可能把左右不同参数的抛物线拼接；这是奇点导致的非唯一性。

### 40c：与抛物线正交的轨线

抛物线族 $y=ax^2$ 在点 $(x,y)$ 的斜率为 $2y/x$。正交曲线斜率取负倒数：

$$
\frac{dy}{dx}=-\frac{x}{2y}.
$$

分离并积分：

$$
2y\,dy=-x\,dx
\Rightarrow
y^2=-\frac{x^2}{2}+C.
$$

所以

$$
\boxed{y^2+\frac{x^2}{2}=C}.
$$

$C>0$ 时是一族椭圆。写成显式分支

$$
y=\pm\sqrt{C-\frac{x^2}{2}}
$$

可看出顶、底半支分别是函数；在 $y=0$ 处有竖直切线，原斜率式分母也为零。

### 配套练习：指数与受限增长

**指数增长**

$$
\frac{dy}{dx}=ry
\Rightarrow
\frac{dy}{y}=r\,dx
\Rightarrow
\boxed{y=Ae^{rx}}.
$$

**Logistic 型受限增长**

$$
\frac{dy}{dx}=ry(s-y),\qquad s>0.
$$

先记录平衡解 $y=0,s$。对其他解：

$$
\frac{dy}{y(s-y)}=r\,dx.
$$

利用

$$
\frac1{y(s-y)}=\frac1s\left(\frac1y+\frac1{s-y}\right),
$$

积分：

$$
\frac1s\ln\left|\frac{y}{s-y}\right|=rx+C.
$$

令指数常数为 $A$：

$$
\frac{y}{s-y}=Ae^{srx}.
$$

解出

$$
\boxed{y=\frac{sAe^{srx}}{1+Ae^{srx}}}.
$$

对典型初值 $0<y_0<s$ 有 $A>0$，且 $x\to\infty$ 时 $y\to s$，与 $0<y<s$ 时 $y'>0$ 的符号分析相符。

### 易错点与边界

- 负倒数斜率只适用于有限非零斜率；水平/竖直需单独解释。
- 分离前列平衡解，分离后检查分母为零的位置。
- 显式开根会产生多个分支；一个隐式曲线未必是全局函数。
- 求出公式后要确定由初值连通得到的最大区间，不能跨越分母零点。

> [!question]- 三道自检题与答案
> 1. $y'=ky$ 的零解是否包含在 $Ae^{kx}$ 中？
> 包含，取 $A=0$。
>
> 2. Logistic 方程在 $y>s$ 时变化方向？
> 若 $r>0$，$y(s-y)<0$，所以 $y$ 下降趋向 $s$。
>
> 3. 正交曲线方程为何可保留隐式？
> 椭圆整体不是单值函数，隐式式同时表示上下分支且更自然。

### 本地材料

- [[Ses40a_Lecture_Notes.pdf#page=1|40a dy/dx=f(x)]]
- [[Ses40b_Lecture_Notes.pdf#page=1|40b Differential Equations and Slope I]]
- [[Ses40c_Lecture_Notes.pdf#page=1|40c Differential Equations and Slope II]]
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise040_Problems.pdf#page=1|Exercise 40 原题]] · [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Exercise040_Solutions.pdf#page=1|Exercise 40 解答]]

**知识链小结：**分离变量把斜率关系转为两个反导数的等式；平衡解、奇点和分支决定公式真正代表哪些曲线。

## Problem Set 5

> [!info] 官方指定范围与材料
> 2G：1b, 2b, 5, 6；3A：1d, 1e, 2a, 2c, 2e, 2g, 2i, 2k, 3a, 3c, 3e, 3g；3F：1c, 1d, 2a, 2e, 4b, 4c, 4d, 8b。
> 2G 使用 [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/PSet02_Problems.pdf#page=12|Applications of Differentiation 原题]] 与 [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/PSet02_Solutions.pdf#page=28|官方解答]]；3A、3F 使用 [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/PSet03_Problems.pdf#page=1|Integration 原题]] 与内容较完整的修订文件 [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/PSet03_Solutions_2.pdf#page=1|Integration 修订解答]]。

### 2G：Mean Value Theorem

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

> [!example]- 2G-5：三个零点推出某处二阶导为零
> 已知 $f''$ 在含 $a<b<c$ 的区间存在，且 $f(a)=f(b)=f(c)=0$。
> 在 $[a,b]$ 对 $f$ 用 Rolle 定理，存在 $q_1\in(a,b)$ 使 $f'(q_1)=0$；在 $[b,c]$ 再用一次，存在 $q_2\in(b,c)$ 使 $f'(q_2)=0$。
> 对 $f'$ 在 $[q_1,q_2]$ 用 Rolle 定理，存在 $p\in(q_1,q_2)\subset(a,c)$：
> $$
> \boxed{f''(p)=0}.
> $$
> 每次应用都要说明相应函数连续、内点可导；$f''$ 存在保证 $f'$ 可导，从而连续。

> [!example]- 2G-6：用 MVT 证明单调与常函数
> 任取 $a\le x_1<x_2\le b$。MVT 给某个 $c\in(x_1,x_2)$：
> $$
> f(x_2)-f(x_1)=f'(c)(x_2-x_1).
> $$
> **(a)** 若区间内 $f'>0$，右边为正，故 $f(x_2)>f(x_1)$，所以 $f$ 严格递增。
> **(b)** 若区间内 $f'=0$，右边为零，任意两点函数值相同，所以 $f$ 为常函数。
> 结论依赖 $f$ 在闭子区间连续、开子区间可导。

### 3A：Differentials and Indefinite Integration

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

> [!example]- 3A-2a：逐项反导
> $$
> \begin{aligned}
> \int(2x^4+3x^2+x+8)\,dx
> &=\frac25x^5+x^3+\frac12x^2+8x+C.
> \end{aligned}
> $$
> 每项求导都恢复对应被积项。

> [!example]- 3A-2c：$\int\sqrt{8+9x}\,dx$
> 令 $u=8+9x$，$du=9dx$：
> $$
> \int\sqrt{8+9x}\,dx
> =\frac19\int u^{1/2}du
> =\boxed{\frac2{27}(8+9x)^{3/2}+C}.
> $$

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

### 3F：Differential Equations — Separation of Variables

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

### Problem Set 5 错误检查

- MVT/Rolle 每次使用都要重新核对连续与可导区间。
- 反导答案用求导回验；这一步识别出了 3A-2e 的官方排版错误。
- 分离变量前列 $g(y)=0$ 的平衡解，解出后再定最大区间。
- 初值问题的常数、符号和定义域是答案的一部分，不是可省略的附注。

**本组小结：**PS5 把 MVT 的严谨推理、微分记号、反向链式法则和微分方程连成一条完整链：导数控制函数，反导数重建函数。

---

## Exam 2

## Session 41：Review for Exam 2

### 本节问题与前置知识

**问题：**如何在考试前把本章看似分散的方法压缩成少数稳定流程？

**前置：**Session 23–40 全部内容，尤其是定义域与边界检查。

### 41a：六类题的最短检查表

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

### 贯穿全章的一个问题

本章所有方法都在回答：

> 已知函数的导数，能对函数本身说什么？

- $f'(a)$ 给局部直线；
- $f''(a)$ 给局部曲率；
- $f'$ 的符号给单调；
- MVT 把局部斜率变成整体差值；
- 反导数从全部斜率恢复函数；
- 微分方程从变化规律选出函数族。

### 易错点与考试策略

- 先写结构再代数，避免在长计算中忘记定义域和所求量。
- 图像题用符号表而非凭外观；优化题的“端点”可能是 $0^+$ 或 $\infty$。
- 若答案维度不对、Newton 残差变大、反导求导不回原式，应立即停下回查。

> [!question]- 三道自检题与答案
> 1. 哪类题最常因漏端点丢分？
> 优化和绝对极值题。
>
> 2. 哪类题必须“先求导后代值”？
> 相关变化率题。
>
> 3. 哪两类题最适合用求导回验？
> 不定积分和微分方程。

### 本地材料

- [[Ses41a_Lecture_Notes.pdf#page=1|41a Review for Test 2]]

**知识链小结：**考试复习不是再记一遍公式，而是把每类题的输入、主步骤、边界检查和回验固定成流程。

## Session 42：Materials for Exam 2

### 本节问题与前置知识

**问题：**能否在一套综合题中，同时正确使用近似、相关变化率、作图、优化、Newton 法和 MVT？

**材料说明：**官网 Session 42 是 Exam 2 材料页；本地没有 Ses42 讲义，以考试原题与官方解答组成。本节对官方简答补足推导，并明确两处可见笔误。

### Exam 2 第 1 题：二次近似与 $\ln1.2$

> [!example] 题目
> (a) 写出二阶可导函数 $f$ 在 $x=a$ 的一般二次近似。
> (b) 用它估计 $\ln1.2$。

**(a)** 令二次多项式在 $a$ 匹配 $f,f',f''$：

$$
\boxed{
Q_a(x)=f(a)+f'(a)(x-a)+\frac{f''(a)}2(x-a)^2
}.
$$

**(b)** 取 $f(x)=\ln x,a=1$：

$$
f(1)=0,\qquad f'(1)=1,\qquad f''(1)=-1.
$$

代 $x=1.2$：

$$
\begin{aligned}
\ln1.2
&\approx0+1(0.2)-\frac12(0.2)^2\\
&=0.2-0.02\\
&=\boxed{0.18}.
\end{aligned}
$$

真值约 $0.18232$，误差约 $0.00232$；二次近似比线性值 $0.2$ 更准。

> [!warning] 常见错误
> 忘记 $f''(a)/2$ 的 $1/2$；把展开点错取为 $0$，而 $\ln0$ 无定义。

### Exam 2 第 2 题：圆锥盐堆

> [!example] 题目
> 盐以 $30\ \mathrm{ft^3/min}$ 落下形成圆锥，圆锥高度始终等于底面直径。高为 $10$ ft 时，求高度增长率。

设半径 $r$、高 $h$。条件 $h=2r$，即 $r=h/2$。体积

$$
V=\frac13\pi r^2h
=\frac13\pi\left(\frac h2\right)^2h
=\frac{\pi}{12}h^3.
$$

对时间求导：

$$
\frac{dV}{dt}
=\frac{\pi}{4}h^2\frac{dh}{dt}.
$$

代入 $dV/dt=30,h=10$：

$$
30=25\pi\frac{dh}{dt}.
$$

$$
\boxed{\frac{dh}{dt}=\frac6{5\pi}\ \mathrm{ft/min}}.
$$

> [!warning] 常见错误
> 把“高度等于直径”写成 $h=r$；或在求导前把 $h=10$ 当成恒定值。

### Exam 2 第 3 题：$f(x)=x-3x^{1/3}$ 作图

> [!example] 题目
> 标出局部极值、增减区间和渐近线；拐点可辅助作图。

定义域为全体实数，且 $f$ 为奇函数。零点：

$$
x-3x^{1/3}=x^{1/3}(x^{2/3}-3)=0,
$$

所以

$$
\boxed{x=-3\sqrt3,\ 0,\ 3\sqrt3}.
$$

对 $x\ne0$：

$$
f'(x)=1-x^{-2/3}=1-\frac1{|x|^{2/3}}.
$$

临界候选为 $x=-1,0,1$；$0$ 处函数存在但导数无穷。符号表：

| 区间 | $(-\infty,-1)$ | $(-1,0)$ | $(0,1)$ | $(1,\infty)$ |
|---|---:|---:|---:|---:|
| $f'$ | $+$ | $-$ | $-$ | $+$ |

因此：

$$
\boxed{\text{递增于 }(-\infty,-1)\text{ 与 }(1,\infty)}
$$

$$
\boxed{\text{递减于 }(-1,0)\text{ 与 }(0,1)}
$$

（也可在单调意义上合写为 $(-1,1)$，但导数在 $0$ 不存在。）

$$
f(-1)=2,\qquad f(1)=-2.
$$

故局部最大 $\boxed{(-1,2)}$，局部最小 $\boxed{(1,-2)}$。$x=0$ 有向下的竖直切线，凹凸在其两侧改变。无竖直、水平或斜渐近线；尽管 $f(x)/x\to1$，但 $f(x)-x=-3x^{1/3}$ 不趋于 $0$。两端 $f(x)\to\pm\infty$。

> [!warning] 官方勘误
> 本地官方解答文字把局部最大点写成了 $(1,2)$；由函数值与奇对称性可确认应为 $\boxed{(-1,2)}$。

### Exam 2 第 4 题：圆柱加半球顶储罐

> [!example] 题目
> 圆柱有底、上接半球顶，固定体积 $V$；求耗金属最少的尺寸。

设共同半径 $r>0$，圆柱部分高 $h\ge0$。体积：

$$
V=\pi r^2h+\frac23\pi r^3.
$$

外表面积包含圆柱底、圆柱侧面、半球曲面：

$$
S=\pi r^2+2\pi rh+2\pi r^2
=3\pi r^2+2\pi rh.
$$

由体积约束

$$
h=\frac{V}{\pi r^2}-\frac23r.
$$

代入：

$$
S(r)=3\pi r^2+2\pi r
\left(\frac{V}{\pi r^2}-\frac23r\right)
=\frac53\pi r^2+\frac{2V}{r}.
$$

$$
S'(r)=\frac{10}{3}\pi r-\frac{2V}{r^2}.
$$

令 $S'=0$：

$$
\frac{10}{3}\pi r^3=2V
\Rightarrow
\boxed{r=\left(\frac{3V}{5\pi}\right)^{1/3}}.
$$

由 $V=\frac53\pi r^3$ 回代：

$$
h=\frac{5r}{3}-\frac{2r}{3}=r.
$$

所以

$$
\boxed{h=r=\left(\frac{3V}{5\pi}\right)^{1/3}}.
$$

并且

$$
S''(r)=\frac{10}{3}\pi+\frac{4V}{r^3}>0,
$$

故 $S$ 严格凸，唯一可行临界点为全局最小；它也满足 $h>0$。

> [!warning] 常见错误
> 漏算底面积或把半球曲面写成 $4\pi r^2$；求出 $r$ 后忘记回答 $h$。

### Exam 2 第 5 题：Newton 法为什么失败

> [!example] 题目
> $f(x)=x^3-3x+7$，初值 $x_1=2$，解释为何迭代最终失败。

$$
f'(x)=3x^2-3.
$$

第一步：

$$
x_2
=2-\frac{f(2)}{f'(2)}
=2-\frac{8-6+7}{12-3}
=2-\frac99
=1.
$$

但

$$
f'(1)=0,\qquad f(1)=5\ne0.
$$

所以下一步

$$
x_3=1-\frac{f(1)}{f'(1)}
$$

分母为零而无定义。几何上，$(1,5)$ 处切线水平，不与 $x$ 轴相交，无法产生下一猜测。

$$
\boxed{\text{迭代在到达 }x=1\text{ 后因水平切线停止。}}
$$

> [!warning] 常见错误
> 只说“除以零”而没有展示哪一步到达该点；或误以为 $f'(1)=0$ 表示 $x=1$ 是方程的根。

### Exam 2 第 6 题：用 MVT 证明平方根不等式

> [!example] 题目
> 证明当 $x>0$ 时
> $$
> \sqrt{1+x}<1+\frac x2.
> $$

令

$$
g(x)=1+\frac x2-\sqrt{1+x}.
$$

$g$ 在 $[0,x]$ 连续、在 $(0,x)$ 可导，且

$$
g(0)=0.
$$

对任意 $t>0$：

$$
g'(t)=\frac12-\frac1{2\sqrt{1+t}}.
$$

因 $\sqrt{1+t}>1$，

$$
\frac1{2\sqrt{1+t}}<\frac12,
$$

故 $g'(t)>0$。MVT 的单调性推论给 $g(x)>g(0)=0$，即

$$
\boxed{\sqrt{1+x}<1+\frac x2}.
$$

也可直接对 $\sqrt{1+t}$ 在 $[0,x]$ 应用 MVT，得到同一证明。

> [!warning] 常见错误
> 用“图上看切线在曲线上方”代替证明；正确逻辑需要由导数符号或凹性定理推出。

### Exam 2 总结与错误诊断

| 题号 | 核心规则 | 必做检查 |
|---|---|---|
| 1 | 二次近似 | 基点、$1/2$、误差量级 |
| 2 | 相关变化率 | 相似/比例关系、先求导后代值、单位 |
| 3 | 曲线作图 | $f'$ 不存在点、符号表、零点、渐近线定义 |
| 4 | 约束优化 | 表面积组成、可行域、全局最小证明 |
| 5 | Newton 法 | 实际迭代、分母 $f'$、几何解释 |
| 6 | MVT | 连续/可导假设、基准值、严格不等号 |

### 易错点与边界

- 计算正确但不回答单位、位置或全部尺寸，仍是不完整答案。
- 图像题的 $f'$ 不存在点与优化题的可行边界都必须主动加入候选。
- 使用官方答案时也应回验；本卷第 3 题的极大点文字确有笔误。

> [!question]- 三道综合自检与答案
> 1. 第 2 题若盐流入率翻倍，高度增长率如何变化？
> 在同一高度下线性翻倍，因为 $h'$ 与 $V'$ 成正比。
>
> 2. 第 4 题的最优关系中，半球总高是否等于 $r$？
> 不是；$h=r$ 指圆柱部分高度，罐体总高为 $h+r=2r$。
>
> 3. 第 6 题为何是严格小于而非小于等于？
> $x>0$ 时区间内部 $g'(t)>0$，故 $g(x)>g(0)$；只有 $x=0$ 取等。

### 本地材料

- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/07_Exams/Exam2_Problems.pdf#page=1|Exam 2 原题（题 1–6 各页）]]
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/07_Exams/Exam2_Solutions.pdf#page=1|Exam 2 官方解答]]

**知识链小结：**Exam 2 检查的是同一能力的六种外观：从导数提取可验证的信息，并对近似范围、定义域、约束和定理假设负责。

---

## 本章总复习：从局部到整体再到逆问题

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

- [ ] 我能解释线性化、二次近似匹配了哪些导数，而不只会套公式。
- [ ] 我画图时总先写定义域，并区分临界点、间断点与真正拐点。
- [ ] 我做优化时会比较端点；做相关变化率时会先求导后代值。
- [ ] 我能从切线方程自行推导 Newton 公式，并识别 $f'=0$ 的失败。
- [ ] 我能完整写出 MVT 的闭区间连续、开区间可导和 $c\in(a,b)$。
- [ ] 我会对不定积分和微分方程解求导回验，并补回平衡解、定义区间。

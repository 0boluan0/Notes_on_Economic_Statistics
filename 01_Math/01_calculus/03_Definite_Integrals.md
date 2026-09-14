---
aliases:
  - MIT 18.01SC Unit 3
  - The Definite Integral and Its Applications
  - 定积分及其应用
tags:
  - math/calculus
  - course/mit-ocw
  - calculus/integration
source: https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/pages/unit-3-the-definite-integral-and-its-applications/
---

# MIT 18.01SC Unit 3: The Definite Integral and Its Applications

连续阅读：[[01_Math/01_calculus/03_定积分与应用]] · 主题关系：[[定积分与应用.canvas]]。本页保留 Session、课堂推导、习题和考试的永久记录。
<!-- bilingual-en:start -->
This page preserves the permanent session record, classroom derivations, exercises, and exam work.
<!-- bilingual-en:end -->

本页沿用 MIT 18.01SC 的编号：FTC I 是[[Newton-Leibniz公式|原函数端点求值]]，FTC II 是[[积分求导定理|累积函数求导]]；分别在 Session 47、51 引入，Session 52 依次证明 FTC II 与 FTC I。其他教材可能采用相反编号，跨来源比较时以公式内容为准。
<!-- bilingual-en:start -->
This page follows MIT 18.01SC's numbering: FTC I is [[Newton-Leibniz公式|evaluation by antiderivative endpoints]], and FTC II is [[积分求导定理|differentiation of an accumulation function]]. They are introduced in Sessions 47 and 51; Session 52 proves FTC II and then FTC I. Other texts may reverse the numbering, so compare the actual formulas across sources.
<!-- bilingual-en:end -->

> [!abstract] 本章主线
> 导数解决“已知量，求瞬时变化率”；定积分反过来解决“已知变化率和初值，求累计变化”。本章从矩形面积的极限构造定积分，证明微积分基本定理把“求和”与“求导”连接起来，再把这一工具用于面积、体积、平均值、概率和数值计算。
> <!-- bilingual-en:start -->
> Derivatives recover an instantaneous rate of change from a known quantity; definite integrals reverse the viewpoint, recovering accumulated change from a rate and an initial value. This chapter constructs the definite integral as a limit of rectangle sums, proves that the Fundamental Theorem of Calculus connects accumulation with differentiation, and then applies the integral to area, volume, averages, probability, and numerical approximation.
> <!-- bilingual-en:end -->

## 阅读地图

- Part A：Session 43–50，定积分、Riemann 和与第一基本定理
- Problem Set 6
- Part B：Session 51–59，第二基本定理、面积与体积
- Problem Set 7
- Part C：Session 60–65，平均值、概率与数值积分
- Problem Set 8
- Exam 3：Session 66–67

## 统一记号
<!-- bilingual-en:start -->
*Notation used throughout*
<!-- bilingual-en:end -->

把 \([a,b]\) 分割为
<!-- bilingual-en:start -->
Split \([a,b]\) into
<!-- bilingual-en:end -->

$$
a=x_0<x_1<\cdots<x_n=b,
$$

第 \(i\) 段宽度为 \(\Delta x_i=x_i-x_{i-1}\)，样本点 \(x_i^*\in[x_{i-1},x_i]\)。[[黎曼和]]（Riemann sum）是
<!-- bilingual-en:start -->
The $i$th subinterval has width $\Delta x_i=x_i-x_{i-1}$, and its sample point is $x_i^*\in[x_{i-1},x_i]$. The [[黎曼和|Riemann sum]] is
<!-- bilingual-en:end -->

$$
\sum_{i=1}^{n}f(x_i^*)\Delta x_i.
$$

等分时 $\Delta x=(b-a)/n$。单位检查始终是“被积函数单位 × \(dx\) 单位”。
<!-- bilingual-en:start -->
For an equal partition, $\Delta x=(b-a)/n$. The units are always “units of the integrand × units of $dx$.”
<!-- bilingual-en:end -->

![[98_attachment/MIT18.01SC/unit03-riemann-sums.png]]

---

## Part A：Definition and First Fundamental Theorem

## Session 43：Definite Integrals

### 问题、定义与直觉
<!-- bilingual-en:start -->
*Questions, Definitions and Intuitions*
<!-- bilingual-en:end -->

若有向速度 \(v(t)\) 在短时间 \(\Delta t\) 内近似不变，小段位移约为 \(v(t_i^*)\Delta t\)，总位移约为
<!-- bilingual-en:start -->
If the signed velocity \(v(t)\) is approximately constant over a short time \(\Delta t\), the displacement is about \(v(t_i^*)\Delta t\), and the total displacement is about
<!-- bilingual-en:end -->

$$
\sum v(t_i^*)\Delta t.
$$

若要求总路程，应累计速率 $|v(t)|$：$\int_a^b|v(t)|dt$；位移则是 $\int_a^bv(t)dt$。见[[净变化与总变化]]。
<!-- bilingual-en:start -->
Total distance accumulates speed: $\int_a^b|v(t)|dt$. Displacement is $\int_a^bv(t)dt$; see [[净变化与总变化|net change and total change]].
<!-- bilingual-en:end -->

面积、质量、成本和概率都是同一种结构：“密度 × 小宽度”再累计。
<!-- bilingual-en:start -->
Area, mass, cost, and probability all share the same structure: accumulate “density × a small width.”
<!-- bilingual-en:end -->

若随着最大分割宽度
<!-- bilingual-en:start -->
If the mesh size of the partition tends to zero,
<!-- bilingual-en:end -->

$$
\|P\|=\max_i\Delta x_i\to0,
$$

所有 Riemann 和都趋于同一个有限值 \(I\)，由此定义[[定积分]]
<!-- bilingual-en:start -->
All Riemann sums tend to the same finite value \(I\), which defines the [[定积分|definite integral]]
<!-- bilingual-en:end -->

$$
\boxed{\int_a^bf(x)\,dx=I}.
$$

它是有向累计量：
<!-- bilingual-en:start -->
It is an oriented accumulation:
<!-- bilingual-en:end -->

$$
\int_a^af=0,\qquad \int_b^af=-\int_a^bf.
$$

负函数贡献负面积，因此定积分不总等于几何面积。
<!-- bilingual-en:start -->
A negative function contributes negative signed area, so a definite integral is not always the same as geometric area.
<!-- bilingual-en:end -->

> [!note] 严格性边界
> 本课使用“闭区间上的连续函数可积”。完整证明依赖一致连续性；这里的重点是 Riemann 和极限及其后果。
> <!-- bilingual-en:start -->
> This course uses the fact that a continuous function on a closed interval is integrable. A complete proof relies on uniform continuity; the focus here is the limit of Riemann sums and its consequences.
> <!-- bilingual-en:end -->

![[连续函数可积]]

### 本地材料

- [[Ses43a_Lecture_Notes.pdf|43a Introduction to Definite Integrals]]
- [[Ses43b_Lecture_Notes.pdf|43b Definition of the Definite Integral]]
- [[Exercise043_Problems.pdf|Exercise 43]] · [[Exercise043_Solutions.pdf|解答]]

> [!example]- Exercise 43：只凭图形估计积分
> 三幅图的纵横坐标刻度不同，必须先读刻度，再把阴影与可计算的三角形、矩形比较。
>
> 1. 第一幅阴影恰为底 $2$、高 $4$ 的三角形，故
>    $$
>    \int f(x)\,dx=\frac12\cdot2\cdot4=\boxed{4},
>    $$
>    选 (b)。
> 2. 第二幅曲线下的面积小于包围它的三角形面积 \(1/2\)，但明显接近该值而非 \(1/4\)，最佳估计为 \(\boxed{1/2}\)，选 (c)。
> 3. 第三幅可粗分成面积约 \(1/2\) 的矩形和面积约 \(1/4\) 的补充部分，故最佳估计为 \(\boxed{3/4}\)，选 (b)。
>
> 这里不是要求精确积分，而是检查“面积必须与坐标尺度相容”。图形见[[Exercise043_Problems.pdf#page=1|原题第 1 页]]。
> <!-- bilingual-en:start -->
> The three plots use different horizontal and vertical scales. Read the scales first, then compare each shaded region with triangles and rectangles whose areas can be calculated.
>
> **1.** The first shaded region is exactly a triangle with base $2$ and height $4$, so<br>
>    $$
>    \int f(x)\,dx=\frac12\cdot2\cdot4=\boxed{4}.
>    $$
>    Choose (b).
> **2.** In the second plot, the area under the curve is less than the enclosing triangle's area $1/2$, but clearly closer to $1/2$ than to $1/4$. The best estimate is $\boxed{1/2}$, so choose (c).<br>
> **3.** The third region can be approximated by a rectangle of area about $1/2$ plus an additional part of area about $1/4$. The best estimate is therefore $\boxed{3/4}$, so choose (b).<br>
>
> The task is not exact integration; it tests whether area estimates are consistent with the coordinate scale. See [[Exercise043_Problems.pdf#page=1|page 1 of the original problem]].
> <!-- bilingual-en:end -->

> [!question]- 三问自检
> 1. 为什么样本点可以任选？2. \(dx\) 对应什么？3. \(f\equiv c\) 时积分是多少？
>
> 答：可积要求选点差异在细分后消失；\(dx\) 来自 \(\Delta x_i\)；结果为 \(c(b-a)\)。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** Why may the sample point in each subinterval be chosen arbitrarily?<br>
> **2.** What does $dx$ correspond to?<br>
> **3.** What is the integral when $f\equiv c$?<br>
>
> Answer: Integrability means that differences caused by the sample-point choices vanish as the partition is refined; $dx$ arises from $\Delta x_i$; the result is $c(b-a)$.
> <!-- bilingual-en:end -->

**知识链：**有限求和 → 分割变细 → Riemann 和极限 → 定积分。
<!-- bilingual-en:start -->
**Knowledge chain:** finite sum → finer partitions → limit of Riemann sums → definite integral.
<!-- bilingual-en:end -->

## Session 44：Adding Areas of Rectangles

取 \(f(x)=x\)、区间 \([0,1]\)、右端点 \(i/n\)：
<!-- bilingual-en:start -->
Take \(f(x)=x\), interval \([0,1]\), right endpoint \(i/n\):
<!-- bilingual-en:end -->

$$
R_n=\sum_{i=1}^n\frac{i}{n}\frac1n
=\frac{n(n+1)}{2n^2}
=\frac12+\frac1{2n}\to\frac12.
$$

左端点和为 \(L_n=\frac12-\frac1{2n}\)，二者从两侧夹住真实面积。
<!-- bilingual-en:start -->
The left-endpoint sum is \(L_n=\frac12-\frac1{2n}\); together, the left- and right-endpoint sums squeeze the true area from below and above.
<!-- bilingual-en:end -->

同理，使用
<!-- bilingual-en:start -->
Similarly, using
<!-- bilingual-en:end -->

$$
\sum_{i=1}^ni^2=\frac{n(n+1)(2n+1)}6
$$

可得
<!-- bilingual-en:start -->
gives
<!-- bilingual-en:end -->

$$
\int_0^1x^2dx
=\lim_{n\to\infty}\frac1{n^3}\sum_{i=1}^ni^2
=\frac13.
$$

> [!warning] 易错点
> 每项必须乘 $\Delta x$。只把高度相加，分割越细总和反而越大。
> <!-- bilingual-en:start -->
> Each item must be multiplied by $\Delta x$.  When you add the heights only, the finer the split, the larger the sum.
> <!-- bilingual-en:end -->

### 本地材料

- [[Ses44a_Lecture_Notes.pdf|44a Rectangle Example]]
- [[Ses44b_Lecture_Notes.pdf|44b Summation Notation]]
- [[Exercise044_Problems.pdf|Exercise 44]] · [[Exercise044_Solutions.pdf|解答]]

> [!example]- Exercise 44：求和记号逐项展开
> 先确定指标的起点、终点和通项，再展开：
> $$
> \begin{aligned}
> \sum_{k=1}^{5}k^2&=1+4+9+16+25=\boxed{55},\\
> \sum_{k=1}^{3}(2k)^2&=4+16+36=\boxed{56},\\
> \sum_{n=1}^{4}(-1)^n n&=-1+2-3+4=\boxed{2},\\
> \sum_{k=0}^{5}2^k&=1+2+4+8+16+32=\boxed{63}.
> \end{aligned}
> $$
> 最后一式也可用等比数列公式 $(2^6-1)/(2-1)$。易错点是把 $k=0$ 项 $2^0=1$ 漏掉。
> <!-- bilingual-en:start -->
> Identify the starting index, ending index, and general term before expanding:
> $$
> \begin{aligned}
> \sum_{k=1}^{5}k^2&=1+4+9+16+25=\boxed{55},\\
> \sum_{k=1}^{3}(2k)^2&=4+16+36=\boxed{56},\\
> \sum_{n=1}^{4}(-1)^n n&=-1+2-3+4=\boxed{2},\\
> \sum_{k=0}^{5}2^k&=1+2+4+8+16+32=\boxed{63}.
> \end{aligned}
> $$
> The last result also follows from the geometric-sum formula $(2^6-1)/(2-1)$. A common error is omitting the $k=0$ term, $2^0=1$.
> <!-- bilingual-en:end -->

> [!question]- 三问自检
> 1. 递增函数的右端点和为何高估？2. 中点和是否改变极限？3. \(\int_0^2x\,dx\) 是多少？
>
> 答：右端点是每段最大值；不改变；结果为 \(2\)。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** Why does a right-endpoint sum overestimate an increasing function?<br>
> **2.** Does using midpoints change the limiting integral?<br>
> **3.** What is $\int_0^2x\,dx$?<br>
>
> Answer: The right endpoint gives the maximum value on each subinterval; no; the integral is $2$.
> <!-- bilingual-en:end -->

## Session 45：Some Easy Integrals

先用几何和对称性：
<!-- bilingual-en:start -->
Start with geometry and symmetry:
<!-- bilingual-en:end -->

$$
\int_a^bc\,dx=c(b-a),
$$

$$
\int_{-r}^{r}\sqrt{r^2-x^2}\,dx=\frac12\pi r^2,
$$

$$
f(-x)=-f(x)\Longrightarrow\int_{-a}^{a}f(x)dx=0,
$$

$$
f(-x)=f(x)\Longrightarrow\int_{-a}^{a}f(x)dx=2\int_0^af(x)dx.
$$

以下假设所用函数在相关闭区间上可积、$a<b$，半圆半径 $r>0$。由 Riemann 和逐项运算可证明[[定积分线性性|线性]]、[[定积分区间可加性|区间可加性]]和[[积分保序性|保序性]]：
<!-- bilingual-en:start -->
Assume the functions are integrable on the relevant closed intervals, $a<b$, and the semicircle radius is $r>0$. [[定积分线性性|Linearity]], [[定积分区间可加性|interval additivity]], and [[积分保序性|order preservation]] follow by applying the corresponding operations term by term to Riemann sums.
<!-- bilingual-en:end -->

$$
\int(cf+dg)=c\int f+d\int g,
$$

$$
\int_a^bf+\int_b^cf=\int_a^cf,
$$

$$
f\le g\Longrightarrow\int_a^bf\le\int_a^bg.
$$

对称区间上的规则见[[奇偶函数积分]]；正负抵消的大小界见[[积分绝对值界]]。
<!-- bilingual-en:start -->
See [[奇偶函数积分|integration of even and odd functions]] for symmetric intervals and [[积分绝对值界|the absolute-value bound for integrals]] for the effect of cancellation.
<!-- bilingual-en:end -->

### 本地材料

- [[Ses45a_Lecture_Notes.pdf|45a Easy Definite Integrals]]
- [[Ses45b_Lecture_Notes.pdf|45b Summary of Examples]]
- [[Exercise045_Problems.pdf|Exercise 45]] · [[Exercise045_Solutions.pdf|解答]]

> [!example]- Exercise 45：\(\int_{-1}^{2}|x|dx\)
> 绝对值在 \(x=0\) 改变公式，所以必须分段：
> $$
> \begin{aligned}
> \int_{-1}^{2}|x|dx
> &=\int_{-1}^{0}(-x)dx+\int_0^2x\,dx\\
> &=\frac12(1)(1)+\frac12(2)(2)
> =\boxed{\frac52}.
> \end{aligned}
> $$
> 两段都位于 \(x\) 轴上方，答案必须为正；直接使用 \(\int x\,dx\) 会把左侧面积错误地抵消。
> <!-- bilingual-en:start -->
> The formula for the absolute value changes at $x=0$, so split the integral:
> $$
> \begin{aligned}
> \int_{-1}^{2}|x|dx
> &=\int_{-1}^{0}(-x)dx+\int_0^2x\,dx\\
> &=\frac12(1)(1)+\frac12(2)(2)
> =\boxed{\frac52}.
> \end{aligned}
> $$
> Both regions lie above the $x$-axis, so the answer must be positive. Integrating $x$ without splitting would incorrectly cancel the area on the left.
> <!-- bilingual-en:end -->

> [!question]- 三问自检
> 1. \(\int_{-2}^2(x^3+4)dx\)？2. 为什么 \(\int|f|\ge|\int f|\)？3. 分点公式如何解释？
>
> 答：\(16\)；正负在 \(\int f\) 中会抵消；相邻区间累计量相加。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** What is \(\int_{-2}^2(x^3+4)dx\)?<br>
> **2.** Why is \(\int|f|\ge|\int f|\)?<br>
> **3.** How should interval additivity be interpreted?<br>
>
> Answer: \(16\); positive and negative contributions can cancel in \(\int f\); accumulated quantities over adjacent intervals add together.
> <!-- bilingual-en:end -->

## Session 46：Riemann Sums

看到
<!-- bilingual-en:start -->
For an expression of the form
<!-- bilingual-en:end -->

$$
\lim_{n\to\infty}\sum_{i=1}^n
f\!\left(a+i\frac{b-a}{n}\right)\frac{b-a}{n},
$$

先识别 \(\Delta x=(b-a)/n\) 和右端点，再写成 \(\int_a^bf(x)dx\)；这一极限转换以 $f$ 在 $[a,b]$ 可积为前提。见[[和式识别为积分]]。
<!-- bilingual-en:start -->
Recognize \(\Delta x=(b-a)/n\) and the right endpoint before writing \(\int_a^bf(x)dx\). This limit conversion assumes that $f$ is integrable on $[a,b]$; see [[和式识别为积分|recognising a sum as an integral]].
<!-- bilingual-en:end -->

若债务增长率为 \(r(t)\) 美元/年，则
<!-- bilingual-en:start -->
If the debt growth rate is \(r(t)\) USD/yr, then
<!-- bilingual-en:end -->

$$
\Delta D=\int_{t_0}^{t_1}r(t)dt.
$$

单位由“美元/年 × 年”恢复成美元。本地 `Ses46b` 和 `Ses46c` 正文相同，只解释一次但保留两个来源。
<!-- bilingual-en:start -->
The units reduce from “dollars per year × years” to dollars. The local files `Ses46b` and `Ses46c` contain the same text, so the content is explained once while both sources remain linked.
<!-- bilingual-en:end -->

### 本地材料

- [[Ses46a_Lecture_Notes.pdf|46a Riemann Sums]]
- [[Ses46b_Lecture_Notes.pdf|46b Cumulative Debt]]
- [[Ses46c_Lecture_Notes.pdf|46c Alternate File]]
- [[Exercise046_Problems.pdf|Exercise 46]] · [[Exercise046_Solutions.pdf|解答]]

> [!example]- Exercise 46：六个左端点矩形
> 估计 $\int_0^2(3x+2)dx$。六等分给出 $\Delta x=2/6=1/3$，左端点为
> $0,1/3,2/3,1,4/3,5/3$，相应高度为 $2,3,4,5,6,7$。因此
> $$
> L_6=\frac13(2+3+4+5+6+7)=\boxed{9}.
> $$
> 精确值为
> $$
> \left[\frac32x^2+2x\right]_0^2=10.
> $$
> 被积函数递增，所以每段左端点高度都是该段最小值，$L_6<10$ 的方向与图形一致。
> <!-- bilingual-en:start -->
> Estimate $\int_0^2(3x+2)dx$. Six equal subintervals give $\Delta x=2/6=1/3$ and left endpoints
> $0,1/3,2/3,1,4/3,5/3$, with corresponding heights $2,3,4,5,6,7$. Therefore,
> $$
> L_6=\frac13(2+3+4+5+6+7)=\boxed{9}.
> $$
> The exact value is
> $$
> \left[\frac32x^2+2x\right]_0^2=10.
> $$
> Since the integrand is increasing, the left-endpoint height is the minimum on each subinterval; the inequality $L_6<10$ therefore agrees with the graph.
> <!-- bilingual-en:end -->

> [!question]- 三问自检
> 1. 怎样从 \(3/n\) 读区间长度？2. 平移 \(a\) 在哪里？3. 单位如何检查？
>
> 答：\(n(3/n)=3\)；样本点写成 \(a+i\Delta x\)；函数单位乘自变量单位。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** How can the interval length be read from $3/n$?<br>
> **2.** Where does the translation by $a$ appear?<br>
> **3.** How should the units be checked?<br>
>
> Answer: $n(3/n)=3$; sample points take the form $a+i\Delta x$; multiply the units of the function by the units of the independent variable.
> <!-- bilingual-en:end -->

## Session 47：Introduction to the FTC

> [!important] [[Newton-Leibniz公式]]（MIT FTC I）
> 若 $f$ 在 $[a,b]$ 连续，$F$ 是其原函数，即 $F'=f$，则
> $$
> \boxed{\int_a^bf(x)dx=F(b)-F(a)}.
> $$
> <!-- bilingual-en:start -->
> If $f$ is continuous on $[a,b]$ and $F'=f$, the definite integral equals the endpoint difference $F(b)-F(a)$.
> <!-- bilingual-en:end -->

使用流程：找原函数 → 写 \([F]_a^b\) → 上限值减下限值 → 检查符号与单位。定积分不写 \(+C\)，因为常数在端点差中抵消。
<!-- bilingual-en:start -->
Use this procedure: find an antiderivative → write \([F]_a^b\) → subtract the lower-endpoint value from the upper-endpoint value → check the sign and units. Do not append \(+C\) to a definite integral, because constants cancel in the endpoint difference.
<!-- bilingual-en:end -->

[[Ses47b_Lecture_Notes.pdf|47b]] 用 $F(x)=x^3/3$ 得到 $\int_a^b x^2dx=(b^3-a^3)/3$，并用 $F(x)=-\cos x$ 得到 $\int_0^\pi\sin x\,dx=2$。这把前面逐项求和、再取极限的计算化为原函数的两次代入；两方向的证明留到 Session 52。
<!-- bilingual-en:start -->
[[Ses47b_Lecture_Notes.pdf|47b]] uses $F(x)=x^3/3$ to obtain $\int_a^b x^2dx=(b^3-a^3)/3$, and $F(x)=-\cos x$ to obtain $\int_0^\pi\sin x\,dx=2$. The earlier sum-and-limit calculation becomes two endpoint evaluations; both directions are proved in Session 52.
<!-- bilingual-en:end -->

![[Newton-Leibniz公式]]

### 本地材料

- [[Ses47a_Lecture_Notes.pdf|47a Fundamental Theorem]]
- [[Ses47b_Lecture_Notes.pdf|47b First FTC]]
- [[Exercise047_Problems.pdf|Exercise 47]] · [[Exercise047_Solutions.pdf|解答]]

> [!example]- Exercise 47：用原函数核对符号
> $$
> \begin{aligned}
> \text{(a)}\quad\int_0^2x^2dx
> &=\left[\frac{x^3}{3}\right]_0^2=\boxed{\frac83},\\
> \text{(b)}\quad\int_1^e\frac{dx}{x}
> &=[\ln|x|]_1^e=1-0=\boxed{1},\\
> \text{(c)}\quad\int_{-\pi/4}^{0}\sin x\,dx
> &=[-\cos x]_{-\pi/4}^{0}
> =-1+\frac1{\sqrt2}
> =\boxed{\frac{\sqrt2-2}{2}}.
> \end{aligned}
> $$
> 第三题约为 $-0.293$。区间内 $\sin x\le0$，所以负号不是计算错误，而是有向面积的必然结果。
> <!-- bilingual-en:start -->
> $$
> \begin{aligned}
> \text{(a)}\quad\int_0^2x^2dx
> &=\left[\frac{x^3}{3}\right]_0^2=\boxed{\frac83},\\
> \text{(b)}\quad\int_1^e\frac{dx}{x}
> &=[\ln|x|]_1^e=1-0=\boxed{1},\\
> \text{(c)}\quad\int_{-\pi/4}^{0}\sin x\,dx
> &=[-\cos x]_{-\pi/4}^{0}
> =-1+\frac1{\sqrt2}
> =\boxed{\frac{\sqrt2-2}{2}}.
> \end{aligned}
> $$
> The third answer is approximately $-0.293$. Since $\sin x\le0$ on the interval, the negative sign is not an error; it is required by signed area.
> <!-- bilingual-en:end -->

> [!question]- 三问自检
> 1. 为什么无 \(+C\)？2. \(\int_1^3(2x+1)dx\)？3. 原函数不唯一为何结果唯一？
>
> 答：常数抵消；\(10\)；任意两原函数只差常数。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** Why is there no $+C$?<br>
> **2.** Evaluate $\int_1^3(2x+1)dx$.<br>
> **3.** Antiderivatives are not unique, so why is the definite integral unique?<br>
>
> Answer: The constant cancels; the value is $10$; any two antiderivatives differ only by a constant.
> <!-- bilingual-en:end -->

## Session 48：The Fundamental Theorem

### 净变化、负被积函数与积分性质
<!-- bilingual-en:start -->
*Net change, negative integrands, and integral properties*
<!-- bilingual-en:end -->

若位置函数 $s$ 连续可导，$v=s'$，FTC I 给出位移 $s(b)-s(a)=\int_a^bv(t)dt$。若途中折返，正负位移抵消；总路程必须另算 $\int_a^b|v(t)|dt$。[[Ses48a_Lecture_Notes.pdf|48a]] 的行程直觉须连同 [[Ses48b_Lecture_Notes.pdf|48b]] 的有向速度解释一起读。
<!-- bilingual-en:start -->
For a continuously differentiable position $s$ with $v=s'$, FTC I gives displacement $s(b)-s(a)=\int_a^bv(t)dt$. Contributions in opposite directions cancel; total distance is $\int_a^b|v(t)|dt$. Read the trip interpretation in [[Ses48a_Lecture_Notes.pdf|48a]] together with the signed-velocity clarification in [[Ses48b_Lecture_Notes.pdf|48b]].
<!-- bilingual-en:end -->

例如 $\int_0^{2\pi}\sin x\,dx=0$，但两拱的几何面积是 $\int_0^{2\pi}|\sin x|dx=4$。负被积函数无需另造积分；它贡献负的累计量。线性、分段相加和保序性随后让我们组合或估计这些累计量。
<!-- bilingual-en:start -->
For example, $\int_0^{2\pi}\sin x\,dx=0$, while the geometric area of the two arches is $\int_0^{2\pi}|\sin x|dx=4$. Negative integrands contribute negative accumulation. Linearity, interval additivity, and order preservation then let us combine or estimate these quantities.
<!-- bilingual-en:end -->

![[净变化与总变化]]

### 本地材料

- [[Ses48a_Lecture_Notes.pdf|48a Interpretation]]
- [[Ses48b_Lecture_Notes.pdf|48b Negative Integrands]]
- [[Ses48c_Lecture_Notes.pdf|48c Integral Properties]]
- [[Exercise048_Problems.pdf|Exercise 48]] · [[Exercise048_Solutions.pdf|解答]]

> [!example]- Exercise 48：线性与对称性
> $$
> \begin{aligned}
> \int_0^\pi(\sin x+\cos x)dx
> &=\int_0^\pi\sin x\,dx+\int_0^\pi\cos x\,dx\\
> &=[-\cos x]_0^\pi+[\sin x]_0^\pi\\
> &=2+0=\boxed{2}.
> \end{aligned}
> $$
> 几何上，$\sin x$ 在区间上方贡献面积 $2$；$\cos x$ 在 $[0,\pi/2]$ 与 $[\pi/2,\pi]$ 的正负面积互相抵消。
> <!-- bilingual-en:start -->
> $$
> \begin{aligned}
> \int_0^\pi(\sin x+\cos x)dx
> &=\int_0^\pi\sin x\,dx+\int_0^\pi\cos x\,dx\\
> &=[-\cos x]_0^\pi+[\sin x]_0^\pi\\
> &=2+0=\boxed{2}.
> \end{aligned}
> $$
> Geometrically, $\sin x$ contributes area $2$ above the axis. The positive area of $\cos x$ on $[0,\pi/2]$ cancels its negative area on $[\pi/2,\pi]$.
> <!-- bilingual-en:end -->

> [!question]- 三问自检
> 1. 往返行程位移为何可为零？2. 怎样求总路程？3. 负被积函数是否使 FTC I 失效？
>
> 答：有向速度积分中正负贡献抵消；对速率 $|v|$ 积分；不失效，定积分本来就是有向累计量。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** Why can a round trip have zero displacement?<br>
> **2.** How is total distance computed?<br>
> **3.** Do negative integrands invalidate FTC I?<br>
>
> Answer: Positive and negative velocity contributions cancel; integrate speed $|v|$; no, definite integrals already represent signed accumulation.
> <!-- bilingual-en:end -->

**知识链：**速度的正负 → 位移的方向 → 净变化；取绝对值再累计 → 总变化。
<!-- bilingual-en:start -->
**Knowledge chain:** velocity sign → displacement direction → net change; take absolute values before accumulating → total change.
<!-- bilingual-en:end -->

## Session 49：Applications of FTC

本节用 FTC I 做估计与[[定积分换元]]。[[Ses49a_Lecture_Notes.pdf|49a]] 从 $e^x\ge1$（$x\ge0$）出发，在 $[0,b]$（$b\ge0$）积分得到 $e^b\ge1+b$，再积分一次得到 $e^b\ge1+b+b^2/2$。每次都是先用[[积分保序性]]，再用原函数端点差计算两边。
<!-- bilingual-en:start -->
This session applies FTC I to estimates and [[定积分换元|definite-integral substitution]]. [[Ses49a_Lecture_Notes.pdf|49a]] starts with $e^x\ge1$ for $x\ge0$, integrates over $[0,b]$ with $b\ge0$ to obtain $e^b\ge1+b$, then integrates again to obtain $e^b\ge1+b+b^2/2$. Each step uses [[积分保序性|order preservation]] followed by endpoint evaluation.
<!-- bilingual-en:end -->

[[Ses49b_Lecture_Notes.pdf|49b]] 的换元同时转换端点：令 $u=x^3+2$，则
<!-- bilingual-en:start -->
The substitution in [[Ses49b_Lecture_Notes.pdf|49b]] also transforms the endpoints: with $u=x^3+2$,
<!-- bilingual-en:end -->

$$
\int_1^2(x^3+2)^5x^2dx=\frac13\int_3^{10}u^5du=\frac{10^6-3^6}{18}.
$$

对完整复合导数型 $f(u(x))u'(x)$，只要 $f$ 在 $u$ 的像所在区间连续且 $u$ 连续可导，[[定积分换元]]可直接使用有向上下限，即使 $u'$ 变号也成立。只有换元后仍需用逆映射消去残余变量，而换元函数又不单调时，才要按 [[非单调换元需分支|单调分支]] 拆开。[[Ses48c_Lecture_Notes.pdf#page=2|48c 第 7 条]] 的变号即分段说法应限于需要逆分支的情形；[[Ses49c_Lecture_Notes.pdf|49c]] 专门讨论这一范围差异。
<!-- bilingual-en:start -->
For an exact composite derivative $f(u(x))u'(x)$, [[定积分换元|definite-integral substitution]] uses oriented bounds whenever $f$ is continuous on an interval containing the image of $u$ and $u$ is continuously differentiable, even when $u'$ changes sign. [[非单调换元需分支|Monotone branches]] are needed when an inverse map must remove residual variables and the substitution is not one-to-one. The sign-change restriction in [[Ses48c_Lecture_Notes.pdf#page=2|48c, property 7]] applies to that inverse-branch setting; [[Ses49c_Lecture_Notes.pdf|49c]] discusses this distinction.
<!-- bilingual-en:end -->

### 本地材料

- [[Ses49a_Lecture_Notes.pdf|49a Estimation]]
- [[Ses49b_Lecture_Notes.pdf|49b FTC Example]]
- [[Ses49c_Lecture_Notes.pdf|49c Substitution When \(u'\) Changes Sign]]
- [[Exercise049_Problems.pdf|Exercise 49]] · [[Exercise049_Solutions.pdf|解答]]

> [!example]- Exercise 49：换元时同步更换积分限
> (a) 令 $u=3x+4$，则 $du=3dx$，且 $x=0,4$ 对应 $u=4,16$：
> $$
> \int_0^4\sqrt{3x+4}\,dx
> =\frac13\int_4^{16}u^{1/2}du
> =\frac29[u^{3/2}]_4^{16}
> =\boxed{\frac{112}{9}}.
> $$
> (b) 令 $u=x^2+1$，$du=2x\,dx$，积分限 $2\to10$：
> $$
> \int_1^3\frac{x}{x^2+1}dx
> =\frac12\int_2^{10}\frac{du}{u}
> =\boxed{\frac12\ln5}.
> $$
> (c) 令 $u=\sin x$，$du=\cos x\,dx$，积分限 $0\to1$：
> $$
> \int_0^{\pi/2}\sin^5x\cos x\,dx
> =\int_0^1u^5du=\boxed{\frac16}.
> $$
> 一旦换成 $u$ 的积分限，最后就不再代回 $x$；两套端点不可混用。
> <!-- bilingual-en:start -->
> **(a)** Let $u=3x+4$, so $du=3dx$, and $x=0,4$ correspond to $u=4,16$:
> $$
> \int_0^4\sqrt{3x+4}\,dx
> =\frac13\int_4^{16}u^{1/2}du
> =\frac29[u^{3/2}]_4^{16}
> =\boxed{\frac{112}{9}}.
> $$
> **(b)** Let $u=x^2+1$, so $du=2x\,dx$ and the new limits are $2$ and $10$:
> $$
> \int_1^3\frac{x}{x^2+1}dx
> =\frac12\int_2^{10}\frac{du}{u}
> =\boxed{\frac12\ln5}.
> $$
> **(c)** Let $u=\sin x$, so $du=\cos x\,dx$ and the new limits are $0$ and $1$:
> $$
> \int_0^{\pi/2}\sin^5x\cos x\,dx
> =\int_0^1u^5du=\boxed{\frac16}.
> $$
> Once the limits have been changed to $u$-values, do not substitute back to $x$ at the end; the two sets of endpoints must not be mixed.
> <!-- bilingual-en:end -->

> [!question]- 三问自检
> 1. 换元后端点用哪一变量？2. $u'$ 变号是否一定分段？3. 如何从 $e^x\ge1$ 得到 $e^b\ge1+b$？
>
> 答：新变量；完整复合导数型无需，需逆映射时检查分支；在 $[0,b]$（$b\ge0$）积分并用 FTC I。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** Which variable determines the limits after substitution?<br>
> **2.** Must an integral be split whenever $u'$ changes sign?<br>
> **3.** How does $e^x\ge1$ give $e^b\ge1+b$?<br>
>
> Answer: The new variable; no for an exact composite derivative, but check branches when using an inverse; integrate on $[0,b]$ with $b\ge0$ and apply FTC I.
> <!-- bilingual-en:end -->

## Session 50：FTC and MVT

### [[积分中值定理]]及证明
<!-- bilingual-en:start -->
*The [[积分中值定理|mean value theorem for integrals]] and its proof*
<!-- bilingual-en:end -->

沿本节官方顺序，先给定连续可导的 $F$，令 $f=F'$，且 $a<b$。FTC I 给出 $F(b)-F(a)=\int_a^bf(x)dx$。对 $F$ 用普通平均值定理，存在 $c\in(a,b)$：
<!-- bilingual-en:start -->
Following the official sequence, start with a continuously differentiable $F$, set $f=F'$, and assume $a<b$. FTC I gives $F(b)-F(a)=\int_a^bf(x)dx$. Applying the ordinary mean value theorem to $F$, there is some $c\in(a,b)$ such that
<!-- bilingual-en:end -->

$$
\frac{F(b)-F(a)}{b-a}=F'(c)=f(c).
$$

所以
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

$$
\boxed{\int_a^bf(x)dx=f(c)(b-a)}.
$$

因此 $f(c)$ 是[[积分平均值]]。若只给定连续 $f$，Session 51 的 FTC II 会构造原函数，使上述证明适用于任意连续被积函数；也可直接由积分保序性和介值定理证明。
<!-- bilingual-en:start -->
Thus $f(c)$ is the [[积分平均值|integral average]]. For an arbitrary continuous $f$, FTC II in Session 51 constructs an antiderivative and extends this proof to every continuous integrand; an alternative proof uses order preservation and the intermediate value theorem.
<!-- bilingual-en:end -->

若 \(m\le f\le M\)，积分保持不等式给出
<!-- bilingual-en:start -->
If \(m\le f\le M\), monotonicity of the integral gives
<!-- bilingual-en:end -->

$$
m(b-a)\le\int_a^bf\le M(b-a).
$$

![[积分上下界估计]]

### 本地材料

- [[Ses50a_Lecture_Notes.pdf|50a FTC Review]]
- [[Ses50b_Lecture_Notes.pdf|50b FTC and MVT]]
- [[Ses50c_Lecture_Notes.pdf|50c Estimation]]
- [[Exercise050_Problems.pdf|Exercise 50]] · [[Exercise050_Solutions.pdf|解答]]

> [!example]- Exercise 50：估计 \(\ln5\)
> 在 \(1\le x\le5\) 上，
> $$
> \frac15\le\frac1x\le1.
> $$
> 在长度为 \(4\) 的区间上积分，得到
> $$
> \boxed{\frac45<\int_1^5\frac{dx}{x}<4}.
> $$
> 又因 \(\int_1^5dx/x=\ln5\approx1.609\)，确实落在两界之间。估计有效但较宽：下界提供数量级，常数上界 \(4\) 则很松。若需要更精确，可把区间细分后分别取 \(1/x\) 的上下界。
> <!-- bilingual-en:start -->
> On $1\le x\le5$,
> $$
> \frac15\le\frac1x\le1.
> $$
> Integrating over an interval of length $4$ gives
> $$
> \boxed{\frac45<\int_1^5\frac{dx}{x}<4}.
> $$
> Since $\int_1^5dx/x=\ln5\approx1.609$, it lies between the two bounds. The estimate is valid but broad: the lower bound gives the right order of magnitude, whereas the constant upper bound $4$ is loose. For a sharper estimate, subdivide the interval and bound $1/x$ separately on each piece.
> <!-- bilingual-en:end -->

> [!question]- 三问自检
> 1. 连续性在哪里使用？2. \(f(c)\) 是哪种平均？3. 不求原函数能否估计积分？
>
> 答：保证 MVT/FTC 可用且平均值被取得；连续平均值；能，用上下界。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** Where is continuity used?<br>
> **2.** What kind of average is $f(c)$?<br>
> **3.** Can an integral be estimated without finding an antiderivative?<br>
>
> Answer: Continuity makes the MVT/FTC applicable and ensures the average value is attained; $f(c)$ is the continuous average value; yes, by using upper and lower bounds.
> <!-- bilingual-en:end -->

## Problem Set 6

### 官方指定范围与原题

- 3B：2a、2b、3b、4a、5
- 3C：1、2a、3a、5a
- 3E：6b、6c
- 4J：1（只列式）、2
- [[PSet03_Problems_2.pdf|Integration Problems, corrected]]
- [[PSet03_Solutions_2.pdf|Integration Solutions, corrected]]
- [[PSet04_Problems.pdf|Applications Problems]]
- [[PSet04_Solutions.pdf|Applications Solutions]]

> [!example]- 完整官方题册与逐题答案
> ![[PSet03_Problems_2.pdf#height=650]]
>
> ![[PSet03_Solutions_2.pdf#height=650]]
>
> ![[PSet04_Problems.pdf#height=650]]
>
> ![[PSet04_Solutions.pdf#height=650]]

> [!example]- 3B 2a、2b：把有限和写成 \(\Sigma\)
> 2a 的各项为 $3,-5,7,-9,11,-13$。第 $n$ 项的绝对值是 $2n+1$，符号由 $(-1)^{n+1}$ 控制，因此
> $$
> \boxed{\sum_{n=1}^{6}(-1)^{n+1}(2n+1)}.
> $$
> 2b 是倒数平方和，首项 $1$、末项 $1/n^2$，故
> $$
> \boxed{\sum_{k=1}^{n}\frac1{k^2}}.
> $$
> 检查方法：分别代入指标的第一个和最后一个值，必须还原原和式。
> <!-- bilingual-en:start -->
> The terms in 2a are $3,-5,7,-9,11,-13$. The magnitude of the $n$th term is $2n+1$, and $(-1)^{n+1}$ controls the sign, so the sum is
> $$
> \boxed{\sum_{n=1}^{6}(-1)^{n+1}(2n+1)}.
> $$
> Part 2b is a sum of reciprocal squares, beginning with $1$ and ending with $1/n^2$:
> $$
> \boxed{\sum_{k=1}^{n}\frac1{k^2}}.
> $$
> To check an indexed expression, substitute the first and last index values and confirm that they reproduce the original sum.
> <!-- bilingual-en:end -->

> [!example]- 3B 3b、4a：上下和及其误差
> 对 $f(x)=x^2$ 在 $[-1,3]$ 作四等分，$\Delta x=1$，节点为 $-1,0,1,2,3$。逐段比较可得
> $$
> L_4=1+0+1+4=\boxed{6},
> \qquad
> R_4=0+1+4+9=\boxed{14}.
> $$
> 因 $x^2$ 在 $[-1,0]$ 递减、在 $[0,3]$ 递增，各段取最大、最小值后
> $$
> \boxed{U_4=15},\qquad \boxed{D_4=5}.
> $$
> 对 4a，$x^2$ 在 $[0,b]$ 单调递增（$b>0$）。每一小段的上、下矩形高度分别是右、左端点值，故望远镜相消：
> $$
> U_n-L_n
> =\frac bn\sum_{i=1}^n
> \left[\left(\frac{ib}{n}\right)^2-
> \left(\frac{(i-1)b}{n}\right)^2\right]
> =\frac bn(b^2-0)=\frac{b^3}{n}\to0.
> $$
> 所以上、下和夹到同一个积分值。
> <!-- bilingual-en:start -->
> Partition $[-1,3]$ into four equal pieces for $f(x)=x^2$. Then $\Delta x=1$ and the nodes are $-1,0,1,2,3$. Direct comparison on each subinterval gives
> $$
> L_4=1+0+1+4=\boxed{6},
> \qquad
> R_4=0+1+4+9=\boxed{14}.
> $$
> Since $x^2$ decreases on $[-1,0]$ and increases on $[0,3]$, choosing the maximum and minimum on each subinterval yields
> $$
> \boxed{U_4=15},\qquad \boxed{D_4=5}.
> $$
> For 4a, $x^2$ is increasing on $[0,b]$ for $b>0$. The upper and lower rectangle heights are therefore the right- and left-endpoint values, and the sum telescopes:
> $$
> U_n-L_n
> =\frac bn\sum_{i=1}^n
> \left[\left(\frac{ib}{n}\right)^2-
> \left(\frac{(i-1)b}{n}\right)^2\right]
> =\frac bn(b^2-0)=\frac{b^3}{n}\to0.
> $$
> Thus the upper and lower sums squeeze to the same integral value.
> <!-- bilingual-en:end -->

> [!example]- 3B 5：从极限识别 Riemann 和
> 题中
> $$
> \frac1n\sum_{i=1}^{n}\sin\frac{ib}{n}
> =\frac1b\sum_{i=1}^{n}\sin\frac{ib}{n}\frac bn.
> $$
> 若 $b>0$，右边是 $[0,b]$ 上 $\sin x$ 的右端点和再乘 $1/b$；若 $b<0$，按从 $0$ 到 $b$ 的方向解释为有向和式。因此（$b\ne0$）
> $$
> \lim_{n\to\infty}\frac1n\sum_{i=1}^{n}\sin\frac{ib}{n}
> =\frac1b\int_0^b\sin x\,dx
> =\boxed{\frac{1-\cos b}{b}}.
> $$
> $b=0$ 时原和每项均为零；右侧的连续延拓也趋于 $0$。
> <!-- bilingual-en:start -->
> The expression is
> $$
> \frac1n\sum_{i=1}^{n}\sin\frac{ib}{n}
> =\frac1b\sum_{i=1}^{n}\sin\frac{ib}{n}\frac bn.
> $$
> For $b>0$, the sum on the right is the right-endpoint sum for $\sin x$ on $[0,b]$, multiplied by $1/b$. For $b<0$, interpret it as an oriented sum from $0$ to $b$. Therefore, for $b\ne0$,
> $$
> \lim_{n\to\infty}\frac1n\sum_{i=1}^{n}\sin\frac{ib}{n}
> =\frac1b\int_0^b\sin x\,dx
> =\boxed{\frac{1-\cos b}{b}}.
> $$
> When $b=0$, every term in the original sum is zero, and the continuous extension of the expression on the right also tends to $0$.
> <!-- bilingual-en:end -->

> [!example]- 3C 1、2a、3a、5a：FTC 与换元
> 1. 令 $u=x-2$，端点 $3,6$ 变为 $1,4$：
>    $$
>    \int_3^6\frac{dx}{\sqrt{x-2}}
>    =\int_1^4u^{-1/2}du
>    =[2\sqrt u]_1^4=\boxed{2}.
>    $$
> 2a. 令 $u=3x+5$，$du=3dx$，端点 $5,11$：
>    $$
>    \int_0^2\sqrt{3x+5}\,dx
>    =\frac13\int_5^{11}u^{1/2}du
>    =\boxed{\frac29\left(11^{3/2}-5^{3/2}\right)}.
>    $$
> 3a. 令 $u=x^2+1$，$du=2x\,dx$，端点 $2,5$：
>    $$
>    \int_1^2\frac{x}{x^2+1}dx
>    =\frac12\int_2^5\frac{du}{u}
>    =\boxed{\frac12\ln\frac52}.
>    $$
> 5a. 一拱 $\sin x$ 位于 $[0,\pi]$ 且非负，所以几何面积就是
>    $$
>    \int_0^\pi\sin x\,dx=[-\cos x]_0^\pi=\boxed{2}.
>    $$
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** Let $u=x-2$; the endpoints $3,6$ become $1,4$:<br>
> $$
> \int_3^6\frac{dx}{\sqrt{x-2}}
> =\int_1^4u^{-1/2}du
> =[2\sqrt u]_1^4=\boxed{2}.
> $$
> **2a.** Let $u=3x+5$, so $du=3dx$ and the endpoints become $5,11$:
> $$
> \int_0^2\sqrt{3x+5}\,dx
> =\frac13\int_5^{11}u^{1/2}du
> =\boxed{\frac29\left(11^{3/2}-5^{3/2}\right)}.
> $$
> **3a.** Let $u=x^2+1$, so $du=2x\,dx$ and the endpoints become $2,5$:
> $$
> \int_1^2\frac{x}{x^2+1}dx
> =\frac12\int_2^5\frac{du}{u}
> =\boxed{\frac12\ln\frac52}.
> $$
> **5a.** One arch of $\sin x$ lies above the axis on $[0,\pi]$, so its geometric area is
> $$
> \int_0^\pi\sin x\,dx=[-\cos x]_0^\pi=\boxed{2}.
> $$
> <!-- bilingual-en:end -->

> [!example]- 3E 6b、6c：不用求原函数的严格比较
> 6b. 在 $[0,\pi]$ 上 $0\le\sin x\le1$，所以
> $$
> \sin^2x\le\sin x,
> $$
> 且除端点和 $x=\pi/2$ 外严格小于。积分保持不等式，故
> $$
> \boxed{\int_0^\pi\sin^2x\,dx<\int_0^\pi\sin x\,dx=2}.
> $$
> 6c. 对 $x\in[10,20]$，$\sqrt{x^2+1}>x$，因此
> $$
> \boxed{\int_{10}^{20}\sqrt{x^2+1}\,dx>
> \int_{10}^{20}x\,dx
> =\frac{20^2-10^2}{2}=150}.
> $$
> <!-- bilingual-en:start -->
> **6b.** On $[0,\pi]$, $0\le\sin x\le1$, so
> $$
> \sin^2x\le\sin x,
> $$
> with strict inequality except at the endpoints and at $x=\pi/2$. Integration preserves the inequality, giving
> $$
> \boxed{\int_0^\pi\sin^2x\,dx<\int_0^\pi\sin x\,dx=2}.
> $$
> **6c.** For $x\in[10,20]$, $\sqrt{x^2+1}>x$, so
> $$
> \boxed{\int_{10}^{20}\sqrt{x^2+1}\,dx>
> \int_{10}^{20}x\,dx
> =\frac{20^2-10^2}{2}=150}.
> $$
> <!-- bilingual-en:end -->

> [!example]- 4J 1、2：从“密度 × 小量”建立积分
> 1. 圆柱形孔直径 $1$，截面积为 $\pi(1/2)^2=\pi/4$。以 $y$ 表示水被提升的距离，厚度 $dy$ 的水层体积为 $(\pi/4)dy$，若单位体积提升单位距离需能量 $k$，则题目要求的“只列式”为
>    $$
>    \boxed{E=\frac{\pi k}{4}\int_0^{100}y\,dy}.
>    $$
>    因素 $y$ 不能漏，它表示不同深度的水提升距离不同；见[[功的积分]]。
> 2. 初始放射性物质量为 $x_0$，时刻 $t$ 剩余 $x_0e^{-kt}$（$k>0$），每单位物质的计数率为 $r$。瞬时计数率是 $rx_0e^{-kt}$，故一小时（$0\le t\le60$，$t$ 以分钟计）总计数为
>    $$
>    R=\int_0^{60}rx_0e^{-kt}dt
>    =\boxed{\frac{rx_0}{k}\left(1-e^{-60k}\right)}.
>    $$
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** A cylindrical shaft has diameter $1$, so its cross-sectional area is $\pi(1/2)^2=\pi/4$. Let $y$ be the distance through which a layer of water is lifted. A layer of thickness $dy$ has volume $(\pi/4)dy$. If lifting one unit of volume through one unit of distance requires energy $k$, the requested setup is<br>
> $$
> \boxed{E=\frac{\pi k}{4}\int_0^{100}y\,dy}.
> $$
> The factor $y$ is essential because water at different depths travels different distances; see [[功的积分|work as an integral]].
> **2.** If the initial amount of radioactive material is $x_0$, the amount remaining at time $t$ is $x_0e^{-kt}$ with $k>0$. If the count rate per unit amount is $r$, the instantaneous count rate is $rx_0e^{-kt}$. The total count over one hour, $0\le t\le60$ with $t$ in minutes, is<br>
> $$
> R=\int_0^{60}rx_0e^{-kt}dt
> =\boxed{\frac{rx_0}{k}\left(1-e^{-60k}\right)}.
> $$
> <!-- bilingual-en:end -->

---

## Part B：Second Fundamental Theorem, Areas and Volumes

## Session 51：The Second Fundamental Theorem

定义[[累积函数]]
<!-- bilingual-en:start -->
Define the [[累积函数|accumulation function]]
<!-- bilingual-en:end -->

$$
F(x)=\int_a^xf(t)dt.
$$

上限从 \(x\) 移到 \(x+h\) 时，
<!-- bilingual-en:start -->
When the upper limit is moved from \(x\) to \(x+h\),
<!-- bilingual-en:end -->

$$
F(x+h)-F(x)=\int_x^{x+h}f(t)dt.
$$

连续性使小条面积近似 \(f(x)h\)，因此差商近似 \(f(x)\)。本课第二基本定理是：
<!-- bilingual-en:start -->
Continuity makes the area of the narrow strip approximately $f(x)h$, so the difference quotient is approximately $f(x)$. This is the course's second Fundamental Theorem:
<!-- bilingual-en:end -->

> [!important] [[积分求导定理]]（MIT FTC II）
> 若 $f$ 在包含 $a$ 与 $x$ 的整个闭区间上可积，且在内点 $x$ 连续，则 $F(x)=\int_a^xf(t)dt$ 满足
> $$
> \boxed{F'(x)=f(x)}.
> $$
> 本课采用更方便的充分条件：$f$ 在所讨论的整个区间连续；端点只取相应单侧导数。
> <!-- bilingual-en:start -->
> If $f$ is integrable on the entire closed interval containing $a$ and $x$, and is continuous at the interior point $x$, then $F(x)=\int_a^xf(t)dt$ satisfies $F'(x)=f(x)$. The course uses the simpler sufficient condition that $f$ is continuous throughout the interval. At endpoints, use the appropriate one-sided derivative.
> <!-- bilingual-en:end -->

这里的 $t$ 是哑变量，$x$ 是函数输入；$F(a)=0$。因此积分为每个连续 $f$ 构造一个满足 $F'=f$ 的[[原函数]]，无需先知道它的初等表达式。
<!-- bilingual-en:start -->
Here $t$ is a dummy variable and $x$ is the function input; $F(a)=0$. The integral therefore constructs an [[原函数|antiderivative]] of every continuous $f$ without requiring an elementary expression in advance.
<!-- bilingual-en:end -->

![[98_attachment/MIT18.01SC/unit03-ftc-accumulator.png|819]]

![[积分求导定理]]

### 变限积分与链式法则
<!-- bilingual-en:start -->
*Variable limits and the chain rule*
<!-- bilingual-en:end -->

若 $f$ 在相关范围连续、$g$ 可导，令 $G(x)=\int_a^{g(x)}f(t)dt$、$H(u)=\int_a^uf(t)dt$，则 $G=H\circ g$。FTC II 与[[链式法则]]给出
<!-- bilingual-en:start -->
If $f$ is continuous on the relevant range and $g$ is differentiable, set $G(x)=\int_a^{g(x)}f(t)dt$ and $H(u)=\int_a^uf(t)dt$. Then $G=H\circ g$, so FTC II and the [[链式法则|chain rule]] give
<!-- bilingual-en:end -->

$$
\boxed{G'(x)=f(g(x))g'(x)}.
$$

上下限都可导时：
<!-- bilingual-en:start -->
When both limits are differentiable:
<!-- bilingual-en:end -->

$$
\frac d{dx}\int_{u(x)}^{v(x)}f(t)dt
=f(v(x))v'(x)-f(u(x))u'(x).
$$

例：无需先求 $\sin(t^3)$ 的原函数，就有
<!-- bilingual-en:start -->
For example, without first finding an antiderivative of $\sin(t^3)$,
<!-- bilingual-en:end -->

$$
\frac d{dx}\int_0^{x^2}\sin(t^3)dt
=2x\sin(x^6).
$$

![[变限积分求导]]

### 本地材料

- [[Ses51a_Lecture_Notes.pdf|51a Second FTC]]
- [[Ses51b_Lecture_Notes.pdf|51b Using Second FTC]]

> [!question]- 三问自检
> 1. 为什么积分变量写 $t$？2. $f<0$ 时累积函数怎样？3. $F(a)$？
>
> 答：$t$ 是哑变量；$F$ 递减；$F(a)=0$。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** Why is the integration variable written as $t$?<br>
> **2.** What happens to $F$ when $f<0$?<br>
> **3.** What is $F(a)$?<br>
>
> Answer: $t$ is a dummy variable; $F$ decreases; $F(a)=0$.
> <!-- bilingual-en:end -->

> [!question]- 变限求导自检
> 假设 $f$ 连续：1. $\frac d{dx}\int_x^0f(t)dt$？2. $\frac d{dx}\int_x^{x^2}e^{t^2}dt$？3. 内层导数为何不能漏？
>
> 答：$-f(x)$；$2xe^{x^4}-e^{x^2}$；因为上限移动速度不一定为 $1$。
> <!-- bilingual-en:start -->
> Assume $f$ is continuous.
>
> &nbsp;
> **1.** What is $\frac d{dx}\int_x^0f(t)dt$?<br>
> **2.** What is $\frac d{dx}\int_x^{x^2}e^{t^2}dt$?<br>
> **3.** Why must the derivative of the inner limit be included?<br>
>
> Answer: $-f(x)$; $2xe^{x^4}-e^{x^2}$; because an integration limit need not move at unit speed.
> <!-- bilingual-en:end -->

## Session 52：微积分基本定理证明
<!-- bilingual-en:start -->
*Session 52: Proofs of the Fundamental Theorem of Calculus*
<!-- bilingual-en:end -->

### FTC II 的逐步证明：累积求导
<!-- bilingual-en:start -->
*Step-by-step proof of FTC II: differentiating accumulation*
<!-- bilingual-en:end -->

沿用 Session 51 的 $F(x)=\int_a^xf(t)dt$ 和连续性条件，目标：
<!-- bilingual-en:start -->
Use $F(x)=\int_a^xf(t)dt$ and the continuity assumptions from Session 51. The goal is
<!-- bilingual-en:end -->

$$
\lim_{h\to0}\frac{F(x+h)-F(x)}h=f(x).
$$

由积分可加性，
<!-- bilingual-en:start -->
By integral additivity,
<!-- bilingual-en:end -->

$$
\frac{F(x+h)-F(x)}h
=\frac1h\int_x^{x+h}f(t)dt.
$$

减去目标值并把常数写入积分：
<!-- bilingual-en:start -->
Subtract the target value and place the constant inside the integral:
<!-- bilingual-en:end -->

$$
\frac{F(x+h)-F(x)}h-f(x)
=\frac1h\int_x^{x+h}[f(t)-f(x)]dt.
$$

给定 \(\varepsilon>0\)。连续性保证存在 \(\delta>0\)，当 \(|t-x|<\delta\) 时，
\(|f(t)-f(x)|<\varepsilon\)。若 \(0<|h|<\delta\)，则
<!-- bilingual-en:start -->
Given \(\varepsilon>0\), continuity provides a \(\delta>0\) such that \(|f(t)-f(x)|<\varepsilon\) whenever \(|t-x|<\delta\). If \(0<|h|<\delta\), then
<!-- bilingual-en:end -->

$$
\left|
\frac{F(x+h)-F(x)}h-f(x)
\right|
\le\frac1{|h|}\varepsilon|h|=\varepsilon.
$$

故差商极限为 \(f(x)\)。证明没有假设 \(f\ge0\)，因此负被积函数同样成立。
<!-- bilingual-en:start -->
Thus the difference quotient tends to \(f(x)\). The proof never assumes \(f\ge0\), so it applies equally when the integrand is negative.
<!-- bilingual-en:end -->

**知识链：**连续性控制小区间内高度变化，区间长度正好约掉差商分母。这个估计补全了 [[Ses52a_Lecture_Notes.pdf|52a]] 的窄条面积论证。
<!-- bilingual-en:start -->
**Knowledge chain:** continuity controls variation on a short interval, while its length cancels the difference-quotient denominator. This estimate makes the thin-strip argument in [[Ses52a_Lecture_Notes.pdf|52a]] precise.
<!-- bilingual-en:end -->

> [!question]- 累积求导自检
> 1. FTC II 证明中连续性控制哪一个量？2. 证明为何也覆盖 $h<0$？3. $\frac d{dx}\int_2^xf(t)dt$ 是什么？
>
> 答：控制 $|f(t)-f(x)|$；估计使用 $|h|$，反向积分的符号也包含在等式中；结果为 $f(x)$。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** Which quantity does continuity control in the proof of FTC II?<br>
> **2.** Why does the proof also cover $h<0$?<br>
> **3.** What is $\frac d{dx}\int_2^xf(t)dt$?<br>
>
> Answer: $|f(t)-f(x)|$; the bound uses $|h|$ and the oriented integral accounts for reversal; the result is $f(x)$.
> <!-- bilingual-en:end -->

### FTC I 的证明：原函数只差常数
<!-- bilingual-en:start -->
*Proof of FTC I: antiderivatives differ by a constant*
<!-- bilingual-en:end -->

令
<!-- bilingual-en:start -->
Let
<!-- bilingual-en:end -->

$$
A(x)=\int_a^xf(t)dt.
$$

FTC II 给出 \(A'=f\)。若 \(F'=f\)，则
<!-- bilingual-en:start -->
FTC II gives \(A'=f\). If \(F'=f\), then
<!-- bilingual-en:end -->

$$
(A-F)'=0.
$$

由平均值定理，导数恒零的函数是常数，所以 \(A-F=C\)。代入 \(x=a\)：
<!-- bilingual-en:start -->
By the mean value theorem, a function with a constant zero derivative is a constant, so \(A-F=C\).  Substitute \(x=a\):
<!-- bilingual-en:end -->

$$
0-F(a)=C.
$$

于是
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

$$
A(x)=F(x)-F(a).
$$

取 \(x=b\) 得
<!-- bilingual-en:start -->
Setting \(x=b\) gives
<!-- bilingual-en:end -->

$$
\int_a^bf=F(b)-F(a).
$$

### 本地材料

- [[Ses52a_Lecture_Notes.pdf|52a Proof of Second FTC]]
- [[Ses52b_Lecture_Notes.pdf|52b Proof of First FTC]]

**知识链：**[[积分求导定理|FTC II]] 造出一个原函数；MVT 说明[[原函数相差常数|所有原函数只差常数]]；因此端点差就是累计量，即[[Newton-Leibniz公式|FTC I]]。
<!-- bilingual-en:start -->
**Conceptual chain:** [[积分求导定理|FTC II]] constructs an antiderivative; MVT shows that [[原函数相差常数|antiderivatives differ only by a constant]]; therefore, their endpoint difference gives the accumulated quantity, which is [[Newton-Leibniz公式|FTC I]].
<!-- bilingual-en:end -->

> [!question]- 三问自检
> 1. 证明中为何比较 \(A-F\)？2. 导数恒零为何推出常数？3. 常数怎样确定？
>
> 答：二者导数都等于 \(f\)；由 MVT；代入 \(x=a\) 使用 \(A(a)=0\)。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** Why compare \(A-F\) in the proof?<br>
> **2.** Why does a zero derivative imply a constant function?<br>
> **3.** How is that constant determined?<br>
>
> Answer: Both derivatives equal \(f\); the mean value theorem makes a function with zero derivative constant; substitute \(x=a\) and use \(A(a)=0\).
> <!-- bilingual-en:end -->

## Session 53：New Functions From Old

即使 \(e^{-x^2}\) 没有初等原函数，仍可定义
<!-- bilingual-en:start -->
Even though \(e^{-x^2}\) has no elementary antiderivative, we can still define
<!-- bilingual-en:end -->

$$
E(x)=\int_0^xe^{-t^2}dt,
\qquad E'(x)=e^{-x^2}.
$$

![[无初等原函数的积分]]

对 \(x>0\) 定义
<!-- bilingual-en:start -->
Define for \(x>0\)
<!-- bilingual-en:end -->

$$
L(x)=\int_1^x\frac{dt}{t}.
$$

则 \(L'(x)=1/x\)、\(L(1)=0\)。下一节证明 \(L\) 具有对数的乘法性质。
<!-- bilingual-en:start -->
Then \(L'(x)=1/x\) and \(L(1)=0\). The next section proves the logarithmic product rule for \(L\).
<!-- bilingual-en:end -->

### 本地材料

- [[Ses53a_Lecture_Notes.pdf|53a Antiderivative of \(1/x\)]]
- [[Ses53b_Lecture_Notes.pdf|53b Bell Curve]]

> [!question]- 三问自检
> 1. \(E\) 奇偶性？2. \(L\) 定义域为何正数？3. 无初等原函数是否等于积分不存在？
>
> 答：\(E\) 为奇函数；\(1/t\) 在 0 奇异；不等于，定积分仍能定义新函数。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** Is \(E\) even or odd?<br>
> **2.** Why is the domain of \(L\) positive?<br>
> **3.** Does the absence of an elementary antiderivative mean that the integral does not exist?<br>
>
> Answer: \(E\) is odd; \(1/t\) is singular at zero; no—a definite integral can still define a new function.
> <!-- bilingual-en:end -->

## Session 54：FTC and \(\ln x\)

固定 \(y>0\)，令
<!-- bilingual-en:start -->
Fix \(y>0\) and define
<!-- bilingual-en:end -->

$$
G(x)=L(xy)-L(x).
$$

求导：
<!-- bilingual-en:start -->
Differentiate:
<!-- bilingual-en:end -->

$$
G'(x)=\frac{y}{xy}-\frac1x=0.
$$

所以 \(G\) 为常数。代 \(x=1\) 得 \(G(1)=L(y)\)，从而
<!-- bilingual-en:start -->
Therefore \(G\) is constant. Substituting \(x=1\) gives \(G(1)=L(y)\), hence
<!-- bilingual-en:end -->

$$
\boxed{L(xy)=L(x)+L(y)}.
$$

又因 $L'(x)=1/x>0$，$L$ 严格递增。乘法性质给出 $L(2^n)=nL(2)\to\infty$ 与 $L(2^{-n})=-nL(2)\to-\infty$，其中 $L(2)>0$。结合连续性和介值定理，$L$ 的值域才可确定为整个 $\mathbb R$，故反函数定义在整个实轴上。
<!-- bilingual-en:start -->
Since $L'(x)=1/x>0$, $L$ is strictly increasing. The product rule gives $L(2^n)=nL(2)\to\infty$ and $L(2^{-n})=-nL(2)\to-\infty$, with $L(2)>0$. Continuity and the intermediate value theorem then establish that its range is all of $\mathbb R$, so its inverse is defined on the whole real line.
<!-- bilingual-en:end -->

这也与已建立的[[自然对数]]对象一致：$(L-\ln)'=1/x-1/x=0$，而 $L(1)=\ln1=0$，由[[原函数相差常数]]得 $L(x)=\ln x$。所以反函数正是[[自然指数函数|$e^x$]]，这里证明的是同一个对数的[[自然对数积分表示|积分表示]]。
<!-- bilingual-en:start -->
This agrees with the established [[自然对数|natural logarithm]]: $(L-\ln)'=0$ and $L(1)=\ln1=0$, so [[原函数相差常数|uniqueness of antiderivatives up to a constant]] gives $L(x)=\ln x$. Its inverse is therefore the [[自然指数函数|natural exponential function]], and the result establishes the [[自然对数积分表示|integral representation of that same logarithm]].
<!-- bilingual-en:end -->

![[自然对数积分表示]]

### 本地材料

- [[Ses54a_Lecture_Notes.pdf|54a Integral Definition of Log]]
- [[Ses54b_Lecture_Notes.pdf|54b Log of a Product]]
- [[Exercise054_Problems.pdf|Exercise 54]] · [[Exercise054_Solutions.pdf|解答]]

> [!example]- Exercise 54：面积形式的幂法则
> 设 $x>1$、$a>0$。题目要证明
> $$
> a\int_{1/x}^{1}\frac{dt}{t}
> =\int_{(1/x)^a}^{1}\frac{dt}{t}.
> $$
> 使用积分定义的对数，左边为
> $$
> a[\ln t]_{1/x}^{1}=a(0+\ln x)=a\ln x,
> $$
> 右边为
> $$
> [\ln t]_{x^{-a}}^{1}=0-\ln(x^{-a})=a\ln x.
> $$
> 因而两边相等。几何上，从 $x^{-a}$ 到 $1$ 的 $1/t$ 曲线下面积是从 $x^{-1}$ 到 $1$ 面积的 $a$ 倍。固定 $x>1$ 并令 $a\to\infty$，下限 $x^{-a}\to0^+$，右侧面积 $a\ln x\to\infty$；这也说明
> $$
> \int_{0}^{1}\frac{dt}{t}
> $$
> 这一[[反常积分]]在 $0$ 附近发散。
> <!-- bilingual-en:start -->
> Let $x>1$ and $a>0$. The claim is
> $$
> a\int_{1/x}^{1}\frac{dt}{t}
> =\int_{(1/x)^a}^{1}\frac{dt}{t}.
> $$
> Using the integral definition of the logarithm, the left-hand side is
> $$
> a[\ln t]_{1/x}^{1}=a(0+\ln x)=a\ln x,
> $$
> while the right-hand side is
> $$
> [\ln t]_{x^{-a}}^{1}=0-\ln(x^{-a})=a\ln x.
> $$
> Hence the two sides are equal. Geometrically, the area under $1/t$ from $x^{-a}$ to $1$ is $a$ times the area from $x^{-1}$ to $1$. If $x>1$ is fixed and $a\to\infty$, then $x^{-a}\to0^+$ and the right-hand area $a\ln x\to\infty$. This also shows that
> $$
> \int_{0}^{1}\frac{dt}{t}
> $$
> is an [[反常积分|improper integral]] that diverges near $0$.
> <!-- bilingual-en:end -->

> [!question]- 三问自检
> 1. 为什么固定 \(y\) 后研究 \(G(x)\)？2. \(G'=0\) 给出什么？3. \(L'(x)>0\) 有何用途？
>
> 答：把二元恒等式化成单变量；\(G\) 为常数；保证 \(L\) 一一对应并存在反函数。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** Why fix \(y\) and study \(G(x)\)?<br>
> **2.** What follows from \(G'=0\)?<br>
> **3.** Why is \(L'(x)>0\) useful?<br>
>
> Answer: It turns a two-variable identity into a one-variable problem; \(G\) is constant; strict monotonicity makes \(L\) one-to-one and gives it an inverse.
> <!-- bilingual-en:end -->

## Session 55：Creating New Functions

若
<!-- bilingual-en:start -->
If
<!-- bilingual-en:end -->

$$
F(x)=\int_a^xf(t)dt,
$$

若 $f$ 在所讨论的整个区间连续，则 $F'=f$；在 $f$ 还可导的点，才有 $F''=f'$。在这些条件下，可从 $f$ 的图像直接判断 $F$：
<!-- bilingual-en:start -->
If $f$ is continuous throughout the interval, then $F'=f$; at points where $f$ is also differentiable, $F''=f'$. Under these assumptions, the graph of $f$ lets us read the following features of $F$ directly:
<!-- bilingual-en:end -->

- \(f>0\)：\(F\) 递增；
- \(f<0\)：\(F\) 递减；
- \(f=0\) 且变号：\(F\) 可能取极值；
- \(f'>0\)：\(F\) 向上凹。
<!-- bilingual-en:start -->
- \(f>0\): \(F\) is increasing;
- \(f<0\): \(F\) is decreasing;
- \(f=0\) with a sign change: \(F\) may have a local extremum;
- \(f'>0\): \(F\) is concave up.
<!-- bilingual-en:end -->

![[累积函数图像]]

标准正态[[累积分布函数|累计函数]]
<!-- bilingual-en:start -->
The standard normal [[累积分布函数|cumulative distribution function]]
<!-- bilingual-en:end -->

$$
\Phi(x)=\frac1{\sqrt{2\pi}}\int_{-\infty}^xe^{-t^2/2}dt
$$

也是“由旧函数造新函数”。
<!-- bilingual-en:start -->
is another example of constructing a new function from an existing one.
<!-- bilingual-en:end -->

### 本地材料

- [[Ses55a_Lecture_Notes.pdf|55a Bell Curve]]
- [[Ses55b_Lecture_Notes.pdf|55b More New Functions]]
- [[Exercise055_Problems.pdf|Exercise 55]] · [[Exercise055_Solutions.pdf|解答]]

> [!example]- Exercise 55：分段指数密度
> 设
> $$
> f(x)=\begin{cases}
> \lambda e^{-\lambda x},&x\ge0,\\
> 0,&x<0,
> \end{cases}
> \qquad \lambda>0.
> $$
> 原函数在正半轴上为 $-e^{-\lambda x}$。因此：
>
> - 若 $b\ge a>0$，
>   $$
>   \int_a^bf(x)dx=[-e^{-\lambda x}]_a^b
>   =\boxed{e^{-\lambda a}-e^{-\lambda b}}.
>   $$
> - 若 $a\le0<b$，负半轴贡献为零，
>   $$
>   \int_a^bf(x)dx=\int_0^b\lambda e^{-\lambda x}dx
>   =\boxed{1-e^{-\lambda b}}.
>   $$
> - 若 \(a\le b\le0\)，则 \(\boxed{\int_a^bf(x)dx=0}\)。
>
> 三种情形必须由 $0$ 是否落在积分区间内决定；不能跨过分段点仍套同一个原函数。
> <!-- bilingual-en:start -->
> Let
> $$
> f(x)=\begin{cases}
> \lambda e^{-\lambda x},&x\ge0,\\
> 0,&x<0,
> \end{cases}
> \qquad \lambda>0.
> $$
> On the positive half-line, an antiderivative is $-e^{-\lambda x}$. Therefore:
>
> - If $b\ge a>0$,
>   $$
>   \int_a^bf(x)dx=[-e^{-\lambda x}]_a^b
>   =\boxed{e^{-\lambda a}-e^{-\lambda b}}.
>   $$
> - If $a\le0<b$, the negative half-line contributes zero:
>   $$
>   \int_a^bf(x)dx=\int_0^b\lambda e^{-\lambda x}dx
>   =\boxed{1-e^{-\lambda b}}.
>   $$
> - If $a\le b\le0$, then $\boxed{\int_a^bf(x)dx=0}$.
>
> The correct case depends on whether $0$ lies inside the interval of integration; one antiderivative formula cannot be applied across the breakpoint without splitting the integral.
> <!-- bilingual-en:end -->

> [!question]- 三问自检
> 1. \(f>0\) 时累积函数怎样？2. \(f'=0\) 对累积函数意味着什么？3. 为什么不必知道累积函数显式式子？
>
> 答：递增；可能是凹凸性改变的候选点；FTC 已把其导数直接写成 \(f\)。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** How does the accumulation function behave when \(f>0\)?<br>
> **2.** What can \(f'=0\) mean for the accumulation function?<br>
> **3.** Why is its explicit formula unnecessary?<br>
>
> Answer: It is increasing; the point may be a candidate for a change in concavity; the FTC already identifies its derivative as \(f\).
> <!-- bilingual-en:end -->

## Session 56：Geometric Interpretation

若 $a<b$，$f,g$ 在 $[a,b]$ 连续且 $f\ge g$，[[曲线间面积|两曲线面积]]为
<!-- bilingual-en:start -->
If $a<b$, $f,g$ are continuous on $[a,b]$, and $f\ge g$, the [[曲线间面积|area between the curves]] is
<!-- bilingual-en:end -->

$$
\boxed{A=\int_a^b[f(x)-g(x)]dx}.
$$

若上下关系改变，在交点处分段。若曲线自然写成 \(x=R(y)\)、\(x=L(y)\)，则
<!-- bilingual-en:start -->
If the upper–lower ordering changes, split the integral at the intersection. If the curves are naturally written as \(x=R(y)\) and \(x=L(y)\), then
<!-- bilingual-en:end -->

$$
A=\int_c^d[R(y)-L(y)]dy.
$$

选择 \(dx\) 或 \(dy\) 的原则是减少反解、分段与绝对值。
<!-- bilingual-en:start -->
Choose \(dx\) or \(dy\) to minimise inverse solving, piecewise cases, and absolute values.
<!-- bilingual-en:end -->

![[98_attachment/MIT18.01SC/unit03-signed-area.png]]

### 本地材料

- [[Ses56a_Lecture_Notes.pdf|56a Areas Between Curves]]
- [[Ses56b_Lecture_Notes.pdf|56b Example]]
- [[Exercise056_Problems.pdf|Exercise 56]] · [[Exercise056_Solutions.pdf|解答]]

> [!example]- Exercise 56：微笑形区域面积
> 两曲线为
> $$
> y=\frac12x^2-\frac12,
> \qquad y=x^4-1.
> $$
> 令二者相等：$2x^4-x^2-1=0$。设 $u=x^2$，则
> $(2u+1)(u-1)=0$，实交点为 $x=\pm1$。在 $x=0$ 处，上曲线是 $-1/2$，下曲线是 $-1$，所以
> $$
> \begin{aligned}
> A
> &=\int_{-1}^{1}\left[\left(\frac12x^2-\frac12\right)-(x^4-1)\right]dx\\
> &=2\int_0^1\left(-x^4+\frac12x^2+\frac12\right)dx\\
> &=2\left[-\frac{x^5}{5}+\frac{x^3}{6}+\frac{x}{2}\right]_0^1
> =\boxed{\frac{14}{15}}.
> \end{aligned}
> $$
> 被积函数在 $[-1,1]$ 非负；偶对称使计算减半。
> <!-- bilingual-en:start -->
> The curves are
> $$
> y=\frac12x^2-\frac12,
> \qquad y=x^4-1.
> $$
> Setting them equal gives $2x^4-x^2-1=0$. With $u=x^2$,
> $(2u+1)(u-1)=0$, so the real intersections are $x=\pm1$. At $x=0$, the upper curve is $-1/2$ and the lower curve is $-1$. Hence
> $$
> \begin{aligned}
> A
> &=\int_{-1}^{1}\left[\left(\frac12x^2-\frac12\right)-(x^4-1)\right]dx\\
> &=2\int_0^1\left(-x^4+\frac12x^2+\frac12\right)dx\\
> &=2\left[-\frac{x^5}{5}+\frac{x^3}{6}+\frac{x}{2}\right]_0^1
> =\boxed{\frac{14}{15}}.
> \end{aligned}
> $$
> The integrand is nonnegative on $[-1,1]$, and even symmetry halves the work.
> <!-- bilingual-en:end -->

> [!question]- 三问自检
> 1. 为何必须“上减下”？2. 何时改用 \(dy\)？3. 曲线交叉时怎么办？
>
> 答：小条高度必须非负；水平切片能减少反解或分段时；在交点拆分并重新判断次序。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** Why must we subtract the lower curve from the upper curve?<br>
> **2.** When is \(dy\) preferable?<br>
> **3.** What should be done when the curves cross?<br>
>
> Answer: The strip height must be nonnegative; use horizontal slices when they require fewer inverse branches or pieces; split at each intersection and re-establish the ordering.
> <!-- bilingual-en:end -->

## Session 57：How to Calculate Volumes

旋转体及一般[[切片体积]]都从截面积开始。若 $a<b$，非负截面积 $A(x)$ 在 $[a,b]$ 可积，
<!-- bilingual-en:start -->
Solids of revolution and general [[切片体积|volumes by slicing]] begin with cross-sectional area. If $a<b$ and the nonnegative cross-sectional area $A(x)$ is integrable on $[a,b]$,
<!-- bilingual-en:end -->

$$
\boxed{V=\int_a^bA(x)dx}.
$$

[[圆盘垫圈法]]：$R(x)\ge r(x)\ge0$ 都是到旋转轴的距离；截面若实心则 $r=0$。
<!-- bilingual-en:start -->
[[圆盘垫圈法|Disk/washer method]]: $R(x)\ge r(x)\ge0$ are distances from the axis; a filled cross-section has $r=0$.
<!-- bilingual-en:end -->

$$
V=\pi\int_a^b[R(x)^2-r(x)^2]dx.
$$

[[圆柱壳法]]：绕竖直轴 $x=c$ 时，壳半径是非负距离 $\rho=|x-c|$。先按半径合并重叠高度，让每个壳只计一次，则
<!-- bilingual-en:start -->
[[圆柱壳法|Cylindrical shell method]]: for a vertical axis $x=c$, the shell radius is the nonnegative distance $\rho=|x-c|$. Merge overlapping height intervals at each radius so each shell is counted once; then
<!-- bilingual-en:end -->

$$
V=2\pi\int_{\rho_{\min}}^{\rho_{\max}}\rho\,H(\rho)d\rho.
$$

熟悉的 $V=2\pi\int_a^bxh(x)dx$ 适用于绕 $y$ 轴、$0\le a<b$ 且竖直切片只覆盖每个壳一次的情形。区域跨轴时，不能把两侧重复生成的实体直接相加；见[[旋转体重复覆盖]]。
<!-- bilingual-en:start -->
The familiar $V=2\pi\int_a^bxh(x)dx$ applies about the $y$-axis when $0\le a<b$ and the vertical slices cover each shell once. If the region crosses the axis, adding both sides may count the same solid twice; see [[旋转体重复覆盖|overlap in solids of revolution]].
<!-- bilingual-en:end -->

washer 切片垂直于旋转轴，shell 切片平行于旋转轴。先画代表性切片，再决定方法。
<!-- bilingual-en:start -->
The washer slice is perpendicular to the axis of rotation, and the shell slice is parallel to the axis of rotation.  Draw representative slices before deciding on a method.
<!-- bilingual-en:end -->

![[98_attachment/MIT18.01SC/unit03-volume-methods.png|809]]

### 本地材料

- [[Ses57a_Lecture_Notes.pdf|57a Volumes by Slicing]]
- [[Ses57b_Lecture_Notes.pdf|57b Solids of Revolution]]
- [[Exercise057_Problems.pdf|Exercise 57]] · [[Exercise057_Solutions.pdf|解答]]

> [!example]- Exercise 57：绕 \(x\) 轴的两个旋转体
> (a) $y=3x-x^2$ 与 $y=0$ 在 $x=0,3$ 相交。圆盘半径为 $3x-x^2$：
> $$
> \begin{aligned}
> V
> &=\pi\int_0^3(3x-x^2)^2dx
> =\pi\int_0^3(9x^2-6x^3+x^4)dx\\
> &=\pi\left[3x^3-\frac32x^4+\frac{x^5}{5}\right]_0^3
> =\boxed{\frac{81\pi}{10}}.
> \end{aligned}
> $$
> (b) 对 $y=\sqrt{ax}$、$0\le x\le a$（$a>0$），圆盘半径平方为 $y^2=ax$：
> $$
> V=\pi\int_0^aax\,dx
> =\pi a\left[\frac{x^2}{2}\right]_0^a
> =\boxed{\frac{\pi a^3}{2}}.
> $$
> 两题答案都具有长度三次的量纲；若把半径而不是半径平方放入圆盘公式，量纲会立刻暴露错误。
> <!-- bilingual-en:start -->
> **(a)** The curves $y=3x-x^2$ and $y=0$ intersect at $x=0$ and $x=3$. The disk radius is $3x-x^2$:
> $$
> \begin{aligned}
> V
> &=\pi\int_0^3(3x-x^2)^2dx
> =\pi\int_0^3(9x^2-6x^3+x^4)dx\\
> &=\pi\left[3x^3-\frac32x^4+\frac{x^5}{5}\right]_0^3
> =\boxed{\frac{81\pi}{10}}.
> \end{aligned}
> $$
> **(b)** For $y=\sqrt{ax}$ on $0\le x\le a$, where $a>0$, the squared disk radius is $y^2=ax$:
> $$
> V=\pi\int_0^aax\,dx
> =\pi a\left[\frac{x^2}{2}\right]_0^a
> =\boxed{\frac{\pi a^3}{2}}.
> $$
> Both answers have dimensions of length cubed. If the radius rather than its square is inserted into the disk formula, dimensional analysis exposes the error immediately.
> <!-- bilingual-en:end -->

> [!question]- 三问自检
> 1. washer 与旋转轴方向关系？2. shell 的周长因子？3. 为什么先画小切片？
>
> 答：切片垂直于轴；\(2\pi r\)；切片直接决定半径、高度、厚度与积分变量。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** How is a washer oriented relative to the axis of rotation?<br>
> **2.** What is the circumference factor in the shell method?<br>
> **3.** Why draw a representative slice first?<br>
>
> Answer: A washer is perpendicular to the axis; the factor is \(2\pi r\); the slice determines the radius, height, thickness, and variable of integration.
> <!-- bilingual-en:end -->

## Session 58：Volume of a Sphere

半径 $R>0$ 的圆满足 $y^2=R^2-x^2$。将圆盘区域绕 $x$ 轴旋转：
<!-- bilingual-en:start -->
A circle of radius $R>0$ satisfies $y^2=R^2-x^2$. Rotate its filled disk about the $x$-axis:
<!-- bilingual-en:end -->

$$
\begin{aligned}
V
&=\pi\int_{-R}^{R}(R^2-x^2)dx\\
&=\pi\left[R^2x-\frac{x^3}{3}\right]_{-R}^{R}\\
&=\boxed{\frac43\pi R^3}.
\end{aligned}
$$

### 本地材料

- [[Ses58a_Lecture_Notes.pdf|58a Sphere]]
- [[Exercise058_Problems.pdf|Exercise 58]] · [[Exercise058_Solutions.pdf|解答]]

> [!example]- Exercise 58：扁球体体积
> 椭圆 $x^2+4y^2=4$ 的纵坐标范围是 $[-1,1]$。绕 $y$ 轴旋转时，水平截面是半径
> $$
> x=2\sqrt{1-y^2}
> $$
> 的圆盘，故
> $$
> \begin{aligned}
> V
> &=\pi\int_{-1}^{1}x^2dy
> =4\pi\int_{-1}^{1}(1-y^2)dy\\
> &=4\pi\left[y-\frac{y^3}{3}\right]_{-1}^{1}
> =\boxed{\frac{16\pi}{3}}.
> \end{aligned}
> $$
> 用左半椭圆或右半椭圆生成的是同一个实体，不能把两半所得体积再相加。
> <!-- bilingual-en:start -->
> The ellipse $x^2+4y^2=4$ has vertical range $[-1,1]$. When it is rotated about the $y$-axis, a horizontal cross-section is a disk of radius
> $$
> x=2\sqrt{1-y^2}.
> $$
> Therefore,
> $$
> \begin{aligned}
> V
> &=\pi\int_{-1}^{1}x^2dy
> =4\pi\int_{-1}^{1}(1-y^2)dy\\
> &=4\pi\left[y-\frac{y^3}{3}\right]_{-1}^{1}
> =\boxed{\frac{16\pi}{3}}.
> \end{aligned}
> $$
> Rotating either the left or the right half of the ellipse produces the same solid; do not add the two volumes.
> <!-- bilingual-en:end -->

> [!question]- 三问自检
> 1. 截面积是什么？2. 为何上下半圆不分别相加？3. 单位为何是三次？
>
> 答：\(\pi(R^2-x^2)\)；旋转后同一圆盘只算一次；面积乘厚度。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** What is the cross-sectional area?<br>
> **2.** Why are the upper and lower semicircles not added separately?<br>
> **3.** Why does the answer have cubic units?<br>
>
> Answer: \(\pi(R^2-x^2)\); rotation produces the same disk from either semicircle, so it is counted once; area multiplied by thickness has units of volume.
> <!-- bilingual-en:end -->

## Session 59：Volume of a Paraboloid

若 $R,h>0$，水平圆盘半径为 $R\sqrt{y/h}$（$0\le y\le h$），则
<!-- bilingual-en:start -->
If $R,h>0$ and the horizontal disk radius is $R\sqrt{y/h}$ for $0\le y\le h$, then
<!-- bilingual-en:end -->

$$
\begin{aligned}
V
&=\pi\int_0^hR^2\frac yh\,dy\\
&=\frac{\pi R^2}{h}\left[\frac{y^2}{2}\right]_0^h\\
&=\boxed{\frac12\pi R^2h}.
\end{aligned}
$$

量纲警告：若 \(x,y\) 带长度单位，\(y=x^2\) 通常不合法，应写 \(y=kx^2\)，其中 \(k\) 带 \(1/\text{length}\) 单位。
<!-- bilingual-en:start -->
Dimensional warning: if both \(x\) and \(y\) have units of length, \(y=x^2\) is dimensionally inconsistent. Write \(y=kx^2\), where \(k\) has units of \(1/\text{length}\).
<!-- bilingual-en:end -->

### 本地材料

- [[Ses59a_Lecture_Notes.pdf|59a Paraboloid]]
- [[Ses59b_Lecture_Notes.pdf|59b Units Warning]]
- [[Exercise059_Problems.pdf|Exercise 59]] · [[Exercise059_Solutions.pdf|解答]]

> [!example]- Exercise 59：绕外部竖直轴的垫圈
> 区域由 $y=0$、$x=4$、$y=\sqrt{x}$ 围成，绕 $x=6$ 旋转。用水平切片最自然：$x=y^2$，且 $0\le y\le2$。外半径来自左边界 $x=y^2$，内半径来自右边界 $x=4$：
> $$
> R(y)=6-y^2,\qquad r(y)=6-4=2.
> $$
> 因此
> $$
> \begin{aligned}
> V
> &=\pi\int_0^2\left[(6-y^2)^2-2^2\right]dy\\
> &=\pi\int_0^2(32-12y^2+y^4)dy\\
> &=\pi\left[32y-4y^3+\frac{y^5}{5}\right]_0^2
> =\boxed{\frac{192\pi}{5}}.
> \end{aligned}
> $$
> “外减内”发生在平方以后：$R^2-r^2$，不是 $(R-r)^2$。
> <!-- bilingual-en:start -->
> The region bounded by $y=0$, $x=4$, and $y=\sqrt x$ is rotated about $x=6$. Horizontal slices are most natural: $x=y^2$ and $0\le y\le2$. The outer radius comes from the left boundary $x=y^2$, and the inner radius from the right boundary $x=4$:
> $$
> R(y)=6-y^2,\qquad r(y)=6-4=2.
> $$
> Hence,
> $$
> \begin{aligned}
> V
> &=\pi\int_0^2\left[(6-y^2)^2-2^2\right]dy\\
> &=\pi\int_0^2(32-12y^2+y^4)dy\\
> &=\pi\left[32y-4y^3+\frac{y^5}{5}\right]_0^2
> =\boxed{\frac{192\pi}{5}}.
> \end{aligned}
> $$
> “Outer minus inner” is applied after squaring: $R^2-r^2$, not $(R-r)^2$.
> <!-- bilingual-en:end -->

> [!question]- 三问自检
> 1. 抛物面体积是同底同高圆柱的几分之几？2. 为什么用 \(dy\)？3. \(y=kx^2\) 中 \(k\) 的量纲？
>
> 答：\(1/2\)；水平圆盘半径容易表示；\(1/\text{length}\)。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** What fraction of the volume of a cylinder with the same base and height does the paraboloid occupy?<br>
> **2.** Why integrate with respect to \(y\)?<br>
> **3.** What are the dimensions of \(k\) in \(y=kx^2\)?<br>
>
> Answer: \(1/2\); horizontal slices make the disk radius easy to express; \(1/\text{length}\).
> <!-- bilingual-en:end -->

## Problem Set 7

- 官方范围：4B 2（使用 1e、1g）、5；4C 2、3；4J 3
- [[PSet04_Problems.pdf|Problems]]
- [[PSet04_Solutions.pdf|Solutions]]

> [!example]- 完整官方题册与逐题答案
> ![[PSet04_Problems.pdf#height=650]]
>
> ![[PSet04_Solutions.pdf#height=650]]

> [!example]- 4B 2：对 1e、1g 使用垫圈法
> 1e 的区域由 $y=2x-x^2$ 与 $y=0$ 围成，并绕 $y$ 轴旋转。为使用垂直于旋转轴的水平垫圈，解出
> $$
> x=1\pm\sqrt{1-y},\qquad 0\le y\le1.
> $$
> 外、内半径分别为 $1+\sqrt{1-y}$ 与 $1-\sqrt{1-y}$，故
> $$
> \begin{aligned}
> V
> &=\pi\int_0^1\left[(1+\sqrt{1-y})^2-(1-\sqrt{1-y})^2\right]dy\\
> &=4\pi\int_0^1\sqrt{1-y}\,dy
> =\boxed{\frac{8\pi}{3}}.
> \end{aligned}
> $$
> 1g 的区域由 $y^2=ax$、$y=0$、$x=a$ 围成（$a>0$），绕 $y$ 轴旋转。对 $0\le y\le a$，外半径为 $a$，内半径为 $y^2/a$：
> $$
> \begin{aligned}
> V
> &=\pi\int_0^a\left[a^2-\left(\frac{y^2}{a}\right)^2\right]dy\\
> &=\pi\left[a^2y-\frac{y^5}{5a^2}\right]_0^a
> =\boxed{\frac{4\pi a^3}{5}}.
> \end{aligned}
> $$
> <!-- bilingual-en:start -->
> In 1e, the region bounded by $y=2x-x^2$ and $y=0$ is rotated about the $y$-axis. To use horizontal washers perpendicular to the axis, solve for
> $$
> x=1\pm\sqrt{1-y},\qquad 0\le y\le1.
> $$
> The outer and inner radii are $1+\sqrt{1-y}$ and $1-\sqrt{1-y}$, so
> $$
> \begin{aligned}
> V
> &=\pi\int_0^1\left[(1+\sqrt{1-y})^2-(1-\sqrt{1-y})^2\right]dy\\
> &=4\pi\int_0^1\sqrt{1-y}\,dy
> =\boxed{\frac{8\pi}{3}}.
> \end{aligned}
> $$
> In 1g, the region bounded by $y^2=ax$, $y=0$, and $x=a$, with $a>0$, is rotated about the $y$-axis. For $0\le y\le a$, the outer radius is $a$ and the inner radius is $y^2/a$:
> $$
> \begin{aligned}
> V
> &=\pi\int_0^a\left[a^2-\left(\frac{y^2}{a}\right)^2\right]dy\\
> &=\pi\left[a^2y-\frac{y^5}{5a^2}\right]_0^a
> =\boxed{\frac{4\pi a^3}{5}}.
> \end{aligned}
> $$
> <!-- bilingual-en:end -->

> [!example]- 4B 5：等边三角形绕一边旋转
> 设边长为 $a>0$，取旋转边为 $x$ 轴。三角形高度为 $\sqrt{3}a/2$，利用关于中线的对称性，左半边界可写成 $y=\sqrt{3}x$，$0\le x\le a/2$。圆盘法给出
> $$
> \begin{aligned}
> V
> &=2\pi\int_0^{a/2}(\sqrt3x)^2dx
> =6\pi\left[\frac{x^3}{3}\right]_0^{a/2}
> =\boxed{\frac{\pi a^3}{4}}.
> \end{aligned}
> $$
> 系数 $2$ 来自左右两半；少乘它会只算出半个旋转体。
> <!-- bilingual-en:start -->
> Let the side length be $a>0$ and take the axis of rotation as the $x$-axis. The triangle's height is $\sqrt{3}a/2$. By symmetry about the median, the left boundary is $y=\sqrt{3}x$ for $0\le x\le a/2$. The disk method gives
> $$
> \begin{aligned}
> V
> &=2\pi\int_0^{a/2}(\sqrt3x)^2dx
> =6\pi\left[\frac{x^3}{3}\right]_0^{a/2}
> =\boxed{\frac{\pi a^3}{4}}.
> \end{aligned}
> $$
> The factor $2$ accounts for the two halves; omitting it computes only half of the solid.
> <!-- bilingual-en:end -->

> [!example]- 4C 2、3：圆柱壳与垫圈互相核对
> 2. 区域 $0\le y\le x^2$、$0\le x\le1$ 绕 $y$ 轴。半径为 $x$、壳高为 $x^2$：
>    $$
>    V=2\pi\int_0^1x(x^2)dx
>    =2\pi\left[\frac{x^4}{4}\right]_0^1
>    =\boxed{\frac\pi2}.
>    $$
> 3. 区域 $\sqrt{x}\le y\le1$、$x\ge0$ 绕 $y$ 轴。用壳法时 $0\le x\le1$，壳高为 $1-\sqrt{x}$：
>    $$
>    V=2\pi\int_0^1x(1-\sqrt x)dx
>    =2\pi\left(\frac12-\frac25\right)
>    =\boxed{\frac\pi5}.
>    $$
>    反解 $x=y^2$ 后，用圆盘法也得
>    $$
>    V=\pi\int_0^1(y^2)^2dy=\frac\pi5,
>    $$
>    两种切片的一致性完成核对。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **2.** Rotate the region $0\le y\le x^2$, $0\le x\le1$ about the $y$-axis. The shell radius is $x$ and its height is $x^2$:<br>
> $$
> V=2\pi\int_0^1x(x^2)dx
> =2\pi\left[\frac{x^4}{4}\right]_0^1
> =\boxed{\frac\pi2}.
> $$
> **3.** Rotate the region $\sqrt x\le y\le1$, $x\ge0$ about the $y$-axis. With shells, $0\le x\le1$ and the shell height is $1-\sqrt x$:<br>
> $$
> V=2\pi\int_0^1x(1-\sqrt x)dx
> =2\pi\left(\frac12-\frac25\right)
> =\boxed{\frac\pi5}.
> $$
> Solving for $x=y^2$ and using disks gives the same result:
> $$
> V=\pi\int_0^1(y^2)^2dy=\frac\pi5.
> $$
> The agreement between the two slicing methods completes the check.
> <!-- bilingual-en:end -->

> [!example]- 4J 3：反射池中的总物质量
> 池深为 $D$、半径为 $R$，距中心 $r$ 处浓度为 $k/(1+r^2)$。半径 $r$、厚度 $dr$ 的圆环体积是
> $$
> dV=(2\pi r\,dr)D.
> $$
> 浓度乘体积并从中心累加到池边：
> $$
> \begin{aligned}
> A
> &=\int_0^R\frac{k}{1+r^2}(2\pi rD)dr\\
> &=\pi kD[\ln(1+r^2)]_0^R
> =\boxed{\pi kD\ln(1+R^2)}.
> \end{aligned}
> $$
> 单位是“浓度 × 体积”，即题目要求的物质量（克）。
> <!-- bilingual-en:start -->
> The pool has depth $D$ and radius $R$. At distance $r$ from the center, the concentration is $k/(1+r^2)$. An annulus of radius $r$ and thickness $dr$ has volume
> $$
> dV=(2\pi r\,dr)D.
> $$
> Multiply concentration by volume and accumulate from the center to the edge:
> $$
> \begin{aligned}
> A
> &=\int_0^R\frac{k}{1+r^2}(2\pi rD)dr\\
> &=\pi kD[\ln(1+r^2)]_0^R
> =\boxed{\pi kD\ln(1+R^2)}.
> \end{aligned}
> $$
> The units are “concentration × volume,” namely the requested mass in grams.
> <!-- bilingual-en:end -->

---

## Part C：Average Value, Probability and Numerical Integration

## Session 60：Integrals and Averages

若 $a<b$ 且 $f$ 可积，[[积分平均值|连续平均值]]为：
<!-- bilingual-en:start -->
For $a<b$ and integrable $f$, the [[积分平均值|continuous average]] is
<!-- bilingual-en:end -->

$$
\boxed{f_{\mathrm{avg}}=\frac1{b-a}\int_a^bf(x)dx}.
$$

因为
<!-- bilingual-en:start -->
because
<!-- bilingual-en:end -->

$$
\int_a^bf=f_{\mathrm{avg}}(b-a),
$$

它是与原有向面积相同的等高矩形高度。沿连续可导图像 $y=y(x)$ 按弧长平均时，以下 $dx$ 按 $x$ 递增方向取正，且曲线只描画一次、弧长有限并为正：
<!-- bilingual-en:start -->
It is the height of a rectangle with the same signed area as the original graph. For an arc-length average along a continuously differentiable graph $y=y(x)$, take $dx$ in the increasing-$x$ direction, trace the curve once, and assume positive finite length:
<!-- bilingual-en:end -->

$$
ds=\sqrt{1+(y')^2}dx,
\qquad
q_{\mathrm{avg}}=\frac{\int_Cq\,ds}{\int_C1\,ds}.
$$

需保证 $\int_C|q|ds<\infty$；一般参数方向用 $ds=\sqrt{x'(t)^2+y'(t)^2}\,dt$（$t$ 递增），见[[参数曲线弧长与重复描画]]。
<!-- bilingual-en:start -->
Require $\int_C|q|ds<\infty$. For a general parametrisation, use $ds=\sqrt{x'(t)^2+y'(t)^2}\,dt$ with increasing $t$; see [[参数曲线弧长与重复描画|parametric arc length and repeated tracing]].
<!-- bilingual-en:end -->

### 本地材料

- [[Ses60a_Lecture_Notes.pdf|60a Average Value]]
- [[Ses60b_Lecture_Notes.pdf|60b Average Height]]
- [[Ses60c_Lecture_Notes.pdf|60c Arc-length Average]]
- [[Exercise060_Problems.pdf|Exercise 60]] · [[Exercise060_Solutions.pdf|解答]]

> [!example]- Exercise 60：复利账户的时间平均余额
> 若 $A(t)=A_0e^{rt}$、$T>0$ 且 $r\ne0$，在 $0\le t\le T$ 上的平均余额为
> $$
> \begin{aligned}
> A_{\mathrm{avg}}
> &=\frac1T\int_0^TA_0e^{rt}dt
> =\frac{A_0}{T}\left[\frac{e^{rt}}r\right]_0^T\\
> &=\boxed{\frac{A_0}{rT}(e^{rT}-1)}.
> \end{aligned}
> $$
> 若 $r=0$，余额恒为 $A_0$，平均值也是 $A_0$；这也等于上述式子的 $r\to0$ 极限。代入 $A_0=100$、$r=0.05$、$T=1$，
> $$
> A_{\mathrm{avg}}=2000(e^{0.05}-1)\approx\boxed{102.54}.
> $$
> 它位于初值 $100$ 与终值 $100e^{0.05}\approx105.13$ 之间，符合单调增长函数的平均值范围。
> <!-- bilingual-en:start -->
> If $A(t)=A_0e^{rt}$, $T>0$, and $r\ne0$, then its average balance over $0\le t\le T$ is
> $$
> \begin{aligned}
> A_{\mathrm{avg}}
> &=\frac1T\int_0^TA_0e^{rt}dt
> =\frac{A_0}{T}\left[\frac{e^{rt}}r\right]_0^T\\
> &=\boxed{\frac{A_0}{rT}(e^{rT}-1)}.
> \end{aligned}
> $$
> If $r=0$, the balance and its average are both $A_0$, also the limit of the formula as $r\to0$. With $A_0=100$, $r=0.05$, and $T=1$,
> $$
> A_{\mathrm{avg}}=2000(e^{0.05}-1)\approx\boxed{102.54}.
> $$
> This lies between the initial balance $100$ and the final balance $100e^{0.05}\approx105.13$, as expected for the average of an increasing function.
> <!-- bilingual-en:end -->

> [!question]- 三问自检
> 1. 连续平均值为何除以 \(b-a\)？2. 平均值是否一定被函数取得？3. 沿曲线平均为何使用 \(ds\)？
>
> 答：除以总权重；连续时由积分平均值定理取得；等权对象是弧长而非水平投影。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** Why is a continuous average divided by \(b-a\)?<br>
> **2.** Must a function attain its average value?<br>
> **3.** Why does an average along a curve use \(ds\)?<br>
>
> Answer: We divide by the total weight; a continuous function attains its average by the mean value theorem for integrals; along a curve, equal weight is assigned by arc length rather than horizontal projection.
> <!-- bilingual-en:end -->

## Session 61：Weighted Averages

[[加权平均]]要求 $a<b$、权重密度 $w(x)\ge0$、$0<\int_a^bw(x)dx<\infty$，且 $\int_a^b|f(x)w(x)|dx<\infty$，此时
<!-- bilingual-en:start -->
For a [[加权平均|weighted average]], require $a<b$, $w(x)\ge0$, $0<\int_a^bw(x)dx<\infty$, and $\int_a^b|f(x)w(x)|dx<\infty$. Then
<!-- bilingual-en:end -->

$$
\boxed{\bar f=\frac{\int_a^bf(x)w(x)dx}{\int_a^bw(x)dx}}.
$$

细杆在 $[a,b]$ 上密度为 $\rho\ge0$、总质量 $M>0$ 时，[[质心]]由[[质心积分计算|一阶矩除以质量]]给出：
<!-- bilingual-en:start -->
For a thin rod on $[a,b]$ with density $\rho\ge0$ and positive mass $M$, its [[质心|center of mass]] is given by [[质心积分计算|first moment divided by mass]]:
<!-- bilingual-en:end -->

$$
M=\int_a^b\rho(x)dx,\qquad
\bar x=\frac1M\int_a^bx\rho(x)dx.
$$

“Boiling Cauldron” 的关键是先明确按长度、面积、体积还是质量抽样；普通算术平均往往用了错误权重。
<!-- bilingual-en:start -->
The key in “Boiling Cauldron” is to decide whether sampling is by length, area, volume, or mass. An ordinary arithmetic mean often assigns the wrong weights.
<!-- bilingual-en:end -->

### 本地材料

- [[Ses61a_Lecture_Notes.pdf|61a Weighted Averages]]
- [[Ses61b_Lecture_Notes.pdf|61b Boiling Cauldron]]
- [[Ses61c_Lecture_Notes.pdf|61c Continued]]
- [[Exercise061_Problems.pdf|Exercise 61]] · [[Exercise061_Solutions.pdf|解答]]

> [!example]- Exercise 61：平面区域的质心
> 区域由 $x=-1$、$x=3$、$y=(x-1)^2$ 与 $y=4$ 围成。假设均匀密度，它关于 $x=1$ 对称，所以
> $$
> \boxed{\bar x=1}.
> $$
> 用竖直切片求面积：
> $$
> A=\int_{-1}^{3}\left[4-(x-1)^2\right]dx=\frac{32}{3}.
> $$
> 求 \(\bar y\) 时改用水平切片更简洁。由 \(y=(x-1)^2\) 得同一高度处宽度
> $x_{\rm right}-x_{\rm left}=2\sqrt y$，其中 $0\le y\le4$。因此
> $$
> \bar y
> =\frac{\int_0^4 y(2\sqrt y)dy}{\int_0^4 2\sqrt y\,dy}
> =\frac{\left[\frac45y^{5/2}\right]_0^4}
> {\left[\frac43y^{3/2}\right]_0^4}
> =\frac{128/5}{32/3}
> =\boxed{\frac{12}{5}}.
> $$
> 故质心为 \(\boxed{(1,12/5)}\)，且确实落在区域的对称轴上。
> <!-- bilingual-en:start -->
> The region is bounded by $x=-1$, $x=3$, $y=(x-1)^2$, and $y=4$. Assuming uniform density, it is symmetric about $x=1$, so
> $$
> \boxed{\bar x=1}.
> $$
> Using vertical slices, its area is
> $$
> A=\int_{-1}^{3}\left[4-(x-1)^2\right]dx=\frac{32}{3}.
> $$
> Horizontal slices are simpler for finding $\bar y$. From $y=(x-1)^2$, the width at height $y$ is
> $x_{\rm right}-x_{\rm left}=2\sqrt y$ for $0\le y\le4$. Hence,
> $$
> \bar y
> =\frac{\int_0^4 y(2\sqrt y)dy}{\int_0^4 2\sqrt y\,dy}
> =\frac{\left[\frac45y^{5/2}\right]_0^4}
> {\left[\frac43y^{3/2}\right]_0^4}
> =\frac{128/5}{32/3}
> =\boxed{\frac{12}{5}}.
> $$
> The centroid is therefore $\boxed{(1,12/5)}$, which indeed lies on the region's axis of symmetry.
> <!-- bilingual-en:end -->

> [!question]- 三问自检
> 1. 分母 \(\int w\) 表示什么？2. \(w\) 已是概率密度时分母？3. 质心公式的分子是什么？
>
> 答：总权重；等于 1；关于原点的一阶矩 \(\int xw(x)dx\)。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** What does the denominator \(\int w\) represent?<br>
> **2.** What is the denominator when \(w\) is already a probability density?<br>
> **3.** What is the numerator in the centroid formula?<br>
>
> Answer: Total weight; it equals \(1\); the first moment about the origin, \(\int xw(x)\,dx\).
> <!-- bilingual-en:end -->

## Session 62：Integrals and Probability

[[概率密度函数|概率密度]] \(p\) 满足
<!-- bilingual-en:start -->
A [[概率密度函数|probability density]] \(p\) satisfies
<!-- bilingual-en:end -->

$$
p(x)\ge0,\qquad \int_{-\infty}^{\infty}p(x)dx=1.
$$

$$
P(a\le X\le b)=\int_a^bp(x)dx,
$$

$$
\mu=E[X]=\int_{-\infty}^{\infty}xp(x)dx,
\qquad
\operatorname{Var}(X)=\int_{-\infty}^{\infty}(x-\mu)^2p(x)dx.
$$

有限[[期望]]要求绝对一阶矩 $\int_{-\infty}^{\infty}|x|p(x)dx<\infty$；有限[[方差]]要求二阶矩 $\int_{-\infty}^{\infty}x^2p(x)dx<\infty$（它也保证一阶矩有限）。对称抵消不能替代这些存在性条件。
<!-- bilingual-en:start -->
A finite [[期望|expectation]] requires $\int_{-\infty}^{\infty}|x|p(x)dx<\infty$. A finite [[方差|variance]] requires $\int_{-\infty}^{\infty}x^2p(x)dx<\infty$, which also ensures a finite first absolute moment. Symmetric cancellation does not replace these existence conditions.
<!-- bilingual-en:end -->

密度可以大于 1；真正受限在 \([0,1]\) 的是面积代表的概率。在这里的绝对连续模型中，单点概率为零。
<!-- bilingual-en:start -->
A density may exceed $1$; it is the area representing a probability that must lie in $[0,1]$. In the absolutely continuous model used here, the probability of any single point is zero.
<!-- bilingual-en:end -->

### 本地材料

- [[Ses62a_Lecture_Notes.pdf|62a Probability Example]]
- [[Ses62b_Lecture_Notes.pdf|62b Summary]]
- [[Ses62c_Lecture_Notes.pdf|62c Errata]]
- [[Ses62d_Lecture_Notes.pdf|62d Extended Example]]

> [!question]- 三问自检
> 1. 密度能大于 1 吗？2. 对称密度均值？3. 为什么单点概率为零？
>
> 答：能；若密度关于 $0$ 对称且期望存在则为 $0$，一般关于 $c$ 对称时存在的均值为 $c$；零宽区间的积分为 $0$。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** Can a density exceed one?<br>
> **2.** What is the mean of a symmetric density?<br>
> **3.** Why does a single point have probability zero?<br>
>
> Answer: Yes; an existing mean is zero for symmetry about zero, and is $c$ for symmetry about $c$; a point is an interval of zero width, whose integral is zero.
> <!-- bilingual-en:end -->

## Session 63：Numerical Integration

数值积分把 $[a,b]$（$a<b$）等分为 $n$ 段，取步长 $h=(b-a)/n$、节点 $x_i=a+ih$。[[左右端点矩形法]]为：
<!-- bilingual-en:start -->
For numerical integration, split $[a,b]$ with $a<b$ into $n$ equal intervals, with $h=(b-a)/n$ and $x_i=a+ih$. The [[左右端点矩形法|left- and right-endpoint rules]] are
<!-- bilingual-en:end -->

$$
L_n=h\sum_{i=0}^{n-1}f(x_i),\quad
R_n=h\sum_{i=1}^{n}f(x_i),
$$

[[中点法]]在各段中点采样；[[梯形法]]则取两端高度的平均：
<!-- bilingual-en:start -->
The [[中点法|midpoint rule]] samples each midpoint; the [[梯形法|trapezoidal rule]] averages the two endpoint heights:
<!-- bilingual-en:end -->

$$
M_n=h\sum_{i=1}^{n}f\!\left(\frac{x_{i-1}+x_i}{2}\right),
$$

$$
\boxed{
T_n=\frac h2\left[f(x_0)+2\sum_{i=1}^{n-1}f(x_i)+f(x_n)\right]
}.
$$

Session 63 第四份本地材料存放在 Problem Sets，内容是[[Simpson法]]。另取偶数 $N$ 个小区间，步长 $h=(b-a)/N$、节点 $x_i=a+ih$（$0\le i\le N$）：
<!-- bilingual-en:start -->
The fourth local file for Session 63 is stored in Problem Sets and covers [[Simpson法|Simpson's rule]]. Use an even number $N$ of subintervals, with $h=(b-a)/N$ and $x_i=a+ih$ for $0\le i\le N$:
<!-- bilingual-en:end -->

$$
\boxed{
S_N=\frac h3[f(x_0)+4f(x_1)+2f(x_2)+\cdots+4f(x_{N-1})+f(x_N)]
}.
$$

### 本地材料

- [[Ses63a_Lecture_Notes.pdf|63a Introduction]]
- [[Ses63b_Lecture_Notes.pdf|63b Riemann Sums]]
- [[Ses63c_Lecture_Notes.pdf|63c Trapezoidal Rule]]
- [[Ses63d_Problems.pdf|63d Simpson's Rule]]
- [[Exercise063_Problems.pdf|Exercise 63]] · [[Exercise063_Solutions.pdf|解答]]

> [!example]- Exercise 63：两种数值近似与精确值
> 对 $f(x)=x^3-2x$ 在 $[-1,2]$ 上积分，题目给出的梯形法取 $n=6$，步长 $h=3/6=1/2$。代入节点 $-1,-1/2,0,1/2,1,3/2,2$ 后，
> $$
> T_6=\frac h2\left[f(-1)+2\sum_{i=1}^{5}f(x_i)+f(2)\right]
> =\boxed{0.93750}.
> $$
> 第二种 Riemann 和在每段取相对位置 $0.5$，即中点法；$n=12$、$h=3/12=1/4$，代入十二个中点得到
> $$
> \boxed{M_{12}=0.72656}.
> $$
> 因为本题有初等原函数，可以核验：
> $$
> \int_{-1}^{2}(x^3-2x)dx
> =\left[\frac{x^4}{4}-x^2\right]_{-1}^{2}
> =0-\left(\frac14-1\right)
> =\boxed{\frac34=0.75}.
> $$
> 中点近似的绝对误差约 $0.02344$，小于梯形近似的 $0.18750$。不要用“梯形用了两倍节点”解释误差：相邻梯形共享端点，且方法的精度取决于误差阶与函数曲率。
> <!-- bilingual-en:start -->
> Integrate $f(x)=x^3-2x$ over $[-1,2]$. The problem specifies the trapezoidal rule with $n=6$, so $h=3/6=1/2$. Substituting the nodes $-1,-1/2,0,1/2,1,3/2,2$ gives
> $$
> T_6=\frac h2\left[f(-1)+2\sum_{i=1}^{5}f(x_i)+f(2)\right]
> =\boxed{0.93750}.
> $$
> The second Riemann sum samples the relative position $0.5$ in each subinterval, so it is the midpoint rule. With $n=12$, $h=3/12=1/4$, and the twelve midpoints,
> $$
> \boxed{M_{12}=0.72656}.
> $$
> Because the integrand has an elementary antiderivative, both approximations can be checked against the exact value:
> $$
> \int_{-1}^{2}(x^3-2x)dx
> =\left[\frac{x^4}{4}-x^2\right]_{-1}^{2}
> =0-\left(\frac14-1\right)
> =\boxed{\frac34=0.75}.
> $$
> The midpoint approximation has absolute error about $0.02344$, smaller than the trapezoidal error $0.18750$. Do not explain this by saying that “the trapezoidal rule uses twice as many nodes”: adjacent trapezoids share endpoints, and accuracy is governed by the method's error order and the function's curvature.
> <!-- bilingual-en:end -->

> [!question]- 三问自检
> 1. \(n\) 个小区间有多少端点？2. 梯形法为何是左右和平均？3. Simpson 法为何要求偶数个小区间？
>
> 答：\(n+1\)；每段梯形面积是两端矩形面积平均；每两个小区间拟合一条抛物线。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** How many endpoints do \(n\) subintervals have?<br>
> **2.** Why is the trapezoidal rule the average of the left- and right-endpoint rules?<br>
> **3.** Why does Simpson's rule require an even number of subintervals?<br>
>
> Answer: \(n+1\); each trapezoid has the average of the two endpoint heights; Simpson's rule fits one parabola across each pair of subintervals.
> <!-- bilingual-en:end -->

![[98_attachment/MIT18.01SC/unit03-numerical-integration.png|897]]

## Session 64：Numerical Integration, Continued

对 \(f(x)=1/x\)，\(f''(x)>0\)，图像凸，弦位于曲线上方，因此梯形法高估 \(\ln2=\int_1^2dx/x\)。
<!-- bilingual-en:start -->
For $f(x)=1/x$, $f''(x)>0$, so the graph is convex and each chord lies above the curve. The trapezoidal rule therefore overestimates $\ln2=\int_1^2dx/x$.
<!-- bilingual-en:end -->

在同一粗网格的 $n$ 个小区间上计算 $T_n$ 和 $M_n$。加入这 $n$ 个中点后，细网格有 $2n$ 个小区间、$2n+1$ 个节点，步长为粗网格的一半；[[Simpson组合恒等式]]为：
<!-- bilingual-en:start -->
Compute $T_n$ and $M_n$ on the same coarse grid with $n$ subintervals. Adding its $n$ midpoints creates a fine grid with $2n$ subintervals and $2n+1$ nodes, at half the coarse step size. The [[Simpson组合恒等式|Simpson combination identity]] is
<!-- bilingual-en:end -->

$$
\boxed{\frac13T_n+\frac23M_n=S_{2n}}.
$$

展开权重即可证明：端点权重为 \(1\)，偶数内点为 \(2\)，奇数中点为 \(4\)。
<!-- bilingual-en:start -->
Expanding the weights shows that the endpoints have weight \(1\), even-indexed interior nodes have weight \(2\), and odd-indexed midpoint nodes have weight \(4\).
<!-- bilingual-en:end -->

- 若 $f\in C^2[a,b]$，[[梯形积分误差界]]与[[中点积分误差界]]由 $K_2=\max_{[a,b]}|f''|$ 控制；
- Simpson 法对三次及以下多项式精确；若 $f\in C^4[a,b]$，[[Simpson误差界]]由 $K_4=\max_{[a,b]}|f^{(4)}|$ 控制。
<!-- bilingual-en:start -->
- If $f\in C^2[a,b]$, the [[梯形积分误差界|trapezoidal]] and [[中点积分误差界|midpoint error bounds]] are controlled by $K_2=\max_{[a,b]}|f''|$.
- Simpson's rule is exact for polynomials of degree at most three. If $f\in C^4[a,b]$, its [[Simpson误差界|error bound]] is controlled by $K_4=\max_{[a,b]}|f^{(4)}|$.
<!-- bilingual-en:end -->

$$
|I-T_n|\le\frac{K_2(b-a)^3}{12n^2},\qquad
|I-M_n|\le\frac{K_2(b-a)^3}{24n^2},\qquad
|I-S_N|\le\frac{K_4(b-a)^5}{180N^4}\quad(N\in2\mathbb N_{>0}).
$$

### 本地材料

- [[Ses64a_Lecture_Notes.pdf|64a Trapezoid and \(\ln2\)]]
- [[Ses64b_Lecture_Notes.pdf|64b Simpson]]
- [[Ses64c_Lecture_Notes.pdf|64c Study Tips]]
- [[Exercise064_Problems.pdf|Exercise 64]] · [[Exercise064_Solutions.pdf|解答]]

> [!example]- Exercise 64：原题函数的 Simpson 积分与归一化核对
> 题册与官方解答都印有以下函数（$x$ 以英寸计），原式保留如下：
> $$
> h(x)=\frac1{2.8\sqrt{2\pi}}
> e^{-(x-69)^2/5.6}.
> $$
> 但[[高斯积分]]给出 $\int_{-\infty}^{\infty}h(x)dx=\sqrt{5.6\pi}/(2.8\sqrt{2\pi})=1/\sqrt{2.8}\approx0.5976$，并非 $1$，所以它不是归一化的[[概率密度函数]]。以下计算复现原式的积分近似，不能直接解释为概率。
> (a) 五到六英尺即 \(60\le x\le72\)。取步长 \(\Delta x=2\)，节点函数值依次约为
> $$
> \begin{array}{c|rrrrrrr}
> x&60&62&64&66&68&70&72\\ \hline
> h(x)&7.45\!\times\!10^{-8}&2.26\!\times\!10^{-5}&0.00160&0.0286&0.119&0.119&0.0286
> \end{array}
> $$
> 六个小区间满足 Simpson 法要求，所以
> $$
> \begin{aligned}
> \int_{60}^{72}h(x)dx
> &\approx\frac23[h(60)+4h(62)+2h(64)+4h(66)\\
> &\qquad\qquad+2h(68)+4h(70)+h(72)]\\
> &\approx\boxed{0.574}.
> \end{aligned}
> $$
> $0.574$ 是官方按所列节点得到的原式 Simpson 积分近似；它不是已验证的 $57.4\%$ 概率。
>
> (b) 八英尺是 $96$ 英寸。原式尾部积分是 $\int_{96}^{\infty}h(x)dx$；官方解答以 $100$ 英寸截断，因为此后函数已极小。取节点 $96,98,100$：
> $$
> h(96)\approx4.15\times10^{-58},\quad
> h(98)\approx8.55\times10^{-67},\quad
> h(100)\approx4.22\times10^{-76}.
> $$
> 故
> $$
> \int_{96}^{100}h(x)dx
> \approx\frac23[h(96)+4h(98)+h(100)]
> =\boxed{2.77\times10^{-58}},
> $$
> 这是极小的尾部积分近似。原解答中写成 $\int_8^\infty$ 是英尺与英寸混写；与函数自变量一致的下限应是 $96$。
>
> 两种修正对应不同分布，不能代替作者作选择：保留指数形状并归一化，得到 $\sqrt{2.8}\,h(x)$，方差为 $2.8$；若另外指定标准差为 $2.8$，则应保留前因子并把指数分母改为 $2(2.8)^2=15.68$。现有材料没有说明应采用哪一种修正。
> <!-- bilingual-en:start -->
> Both the exercise and its official solution print the following function, with $x$ measured in inches; the source formula is retained:
> $$
> h(x)=\frac1{2.8\sqrt{2\pi}}
> e^{-(x-69)^2/5.6}.
> $$
> The [[高斯积分|Gaussian integral]] gives $\int_{-\infty}^{\infty}h(x)dx=\sqrt{5.6\pi}/(2.8\sqrt{2\pi})=1/\sqrt{2.8}\approx0.5976$, rather than $1$. Thus it is not a normalised [[概率密度函数|probability density]]. The following computations reproduce integral approximations for the printed formula; they cannot be interpreted directly as probabilities.
> **(a)** Five to six feet corresponds to $60\le x\le72$. With step size $\Delta x=2$, the node values are approximately
> $$
> \begin{array}{c|rrrrrrr}
> x&60&62&64&66&68&70&72\\ \hline
> h(x)&7.45\!\times\!10^{-8}&2.26\!\times\!10^{-5}&0.00160&0.0286&0.119&0.119&0.0286
> \end{array}
> $$
> Six subintervals satisfy Simpson's requirement, so
> $$
> \begin{aligned}
> \int_{60}^{72}h(x)dx
> &\approx\frac23[h(60)+4h(62)+2h(64)+4h(66)\\
> &\qquad\qquad+2h(68)+4h(70)+h(72)]\\
> &\approx\boxed{0.574}.
> \end{aligned}
> $$
> The official value $0.574$ is a Simpson integral approximation for the printed function using the listed nodes, not a verified probability of $57.4\%$.
>
> **(b)** Eight feet is $96$ inches. The printed function's tail integral is $\int_{96}^{\infty}h(x)dx$; the official solution truncates at $100$ inches because the function is already extremely small there. At the nodes $96,98,100$,
> $$
> h(96)\approx4.15\times10^{-58},\quad
> h(98)\approx8.55\times10^{-67},\quad
> h(100)\approx4.22\times10^{-76}.
> $$
> Hence,
> $$
> \int_{96}^{100}h(x)dx
> \approx\frac23[h(96)+4h(98)+h(100)]
> =\boxed{2.77\times10^{-58}},
> $$
> a very small approximation to the tail integral. The original solution writes $\int_8^\infty$, mixing feet with inches; the lower limit consistent with the function's input unit is $96$.
>
> Two repairs describe different distributions and cannot be selected on the author's behalf. Preserving the exponent's shape and normalising gives $\sqrt{2.8}\,h(x)$, with variance $2.8$. Alternatively, specifying standard deviation $2.8$ requires keeping the prefactor and changing the exponent denominator to $2(2.8)^2=15.68$. The available materials do not establish which repair was intended.
> <!-- bilingual-en:end -->

> [!question]- 三问自检
> 1. 凸函数的梯形法为何高估？2. Simpson 权重怎样排列？3. 哪阶导数控制 Simpson 主误差？
>
> 答：弦位于图像上方；\(1,4,2,\ldots,4,1\)；四阶导数。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** Why does the trapezoidal rule overestimate a convex function?<br>
> **2.** How are Simpson's weights arranged?<br>
> **3.** Which derivative controls Simpson's leading error?<br>
>
> Answer: Each chord lies above the graph; $1,4,2,\ldots,4,1$; the fourth derivative.
> <!-- bilingual-en:end -->

## Session 65：Bell Curve, Conclusion

标准正态密度
<!-- bilingual-en:start -->
The standard normal density is
<!-- bilingual-en:end -->

$$
\phi(x)=\frac1{\sqrt{2\pi}}e^{-x^2/2}
$$

总面积为 $1$，归一化由[[高斯积分]]支持。它没有初等原函数，但可通过数值积分求概率。因 $\phi$ 为偶函数，对 $a\ge0$，
<!-- bilingual-en:start -->
Its total area is $1$, as established by the [[高斯积分|Gaussian integral]]. Although it has no elementary antiderivative, probabilities can be computed numerically. Since $\phi$ is even, for $a\ge0$,
<!-- bilingual-en:end -->

$$
P(X\le0)=\frac12,
\qquad
P(-a\le X\le a)=2\int_0^a\phi(x)dx.
$$

### 本地材料

- [[Ses65a_Lecture_Notes.pdf|65a Bell Curve Conclusion]]

> [!question]- 三问自检
> 1. 标准化常数的作用？2. 对称性怎样减少计算？3. 无初等原函数时怎样求概率？
>
> 答：使总面积为 1；只算半边再倍增；用数值积分或累计分布函数表。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** What does the normalising constant do?<br>
> **2.** How does symmetry reduce the computation?<br>
> **3.** How can probabilities be found without an elementary antiderivative?<br>
>
> Answer: It makes the total area \(1\); compute one half and reflect it; use numerical integration or a cumulative-distribution table.
> <!-- bilingual-en:end -->

## Problem Set 8

- 官方范围：4D 2、3、5
- [[PSet04_Problems.pdf|Problems]]
- [[PSet04_Solutions.pdf|Solutions]]

> [!example]- 完整官方题册与逐题答案
> ![[PSet04_Problems.pdf#height=650]]
>
> ![[PSet04_Solutions.pdf#height=650]]

> [!example]- 4D 2：\(1/x\) 在 \([a,2a]\) 上的平均值
> 假设 \(a>0\)，区间长度是 \(2a-a=a\)，所以
> $$
> \begin{aligned}
> \left(\frac1x\right)_{\rm avg}
> &=\frac1a\int_a^{2a}\frac{dx}{x}
> =\frac1a[\ln x]_a^{2a}\\
> &=\frac{\ln(2a)-\ln a}{a}
> =\boxed{\frac{\ln2}{a}}.
> \end{aligned}
> $$
> 因而题目所求常数为 \(\boxed{C=\ln2}\)。条件 \(a>0\) 保证区间不跨过 \(1/x\) 的奇点 \(0\)。
> <!-- bilingual-en:start -->
> Assume $a>0$. The interval has length $2a-a=a$, so
> $$
> \begin{aligned}
> \left(\frac1x\right)_{\rm avg}
> &=\frac1a\int_a^{2a}\frac{dx}{x}
> =\frac1a[\ln x]_a^{2a}\\
> &=\frac{\ln(2a)-\ln a}{a}
> =\boxed{\frac{\ln2}{a}}.
> \end{aligned}
> $$
> Thus the requested constant is $\boxed{C=\ln2}$. The condition $a>0$ keeps the interval from crossing the singularity of $1/x$ at $0$.
> <!-- bilingual-en:end -->

> [!example]- 4D 3：速度的函数平均等于平均速度
> 假设 $a<b$，位置 $s\in C^1[a,b]$ 且 $v=s'$。连续可导保证导数可积，仅有 $s$ 可导并不足以保证 $s'$ 黎曼可积。$v$ 在 $[a,b]$ 上的函数平均为
> $$
> v_{\rm avg}
> =\frac1{b-a}\int_a^bv(t)dt
> =\frac1{b-a}\int_a^bs'(t)dt.
> $$
> 由 [[Newton-Leibniz公式|FTC I]]，
> $$
> \boxed{v_{\rm avg}=\frac{s(b)-s(a)}{b-a}},
> $$
> 右边正是“总位移 ÷ 总时间”的平均速度。这里用的是位移而非总路程；若速度变号，两者不能混为一谈。
> <!-- bilingual-en:start -->
> Assume $a<b$, $s\in C^1[a,b]$, and $v=s'$. Continuous differentiability ensures that $v$ is integrable; differentiability alone does not ensure Riemann integrability of $s'$. The average value of $v$ on $[a,b]$ is
> $$
> v_{\rm avg}
> =\frac1{b-a}\int_a^bv(t)dt
> =\frac1{b-a}\int_a^bs'(t)dt.
> $$
> By [[Newton-Leibniz公式|FTC I]],
> $$
> \boxed{v_{\rm avg}=\frac{s(b)-s(a)}{b-a}}.
> $$
> The right-hand side is total displacement divided by total time. It uses displacement, not total distance; if velocity changes sign, the two quantities must not be confused.
> <!-- bilingual-en:end -->

> [!example]- 4D 5：由区间平均恢复被积函数
> 已知连续函数 $f$ 在 $[0,x]$ 上的平均值为 $g(x)$，即对 $x>0$
> $$
> g(x)=\frac1x\int_0^xf(t)dt.
> $$
> 先乘以 $x$：
> $$
> xg(x)=\int_0^xf(t)dt.
> $$
> 两边对 $x$ 求导。左边用乘积法则，右边用 [[积分求导定理|FTC II]]：
> $$
> g(x)+xg'(x)=f(x).
> $$
> 因此
> $$
> \boxed{f(x)=g(x)+xg'(x)}.
> $$
> 易错点是直接对 $g=(1/x)\int_0^xf$ 求导却漏掉 $1/x$ 的导数；先清除分母最稳妥。
> <!-- bilingual-en:start -->
> Suppose the average value of the continuous function $f$ on $[0,x]$ is $g(x)$, so for $x>0$,
> $$
> g(x)=\frac1x\int_0^xf(t)dt.
> $$
> First multiply by $x$:
> $$
> xg(x)=\int_0^xf(t)dt.
> $$
> Differentiate both sides with respect to $x$. The left side uses the product rule and the right side uses [[积分求导定理|FTC II]]:
> $$
> g(x)+xg'(x)=f(x).
> $$
> Therefore,
> $$
> \boxed{f(x)=g(x)+xg'(x)}.
> $$
> A common error is to differentiate $g=(1/x)\int_0^xf$ directly and omit the derivative of $1/x$. Clearing the denominator first is the safest route.
> <!-- bilingual-en:end -->

---

## Exam 3

## Session 66：Review for Exam 3

以下四份讲义虽放在 Unit 4 目录，实际属于 Exam 3 复习：

- [[Ses66a_Lecture_Notes.pdf|66a Questions on Test 3]]
- [[Ses66b_Lecture_Notes.pdf|66b Types of Riemann Sums]]
- [[Ses66c_Lecture_Notes.pdf|66c Asymptotes of Antiderivatives]]
- [[Ses66d_Lecture_Notes.pdf|66d Choosing a Technique]]

### 考前检查
<!-- bilingual-en:start -->
*Pre-exam checklist*
<!-- bilingual-en:end -->

1. 从和式读出区间、\(\Delta x\)、样本点和被积函数；
2. 区分“FTC 求导”和“原函数端点求值”；
3. 面积选择 \(dx/dy\)，体积先画切片；
4. 平均值和概率写正确权重；
5. 数值法不混淆节点数与区间数；
6. 混合问题写“流入率 − 流出率”及初值。
<!-- bilingual-en:start -->

&nbsp;
**1.** Read the interval, \(\Delta x\), sample points, and integrand from a sum.<br>
**2.** Distinguish differentiating an integral with FTC II from evaluating endpoint values with FTC I.<br>
**3.** Choose \(dx\) or \(dy\) for area; for volume, draw the representative slice first.<br>
**4.** Use the correct weights for averages and probabilities.<br>
**5.** Do not confuse the number of nodes with the number of subintervals in a numerical rule.<br>
**6.** For a mixing problem, write “inflow rate − outflow rate” and the initial condition.<br>
<!-- bilingual-en:end -->

> [!question]- 三问自检
> 1. Riemann 和最先找什么？2. 旋转体最先画什么？3. 混合模型浓度怎样写？
>
> 答：\(\Delta x\) 与样本点；代表性切片；当前溶质量除以当前总体积。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** What should be identified first in a Riemann sum?<br>
> **2.** What should be drawn first for a solid of revolution?<br>
> **3.** How is concentration written in a mixing model?<br>
>
> Answer: \(\Delta x\) and the sample points; a representative slice; the current mass of dissolved material divided by the current total volume.
> <!-- bilingual-en:end -->

## Session 67：Materials for Exam 3

- [[Exam3_Problems.pdf|Exam 3 Problems]]
- [[Exam3_Solutions.pdf|Exam 3 Official Solutions]]

### Problem 1：两曲线面积
<!-- bilingual-en:start -->
*Problem 1: Area between two curves*
<!-- bilingual-en:end -->

令
<!-- bilingual-en:start -->
Set
<!-- bilingual-en:end -->

$$
y^2-4y=2y-y^2
\Longrightarrow2y(y-3)=0,
$$

故积分限为 \(0,3\)。右减左：
<!-- bilingual-en:start -->
Thus the limits of integration are \(0\) and \(3\). Subtract the left boundary from the right boundary:
<!-- bilingual-en:end -->

$$
\begin{aligned}
A
&=\int_0^3[(2y-y^2)-(y^2-4y)]dy\\
&=\left[3y^2-\frac23y^3\right]_0^3\\
&=\boxed{9}.
\end{aligned}
$$

### Problem 2：绕 \(y=-1\) 的体积
<!-- bilingual-en:start -->
*Problem 2: Volume around \(y=-1\)*
<!-- bilingual-en:end -->

交点 \(e^x=2\) 给 \(x=\ln2\)。外半径 \(3\)，内半径 \(e^x+1\)：
<!-- bilingual-en:start -->
The intersection equation \(e^x=2\) gives \(x=\ln2\). The outer radius is \(3\), and the inner radius is \(e^x+1\):
<!-- bilingual-en:end -->

$$
\boxed{V=\pi\int_0^{\ln2}[9-(1+e^x)^2]dx}.
$$

### Problem 3：Riemann 和与 FTC
<!-- bilingual-en:start -->
*Problem 3: Riemann sums and the FTC*
<!-- bilingual-en:end -->

$$
\lim_{n\to\infty}\sum_{i=1}^n
\left(1+\frac{3i}{n}\right)^2\frac3n
=\int_0^3(1+x)^2dx
=\boxed{21}.
$$

若 $f$ 连续且
<!-- bilingual-en:start -->
If $f$ is continuous and
<!-- bilingual-en:end -->

$$
x\sin(\pi x)=\int_0^{x^2}f(t)dt,
$$

求导得
<!-- bilingual-en:start -->
Differentiating gives
<!-- bilingual-en:end -->

$$
\sin(\pi x)+\pi x\cos(\pi x)=2xf(x^2).
$$

代 \(x=2\)：
<!-- bilingual-en:start -->
Substitute \(x=2\):
<!-- bilingual-en:end -->

$$
2\pi=4f(4)\Longrightarrow\boxed{f(4)=\pi/2}.
$$

### Problem 4：三角形质心与 Pappus
<!-- bilingual-en:start -->
*Problem 4: Triangle Centroid and Pappus*
<!-- bilingual-en:end -->

设 $r,h>0$，均匀密度直角三角形的上边界为 $y=h-(h/r)x$，面积 $A=rh/2$；用[[质心积分计算]]：
<!-- bilingual-en:start -->
Let $r,h>0$. The uniform-density right triangle has upper boundary $y=h-(h/r)x$ and area $A=rh/2$. Its [[质心积分计算|centroid is calculated by integration]]:
<!-- bilingual-en:end -->

$$
\bar x=
\frac{\int_0^rx[h-(h/r)x]dx}{A}
=\frac{hr^2/6}{hr/2}
=\frac r3.
$$

水平切片同理给 $\bar y=h/3$。区域在旋转轴一侧且轴不穿过内部，可用[[Pappus体积定理]]；绕 $y$ 轴时质心走过 $2\pi r/3$：
<!-- bilingual-en:start -->
Horizontal slices similarly give $\bar y=h/3$. The region lies on one side of the axis and the axis does not cross its interior, so [[Pappus体积定理|Pappus's volume theorem]] applies. The centroid travels $2\pi r/3$ about the $y$-axis:
<!-- bilingual-en:end -->

$$
V=A\cdot\frac{2\pi r}{3}
=\boxed{\frac13\pi r^2h}.
$$

### Problem 5：Simpson 恒等式
<!-- bilingual-en:start -->
*Problem 5: Simpson's identity*
<!-- bilingual-en:end -->

在 $2n$ 个小区间、步长 $h=(b-a)/(2n)$ 的细网格上，令 $f_i=f(a+ih)$。$T_n,M_n$ 各用 $n$ 个粗区间，粗步长为 $2h$：
<!-- bilingual-en:start -->
On a fine grid with $2n$ subintervals and $h=(b-a)/(2n)$, write $f_i=f(a+ih)$. Both $T_n$ and $M_n$ use $n$ coarse subintervals of width $2h$:
<!-- bilingual-en:end -->

$$
T_n=h[f_0+2f_2+\cdots+2f_{2n-2}+f_{2n}],
$$

$$
M_n=2h[f_1+f_3+\cdots+f_{2n-1}].
$$

所以
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

$$
\frac13T_n+\frac23M_n
=\frac h3[f_0+4f_1+2f_2+\cdots+4f_{2n-1}+f_{2n}]
=\boxed{S_{2n}}.
$$

### Problem 6：盐水混合
<!-- bilingual-en:start -->
*Problem 6: Saltwater mixing*
<!-- bilingual-en:end -->

令 $s(t)$ 为盐量 kg。水箱初始体积为 $1000$ L，纯水流入与溶液流出均为 $10$ L/min，因此体积恒为 $1000$ L。充分混合保证流出液浓度等于箱内浓度 $s/1000$ kg/L：
<!-- bilingual-en:start -->
Let $s(t)$ be the salt mass in kilograms. The initial tank volume is $1000$ L; pure water enters and solution leaves at the same rate of $10$ L/min, so the volume remains $1000$ L. Thorough mixing makes the outflow concentration equal to the tank concentration $s/1000$ kg/L:
<!-- bilingual-en:end -->

$$
\frac{ds}{dt}=-10\frac{s}{1000}=-\frac{s}{100},
\qquad s(0)=15.
$$

用[[分离变量]]求解：
<!-- bilingual-en:start -->
Use [[分离变量|separation of variables]]:
<!-- bilingual-en:end -->

$$
\frac{ds}{s}=-\frac{dt}{100}
\Longrightarrow
\ln s=-\frac t{100}+C.
$$

所以
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

$$
\boxed{s(t)=15e^{-t/100}}.
$$

这是[[指数增长与衰减|指数衰减]]；半衰期为：
<!-- bilingual-en:start -->
This is [[指数增长与衰减|exponential decay]]; its half-life is
<!-- bilingual-en:end -->

$$
e^{-t/100}=\frac12
\Longrightarrow
\boxed{t=100\ln2\approx69.3\text{ min}}.
$$

> [!question]- Exam 3 收尾自检
> 1. 面积为何必须非负？2. Simpson 恒等式靠什么证明？3. 指数衰减半衰期怎样求？
>
> 答：几何面积是小条绝对高度累计；逐点展开并比较权重；解 \(e^{kt}=1/2\)。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** Why must geometric area be nonnegative?<br>
> **2.** How is Simpson's identity proved?<br>
> **3.** How is the half-life of exponential decay found?<br>
>
> Answer: Geometric area accumulates nonnegative strip heights; expand the formulas and compare the weights node by node; solve \(e^{kt}=1/2\).
> <!-- bilingual-en:end -->

## 全章总结
<!-- bilingual-en:start -->
*Chapter Summary*
<!-- bilingual-en:end -->

1. Riemann 和把有限近似升级为定积分；
2. FTC I 把定积分化为原函数端点差；
3. FTC II 说明连续密度的累计函数求导后回到该密度；
4. 面积、体积、质量、平均值和概率都是“密度 × 小尺度”的累计；
5. 没有初等原函数时仍可定义新函数并数值计算；
6. 应用题先明确小量、单位、积分方向和边界，再选择公式。
<!-- bilingual-en:start -->

&nbsp;
**1.** Riemann sums turn finite approximations into definite integrals.<br>
**2.** FTC I evaluates a definite integral as the endpoint difference of an antiderivative.<br>
**3.** FTC II differentiates the accumulation of a continuous density to recover that density.<br>
**4.** Area, volume, mass, averages, and probabilities all accumulate “density × a small scale.”<br>
**5.** Even without an elementary antiderivative, an integral can define a new function and be approximated numerically.<br>
**6.** In an application, first identify the small quantity, units, orientation, and boundaries; only then choose a formula.<br>
<!-- bilingual-en:end -->

> [!tip] 一遍读懂后的最低验收
> 不看公式表，能够从 Riemann 和写出积分、复述 FTC 的差商证明、画切片建立面积或体积、解释加权平均，并独立完成 Exam 3 六题。
> <!-- bilingual-en:start -->
> Without consulting a formula sheet, you should be able to turn a Riemann sum into an integral, reproduce the difference-quotient proof of the FTC, model an area or volume from a representative slice, explain a weighted average, and solve all six Exam 3 problems independently.
> <!-- bilingual-en:end -->

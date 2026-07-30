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

> [!abstract] 本章主线
> 导数解决“已知量，求瞬时变化率”；定积分反过来解决“已知变化率和初值，求累计变化”。本章从矩形面积的极限构造定积分，证明微积分基本定理把“求和”与“求导”连接起来，再把这一工具用于面积、体积、平均值、概率和数值计算。

## 阅读地图

- Part A：Session 43–50，定积分、Riemann 和与第一基本定理
- Problem Set 6
- Part B：Session 51–59，第二基本定理、面积与体积
- Problem Set 7
- Part C：Session 60–65，平均值、概率与数值积分
- Problem Set 8
- Exam 3：Session 66–67

## 统一记号

把 \([a,b]\) 分割为

$$
a=x_0<x_1<\cdots<x_n=b,
$$

第 \(i\) 段宽度为 \(\Delta x_i=x_i-x_{i-1}\)，样本点 \(x_i^*\in[x_{i-1},x_i]\)。[[Riemann Sum|黎曼和]]（Riemann sum）是

$$
\sum_{i=1}^{n}f(x_i^*)\Delta x_i.
$$

等分时 $(\Delta x=(b-a)/n)$。单位检查始终是“被积函数单位 × \(dx\) 单位”。

![[98_attachment/MIT18.01SC/unit03-riemann-sums.png]]

---

## Part A：Definition and First Fundamental Theorem

## Session 43：Definite Integrals

### 问题、定义与直觉

若速度 \(v(t)\) 在短时间 \(\Delta t\) 内近似不变，小段路程约为 \(v(t_i^*)\Delta t\)，总路程约为

$$
\sum v(t_i^*)\Delta t.
$$

面积、质量、成本和概率都是同一种结构：“密度 × 小宽度”再累计。

若随着最大分割宽度

$$
\|P\|=\max_i\Delta x_i\to0,
$$

所有 Riemann 和都趋于同一个有限值 \(I\)，由此定义[[Definite Integral|定积分]]

$$
\boxed{\int_a^bf(x)\,dx=I}.
$$

它是有向累计量：

$$
\int_a^af=0,\qquad \int_b^af=-\int_a^bf.
$$

负函数贡献负面积，因此定积分不总等于几何面积。

> [!note] 严格性边界
> 本课使用“闭区间上的连续函数可积”。完整证明依赖一致连续性；这里的重点是 Riemann 和极限及其后果。

### 本地材料

- [[Ses43a_Lecture_Notes.pdf|43a Introduction to Definite Integrals]]
- [[Ses43b_Lecture_Notes.pdf|43b Definition of the Definite Integral]]
- [[Exercise043_Problems.pdf|Exercise 43]] · [[Exercise043_Solutions.pdf|解答]]

> [!example]- Exercise 43：只凭图形估计积分
> 三幅图的纵横坐标刻度不同，必须先读刻度，再把阴影与可计算的三角形、矩形比较。
>
> 1. 第一幅阴影恰为底 (2)、高 (4) 的三角形，故
>    $$
>    \int f(x)\,dx=\frac12\cdot2\cdot4=\boxed{4},
>    $$
>    选 (b)。
> 2. 第二幅曲线下的面积小于包围它的三角形面积 \(1/2\)，但明显接近该值而非 \(1/4\)，最佳估计为 \(\boxed{1/2}\)，选 (c)。
> 3. 第三幅可粗分成面积约 \(1/2\) 的矩形和面积约 \(1/4\) 的补充部分，故最佳估计为 \(\boxed{3/4}\)，选 (b)。
>
> 这里不是要求精确积分，而是检查“面积必须与坐标尺度相容”。图形见[[Exercise043_Problems.pdf#page=1|原题第 1 页]]。

> [!question]- 三问自检
> 1. 为什么样本点可以任选？2. \(dx\) 对应什么？3. \(f\equiv c\) 时积分是多少？
>
> 答：可积要求选点差异在细分后消失；\(dx\) 来自 \(\Delta x_i\)；结果为 \(c(b-a)\)。

**知识链：**有限求和 → 分割变细 → Riemann 和极限 → 定积分。

## Session 44：Adding Areas of Rectangles

取 \(f(x)=x\)、区间 \([0,1]\)、右端点 \(i/n\)：

$$
R_n=\sum_{i=1}^n\frac{i}{n}\frac1n
=\frac{n(n+1)}{2n^2}
=\frac12+\frac1{2n}\to\frac12.
$$

左端点和为 \(L_n=\frac12-\frac1{2n}\)，二者从两侧夹住真实面积。

同理，使用

$$
\sum_{i=1}^ni^2=\frac{n(n+1)(2n+1)}6
$$

可得

$$
\int_0^1x^2dx
=\lim_{n\to\infty}\frac1{n^3}\sum_{i=1}^ni^2
=\frac13.
$$

> [!warning] 易错点
> 每项必须乘 $\Delta x$。只把高度相加，分割越细总和反而越大。

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
> 最后一式也可用等比数列公式 ((2^6-1)/(2-1))。易错点是把 (k=0) 项 (2^0=1) 漏掉。

> [!question]- 三问自检
> 1. 递增函数的右端点和为何高估？2. 中点和是否改变极限？3. \(\int_0^2x\,dx\) 是多少？
>
> 答：右端点是每段最大值；不改变；结果为 \(2\)。

## Session 45：Some Easy Integrals

先用几何和对称性：

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

由 Riemann 和逐项运算可证明线性、区间可加性和单调性：

$$
\int(cf+dg)=c\int f+d\int g,
$$

$$
\int_a^bf+\int_b^cf=\int_a^cf,
$$

$$
f\le g\Longrightarrow\int_a^bf\le\int_a^bg.
$$

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

> [!question]- 三问自检
> 1. \(\int_{-2}^2(x^3+4)dx\)？2. 为什么 \(\int|f|\ge|\int f|\)？3. 分点公式如何解释？
>
> 答：\(16\)；正负在 \(\int f\) 中会抵消；相邻区间累计量相加。

## Session 46：Riemann Sums

看到

$$
\lim_{n\to\infty}\sum_{i=1}^n
f\!\left(a+i\frac{b-a}{n}\right)\frac{b-a}{n},
$$

先识别 \(\Delta x=(b-a)/n\) 和右端点，再写成 \(\int_a^bf(x)dx\)。

若债务增长率为 \(r(t)\) 美元/年，则

$$
\Delta D=\int_{t_0}^{t_1}r(t)dt.
$$

单位由“美元/年 × 年”恢复成美元。本地 `Ses46b` 和 `Ses46c` 正文相同，只解释一次但保留两个来源。

### 本地材料

- [[Ses46a_Lecture_Notes.pdf|46a Riemann Sums]]
- [[Ses46b_Lecture_Notes.pdf|46b Cumulative Debt]]
- [[Ses46c_Lecture_Notes.pdf|46c Alternate File]]
- [[Exercise046_Problems.pdf|Exercise 46]] · [[Exercise046_Solutions.pdf|解答]]

> [!example]- Exercise 46：六个左端点矩形
> 估计 (int_0^2(3x+2)dx)。六等分给出 (Delta x=2/6=1/3)，左端点为
> (0,1/3,2/3,1,4/3,5/3)，相应高度为 (2,3,4,5,6,7)。因此
> $$
> L_6=\frac13(2+3+4+5+6+7)=\boxed{9}.
> $$
> 精确值为
> $$
> \left[\frac32x^2+2x\right]_0^2=10.
> $$
> 被积函数递增，所以每段左端点高度都是该段最小值，(L_6<10) 的方向与图形一致。

> [!question]- 三问自检
> 1. 怎样从 \(3/n\) 读区间长度？2. 平移 \(a\) 在哪里？3. 单位如何检查？
>
> 答：\(n(3/n)=3\)；样本点写成 \(a+i\Delta x\)；函数单位乘自变量单位。

## Session 47：Introduction to the FTC

定义累积函数

$$
F(x)=\int_a^xf(t)dt.
$$

上限从 \(x\) 移到 \(x+h\) 时，

$$
F(x+h)-F(x)=\int_x^{x+h}f(t)dt.
$$

连续性使小条面积近似 \(f(x)h\)，因此差商近似 \(f(x)\)。第一基本定理是：

> [!important] [[Fundamental Theorem of Calculus|微积分基本定理]]第一部分（FTC I）
> 若 \(f\) 在 \(x\) 附近连续，\(F(x)=\int_a^xf(t)dt\)，则
> $$
> \boxed{F'(x)=f(x)}.
> $$

![[98_attachment/MIT18.01SC/unit03-ftc-accumulator.png|819]]

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
> 第三题约为 (-0.293)。区间内 (sin x\le0)，所以负号不是计算错误，而是有向面积的必然结果。

> [!question]- 三问自检
> 1. 为什么积分变量写 \(t\)？2. \(f<0\) 时 \(F\) 怎样？3. \(F(a)\)？
>
> 答：\(t\) 是哑变量；\(F\) 递减；\(F(a)=0\)。

## Session 48：The Fundamental Theorem

### FTC I 的逐步证明

目标：

$$
\lim_{h\to0}\frac{F(x+h)-F(x)}h=f(x).
$$

由积分可加性，

$$
\frac{F(x+h)-F(x)}h
=\frac1h\int_x^{x+h}f(t)dt.
$$

减去目标值并把常数写入积分：

$$
\frac{F(x+h)-F(x)}h-f(x)
=\frac1h\int_x^{x+h}[f(t)-f(x)]dt.
$$

给定 \(\varepsilon>0\)。连续性保证存在 \(\delta>0\)，当 \(|t-x|<\delta\) 时，
\(|f(t)-f(x)|<\varepsilon\)。若 \(0<|h|<\delta\)，则

$$
\left|
\frac{F(x+h)-F(x)}h-f(x)
\right|
\le\frac1{|h|}\varepsilon|h|=\varepsilon.
$$

故差商极限为 \(f(x)\)。证明没有假设 \(f\ge0\)，因此负被积函数同样成立。

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
> 几何上，(sin x) 在区间上方贡献面积 (2)；(cos x) 在 ([0,\pi/2]) 与 ([\pi/2,\pi]) 的正负面积互相抵消。

> [!question]- 三问自检
> 1. FTC I 证明中连续性控制哪一个量？2. 证明为何也覆盖 \(h<0\)？3. \(\frac d{dx}\int_2^x f(t)dt\) 是什么？
>
> 答：控制 (|f(t)-f(x)|)；估计使用 (|h|)，反向积分的符号也包含在等式中；结果为 (f(x))。

**知识链：**连续性控制小区间内高度变化，区间长度正好约掉差商分母。

## Session 49：Applications of FTC

若

$$
G(x)=\int_a^{g(x)}f(t)dt,
$$

令 \(H(u)=\int_a^uf(t)dt\)，则 \(G=H\circ g\)。FTC 与链式法则给出

$$
\boxed{G'(x)=f(g(x))g'(x)}.
$$

上下限都变时：

$$
\frac d{dx}\int_{u(x)}^{v(x)}f(t)dt
=f(v(x))v'(x)-f(u(x))u'(x).
$$

例：

$$
\frac d{dx}\int_0^{x^2}\sin(t^3)dt
=2x\sin(x^6).
$$

无需先求 \(\sin(t^3)\) 的原函数。换元函数若不单调，要按单调区间拆开，避免重复覆盖。

### 本地材料

- [[Ses49a_Lecture_Notes.pdf|49a Estimation]]
- [[Ses49b_Lecture_Notes.pdf|49b FTC Example]]
- [[Ses49c_Lecture_Notes.pdf|49c Substitution When \(u'\) Changes Sign]]
- [[Exercise049_Problems.pdf|Exercise 49]] · [[Exercise049_Solutions.pdf|解答]]

> [!example]- Exercise 49：换元时同步更换积分限
> (a) 令 (u=3x+4)，则 (du=3dx)，且 (x=0,4) 对应 (u=4,16)：
> $$
> \int_0^4\sqrt{3x+4}\,dx
> =\frac13\int_4^{16}u^{1/2}du
> =\frac29[u^{3/2}]_4^{16}
> =\boxed{\frac{112}{9}}.
> $$
> (b) 令 (u=x^2+1)，(du=2x\,dx)，积分限 (2\to10)：
> $$
> \int_1^3\frac{x}{x^2+1}dx
> =\frac12\int_2^{10}\frac{du}{u}
> =\boxed{\frac12\ln5}.
> $$
> (c) 令 (u=\sin x)，(du=\cos x\,dx)，积分限 (0\to1)：
> $$
> \int_0^{\pi/2}\sin^5x\cos x\,dx
> =\int_0^1u^5du=\boxed{\frac16}.
> $$
> 一旦换成 (u) 的积分限，最后就不再代回 (x)；两套端点不可混用。

> [!question]- 三问自检
> 1. \(d/dx\int_x^0f(t)dt\)？2. \(d/dx\int_x^{x^2}e^{t^2}dt\)？3. 内层导数为何不能漏？
>
> 答：\(-f(x)\)；\(2xe^{x^4}-e^{x^2}\)；因为上限移动速度不一定为 1。

## Session 50：FTC and MVT

### [[Average Value of a Function|积分平均值]]定理及证明

若 \(f\) 在 \([a,b]\) 连续，令 \(F(x)=\int_a^xf(t)dt\)。FTC 给出 \(F'=f\)。对 \(F\) 用普通平均值定理，存在 \(c\in(a,b)\)：

$$
\frac{F(b)-F(a)}{b-a}=F'(c)=f(c).
$$

所以

$$
\boxed{\int_a^bf(x)dx=f(c)(b-a)}.
$$

若 \(m\le f\le M\)，积分保持不等式给出

$$
m(b-a)\le\int_a^bf\le M(b-a).
$$

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

> [!question]- 三问自检
> 1. 连续性在哪里使用？2. \(f(c)\) 是哪种平均？3. 不求原函数能否估计积分？
>
> 答：保证 MVT/FTC 可用且平均值被取得；连续平均值；能，用上下界。

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
> 2a 的各项为 (3,-5,7,-9,11,-13)。第 (n) 项的绝对值是 (2n+1)，符号由 ((-1)^{n+1}) 控制，因此
> $$
> \boxed{\sum_{n=1}^{6}(-1)^{n+1}(2n+1)}.
> $$
> 2b 是倒数平方和，首项 (1)、末项 (1/n^2)，故
> $$
> \boxed{\sum_{k=1}^{n}\frac1{k^2}}.
> $$
> 检查方法：分别代入指标的第一个和最后一个值，必须还原原和式。

> [!example]- 3B 3b、4a：上下和及其误差
> 对 (f(x)=x^2) 在 ([-1,3]) 作四等分，(Delta x=1)，节点为 (-1,0,1,2,3)。逐段比较可得
> $$
> L_4=1+0+1+4=\boxed{6},
> \qquad
> R_4=0+1+4+9=\boxed{14}.
> $$
> 因 (x^2) 在 ([-1,0]) 递减、在 ([0,3]) 递增，各段取最大、最小值后
> $$
> \boxed{U_4=15},\qquad \boxed{D_4=5}.
> $$
> 对 4a，(x^2) 在 ([0,b]) 单调递增（(b>0)）。每一小段的上、下矩形高度分别是右、左端点值，故望远镜相消：
> $$
> U_n-L_n
> =\frac bn\sum_{i=1}^n
> \left[\left(\frac{ib}{n}\right)^2-
> \left(\frac{(i-1)b}{n}\right)^2\right]
> =\frac bn(b^2-0)=\frac{b^3}{n}\to0.
> $$
> 所以上、下和夹到同一个积分值。

> [!example]- 3B 5：从极限识别 Riemann 和
> 题中
> $$
> \frac1n\sum_{i=1}^{n}\sin\frac{ib}{n}
> =\frac1b\sum_{i=1}^{n}\sin\frac{ib}{n}\frac bn.
> $$
> 右边是 ([0,b]) 上 (sin x) 的右端点和再乘 (1/b)，因此（(b\ne0)）
> $$
> \lim_{n\to\infty}\frac1n\sum_{i=1}^{n}\sin\frac{ib}{n}
> =\frac1b\int_0^b\sin x\,dx
> =\boxed{\frac{1-\cos b}{b}}.
> $$
> (b=0) 时原和每项均为零；右侧的连续延拓也趋于 (0)。

> [!example]- 3C 1、2a、3a、5a：FTC 与换元
> 1. 令 (u=x-2)，端点 (3,6) 变为 (1,4)：
>    $$
>    \int_3^6\frac{dx}{\sqrt{x-2}}
>    =\int_1^4u^{-1/2}du
>    =[2\sqrt u]_1^4=\boxed{2}.
>    $$
> 2a. 令 (u=3x+5)，(du=3dx)，端点 (5,11)：
>    $$
>    \int_0^2\sqrt{3x+5}\,dx
>    =\frac13\int_5^{11}u^{1/2}du
>    =\boxed{\frac29\left(11^{3/2}-5^{3/2}\right)}.
>    $$
> 3a. 令 (u=x^2+1)，(du=2x\,dx)，端点 (2,5)：
>    $$
>    \int_1^2\frac{x}{x^2+1}dx
>    =\frac12\int_2^5\frac{du}{u}
>    =\boxed{\frac12\ln\frac52}.
>    $$
> 5a. 一拱 (sin x) 位于 ([0,\pi]) 且非负，所以几何面积就是
>    $$
>    \int_0^\pi\sin x\,dx=[-\cos x]_0^\pi=\boxed{2}.
>    $$

> [!example]- 3E 6b、6c：不用求原函数的严格比较
> 6b. 在 ([0,\pi]) 上 (0\le\sin x\le1)，所以
> $$
> \sin^2x\le\sin x,
> $$
> 且除端点和 (x=\pi/2) 外严格小于。积分保持不等式，故
> $$
> \boxed{\int_0^\pi\sin^2x\,dx<\int_0^\pi\sin x\,dx=2}.
> $$
> 6c. 对 (x\in[10,20])，(sqrt{x^2+1}>x)，因此
> $$
> \boxed{\int_{10}^{20}\sqrt{x^2+1}\,dx>
> \int_{10}^{20}x\,dx
> =\frac{20^2-10^2}{2}=150}.
> $$

> [!example]- 4J 1、2：从“密度 × 小量”建立积分
> 1. 圆柱形孔直径 (1)，截面积为 (pi(1/2)^2=pi/4)。以 (y) 表示水被提升的距离，厚度 (dy) 的水层体积为 ((\pi/4)dy)，若单位体积提升单位距离需能量 (k)，则题目要求的“只列式”为
>    $$
>    \boxed{E=\frac{\pi k}{4}\int_0^{100}y\,dy}.
>    $$
>    因素 (y) 不能漏，它表示不同深度的水提升距离不同。
> 2. 初始放射性物质量为 (x_0)，时刻 (t) 剩余 (x_0e^{-kt})，每单位物质的计数率为 (r)。一分钟内的计数率是 (rx_0e^{-kt})，故一小时（(0\le t\le60)）总计数为
>    $$
>    R=\int_0^{60}rx_0e^{-kt}dt
>    =\boxed{\frac{rx_0}{k}\left(1-e^{-60k}\right)}.
>    $$

---

## Part B：Second Fundamental Theorem, Areas and Volumes

## Session 51：The Second Fundamental Theorem

> [!important] FTC II
> 若 \(f\) 连续且 \(F'=f\)，则
> $$
> \boxed{\int_a^bf(x)dx=F(b)-F(a)}.
> $$

使用流程：找原函数 → 写 \([F]_a^b\) → 上限值减下限值 → 检查符号与单位。定积分不写 \(+C\)，因为常数在端点差中抵消。

### 本地材料

- [[Ses51a_Lecture_Notes.pdf|51a Second FTC]]
- [[Ses51b_Lecture_Notes.pdf|51b Using Second FTC]]

> [!question]- 三问自检
> 1. 为什么无 \(+C\)？2. \(\int_1^3(2x+1)dx\)？3. 原函数不唯一为何结果唯一？
>
> 答：常数抵消；\(10\)；任意两原函数只差常数。

## Session 52：[[Fundamental Theorem of Calculus Proof|微积分基本定理证明]]

令

$$
A(x)=\int_a^xf(t)dt.
$$

FTC I 给出 \(A'=f\)。若 \(F'=f\)，则

$$
(A-F)'=0.
$$

由平均值定理，导数恒零的函数是常数，所以 \(A-F=C\)。代入 \(x=a\)：

$$
0-F(a)=C.
$$

于是

$$
A(x)=F(x)-F(a).
$$

取 \(x=b\) 得

$$
\int_a^bf=F(b)-F(a).
$$

### 本地材料

- [[Ses52a_Lecture_Notes.pdf|52a Proof of Second FTC]]
- [[Ses52b_Lecture_Notes.pdf|52b Proof of First FTC]]

**知识链：**FTC I 造出一个原函数；MVT 说明所有原函数只差常数；因此端点差就是累计量。

> [!question]- 三问自检
> 1. 证明中为何比较 \(A-F\)？2. 导数恒零为何推出常数？3. 常数怎样确定？
>
> 答：二者导数都等于 \(f\)；由 MVT；代入 \(x=a\) 使用 \(A(a)=0\)。

## Session 53：New Functions From Old

即使 \(e^{-x^2}\) 没有初等原函数，仍可定义

$$
E(x)=\int_0^xe^{-t^2}dt,
\qquad E'(x)=e^{-x^2}.
$$

对 \(x>0\) 定义

$$
L(x)=\int_1^x\frac{dt}{t}.
$$

则 \(L'(x)=1/x\)、\(L(1)=0\)。下一节证明 \(L\) 具有对数的乘法性质。

### 本地材料

- [[Ses53a_Lecture_Notes.pdf|53a Antiderivative of \(1/x\)]]
- [[Ses53b_Lecture_Notes.pdf|53b Bell Curve]]

> [!question]- 三问自检
> 1. \(E\) 奇偶性？2. \(L\) 定义域为何正数？3. 无初等原函数是否等于积分不存在？
>
> 答：\(E\) 为奇函数；\(1/t\) 在 0 奇异；不等于，定积分仍能定义新函数。

## Session 54：FTC and \(\ln x\)

固定 \(y>0\)，令

$$
G(x)=L(xy)-L(x).
$$

求导：

$$
G'(x)=\frac{y}{xy}-\frac1x=0.
$$

所以 \(G\) 为常数。代 \(x=1\) 得 \(G(1)=L(y)\)，从而

$$
\boxed{L(xy)=L(x)+L(y)}.
$$

又因 \(L'(x)=1/x>0\)，\(L\) 严格递增并有反函数；这个反函数就是 \(e^x\)。

### 本地材料

- [[Ses54a_Lecture_Notes.pdf|54a Integral Definition of Log]]
- [[Ses54b_Lecture_Notes.pdf|54b Log of a Product]]
- [[Exercise054_Problems.pdf|Exercise 54]] · [[Exercise054_Solutions.pdf|解答]]

> [!example]- Exercise 54：面积形式的幂法则
> 设 (x>1)、(a>0)。题目要证明
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
> 因而两边相等。几何上，从 (x^{-a}) 到 (1) 的 (1/t) 曲线下面积是从 (x^{-1}) 到 (1) 面积的 (a) 倍。固定 (x>1) 并令 (a\to\infty)，下限 (x^{-a}\to0^+)，右侧面积 (a\ln x\to\infty)；这也说明
> $$
> \int_{0}^{1}\frac{dt}{t}
> $$
> 在 (0) 附近发散。

> [!question]- 三问自检
> 1. 为什么固定 \(y\) 后研究 \(G(x)\)？2. \(G'=0\) 给出什么？3. \(L'(x)>0\) 有何用途？
>
> 答：把二元恒等式化成单变量；\(G\) 为常数；保证 \(L\) 一一对应并存在反函数。

## Session 55：Creating New Functions

若

$$
F(x)=\int_a^xf(t)dt,
$$

则 \(F'=f\)、\(F''=f'\)。因此从 \(f\) 的图像直接判断 \(F\)：

- \(f>0\)：\(F\) 递增；
- \(f<0\)：\(F\) 递减；
- \(f=0\) 且变号：\(F\) 可能取极值；
- \(f'>0\)：\(F\) 向上凹。

标准正态累计函数

$$
\Phi(x)=\frac1{\sqrt{2\pi}}\int_{-\infty}^xe^{-t^2/2}dt
$$

也是“由旧函数造新函数”。

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
> 原函数在正半轴上为 (-e^{-\lambda x})。因此：
>
> - 若 (b\ge a>0)，
>   $$
>   \int_a^bf(x)dx=[-e^{-\lambda x}]_a^b
>   =\boxed{e^{-\lambda a}-e^{-\lambda b}}.
>   $$
> - 若 (a\le0<b)，负半轴贡献为零，
>   $$
>   \int_a^bf(x)dx=\int_0^b\lambda e^{-\lambda x}dx
>   =\boxed{1-e^{-\lambda b}}.
>   $$
> - 若 \(a\le b\le0\)，则 \(\boxed{\int_a^bf(x)dx=0}\)。
>
> 三种情形必须由 (0) 是否落在积分区间内决定；不能跨过分段点仍套同一个原函数。

> [!question]- 三问自检
> 1. \(f>0\) 时累积函数怎样？2. \(f'=0\) 对累积函数意味着什么？3. 为什么不必知道累积函数显式式子？
>
> 答：递增；可能是凹凸性改变的候选点；FTC 已把其导数直接写成 \(f\)。

## Session 56：Geometric Interpretation

若 \(f\ge g\)，两曲线面积为

$$
\boxed{A=\int_a^b[f(x)-g(x)]dx}.
$$

若上下关系改变，在交点处分段。若曲线自然写成 \(x=R(y)\)、\(x=L(y)\)，则

$$
A=\int_c^d[R(y)-L(y)]dy.
$$

选择 \(dx\) 或 \(dy\) 的原则是减少反解、分段与绝对值。

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
> 令二者相等：(2x^4-x^2-1=0)。设 (u=x^2)，则
> ((2u+1)(u-1)=0)，实交点为 (x=\pm1)。在 (x=0) 处，上曲线是 (-1/2)，下曲线是 (-1)，所以
> $$
> \begin{aligned}
> A
> &=\int_{-1}^{1}\left[\left(\frac12x^2-\frac12\right)-(x^4-1)\right]dx\\
> &=2\int_0^1\left(-x^4+\frac12x^2+\frac12\right)dx\\
> &=2\left[-\frac{x^5}{5}+\frac{x^3}{6}+\frac{x}{2}\right]_0^1
> =\boxed{\frac{14}{15}}.
> \end{aligned}
> $$
> 被积函数在 ([-1,1]) 非负；偶对称使计算减半。

> [!question]- 三问自检
> 1. 为何必须“上减下”？2. 何时改用 \(dy\)？3. 曲线交叉时怎么办？
>
> 答：小条高度必须非负；水平切片能减少反解或分段时；在交点拆分并重新判断次序。

## Session 57：How to Calculate Volumes

[[Solids of Revolution|旋转体]]及一般切片体积都从截面积开始。若截面积为 \(A(x)\)，

$$
\boxed{V=\int_a^bA(x)dx}.
$$

圆盘/垫圈法：

$$
V=\pi\int_a^b[R(x)^2-r(x)^2]dx.
$$

圆柱壳法：

$$
V=2\pi\int_a^bx\,h(x)dx.
$$

washer 切片垂直于旋转轴，shell 切片平行于旋转轴。先画代表性切片，再决定方法。

![[98_attachment/MIT18.01SC/unit03-volume-methods.png|809]]

### 本地材料

- [[Ses57a_Lecture_Notes.pdf|57a Volumes by Slicing]]
- [[Ses57b_Lecture_Notes.pdf|57b Solids of Revolution]]
- [[Exercise057_Problems.pdf|Exercise 57]] · [[Exercise057_Solutions.pdf|解答]]

> [!example]- Exercise 57：绕 \(x\) 轴的两个旋转体
> (a) (y=3x-x^2) 与 (y=0) 在 (x=0,3) 相交。圆盘半径为 (3x-x^2)：
> $$
> \begin{aligned}
> V
> &=\pi\int_0^3(3x-x^2)^2dx
> =\pi\int_0^3(9x^2-6x^3+x^4)dx\\
> &=\pi\left[3x^3-\frac32x^4+\frac{x^5}{5}\right]_0^3
> =\boxed{\frac{81\pi}{10}}.
> \end{aligned}
> $$
> (b) 对 (y=\sqrt{ax})、(0\le x\le a)（(a>0)），圆盘半径平方为 (y^2=ax)：
> $$
> V=\pi\int_0^aax\,dx
> =\pi a\left[\frac{x^2}{2}\right]_0^a
> =\boxed{\frac{\pi a^3}{2}}.
> $$
> 两题答案都具有长度三次的量纲；若把半径而不是半径平方放入圆盘公式，量纲会立刻暴露错误。

> [!question]- 三问自检
> 1. washer 与旋转轴方向关系？2. shell 的周长因子？3. 为什么先画小切片？
>
> 答：切片垂直于轴；\(2\pi r\)；切片直接决定半径、高度、厚度与积分变量。

## Session 58：Volume of a Sphere

半径 \(R\) 的圆满足 \(y^2=R^2-x^2\)。绕 \(x\) 轴：

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
> 椭圆 (x^2+4y^2=4) 的纵坐标范围是 ([-1,1])。绕 (y) 轴旋转时，水平截面是半径
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

> [!question]- 三问自检
> 1. 截面积是什么？2. 为何上下半圆不分别相加？3. 单位为何是三次？
>
> 答：\(\pi(R^2-x^2)\)；旋转后同一圆盘只算一次；面积乘厚度。

## Session 59：Volume of a Paraboloid

若水平圆盘半径为 \(R\sqrt{y/h}\)，则

$$
\begin{aligned}
V
&=\pi\int_0^hR^2\frac yh\,dy\\
&=\frac{\pi R^2}{h}\left[\frac{y^2}{2}\right]_0^h\\
&=\boxed{\frac12\pi R^2h}.
\end{aligned}
$$

量纲警告：若 \(x,y\) 带长度单位，\(y=x^2\) 通常不合法，应写 \(y=kx^2\)，其中 \(k\) 带 \(1/\text{length}\) 单位。

### 本地材料

- [[Ses59a_Lecture_Notes.pdf|59a Paraboloid]]
- [[Ses59b_Lecture_Notes.pdf|59b Units Warning]]
- [[Exercise059_Problems.pdf|Exercise 59]] · [[Exercise059_Solutions.pdf|解答]]

> [!example]- Exercise 59：绕外部竖直轴的垫圈
> 区域由 (y=0)、(x=4)、(y=\sqrt x) 围成，绕 (x=6) 旋转。用水平切片最自然：(x=y^2)，且 (0\le y\le2)。外半径来自左边界 (x=y^2)，内半径来自右边界 (x=4)：
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
> “外减内”发生在平方以后：(R^2-r^2)，不是 ((R-r)^2)。

> [!question]- 三问自检
> 1. 抛物面体积是同底同高圆柱的几分之几？2. 为什么用 \(dy\)？3. \(y=kx^2\) 中 \(k\) 的量纲？
>
> 答：\(1/2\)；水平圆盘半径容易表示；\(1/\text{length}\)。

## Problem Set 7

- 官方范围：4B 2（使用 1e、1g）、5；4C 2、3；4J 3
- [[PSet04_Problems.pdf|Problems]]
- [[PSet04_Solutions.pdf|Solutions]]

> [!example]- 完整官方题册与逐题答案
> ![[PSet04_Problems.pdf#height=650]]
>
> ![[PSet04_Solutions.pdf#height=650]]

> [!example]- 4B 2：对 1e、1g 使用垫圈法
> 1e 的区域由 (y=2x-x^2) 与 (y=0) 围成，并绕 (y) 轴旋转。为使用垂直于旋转轴的水平垫圈，解出
> $$
> x=1\pm\sqrt{1-y},\qquad 0\le y\le1.
> $$
> 外、内半径分别为 (1+\sqrt{1-y}) 与 (1-\sqrt{1-y})，故
> $$
> \begin{aligned}
> V
> &=\pi\int_0^1\left[(1+\sqrt{1-y})^2-(1-\sqrt{1-y})^2\right]dy\\
> &=4\pi\int_0^1\sqrt{1-y}\,dy
> =\boxed{\frac{8\pi}{3}}.
> \end{aligned}
> $$
> 1g 的区域由 (y^2=ax)、(y=0)、(x=a) 围成（(a>0)），绕 (y) 轴旋转。对 (0\le y\le a)，外半径为 (a)，内半径为 (y^2/a)：
> $$
> \begin{aligned}
> V
> &=\pi\int_0^a\left[a^2-\left(\frac{y^2}{a}\right)^2\right]dy\\
> &=\pi\left[a^2y-\frac{y^5}{5a^2}\right]_0^a
> =\boxed{\frac{4\pi a^3}{5}}.
> \end{aligned}
> $$

> [!example]- 4B 5：等边三角形绕一边旋转
> 设边长为 (a)，取旋转边为 (x) 轴。三角形高度为 (sqrt3a/2)，利用关于中线的对称性，左半边界可写成 (y=\sqrt3x)，(0\le x\le a/2)。圆盘法给出
> $$
> \begin{aligned}
> V
> &=2\pi\int_0^{a/2}(\sqrt3x)^2dx
> =6\pi\left[\frac{x^3}{3}\right]_0^{a/2}
> =\boxed{\frac{\pi a^3}{4}}.
> \end{aligned}
> $$
> 系数 (2) 来自左右两半；少乘它会只算出半个旋转体。

> [!example]- 4C 2、3：圆柱壳与垫圈互相核对
> 2. 区域 (0\le y\le x^2)、(0\le x\le1) 绕 (y) 轴。半径为 (x)、壳高为 (x^2)：
>    $$
>    V=2\pi\int_0^1x(x^2)dx
>    =2\pi\left[\frac{x^4}{4}\right]_0^1
>    =\boxed{\frac\pi2}.
>    $$
> 3. 区域 (sqrt x\le y\le1)、(x\ge0) 绕 (y) 轴。用壳法时 (0\le x\le1)，壳高为 (1-\sqrt x)：
>    $$
>    V=2\pi\int_0^1x(1-\sqrt x)dx
>    =2\pi\left(\frac12-\frac25\right)
>    =\boxed{\frac\pi5}.
>    $$
>    反解 (x=y^2) 后，用圆盘法也得
>    $$
>    V=\pi\int_0^1(y^2)^2dy=\frac\pi5,
>    $$
>    两种切片的一致性完成核对。

> [!example]- 4J 3：反射池中的总物质量
> 池深为 (D)、半径为 (R)，距中心 (r) 处浓度为 (k/(1+r^2))。半径 (r)、厚度 (dr) 的圆环体积是
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

---

## Part C：Average Value, Probability and Numerical Integration

## Session 60：Integrals and Averages

连续平均值：

$$
\boxed{f_{\mathrm{avg}}=\frac1{b-a}\int_a^bf(x)dx}.
$$

因为

$$
\int_a^bf=f_{\mathrm{avg}}(b-a),
$$

它是与原有向面积相同的等高矩形高度。沿曲线弧长平均时，

$$
ds=\sqrt{1+(y')^2}dx,
\qquad
q_{\mathrm{avg}}=\frac{\int_Cq\,ds}{\int_C1\,ds}.
$$

### 本地材料

- [[Ses60a_Lecture_Notes.pdf|60a Average Value]]
- [[Ses60b_Lecture_Notes.pdf|60b Average Height]]
- [[Ses60c_Lecture_Notes.pdf|60c Arc-length Average]]
- [[Exercise060_Problems.pdf|Exercise 60]] · [[Exercise060_Solutions.pdf|解答]]

> [!example]- Exercise 60：复利账户的时间平均余额
> 若 (A(t)=A_0e^{rt})，在 (0\le t\le T) 上的平均余额为
> $$
> \begin{aligned}
> A_{\mathrm{avg}}
> &=\frac1T\int_0^TA_0e^{rt}dt
> =\frac{A_0}{T}\left[\frac{e^{rt}}r\right]_0^T\\
> &=\boxed{\frac{A_0}{rT}(e^{rT}-1)}.
> \end{aligned}
> $$
> 代入 (A_0=100)、(r=0.05)、(T=1)，
> $$
> A_{\mathrm{avg}}=2000(e^{0.05}-1)\approx\boxed{102.54}.
> $$
> 它位于初值 (100) 与终值 (100e^{0.05}\approx105.13) 之间，符合单调增长函数的平均值范围。

> [!question]- 三问自检
> 1. 连续平均值为何除以 \(b-a\)？2. 平均值是否一定被函数取得？3. 沿曲线平均为何使用 \(ds\)？
>
> 答：除以总权重；连续时由积分平均值定理取得；等权对象是弧长而非水平投影。

## Session 61：Weighted Averages

若权重密度 \(w(x)\ge0\)，

$$
\boxed{\bar f=\frac{\int_a^bf(x)w(x)dx}{\int_a^bw(x)dx}}.
$$

细杆密度 \(\rho\) 时：

$$
M=\int\rho(x)dx,\qquad
\bar x=\frac1M\int x\rho(x)dx.
$$

“Boiling Cauldron” 的关键是先明确按长度、面积、体积还是质量抽样；普通算术平均往往用了错误权重。

### 本地材料

- [[Ses61a_Lecture_Notes.pdf|61a Weighted Averages]]
- [[Ses61b_Lecture_Notes.pdf|61b Boiling Cauldron]]
- [[Ses61c_Lecture_Notes.pdf|61c Continued]]
- [[Exercise061_Problems.pdf|Exercise 61]] · [[Exercise061_Solutions.pdf|解答]]

> [!example]- Exercise 61：平面区域的质心
> 区域由 (x=-1)、(x=3)、(y=(x-1)^2) 与 (y=4) 围成。它关于 (x=1) 对称，所以
> $$
> \boxed{\bar x=1}.
> $$
> 用竖直切片求面积：
> $$
> A=\int_{-1}^{3}\left[4-(x-1)^2\right]dx=\frac{32}{3}.
> $$
> 求 \(\bar y\) 时改用水平切片更简洁。由 \(y=(x-1)^2\) 得同一高度处宽度
> (x_{\rm right}-x_{\rm left}=2\sqrt y)，其中 (0\le y\le4)。因此
> $$
> \bar y
> =\frac{\int_0^4 y(2\sqrt y)dy}{\int_0^4 2\sqrt y\,dy}
> =\frac{\left[\frac45y^{5/2}\right]_0^4}
> {\left[\frac43y^{3/2}\right]_0^4}
> =\frac{128/5}{32/3}
> =\boxed{\frac{12}{5}}.
> $$
> 故质心为 \(\boxed{(1,12/5)}\)，且确实落在区域的对称轴上。

> [!question]- 三问自检
> 1. 分母 \(\int w\) 表示什么？2. \(w\) 已是概率密度时分母？3. 质心公式的分子是什么？
>
> 答：总权重；等于 1；关于原点的一阶矩 \(\int xw(x)dx\)。

## Session 62：Integrals and Probability

[[Probability Density Function|概率密度]] \(p\) 满足

$$
p(x)\ge0,\qquad \int_{-\infty}^{\infty}p(x)dx=1.
$$

$$
P(a\le X\le b)=\int_a^bp(x)dx,
$$

$$
E[X]=\int xp(x)dx,
\qquad
\operatorname{Var}(X)=\int(x-\mu)^2p(x)dx.
$$

密度可以大于 1；真正受限在 \([0,1]\) 的是面积代表的概率。连续模型中单点概率为零。

### 本地材料

- [[Ses62a_Lecture_Notes.pdf|62a Probability Example]]
- [[Ses62b_Lecture_Notes.pdf|62b Summary]]
- [[Ses62c_Lecture_Notes.pdf|62c Errata]]
- [[Ses62d_Lecture_Notes.pdf|62d Extended Example]]

> [!question]- 三问自检
> 1. 密度能大于 1 吗？2. 对称密度均值？3. 为什么单点概率为零？
>
> 答：能；若期望存在则为 0；零宽区间的积分为 0。

## Session 63：Numerical Integration

[[Numerical Integration Methods|数值积分]]取步长 \(h=(b-a)/n\)：

$$
L_n=h\sum_{i=0}^{n-1}f(x_i),\quad
R_n=h\sum_{i=1}^{n}f(x_i),
$$

$$
M_n=h\sum_{i=1}^{n}f\!\left(\frac{x_{i-1}+x_i}{2}\right),
$$

$$
\boxed{
T_n=\frac h2\left[f(x_0)+2\sum_{i=1}^{n-1}f(x_i)+f(x_n)\right]
}.
$$

Session 63 第四段误放在 Problem Sets：

$$
\boxed{
S_{2n}=\frac h3[f(x_0)+4f(x_1)+2f(x_2)+\cdots+4f(x_{2n-1})+f(x_{2n})]
}.
$$

### 本地材料

- [[Ses63a_Lecture_Notes.pdf|63a Introduction]]
- [[Ses63b_Lecture_Notes.pdf|63b Riemann Sums]]
- [[Ses63c_Lecture_Notes.pdf|63c Trapezoidal Rule]]
- [[Ses63d_Problems.pdf|63d Simpson's Rule]]
- [[Exercise063_Problems.pdf|Exercise 63]] · [[Exercise063_Solutions.pdf|解答]]

> [!example]- Exercise 63：两种数值近似与精确值
> 对 (f(x)=x^3-2x) 在 ([-1,2]) 上积分，题目给出的梯形法取 (n=6)，步长 (h=3/6=1/2)。代入节点 (-1,-1/2,0,1/2,1,3/2,2) 后，
> $$
> T_6=\frac h2\left[f(-1)+2\sum_{i=1}^{5}f(x_i)+f(2)\right]
> =\boxed{0.93750}.
> $$
> 第二种 Riemann 和在每段取相对位置 (0.5)，即中点法；(n=12)、(h=3/12=1/4)，代入十二个中点得到
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
> 中点近似的绝对误差约 (0.02344)，小于梯形近似的 (0.18750)。不要用“梯形用了两倍节点”解释误差：相邻梯形共享端点，且方法的精度取决于误差阶与函数曲率。

> [!question]- 三问自检
> 1. \(n\) 个小区间有多少端点？2. 梯形法为何是左右和平均？3. Simpson 法为何要求偶数个小区间？
>
> 答：\(n+1\)；每段梯形面积是两端矩形面积平均；每两个小区间拟合一条抛物线。

![[98_attachment/MIT18.01SC/unit03-numerical-integration.png|897]]

## Session 64：Numerical Integration, Continued

对 \(f(x)=1/x\)，\(f''(x)>0\)，图像凸，弦位于曲线上方，因此梯形法高估 \(\ln2=\int_1^2dx/x\)。

在同一组 \(2n\) 网格点上：

$$
\boxed{\frac13T_n+\frac23M_n=S_{2n}}.
$$

展开权重即可证明：端点权重为 \(1\)，偶数内点为 \(2\)，奇数中点为 \(4\)。

- 梯形法主要误差由 \(f''\) 决定；
- Simpson 法对三次及以下多项式精确，主要误差与 \(f^{(4)}\) 有关。

### 本地材料

- [[Ses64a_Lecture_Notes.pdf|64a Trapezoid and \(\ln2\)]]
- [[Ses64b_Lecture_Notes.pdf|64b Simpson]]
- [[Ses64c_Lecture_Notes.pdf|64c Study Tips]]
- [[Exercise064_Problems.pdf|Exercise 64]] · [[Exercise064_Solutions.pdf|解答]]

> [!example]- Exercise 64：Simpson 法估计身高概率
> 题册给出的模型（(x) 以英寸计）是
> $$
> h(x)=\frac1{2.8\sqrt{2\pi}}
> e^{-(x-69)^2/5.6}.
> $$
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
> P(60\le X\le72)
> &\approx\frac23[h(60)+4h(62)+2h(64)+4h(66)\\
> &\qquad\qquad+2h(68)+4h(70)+h(72)]\\
> &\approx\boxed{0.574}.
> \end{aligned}
> $$
> 即约 \(57.4\%\)。
>
> (b) 八英尺是 \(96\) 英寸。严格概率是 \(\int_{96}^{\infty}h(x)dx\)；官方解答以 \(100\) 英寸截断，因为此后函数已极小。取节点 \(96,98,100\)：
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
> 在该模型下可视为零。原解答中写成 \(\int_8^\infty\) 是英尺与英寸混写；与密度自变量一致的下限应是 \(96\)。

> [!question]- 三问自检
> 1. 凸函数的梯形法为何高估？2. Simpson 权重怎样排列？3. 哪阶导数控制 Simpson 主误差？
>
> 答：弦位于图像上方；\(1,4,2,\ldots,4,1\)；四阶导数。

## Session 65：Bell Curve, Conclusion

标准正态密度

$$
\phi(x)=\frac1{\sqrt{2\pi}}e^{-x^2/2}
$$

总面积为 1。它没有初等原函数，但可通过数值积分求概率。因 \(\phi\) 为偶函数，

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

> [!example]- 4D 3：速度的函数平均等于平均速度
> 若位置 (s(t)) 可导且 (v(t)=s'(t))，则 (v) 在 ([a,b]) 上的函数平均为
> $$
> v_{\rm avg}
> =\frac1{b-a}\int_a^bv(t)dt
> =\frac1{b-a}\int_a^bs'(t)dt.
> $$
> 由 FTC II，
> $$
> \boxed{v_{\rm avg}=\frac{s(b)-s(a)}{b-a}},
> $$
> 右边正是“总位移 ÷ 总时间”的平均速度。这里用的是位移而非总路程；若速度变号，两者不能混为一谈。

> [!example]- 4D 5：由区间平均恢复原函数
> 已知 (f) 在 ([0,x]) 上的平均值为 (g(x))，即对 (x>0)
> $$
> g(x)=\frac1x\int_0^xf(t)dt.
> $$
> 先乘以 (x)：
> $$
> xg(x)=\int_0^xf(t)dt.
> $$
> 两边对 (x) 求导。左边用乘积法则，右边用 FTC I：
> $$
> g(x)+xg'(x)=f(x).
> $$
> 因此
> $$
> \boxed{f(x)=g(x)+xg'(x)}.
> $$
> 易错点是直接对 (g=(1/x)\int_0^xf) 求导却漏掉 (1/x) 的导数；先清除分母最稳妥。

---

## Exam 3

## Session 66：Review for Exam 3

以下四份讲义虽放在 Unit 4 目录，实际属于 Exam 3 复习：

- [[Ses66a_Lecture_Notes.pdf|66a Questions on Test 3]]
- [[Ses66b_Lecture_Notes.pdf|66b Types of Riemann Sums]]
- [[Ses66c_Lecture_Notes.pdf|66c Asymptotes of Antiderivatives]]
- [[Ses66d_Lecture_Notes.pdf|66d Choosing a Technique]]

### 考前检查

1. 从和式读出区间、\(\Delta x\)、样本点和被积函数；
2. 区分“FTC 求导”和“原函数端点求值”；
3. 面积选择 \(dx/dy\)，体积先画切片；
4. 平均值和概率写正确权重；
5. 数值法不混淆节点数与区间数；
6. 混合问题写“流入率 − 流出率”及初值。

> [!question]- 三问自检
> 1. Riemann 和最先找什么？2. 旋转体最先画什么？3. 混合模型浓度怎样写？
>
> 答：\(\Delta x\) 与样本点；代表性切片；当前溶质量除以当前总体积。

## Session 67：Materials for Exam 3

- [[Exam3_Problems.pdf|Exam 3 Problems]]
- [[Exam3_Solutions.pdf|Exam 3 Official Solutions]]

### Problem 1：两曲线面积

令

$$
y^2-4y=2y-y^2
\Longrightarrow2y(y-3)=0,
$$

故积分限为 \(0,3\)。右减左：

$$
\begin{aligned}
A
&=\int_0^3[(2y-y^2)-(y^2-4y)]dy\\
&=\left[3y^2-\frac23y^3\right]_0^3\\
&=\boxed{9}.
\end{aligned}
$$

### Problem 2：绕 \(y=-1\) 的体积

交点 \(e^x=2\) 给 \(x=\ln2\)。外半径 \(3\)，内半径 \(e^x+1\)：

$$
\boxed{V=\pi\int_0^{\ln2}[9-(1+e^x)^2]dx}.
$$

### Problem 3：Riemann 和与 FTC

$$
\lim_{n\to\infty}\sum_{i=1}^n
\left(1+\frac{3i}{n}\right)^2\frac3n
=\int_0^3(1+x)^2dx
=\boxed{21}.
$$

若

$$
x\sin(\pi x)=\int_0^{x^2}f(t)dt,
$$

求导得

$$
\sin(\pi x)+\pi x\cos(\pi x)=2xf(x^2).
$$

代 \(x=2\)：

$$
2\pi=4f(4)\Longrightarrow\boxed{f(4)=\pi/2}.
$$

### Problem 4：三角形质心与 Pappus

直角三角形上边界 \(y=h-(h/r)x\)，面积 \(A=rh/2\)：

$$
\bar x=
\frac{\int_0^rx[h-(h/r)x]dx}{A}
=\frac{hr^2/6}{hr/2}
=\frac r3.
$$

水平切片同理给 \(\bar y=h/3\)。绕 \(y\) 轴时质心走过 \(2\pi r/3\)：

$$
V=A\cdot\frac{2\pi r}{3}
=\boxed{\frac13\pi r^2h}.
$$

### Problem 5：Simpson 恒等式

在 \(2n\) 个小区间、步长 \(h=(b-a)/(2n)\) 上，

$$
T_n=h[f_0+2f_2+\cdots+2f_{2n-2}+f_{2n}],
$$

$$
M_n=2h[f_1+f_3+\cdots+f_{2n-1}].
$$

所以

$$
\frac13T_n+\frac23M_n
=\frac h3[f_0+4f_1+2f_2+\cdots+4f_{2n-1}+f_{2n}]
=\boxed{S_{2n}}.
$$

### Problem 6：盐水混合

令 \(s(t)\) 为盐量 kg。浓度 \(s/1000\) kg/L，流出 \(10\) L/min，流入纯水：

$$
\frac{ds}{dt}=-10\frac{s}{1000}=-\frac{s}{100},
\qquad s(0)=15.
$$

分离变量：

$$
\frac{ds}{s}=-\frac{dt}{100}
\Longrightarrow
\ln s=-\frac t{100}+C.
$$

所以

$$
\boxed{s(t)=15e^{-t/100}}.
$$

半衰期：

$$
e^{-t/100}=\frac12
\Longrightarrow
\boxed{t=100\ln2\approx69.3\text{ min}}.
$$

> [!question]- Exam 3 收尾自检
> 1. 面积为何必须非负？2. Simpson 恒等式靠什么证明？3. 指数衰减半衰期怎样求？
>
> 答：几何面积是小条绝对高度累计；逐点展开并比较权重；解 \(e^{kt}=1/2\)。

## 全章总结

1. Riemann 和把有限近似升级为定积分；
2. FTC I 说明累计函数的导数是当前密度；
3. FTC II 把定积分化为原函数端点差；
4. 面积、体积、质量、平均值和概率都是“密度 × 小尺度”的累计；
5. 没有初等原函数时仍可定义新函数并数值计算；
6. 应用题先明确小量、单位、积分方向和边界，再选择公式。

> [!tip] 一遍读懂后的最低验收
> 不看公式表，能够从 Riemann 和写出积分、复述 FTC 的差商证明、画切片建立面积或体积、解释加权平均，并独立完成 Exam 3 六题。

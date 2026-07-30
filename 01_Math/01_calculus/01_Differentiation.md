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

- 课程来源：[MIT OpenCourseWare 18.01SC - Unit 1: Differentiation](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/pages/1.-differentiation/)
- 教师：David Jerison；学期：Fall 2010。
- 官方顺序：Part A（Session 1-12）→ Problem Set 1 → Part B（Session 13-20）→ Problem Set 2 → Exam 1（Session 21-22）。
- 本地材料说明：同一 Session 的 `a/b/c...` 是视频片段顺序；笔记正文按这个顺序整合。PDF 不负责导航，但每节末保留精确入口。

## 怎样使用这篇笔记

1. 先读每节的“问题与前置知识”，明确本节究竟在解决什么。
2. 证明不要只背结论：依次检查目标、构造、每一步依据、使用的假设和边界情形。
3. 代表例题先遮住解答自己做；再用“符号、定义域、单位、图像趋势”四项检查。
4. 每节最后完成三道自检题。答案折叠，适合第二次复习时主动回忆。
5. 题目图形若依赖原印刷页，正文会给出解析描述，同时链接对应 PDF 页。

## 学习目标

学完本章后，应当能够：

1. 从割线极限定义导数，并在几何、运动、单位和误差传播之间切换解释。
2. 区分函数值、极限、连续与可导；识别可去、跳跃、无穷和振荡间断。
3. 从定义证明线性法则、积法则，并熟练使用幂、积、商、链式法则。
4. 在弧度制下证明基本三角极限，进而推导正弦、余弦及其他三角导数。
5. 用隐函数求导处理关系式，并推导反函数、反三角函数的导数。
6. 理解 $e$ 的选择、指数与对数互逆、对数求导、变量幂和双曲函数。
7. 独立完成本章两套指定作业和 Exam 1，并能说明每个步骤使用了什么规则。

## 课程导航

| 官方位置      | 内容                                       |                                                                    |                                                                            |                                                   |                                                                                       |                                                                                  |                                                               |                                                            |                                       |                                        |                                      |                                               |            |
| --------- | ---------------------------------------- | ------------------------------------------------------------------ | -------------------------------------------------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------- | ---------------------------------------------------------- | ------------------------------------- | -------------------------------------- | ------------------------------------ | --------------------------------------------- | ---------- |
| Part A    | [[#Session 1：Introduction to Derivatives | S1 导数简介]] · [[#Sesion 2：Examples of Derivatives                    | S2 定义计算]] · [[#Session 3：Derivative as Rate of Change                      | S3 变化率]] · [[#Session 4：Limits and Continuity     | S4 极限与连续]] · [[#Session 5：Discontinuity                                               | S5 间断]] · [[#Session 6：Calculating Derivatives                                   | S6 基本规则]] · [[#Session 7：Derivatives of Sine and Cosine       | S7 正余弦导数]] · [[#Session 8：Limits of Sine and Cosine        | S8 三角极限]] · [[#Session 9：Product Rule | S9 积法则]] · [[#Session 10：Quotient Rule | S10 商法则]] · [[#Session 11：Chain Rule | S11 链式法则]] · [[#Session 12：Higher Derivatives | S12 高阶导数]] |
| Part A 练习 | [[#Problem Set 1                         | Problem Set 1]]                                                    |                                                                            |                                                   |                                                                                       |                                                                                  |                                                               |                                                            |                                       |                                        |                                      |                                               |            |
| Part B    | [[#Session 13：Implicit Differentiation   | S13 隐函数与有理幂]] · [[#Session 14：Examples of Implicit Differentiation | S14 隐函数例题]] · [[#Session 15：Implicit Differentiation and Inverse Functions | S15 反函数]] · [[#Session 16：The Derivative of $a^x$ | S16 $a^x$]] · [[#Session 17：The Exponential Function, Its Derivative, and Its Inverse | S17 $e^x$ 与 $\ln x$]] · [[#Session 18：Derivatives of Other Exponential Functions | S18 对数求导]] · [[#Session 19：An Interesting Limit Involving $e$ | S19 关于 $e$ 的极限]] · [[#Session 20：Hyperbolic Trig Functions | S20 双曲函数]]                            |                                        |                                      |                                               |            |
| Part B 练习 | [[#Problem Set 2                         | Problem Set 2]]                                                    |                                                                            |                                                   |                                                                                       |                                                                                  |                                                               |                                                            |                                       |                                        |                                      |                                               |            |
| Exam 1    | [[#Session 21：Review for Exam 1          | S21 综合复习]] · [[#Session 22：Materials for Exam 1                    | S22 完整题解]]                                                                 |                                                   |                                                                                       |                                                                                  |                                                               |                                                            |                                       |                                        |                                      |                                               |            |

---

## Part A：Definition and Basic Rules

## Session 1：Introduction to Derivatives

### 本节问题与前置知识

问题 :只给一条曲线和曲线上一点，怎样定义并计算该点的“方向”？

前置知识：两点间直线斜率 $m=(y_2-y_1)/(x_2-x_1)$、点斜式 $y-y_0=m(x-x_0)$、极限的直观含义。

### 按片段展开：从几何问题到主公式

**01a - Welcome。** 微分的用途远超画切线：所有“量怎样随另一个量改变”的问题都可能需要导数。课程最终目标之一，是能计算诸如 $e^{x\arctan x}$ 这类多层复合函数的导数。

**01b - Geometric interpretation。** 设曲线为 $y=f(x)$，固定

$$
P=(x_0,f(x_0)).
$$

经过 $P$ 的切线（tangent line）不是“只接触曲线一次的线”。一条切线可以在别处再次穿过曲线；有些只交一次的线也未必反映局部方向。正确的几何思想是：切线是附近割线（secant line）的极限位置，也是曲线在 $P$ 附近的最佳直线近似。

**01c - Geometric definition。** 在曲线上再取移动点

$$
Q=(x_0+h,f(x_0+h)),\qquad h\ne0.
$$

$P,Q$ 确定割线。让 $Q$ 沿曲线趋近 $P$，等价于让 $h\to0$。若割线斜率趋向一个唯一有限数，就把这个数定义为切线斜率。

**01d - Slope as ratio。** 水平改变量和竖直改变量分别是

$$
\Delta x=h,\qquad \Delta f=f(x_0+h)-f(x_0).
$$

割线斜率是

$$
\frac{\Delta f}{\Delta x}
=\frac{f(x_0+h)-f(x_0)}h.
$$

**01e - Main formula。** 取极限便得到[[Derivative|导数]]定义；几何上它对应[[geometric interpretation of derivative|导数的几何意义]]：

> [!important] 点处导数
> 若下列双侧极限存在且为有限数，则 $f$ 在 $x_0$ 可导（differentiable）：
> $$
> f'(x_0)=\lim_{h\to0}\frac{f(x_0+h)-f(x_0)}h.
> $$
> 对应切线为
> $$
> y-f(x_0)=f'(x_0)(x-x_0).
> $$

![[98_attachment/MIT18.01SC/unit01-secant-tangent.png|900]]

### 代表例题：讲义中的割线实验

课堂练习取

$$
f(x)=\frac12x^3-x.
$$

在一般点 $x$，步长为 $h$ 的割线斜率为

$$
\begin{aligned}
\frac{f(x+h)-f(x)}h
&=\frac{\frac12(x+h)^3-(x+h)-\left(\frac12x^3-x\right)}h\\
&=\frac{\frac12(3x^2h+3xh^2+h^3)-h}{h}\\
&=\frac32x^2+\frac32xh+\frac12h^2-1.
\end{aligned}
$$

令 $h\to0$：

$$
f'(x)=\frac32x^2-1.
$$

例如 $x=-0.75$ 时，切线斜率

$$
f'(-0.75)=\frac32(0.75)^2-1=-0.15625.
$$

练习表中 $h=-0.5,-0.25,0.25,0.5$ 得到的割线斜率约为 $0.53,0.16,-0.41,-0.59$；它们并不都很接近 $-0.15625$。这揭示一个重要事实：同样大小的 $h$ 在曲率较大的地方可能不够小。“趋近”不是固定精度，而是可按误差要求继续缩小 $|h|$。

### 为什么不能直接令 $h=0$

在原差商中令 $h=0$ 会得到 $0/0$，这不是一个数。正确顺序是：

1. 只讨论 $h\ne0$；
2. 用代数变形消去导致 $0/0$ 的公共因子；
3. 再研究 $h\to0$ 时化简后表达式趋向什么。

极限允许变量任意接近零，但不要求它在计算过程中等于零。

### 边界情况与易错点

- 双侧差商极限必须相同。尖角处左右斜率不同，因此不可导。
- 极限趋于 $\pm\infty$ 时可以说有竖直切线，但按本课程“有限导数”约定仍不可导。
- $f'(x_0)$ 是一个数；$f'(x)$ 是随 $x$ 改变的新函数，不要混淆。
- 切线是局部近似，不保证在整个图像上接近曲线。

### 三道自检

1. 写出 $f(x)=x^2$ 在 $x=3$ 的差商，并求切线。
2. 为什么“直线只与曲线相交一点”不能定义切线？
3. 对 $f(x)=\frac12x^3-x$，在 $x=0$ 且 $h=0.25$ 时，割线斜率与切线斜率相差多少？

> [!success]- 自检答案
> 1. $[(3+h)^2-9]/h=6+h\to6$，切线 $y-9=6(x-3)$。
> 2. 切线可能在别处再交曲线；相交次数是全局性质，不能刻画一点附近的方向。割线极限才是局部定义。
> 3. 割线斜率为 $\frac12h^2-1=-0.96875$，切线斜率 $f'(0)=-1$，绝对误差 $0.03125$。

### 本地材料

- [[Ses01a_Lecture_Notes.pdf#page=1|01a Welcome to 18.01（p.1）]]
- [[Ses01b_Lecture_Notes.pdf#page=1|01b Geometric Interpretation（p.1）]]
- [[Ses01c_Lecture_Notes.pdf#page=1|01c Geometric Definition（p.1）]]
- [[Ses01d_Lecture_Notes.pdf#page=1|01d Slope as Ratio（p.1）]]
- [[Ses01e_Lecture_Notes.pdf#page=1|01e Main Formula（p.1）]]
-  [[Ses01e_lec1ses1ex1_secants.pdf#page=1|课堂练习与答案：Secants and Tangents（pp.1-4）]]
- [x] [[Exercise001_Problems.pdf#page=1|Exercise 001 原题]] · [[Exercise001_Solutions.pdf#page=1|Exercise 001 解答]] ✅ 2026-07-30

**知识链：** 两点斜率 → 割线 → 令第二点逼近第一点 → 切线斜率 → 导数定义。

## Session 2：Examples of Derivatives

### 本节问题与前置知识

**问题：** 怎样把抽象差商真正算出来？幂函数的统一规则从哪里来？

**前置知识：** 导数定义、分式通分、二项式展开、直线截距与三角形面积。

### 02a：由定义求 $f(x)=1/x$

固定 $x_0\ne0$。先写差商：

$$
\begin{aligned}
\frac{f(x_0+h)-f(x_0)}h
&=\frac{\frac1{x_0+h}-\frac1{x_0}}h\\
&=\frac{x_0-(x_0+h)}{h\,x_0(x_0+h)}\\
&=-\frac1{x_0(x_0+h)}.
\end{aligned}
$$

这里第二行用共同分母 $x_0(x_0+h)$，第三行才约去 $h$。于是

$$
\boxed{\left(\frac1x\right)'=-\frac1{x^2}},\qquad x\ne0.
$$

合理性检查：$1/x$ 在定义域两支都随 $x$ 增大而下降，所以导数应为负；当 $|x|$ 很大，图像变平，$-1/x^2\to0$，也与图像一致。

### 02b：双曲线切线围成的三角形

**题目。** $y=1/x$ 在第一象限任一点 $P=(x_0,1/x_0)$ 的切线，与两坐标轴围成的三角形面积是多少？

导数给出切线斜率 $m=-1/x_0^2$，故

$$
y-\frac1{x_0}=-\frac1{x_0^2}(x-x_0).
$$

求 $x$ 截距：令 $y=0$，

$$
-\frac1{x_0}=-\frac{x-x_0}{x_0^2}
\quad\Longrightarrow\quad x=2x_0.
$$

求 $y$ 截距：令 $x=0$，

$$
y-\frac1{x_0}=\frac1{x_0}
\quad\Longrightarrow\quad y=\frac2{x_0}.
$$

因此

$$
A=\frac12\cdot2x_0\cdot\frac2{x_0}=\boxed{2}.
$$

面积与切点无关。这里微积分只负责求斜率；坐标几何和代数负责余下步骤。变量 $x,y$ 在“曲线上点”“切线上任一点”“坐标轴截距”中扮演不同角色，必须由方程语境辨认。

### 02c：导数记号

若 $y=f(x)$，常见记号为

$$
f'(x),\quad y',\quad Df(x),\quad \frac{df}{dx},\quad \frac{dy}{dx},\quad \frac d{dx}f(x).
$$

$f'(x_0)$ 明确表示在 $x_0$ 的值；$dy/dx$ 强调“相对于谁求导”。Leibniz 记号形似分数，但定义上是一个整体运算符；以后链式法则中可以像分数一样帮助记忆，却不能不加条件地任意约分。

### 02d：正整数幂法则的完整推导

对正整数 $n$，从定义出发：

$$
\frac{(x+h)^n-x^n}{h}.
$$

二项式定理给出

$$
(x+h)^n=x^n+nx^{n-1}h+\binom n2x^{n-2}h^2+\cdots+h^n.
$$

减去 $x^n$，每一项都有因子 $h$：

$$
\frac{(x+h)^n-x^n}{h}
=nx^{n-1}+\binom n2x^{n-2}h+\cdots+h^{n-1}.
$$

当 $h\to0$，除第一项外其余项都至少含一个 $h$，所以趋于零：

> [!important] 正整数幂法则
> $$
> \boxed{\frac d{dx}x^n=nx^{n-1}},\qquad n=1,2,3,\ldots
> $$

这也解释了课件中的 $O(h^2)$：它代表至少含 $h^2$ 的所有项；除以 $h$ 后成为 $O(h)$，取极限时消失。

### 02e：线性近似乘积（超前材料）

本地 `Ses02e` 的文件名归入 Session 2，但内容使用了稍后才正式证明的积法则，并预告 Unit 2 的线性近似。若

$$
f(x)\approx f_0+f'_0\Delta x,\qquad
g(x)\approx g_0+g'_0\Delta x,
$$

则相乘得到

$$
f_0g_0+(f'_0g_0+f_0g'_0)\Delta x+f'_0g'_0(\Delta x)^2.
$$

忽略二阶小量 $(\Delta x)^2$，一次项系数正是积法则。这里应把它理解为直觉预告，不用它反过来证明本节幂法则。

### 边界情况与易错点

- $1/x$ 的导数公式只在 $x\ne0$ 有意义；求导不能创造原函数没有的定义点。
- 二项式证明当前只覆盖正整数指数；负整数、有理数、实数指数将在 Session 10、13、18 逐步扩展。
- $O(h^2)$ 不是一个固定常数，而是“量级至多与 $h^2$ 相当”的项集合。
- 直线题要先写点斜式，再分别令 $x=0$、$y=0$ 求截距；不要把切点坐标误当截距。

### 三道自检

1. 用定义求 $f(x)=x^3$ 的导数，不直接引用幂法则。
2. $y=1/x$ 在 $x_0=-2$ 的切线是什么？
3. 为什么二项式展开中只有 $nx^{n-1}h$ 对极限留下贡献？

> [!success]- 自检答案
> 1. $[(x+h)^3-x^3]/h=3x^2+3xh+h^2\to3x^2$。
> 2. 点 $(-2,-1/2)$，斜率 $-1/4$，故 $y+1/2=-\frac14(x+2)$，即 $y=-x/4-1$。
> 3. 减去 $x^n$ 并除以 $h$ 后，第一项不再含 $h$；其他项仍至少含一个 $h$，在 $h\to0$ 时趋零。

### 本地材料

- [[Ses02a_Lecture_Notes.pdf#page=1|02a $1/x$（pp.1-2）]]
- [[Ses02b_Lecture_Notes.pdf#page=1|02b A Harder Problem（pp.1-3）]]
- [[Ses02c_Lecture_Notes.pdf#page=1|02c Notations（p.1）]]
- [[Ses02d_MIT18_10SCF10_Ses2d.pdf#page=1|02d Positive Integer Power Rule（pp.1-2）]]
- [[Ses02e_lec9ses2ex1_linearprod.pdf#page=1|02e Product of Linear Approximations（p.1，超前材料）]]
- [x] [[Exercise002_Problems.pdf#page=1|Exercise 002：$|x|$ 的导数]] · [[Exercise002_Solutions.pdf#page=1|答案]] ✅ 2026-07-30

**知识链：** 差商 → 代数消去 $0/0$ → 具体导数 → 二项式结构 → 正整数幂法则。
　
## Session 3：Derivative as Rate of Change

### 本节问题与前置知识

**问题：** 切线斜率为什么也能表示速度、电流、温度梯度和测量灵敏度？

**前置知识：** 差商、导数定义、变量及单位、勾股定理。

### 03a-03b：平均变化率与瞬时变化率

若 $y=f(x)$，在 $x$ 到 $x+h$ 之间，

$$
\frac{\Delta y}{\Delta x}=\frac{f(x+h)-f(x)}h
$$

是单位输入变化所对应的平均输出变化。令区间长度 $h\to0$，得到瞬时变化率

$$
\frac{dy}{dx}=f'(x).
$$

“斜率”与“变化率”是同一个比值的两种语言：在坐标图上看是 rise/run；把横轴解释为时间、距离或产量后，就成为物理或经济量。

### 03c：80 米南瓜下落

课件设南瓜从约 $80$ 米高处静止落下，忽略空气阻力：

$$
h(t)=80-5t^2\quad(\text{m}).
$$

落地时间由 $h(t)=0$ 得

$$
80-5t^2=0\Longrightarrow t=4\text{ s}.
$$

全程平均速度为

$$
\frac{h(4)-h(0)}{4-0}=\frac{0-80}{4}=-20\text{ m/s}.
$$

由幂法则，瞬时速度

$$
v(t)=h'(t)=-10t,
$$

所以撞地前

$$
v(4)=-40\text{ m/s}.
$$

负号表示向下；速率（speed）是速度大小 $|v|=40\text{ m/s}$。平均速度与终点瞬时速度不同，因为下落过程中速度一直在改变。

### 单位检查

若 $h$ 用米、$t$ 用秒，则 $dh/dt$ 的单位是 m/s；再求导得到加速度

$$
a(t)=v'(t)=h''(t)=-10\text{ m/s}^2.
$$

单位是答案的一部分：

- 电荷 $q$（库仑）对时间求导 $dq/dt$ 是电流（安培）；
- 温度 $T$ 对位置 $x$ 求导 $dT/dx$ 是温度梯度（度/米）；
- 成本 $C$ 对产量 $q$ 求导 $dC/dq$ 是边际成本（货币/件）。

### 03d：GPS 灵敏度

简化的平面模型中，卫星高度 $s$ 已知，接收机测得斜距 $h$，水平距离为 $L$：

$$
h^2=s^2+L^2,\qquad L(h)=\sqrt{h^2-s^2}.
$$

对 $h$ 求导：
#confused
$$
\frac{dL}{dh}=\frac{h}{\sqrt{h^2-s^2}}=\frac hL.
$$

测距有小误差 $\Delta h$ 时，水平误差近似

$$
\Delta L\approx\frac{dL}{dh}\Delta h=\frac hL\Delta h.
$$

当接收机几乎在卫星正下方时 $L$ 很小，放大因子 $h/L$ 很大：很小的斜距误差也会造成明显的水平位置误差。这就是导数作为**灵敏度（sensitivity）** 的含义。

> [!note] 近似的逻辑
> $\Delta L/\Delta h$ 是真实有限误差比；$dL/dh$ 是其在 $\Delta h\to0$ 的极限。写 $\Delta L\approx(dL/dh)\Delta h$ 需要误差足够小，并不意味着二者在任意步长下完全相等。

### 边界情况与易错点

- 求导前先说明自变量。$dT/dx$ 与 $dT/dt$ 回答不同问题。
- 灵敏度接近无穷不表示实际误差必定无穷，只说明线性放大因子很大、测量几何很不利。

### 自检

1. 对 $s(t)=3t^2-2t$，求 $[1,3]$ 的平均速度和 $t=3$ 的瞬时速度。
2. GPS 模型中若 $s=3,h=5$，测距误差约 $0.01$，估计 $L$ 的误差。

> [!success]- 自检答案
> 1. $[s(3)-s(1)]/2=(21-1)/2=10$；$s'(t)=6t-2$，故 $s'(3)=16$。
> 2. 选向上为正时下落方向为负，所以速度 $-40\text{ m/s}$；若问速率或速度大小，写 $40\text{ m/s}$。
> 3. $L=\sqrt{25-9}=4$，$dL/dh=5/4$，故 $|\Delta L|\approx(5/4)(0.01)=0.0125$。

### 本地材料

- [[Ses03a_Lecture_Notes.pdf#page=1|03a Introduction to Rates of Change（p.1）]]
- [[Ses03b_Lecture_Notes.pdf#page=1|03b Rates of Change（p.1）]]
- [[Ses03c_Lecture_Notes.pdf#page=1|03c Pumpkin Drop（pp.1-2）]]
- [[Ses03d_Lecture_Notes.pdf#page=1|03d Temperature Gradient and GPS（pp.1-2）]]
- [x] [[Exercise003_Problems.pdf#page=1|Exercise 003：Checking Account Balances]] · [[Exercise003_Solutions.pdf#page=1|答案]] ✅ 2026-07-30

**知识链：**几何斜率 → 单位输出/单位输入 → 平均变化率 → 区间缩到一点 → 瞬时变化率与灵敏度。

## Session 4：Limits and Continuity

### 本节问题与前置知识

**问题：**“趋近”究竟依赖函数在目标点的值吗？什么条件保证可以直接代入？

**前置知识：**函数图像、单侧趋近、代数化简。

### 04a：极限、容易极限与困难极限

记号

$$
\lim_{x\to a}f(x)=L
$$

表示当 $x$ 取足够靠近但不等于 $a$ 的值时，$f(x)$ 可任意靠近 $L$。极限考察的是**附近行为**，所以 $f(a)$ 可以未定义，也可以与 $L$ 不同。

例如

$$
\lim_{x\to3}\frac{x^2+x}{x+1}
=\frac{9+3}{4}=3,
$$

因为分母在 $3$ 附近不为零，函数在此连续，可直接代入。相反，导数差商在 $h=0$ 总得到 $0/0$，必须先化简；涉及除零或无穷远的极限也通常不能直接代入。

### 左右极限

$$
\lim_{x\to a^-}f(x)=L_-,\qquad
\lim_{x\to a^+}f(x)=L_+.
$$

双侧极限存在且等于 $L$，当且仅当左右极限都存在并且

$$
L_-=L_+=L.
$$

课件例子取

$$
f(x)=
\begin{cases}
x+1,&x>0,\\
-x,&x\le0.
\end{cases}
$$

于是

$$
\lim_{x\to0^+}f(x)=1,
\qquad
\lim_{x\to0^-}f(x)=0.
$$

左右不同，所以 $\lim_{x\to0}f(x)$ 不存在；虽然 $f(0)=0$，它不能改变右侧附近的行为。

### 04b：连续的三个条件

> [!important] 点连续
> $f$ 在 $x=a$ 具有[[Continuity|连续性]]，当且仅当：
> 1. $f(a)$ 有定义；
> 2. $\lim_{x\to a}f(x)$ 存在；
> 3. $\lim_{x\to a}f(x)=f(a)$。

等价地说，左右极限和函数值三者相等。连续意味着小输入变化只造成小输出变化，不会突然跳跃；但这不等同于“可导”，因为曲线仍可能有尖角。

### 极限定律为何能用

若 $\lim f=L$、$\lim g=M$，则在相应极限存在时：

$$
\lim(f+g)=L+M,
\quad
\lim(fg)=LM,
\quad
\lim\frac fg=\frac LM\ (M\ne0).
$$

这些定律将在求导规则证明中拆分复杂差商。分母极限为零时，最后一条不能直接使用。

### 边界情况与易错点

- “极限不存在”和“极限为无穷”在严格意义上不同；后者描述一种确定的发散方式。
- 端点只要求定义域内部一侧的连续性；例如 $\sqrt{x}$ 在 $0$ 处讨论右连续。
- 直接代入是连续性的结果，不是极限定义本身。
- 图上空心点表示该点未取值；实心点表示函数值，二者不要混作极限。

### 三道自检

1. 计算 $\lim_{x\to2}(x^2-4)/(x-2)$，并说明为何不能一开始代入。
2. 构造一个 $f(0)=7$ 但 $\lim_{x\to0}f(x)=2$ 的函数。
3. 分段函数 $f(x)=x+a$（$x>1$），$f(x)=x^2$（$x\le1$）在 $1$ 连续时 $a$ 为何值？

> [!success]- 自检答案
> 1. 对 $x\ne2$，原式 $=x+2$，极限为 $4$；直接代入原式是 $0/0$，没有数值。
> 2. 例如 $f(x)=2$（$x\ne0$），$f(0)=7$。
> 3. 左侧及函数值为 $1$，右极限为 $1+a$，故 $a=0$。

### 本地材料

- [[Ses04a_Lecture_Notes.pdf#page=1|04a Limits（pp.1-2）]]
- [[Ses04b_Lecture_Notes.pdf#page=1|04b Continuity（p.1）]]
- [ ] [[Exercise004_Problems.pdf#page=1|Exercise 004：Continuous but not Smooth]] · [[Exercise004_Solutions.pdf#page=1|答案]]

**知识链：**附近行为 → 左右极限 → 双侧极限 → 极限等于函数值 → 连续。

## Session 5：Discontinuity

### 本节问题与前置知识

**问题：**连续性会以哪些方式失败？为什么可导一定连续，而连续未必可导？

**前置知识：**左右极限、点连续定义、导数差商。

### 05a-05d：四类间断

[[Discontinuity|间断]]按连续条件失败的方式分类：

1. **跳跃间断（jump discontinuity）**：左右极限都存在但不相等。上一节分段函数在 $0$ 即为例子。
2. **可去间断（removable discontinuity）**：左右极限相等且有限，但函数未定义或函数值不等于极限。补上正确函数值即可连续。例如 $(x^2-1)/(x-1)$ 在 $x=1$ 的洞。
3. **无穷间断（infinite discontinuity）**：至少一个单侧极限为 $+\infty$ 或 $-\infty$。例如 $1/x$ 在 $0$；这里有竖直渐近线。
4. **振荡间断（oscillatory discontinuity）**：靠近目标点时无限振荡，没有单侧极限。例如 $\sin(1/x)$ 在 $0$。

课件还比较 $f(x)=1/x$ 与 $f'(x)=-1/x^2$：原函数为奇函数，导函数为偶函数；$f'(x)<0$ 准确记录两支图像都向右下降，而导函数形状无需像原函数。

### 05e：[[Differentiability Implies Continuity|可导蕴含连续]]的逐步证明

> [!important] 定理
> 若 $f$ 在 $x_0$ 可导，则 $f$ 在 $x_0$ 连续。

**目标。** 证明 $\lim_{x\to x_0}[f(x)-f(x_0)]=0$。

**构造。** 对 $x\ne x_0$，把函数增量拆成“差商 × 输入增量”：

$$
f(x)-f(x_0)
=\frac{f(x)-f(x_0)}{x-x_0}(x-x_0).
$$

**取极限。** 可导假设保证第一因子趋于有限数 $f'(x_0)$；第二因子趋于 $0$：

$$
\begin{aligned}
\lim_{x\to x_0}[f(x)-f(x_0)]
&=\left(\lim_{x\to x_0}\frac{f(x)-f(x_0)}{x-x_0}\right)
\left(\lim_{x\to x_0}(x-x_0)\right)\\
&=f'(x_0)\cdot0=0.
\end{aligned}
$$

因此 $\lim_{x\to x_0}f(x)=f(x_0)$，即连续。

**边界条件。** 证明依赖导数为有限数；若差商趋于无穷，就不能写成“有限数乘零”。又因为极限过程始终取 $x\ne x_0$，中间除以 $x-x_0$ 合法。

### 逆命题为什么错：$|x|$

$f(x)=|x|$ 在 $0$ 连续，但

$$
\frac{|h|-|0|}{h}=\frac{|h|}{h}
=\begin{cases}1,&h>0,\\-1,&h<0.\end{cases}
$$

左右差商极限分别为 $1,-1$，所以不可导。这说明：

$$
\text{可导}\Longrightarrow\text{连续},
\qquad
\text{连续}\centernot\Longrightarrow\text{可导}.
$$

### 边界情况与易错点

- 可去间断处若重新定义函数值，可修复连续性；跳跃、无穷、振荡间断不能只改一个点修复。
- 分段函数要可导，先匹配函数值，再匹配左右导数；只匹配斜率不够。
- $\infty$ 不是普通实数，不能在代数式中任意做 $\infty-\infty$。
- 一条结论的逆命题必须单独证明；不能因“可导蕴含连续”就反向使用。

### 三道自检

1. 分类 $f(x)=(x^2-4)/(x-2)$ 在 $x=2$ 的间断。
2. 分类 $1/(x-3)^2$ 在 $x=3$ 的间断，并给左右行为。
3. 证明若函数在一点不连续，则它在该点一定不可导。

> [!success]- 自检答案
> 1. 对 $x\ne2$，$f=x+2$，极限为 $4$，但原式未定义，是可去间断。
> 2. 无穷间断；左右都趋于 $+\infty$。
> 3. 使用“可导蕴含连续”的逆否命题：若不连续，则不可能可导。这不是逆命题，而是逻辑等价的逆否命题。

### 本地材料

- [[Ses05a_Lecture_Notes.pdf#page=1|05a Jump Discontinuity（pp.1-2）]]
- [[Ses05b_Lecture_Notes.pdf#page=1|05b Removable Discontinuity（p.1）]]
- [[Ses05c_Lecture_Notes.pdf#page=1|05c Infinite Discontinuity（pp.1-2）]]
- [[Ses05d_Lecture_Notes.pdf#page=1|05d Oscillatory Discontinuity（p.1）]]
- [[Ses05e_Lecture_Notes.pdf#page=1|05e Differentiable Implies Continuous（p.1）]]
- [[Exercise005_Problems.pdf#page=1|Exercise 005：Limits and Discontinuity]] · [[Exercise005_Solutions.pdf#page=1|答案]]

**知识链：**连续条件的不同失败方式 → 间断分类 → 差商分解 → 可导必连续 → 用 $|x|$ 否定逆命题。

## Session 6：Calculating Derivatives

### 本节问题与前置知识

**问题：**怎样把已知简单导数组合成多项式等新函数的导数？

**前置知识：**导数定义、极限定律、幂法则、可导蕴含连续。

### 06a：两类公式

课件区分：

- **特定函数公式**：例如 $(x^n)'=nx^{n-1}$；
- **一般组合规则**：例如 $(u+v)'=u'+v'$、$(cu)'=cu'$。

有了二者，就能把多项式逐项求导。

### 常数、常数倍与和法则

常数函数 $f(x)=C$ 的差商恒为零：

$$
\frac{C-C}{h}=0\quad\Longrightarrow\quad C'=0.
$$

若 $c$ 为常数：

$$
\begin{aligned}
(cu)'(x)
&=\lim_{h\to0}\frac{cu(x+h)-cu(x)}h\\
&=c\lim_{h\to0}\frac{u(x+h)-u(x)}h=cu'(x).
\end{aligned}
$$

### 06b：和法则的完整证明

假设 $u,v$ 在 $x$ 可导。由定义：

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

$$
\boxed{(u+v)'=u'+v'}.
$$

同理 $(u-v)'=u'-v'$。推广到有限多项之和，可以逐项求导。

### 例：多项式

$$
\begin{aligned}
\frac d{dx}(4x^5-3x^2+7x-9)
&=4(5x^4)-3(2x)+7(1)-0\\
&=20x^4-6x+7.
\end{aligned}
$$

每一项同时使用了常数倍法则与幂法则。

### 易错点与适用条件

- $(u+v)'=u'+v'$，但后面将看到 $(uv)'\ne u'v'$。
- “常数”指相对于当前自变量不变。例如对 $x$ 求导时 $a$ 可为常数；若 $a=a(x)$ 就不能用常数倍法则。
- 极限和法则要求各极限存在；不能把两个各自发散的量随意拆开后相消。
- 幂法则对不同指数的适用范围要按课程进度区分，不要提前把未证明范围当作已证。

### 三道自检

1. 由定义证明 $(u-v)'=u'-v'$。
2. 求 $D(2x^7-\pi x^2+e^2)$；此处 $e^2$ 是什么？
3. 若 $u'(2)=3,v'(2)=-5$，求 $(4u-2v)'(2)$。

> [!success]- 自检答案
> 1. 将差商拆为 $[u(x+h)-u(x)]/h-[v(x+h)-v(x)]/h$，分别取极限。
> 2. $14x^6-2\pi x$；$e^2$ 是常数，导数为零。
> 3. $4u'(2)-2v'(2)=12+10=22$。

### 本地材料

- [[Ses06a_Lecture_Notes.pdf#page=1|06a Introduction to Differentiation（p.1）]]
- [[Ses06b_Lecture_Notes.pdf#page=1|06b Derivative of a Sum（p.1）]]
- 本地资料库没有 `Exercise006`；本节自检题用于补足练习入口。

**知识链：**特定导数公式 + 极限线性 → 常数、常数倍、和差法则 → 多项式逐项求导。

## Session 7：Derivatives of Sine and Cosine

### 本节问题与前置知识

**问题：**如何只用导数定义和三角加法公式推导 $\sin x$、$\cos x$ 的导数？

**前置知识：**差商、极限线性、

$$
\sin(a+b)=\sin a\cos b+\cos a\sin b,
$$

$$
\cos(a+b)=\cos a\cos b-\sin a\sin b.
$$

本节暂时把两个基本极限当作已知；Session 8 再证明它们：

$$
\lim_{h\to0}\frac{\sin h}{h}=1,
\qquad
\lim_{h\to0}\frac{\cos h-1}{h}=0.
$$

### 07a：正弦导数的代数推导

从定义开始：

$$
\begin{aligned}
\frac d{dx}\sin x
&=\lim_{h\to0}\frac{\sin(x+h)-\sin x}{h}\\
&=\lim_{h\to0}
\frac{\sin x\cos h+\cos x\sin h-\sin x}{h}.
\end{aligned}
$$

把含 $\sin x$ 和 $\cos x$ 的部分分开：

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

### 07b：余弦导数的代数推导

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

### 为什么必须用弧度

若角度以度为单位，令 $x_{\rm rad}=\pi x_{\rm deg}/180$，则链式法则会给

$$
\frac d{dx_{\rm deg}}\sin(x_{\rm deg})
=\frac\pi{180}\cos(x_{\rm deg}).
$$

只有弧度制使单位圆弧长等于角度数值，从而 $\lim_{h\to0}\sin h/h=1$，也只有此时导数公式具有最简形式。

### 图像检查

- $\sin x$ 在 $x=\pi/2+k\pi$ 处切线水平，对应 $\cos x=0$。
- $\sin x$ 在 $x=2k\pi$ 处上升最快，导数为 $1$；在 $x=(2k+1)\pi$ 处下降最快，导数为 $-1$。
- $\cos x$ 是偶函数，导数 $-\sin x$ 是奇函数，与“偶函数导数为奇函数”一致。

### 边界情况与易错点

- 不可循环论证：本节推导依赖两个基本极限，Session 8 必须独立证明它们。
- $\cos h-1$ 与 $1-\cos h$ 互为相反数；极限都为零，但中间符号不能漏。
- $\sin^2x$ 表示 $(\sin x)^2$，其导数不是 $\cos^2x$；要等链式法则后算作 $2\sin x\cos x$。
- 角度单位若不是弧度，必须额外乘换算因子。

### 三道自检

1. 只用余弦加法公式和两个基本极限，重写 $(\cos x)'$ 的每一步。
2. 求 $\sin x$ 在 $x=\pi$ 的切线。
3. 不画图，说明 $\sin x$ 在 $x=3\pi/2$ 附近为何切线水平且是局部最低点。

> [!success]- 自检答案
> 1. 正文 07b 的四行；关键分组是 $\cos x(\cos h-1)-\sin x\sin h$。
> 2. 点 $(\pi,0)$，斜率 $\cos\pi=-1$，切线 $y=-(x-\pi)$。
> 3. 导数 $\cos(3\pi/2)=0$；其左右 $\cos x$ 从负变正，所以函数先降后升。

### 本地材料

- [[Ses07a_Lecture_Notes.pdf#page=1|07a Derivative of Sine（p.1）]]
- [[Ses07b_Lecture_Notes.pdf#page=1|07b Derivative of Cosine（p.1）]]
- [[Exercise007_Problems.pdf#page=1|Exercise 007：Derivatives of Sine and Cosine]] · [[Exercise007_Solutions.pdf#page=1|答案]]

**知识链：**三角加法公式 → 差商拆成两个基本极限 → 正弦、余弦导数。

## Session 8：Limits of Sine and Cosine

### 本节问题与前置知识

**问题：**上一节使用的两个三角极限为何成立？其几何本质是什么？

**前置知识：**单位圆、弧度、三角形面积、夹逼定理、共轭式。

### 08a：$\lim_{\theta\to0}\sin\theta/\theta=1$

课件先给出直觉：单位圆中，弦的竖直投影长为 $\sin\theta$，弧长为 $\theta$；小角度时弧与弦越来越接近。为了使流程可检查，写成[[Sine Limit by Squeeze Theorem|三角极限夹逼证明]]。

对 $0<\theta<\pi/2$，单位圆内三角形、扇形、外切三角形面积满足

$$
\frac12\sin\theta\cos\theta
<\frac12\theta
<\frac12\tan\theta.
$$

左边是圆内接直角三角形面积，中央是扇形面积，右边是外切三角形面积。分别整理两侧不等式：

$$
\sin\theta\cos\theta<\theta
\quad\Longrightarrow\quad
\frac{\sin\theta}{\theta}<\frac1{\cos\theta},
$$

以及

$$
\theta<\tan\theta=\frac{\sin\theta}{\cos\theta}
\quad\Longrightarrow\quad
\cos\theta<\frac{\sin\theta}{\theta}.
$$

合并得到

$$
\cos\theta<\frac{\sin\theta}{\theta}<\frac1{\cos\theta}.
$$

当 $\theta\to0^+$，两端 $\cos\theta$ 与 $1/\cos\theta$ 都趋于 $1$，夹逼定理给

$$
\lim_{\theta\to0^+}\frac{\sin\theta}{\theta}=1.
$$

又因 $\sin(-\theta)/(-\theta)=\sin\theta/\theta$，该比值是偶函数，左极限相同：

$$
\boxed{\lim_{\theta\to0}\frac{\sin\theta}{\theta}=1}.
$$

![[98_attachment/MIT18.01SC/unit01-trig-squeeze.png|900]]

### 08b-08c：$\lim_{\theta\to0}(1-\cos\theta)/\theta=0$

课件用“圆弧与弦之间的水平缝隙比弧长缩得更快”解释。代数上可以由第一个极限严格推出：

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

$$
\boxed{\lim_{\theta\to0}\frac{1-\cos\theta}{\theta}=0}.
$$

相应地 $(\cos\theta-1)/\theta$ 也趋于 $0$。这里分子和分母都趋零，但分子是二阶小量：由半角公式 $1-\cos\theta=2\sin^2(\theta/2)$ 可看出其数量级约为 $\theta^2/2$。

### 08d：正弦导数的几何图像

单位圆上点 $P$ 的角为 $\theta$，邻点 $Q$ 的角为 $\theta+\Delta\theta$。小弧 $PQ$ 的长度为 $\Delta\theta$，而其竖直变化为

$$
\Delta y=\sin(\theta+\Delta\theta)-\sin\theta.
$$

当 $\Delta\theta$ 很小时，弦方向接近切线方向；圆的切线垂直于半径，因此切线与竖直方向的夹角对应 $\theta$，竖直分量与弧长之比趋于 $\cos\theta$：

$$
\frac{\Delta y}{\Delta\theta}\to\cos\theta.
$$

这是几何直观；严格代数证明仍以 07a 的角和公式与刚证明的极限为准。

### sinc 函数

$$
\operatorname{sinc}(x)=\frac{\sin x}{x},\qquad x\ne0.
$$

极限告诉我们在 $x=0$ 只差一个可去间断。定义 $\operatorname{sinc}(0)=1$ 后连续。它是偶函数，在 $x=k\pi$（非零整数 $k$）取零，振幅受 $1/|x|$ 包络并逐渐衰减；这正是信号处理中常见的振荡形状。

### 边界情况与易错点

- 面积不等式先只对 $0<\theta<\pi/2$ 写；左侧由偶性补齐。
- 取倒数时必须确认各量为正，并反转不等号。
- $\sin\theta\sim\theta$ 是极限等价，不是对非零 $\theta$ 的恒等式。
- 本证明依赖弧度；角度制下单位圆弧长不等于角度数值。

### 三道自检

1. 求 $\lim_{x\to0}\sin(5x)/x$。
2. 求 $\lim_{x\to0}(1-\cos x)/x^2$，可用半角公式。
3. sinc 补定义后为何在 $0$ 连续，但原始分式在 $0$ 不可谈导数？

> [!success]- 自检答案
> 1. $5\,[\sin(5x)/(5x)]\to5$。
> 2. $2\sin^2(x/2)/x^2=\frac12[\sin(x/2)/(x/2)]^2\to1/2$。
> 3. 原始分式在 $0$ 未定义；先定义 sinc$(0)=1$ 才得到新函数。连续性由极限等于补入值保证，之后才可进一步研究导数。

### 本地材料

- [[Ses08a_Lecture_Notes.pdf#page=1|08a $\sin x/x$（pp.1-2）]]
- [[Ses08b_Lecture_Notes.pdf#page=1|08b $(1-\cos x)/x$（pp.1-2）]]
- [[Ses08c_Lecture_Notes.pdf#page=1|08c Questions and Answers（pp.1-2）]]
- [[Ses08d_Lecture_Notes.pdf#page=1|08d Geometric Proof of $(\sin x)'$（pp.1-2）]]
- [[Exercise008_Problems.pdf#page=1|Exercise 008：The Function sinc]] · [[Exercise008_Solutions.pdf#page=1|答案]]

**知识链：**单位圆面积比较 → 夹逼基本极限 → 共轭式推出余弦极限 → 补全三角导数证明。

## Session 9：Product Rule

### 本节问题与前置知识

**问题：**两个都在变化的量相乘，乘积的瞬时变化为何是两项之和？

**前置知识：**导数定义、可导蕴含连续、极限乘法与加法法则。

### 09a-09b：规则与直觉

若矩形边长为 $u,v$，小变化为 $\Delta u,\Delta v$，面积增量为

$$
(u+\Delta u)(v+\Delta v)-uv
=v\Delta u+u\Delta v+\Delta u\Delta v.
$$

除以输入变化后，最后一项是两个小量的乘积，取极限时消失；保留下来的正是“一次只改变一个因子”的两部分。

> [!important] 积法则
> 若 $u,v$ 在 $x$ 可导，则
> $$
> \boxed{(uv)'=u'v+uv'}.
> $$

### 09c：由定义证明

从积的差商开始，并加减中间项 $u(x+h)v(x)$：

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

$$
(uv)'(x)=u(x)v'(x)+v(x)u'(x).
$$

这一步明确使用了“可导蕴含连续”。若不知道 $u(x+h)\to u(x)$，第一项不能直接替换。

### 代表例题

$$
\frac d{dx}(x^n\sin x)
=nx^{n-1}\sin x+x^n\cos x.
$$

三个因子时重复应用：

$$
(uvw)'=u'vw+uv'w+uvw'.
$$

一般而言，有限多个因子求导就是“每一项只对一个因子求导，其余保持原样，再相加”。

### Exercise 009：拼接多项式

分段函数在连接点可导必须满足：函数值连续 + 左右导数相等。对

$$
f(x)=
\begin{cases}
ax^2+bx+6,&x\le0,\\
2x^5+3x^4+4x^2+5x+6,&x>0,
\end{cases}
$$

在 $0$ 两侧函数值自动都是 $6$；左导数 $b$，右导数 $5$，故 $b=5$，而 $a$ 任意。这个练习虽归在 Product Rule Session，实质复习了可导拼接。

### 边界情况与易错点

- 最大误区是写 $(uv)'=u'v'$；常数因子即可反驳：若 $u=x,v=x$，错误公式给 $1$，真实导数是 $2x$。
- 加减中间项有两种选择，所得两项顺序可不同，但结论相同。
- 因子超过两个时不要漏掉任何“只求一个因子导数”的项。
- 先化简再求导有时更短，但必须保持定义域；约去因子可能掩盖原函数的洞。

### 三道自检

1. 用定义证明时为何要加减 $u(x+h)v(x)$？
2. 求 $D[x^2\cos x]$，并在 $x=0$ 检查。
3. 写出 $(uvwx)'$（最后一个 $x$ 是自变量）的展开式。

> [!success]- 自检答案
> 1. 它把一个无法识别的乘积增量拆成两个标准差商；所加所减相同，不改变分子。
> 2. $2x\cos x-x^2\sin x$；在 $0$ 为 $0$，与 $x^2\cos x$ 在原点的水平切线一致。
> 3. $u'vwx+uv'wx+uvw'x+uvw$。

### 本地材料

- [[Ses09a_Lecture_Notes.pdf#page=1|09a General Derivative Rules（p.1）]]
- [[Ses09b_Lecture_Notes.pdf#page=1|09b Introduction to General Rules（p.1）]]
- [[Ses09c_Lecture_Notes.pdf#page=1|09c Product Formula and Proof（pp.1-2）]]
- [[Exercise009_Problems.pdf#page=1|Exercise 009：Smoothing a Piecewise Polynomial]] · [[Exercise009_Solutions.pdf#page=1|答案]]

**知识链：**乘积增量 → 加减中间项 → 两个差商 + 连续性 → 积法则。

## Session 10：Quotient Rule

### 本节问题与前置知识

**问题：**分子、分母都变化时，商的导数如何由各自导数组成？

**前置知识：**分式通分、积法则、可导蕴含连续。

### 10a：商法则推导

设 $v(x)\ne0$，且在 $x$ 附近分母也不为零。记

$$
\Delta u=u(x+h)-u(x),\qquad
\Delta v=v(x+h)-v(x).
$$

则 $u(x+h)=u+\Delta u$、$v(x+h)=v+\Delta v$。商的增量为

$$
\begin{aligned}
\frac{u+\Delta u}{v+\Delta v}-\frac uv
&=\frac{(u+\Delta u)v-u(v+\Delta v)}{v(v+\Delta v)}\\
&=\frac{v\Delta u-u\Delta v}{v(v+\Delta v)}.
\end{aligned}
$$

再除以 $h$：

$$
\frac{\Delta(u/v)}h
=\frac{v(\Delta u/h)-u(\Delta v/h)}{v(v+\Delta v)}.
$$

令 $h\to0$。因 $v$ 可导所以连续，$\Delta v\to0$；差商趋于导数：

> [!important] 商法则
> $$
> \boxed{\left(\frac uv\right)'=\frac{u'v-uv'}{v^2}},\qquad v\ne0.
> $$

记忆时读作“下乘上导，减上乘下导，除以下平方”。负号次序与分子原顺序绑定。

### 10b：倒数与负整数幂

令 $u=1$：

$$
\left(\frac1v\right)'=-\frac{v'}{v^2}=-v^{-2}v'.
$$

取 $v=x^n$：

$$
\frac d{dx}x^{-n}
=-x^{-2n}\cdot nx^{n-1}
=-nx^{-n-1}.
$$

因此幂法则从正整数扩展到负整数：

$$
\frac d{dx}x^m=mx^{m-1},\qquad m\in\mathbb Z, x\ne0\text{（若 }m<0\text{）}.
$$

### 代表例题：正切与正割

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

- 结果分母是 $v^2$，并不取消原限制 $v\ne0$。
- 分子次序写反会整体差一个负号；可用简单函数如 $1/x$ 检查。
- 若可改写为 $uv^{-1}$，积法则 + 链式法则常比死记商法则更可靠；这是下一节练习主题。
- 先约分可能改变定义域。例如 $(x^2-1)/(x-1)=x+1$ 只在 $x\ne1$ 相等。

### 三道自检

1. 求 $D[x^2/(x+1)]$，并化简。
2. 用倒数公式重新求 $(1/x)'$。
3. 为什么 $\tan x$ 的导数公式不能在 $x=\pi/2$ 使用？

> [!success]- 自检答案
> 1. $[2x(x+1)-x^2]/(x+1)^2=x(x+2)/(x+1)^2$，$x\ne-1$。
> 2. 取 $v=x,v'=1$，得到 $-1/x^2$。
> 3. $\cos(\pi/2)=0$，原函数 $\tan x$ 未定义；求导公式不能补出原函数的定义点。

### 本地材料

- [[Ses10a_Lecture_Notes.pdf#page=1|10a Quotient Rule（pp.1-2）]]
- [[Ses10b_Lecture_Notes.pdf#page=1|10b Reciprocals and Negative Powers（p.1）]]
- [[Exercise010_Problems.pdf#page=1|Exercise 010：Quotient Rule Practice]] · [[Exercise010_Solutions.pdf#page=1|答案]]

**知识链：**商的有限增量通分 → 分子分解成 $v\Delta u-u\Delta v$ → 取极限 → 商法则与倒数法则。

## Session 11：Chain Rule

### 本节问题与前置知识

**问题：**输入经过多层函数转换时，总变化率为何是各层局部变化率的乘积？

**前置知识：**函数复合、导数、连续、积法则。

### 11a：中间变量与变化率相乘

设

$$
x=g(t),\qquad y=f(x)=f(g(t)).
$$

有限变化时，若 $\Delta x\ne0$，

$$
\frac{\Delta y}{\Delta t}
=\frac{\Delta y}{\Delta x}\frac{\Delta x}{\Delta t}.
$$

当 $\Delta t\to0$，可导性使 $\Delta x\to0$，两个倍率分别趋于 $dy/dx$ 与 $dx/dt$，于是

> [!important] [[Chain Rule|链式法则]]
> $$
> \boxed{\frac{dy}{dt}=\frac{dy}{dx}\frac{dx}{dt}},
> $$
> 或
> $$
> \boxed{(f\circ g)'(t)=f'(g(t))g'(t)}.
> $$

直觉是“每单位 $t$ 产生多少 $x$”乘“每单位 $x$ 产生多少 $y$”。单位也会相消：$(y/x)(x/t)=y/t$。

### 不跳过 $\Delta x=0$ 的证明细节

直接约掉 $\Delta x$ 在某些步长上可能遇到 $\Delta x=0$。定义辅助函数

$$
\phi(u)=
\begin{cases}
\dfrac{f(g(t)+u)-f(g(t))}{u},&u\ne0,\\[6pt]
f'(g(t)),&u=0.
\end{cases}
$$

$f$ 在 $g(t)$ 可导意味着 $\phi(u)\to f'(g(t))$，并且补值后 $\phi$ 在 $0$ 连续。令 $u=g(t+h)-g(t)$，则无论 $u$ 是否为零都有

$$
f(g(t+h))-f(g(t))=\phi(u)u.
$$

所以

$$
\frac{f(g(t+h))-f(g(t))}{h}
=\phi(u)\frac{g(t+h)-g(t)}h.
$$

令 $h\to0$：因 $g$ 可导故连续，$u\to0$；第一因子趋于 $f'(g(t))$，第二因子趋于 $g'(t)$。链式法则得证。

### 11a 例题：$\sin^{10}t$

令 $x=\sin t$、$y=x^{10}$：

$$
\frac{dy}{dt}
=\frac{dy}{dx}\frac{dx}{dt}
=10x^9\cos t
=\boxed{10\sin^9t\cos t}.
$$

### 11b 例题：$\sin(10t)$

外函数是 $\sin x$，内函数是 $x=10t$：

$$
\frac d{dt}\sin(10t)
=\cos(10t)\cdot10
=\boxed{10\cos(10t)}.
$$

注意 $\sin^{10}t$ 与 $\sin(10t)$ 完全不同：前者是函数值的十次幂，后者是角度放大十倍。

### 多层复合

$$
\frac d{dx}\sin\bigl((x^2+1)^3\bigr)
=\cos\bigl((x^2+1)^3\bigr)
\cdot3(x^2+1)^2\cdot2x.
$$

从最外层向内层逐层求导；每经过一层就乘该层内函数的导数，直到自变量。

### 边界情况与易错点

- 最常见错误是只求外层导数，漏乘内层导数。
- $f'(g(x))$ 表示先把 $g(x)$ 代入 $f'$；不是 $f'(x)g(x)$。
- 链式法则要求外函数在内函数输出处可导、内函数在当前点可导。
- Leibniz 记号提供记忆直觉，但正式依据是复合函数的极限证明。

### 三道自检

1. 求 $D[(3x^2-1)^5]$。
2. 比较 $D[\cos^2x]$ 与 $D[\cos(x^2)]$。
3. 若温度 $T$ 随高度 $z$ 变化，气球高度随时间变化，解释 $dT/dt=(dT/dz)(dz/dt)$ 的单位。

> [!success]- 自检答案
> 1. $5(3x^2-1)^4\cdot6x=30x(3x^2-1)^4$。
> 2. $D[\cos^2x]=-2\sin x\cos x$；$D[\cos(x^2)]=-2x\sin(x^2)$。
> 3. $(\text{度}/\text{米})(\text{米}/\text{秒})=\text{度}/\text{秒}$，表示气球经历的温度随时间变化率。

### 本地材料

- [[Ses11a_Lecture_Notes.pdf#page=1|11a Chain Rule and $\sin^{10}t$（pp.1-2）]]
- [[Ses11b_Lecture_Notes.pdf#page=1|11b Example $\sin(10t)$（p.1）]]
- [[Exercise011_Problems.pdf#page=1|Exercise 011：Do We Need the Quotient Rule?]] · [[Exercise011_Solutions.pdf#page=1|答案]]

**知识链：**复合函数 → 引入中间变量 → 局部倍率相乘 → 处理 $\Delta x=0$ 的严格细节 → 多层链式法则。

## Session 12：Higher Derivatives

### 本节问题与前置知识

**问题：**导数本身继续变化时，怎样表示并解释这种“变化率的变化率”？

**前置知识：**基本求导规则、正余弦导数、运动的速度解释。

### 12a：定义、记号与解释

若 $f'$ 仍可导，则

$$
f''(x)=D^2f(x)=\frac{d^2f}{dx^2}.
$$

继续求导：

$$
f^{(n)}(x)=D^nf(x)=\frac{d^nf}{dx^n}.
$$

这里 $f^{(n)}$ 是[[Higher-Order Derivative|高阶导数]]，不是 $f$ 的 $n$ 次幂。Leibniz 记号中的 $d^2f/dx^2$ 也是一个整体，不应读成普通分数平方。

运动中

$$
s'(t)=v(t),\qquad s''(t)=v'(t)=a(t).
$$

图像上，$f''>0$ 表示斜率 $f'$ 随 $x$ 增大而增加（向上凹）；$f''<0$ 表示斜率减少（向下凹）。这一曲线描绘解释将在 Unit 2 系统使用。

### 正弦的四阶循环

$$
\sin x\xrightarrow D\cos x
\xrightarrow D-\sin x
\xrightarrow D-\cos x
\xrightarrow D\sin x.
$$

因此阶数对 $4$ 取余即可。例如 $101\equiv1\pmod4$，

$$
\frac{d^{101}}{dx^{101}}\sin x=\cos x.
$$

同时 $\sin x$ 与 $\cos x$ 都满足

$$
y''=-y,
$$

这是简谐振动微分方程的核心形式。

### 12b：$D^nx^n=n!$

逐次求导：

$$
\begin{aligned}
Dx^n&=nx^{n-1},\\
D^2x^n&=n(n-1)x^{n-2},\\
D^3x^n&=n(n-1)(n-2)x^{n-3}.
\end{aligned}
$$

第 $k$ 阶为

$$
D^kx^n=\frac{n!}{(n-k)!}x^{n-k},\qquad 0\le k\le n.
$$

取 $k=n$：

$$
\boxed{D^nx^n=n!}.
$$

再求一次，常数导数为零：$D^{n+1}x^n=0$；更高阶也全为零。

### 乘积高阶导数：Leibniz 公式

重复使用积法则会出现二项式系数：

$$
(uv)^{(n)}=
\sum_{k=0}^n\binom nk u^{(k)}v^{(n-k)}.
$$

例如

$$
(uv)''=u''v+2u'v'+uv'',
$$

中间系数 $2$ 来自两条不同求导路径。

### 边界情况与易错点

- 二阶导数记号是 $d^2y/dx^2$，不是 $(dy/dx)^2$。
- $f''(a)=0$ 不足以断言拐点；还要检查凹向是否改变。
- 高阶求导时积法则产生的系数会累积，不能只对每个因子各求同阶一次。
- 物理中加速度与速度同号表示速率增加，异号表示速率减少；不能只看 $a$ 正负。

### 三道自检

1. 求 $D^6x^4$。
2. 求 $D^{2026}\cos x$。
3. 展开 $(uv)'''$。

> [!success]- 自检答案
> 1. 四阶后为 $4!=24$，五阶起为零，所以答案 $0$。
> 2. $2026\equiv2\pmod4$，故为 $-\cos x$。
> 3. $u'''v+3u''v'+3u'v''+uv'''$。

### 本地材料

- [[Ses12a_Lecture_Notes.pdf#page=1|12a Higher Derivatives and Notation（p.1）]]
- [[Ses12b_Lecture_Notes.pdf#page=1|12b Example $D^nx^n$（p.1）]]
- [[Exercise012_Problems.pdf#page=1|Exercise 012：Repeated Differentiation]] · [[Exercise012_Solutions.pdf#page=1|答案]]

**知识链：**导函数仍是函数 → 重复求导 → 速度/加速度与凹向 → 阶数模式、阶乘和 Leibniz 公式。

## Problem Set 1

官方在 Part A 后指定同一份 Differentiation 题册中的 1A、1B、1C、1D、1E、1F、1G、1J 选题。下列编号严格按[官网 Problem Set 1](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/pages/1.-differentiation/part-a-definition-and-basic-rules/problem-set-1/)；不是整本题册的所有题。

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

> [!warning] Problem Set 1 常见错误
> - 图像变换只写答案而不交代平移/尺度；分段题只配函数值不配左右导数。
> - “极限为 $\infty$”与左右分别为 $\pm\infty$ 混为一谈。
> - 求切线只算斜率，漏掉切点；问总路程却只算净位移。
> - 链式法则漏内层导数；商法则负号次序写反；奇偶性证明漏掉 $d(-x)/dx=-1$。

**Problem Set 1 小结：**这些题把 Part A 的三层能力连在一起：先读图和定义域，再选择规则，最后用极限、奇偶性、单位或函数值回代检查。

---

## Part B：Implicit Differentiation and Inverse Functions

## Session 13：Implicit Differentiation

### 本节问题与前置知识

**问题：**当 $y$ 没有方便地单独解出时，怎样求 $dy/dx$？正整数幂法则如何扩展到有理指数？

**前置知识：**链式法则、整数幂法则、指数运算、局部把 $y$ 看成 $y(x)$。

### 13a：隐函数求导的思想

[[Implicit Differentiation|隐函数求导]]从关系式

$$
F(x,y)=0
$$

可能描述多条分支，甚至不能在整个图形上写成单一函数 $y=f(x)$。但在非竖直切线附近，一小段曲线通常仍可把 $y$ 看成由 $x$ 决定。于是对等式两边关于 $x$ 求导：

$$
\frac d{dx}F(x,y(x))=0.
$$

关键规则：每当对含 $y$ 的表达式求导，都要乘 $y'=dy/dx$。例如

$$
\frac d{dx}y^n=ny^{n-1}y'.
$$

这不是“额外规则”，正是外函数 $u^n$ 与内函数 $u=y(x)$ 的链式法则。

### 13b：有理指数幂法则的完整推导

设

$$
y=x^{m/n},
$$

其中 $m\in\mathbb Z,n\in\mathbb N$。为消去分数指数，两边取 $n$ 次幂：

$$
y^n=x^m.
$$

关于 $x$ 求导：

$$
ny^{n-1}y'=mx^{m-1}.
$$

在 $y\ne0$ 处解出：

$$
y'=\frac{m}{n}\frac{x^{m-1}}{y^{n-1}}.
$$

代回 $y=x^{m/n}$：

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

### 定义域和零点必须另查

代数推导中除以了 $y^{n-1}$，所以 $y=0$ 处不能仅凭该步骤断言。还要考虑：

- $n$ 为偶数时，实函数 $x^{m/n}$ 通常只在 $x\ge0$ 定义；
- 同一个数可有不同分数表示，如 $x^{2/6}=x^{1/3}$，实数幂的定义要先约分；
- $r<1$ 时公式常在 $x=0$ 发散。例如 $\sqrt{x}$ 在 $0$ 有竖直切线而无有限双侧导数；
- $r>1$ 时零点可能可导，例如 $(x^{3/2})'=(3/2)\sqrt{x}$ 在右端点为零，但这里讨论的是单侧导数。

### Exercise 013：隐函数二阶导数

由

$$
x^2+4y^2=1
$$

第一次求导：

$$
2x+8yy'=0
\quad\Longrightarrow\quad
y'=-\frac{x}{4y}.
$$

再次求导，使用商法则：

$$
\begin{aligned}
y''
&=-\frac14\frac{y-xy'}{y^2}\\
&=-\frac14\frac{y+x^2/(4y)}{y^2}\\
&=-\frac{4y^2+x^2}{16y^3}.
\end{aligned}
$$

原曲线给 $x^2+4y^2=1$，故

$$
\boxed{y''=-\frac1{16y^3}}.
$$

当 $y=0$ 时第一次导数公式已无定义，对应椭圆左右端点的竖直切线；二阶公式也不适用。

### 边界情况与易错点

- 隐式求导不是把 $y$ 当常数；$D(y^4)=4y^3y'$。
- 先求一般公式再代点，能避免过早把变量变成常数。
- 除以含 $x,y$ 的因子后，要记录该因子为零的点并单独分析。
- $F(x,y)=0$ 在某点未必真能局部表示为 $y(x)$；若 $F_y=0$，公式 $y'=-F_x/F_y$ 失效，常对应竖直切线或更复杂奇点。

### 三道自检

1. 对 $x^2+y^2=25$ 求 $y'$，并求 $(3,4)$ 处切线。
2. 从 $y=x^{2/3}$ 出发，用 $y^3=x^2$ 推导导数，并讨论 $x=0$。
3. 若 $F(x,y)=0$，形式上为什么 $y'=-F_x/F_y$？

> [!success]- 自检答案
> 1. $2x+2yy'=0$，$y'=-x/y$；在 $(3,4)$ 斜率 $-3/4$，切线 $y-4=-\frac34(x-3)$。
> 2. $3y^2y'=2x$，$y'=2x/(3y^2)=\frac23x^{-1/3}$（$x\ne0$）。在 $0$ 差商为 $|h|^{2/3}/h$，大小趋无穷且左右符号相反，形成尖点/竖直切线，不可导。
> 3. 链式法则给 $F_x+F_y y'=0$；若 $F_y\ne0$，解得 $y'=-F_x/F_y$。

### 本地材料

- [[Ses13a_Lecture_Notes.pdf#page=1|13a Introduction to Implicit Differentiation（p.1）]]
- [[Ses13b_Lecture_Notes.pdf#page=1|13b Rational Exponent Rule（pp.1-3）]]
- [[Exercise013_Problems.pdf#page=1|Exercise 013：Implicit Differentiation and Second Derivative]] · [[Exercise013_Solutions.pdf#page=1|答案]]

**知识链：**把 $y$ 看作 $y(x)$ → 链式法则产生 $y'$ → 解出斜率 → 用同一方法证明有理幂法则。

## Session 14：Examples of Implicit Differentiation

### 本节问题与前置知识

**问题：**隐式法何时比先解出 $y$ 更短？如何处理同时含 $x,y$ 的积？

**前置知识：**链式、积法则、分支与定义域。

### 14a：圆的直接法

单位圆

$$
x^2+y^2=1
$$

不是全局的 $y=f(x)$，因为同一 $x\in(-1,1)$ 对应两个 $y$。若只取上半圆，

$$
y=\sqrt{1-x^2}=(1-x^2)^{1/2}.
$$

链式法则给

$$
y'=\frac12(1-x^2)^{-1/2}(-2x)
=-\frac{x}{\sqrt{1-x^2}}
=-\frac xy.
$$

这个直接法必须先选择上支；下支要另算。

### 14b：圆的隐式法

直接对原关系求导：

$$
2x+2yy'=0
\quad\Longrightarrow\quad
\boxed{y'=-\frac xy}.
$$

它同时覆盖上下半圆。几何检查：半径向量 $(x,y)$ 与切向量 $(1,y')$ 的点积

$$
(x,y)\cdot(1,-x/y)=x-x=0,
$$

因此切线确实垂直于半径。$y=0$ 时公式分母为零，对应 $(\pm1,0)$ 的竖直切线。

### 14c：课件原例 $y^4+xy^2-2=0$

这道题不能误记成别的三次曲线。逐项求导：

$$
\frac d{dx}y^4+\frac d{dx}(xy^2)-\frac d{dx}2=0.
$$

第一项用链式法则，第二项同时用积法则和链式法则：

$$
4y^3y'+\left(y^2+x\cdot2yy'\right)=0.
$$

收集所有含 $y'$ 的项：

$$
(4y^3+2xy)y'=-y^2.
$$

所以

$$
\boxed{y'=-\frac{y^2}{4y^3+2xy}}
=-\frac{y}{4y^2+2x}\quad(y\ne0\text{ 时}).
$$

保留未约分形式可更清楚看出从哪一步除以了什么；约分后必须补记 $y=0$ 的排除情况，而原曲线在 $y=0$ 时给 $-2=0$，实际没有这样的点，所以此处约分安全。

### Exercise 014：双曲线分支

$$
y^2-x^2=1.
$$

隐式求导：

$$
2yy'-2x=0
\quad\Longrightarrow\quad
\boxed{y'=\frac xy}.
$$

当 $y=-1$ 时曲线上只有 $x=0$，斜率 $0$；当 $x=1$ 时 $y=\pm\sqrt2$，两支斜率分别 $\pm1/\sqrt2$。对上支直接写 $y=\sqrt{x^2+1}$，得到 $y'=x/\sqrt{x^2+1}=x/y$，与隐式法一致。

### 通用工作流

1. 写清每一项使用和、积、链式中的哪一条规则；
2. 所有含 $y'$ 的项移到同一侧；
3. 提取 $y'$；
4. 除以前先记录可能为零的因子；
5. 最后才代入指定点并写切线。

### 边界情况与易错点

- $D(xy^2)=y^2+2xyy'$，漏掉任一项都错。
- 一个隐式方程可有多个分支；同一个 $x$ 上不同 $y$ 可能给不同斜率。
- $y'=0$ 表示水平切线；公式分母为零且分子非零常表示竖直切线。
- 若分子、分母同时为零，不能直接判断，可能是交叉点、尖点或更高阶接触。

### 三道自检

1. 对 $x^3+y^3=6xy$ 求 $y'$。
2. 求 $x^2+xy+y^2=7$ 在 $(1,2)$ 的切线。
3. 对 $y^4+xy^2-2=0$，说明为何不能把 $D(xy^2)$ 写成 $2xyy'$。

> [!success]- 自检答案
> 1. $3x^2+3y^2y'=6y+6xy'$，故 $y'=(2y-x^2)/(y^2-2x)$。
> 2. $2x+y+xy'+2yy'=0$，$y'=-(2x+y)/(x+2y)$；在 $(1,2)$ 为 $-4/5$，切线 $y-2=-\frac45(x-1)$。
> 3. $x$ 与 $y^2$ 都随 $x$ 变化，积法则给 $x' y^2+x(y^2)'=y^2+2xyy'$；漏掉 $y^2$ 等于错误地把 $x$ 当常数。

### 本地材料

- [[Ses14a_Lecture_Notes.pdf#page=1|14a Circle - Direct Version（p.1）]]
- [[Ses14b_Lecture_Notes.pdf#page=1|14b Circle - Implicit Version（p.1）]]
- [[Ses14c_Lecture_Notes.pdf#page=1|14c $y^4+xy^2-2=0$（p.1）]]
- [[Exercise014_Problems.pdf#page=1|Exercise 014：Implicit Differentiation and Chain Rule]] · [[Exercise014_Solutions.pdf#page=1|答案]]

**知识链：**显式分支的繁琐 → 直接对关系求导 → 积与链式法则同时出现 → 一次覆盖多分支。

## Session 15：Implicit Differentiation and Inverse Functions

### 本节问题与前置知识

**问题：**函数与反函数的斜率为何互为倒数？如何由此推导反正切、反正弦的导数？

**前置知识：**一一对应、函数复合、隐函数和链式法则、三角恒等式。

### 15a：反函数导数定理

若 $g=f^{-1}$，则

$$
f(g(x))=x.
$$

两边求导：

$$
f'(g(x))g'(x)=1.
$$

只要 $f'(g(x))\ne0$，

> [!important] [[Inverse Function Derivative|反函数导数]]
> $$
> \boxed{(f^{-1})'(x)=\frac1{f'(f^{-1}(x))}}.
> $$

若写对应点 $y_0=f(x_0)$，则

$$
(f^{-1})'(y_0)=\frac1{f'(x_0)}.
$$

图像上交换坐标 $(x_0,y_0)\leftrightarrow(y_0,x_0)$，即关于 $y=x$ 反射；切线的 rise/run 也交换，所以斜率取倒数。

![[98_attachment/MIT18.01SC/unit01-inverse-reflection.png|900]]

**假设不能省略：**$f$ 必须在相关区间一一对应，反函数才存在；且 $f'(x_0)\ne0$，否则倒数公式分母为零，反函数可能出现竖直切线。

### 15b：$\arctan x$

令

$$
y=\arctan x,
$$

按主值范围 $y\in(-\pi/2,\pi/2)$，等价于 $\tan y=x$。求导：

$$
\sec^2y\,y'=1
\quad\Longrightarrow\quad
y'=\frac1{\sec^2y}.
$$

用 $\sec^2y=1+\tan^2y=1+x^2$：

$$
\boxed{\frac d{dx}\arctan x=\frac1{1+x^2}},\qquad x\in\mathbb R.
$$

导数恒正，与 $\arctan x$ 单调增加一致；当 $|x|\to\infty$，导数趋零，与水平渐近线 $y=\pm\pi/2$ 一致。

### 15c：$\arcsin x$

令

$$
y=\arcsin x,
$$

主值范围 $y\in[-\pi/2,\pi/2]$，等价于 $\sin y=x$。求导：

$$
\cos y\,y'=1
\quad\Longrightarrow\quad
y'=\frac1{\cos y}.
$$

因主值范围内 $\cos y\ge0$，

$$
\cos y=\sqrt{1-\sin^2y}=\sqrt{1-x^2}.
$$

所以

$$
\boxed{\frac d{dx}\arcsin x=\frac1{\sqrt{1-x^2}}},\qquad |x|<1.
$$

在 $x=\pm1$ 分母为零，反正弦图像有竖直切线；函数在端点连续，但没有有限导数。

### Exercise 015：平方根是平方函数的反函数

限制 $f(x)=x^2$ 的定义域为 $x>0$，使其一一对应。若 $y=f^{-1}(x)$，则 $y^2=x$。隐式求导：

$$
2yy'=1
\quad\Longrightarrow\quad
y'=\frac1{2y}=\boxed{\frac1{2\sqrt x}}.
$$

直接写 $y=\sqrt x=x^{1/2}$ 用有理幂法则得到同一结果。

### 边界情况与易错点

- $(f^{-1})'(x)$ 不是 $1/f'(x)$；分母应在 $f^{-1}(x)$ 处评价。
- $f^{-1}$ 表示反函数，不表示倒数 $1/f$。
- 推导反三角函数必须声明主值范围，平方根符号才能确定。
- 反函数存在需要一一对应；$x^2$ 在整个实数轴上没有函数意义的全局反函数。

### 三道自检

1. 若 $f(2)=5,f'(2)=-3$，求 $(f^{-1})'(5)$。
2. 推导 $(\arccos x)'$，注意主值范围。
3. 求 $D[\arctan(3x)]$。

> [!success]- 自检答案
> 1. $-1/3$。
> 2. 令 $y=\arccos x\in[0,\pi]$，$\cos y=x$，故 $-\sin y\,y'=1$；$\sin y=\sqrt{1-x^2}\ge0$，所以 $y'=-1/\sqrt{1-x^2}$。
> 3. 链式法则给 $3/(1+9x^2)$。

### 本地材料

- [[Ses15a_Lecture_Notes.pdf#page=1|15a Derivative of the Inverse（pp.1-2）]]
- [[Ses15b_Lecture_Notes.pdf#page=1|15b Derivative of $\arctan x$（pp.1-3）]]
- [[Ses15c_Lecture_Notes.pdf#page=1|15c Derivative of $\arcsin x$（p.1）]]
- [[Exercise015_Problems.pdf#page=1|Exercise 015：Derivative of the Square Root]] · [[Exercise015_Solutions.pdf#page=1|答案]]

**知识链：**反函数复合为恒等函数 → 链式法则 → 对应斜率互为倒数 → 反三角导数与主值范围。

## Session 16：The Derivative of $a^x$

### 本节问题与前置知识

**问题：**$a^x$ 的导数为何必定是它自身乘一个只依赖底数的常数？

**前置知识：**指数律、连续性、导数定义。此节尚未选定自然底数 $e$。

### 16a-16b：把指数函数定义到实数

[[Exponential Function|指数函数]]取 $a>0$。整数指数由重复相乘与倒数定义；对有理数 $p/q$，定义

$$
a^{p/q}=\sqrt[q]{a^p}.
$$

指数律

$$
a^{x_1+x_2}=a^{x_1}a^{x_2}
$$

在有理指数上成立。对无理 $x$，用趋近 $x$ 的有理数序列补齐，并要求 $a^x$ 连续。这给出熟悉的连续指数曲线。课件为便于画图先设 $a>1$；$0<a<1$ 时函数递减，$a=1$ 时恒为 $1$。

### 16c：从定义分离出 $x$

$$
\begin{aligned}
\frac d{dx}a^x
&=\lim_{h\to0}\frac{a^{x+h}-a^x}{h}\\
&=\lim_{h\to0}\frac{a^xa^h-a^x}{h}\\
&=a^x\lim_{h\to0}\frac{a^h-1}{h}.
\end{aligned}
$$

定义只依赖底数的常数

$$
M(a)=\lim_{h\to0}\frac{a^h-1}{h}.
$$

于是

> [!important] 一般结构
> $$
> \boxed{\frac d{dx}a^x=M(a)a^x}.
> $$

这一步已经很强：指数函数在任何点的斜率，等于函数高度乘同一个相对增长常数。

### 16d：$M(a)$ 的几何意义

在 $x=0$，$a^0=1$，所以

$$
\left.\frac d{dx}a^x\right|_{x=0}=M(a).
$$

因此 $M(a)$ 是 $y=a^x$ 在 $(0,1)$ 的切线斜率。知道这一点的斜率，就通过 $M(a)a^x$ 知道整条曲线上每一点的斜率。对于 $a>1$，$M(a)>0$；$a=1$ 时 $M(1)=0$；$0<a<1$ 时 $M(a)<0$。

### Exercise 016：复利

本金 $P$、名义年利率 $r$，一年复利 $k$ 次：

$$
A=P\left(1+\frac rk\right)^k.
$$

等效年收益率为

$$
\mathrm{APR}_{\rm eff}=\left(1+\frac rk\right)^k-1.
$$

代入：

- $5\%$ 月复利：$5.1162\%$；日复利：$5.1267\%$；
- $10\%$ 月复利：$10.4713\%$；双周复利（$k=26$）：$10.4959\%$；日复利：$10.5156\%$。

连续复利的极限将在 Session 19 得到 $e^r-1$。

### 边界情况与易错点

- 实指数底数要求 $a>0$；负底数不能对所有实指数给出实值连续函数。
- $a$ 是固定底数；若底数也随 $x$ 变成 $x^x$，本节公式不能直接套用。
- $M(a)$ 目前只是极限定义，尚未证明等于 $\ln a$；Session 17 才完成识别。
- 名义利率 $r$ 与等效年收益率不同；复利次数越多，后者通常越大但有有限上界。

### 三道自检

1. 从定义证明 $(a^x)'/a^x$ 与 $x$ 无关。
2. $M(1)$ 是多少？与图像如何对应？
3. 若 $M(4)=2M(2)$，可从哪条指数关系直观预期这一点？

> [!success]- 自检答案
> 1. 正文 16c 把 $a^x$ 提到极限外，剩余极限只含 $a,h$。
> 2. $M(1)=\lim(1^h-1)/h=0$；$y=1$ 是水平线。
> 3. $4^x=(2^x)^2$；用积法则求导得 $(4^x)'=2\cdot2^x(2^x)'=2M(2)4^x$，故 $M(4)=2M(2)$。

### 本地材料

- [[Ses16a_Lecture_Notes.pdf#page=1|16a Differentiating Logs and Exponentials（p.1）]]
- [[Ses16b_Lecture_Notes.pdf#page=1|16b Working with Exponents（p.1）]]
- [[Ses16c_Lecture_Notes.pdf#page=1|16c $a^x$ and the Definition（p.1）]]
- [[Ses16d_Lecture_Notes.pdf#page=1|16d Slope of the Tangent to $a^x$（pp.1-2）]]
- [[Exercise016_Problems.pdf#page=1|Exercise 016：Compound Interest]] · [[Exercise016_Solutions.pdf#page=1|答案]]

**知识链：**指数律 → 差商中提出 $a^x$ → 剩余常数 $M(a)$ → 一点斜率控制整条指数曲线。

## Session 17：The Exponential Function, Its Derivative, and Its Inverse

### 本节问题与前置知识

**问题：**能否选择一个底数，使指数函数的导数恰好等于自身？它的反函数为何导数为 $1/x$？

**前置知识：**$M(a)$、反函数导数、指数律。

### 17a：用斜率定义 $e$

Session 16 得到

$$
(a^x)'=M(a)a^x,
\qquad
M(a)=\lim_{h\to0}\frac{a^h-1}{h}.
$$

定义 $e$ 为使 $M(e)=1$ 的唯一正底数：

$$
\lim_{h\to0}\frac{e^h-1}{h}=1.
$$

于是

> [!important] 自然指数函数
> $$
> \boxed{\frac d{dx}e^x=e^x}.
> $$

几何上，$y=e^x$ 在 $(0,1)$ 的切线斜率为 $1$。课件用图形夹出 $2<e<4$：$2^x$ 在 $0$ 的切线比连接 $(0,1),(1,2)$ 的割线平，故 $M(2)<1$；$4^x$ 足够陡，$M(4)>1$。若 $M(a)$ 随 $a$ 连续且严格增加，中间必有唯一底数使斜率为 $1$。这是本课层级的存在唯一性说明；更完整证明需要建立指数函数的连续单调理论。

### 17b：[[Natural Logarithm|自然对数]]是 $e^x$ 的反函数

定义

$$
y=e^x\quad\Longleftrightarrow\quad x=\ln y.
$$

因此 $\ln x$ 的定义域是 $x>0$，值域为全体实数；图像是 $e^x$ 关于 $y=x$ 的反射，且

$$
\ln1=0,
\qquad
\ln e=1,
\qquad
\ln(x_1x_2)=\ln x_1+\ln x_2.
$$

令 $w=\ln x$，则 $e^w=x$。隐式求导：

$$
e^w\frac{dw}{dx}=1.
$$

又 $e^w=x$，所以

> [!important] 自然对数导数
> $$
> \boxed{\frac d{dx}\ln x=\frac1x},\qquad x>0.
> $$

![[98_attachment/MIT18.01SC/unit01-exp-log.png|900]]

### 17c：识别一般底数的 $M(a)$

因为 $a=e^{\ln a}$，

$$
a^x=e^{x\ln a}.
$$

链式法则给

$$
\frac d{dx}a^x
=e^{x\ln a}\cdot\ln a
=a^x\ln a.
$$

与 $(a^x)'=M(a)a^x$ 比较：

$$
\boxed{M(a)=\ln a},
$$

从而

$$
\boxed{(a^x)'=a^x\ln a},\qquad a>0.
$$

若 $a>1$，$\ln a>0$，指数函数递增；若 $0<a<1$，$\ln a<0$，指数函数递减。

### 17d：为什么自然对数“自然”

若价格 $p(t)>0$，相对变化率为

$$
\frac{p'(t)}{p(t)}.
$$

链式法则恰好给

$$
\frac d{dt}\ln p(t)=\frac{p'(t)}{p(t)}.
$$

它把乘法增长变成加法，把绝对变化除以当前规模，因此适合比较不同规模资产、人口或浓度的增长。若改用 $\log_{10}$，导数会多出 $1/\ln10$，形式不再直接等于相对增长率。

### Exercise 017：指数与对数方程的检查法

例如

$$
\ln(y+1)+\ln(y-1)=2x+\ln x.
$$

先由真数要求 $y>1,x>0$。合并并指数化：

$$
\ln(y^2-1)=\ln(xe^{2x})
\Longrightarrow y^2-1=xe^{2x}.
$$

由 $y>1$ 选正根：

$$
\boxed{y=\sqrt{xe^{2x}+1}}.
$$

取对数或指数化都可能引入分支选择，最后必须回到原定义域检查。

### 边界情况与易错点

- $\ln x$ 只对 $x>0$ 定义；$(\ln|x|)'=1/x$ 才能覆盖 $x<0$ 的区间。
- $\ln(u+v)$ 不能拆成 $\ln u+\ln v$；只有乘积可拆。
- $(e^{u(x)})'=e^u u'$，不能因 $e^x$ 自导就漏掉链式因子。
- $\log$ 的底数在不同学科可能表示 $10$、$e$ 或 $2$；本笔记用 $\ln$ 明确自然对数。

### 三道自检

1. 求 $D[e^{3x^2}]$。
2. 求 $D[\ln(5x)]$，并解释常数 $5$ 为何消失。
3. 若 $p'/p=0.04$，$D(\ln p)$ 是多少？其单位是什么？

> [!success]- 自检答案
> 1. $6xe^{3x^2}$。
> 2. 直接链式为 $5/(5x)=1/x$；或 $\ln(5x)=\ln5+\ln x$，常数导数为零。定义域 $x>0$。
> 3. $0.04$；若自变量是年，则单位为每年。$\ln p$ 无量纲，相对增长率的单位来自时间倒数。

### 本地材料

- [[Ses17a_Lecture_Notes.pdf#page=1|17a Definition of $e$（pp.1-3）]]
- [[Ses17b_Lecture_Notes.pdf#page=1|17b Natural Log and Its Derivative（pp.1-2）]]
- [[Ses17c_Lecture_Notes.pdf#page=1|17c Derivative of $a^x$（p.1）]]
- [[Ses17d_Lecture_Notes.pdf#page=1|17d The Most Natural Logarithm（p.1）]]
- [[Exercise017_Problems.pdf#page=1|Exercise 017：Solving Equations with $e$ and $\ln$]] · [[Exercise017_Solutions.pdf#page=1|答案]]

**知识链：**$M(a)$ → 选 $M(e)=1$ → $e^x$ 自导 → 反函数 $\ln x$ → 一般 $a^x$ 与相对变化率。

## Session 18：Derivatives of Other Exponential Functions

### 本节问题与前置知识

**问题：**怎样系统求一般指数、变量幂和任意实数幂的导数？

**前置知识：**$e^x,\ln x$ 的导数、积与链式法则、对数性质。

### 18a：$2^x$ 与 $10^x$

一般公式立即给

$$
(2^x)'=(\ln2)2^x,
\qquad
(10^x)'=(\ln10)10^x.
$$

即便从人类习惯的底数 $2$ 或 $10$ 出发，自然对数仍自动出现；$e$ 的特殊之处正是 $\ln e=1$。

### 18b：[[Logarithmic Differentiation|对数求导]]的核心公式

若 $u(x)>0$，链式法则给

$$
\boxed{(\ln u)'=\frac{u'}u}.
$$

因此若先容易求 $(\ln u)'$，可反解

$$
u'=u(\ln u)'.
$$

例如对 $u=a^x$：

$$
\ln u=x\ln a
\Longrightarrow
\frac{u'}u=\ln a
\Longrightarrow
u'=a^x\ln a.
$$

### 18c：移动底数与移动指数 $x^x$

设 $v=x^x$，在实数课程中先取 $x>0$。取自然对数：

$$
\ln v=x\ln x.
$$

两边求导，右边使用积法则：

$$
\frac{v'}v=\ln x+x\frac1x=\ln x+1.
$$

乘回 $v=x^x$：

$$
\boxed{\frac d{dx}x^x=x^x(1+\ln x)},\qquad x>0.
$$

一般地，若 $y=u(x)^{v(x)}$ 且 $u>0$：

$$
\ln y=v\ln u.
$$

求导后得到

$$
\boxed{y'=u^v\left(v'\ln u+v\frac{u'}u\right)}.
$$

两项分别记录指数变化与底数变化；漏掉任一项都错。

### 18d：实数幂法则

对固定实数 $r$、$x>0$：

$$
x^r=e^{r\ln x}.
$$

链式法则：

$$
\frac d{dx}x^r
=e^{r\ln x}\frac r x
=x^r\frac r x
=\boxed{rx^{r-1}}.
$$

至此幂法则从正整数 → 负整数 → 有理数 → 实数完成扩展。对特殊 $r$，定义域有时可延伸到 $x\le0$；但上述 $e^{r\ln x}$ 证明本身只覆盖 $x>0$。

### 对数求导何时特别有用

- 多个幂的乘除：$y=(x-1)^3(x+2)^5/x^7$；
- 变量在指数中：$x^{\sin x}$、$(1+x)^{1/x}$；
- 直接反复积法则过长的表达式。

例如 $y=(x-1)^3(x+2)^5/x^7$（在各因子符号固定且可取对数的区间）：

$$
\frac{y'}y=\frac3{x-1}+\frac5{x+2}-\frac7x,
$$

最后乘回 $y$ 即可。若因子可能为负，可在固定不穿零的区间使用 $\ln|u|$。

### 边界情况与易错点

- 对数求导前先保证表达式正，或在不跨零区间使用绝对值。
- $D(x^x)$ 既不是 $xx^{x-1}$，也不是 $x^x\ln x$；底数与指数都变，必须有两项。
- $x^r=e^{r\ln x}$ 的证明只在 $x>0$；不能无说明地把结论扩到负底数的任意实数幂。
- 求出 $y'/y$ 后不要忘记乘回 $y$。

### 三道自检

1. 求 $D[x^{\sin x}]$（$x>0$）。
2. 求 $D[(2x+1)^{x}]$（$2x+1>0$）。
3. 用对数求导求 $D[(x^2+1)^5/x^3]$。

> [!success]- 自检答案
> 1. $x^{\sin x}[\cos x\ln x+(\sin x)/x]$。
> 2. $(2x+1)^x[\ln(2x+1)+2x/(2x+1)]$。
> 3. 令 $y=(x^2+1)^5x^{-3}$，$y'/y=10x/(x^2+1)-3/x$，故 $y'=y[10x/(x^2+1)-3/x]$；定义域 $x\ne0$。

### 本地材料

- [[Ses18a_Lecture_Notes.pdf#page=1|18a The Functions $10^x$ and $2^x$（p.1）]]
- [[Ses18b_Lecture_Notes.pdf#page=1|18b Logarithmic Differentiation（p.1）]]
- [[Ses18c_Lecture_Notes.pdf#page=1|18c Example $x^x$（p.1）]]
- [[Ses18d_Lecture_Notes.pdf#page=1|18d Real Power Rule（pp.1-2）]]
- 本地资料库没有 `Exercise018`；本节三道自检覆盖变量幂与对数求导。

**知识链：**$\ln$ 把指数移到乘法位置 → 隐式求导 → 变量幂统一公式 → 实数幂法则。

## Session 19：An Interesting Limit Involving $e$

### 本节问题与前置知识

**问题：**为什么“每次增长很少、次数无限多”的极限会产生 $e$？

**前置知识：**对数与指数互逆、导数型极限、无穷极限变量替换。

### 19a：逐步计算 $\left(1+1/n\right)^n$

设

$$
a_n=\left(1+\frac1n\right)^n.
$$

直接看是 $1^\infty$ 型，不能把底数极限和指数极限分别代入。先取对数，把移动指数变成乘法：

$$
\ln a_n=n\ln\left(1+\frac1n\right).
$$

令 $h=1/n$，则 $n\to\infty$ 时 $h\to0^+$：

$$
\ln a_n=\frac{\ln(1+h)}h
=\frac{\ln(1+h)-\ln1}{h}.
$$

这正是 $\ln x$ 在 $x=1$ 的导数：

$$
\lim_{h\to0}\frac{\ln(1+h)-\ln1}{h}
=(\ln x)'|_{x=1}=1.
$$

所以 $\ln a_n\to1$。指数函数连续，

$$
a_n=e^{\ln a_n}\to e^1=e.
$$

> [!important] 关于 $e$ 的基本极限
> $$
> \boxed{\lim_{n\to\infty}\left(1+\frac1n\right)^n=e}.
> $$

这也给 $e$ 的数值近似，例如 $n=10000$ 时约为 $2.71815$。

### 19b：为什么不是 $1$

底数 $1+1/n\to1$，但指数 $n\to\infty$；“非常小的增长”累计“非常多次”，总效应不能只看底数。对数后变成

$$
n\ln(1+1/n),
$$

其中一个因子趋无穷、一个趋零，乘积极限需要上述导数计算，不能写成 $\infty\cdot0=0$。

### 推广

对固定常数 $c,d$，在底数最终为正时：

$$
\left(1+\frac cn\right)^{dn}
=\left[\left(1+\frac cn\right)^{n/c}\right]^{cd}\to e^{cd}.
$$

更稳妥的对数推导是

$$
dn\ln(1+c/n)
=cd\frac{\ln(1+h)}h\to cd,
\quad h=c/n.
$$

连续复利正是

$$
\lim_{n\to\infty}P\left(1+\frac rn\right)^n=Pe^r.
$$

### 边界情况与易错点

- $1^\infty$ 是未定式标签，不是答案 $1$。
- 取对数后必须在最后用指数函数连续性返回原极限。
- $h=1/n$ 只从正侧趋零；本题足够，因为 $\ln$ 在 $1$ 两侧导数一致。
- 有限 $n$ 的复利值不等于 $e^r$，只是随 $n$ 增大趋近。

### 三道自检

1. 求 $\lim_{n\to\infty}(1+1/n)^{3n}$。
2. 求 $\lim_{n\to\infty}(1+2/n)^{5n}$。
3. 解释 $\lim_{h\to0}(1+h)^{1/h}=e$ 与本节公式的关系。

> [!success]- 自检答案
> 1. $e^3$。
> 2. $e^{10}$。
> 3. 令 $h=1/n$ 得同一序列形式；更一般地取对数，$\ln[(1+h)^{1/h}]=\ln(1+h)/h\to1$，所以原式趋 $e$。

### 本地材料

- [[Ses19a_Lecture_Notes.pdf#page=1|19a Another Moving Exponent（pp.1-2）]]
- [[Ses19b_Lecture_Notes.pdf#page=1|19b A Formula for $e$（p.1）]]
- [[Exercise019_Problems.pdf#page=1|Exercise 019：Evaluating an Interesting Limit]] · [[Exercise019_Solutions.pdf#page=1|答案]]
- 资料库另有同内容副本 `Exercise019_Problems_2.pdf` / `Solutions_2.pdf`；正文只链接一份，避免重复导航。

**知识链：**移动指数 → 取对数 → 换元成 $\ln$ 在 $1$ 的导数 → 指数化返回 → 连续复利。

## Session 20：Hyperbolic Trig Functions

### 本节问题与前置知识

**问题：**由 $e^x,e^{-x}$ 的对称组合产生哪些类似三角函数的结构？

**前置知识：**指数求导、积与链式法则、双曲线方程。

### 20a：定义与导数

[[Hyperbolic Functions|双曲函数]]中的双曲正弦（hyperbolic sine）和双曲余弦（hyperbolic cosine）定义为：

$$
\sinh x=\frac{e^x-e^{-x}}2,
\qquad
\cosh x=\frac{e^x+e^{-x}}2.
$$

注意 $(e^{-x})'=-e^{-x}$。因此

$$
\begin{aligned}
(\sinh x)'
&=\frac{e^x-(-e^{-x})}{2}=\cosh x,\\
(\cosh x)'
&=\frac{e^x+(-e^{-x})}{2}=\sinh x.
\end{aligned}
$$

与圆三角函数不同，$\cosh$ 求导没有负号。

### 核心恒等式的逐步证明

$$
\begin{aligned}
\cosh^2x-\sinh^2x
&=\frac{(e^x+e^{-x})^2-(e^x-e^{-x})^2}{4}\\
&=\frac{[e^{2x}+2+e^{-2x}]-[e^{2x}-2+e^{-2x}]}4\\
&=\frac44=\boxed{1}.
\end{aligned}
$$

所以点

$$
(u,v)=(\cosh x,\sinh x)
$$

满足 $u^2-v^2=1$，位于单位双曲线右支；这就是“hyperbolic”的来源。圆三角对应 $\cos^2x+\sin^2x=1$。

![[98_attachment/MIT18.01SC/unit01-hyperbolic.png|900]]

### 由定义推出其他公式

$$
\tanh x=\frac{\sinh x}{\cosh x}.
$$

商法则与恒等式给

$$
(\tanh x)'
=\frac{\cosh^2x-\sinh^2x}{\cosh^2x}
=\boxed{\operatorname{sech}^2x}.
$$

奇偶性：$\sinh$ 为奇函数，$\cosh$ 为偶函数；这也与导数的奇偶转换吻合。

### Exercise 020：双曲加法公式

从指数定义展开：

$$
\begin{aligned}
\sinh(x+y)
&=\frac{e^xe^y-e^{-x}e^{-y}}2\\
&=\frac{(e^x-e^{-x})(e^y+e^{-y})+(e^x+e^{-x})(e^y-e^{-y})}{4}\\
&=\boxed{\sinh x\cosh y+\cosh x\sinh y}.
\end{aligned}
$$

同理

$$
\boxed{\cosh(x+y)=\cosh x\cosh y+\sinh x\sinh y}.
$$

第二式中间是加号，而圆三角的 $\cos(x+y)$ 公式中间是减号；根源是双曲恒等式使用差平方。

### 边界情况与易错点

- $\cosh x\ge1$，从不为零，所以 $\tanh x$ 对所有实数定义。
- $\sinh$、$\sin$ 名称相似但导数循环不同；不要凭符号机械搬用。
- $\cosh^{-1}x$ 常表示反双曲余弦，不是 $1/\cosh x$；倒数写作 $\operatorname{sech}x$。
- 双曲角的几何解释与普通圆角不同，本章只需要指数定义和代数恒等式。

### 三道自检

1. 求 $D[\sinh(3x)]$。
2. 证明 $\cosh x\ge1$。
3. 求 $D[\operatorname{sech}x]$。

> [!success]- 自检答案
> 1. $3\cosh(3x)$。
> 2. $\cosh x=(e^x+e^{-x})/2\ge\sqrt{e^xe^{-x}}=1$（AM-GM），等号仅在 $x=0$。
> 3. sech$\,x=1/\cosh x$，故导数 $-\sinh x/\cosh^2x=-\operatorname{sech}x\tanh x$。

### 本地材料

- [[Ses20a_Lecture_Notes.pdf#page=1|20a Derivatives of Hyperbolic Sine and Cosine（p.1）]]
- [[Exercise020_Problems.pdf#page=1|Exercise 020：Hyperbolic Angle Sum Formula]] · [[Exercise020_Solutions.pdf#page=1|答案]]

**知识链：**指数函数的对称/反对称组合 → 双曲正余弦 → 导数互换 → 差平方恒等式与双曲线。

## Problem Set 2

官网在 Part B 后同时指定两本题册：Differentiation 的 1F-1I，以及 Integration Techniques 的 5A。5A 在此只承担反三角与双曲函数练习，不表示课程已经进入积分技巧。

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

> [!warning] Problem Set 2 常见错误
> - 反三角题忽略主值范围；把 $\sqrt{x^2}$ 直接写成 $x$ 而非 $|x|$。
> - 对数方程不先检查真数为正；二次方程两根不回代定义域。
> - 变量幂只对底数或指数的一方求导；求出 $y'/y$ 后忘记乘回 $y$。
> - 隐式题除以可能为零的因子而不记录；水平切线只令分子为零却不检查分母。

**Problem Set 2 小结：**Part B 的核心不是多记几个公式，而是把链式法则用于“没有显式写出的依赖关系”：$y(x)$、反函数、对数后的变量幂，以及由指数定义的双曲函数。

---

## Exam 1

## Session 21：Review for Exam 1

### 本节问题与前置知识

**问题：**面对综合求导、定义证明、切线和分段函数题，怎样快速辨认结构并检查答案？

**前置知识：**Session 1-20 全部内容。目标不是再添新规则，而是建立调用顺序。

### 21a：公式总表与隐式流程

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

$$
3y^2y'+3y^2+6xyy'=0,
$$

所以

$$
\boxed{y'=-\frac{3y^2}{3y^2+6xy}}.
$$

### 21b：链式法则为何是乘法

若 $y=10x+b$，则 $dy/dx=10$；若 $x=5t+a$，则 $dx/dt=5$。代入得到

$$
y=50t+(10a+b),
$$

故 $dy/dt=50=10\cdot5$。这是“输出相对中间量的倍率 × 中间量相对输入的倍率”。链式法则还可把商改写为积：

$$
\left(\frac uv\right)'=(uv^{-1})'
=u'v^{-1}-uv^{-2}v'
=\frac{u'v-uv'}{v^2}.
$$

### 21c-21f：按真实片段顺序的四个例子

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

**21f - $e^{x\arctan x}$：** 课件实际是整个 $x\arctan x$ 位于指数中，不是 $e^x\arctan x$。设 $u=x\arctan x$：

$$
u'=\arctan x+\frac{x}{1+x^2}.
$$

于是

$$
\boxed{
\frac d{dx}e^{x\arctan x}
=e^{x\arctan x}\left(\arctan x+\frac{x}{1+x^2}\right)}.
$$

### 21g：定义、极限、反函数与图像复习

考试可能把导数定义伪装成极限。例如

$$
\lim_{u\to0}\frac{e^u-1}{u}
=\left.\frac d{dx}e^x\right|_{x=0}=1.
$$

看到

$$
\lim_{h\to0}\frac{f(a+h)-f(a)}h
$$

应立即识别为 $f'(a)$。若问从图像判断可导性，要比较左右割线极限：跳跃、尖角、尖点、竖直切线均不具有有限且相同的双侧导数。

### 考场决策顺序

1. 先写原函数定义域，标出不可求导点。
2. 圈出最外层运算：和、积、商还是复合。
3. 复合函数从外向内逐层；隐函数每个 $y$ 项都带链式因子。
4. 切线题写“点 + 斜率”；运动题先找速度零点再分段算路程。
5. 最后检查奇偶性、符号、单位、简单点和是否漏了定义域。

### 三道自检

1. 求 $D[e^{x\arctan x}]$，并说明用了哪两层规则。
2. 把 $\lim_{h\to0}[\ln(2+h)-\ln2]/h$ 识别为导数并求值。
3. 分段函数在接点处可导需要哪两个独立条件？

> [!success]- 自检答案
> 1. 正文 21f；外层用指数链式法则，指数内部用积法则和 $\arctan$ 导数。
> 2. 是 $(\ln x)'|_{x=2}=1/2$。
> 3. 先连续：左右极限和函数值相等；再匹配左右导数。第二条件不能替代第一条件。

### 本地材料

- [[Ses21a_Lecture_Notes.pdf#page=1|21a Differentiation Formulas（pp.1-2）]]
- [[Ses21b_Lecture_Notes.pdf#page=1|21b Chain Rule Revisited（p.1）]]
- [[Ses21c_Lecture_Notes.pdf#page=1|21c Derivative of Secant（p.1）]]
- [[Ses21d_Lecture_Notes.pdf#page=1|21d Derivative of $\ln(\sec x)$（p.1）]]
- [[Ses21e_Lecture_Notes.pdf#page=1|21e Derivative of $(x^{10}+8x)^6$（p.1）]]
- [[Ses21f_Lecture_Notes.pdf#page=1|21f Derivative of $e^{x\arctan x}$（p.1）]]
- [[Ses21g_MIT18_01SCF10_Ses21g.pdf#page=1|21g Exam 1 Review Continued（pp.1-3）]]

**知识链：**公式识别 → 外层结构决策 → 定义与规则双向使用 → 定义域、图像和单位检查。

## Session 22：Materials for Exam 1

### 本节问题与作答标准

**问题：**怎样把本章的定义、规则、隐式关系和运动解释组合成一套完整考试解答？

**作答标准：**每题都写“已知/目标 → 选择规则 → 逐步计算 → 定义域或几何检查 → 最终答案”。本地 Session 22 没有 `Ses22` 讲义；按官方结构，本节由 Exam 1 原题与官方答案构成。

- [[Exam1_Problems.pdf#page=1|Exam 1 Problems（pp.1-7）]]
- [[Exam1_Solutions.pdf#page=1|Exam 1 Official Solutions（pp.1-8）]]

### Problem 1：计算导数

#### 1(a) $f(x)=x/(1-x^2)$

**规则。** 商法则；原定义域 $x\ne\pm1$。

$$
\begin{aligned}
f'(x)
&=\frac{1(1-x^2)-x(-2x)}{(1-x^2)^2}\\
&=\frac{1-x^2+2x^2}{(1-x^2)^2}\\
&=\boxed{\frac{1+x^2}{(1-x^2)^2}}.
\end{aligned}
$$

**检查。** 分子、分母在定义域内均正，因此 $f'>0$；原函数在每个定义域区间单调增加，与图像一致。答案不能把 $x=\pm1$ 补回。

#### 1(b) $f(x)=\ln(\cos x)-\frac12\sin^2x$

**规则。** 对数链式法则 + 幂的链式法则：

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

#### 1(c) $f(x)=xe^x$ 的五阶导数

先逐阶观察：

$$
\begin{aligned}
f'&=(x+1)e^x,\\
f''&=(x+2)e^x,\\
f^{(3)}&=(x+3)e^x.
\end{aligned}
$$

提出一般命题 $f^{(n)}=(x+n)e^x$。若第 $n$ 阶成立，则

$$
f^{(n+1)}=e^x+(x+n)e^x=(x+n+1)e^x,
$$

所以由归纳法成立。取 $n=5$：

$$
\boxed{f^{(5)}(x)=(x+5)e^x}.
$$

也可用 Leibniz 公式：$x$ 的二阶及以上导数为零，只留下 $xe^x$ 与 $5e^x$ 两项。

### Problem 2：星形线（astroid）的切线

曲线

$$
x^{2/3}+y^{2/3}=4
$$

在 $(-\sqrt{27},1)$ 处。先验点：$-\sqrt{27}=-3^{3/2}$，故

$$
(-3^{3/2})^{2/3}+1^{2/3}=3+1=4.
$$

隐式求导：

$$
\frac23x^{-1/3}+\frac23y^{-1/3}y'=0.
$$

解出

$$
y'=-\frac{x^{-1/3}}{y^{-1/3}}.
$$

在指定点，$x^{1/3}=-\sqrt3$，所以 $x^{-1/3}=-1/\sqrt3$，而 $y^{-1/3}=1$：

$$
m=\frac1{\sqrt3}.
$$

点斜式：

$$
y-1=\frac1{\sqrt3}(x+\sqrt{27}).
$$

因 $\sqrt{27}/\sqrt3=3$，

$$
\boxed{y=\frac{x}{\sqrt3}+4}.
$$

**检查。** 代切点：$(-\sqrt{27})/\sqrt3+4=-3+4=1$；切线确实过指定点。分数幂在负数处按实立方根理解，不能错误地把 $x^{1/3}$ 取成正根。

### Problem 3：前三秒的总路程

位置

$$
y(t)=t^3-3t+3,\qquad t\ge0.
$$

速度

$$
v(t)=y'(t)=3t^2-3=3(t-1)(t+1).
$$

在 $[0,3]$ 内只有 $t=1$ 使速度为零；$0<t<1$ 时 $v<0$，$t>1$ 时 $v>0$，所以粒子在 $t=1$ 改变方向。各关键位置：

$$
y(0)=3,\qquad y(1)=1,\qquad y(3)=21.
$$

总路程是分段位移绝对值之和：

$$
|y(1)-y(0)|+|y(3)-y(1)|
=|1-3|+|21-1|
=2+20.
$$

$$
\boxed{\text{总路程}=22\text{ m}}.
$$

**常见错误。** 净位移是 $y(3)-y(0)=18$ 米，不是总路程；必须先用速度零点确定是否改变方向。

### Problem 4：由定义证明积法则

**定理。** 若 $f,g$ 在 $x$ 可导，则

$$
(fg)'(x)=f'(x)g(x)+f(x)g'(x).
$$

**证明目标。** 把积的差商变成 $f,g$ 的两个标准差商。

**构造。** 加减中间项 $f(x)g(x+h)$：

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

$$
\boxed{(fg)'(x)=f'(x)g(x)+f(x)g'(x)}.
$$

**边界说明。** 中间没有除以 $f$ 或 $g$，所以不要求它们非零。可导性不仅提供差商极限，也通过“可导蕴含连续”提供 $g(x+h)\to g(x)$。

### Problem 5：分段函数能否处处可导

$$
f(x)=
\begin{cases}
\arctan x,&x\le0,\\
ax^2+bx+c,&0<x<2,\\
x^3-\frac14x^2+5,&x\ge2.
\end{cases}
$$

只需检查连接点 $0,2$。

**第一步：连续性。** 在 $0$：

$$
c=\arctan0=0.
$$

在 $2$，右段函数值

$$
2^3-\frac14(2^2)+5=8-1+5=12.
$$

所以

$$
4a+2b+c=12
\quad\Longrightarrow\quad
2a+b=6.
$$

**第二步：$x=2$ 处导数匹配。** 中段左导数 $4a+b$；右段导数

$$
3x^2-\frac12x
$$

在 $2$ 为 $12-1=11$。故

$$
4a+b=11.
$$

联立 $2a+b=6$：

$$
2a=5\Longrightarrow a=\frac52,\qquad b=1.
$$

**第三步：回查 $x=0$ 的导数。** 中段右导数为 $b=1$；左段

$$
(\arctan x)'|_{x=0}=\frac1{1+0^2}=1.
$$

左右相等。因此答案确实存在：

$$
\boxed{a=\frac52,\qquad b=1,\qquad c=0}.
$$

**错误诊断。** 只解 $x=2$ 的两条条件会得到参数，却仍必须回查 $x=0$；本题恰好通过，不代表一般题会自动通过。

### Problem 6：函数方程与导数定义

已知对所有实数 $x,y$：

$$
f(x+y)=f(x)+f(y)+x^2y+xy^2,
$$

且

$$
\lim_{x\to0}\frac{f(x)}x=1.
$$

#### 6(a) 求 $f(0)$

令 $x=y=0$：

$$
f(0)=2f(0)
\quad\Longrightarrow\quad
\boxed{f(0)=0}.
$$

极限条件也说明 $f(x)=x[f(x)/x]\to0$，与结果相容。

#### 6(b) 求 $f'(0)$

直接使用定义和 (a)：

$$
f'(0)=\lim_{h\to0}\frac{f(h)-f(0)}h
=\lim_{h\to0}\frac{f(h)}h
=\boxed{1}.
$$

#### 6(c) 求 $f'(x)$

在函数方程中把第二个变量取为 $h$：

$$
f(x+h)=f(x)+f(h)+x^2h+xh^2.
$$

移项并除以 $h\ne0$：

$$
\frac{f(x+h)-f(x)}h
=\frac{f(h)}h+x^2+xh.
$$

令 $h\to0$，使用已知极限：

$$
\begin{aligned}
f'(x)
&=1+x^2+0\\
&=\boxed{1+x^2}.
\end{aligned}
$$

**进一步检查。** 对 $f'(x)$ 求一个候选原函数 $f(x)=x+x^3/3+C$；由 $f(0)=0$ 得 $C=0$。代回函数方程：

$$
\frac{(x+y)^3-x^3-y^3}{3}=x^2y+xy^2,
$$

恰好成立，验证答案一致。

### Exam 1 三道收尾自检

1. Problem 2 中若忘记先验点，会漏掉什么潜在问题？
2. Problem 5 为什么必须先连续、再匹配导数？
3. Problem 6(c) 中哪一步真正使用了额外极限条件？

> [!success]- 自检答案
> 1. 指定点可能根本不在曲线上；此时“该点切线”无意义。验点也检查分数幂的实数解释。
> 2. 可导必连续；即使左右导数形式碰巧相等，函数值若跳跃仍不可导。连续条件还决定参数的一部分。
> 3. 令 $h\to0$ 时把 $f(h)/h$ 替换为 $1$；其余项只用代数和导数定义。

**Session 22 小结：**Exam 1 同时检查计算与论证。高分答案不只给最终式，还会指出定义域、转向点、连续性条件，以及证明中使用“可导蕴含连续”的位置。

---

## 全章知识链与复习清单

### 一条主链

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

1. 可导蕴含连续：把函数增量写成“差商 × 输入增量”。
2. 正整数幂法则：二项式展开后除以 $h$。
3. $\sin x/x$ 极限：单位圆面积夹逼，且说明弧度条件。
4. 正余弦导数：角和公式 + 两个基本极限。
5. 积法则：加减中间项 + 可导蕴含连续。
6. 链式法则：局部倍率相乘，并处理 $\Delta x=0$ 的辅助函数。
7. 反函数导数：对 $f(f^{-1}(x))=x$ 求导。
8. $e^x,\ln x$ 导数：$M(e)=1$ 与反函数求导。
9. $\lim(1+1/n)^n=e$：取对数化成 $\ln x$ 在 $1$ 的导数。

### 每次求导的五项检查

- **定义域：**原函数在哪里有意义？答案不能扩张原定义域。
- **结构：**最外层是和、积、商还是复合？内层导数是否齐全？
- **符号：**递增/递减、奇偶性、简单点斜率是否吻合？
- **单位：**导数单位是否为“输出单位/输入单位”？
- **边界：**除过的因子能否为零？端点、尖角、竖直切线是否需单独讨论？

> [!tip] 一遍看懂后的主动复习
> 合上正文，依次从定义重算 $1/x$、证明积法则、证明三角基本极限、推导反函数公式、求 $D(x^x)$。若不仅能写结论，还能说出每一步的假设和检查方法，本章就真正形成了可迁移的知识链。

---
aliases:
  - MIT 18.01SC Final Exam
  - MIT 单变量微积分期末复习
tags:
  - math/calculus
  - course/mit-ocw
  - exam/final
source: https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/pages/final-exam/
---

# MIT 18.01SC Final Exam

> [!abstract] 使用方式
> 先独立完成原卷，再逐题阅读下方解答。每题不仅给计算结果，还标明它在考哪一条知识链、为什么选择该方法以及怎样检查答案。

- [[Final_Exam_Problems.pdf|Final Exam Problems]]
- [[Final_Exam_Solutions.pdf|Official Solutions]]

> [!example]- 完整原卷与官方答案
> ![[Final_Exam_Problems.pdf#height=650]]
>
> ![[Final_Exam_Solutions.pdf#height=650]]

## 全课程检查表

### Unit 1：Differentiation

- 差商定义、切线与变化率
- 连续、可导及左右极限
- 积、商、链式与隐函数求导
- 反函数、指数、对数与高阶导数

> [!question]- Unit 1 三问自检
> 1. 差商中何时才能约去趋于零的因子？答案：先在穿孔邻域代数化简，再取极限。
> 2. 隐式曲线的水平切线检查哪一项？答案：在 $F_y\ne0$ 时检查 $F_x=0$，因为 $y'=-F_x/F_y$。
> 3. 七阶三角导数如何防止符号错位？答案：同时追踪 $2^7$ 与四阶函数循环。

### Unit 2：Applications of Differentiation

- 线性/二次近似与误差
- 曲线描绘、最值、相关变化率
- Rolle/MVT、Newton 法
- 原函数、换元和可分离微分方程

> [!question]- Unit 2 三问自检
> 1. 优化题找到 $A'=0$ 后还缺什么？答案：可行域、端点/极限及全局最优检查。
> 2. 相关变化率为何必须最后代瞬时数值？答案：变量在整个过程中变化，提前代值会把变量误当常数。
> 3. MVT 的两个正则性假设？答案：闭区间连续、开区间可导。

### Unit 3：Definite Integrals

- Riemann 和与两种 FTC
- 面积、体积、平均值、概率
- 梯形与 Simpson 数值积分

> [!question]- Unit 3 三问自检
> 1. 怎样从和式读出 $\Delta x$？答案：找乘在每项后的公共小宽度。
> 2. $\frac1h\int_a^{a+h}f$ 在 $h\to0$ 时趋于什么？答案：若 $f$ 在 $a$ 连续，则趋于 $f(a)$。
> 3. 两个梯形需要几个等距节点？答案：三个。

### Unit 4：Techniques of Integration

- 三角积分与三角代换
- 部分分式、分部积分
- 弧长、曲面、参数与极坐标

> [!question]- Unit 4 三问自检
> 1. $\sqrt{a^2-x^2}$ 首选什么代换？答案：在实数区间内取 $x=a\sin\theta$。
> 2. 分部积分如何回验？答案：对结果求导，确认边界项与剩余积分的符号。
> 3. 参数曲线弧长为何用速度大小？答案：长度累积的是位移向量的欧氏范数，不能让方向相消。

### Unit 5：Exploring the Infinite

- L’Hôpital 法则与反常积分
- 级数、幂级数、Taylor 展开和余项

> [!question]- Unit 5 三问自检
> 1. 比值检验给出 $|x|<R$ 后还要做什么？答案：分别检查两个端点。
> 2. Taylor 不等式的 $M$ 应在哪个区间取？答案：只需覆盖展开点与目标点之间的区间。
> 3. 交错级数的首项误差界是什么？答案：不超过第一个被省略项的绝对值。

---

## Session 102：Materials for Final Exam

## Problem 1：导数与高阶导数

### (a) \(f(x)=x^3e^x\)

这是两个单变量函数因子的乘积：

$$
\begin{aligned}
f'(x)
&=(x^3)'e^x+x^3(e^x)'\\
&=3x^2e^x+x^3e^x\\
&=\boxed{e^x(3x^2+x^3)}.
\end{aligned}
$$

检查：两项都应含 \(e^x\)，因为对任一因子求导时另一因子保留。

### (b) \(f(x)=\sin(2x)\) 的七阶导数

每求导一次，系数乘 2；三角函数按

$$
\sin\to\cos\to-\sin\to-\cos\to\sin
$$

四阶循环：

$$
\begin{aligned}
f'&=2\cos2x,\\
f''&=-4\sin2x,\\
f^{(3)}&=-8\cos2x,\\
f^{(4)}&=16\sin2x.
\end{aligned}
$$

再循环三次：

$$
\boxed{f^{(7)}(x)=-128\cos2x}.
$$

## Problem 2：切线与隐函数

### (a) 抛物线切线

$$
y=3x^2-5x+2,\qquad y'=6x-5.
$$

在 \(x=2\)：

$$
y(2)=4,\qquad m=y'(2)=7.
$$

点斜式：

$$
y-4=7(x-2)
\Longrightarrow
\boxed{y=7x-10}.
$$

### (b) 证明曲线没有水平切线

曲线满足

$$
xy^3+x^3y=4.
$$

对 \(x\) 求导；每个含 \(y\) 的项都使用链式法则：

$$
y^3+3xy^2y'+3x^2y+x^3y'=0.
$$

令

$$
F(x,y)=xy^3+x^3y-4.
$$

则

$$
F_x=y^3+3x^2y=y(y^2+3x^2),
\qquad
F_y=3xy^2+x^3.
$$

在 \(F_y\ne0\) 的正则点，

$$
y'=-\frac{F_x}{F_y}
=-\frac{y^3+3x^2y}{3xy^2+x^3}.
$$

水平切线要求 \(F_x=0\) 且 \(F_y\ne0\)。但 \(F_x=0\) 在实数中只能导致 \(y=0\)：若第二因子也为零，则更有 \(x=y=0\)。任何 \(y=0\) 的点都使

$$
xy^3+x^3y=0\ne4,
$$

所以不在曲线上。于是 \(F_x\) 在曲线上从不为零；即使某点 \(F_y=0\)，那里也只可能是竖直切线而非水平切线。因此

$$
\boxed{\text{曲线上不存在水平切线}}.
$$

> [!warning] 官方答案排版勘误
> 官方解答收集 \(y'\) 时把系数误排成了 \(x^3y^2+x^3\)；逐项求导给出的正确分母是 \(3xy^2+x^3\)。最终“不存在水平切线”的结论仍正确。

## Problem 3：定义与导数型极限

### (a) 从定义求 \(f(x)=x/(x+1)\) 的导数

使用 \(t\to x\) 形式：

$$
\begin{aligned}
f'(x)
&=\lim_{t\to x}\frac{\frac{t}{t+1}-\frac{x}{x+1}}{t-x}\\
&=\lim_{t\to x}
\frac{t(x+1)-x(t+1)}
{(t-x)(t+1)(x+1)}\\
&=\lim_{t\to x}
\frac{t-x}{(t-x)(t+1)(x+1)}\\
&=\boxed{\frac1{(x+1)^2}},\qquad x\ne-1.
\end{aligned}
$$

只有在 \(t\ne x\) 的穿孔邻域中才约去 \(t-x\)，随后再取极限。

### (b) 反正切极限

$$
\lim_{x\to\sqrt3}
\frac{\arctan x-\pi/3}{x-\sqrt3}.
$$

因为 \(\arctan(\sqrt3)=\pi/3\)，这正是 \(\arctan x\) 在 \(x=\sqrt3\) 的导数定义：

$$
\boxed{
\frac1{1+(\sqrt3)^2}=\frac14
}.
$$

不必使用 L’Hôpital；识别导数型极限更直接。

## Problem 4：完整曲线描绘

$$
f(x)=\frac{x}{x^2+1}.
$$

定义域为全体实数，且

$$
f(-x)=-f(x),
$$

所以图像关于原点中心对称。

### 一阶导数与极值

$$
f'(x)=\frac{1-x^2}{(x^2+1)^2}.
$$

- \(|x|<1\)：\(f'>0\)，递增；
- \(|x|>1\)：\(f'<0\)，递减；
- \(x=-1\)：由减转增，局部最小点 \(\boxed{(-1,-1/2)}\)；
- \(x=1\)：由增转减，局部最大点 \(\boxed{(1,1/2)}\)。

又因

$$
2|x|\le x^2+1,
$$

等号仅在 \(|x|=1\) 成立，所以 \(|f(x)|\le1/2\)；这两个局部极值同时也是全局极值。

### 二阶导数与拐点

$$
f''(x)=\frac{2x(x^2-3)}{(x^2+1)^3}.
$$

二阶导数在

$$
x=-\sqrt3,\quad0,\quad\sqrt3
$$

变号，所以三点都是拐点：

$$
\left(-\sqrt3,-\frac{\sqrt3}{4}\right),
\quad(0,0),
\quad\left(\sqrt3,\frac{\sqrt3}{4}\right).
$$

凹凸区间：

- \((-\infty,-\sqrt3)\)：向下凹；
- \((-\sqrt3,0)\)：向上凹；
- \((0,\sqrt3)\)：向下凹；
- \((\sqrt3,\infty)\)：向上凹。

### 渐近线

分母从不为零，所以没有竖直渐近线；且

$$
\lim_{x\to\pm\infty}\frac{x}{x^2+1}=0,
$$

水平渐近线为 \(\boxed{y=0}\)。

## Problem 5：海报面积最小

设印刷区宽 \(x\)、高 \(y\)，则

$$
xy=50.
$$

左右边距各 2 英寸，上下各 4 英寸，所以海报总面积

$$
A=(x+4)(y+8).
$$

代入 \(y=50/x\)：

$$
A(x)=82+8x+\frac{200}{x},\qquad x>0.
$$

$$
A'(x)=8-\frac{200}{x^2}.
$$

令 \(A'=0\)：

$$
x^2=25\Longrightarrow x=5.
$$

只取正根。又

$$
A''(x)=\frac{400}{x^3}>0,
$$

所以这是严格局部最小；并且 \(A\to\infty\) 当 \(x\to0^+\) 或 \(x\to\infty\)，故也是全局最小。此时 \(y=10\)，海报尺寸为

$$
\boxed{9\text{ in}\times18\text{ in}}.
$$

## Problem 6：相关变化率

飞机高度恒为 1 mile，斜距 \(r=1.5\) mile 且

$$
\frac{dr}{dt}=-136\text{ mph}.
$$

水平距离 \(x\) 满足

$$
r^2=x^2+1.
$$

当 \(r=3/2\)：

$$
x=\sqrt{\frac94-1}=\frac{\sqrt5}{2}.
$$

求导：

$$
r\frac{dr}{dt}=x\frac{dx}{dt}.
$$

所以相对水平速度

$$
\frac{dx}{dt}
=\frac{(3/2)(-136)}{\sqrt5/2}
=-\frac{408}{\sqrt5}\text{ mph}.
$$

水平间距缩短的速率大小约为 \(182.5\) mph。采用官方解答图中的预期情形——飞机与汽车沿公路相向运动——闭合速率等于二者速率之和，所以汽车速率为

$$
\boxed{\frac{408}{\sqrt5}-120\approx62.5\text{ mph}}.
$$

若使用题目给的 \(\sqrt5\approx2.2\)，得到约 \(65.5\) mph。

> [!warning] 方向假设
> 原题文字没有明说汽车方向；雷达数据本身只确定相对水平速率的大小。若汽车以极高速度同向追赶飞机，另一数学可能是
> $$
> v_{\rm car}=120+\frac{408}{\sqrt5}\approx302.5\text{ mph}.
> $$
> 官方解答中的相向运动示意图与通常交通情境选择了前一支。

## Problem 7：两个积分型极限

### (a) Riemann 和

$$
\lim_{n\to\infty}
\sum_{i=1}^n
\sqrt{1+\frac{2i}{n}}\frac2n
=\int_0^2\sqrt{1+x}\,dx.
$$

$$
\begin{aligned}
\int_0^2\sqrt{1+x}\,dx
&=\left[\frac23(1+x)^{3/2}\right]_0^2\\
&=\boxed{2\sqrt3-\frac23}.
\end{aligned}
$$

### (b) 小区间平均值

$$
\lim_{h\to0}\frac1h\int_2^{2+h}\sin(x^2)dx.
$$

令 \(F(u)=\int_2^u\sin(x^2)dx\)，则极限是

$$
\lim_{h\to0}\frac{F(2+h)-F(2)}h=F'(2).
$$

由 FTC，

$$
\boxed{F'(2)=\sin4}.
$$

## Problem 8：两个定积分

### (a)

令 \(u=\tan x\)，则 \(du=\sec^2x\,dx\)。当 \(x=0\)，\(u=0\)；当 \(x=\pi/4\)，\(u=1\)：

$$
\int_0^{\pi/4}\tan x\sec^2x\,dx
=\int_0^1u\,du
=\boxed{\frac12}.
$$

### (b)

对

$$
\int_1^2x\ln x\,dx
$$

分部积分，取 \(u=\ln x\)、\(dv=x\,dx\)，于是 \(du=dx/x\)、\(v=x^2/2\)：

$$
\begin{aligned}
\int_1^2x\ln x\,dx
&=\left[\frac{x^2}{2}\ln x\right]_1^2
-\frac12\int_1^2x\,dx\\
&=2\ln2-\frac14(4-1)\\
&=\boxed{2\ln2-\frac34}.
\end{aligned}
$$

## Problem 9：三角代换

计算

$$
\int\frac{x^2}{\sqrt{9-x^2}}dx.
$$

因根式是 \(\sqrt{a^2-x^2}\)，取

$$
x=3\sin\theta,\quad
dx=3\cos\theta\,d\theta,\quad
\sqrt{9-x^2}=3\cos\theta.
$$

在任一满足 \(|x|<3\) 的实区间上，取

$$
\theta=\arcsin(x/3)\in[-\pi/2,\pi/2],
$$

故 \(\cos\theta\ge0\)，根式还原没有符号歧义。

积分化为

$$
9\int\sin^2\theta\,d\theta
=\frac92\int(1-\cos2\theta)d\theta
=\frac92\theta-\frac94\sin2\theta+C.
$$

还原

$$
\theta=\arcsin(x/3),
\qquad
\sin2\theta
=2\frac{x}{3}\frac{\sqrt{9-x^2}}3.
$$

因此

$$
\boxed{
\int\frac{x^2}{\sqrt{9-x^2}}dx
=\frac92\arcsin\frac{x}{3}
-\frac12x\sqrt{9-x^2}+C
}.
$$

## Problem 10：Napkin Ring

假设 \(a>0\)。球半径为 \(a\)，钻孔直径为 \(a\)，所以孔半径为 \(a/2\)。使用圆柱壳：半径 \(x\)，剩余高度 \(2\sqrt{a^2-x^2}\)，\(a/2\le x\le a\)：

$$
V=4\pi\int_{a/2}^{a}x\sqrt{a^2-x^2}\,dx.
$$

令 \(u=a^2-x^2\)，\(du=-2x\,dx\)：

$$
\begin{aligned}
V
&=-2\pi\int_{3a^2/4}^{0}u^{1/2}du\\
&=\frac{4\pi}{3}\left(\frac{3a^2}{4}\right)^{3/2}\\
&=\boxed{\frac{\sqrt3}{2}\pi a^3}.
\end{aligned}
$$

检查：钻孔后体积应小于原球体积 \(4\pi a^3/3\)，本结果满足。

量纲也正确：体积必须是无量纲常数乘 \(a^3\)。

## Problem 11：两梯形近似

区间 \([1,5]\) 用两个梯形，步长 \(h=2\)，只用节点 \(1,3,5\)：

$$
T_2=\frac h2[f(1)+2f(3)+f(5)].
$$

由表：

$$
T_2=1[2.7+2(6.7)+29.7]
=\boxed{45.8}.
$$

## Problem 12：放射性衰变

### (a) 建模并求解

“衰变率与当前质量成正比”写成

$$
\frac{dm}{dt}=km,\qquad k<0,\qquad m(0)=100.
$$

这里 \(t\) 以年计，所以 \(k\) 的单位是 \(\mathrm{year}^{-1}\)，从而指数 \(kt\) 无量纲。

分离变量得到

$$
m(t)=100e^{kt}.
$$

半衰期 1600 年：

$$
50=100e^{1600k}
\Longrightarrow
k=-\frac{\ln2}{1600}.
$$

因此

$$
\boxed{m(t)=100e^{-(\ln2)t/1600}
=100\cdot2^{-t/1600}\text{ mg}}.
$$

### (b) 一千年后

$$
m(1000)
=100\cdot2^{-10/16}
\approx\boxed{65\text{ mg}}.
$$

## Problem 13：Cornu Spiral 的弧长

$$
x(t)=\int_0^t\cos(\pi u^2/2)du,
\qquad
y(t)=\int_0^t\sin(\pi u^2/2)du.
$$

FTC 给出

$$
x'(t)=\cos(\pi t^2/2),
\qquad
y'(t)=\sin(\pi t^2/2).
$$

速度大小：

$$
\sqrt{x'(t)^2+y'(t)^2}
=\sqrt{\cos^2(\pi t^2/2)+\sin^2(\pi t^2/2)}
=1.
$$

所以从 \(0\) 到 \(t_0\) 的弧长

$$
L=\int_0^{t_0}1\,dt
=\boxed{t_0}
$$

（默认 \(t_0\ge0\)；若允许负值，长度为 \(|t_0|\)）。

严格地说，\(t_0<0\) 时应反转积分限：

$$
L=\int_{t_0}^{0}\sqrt{x'(t)^2+y'(t)^2}\,dt=-t_0.
$$

## Problem 14：\(\ln(1+x)\) 的 Taylor 级数

### (a) 展开

几何级数

$$
\frac1{1+x}=1-x+x^2-x^3+\cdots,\qquad |x|<1.
$$

从 0 到 \(x\) 逐项积分：

$$
\boxed{
\ln(1+x)
=x-\frac{x^2}{2}+\frac{x^3}{3}-\frac{x^4}{4}+\cdots
=\sum_{n=1}^{\infty}(-1)^{n+1}\frac{x^n}{n}
}.
$$

### (b) 收敛半径

比值检验给

$$
\lim_{n\to\infty}
\left|\frac{x^{n+1}/(n+1)}{x^n/n}\right|
=|x|.
$$

故 \(|x|<1\)，收敛半径

$$
\boxed{R=1}.
$$

端点需另查：\(x=1\) 为交错调和级数，收敛到 \(\ln2\)；\(x=-1\) 为负调和级数，发散。因此区间为 \((-1,1]\)。

### (c) 两项近似

$$
\ln\frac32=\ln(1+1/2)
\approx\frac12-\frac{(1/2)^2}{2}
=\boxed{\frac38}.
$$

### (d) Taylor 不等式

对二次 Taylor 多项式，

$$
R_2(x)=\frac{f^{(3)}(\xi)}{3!}x^3,
\qquad
f^{(3)}(x)=\frac{2}{(1+x)^3}.
$$

在连接 \(0\) 与 \(1/2\) 的区间 \([0,1/2]\) 上，最大值为 \(M=2\)，所以更紧的 Taylor 界是

$$
|R_2(1/2)|
\le\frac{2(1/2)^3}{6}
=\boxed{\frac1{24}}.
$$

官方解答选择更大的对称区间 \([-1/2,1/2]\)，在其上可取 \(M=16\)，于是得到同样有效但较松的界

$$
|R_2(1/2)|\le\frac13.
$$

Taylor 定理只要求区间包含展开点 \(0\) 与目标点 \(1/2\)，所以在 \([0,1/2]\) 上得到的 \(1/24\) 更紧；它也正好等于交错级数首个被省略项的大小。实际误差约为

$$
\ln(3/2)-\frac38\approx0.03047<\frac1{24}.
$$

## Problem 15（Bonus）：反正切不等式

要证对 \(x>0\)：

$$
\frac{x}{1+x^2}<\arctan x<x.
$$

先比较导数：

$$
\left(\frac{x}{1+x^2}\right)'
=\frac{1-x^2}{(1+x^2)^2},
$$

$$
(\arctan x)'=\frac1{1+x^2},
\qquad
(x)'=1.
$$

对 \(x>0\)：

$$
\frac{1-x^2}{(1+x^2)^2}
<
\frac1{1+x^2}
<1.
$$

三函数在 \(x=0\) 都等于 0。把导数不等式从 0 积分到 \(x\)：

$$
\int_0^x
\left(\frac{t}{1+t^2}\right)'dt
<
\int_0^x(\arctan t)'dt
<
\int_0^x1dt.
$$

由 FTC：

$$
\boxed{\frac{x}{1+x^2}<\arctan x<x,\qquad x>0}.
$$

## 交卷前错误检查

1. 导数题检查链式因子；
2. 定积分换元同步换限；
3. 面积和体积结果必须非负且单位正确；
4. 极限先识别导数或 Riemann 和，再考虑 L’Hôpital；
5. 优化题说明临界点为何是最小值；
6. 级数端点与余项必须单独检查；
7. 相关变化率明确符号方向；
8. 微分方程必须带初值并检查指数无量纲。

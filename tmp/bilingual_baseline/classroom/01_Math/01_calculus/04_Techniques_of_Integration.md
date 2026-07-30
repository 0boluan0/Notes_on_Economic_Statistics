---
aliases:
  - MIT 18.01SC Techniques of Integration
  - MIT 18.01SC 积分技巧
  - Techniques of Integration
tags:
  - math/calculus
  - course/mit-ocw
  - calculus/integration
source: https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/pages/unit-4-techniques-of-integration/
---

# Techniques of Integration

> [!abstract] 本章主线
> 求导有一套局部规则，积分却没有能机械处理一切函数的单一算法。本章的核心是识别结构并进行[[Choosing an Integration Technique|积分技巧选择]]：三角恒等式处理三角幂，三角代换消去二次根式，部分分式拆开有理函数，分部积分逆用乘积法则。随后把同一个“微元—累加—取极限”的思想用于弧长、旋转曲面、参数曲线与极坐标面积。

- 官方课程：[MIT OCW 18.01SC — Unit 4: Techniques of Integration](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/pages/unit-4-techniques-of-integration/)
- 教师：David Jerison；学期：Fall 2010
- 官方顺序：Part A（Session 68–73）→ Problem Set 9 → Part B（Session 74–79）→ Problem Set 10 → Part C（Session 80–84）→ Problem Set 11 → Exam 4（Session 85–86）
- 本地 `SesXXa/b/c...` 是同一 Session 的视频片段；下文严格按字母顺序整合。目录中的 `Ses66a–d` 属于上一单元 Exam 3 复习，故本章不引用。

## 学习目标

完成本章后，应当能够：

1. 依据奇偶性处理 $\int\sin^n x\cos^m x\,dx$，并处理 $\tan,\sec,\cot,\csc$ 的幂。
2. 从根式形状选择正确三角代换，控制角度范围、绝对值与回代。
3. 对任意有理函数先长除、再因式分解并写出完整的部分分式模板。
4. 从乘积法则推导分部积分，并判断哪一部分应当被微分。
5. 从折线极限推导弧长，从窄圆台推导旋转曲面面积。
6. 在直角坐标、参数方程和极坐标之间转换，并正确判断轨迹、方向、重复描画和积分范围。
7. 独立完成 Problem Set 9–11 与 Exam 4 的全部指定题。

## 一张决策表：看到积分先问什么

| 结构 | 第一选择 | 必查条件 |
|---|---|---|
| 复合函数与其导数同时出现 | 直接换元 $u=g(x)$ | $du$ 是否真的匹配，定积分是否换上下限 |
| $\sin^n x\cos^m x$ | 有奇次幂就留一个因子；全偶用半角公式 | 不要同时把两种函数都换成新变量 |
| $\tan^m x\sec^n x$ | 留 $\sec^2x$ 或 $\sec x\tan x$ | 依据可用导数配对，而非死背 |
| $\sqrt{a^2-x^2},\sqrt{a^2+x^2},\sqrt{x^2-a^2}$ | 分别用 $a\sin\theta,a\tan\theta,a\sec\theta$ | 根式的符号、角度范围、回代绝对值 |
| 有理函数 $P/Q$ | 长除 → 分解 $Q$ → 部分分式 | 必须先使 $\deg P<\deg Q$；重复因子不能漏项 |
| 多项式乘指数/三角，或含 $\ln x,\arctan x$ | 分部积分 | 选择 $u$ 后是否变简单；定积分边界项 |
| 几何长度或旋转表面积 | 先写 $ds$，再乘相应半径 | 半径是到旋转轴的距离，不能带负号 |

## 课程目录

### Part A：Trigonometric Powers, Trigonometric Substitution and Completing the Square

1. [[#Session 68：Integral of $\sin^n x\cos^m x$，odd exponents|Session 68：奇次三角幂]]
2. [[#Session 69：Integral of $\sin^n x\cos^m x$，even exponents|Session 69：偶次三角幂]]
3. [[#Session 70：Preview of trig substitution and polar coordinates|Session 70：三角代换预览]]
4. [[#Session 71：Integrals involving secant, cosecant and cotangent|Session 71：正割、余割与余切]]
5. [[#Session 72：Trig substitution|Session 72：三角代换]]
6. [[#Session 73：Completing the square|Session 73：配方法]]
7. [[#Problem Set 9|Problem Set 9]]

### Part B：Partial Fractions, Integration by Parts, Arc Length, and Surface Area

1. [[#Session 74：Integration by partial fractions|Session 74：部分分式]]
2. [[#Session 75：Advanced partial fractions|Session 75：进阶部分分式]]
3. [[#Session 76：Integration by parts|Session 76：分部积分]]
4. [[#Session 77：Volume of a wine glass|Session 77：酒杯体积]]
5. [[#Session 78：Computing the length of a curve|Session 78：弧长]]
6. [[#Session 79：Surface area|Session 79：旋转曲面面积]]
7. [[#Problem Set 10|Problem Set 10]]

### Part C：Parametric Equations and Polar Coordinates

1. [[#Session 80：Parametric curves|Session 80：参数曲线]]
2. [[#Session 81：Examples using parametrized curves|Session 81：参数曲线例题]]
3. [[#Session 82：Polar coordinates|Session 82：极坐标]]
4. [[#Session 83：Polar coordinates，continued|Session 83：极坐标面积]]
5. [[#Session 84：Polar coordinates and graphing|Session 84：极坐标作图]]
6. [[#Problem Set 11|Problem Set 11]]
7. [[#Session 85：Review for Exam 4|Session 85：考试复习]]
8. [[#Session 86：Materials for Exam 4|Session 86：Exam 4 完整题解]]

---

## Part A：Trigonometric Powers, Trigonometric Substitution and Completing the Square

## Session 68：Integral of $\sin^n x\cos^m x$，odd exponents

### 本节问题与前置知识

要计算

$$
I_{n,m}=\int\sin^n x\cos^m x\,dx,
$$

什么时候能把它化为普通多项式积分？前置知识是

$$
\sin^2x+\cos^2x=1,
\qquad d(\sin x)=\cos x\,dx,
\qquad d(\cos x)=-\sin x\,dx.
$$

### 68a：恒等式为何决定换元

若 $m$ 为奇数，写 $m=2k+1$，保留一个 $\cos x\,dx$ 给 $u=\sin x$，其余偶次幂改写：

$$
\cos^{2k}x=(1-\sin^2x)^k.
$$

于是

$$
\int\sin^n x\cos^{2k+1}x\,dx
=\int u^n(1-u^2)^k\,du.
$$

若 $n$ 为奇数，则保留 $\sin x\,dx$ 给 $u=\cos x$，并记住 $du=-\sin x\,dx$。关键不是“看到正弦就换正弦”，而是**新变量的微分必须在剩余因子里出现**。

### 68b–68d：从最简单情形到完整例题

先看 $\int\sin^3x\cos^2x\,dx$。正弦为奇次，拆出一个 $\sin x$：

$$
\begin{aligned}
\int\sin^3x\cos^2x\,dx
&=\int(1-\cos^2x)\cos^2x\sin x\,dx\\
&=-\int(1-u^2)u^2\,du\quad(u=\cos x)\\
&=-\frac{u^3}{3}+\frac{u^5}{5}+C\\
&=-\frac{\cos^3x}{3}+\frac{\cos^5x}{5}+C.
\end{aligned}
$$

只有一种函数也一样，例如

$$
\begin{aligned}
\int\sin^3x\,dx
&=\int(1-\cos^2x)\sin x\,dx\\
&=-\int(1-u^2)\,du
=-\cos x+\frac{\cos^3x}{3}+C.
\end{aligned}
$$

这里可视为 $\cos^0x$：零是偶数，而正弦指数 $3$ 是奇数，所以仍属于“留一个正弦”的情形。

> [!example]- 配套 Exercise 068：计算 $\int\cos^3(2x)\,dx$
> 先处理内部线性函数。令 $u=2x$，$dx=du/2$：
> $$
> \begin{aligned}
> \int\cos^3(2x)\,dx
> &=\frac12\int(1-\sin^2u)\cos u\,du\\
> &=\frac12\left(\sin u-\frac{\sin^3u}{3}\right)+C\\
> &=\frac12\sin(2x)-\frac16\sin^3(2x)+C.
> \end{aligned}
> $$
> 验算时第二项要两次使用链式法则，导数为 $-\sin^2(2x)\cos(2x)$；与第一项的 $\cos(2x)$ 合并，恰为 $\cos^3(2x)$。

> [!warning] 易错点
> - $u=\cos x$ 时漏掉负号。
> - 把 $\sin^2x$ 错写成 $1-\cos x$，正确式是 $1-\cos^2x$。
> - 被积函数含 $2x$、$ax$ 时漏掉链式法则带来的常数。

> [!question]- 三道自检题与答案
> 1. $\int\sin x\cos^4x\,dx=-\cos^5x/5+C$。
> 2. $\int\sin^2x\cos^3x\,dx=\sin^3x/3-\sin^5x/5+C$。
> 3. 若 $n,m$ 都是奇数，两种保留法都可用；通常保留次数较低的一侧使展开更短。

### 本地材料与知识链

- [[Ses68a_Lecture_Notes.pdf#page=1|68a 三角恒等式复习]] · [[Ses68b_Lecture_Notes.pdf#page=1|68b 一个指数为 1]] · [[Ses68c_Lecture_Notes.pdf#page=1|68c $\sin^3x\cos^2x$]] · [[Ses68d_Lecture_Notes.pdf#page=1|68d $\sin^3x$]]
- [[Exercise068_Problems.pdf#page=1|Exercise 068 原题]] · [[Exercise068_Solutions.pdf#page=1|官方解答]]

**小结：**奇次幂提供一个可充当 $du$ 的因子，平方恒等式把其余部分全部变成同一个新变量。

## Session 69：Integral of $\sin^n x\cos^m x$，even exponents

### 本节问题与知识目标

若 $n,m$ 都为偶数，拆出一个正弦或余弦都不会得到另一者的导数。此时使用降幂（power-reduction）公式：

$$
\cos^2x=\frac{1+\cos2x}{2},
\qquad
\sin^2x=\frac{1-\cos2x}{2}.
$$

它们由 $\cos2x=2\cos^2x-1=1-2\sin^2x$ 直接移项得到，所以不是互不相关的新公式。

### 69a：基本例子 $\int\cos^2x\,dx$

$$
\int\cos^2x\,dx
=\frac12\int(1+\cos2x)\,dx
=\frac{x}{2}+\frac{\sin2x}{4}+C.
$$

注意 $\int\cos2x\,dx=\tfrac12\sin2x$，第二个 $1/2$ 来自链式法则。

### 69b：两种偶次幂同时出现

一个常用捷径是

$$
\sin^2x\cos^2x
=\frac14\sin^22x
=\frac18(1-\cos4x).
$$

因此

$$
\int\sin^2x\cos^2x\,dx
=\frac{x}{8}-\frac{\sin4x}{32}+C.
$$

若次数更高，就反复降幂；目标是让最高次数严格下降，而不是只把式子展开得更长。

> [!example]- 配套 Exercise 069：$\int\sin^4x\cos^2x\,dx$
> 先在草稿区降幂：
> $$
> \sin^4x\cos^2x
> =\frac{1-\cos2x-\cos^22x+\cos^32x}{8}.
> $$
> 再用
> $$
> \int\cos^22x\,dx=\frac x2+\frac{\sin4x}{8},
> \quad
> \int\cos^32x\,dx=\frac12\sin2x-\frac16\sin^32x.
> $$
> 合并后
> $$
> \boxed{\int\sin^4x\cos^2x\,dx
> =\frac{x}{16}-\frac{\sin2x}{32}-\frac{\sin^32x}{48}+C.}
> $$
> 被积函数为偶函数，取 $C=0$ 时原函数应为奇函数；上式满足这一快速一致性检查。

> [!warning] 易错点
> 半角公式会改变角频率；每次积分 $\cos(kx)$ 都必须除以 $k$。另外，“一个指数为偶数”并不足以使用本节策略；若另一个指数为奇数，Session 68 的换元通常更短。

> [!question]- 三道自检题与答案
> 1. $\int\sin^2x\,dx=x/2-\sin2x/4+C$。
> 2. $\int\sin^2(3x)\,dx=x/2-\sin6x/12+C$。
> 3. $\int\sin^4x\,dx=3x/8-\sin2x/4+\sin4x/32+C$。

### 本地材料与知识链

- [[Ses69a_Lecture_Notes.pdf#page=1|69a $\cos^2x$]] · [[Ses69b_Lecture_Notes.pdf#page=1|69b $\sin^2x\cos^2x$]]
- [[Exercise069_Problems.pdf#page=1|Exercise 069 原题]] · [[Exercise069_Solutions.pdf#page=1|官方解答]]

**小结：**全偶次时，降幂公式用“角度加倍”换取“次数减半”。

## Session 70：Preview of trig substitution and polar coordinates

### 本节问题

为何圆会自然导出[[Trigonometric Substitution|三角代换]]？课堂以半径为 $a$ 的圆中、高为 $b$ 的区域为例。横切片长度为 $x=\sqrt{a^2-y^2}$，所以

$$
A=\int_0^b\sqrt{a^2-y^2}\,dy.
$$

### 70a：从几何坐标到代数消根

令 $y=a\sin\theta$，则 $dy=a\cos\theta\,d\theta$。规定 $0\le\theta\le\pi/2$ 后 $\cos\theta\ge0$，故

$$
\sqrt{a^2-y^2}
=a\sqrt{1-\sin^2\theta}
=a|\cos\theta|
=a\cos\theta.
$$

若 $\theta_0=\arcsin(b/a)$，则

$$
\begin{aligned}
A&=a^2\int_0^{\theta_0}\cos^2\theta\,d\theta\\
&=\frac{a^2\theta_0}{2}+\frac{a^2\sin2\theta_0}{4}\\
&=\frac{a^2}{2}\arcsin\frac ba+\frac b2\sqrt{a^2-b^2}.
\end{aligned}
$$

第一项是扇形面积，第二项是三角形面积，因而代数答案与几何分割完全一致。这也是很强的结果检查。

![[unit04-trig-substitution-triangles.png|820]]

> [!example]- 配套 Exercise 070：改用 $x=a\cos\theta$ 计算 $\int_0^b\sqrt{a^2-x^2}\,dx$
> 当 $x=0$ 时 $\theta=\pi/2$；当 $x=b$ 时 $\theta=\arccos(b/a)$。由于 $dx=-a\sin\theta\,d\theta$，上下限随 $x$ 增大反而下降：
> $$
> \begin{aligned}
> \int_0^b\sqrt{a^2-x^2}\,dx
> &=a^2\int_{\arccos(b/a)}^{\pi/2}\sin^2\theta\,d\theta\\
> &=\frac{a^2}{2}\arcsin\frac ba+\frac b2\sqrt{a^2-b^2}.
> \end{aligned}
> $$
> 两种代换给出同一结果，因为 $\arcsin(b/a)+\arccos(b/a)=\pi/2$。

> [!warning] 易错点
> $\sqrt{\cos^2\theta}=|\cos\theta|$，不是无条件等于 $\cos\theta$。选择角度区间的作用正是控制符号。定积分代换时还必须同步改变上下限。

> [!question]- 三道自检题与答案
> 1. $b=a$ 时得到 $\pi a^2/4$，即四分之一圆面积。
> 2. $b=0$ 时两项都为零。
> 3. $dA/db=\sqrt{a^2-b^2}$，正好等于新增水平薄片的长度。

### 本地材料与知识链

- [[Ses70a_Lecture_Notes.pdf#page=1|70a 圆弓形面积与三角代换预览]]
- [[Exercise070_Problems.pdf#page=1|Exercise 070 原题]] · [[Exercise070_Solutions.pdf#page=1|官方解答]]

**小结：**三角代换不是魔法；它把勾股恒等式嵌入根式，使根式成为直角三角形的一条边。

## Session 71：Integrals involving secant, cosecant and cotangent

### 本节问题与基础恒等式

$$
\sec x=\frac1{\cos x},\quad
\tan x=\frac{\sin x}{\cos x},\quad
\csc x=\frac1{\sin x},\quad
\cot x=\frac{\cos x}{\sin x},
$$

$$
1+\tan^2x=\sec^2x,
\qquad
1+\cot^2x=\csc^2x.
$$

### 71a–71c：基本原函数的逐步推导

先把正切写成商：

$$
\int\tan x\,dx
=\int\frac{\sin x}{\cos x}\,dx
=-\ln|\cos x|+C
=\ln|\sec x|+C.
$$

正割看似没有直接换元。观察

$$
\frac d{dx}(\sec x+\tan x)
=\sec x(\sec x+\tan x),
$$

于是有意乘以 $1=(\sec x+\tan x)/(\sec x+\tan x)$：

$$
\begin{aligned}
\int\sec x\,dx
&=\int\frac{\sec x(\sec x+\tan x)}{\sec x+\tan x}\,dx\\
&=\ln|\sec x+\tan x|+C.
\end{aligned}
$$

类似地

$$
\int\csc x\,dx=-\ln|\csc x+\cot x|+C
=\ln|\csc x-\cot x|+C.
$$

### 71d：幂积分的配对策略

- $\int\tan^m x\sec^n x\,dx$ 中若 $n$ 为正偶数，留一个 $\sec^2x\,dx$，其余用 $\sec^2x=1+\tan^2x$，令 $u=\tan x$。
- 若 $m$ 为正奇数，留 $\sec x\tan x\,dx$，其余把 $\tan^2x$ 攐成 $\sec^2x-1$，令 $u=\sec x$。
- 余切与余割完全平行，但 $d(\cot x)=-\csc^2x\,dx$、$d(\csc x)=-\csc x\cot x\,dx$ 带负号。

> [!example]- 配套 Exercise 071：$\int\tan^3x\,dx$
> $$
> \begin{aligned}
> \int\tan^3x\,dx
> &=\int\tan x(\sec^2x-1)\,dx\\
> &=\int\tan x\sec^2x\,dx-\int\tan x\,dx\\
> &=\frac12\tan^2x+\ln|\cos x|+C.
> \end{aligned}
> $$
> 第一项用 $u=\tan x$；第二项用刚推导的基本原函数。

> [!warning] 易错点
> $\int\sec x\,dx$ 不是 $\ln|\sec x|$；后者的导数是 $\tan x$。对数必须写绝对值，除非已明确限制在表达式恒正的区间。

> [!question]- 三道自检题与答案
> 1. $\int\sec^4x\,dx=\tan x+\tan^3x/3+C$。
> 2. $\int\csc^2x\,dx=-\cot x+C$。
> 3. $\int\sec x\tan x\,dx=\sec x+C$。

### 本地材料与知识链

- [[Ses71a_Lecture_Notes.pdf#page=1|71a 恒等式复习]] · [[Ses71b_Lecture_Notes.pdf#page=1|71b $\int\tan x$]] · [[Ses71c_Lecture_Notes.pdf#page=1|71c $\int\sec x$]] · [[Ses71d_Lecture_Notes.pdf#page=1|71d 三角积分总结]]
- [[Exercise071_Problems.pdf#page=1|Exercise 071 原题]] · [[Exercise071_Solutions.pdf#page=1|官方解答]]

**小结：**处理 $\tan,\sec$ 的幂仍是“留出一个导数因子，再用恒等式统一变量”。

## Session 72：Trig substitution

### 本节问题与三类模板

根式中的二次式提示勾股恒等式：

| 根式 | 代换 | 根式化简 | 常用角度范围 |
|---|---|---|---|
| $\sqrt{a^2-x^2}$ | $x=a\sin\theta$ | $a\cos\theta$ | $-\pi/2\le\theta\le\pi/2$ |
| $\sqrt{a^2+x^2}$ | $x=a\tan\theta$ | $a\sec\theta$ | $-\pi/2<\theta<\pi/2$ |
| $\sqrt{x^2-a^2}$ | $x=a\sec\theta$ | $a\tan\theta$ | 按 $x$ 的符号选支 |

### 72a：完整例题

计算 $\int dx/(x^2\sqrt{1+x^2})$。令 $x=\tan\theta$：

$$
\begin{aligned}
\int\frac{dx}{x^2\sqrt{1+x^2}}
&=\int\frac{\sec^2\theta}{\tan^2\theta\sec\theta}\,d\theta\\
&=\int\frac{\cos\theta}{\sin^2\theta}\,d\theta\\
&=-\frac1{\sin\theta}+C.
\end{aligned}
$$

由直角三角形 $\tan\theta=x/1$ 得 $\sin\theta=x/\sqrt{1+x^2}$，所以

$$
\boxed{\int\frac{dx}{x^2\sqrt{1+x^2}}
=-\frac{\sqrt{1+x^2}}{x}+C.}
$$

### 72b：如何可靠回代

回代不是猜公式。若 $\theta=\arccsc x$，则 $\csc\theta=x=\text{斜边}/\text{对边}$；取斜边 $x$、对边 $1$，邻边为 $\sqrt{x^2-1}$，故

$$
\tan(\arccsc x)=\frac1{\sqrt{x^2-1}}
$$

（严格地说要结合 $x$ 的符号和反函数值域处理分支）。

### 72c：算法与选择

1. 先把二次式规范成三种模板之一。
2. 写出 $dx$，同时声明 $\theta$ 的范围。
3. 用恒等式消掉根式并完成三角积分。
4. 用三角形或代数恒等式回到 $x$。
5. 对结果求导；若是定积分，优先在代换时直接改上下限，避免回代。

> [!example]- 配套 Exercise 072：两种方法计算 $\int x\sqrt{x^2-9}\,dx$
> 三角代换 $x=3\sec\theta$ 给出
> $$
> \int27\sec^2\theta\tan^2\theta\,d\theta
> =9\tan^3\theta+C
> =\frac13(x^2-9)^{3/2}+C.
> $$
> 但直接令 $u=x^2-9$，$du=2x\,dx$，立刻得到同一答案。结论不是“三角代换总要用”，而是**先检查更简单的直接换元**。

> [!warning] 适用条件
> $\sqrt{x^2-a^2}$ 只在 $|x|\ge a$ 有实值。把 $\sqrt{a^2\sec^2\theta-a^2}$ 写成 $a\tan\theta$ 前必须选择使 $\tan\theta\ge0$ 的分支，否则应保留绝对值。

> [!question]- 三道自检题与答案
> 1. $\int dx/\sqrt{a^2-x^2}=\arcsin(x/a)+C$。
> 2. $\sqrt{a^2+x^2}$ 选 $x=a\tan\theta$，因为 $1+\tan^2\theta=\sec^2\theta$。
> 3. 定积分代换后上下限的顺序若反转，应保留负号或交换上下限并改号，不能两者都做。

### 本地材料与知识链

- [[Ses72a_Lecture_Notes.pdf#page=1|72a 完整三角代换]] · [[Ses72b_Lecture_Notes.pdf#page=1|72b 回代]] · [[Ses72c_Lecture_Notes.pdf#page=1|72c 三类模板总结]]
- [[Exercise072_Problems.pdf#page=1|Exercise 072 原题]] · [[Exercise072_Solutions.pdf#page=1|官方解答]]

**小结：**代换的目的不是引入三角函数，而是利用恒等式消去根式；角度范围保证所有平方根与符号步骤合法。

## Session 73：Completing the square

### 本节问题

三角代换模板只处理“平方加减常数”。一般二次式先配方：

$$
Ax^2+Bx+C
=A\left(x+\frac{B}{2A}\right)^2+left(C-\frac{B^2}{4A}\right),\qquad A\ne0.
$$

这来自展开右侧，不是需要另背的结论。

### 73a：课件例题

计算 $\int dx/\sqrt{x^2+4x}$。先写

$$
x^2+4x=(x+2)^2-4.
$$

令 $u=x+2=2\sec\theta$：

$$
\begin{aligned}
\int\frac{dx}{\sqrt{x^2+4x}}
&=\int\frac{2\sec\theta\tan\theta}{2\tan\theta}\,d\theta\\
&=\int\sec\theta\,d\theta\\
&=\ln|\sec\theta+\tan\theta|+C\\
&=\ln|x+2+\sqrt{x^2+4x}|+C.
\end{aligned}
$$

最后一行吸收了常数 $-\ln2$。实数定义域为 $x\le-4$ 或 $x\ge0$；在每个连通区间上原函数常数可不同。

> [!example]- 配套 Exercise 073：$\int dx/\sqrt{2x-x^2}$
> $$
> 2x-x^2=1-(x-1)^2.
> $$
> 令 $x-1=\sin\theta$，则 $dx=\cos\theta\,d\theta$，在 $[-\pi/2,\pi/2]$ 上根式为 $\cos\theta$，故
> $$
> \int\frac{dx}{\sqrt{2x-x^2}}
> =\int d\theta
> =\arcsin(x-1)+C,
> $$
> 其定义区间是 $0<x<2$。

> [!warning] 易错点
> 配方时必须先提出二次项系数。例如 $2x^2+8x+1=2[(x+2)^2-4]+1=2(x+2)^2-7$，不能直接把“8 的一半”平方后加减。

> [!question]- 三道自检题与答案
> 1. $x^2-6x+13=(x-3)^2+4$。
> 2. $-x^2+4x+5=9-(x-2)^2$，因此适合正弦代换。
> 3. $\int dx/(x^2+4x+13)^{3/2}=(x+2)/(9\sqrt{x^2+4x+13})+C$。

### 本地材料与知识链

- [[Ses73a_Lecture_Notes.pdf#page=1|73a 配方法与三角代换]]
- [[Exercise073_Problems.pdf#page=1|Exercise 073 原题]] · [[Exercise073_Solutions.pdf#page=1|官方解答]]

**小结：**配方把一般二次式翻译成“平移后的标准根式”；随后再按符号选择正弦、正切或正割代换。

## Problem Set 9

官网指定题：5B 的 9、11、13、16；5C 的 5、7、9、11；5D 的 1、2、7、10。题册与官方解答分别为 [[PSet05_Problems.pdf#page=1|Integration Techniques Problems]]、[[PSet05_Solutions.pdf#page=1|Solutions]]。

> [!example]- 5B-9　计算 $\int e^x(1+e^x)^{-1/3}\,dx$
> 令 $u=1+e^x$，$du=e^x dx$：
> $$
> \int u^{-1/3}\,du=\frac32u^{2/3}+C
> =\boxed{\frac32(1+e^x)^{2/3}+C}.
> $$
> **规则：**直接换元。常见错误是把幂积分系数写成 $2/3$。

> [!example]- 5B-11　计算 $\int\sec^2(9x)\,dx$
> 令 $u=9x$，$dx=du/9$：
> $$
> \boxed{\int\sec^2(9x)\,dx=\frac19\tan(9x)+C.}
> $$
> 验算时链式法则产生 $9$，与 $1/9$ 抵消。

> [!example]- 5B-13　计算 $\int\dfrac{x^2}{1+x^6}\,dx$
> 令 $u=x^3$，$du=3x^2dx$，且 $x^6=u^2$：
> $$
> \boxed{\int\frac{x^2}{1+x^6}\,dx=\frac13\arctan(x^3)+C.}
> $$

> [!example]- 5B-16　计算 $\int_{-1}^{1}\dfrac{\arctan x}{1+x^2}\,dx$
> 令 $u=\arctan x$，则 $du=dx/(1+x^2)$，上下限变为 $-\pi/4,\pi/4$：
> $$
> \int_{-\pi/4}^{\pi/4}u\,du=\left.\frac{u^2}{2}\right|_{-\pi/4}^{\pi/4}=\boxed{0}.
> $$
> 也可先看出被积函数是奇函数。两种方法互相检查。

> [!example]- 5C-5　计算 $\int\sin^3x\cos^2x\,dx$
> 保留 $\sin xdx$，令 $u=\cos x$：
> $$
> \boxed{-\frac13\cos^3x+\frac15\cos^5x+C.}
> $$
> 这里使用 $\sin^2x=1-\cos^2x$，并因 $du=-\sin xdx$ 得负号。

> [!example]- 5C-7　计算 $\int\sin^2(4x)\cos^2(4x)\,dx$
> $$
> \sin^2(4x)\cos^2(4x)=\frac14\sin^2(8x)=\frac18(1-\cos16x),
> $$
> 所以
> $$
> \boxed{\frac x8-\frac{\sin16x}{128}+C.}
> $$

> [!example]- 5C-9　计算 $\int\sin^3x\sec^2x\,dx$
> 写成 $(1-\cos^2x)\sin x/\cos^2x$，令 $u=\cos x$：
> $$
> -\int\frac{1-u^2}{u^2}\,du
> =-\int(u^{-2}-1)du
> =u+u^{-1}+C.
> $$
> 因此
> $$
> \boxed{\cos x+\sec x+C},
> $$
> 只在避开 $\cos x=0$ 的区间上成立。

> [!example]- 5C-11　计算 $\int\sin x\cos(2x)\,dx$
> 用 $\cos2x=2\cos^2x-1$，令 $u=\cos x$、$du=-\sin xdx$：
> $$
> -\int(2u^2-1)du
> =\boxed{\cos x-\frac23\cos^3x+C}.
> $$

> [!example]- 5D-1　计算 $\int\dfrac{dx}{(a^2-x^2)^{3/2}}$（$a>0$）
> 令 $x=a\sin\theta$，$dx=a\cos\theta d\theta$：
> $$
> \int\frac{a\cos\theta}{a^3\cos^3\theta}d\theta
> =\frac1{a^2}\int\sec^2\theta d\theta
> =\frac1{a^2}\tan\theta+C.
> $$
> 由 $\tan\theta=x/\sqrt{a^2-x^2}$：
> $$
> \boxed{\frac{x}{a^2\sqrt{a^2-x^2}}+C},\qquad |x|<a.
> $$

> [!example]- 5D-2　计算 $\int\dfrac{x^3}{\sqrt{a^2-x^2}}\,dx$
> 令 $x=a\sin\theta$ 后得到 $a^3\int\sin^3\theta d\theta$：
> $$
> a^3\left(-\cos\theta+\frac13\cos^3\theta\right)+C.
> $$
> 用 $\cos\theta=\sqrt{a^2-x^2}/a$ 回代：
> $$
> \boxed{-a^2\sqrt{a^2-x^2}+\frac13(a^2-x^2)^{3/2}+C}.
> $$
> 也可先写 $x^3=x(a^2-(a^2-x^2))$ 后直接换元，通常更短。

> [!example]- 5D-7　计算 $\int\dfrac{\sqrt{x^2-a^2}}{x^2}\,dx$
> 在 $x>a>0$ 上令 $x=a\sec\theta$：
> $$
> \int\frac{a\tan\theta}{a^2\sec^2\theta}a\sec\theta\tan\theta d\theta
> =\int(\sec\theta-\cos\theta)d\theta.
> $$
> 所以
> $$
> \boxed{\ln|x+\sqrt{x^2-a^2}|-\frac{\sqrt{x^2-a^2}}{x}+C}.
> $$
> $|x|>a$ 的另一支需按所选角区间解释，但对数绝对值使公式在各连通区间可用。

> [!example]- 5D-10　计算 $\int\dfrac{dx}{(x^2+4x+13)^{3/2}}$
> 配方为 $(x+2)^2+9$。令 $x+2=3\tan\theta$：
> $$
> \int\frac{3\sec^2\theta}{27\sec^3\theta}d\theta
> =\frac19\int\cos\theta d\theta
> =\frac19\sin\theta+C.
> $$
> 因 $\sin\theta=(x+2)/\sqrt{x^2+4x+13}$，
> $$
> \boxed{\frac{x+2}{9\sqrt{x^2+4x+13}}+C}.
> $$

**Problem Set 9 小结：**先识别直接换元；只有二次根式确实不能消去时才启用三角代换。每个答案都应通过求导检查，定积分还应检查奇偶性与上下限。

---

## Part B：Partial Fractions, Integration by Parts, Arc Length, and Surface Area

## Session 74：Integration by partial fractions

### 本节问题与前置知识

有理函数（rational function）是 $R(x)=P(x)/Q(x)$，其中 $P,Q$ 为多项式且 $Q\ne0$。[[Partial Fraction Decomposition|部分分式分解]]（partial fractions）的目标是把复杂的一个商拆成若干个原函数已知的简单商。Session 74 先处理两个条件：

1. $\deg P<\deg Q$，即真分式；
2. $Q$ 分解为互异的一次因子。

### 74a：为什么拆分有效

例如

$$
\frac{4x-1}{(x-1)(x+2)}
=\frac{A}{x-1}+\frac{B}{x+2}.
$$

乘回公共分母得多项式恒等式

$$
4x-1=A(x+2)+B(x-1).
$$

令 $x=1$ 消掉 $B$ 项，得 $3=3A$，故 $A=1$；令 $x=-2$ 消掉 $A$ 项，得 $-9=-3B$，故 $B=3$。因此

$$
\int\frac{4x-1}{(x-1)(x+2)}dx
=\ln|x-1|+3\ln|x+2|+C.
$$

### 74b–74c：遮盖法的本质与算法

对互异一次因子，系数可由“遮盖法”（cover-up method）快速求出：若

$$
\frac{P(x)}{(x-r_1)\cdots(x-r_k)}
=\sum_{j=1}^k\frac{A_j}{x-r_j},
$$

则乘以 $x-r_j$ 后令 $x=r_j$，得到

$$
A_j=\frac{P(r_j)}{\prod_{i\ne j}(r_j-r_i)}.
$$

这不是取巧，而是在多项式恒等式中选择一个使其他项为零的输入。完整流程是：先因式分解 → 写目标形式 → 求系数 → 用一个普通 $x$ 值验算 → 逐项积分。

> [!example]- 配套 Exercise 074
> 计算
> $$
> \int\frac{x^2+2x+3}{(x+1)(x+2)(x+3)}dx.
> $$
> 写成 $A/(x+1)+B/(x+2)+C/(x+3)$。分别代入 $x=-1,-2,-3$：
> $$
> A=1,\qquad B=-3,\qquad C=3.
> $$
> 因而
> $$
> \boxed{\ln|x+1|-3\ln|x+2|+3\ln|x+3|+C.}
> $$
> 用 $x=0$ 检查：原分式为 $1/2$，右侧分解为 $1-3/2+1=1/2$。

> [!warning] 易错点
> - 分母的零点不在原函数定义域；代入零点只是求多项式恒等式的系数，并非把原有理函数在该点取值。
> - 对数必须有绝对值。
> - 若分子次数不低于分母，或分母有重复/不可约二次因子，本节的简单模板不完整，需用 Session 75。

> [!question]- 三道自检题与答案
> 1. $\dfrac1{(x-1)(x+1)}=\dfrac12\left(\dfrac1{x-1}-\dfrac1{x+1}\right)$。
> 2. $\int dx/(x^2-1)=\tfrac12\ln|(x-1)/(x+1)|+C$。
> 3. 分解后随机代入一个非极点能发现大多数符号错误，但严格保证来自乘回分母后比较恒等式。

### 本地材料与知识链

- [[Ses74a_Lecture_Notes.pdf#page=1|74a 部分分式概念]] · [[Ses74b_Lecture_Notes.pdf#page=1|74b 遮盖法引入]] · [[Ses74c_Lecture_Notes.pdf#page=1|74c 遮盖法算法]]
- [[Exercise074_Problems.pdf#page=1|Exercise 074 原题]] · [[Exercise074_Solutions.pdf#page=1|官方解答]]

**小结：**部分分式先完成一个代数问题，再完成多个简单积分；求系数发生在积分之前。

## Session 75：Advanced partial fractions

### 本节问题

重复因子、不可约二次因子和假分式分别怎样处理？答案是一套不能漏项的模板。

### 75a：重复一次因子

若分母含 $(x-a)^k$，必须包含从一阶到 $k$ 阶的所有项：

$$
\frac{P(x)}{(x-a)^kS(x)}
=\frac{A_1}{x-a}+\frac{A_2}{(x-a)^2}+\cdots+\frac{A_k}{(x-a)^k}+\text{其余因子对应项}.
$$

只写最高次项无法表示一般分子。遮盖法通常只能直接求最高阶系数，其余用比较系数或代入普通值求解。

### 75b：不可约二次因子

实数范围内不能分解的 $x^2+bx+c$ 对应**一次分子**：

$$
\frac{Ax+B}{x^2+bx+c}.
$$

原因是分子次数必须低于该因子的次数；常数分子不足以覆盖所有可能。积分时把一次分子拆成“分母导数的倍数 + 常数”，前者给对数，后者配方后给反正切。

例如

$$
\frac{x-11}{(x^2+9)(x+2)}
=\frac{x-1}{x^2+9}-\frac1{x+2},
$$

所以

$$
\int\frac{x-11}{(x^2+9)(x+2)}dx
=\frac12\ln(x^2+9)-\frac13\arctan\frac x3-\ln|x+2|+C.
$$

### 75c：长除法是 Step 0

若 $\deg P\ge\deg Q$，先做多项式长除：

$$
\frac{P}{Q}=S+\frac{R}{Q},\qquad \deg R<\deg Q.
$$

例如

$$
\frac{x^3}{x^2-1}=x+\frac{x}{x^2-1},
$$

故

$$
\boxed{\int\frac{x^3}{x^2-1}dx=\frac{x^2}{2}+\frac12\ln|x^2-1|+C.}
$$

### 75d：一般算法

1. **长除**直到真分式。
2. 在实数上把分母完全分成一次因子和不可约二次因子。
3. 对每个重复一次因子列出全部幂；对每个二次因子的每一重列出线性分子。
4. 乘回公共分母，用遮盖、代值和比较系数求解。
5. 逐项积分并求导检查。

> [!example]- 配套 Exercise 075（两份）
> 第一题正是上面的二次因子例题：
> $$
> \boxed{\int\frac{x-11}{(x^2+9)(x+2)}dx
> =\frac12\ln(x^2+9)-\frac13\arctan(x/3)-\ln|x+2|+C.}
> $$
> 第二题先长除：
> $$
> \boxed{\int\frac{x^3}{x^2-1}dx
> =\frac{x^2}{2}+\frac12\ln|x^2-1|+C.}
> $$

> [!warning] 易错点
> $(x-a)^3$ 必须写三项；$(x^2+1)^2$ 必须写 $(Ax+B)/(x^2+1)+(Cx+D)/(x^2+1)^2$。任何漏项都会使系数方程无解或只对特殊分子有效。

> [!question]- 三道自检题与答案
> 1. $1/[x(x+1)^2]$ 的模板是 $A/x+B/(x+1)+C/(x+1)^2$。
> 2. $1/[(x^2+1)(x-2)]$ 的模板是 $(Ax+B)/(x^2+1)+C/(x-2)$。
> 3. $x^2/(x-1)=x+1+1/(x-1)$。

### 本地材料与知识链

- [[Ses75a_Lecture_Notes.pdf#page=1|75a 重复因子]] · [[Ses75b_Lecture_Notes.pdf#page=1|75b 二次因子]] · [[Ses75c_Lecture_Notes.pdf#page=1|75c 长除]] · [[Ses75d_Lecture_Notes.pdf#page=1|75d 综合算法]]
- [[Exercise075_Problems.pdf#page=1|Exercise 075：二次因子]] · [[Exercise075_Solutions.pdf#page=1|解答]] · [[Exercise075_Problems_2.pdf#page=1|Exercise 075：长除]] · [[Exercise075_Solutions_2.pdf#page=1|解答]]

**小结：**“先长除、后完整列项”使部分分式成为覆盖所有有理函数的确定算法。

## Session 76：[[Integration by Parts|分部积分]]

### 本节问题：公式从哪里来

乘积法则

$$
(uv)'=u'v+uv'
$$

移项并积分：

$$
\boxed{\int u\,dv=uv-\int v\,du.}
$$

定积分版本保留边界项：

$$
\boxed{\int_a^b u(x)v'(x)dx=[u(x)v(x)]_a^b-\int_a^b v(x)u'(x)dx.}
$$

公式的目标是用右边的新积分替换左边；只有 $du$ 比 $u$ 更简单且 $dv$ 容易积分时，这个替换才有价值。

### 76a–76c：选择与基本例题

常用经验 LIATE：对数（Logarithmic）→ 反三角（Inverse trig）→ 代数（Algebraic）→ 三角（Trig）→ 指数（Exponential）优先选作 $u$。它只是经验，最终标准仍是新积分是否更简单。

计算 $\int\ln x\,dx$（$x>0$），把未写出的 $1$ 当成 $dv$：

$$
u=\ln x,\quad dv=dx,
\quad du=\frac{dx}{x},\quad v=x,
$$

$$
\boxed{\int\ln x\,dx=x\ln x-x+C.}
$$

再算 $\int(\ln x)^2dx$：

$$
\begin{aligned}
\int(\ln x)^2dx
&=x(\ln x)^2-2\int\ln x\,dx\\
&=x[(\ln x)^2-2\ln x+2]+C.
\end{aligned}
$$

### 76d–76e：递推公式

令 $I_n=\int(\ln x)^n dx$，同样选择得

$$
I_n=x(\ln x)^n-nI_{n-1}.
$$

每次把指数降低 $1$，最终到 $I_0=x$。对

$$
J_n(a)=\int x^ne^{ax}dx\qquad(a\ne0),
$$

取 $u=x^n$、$dv=e^{ax}dx$，得到

$$
\boxed{J_n(a)=\frac{x^ne^{ax}}a-\frac n aJ_{n-1}(a).}
$$

> [!example]- 配套 Exercise 076：$\int x^4\cos x\,dx$
> 反复分部积分，代数幂每次降低：
> $$
> \boxed{x^4\sin x+4x^3\cos x-12x^2\sin x-24x\cos x+24\sin x+C.}
> $$
> 最可靠的检查是整式求导：相邻项产生的 $x^3\sin x,x^2\cos x,x\sin x,\cos x$ 会逐对抵消，只剩 $x^4\cos x$。

> [!warning] 易错点
> - 把 $\int v\,du$ 前的负号漏掉。
> - 选择 $dv=\ln xdx$ 等于先假定已会求原积分，形成循环。
> - 定积分只在 $uv$ 项代上下限而忘记第二个积分仍有同样上下限。

> [!question]- 三道自检题与答案
> 1. $\int xe^x dx=e^x(x-1)+C$。
> 2. $\int x\sin xdx=-x\cos x+\sin x+C$。
> 3. $\int\arctan xdx=x\arctan x-\tfrac12\ln(1+x^2)+C$。

### 本地材料与知识链

- [[Ses76a_Lecture_Notes.pdf#page=1|76a 公式推导]] · [[Ses76b_Lecture_Notes.pdf#page=1|76b $\int\ln x$]] · [[Ses76c_Lecture_Notes.pdf#page=1|76c 进阶例题]] · [[Ses76d_Lecture_Notes.pdf#page=1|76d 对数递推]] · [[Ses76e_Lecture_Notes.pdf#page=1|76e $x^ne^x$ 递推]]
- [[Exercise076_Problems.pdf#page=1|Exercise 076 原题]] · [[Exercise076_Solutions.pdf#page=1|官方解答]]

**小结：**分部积分是乘积法则的逆向工程；好的选择会让某一“复杂度指标”在每一步下降。

## Session 77：Volume of a wine glass

### 本节问题与几何模型

把 $y=e^x$ 从 $(0,1)$ 到 $(1,e)$ 的弧绕 $y$ 轴旋转，并以 $y=e$ 封口，所得“指数酒杯”的容积是多少？同一立体可用水平圆盘或竖直圆柱壳计算，答案应一致。

### 77a：水平切片——圆盘法

由 $y=e^x$ 得 $x=\ln y$。高度 $y$ 处截面是半径 $\ln y$ 的圆盘：

$$
V=\pi\int_1^e(\ln y)^2dy.
$$

用 Session 76 的递推：

$$
\int(\ln y)^2dy=y[(\ln y)^2-2\ln y+2]+C.
$$

因此

$$
V=\pi[e-2]=\boxed{\pi(e-2)}.
$$

### 77b：竖直切片——圆柱壳法

在 $x$ 处，壳半径为 $x$，周长 $2\pi x$，高度 $e-e^x$，厚度 $dx$：

$$
\begin{aligned}
V&=2\pi\int_0^1x(e-e^x)dx\\
&=2\pi\left(\frac e2-\int_0^1xe^x dx\right).
\end{aligned}
$$

分部积分给 $\int_0^1xe^xdx=[e^x(x-1)]_0^1=1$，故仍为 $\pi(e-2)$。

两种方法使用不同积分变量与技巧，却描述同一组体积微元；相等不是巧合，而是体积的唯一性。

> [!warning] 易错点
> 壳法微元是“周长 × 高 × 厚”，不是 $\pi r^2dx$。圆盘法的积分变量为 $y$，必须先把半径写成 $x=\ln y$。

> [!question]- 三道自检题与答案
> 1. 壳高为何是 $e-e^x$？因为竖片从下边界 $y=e^x$ 延伸到上盖 $y=e$。
> 2. $e-2>0$，故体积为正，且小于外包圆柱 $\pi e$。
> 3. 若上端改为 $x=b$，圆盘形式为 $\pi\int_1^{e^b}(\ln y)^2dy$。

### 本地材料与知识链

- [[Ses77a_Lecture_Notes.pdf#page=1|77a 水平切片]] · [[Ses77b_Lecture_Notes.pdf#page=1|77b 竖直壳层]]

**小结：**几何切片决定积分形式；积分技巧随后服务于计算，不能先选技巧再硬套几何。

## Session 78：Computing the length of a curve

### 本节问题：弧长公式从哪里来

把 $[a,b]$ 分成很多小区间。曲线上相邻点间的小弦长度为

$$
\Delta s_i=\sqrt{(\Delta x_i)^2+(\Delta y_i)^2}.
$$

若 $y=f(x)$ 可微，把 $\Delta y_i/\Delta x_i$ 近似为 $f'(x_i^*)$：

$$
\Delta s_i
=\sqrt{1+\left(\frac{\Delta y_i}{\Delta x_i}\right)^2}\,\Delta x_i
\approx\sqrt{1+[f'(x_i^*)]^2}\,\Delta x_i.
$$

弦长和取极限得到

$$
\boxed{L=\int_a^b\sqrt{1+[f'(x)]^2}\,dx.}
$$

严格条件通常取 $f'$ 连续；这样上述 Riemann 和确实收敛到[[Arc Length|弧长]]。

![[unit04-arc-length-element.png|820]]

### 78a–78d：逐级例题

直线 $y=mx$ 在 $0\le x\le10$ 上：

$$
L=\int_0^{10}\sqrt{1+m^2}dx=10\sqrt{1+m^2},
$$

与两端点距离完全一致。

单位圆上半圆 $y=\sqrt{1-x^2}$，$0\le x\le a<1$：

$$
y'=-\frac{x}{\sqrt{1-x^2}},
\quad
\sqrt{1+(y')^2}=\frac1{\sqrt{1-x^2}},
$$

故 $L=\arcsin a$，正是半径 $1$ 乘圆心角。

抛物线 $y=x^2$，$0\le x\le a$：

$$
L=\int_0^a\sqrt{1+4x^2}dx
=\frac a2\sqrt{1+4a^2}+\frac14\ln\left(2a+\sqrt{1+4a^2}\right).
$$

这里用 $2x=\tan\theta$；弧长即使曲线简单，也常产生对数或反双曲函数。

> [!example]- 配套 Exercise 078
> 对 $y=\ln x$，$1/10\le x\le1$：
> $$
> \boxed{L=\int_{1/10}^{1}\sqrt{1+\frac1{x^2}}dx
> =\int_{1/10}^{1}\frac{\sqrt{x^2+1}}x\,dx.}
> $$
> 因区间内 $x>0$，$\sqrt{x^2}/x=1$，无需绝对值；若区间在负半轴则不能这样写。题目只要求列式。

> [!warning] 易错点
> $ds\ne dx+dy$；勾股关系是 $ds^2=dx^2+dy^2$ 的极限记号。弧长积分中的平方根保证长度非负，不能把 $\sqrt{(f')^2}$ 随意写成 $f'$。

> [!question]- 三道自检题与答案
> 1. 水平线在 $[a,b]$ 上长度为 $b-a$。
> 2. 若改用 $x=g(y)$，则 $L=\int\sqrt{1+(dx/dy)^2}\,dy$。
> 3. 弧长不小于端点直线距离；数值结果若更小，必有错误。

### 本地材料与知识链

- [[Ses78a_Lecture_Notes.pdf#page=1|78a 弧长推导]] · [[Ses78b_Lecture_Notes.pdf#page=1|78b 直线]] · [[Ses78c_Lecture_Notes.pdf#page=1|78c 圆弧]] · [[Ses78d_Lecture_Notes.pdf#page=1|78d 抛物线]]
- [[Exercise078_Problems.pdf#page=1|Exercise 078 原题]] · [[Exercise078_Solutions.pdf#page=1|官方解答]]

**小结：**弧长是“局部勾股长度”的累加；$ds$ 是几何对象，选择 $x,y$ 或参数只是计算它的不同方式。

## Session 79：Surface area

### 本节问题：窄圆台为何给出公式

曲线 $y=f(x)\ge0$ 绕 $x$ 轴旋转。极短弧段长度为 $ds$，旋转后形成窄圆台；其面积在极限中等于圆周 $2\pi y$ 乘斜宽 $ds$：

$$
\boxed{A=\int 2\pi y\,ds
=2\pi\int_a^b f(x)\sqrt{1+[f'(x)]^2}\,dx.}
$$

绕 $y$ 轴时半径为到 $y$ 轴的距离 $|x|$。一般记忆法是

$$
dA=2\pi(\text{到旋转轴的距离})\,ds.
$$

![[unit04-surface-of-revolution.png|820]]

### 79a：抛物线旋转

$y=x^2$、$0\le x\le a$ 绕 $x$ 轴：

$$
A=2\pi\int_0^a x^2\sqrt{1+4x^2}\,dx.
$$

本节重点是从几何正确列式；真正计算需要 Session 72 的三角代换。

### 79b：球面面积的完整推导

半径 $R$ 的上半圆 $y=\sqrt{R^2-x^2}$ 绕 $x$ 轴。因为

$$
y'=-\frac{x}{\sqrt{R^2-x^2}},
\qquad
ds=\frac R{\sqrt{R^2-x^2}}dx,
$$

所以发生关键抵消：

$$
dA=2\pi y\,ds=2\pi R\,dx.
$$

球面在 $a\le x\le b$ 的带面积为

$$
A=2\pi R(b-a).
$$

取 $a=-R,b=R$ 得整个球面面积 $4\pi R^2$；取任意高 $h=b-a$，球带面积只依赖 $h$，不依赖位于球面何处。

> [!example]- 配套 Exercise 079：指数酒杯的表面积
> $y=e^x$、$0\le x\le1$ 绕 $y$ 轴，半径是 $x$，且 $ds=\sqrt{1+e^{2x}}dx$：
> $$
> \boxed{A=2\pi\int_0^1x\sqrt{1+e^{2x}}\,dx.}
> $$
> 题目只要求列式。不要把 Session 77 的壳高 $e-e^x$ 带入表面积公式；体积壳与曲面带是不同微元。

> [!warning] 易错点
> - 半径是距离，必要时写绝对值。
> - $ds$ 是斜宽，不是水平宽 $dx$。
> - 只旋转一条边界曲线时，不要误加底面或端盖；是否包含端盖由题目决定。

> [!question]- 三道自检题与答案
> 1. $y=c>0$ 在长度 $L$ 的区间绕 $x$ 轴，侧面积为 $2\pi cL$。
> 2. 单位上半圆绕 $x$ 轴给 $4\pi$，不是 $2\pi$，因为 $x$ 从 $-1$ 到 $1$。
> 3. 绕直线 $y=k$ 时半径为 $|f(x)-k|$。

### 本地材料与知识链

- [[Ses79a_Lecture_Notes.pdf#page=1|79a 旋转曲面微元]] · [[Ses79b_Lecture_Notes.pdf#page=1|79b 球面面积]]
- [[Exercise079_Problems.pdf#page=1|Exercise 079 原题]] · [[Exercise079_Solutions.pdf#page=1|官方解答]]

**小结：**[[Surface of Revolution|旋转曲面面积]]是“圆周 × 弧长微元”的累加；真正容易错的是几何半径和 $ds$，而不是最后的积分技巧。

## Problem Set 10

官网指定题：5E 的 2、3、5、6、10h；5F 的 1a、2d、再做 2b、3。题册与解答为 [[PSet05_Problems.pdf#page=4|Integration Techniques 第 5E–5F 节]]、[[PSet05_Solutions.pdf#page=12|官方解答相应页]]。

> [!example]- 5E-2　$\int\dfrac{x}{(x-2)(x+3)}dx$
> $$
> \frac{x}{(x-2)(x+3)}=\frac{2/5}{x-2}+\frac{3/5}{x+3},
> $$
> 因而
> $$
> \boxed{\frac25\ln|x-2|+\frac35\ln|x+3|+C.}
> $$

> [!example]- 5E-3　$\int\dfrac{x}{(x^2-4)(x+3)}dx$
> 分母为 $(x-2)(x+2)(x+3)$。遮盖法给
> $$
> \frac{x}{(x-2)(x+2)(x+3)}
> =\frac{1/10}{x-2}+\frac{1/2}{x+2}-\frac{3/5}{x+3}.
> $$
> 所以
> $$
> \boxed{\frac1{10}\ln|x-2|+\frac12\ln|x+2|-\frac35\ln|x+3|+C.}
> $$

> [!example]- 5E-5　$\int\dfrac{3x+2}{x(x+1)^2}dx$
> 重复因子模板为
> $$
> \frac{3x+2}{x(x+1)^2}=\frac2x-\frac2{x+1}+\frac1{(x+1)^2}.
> $$
> 逐项积分：
> $$
> \boxed{2\ln|x|-2\ln|x+1|-\frac1{x+1}+C.}
> $$

> [!example]- 5E-6　$\int\dfrac{2x-9}{(x^2+9)(x+2)}dx$
> 写 $(Ax+B)/(x^2+9)+C/(x+2)$。代 $x=-2$ 得 $C=-1$；乘回后得到 $Ax+B=x$，故
> $$
> \frac{2x-9}{(x^2+9)(x+2)}=\frac{x}{x^2+9}-\frac1{x+2}.
> $$
> 因此
> $$
> \boxed{\frac12\ln(x^2+9)-\ln|x+2|+C.}
> $$

> [!example]- 5E-10h　$\int\dfrac{x^2+1}{x^2+2x+2}dx$
> 先长除并让分子出现分母导数：
> $$
> \frac{x^2+1}{x^2+2x+2}
> =1-\frac{2x+1}{x^2+2x+2}
> =1-\frac{2x+2}{x^2+2x+2}+\frac1{(x+1)^2+1}.
> $$
> 所以
> $$
> \boxed{x-\ln(x^2+2x+2)+\arctan(x+1)+C.}
> $$
> 分母恒正，第一处对数不必写绝对值。

> [!example]- 5F-1a　$\int x^a\ln x\,dx$（$a\ne-1,x>0$）
> 取 $u=\ln x$、$dv=x^adx$：
> $$
> \boxed{\frac{x^{a+1}\ln x}{a+1}-\frac{x^{a+1}}{(a+1)^2}+C.}
> $$
> $a=-1$ 时 $v$ 的公式失效，须另做换元，结果为 $(\ln x)^2/2+C$。

> [!example]- 5F-2d　推导递推公式
> 对 $I_n(a)=\int x^ne^{ax}dx$，取 $u=x^n$、$dv=e^{ax}dx$：
> $$
> \boxed{I_n(a)=\frac{x^ne^{ax}}a-\frac n aI_{n-1}(a)},\qquad a\ne0.
> $$
> 这一步同时标明了假设 $a\ne0$ 和终点 $I_0=e^{ax}/a$。

> [!example]- 5F-2b　用递推计算 $\int x^2e^xdx$
> $$
> I_2=x^2e^x-2I_1,
> \qquad I_1=xe^x-e^x,
> $$
> 因而
> $$
> \boxed{\int x^2e^xdx=e^x(x^2-2x+2)+C.}
> $$

> [!example]- 5F-3　$\int\arcsin(4x)dx$
> 取 $u=\arcsin(4x)$、$dv=dx$：
> $$
> \begin{aligned}
> \int\arcsin(4x)dx
> &=x\arcsin(4x)-\int\frac{4x}{\sqrt{1-16x^2}}dx\\
> &=\boxed{x\arcsin(4x)+\frac14\sqrt{1-16x^2}+C}.
> \end{aligned}
> $$
> 实数范围为 $|x|\le1/4$；区间内部可直接求导检查。

**Problem Set 10 小结：**部分分式的难点是完整代数分解；分部积分的难点是让复杂度下降。写完答案后逐项求导，比重新做一遍更快地发现系数错误。

---

## Part C：Parametric Equations and Polar Coordinates

## Session 80：Parametric curves

### 本节问题与定义

[[Parametric Curve|参数曲线]]（parametric curve）把位置的两个坐标都写成第三个变量的函数：

$$
x=x(t),\qquad y=y(t),\qquad \alpha\le t\le\beta.
$$

$t$ 常表示时间，但也可以只是编号点的参数。参数化不只给出点集，还给出起点、方向、速度以及某些部分被描画多少次；同一几何曲线可以有许多不同参数化。

### 80a：从运动理解曲线

$$
x=a\cos t,\qquad y=a\sin t
$$

满足 $x^2+y^2=a^2$。当 $t=0$ 位于 $(a,0)$，且速度 $(-a\sin t,a\cos t)$ 在 $t=0$ 指向上方，所以它以恒速 $a$ 逆时针描圆。

若 $x'(t)\ne0$，链式法则给切线斜率

$$
\boxed{\frac{dy}{dx}=\frac{dy/dt}{dx/dt}}.
$$

不能把分母为零简单解释为“没有切线”；此时可能存在竖直切线，需看 $y'(t)$ 是否非零。

### 80b：参数曲线弧长

局部位移的勾股关系为

$$
ds^2=dx^2+dy^2.
$$

除以 $dt^2$ 并取非负平方根：

$$
\boxed{\frac{ds}{dt}=\sqrt{[x'(t)]^2+[y'(t)]^2}},
$$

故

$$
\boxed{L=\int_\alpha^\beta\sqrt{[x'(t)]^2+[y'(t)]^2}\,dt.}
$$

若曲线在参数区间内重复描画，公式会按实际运动路程重复计数；要算几何曲线一次的长度，必须选择只描一次的区间。

### 80c：微分记号的严格解释

$ds^2=dx^2+dy^2$ 是上述导数公式的紧凑记法，不是把有限小量当普通代数数而无需极限。合法操作可由链式法则验证：

$$
ds=\sqrt{\left(\frac{dx}{dt}\right)^2+\left(\frac{dy}{dt}\right)^2}\,dt
$$

在 $t$ 递增且 $dt>0$ 的定积分解释下成立；一般方向问题应使用 $|dt|$ 或确保上下限有序。

![[unit04-parametric-motion.png|820]]

> [!example]- 配套 Exercise 080：阿基米德螺线
> 对 $x=t\cos t,y=t\sin t$，$0\le t\le4\pi$，有 $x^2+y^2=t^2$，且极角为 $t$，故半径以恒定速率增加并逆时针绕两圈。导数为
> $$
> x'=\cos t-t\sin t,\qquad y'=\sin t+t\cos t.
> $$
> 交叉项抵消：
> $$
> (x')^2+(y')^2=1+t^2.
> $$
> 所以弧长列式为
> $$
> \boxed{L=\int_0^{4\pi}\sqrt{1+t^2}\,dt.}
> $$

> [!warning] 易错点
> 消去参数所得直角坐标方程会丢失方向、范围和重复描画信息；因此“方程相同”不等于“参数曲线完全相同”。

> [!question]- 三道自检题与答案
> 1. $x=\cos2t,y=\sin2t,0\le t\le2\pi$ 把单位圆逆时针描两次。
> 2. $x=t^2,y=t^4$ 满足 $y=x^2$，但 $t\in\mathbb R$ 时只覆盖 $x\ge0$ 且除原点外每点描两次。
> 3. $d^2y/dx^2=\dfrac{d}{dt}(dy/dx)\big/x'(t)$，前提是相关导数存在且 $x'(t)\ne0$。

### 本地材料与知识链

- [[Ses80a_Lecture_Notes.pdf#page=1|80a 参数曲线定义]] · [[Ses80b_Lecture_Notes.pdf#page=1|80b 参数弧长]] · [[Ses80c_Lecture_Notes.pdf#page=1|80c 微分记号]]
- [[Exercise080_Problems.pdf#page=1|Exercise 080 原题]] · [[Exercise080_Solutions.pdf#page=1|官方解答]]

**小结：**参数化把曲线变成运动；$x',y'$ 是速度分量，$ds/dt$ 是速率，积分速率得到路程。

## Session 81：Examples using parametrized curves

### 本节问题

怎样从参数方程识别轨迹、判断速度，并计算参数曲线旋转所得的表面积？

### 81a：非匀速椭圆参数化

$$
x=2\sin t,\qquad y=\cos t
$$

满足

$$
\frac{x^2}{4}+y^2=1,
$$

所以轨迹是椭圆。但速率

$$
\frac{ds}{dt}=\sqrt{4\cos^2t+\sin^2t}=\sqrt{1+3\cos^2t}
$$

随 $t$ 改变：在椭圆顶端 $t=0$ 速率为 $2$，在最右端 $t=\pi/2$ 速率为 $1$。三角参数的角并不必然对应沿曲线的匀速运动。

### 81b：旋转椭球面的列式

将上面椭圆的右半边绕 $y$ 轴旋转。$0\le t\le\pi$ 已从顶端走到下端并且 $x=2\sin t\ge0$，恰好描右半边一次。旋转半径为 $x$：

$$
\begin{aligned}
A
&=\int_0^\pi2\pi x\,ds\\
&=2\pi\int_0^\pi(2\sin t)\sqrt{4\cos^2t+\sin^2t}\,dt.
\end{aligned}
$$

令 $u=\cos t$ 后化为 $4\pi\int_{-1}^{1}\sqrt{1+3u^2}\,du$，后续可用三角代换。若再加入左半边，会把同一旋转曲面重复计算一次。

> [!example]- 配套 Exercise 081：屋顶落球（按题目给定的 $y''=-9.8$）
> 已知 $x(t)=t$、$y''(t)=-9.8$、$y'(0)=0$、$y(0)=5$。两次积分并依次使用初值：
> $$
> y'(t)=-9.8t,
> \qquad
> y(t)=5-4.9t^2.
> $$
> 因而路径参数化为
> $$
> \boxed{(x(t),y(t))=(t,5-4.9t^2)},
> $$
> 速率为
> $$
> \boxed{\frac{ds}{dt}=\sqrt{1+(9.8t)^2}\ \text{m/s}}.
> $$
> 只在落地前有效；落地时间由 $5-4.9t^2=0$ 得 $t=\sqrt{5/4.9}\approx1.01$ 秒。这里负号是物理方向的核心，不能在反求原函数时丢失。

> [!warning] 易错点
> 曲面公式的半径是 $|x(t)|$；选 $0\le t\le\pi$ 正是为了让右半边 $x\ge0$ 且不重复。若参数化走回头路，$ds$ 仍为正并会重复计面积。

> [!question]- 三道自检题与答案
> 1. 椭圆在 $t=\pi/2$ 的切线竖直，因为 $x'=0,y'=-1$。
> 2. 同一椭圆若 $t$ 从 $0$ 到 $2\pi$ 再绕 $y$ 轴，曲面被计算两次。
> 3. 参数曲线绕 $x$ 轴的表面积是 $2\pi\int|y(t)|\sqrt{x'^2+y'^2}dt$。

### 本地材料与知识链

- [[Ses81a_Lecture_Notes.pdf#page=1|81a 非匀速参数化]] · [[Ses81b_Lecture_Notes.pdf#page=1|81b 椭球表面积]]
- [[Exercise081_Problems.pdf#page=1|Exercise 081 原题]] · [[Exercise081_Solutions.pdf#page=1|官方解答]]

**小结：**轨迹由消参识别，运动由参数区间与导数识别；计算弧长或面积时两类信息缺一不可。

## Session 82：Polar coordinates

### 本节问题与定义

[[Polar Coordinates|极坐标]]（polar coordinates）用到原点的有向距离 $r$ 和方向角 $\theta$ 描述点：

$$
\boxed{x=r\cos\theta,\qquad y=r\sin\theta},
$$

$$
\boxed{r^2=x^2+y^2,\qquad \tan\theta=y/x}
$$

（最后一式不能单独判断象限）。

### 82a–82b：同一点有无穷多表示

$$
(r,\theta)=(r,\theta+2k\pi)=(-r,\theta+(2k+1)\pi),\qquad k\in\mathbb Z.
$$

负半径不是“负距离”，而是沿角度 $\theta$ 的反方向走 $|r|$。因此从直角坐标求角时应结合点所在象限，实际计算可用双参数反正切的思想，而非只用 $\arctan(y/x)$。

例如 $(x,y)=(1,-1)$ 可写为 $(\sqrt2,-\pi/4)$，也可写为 $(\sqrt2,7\pi/4)$ 或 $(-\sqrt2,3\pi/4)$。

### 82c：直线 $y=1$

代入 $y=r\sin\theta$：

$$
r\sin\theta=1
\quad\Longrightarrow\quad
r=\csc\theta.
$$

这条公式只在 $\sin\theta\ne0$ 时有意义；当 $0<\theta<\pi$，$r>0$ 描出整条直线一次。极坐标中简单直线可能看似复杂，反之亦然。

### 82d：过原点的偏心圆

圆心 $(a,0)$、半径 $a$ 的圆：

$$
(x-a)^2+y^2=a^2.
$$

展开后用 $x^2+y^2=r^2$、$x=r\cos\theta$：

$$
r^2-2ar\cos\theta=0
\quad\Longrightarrow\quad
r=0\quad\text{或}\quad\boxed{r=2a\cos\theta}.
$$

对 $a>0$，取 $-\pi/2\le\theta\le\pi/2$ 时 $r\ge0$ 并将圆描一次。

> [!warning] 易错点
> 从 $r^2=2ar\cos\theta$ 除以 $r$ 会暂时丢掉 $r=0$；虽然连续曲线 $r=2a\cos\theta$ 在端点仍包含原点，代数上应说明这一点。平方消元还可能引入额外点，必须回到未平方方程核查。

> [!question]- 三道自检题与答案
> 1. $x=1$ 化为 $r=\sec\theta$。
> 2. $x^2+y^2=4$ 化为 $r=2$（也可用 $r=-2$ 配合角平移）。
> 3. $r=2a\sin\theta$ 是圆心 $(0,a)$、半径 $a$ 的圆。

### 本地材料与知识链

- [[Ses82a_Lecture_Notes.pdf#page=1|82a 极坐标定义]] · [[Ses82b_Lecture_Notes.pdf#page=1|82b 简单例子]] · [[Ses82c_Lecture_Notes.pdf#page=1|82c $y=1$ 的转换]] · [[Ses82d_Lecture_Notes.pdf#page=1|82d 偏心圆]]

**小结：**极坐标把“半径—方向”作为基本变量；表示不唯一，因此范围与负半径是理解曲线的组成部分。

## Session 83：Polar coordinates，continued

### 本节问题：面积微元如何推导

半径 $r$、圆心角 $\Delta\theta$ 的扇形面积是

$$
\frac{\Delta\theta}{2\pi}\cdot\pi r^2
=\frac12r^2\Delta\theta.
$$

令角度宽度趋于零，得到

$$
\boxed{dA=\frac12r^2d\theta}.
$$

所以由 $0\le r\le f(\theta)$、$\alpha\le\theta\le\beta$ 描述的区域面积为

$$
\boxed{A=\frac12\int_\alpha^\beta[f(\theta)]^2d\theta.}
$$

若位于外曲线 $r=R(\theta)$ 与内曲线 $r=r_0(\theta)$ 之间，则

$$
A=\frac12\int_\alpha^\beta(R^2-r_0^2)d\theta.
$$

![[unit04-polar-area-element.png|820]]

### 83a–83b：偏心圆面积的完整检查

对 $r=2a\cos\theta$，选择 $-\pi/2\le\theta\le\pi/2$ 恰好描圆一次：

$$
\begin{aligned}
A
&=\frac12\int_{-\pi/2}^{\pi/2}(2a\cos\theta)^2d\theta\\
&=2a^2\int_{-\pi/2}^{\pi/2}\cos^2\theta d\theta\\
&=2a^2\cdot\frac\pi2
=\boxed{\pi a^2}.
\end{aligned}
$$

结果与半径为 $a$ 的圆面积一致，验证了积分范围和 $1/2$ 因子。

> [!warning] 易错点
> - 忘记 $1/2$。
> - 选择使整条曲线重复描画的角区间，面积被重复计算。
> - 求两曲线间面积时必须先判断每个角度谁是外半径；交点处可能需要分段。

> [!question]- 三道自检题与答案
> 1. $r=R$、$0\le\theta\le2\pi$ 给 $A=\pi R^2$。
> 2. $r=2\cos\theta$ 只算上半圆可取 $0\le\theta\le\pi/2$，面积为 $\pi/2$。
> 3. 若 $r$ 在某区间为负，$r^2$ 仍为正，但区域解释和是否重复描画必须单独检查。

### 本地材料与知识链

- [[Ses83a_Lecture_Notes.pdf#page=1|83a 极坐标面积公式]] · [[Ses83b_Lecture_Notes.pdf#page=1|83b 偏心圆面积]]

**小结：**极坐标面积是“窄扇形”的 Riemann 和；积分前最重要的工作是确定只覆盖目标区域一次的角区间。

## Session 84：Polar coordinates and graphing

### 本节目标：不靠盲目描点作图

画 $r=f(\theta)$ 时按以下顺序：

1. 找周期与对称性：$f(-\theta)$、$f(\pi-\theta)$、$f(\theta+\pi)$。
2. 找 $r=0$、$|r|$ 极值和无定义角度。
3. 分区间判断 $r$ 正负；负半径把点放到反方向。
4. 用少量关键角度描点，标出随 $\theta$ 增大的方向。
5. 确认完整曲线何时开始重复。

### 84a：$r=2a\cos\theta$ 的范围

在 $[-\pi/2,\pi/2]$ 已描完整个圆。继续增加 $\theta$ 时 $r<0$，点转到反方向，再次描同一个圆；因此 $[0,2\pi]$ 会重复，而不是产生第二个圆。

### 84b：玫瑰线 $r=\sin2\theta$

$0\le\theta\le\pi/2$ 时 $r\ge0$，从原点出发，在 $\theta=\pi/4$ 达 $r=1$，再回原点，形成第一瓣。接着 $r<0$ 把点画到相反方向；完整曲线有四瓣。这里“频率 2”不能简单理解为“两瓣”，负半径使偶数频率 $n$ 的 $r=\sin n\theta$ 通常产生 $2n$ 瓣。

### 84c：圆锥曲线与开普勒联系

对

$$
r=\frac1{1+2\cos\theta},
$$

当 $\cos\theta=-1/2$，即 $\theta=\pm2\pi/3$ 时分母趋零，曲线趋向无穷。转换为直角坐标：

$$
r+2r\cos\theta=1
\Longrightarrow r=1-2x.
$$

平方并用 $r^2=x^2+y^2$：

$$
-3x^2+y^2+4x-1=0,
$$

二次项异号，故为双曲线。平方后还应以 $r=1-2x$ 检查分支。

若天体轨迹写成焦点在原点的极坐标曲线，开普勒第二定律“等时扫过等面积”就是

$$
\frac{dA}{dt}=\frac12r^2\frac{d\theta}{dt}=\text{常数},
$$

即角动量守恒的几何形式。

> [!example]- 配套 Exercise 084：双纽线 $r^2=\cos2\theta$
> 实数点要求 $\cos2\theta\ge0$，例如
> $$
> -\frac\pi4\le\theta\le\frac\pi4,
> \qquad
> \frac{3\pi}4\le\theta\le\frac{5\pi}4
> $$
> 及其 $2\pi$ 平移。$r=0$ 在 $\theta$ 为奇数倍 $\pi/4$ 时出现；$|r|$ 最大为 $1$，在 $\theta$ 为 $\pi/2$ 的整数倍且 $\cos2\theta=1$ 的方向，即 $\theta=k\pi$（负 $r$ 表示同一两瓣）达到。曲线由左右两个在原点相接的环组成。作图时写 $r=\pm\sqrt{\cos2\theta}$，不能只取正根后误以为少一半。

> [!warning] 易错点
> 极坐标曲线的“函数图像”不是在 $\theta r$ 平面作图。$r<0$ 时必须把点旋转 $\pi$；仅看数值表而不处理负半径，常会把花瓣放错象限。

> [!question]- 三道自检题与答案
> 1. $r=1-\cos\theta$ 关于 $x$ 轴对称，因为 $f(-\theta)=f(\theta)$。
> 2. $r=\cos3\theta$ 有三瓣；一段完整描画可取 $0\le\theta\le\pi$。
> 3. $r=1/(1+e\cos\theta)$ 中 $0<e<1,e=1,e>1$ 分别对应椭圆、抛物线、双曲线（适当缩放下）。

### 本地材料与知识链

- [[Ses84a_Lecture_Notes.pdf#page=1|84a 偏心圆的完整描画]] · [[Ses84b_Lecture_Notes.pdf#page=1|84b 玫瑰线]] · [[Ses84c_Lecture_Notes.pdf#page=1|84c 圆锥曲线与开普勒定律]]
- [[Exercise084_Problems.pdf#page=1|Exercise 084 原题]] · [[Exercise084_Solutions.pdf#page=1|官方解答]]

**小结：**极坐标作图的核心不是多取点，而是周期、对称、零点、符号和重复描画。

## Problem Set 11

官网指定题：4E 的 2、3、8；4F 的 1d、4、5、8；4G 的 2、5。题册为 [[PSet04_Problems.pdf#page=4|Applications of Integration：4E–4G]]，官方解答从 [[PSet04_Solutions.pdf#page=12|4E 解答]]、[[PSet04_Solutions.pdf#page=13|4F 解答]]、[[PSet04_Solutions.pdf#page=15|4G 解答]] 开始。

> [!example]- 4E-2　消去 $x=t+1/t,\ y=t-1/t$ 中的参数
> 平方后相减：
> $$
> x^2=t^2+2+t^{-2},\qquad y^2=t^2-2+t^{-2},
> $$
> $$
> \boxed{x^2-y^2=4}.
> $$
> 这是双曲线。$t\ne0$；$t>0$ 覆盖右支 $x\ge2$，$t<0$ 覆盖左支 $x\le-2$，所以两支都能得到。

> [!example]- 4E-3　消去 $x=1+\sin t,\ y=4+\cos t$
> $$
> (x-1)^2+(y-4)^2=\sin^2t+\cos^2t=1.
> $$
> 故为圆心 $(1,4)$、半径 $1$ 的圆。$t=0$ 时在顶点 $(1,5)$，速度 $(x',y')=(1,0)$ 向右，故随 $t$ 增大顺时针：
> $$
> \boxed{(x-1)^2+(y-4)^2=1\quad\text{（顺时针）}}.
> $$

> [!example]- 4E-8　钟面上的蜗牛
> 令 $t$ 为正午后经过的小时数，$0\le t\le1$；原点在钟心，$x$ 向右、$y$ 向上。蜗牛匀速沿长度为 $1$ 的时针向外，故到中心距离为 $t$；时针从正 $y$ 轴顺时针转角 $\pi t/6$：
> $$
> \boxed{x(t)=t\sin\frac{\pi t}{6},\qquad y(t)=t\cos\frac{\pi t}{6},\qquad0\le t\le1.}
> $$
> 检查：$t=0$ 在中心；$t=1$ 在一点钟方向且半径为 $1$。

> [!example]- 4F-1d　$y=\frac13(2+x^2)^{3/2},\ 1\le x\le2$ 的弧长
> $$
> y'=x\sqrt{2+x^2},
> \quad
> 1+(y')^2=1+2x^2+x^4=(x^2+1)^2.
> $$
> 区间内 $x^2+1>0$，故
> $$
> L=\int_1^2(x^2+1)dx
> =\left[\frac{x^3}{3}+x\right]_1^2
> =\boxed{\frac{10}{3}}.
> $$

> [!example]- 4F-4　$x=t^2,y=t^3,0\le t\le2$ 的弧长
> $$
> \frac{ds}{dt}=\sqrt{4t^2+9t^4}=t\sqrt{4+9t^2}
> $$
> （因 $t\ge0$）。令 $u=4+9t^2$：
> $$
> L=\left.\frac1{27}(4+9t^2)^{3/2}\right|_0^2
> =\boxed{\frac{80\sqrt{10}-8}{27}}.
> $$

> [!example]- 4F-5　为 4E-2 的曲线在 $1\le t\le2$ 列弧长积分
> $$
> x'=1-t^{-2},\qquad y'=1+t^{-2},
> $$
> $$
> (x')^2+(y')^2=2+2t^{-4}.
> $$
> 因 $t>0$，
> $$
> \boxed{L=\int_1^2\sqrt{2+\frac2{t^4}}dt
> =\sqrt2\int_1^2\frac{\sqrt{t^4+1}}{t^2}dt.}
> $$
> 题目只要求化简列式，不要求求值。

> [!example]- 4F-8　$x=e^t\cos t,y=e^t\sin t,0\le t\le10$
> $$
> x'=e^t(\cos t-\sin t),\quad y'=e^t(\sin t+\cos t),
> $$
> $$
> (x')^2+(y')^2=2e^{2t}.
> $$
> 因而
> $$
> \boxed{L=\sqrt2\int_0^{10}e^tdt=\sqrt2(e^{10}-1)}.
> $$

> [!example]- 4G-2　$y=1-2x$ 第一象限线段绕 $x$ 轴
> $0\le x\le1/2$，$y'=-2$，所以 $ds=\sqrt5dx$：
> $$
> A=2\pi\sqrt5\int_0^{1/2}(1-2x)dx
> =2\pi\sqrt5\left[x-x^2\right]_0^{1/2}
> =\boxed{\frac{\pi\sqrt5}{2}}.
> $$
> 这也等于半径 $1$、母线长 $\sqrt5/2$ 的圆锥侧面积 $\pi r\ell$。

> [!example]- 4G-5　$y=x^2,0\le x\le4$ 绕 $y$ 轴
> 半径为 $x$，$ds=\sqrt{1+4x^2}dx$：
> $$
> \begin{aligned}
> A&=2\pi\int_0^4x\sqrt{1+4x^2}dx\\
> &=\frac\pi6\left[(1+4x^2)^{3/2}\right]_0^4\\
> &=\boxed{\frac\pi6(65\sqrt{65}-1)}.
> \end{aligned}
> $$

**Problem Set 11 小结：**先确定参数范围和旋转半径，再写导数与 $ds$；漂亮的代数平方、交叉项抵消或几何公式都是检查答案的重要信号。

---

## Exam 4

## Session 85：Review for Exam 4

### 考试范围与策略

课件给出的权重约为：积分技巧 55%，参数曲线与几何 45%。考场上先完成以下识别：

1. 三角幂：检查奇偶与可配对导数。
2. 根式二次式：先配方，再选三角代换。
3. 有理函数：先长除与因式分解，再列完整部分分式。
4. 对数/反三角/多项式乘指数：考虑分部积分。
5. 弧长/表面积：先写 $ds$ 和半径，再代入参数。
6. 极坐标：先确定只描目标区域一次的角区间。

### 85a–85b：边界说明

Exam 4 不要求极坐标弧长；但要求参数曲线弧长、旋转表面积和极坐标面积。题目不一定说明应使用哪种积分技巧，因此应按上面的结构诊断，不从结果倒猜方法。

### 85c：综合例题 $\int x\arctan x\,dx$

反正切被微分后简化，故取

$$
u=\arctan x,quad dv=x\,dx,
\quad du=\frac{dx}{1+x^2},quad v=\frac{x^2}{2}.
$$

$$
\int x\arctan xdx
=\frac{x^2}{2}\arctan x-\frac12\int\frac{x^2}{1+x^2}dx.
$$

长除式

$$
\frac{x^2}{1+x^2}=1-\frac1{1+x^2}
$$

给出

$$
\boxed{\int x\arctan xdx
=\frac{x^2-1}{2}\arctan x-\frac x2+C.}
$$

这道题把分部积分、有理式化简与反三角原函数串在一起。

> [!question]- 三道自检题与答案
> 1. $\int x^3/(x^2+1)dx$ 应先长除为 $x-x/(x^2+1)$，答案 $x^2/2-\tfrac12\ln(x^2+1)+C$。
> 2. 参数曲面公式必须使用速率 $\sqrt{x'^2+y'^2}$，不能用带符号的 $dx/dt$。
> 3. 最后五分钟优先检查：链式常数、对数绝对值、定积分上下限、表面积半径和极坐标重复描画。

### 本地材料与知识链

- [[Ses85a_Lecture_Notes.pdf#page=1|85a 考试范围]] · [[Ses85b_Lecture_Notes.pdf#page=1|85b 学生问答]] · [[Ses85c_Lecture_Notes.pdf#page=1|85c 综合积分例题]]

**小结：**Exam 4 真正考查的是识别结构与串联方法；先写“为什么选它”，通常就能避免走进死路。

## Session 86：Materials for Exam 4

本地没有 `Ses86` 讲义；按官方结构，本节由 [[Exam4_Problems.pdf#page=1|Exam 4 原题]] 与 [[Exam4_Solutions.pdf#page=1|官方解答]] 组成。以下逐题完整解答。

### 题 1：$\int_1^4\sqrt t\ln t\,dt$

取 $u=\ln t$、$dv=t^{1/2}dt$，则 $du=dt/t$、$v=\tfrac23t^{3/2}$：

$$
\begin{aligned}
I
&=\left[\frac23t^{3/2}\ln t\right]_1^4
-\frac23\int_1^4t^{1/2}dt\\
&=\frac{16}{3}\ln4-\frac49\left[t^{3/2}\right]_1^4\\
&=\boxed{\frac{16}{3}\ln4-\frac{28}{9}}.
\end{aligned}
$$

检查：被积函数在 $[1,4]$ 非负，数值约 $4.28>0$。

### 题 2：$\int_0^{\pi/4}\tan^4\theta\sec^6\theta\,d\theta$

为 $u=\tan\theta$ 留出一个 $\sec^2\theta d\theta$，其余 $\sec^4\theta=(1+\tan^2\theta)^2$。上下限变为 $0,1$：

$$
\begin{aligned}
I
&=\int_0^1u^4(1+u^2)^2du\\
&=\int_0^1(u^4+2u^6+u^8)du\\
&=\frac15+\frac27+\frac19
=\boxed{\frac{188}{315}}.
\end{aligned}
$$

常见错误是把六个 $\sec$ 全部替换，从而没有留下 $du$。

### 题 3：$\int\dfrac{10}{(x-1)(x^2+9)}dx$

一次因子加不可约二次因子的模板为

$$
\frac{10}{(x-1)(x^2+9)}
=\frac A{x-1}+\frac{Bx+C}{x^2+9}.
$$

代 $x=1$ 得 $A=1$；乘回并比较可得 $B=-1,C=-1$：

$$
\frac{10}{(x-1)(x^2+9)}
=\frac1{x-1}-\frac{x+1}{x^2+9}.
$$

逐项积分：

$$
\boxed{\ln|x-1|-\frac12\ln(x^2+9)-\frac13\arctan\frac x3+C.}
$$

答案只在不跨越 $x=1$ 的区间上作为一个原函数使用。

### 题 4：$\int\dfrac{dx}{(5-4x-x^2)^{5/2}}$

先配方：

$$
5-4x-x^2=9-(x+2)^2.
$$

实数被积函数要求 $-5<x<1$。令 $x+2=3\sin\theta$，则 $dx=3\cos\theta d\theta$：

$$
I=\frac1{81}\int\sec^4\theta d\theta.
$$

再令 $u=\tan\theta$，$du=\sec^2\theta d\theta$：

$$
\int\sec^4\theta d\theta
=\int(1+u^2)du
=\tan\theta+\frac13\tan^3\theta+C.
$$

由直角三角形

$$
\tan\theta=\frac{x+2}{\sqrt{5-4x-x^2}},
$$

所以

$$
\boxed{
I=\frac1{81}\left[
\frac{x+2}{\sqrt{5-4x-x^2}}
+\frac{(x+2)^3}{3(5-4x-x^2)^{3/2}}
\right]+C.}
$$

这里最易漏掉的是 $9^{5/2}=3^5=243$，与 $dx$ 的系数 $3$ 相除后才得到 $1/81$。

### 题 5a：列出 $x=y+y^3$ 从 $y=1$ 到 $y=4$ 的弧长

以 $y$ 为自变量最直接：

$$
\frac{dx}{dy}=1+3y^2,
\qquad
ds=\sqrt{1+\left(\frac{dx}{dy}\right)^2}dy.
$$

所以

$$
\boxed{L=\int_1^4\sqrt{1+(1+3y^2)^2}\,dy.}
$$

题目只要求列式；不要为了形式熟悉而强行解出 $y(x)$。

### 题 5b：参数曲线旋转曲面的列式

$$
x=a\cos^3t,\qquad y=a\sin^3t,\qquad 0\le t\le\frac\pi2
$$

绕 $x$ 轴。区间内 $\sin t,\cos t\ge0$。导数为

$$
x'=-3a\cos^2t\sin t,
\qquad
y'=3a\sin^2t\cos t.
$$

因此

$$
\begin{aligned}
\frac{ds}{dt}
&=3|a|\sqrt{\cos^4t\sin^2t+\sin^4t\cos^2t}\\
&=3|a|\sin t\cos t.
\end{aligned}
$$

半径为 $|y|=|a|\sin^3t$，故

$$
\boxed{A=6\pi a^2\int_0^{\pi/2}\sin^4t\cos t\,dt.}
$$

题目要求“列出但不计算”。写 $a^2$ 而非假设 $a>0$，使任意实常数 $a$ 时面积仍非负。

> [!success] Exam 4 最终检查表
> - 题 1：分部积分边界项与幂指数。
> - 题 2：留出 $\sec^2\theta d\theta$ 并换上下限。
> - 题 3：二次因子上方必须是线性分子。
> - 题 4：先配方、声明定义域、控制根号符号。
> - 题 5：弧长用 $ds$，曲面再乘 $2\pi\times$ 到旋转轴的距离。

---

## 全章知识链与公式总表

$$
\text{识别结构}
\longrightarrow
\text{合法变形或几何微元}
\longrightarrow
\text{已知原函数}
\longrightarrow
\text{回代、范围与求导检查}.
$$

| 主题 | 核心公式 |
|---|---|
| 奇次三角幂 | 留一个导数因子，其余用 $\sin^2x+\cos^2x=1$ |
| 偶次三角幂 | $\sin^2x=(1-\cos2x)/2$，$\cos^2x=(1+\cos2x)/2$ |
| 三角代换 | $a^2-x^2\leftrightarrow a\sin\theta$；$a^2+x^2\leftrightarrow a\tan\theta$；$x^2-a^2\leftrightarrow a\sec\theta$ |
| 部分分式 | 长除 → 因式分解 → 完整列项 → 求系数 |
| 分部积分 | $\int u\,dv=uv-\int v\,du$ |
| 参数弧长 | $L=\int\sqrt{x'^2+y'^2}\,dt$ |
| 旋转曲面 | $A=\int2\pi(\text{到轴距离})\,ds$ |
| 极坐标转换 | $x=r\cos\theta,y=r\sin\theta,r^2=x^2+y^2$ |
| 极坐标面积 | $A=\tfrac12\int r^2d\theta$ |

> [!tip] 一遍读懂后的复习方式
> 合上推导，只看每节“本节问题”，先口述为什么选择该方法；再独立做三个自检题；最后用 Problem Set 与 Exam 4 检查能否在没有方法提示时完成识别。若答案出错，优先回查定义域、绝对值、参数范围和重复描画，而不是只查代数运算。

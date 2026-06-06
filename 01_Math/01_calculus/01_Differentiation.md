---
aliases:
  - Differentiation
  - Derivatives
  - 导数
  - 求导
tags:
  - math/calculus
  - course/mit-ocw
  - calculus/differentiation
科目: Calculus
---

# Unit outline 课程大纲

MIT 18.01 的 Unit 1 先回答两个问题：

1. **A. What is a derivative? 什么是导数？**
   - Geometric interpretation 几何直观。
   - Physical interpretation 物理直观。
   - Importance to all measurements 对测量和敏感性分析的重要性。

2. **B. How to differentiate any function 如何对任意函数求导？**
   - 幂函数、三角函数、指数函数、对数函数。
   - 加减、常数倍、乘积、商、复合函数。
   - 隐函数、反函数、对数求导。

导数的核心概念可以压缩成一句话：

$$
f'(x_0)=\lim_{\Delta x\to 0}\frac{f(x_0+\Delta x)-f(x_0)}{\Delta x}
$$

其中

$$
\frac{\Delta f}{\Delta x}
=\frac{f(x_0+\Delta x)-f(x_0)}{\Delta x}
$$

叫做 **difference quotient 差商**。先算差商，再让 $\Delta x\to 0$，得到瞬时变化率。

Related: [[Derivative|导数]]、[[Limit|极限]]。

# A. What is a derivative? 什么是导数？

先建立三个直觉：几何上它是切线斜率，物理上它是瞬时变化率，在测量问题里它描述一个变量对另一个变量的敏感程度。

## A.1 geometric interpretation 几何直观

### A.1.1 从割线到切线

设曲线上固定一点

$$
P=(x_0,f(x_0))
$$

再取附近一点

$$
Q=(x_0+\Delta x,f(x_0+\Delta x))
$$

连接 $P,Q$ 得到一条 **secant line 割线**。割线斜率是

$$
\frac{\Delta f}{\Delta x}
=\frac{f(x_0+\Delta x)-f(x_0)}{\Delta x}
$$

当 $Q$ 沿着曲线靠近 $P$，即 $\Delta x\to 0$ 时，割线的极限位置就是 **tangent line 切线**。因此：

$$
\text{derivative at }x_0
=\text{slope of tangent line at }x_0
$$

即

$$
f'(x_0)=\lim_{\Delta x\to 0}\frac{\Delta f}{\Delta x}
$$

Related: [[geometric interpretation of derivative|导数的几何意义]]。

### A.1.2 为什么不能直接令 $\Delta x=0$

如果直接把 $\Delta x=0$ 代入差商，会得到

$$
\frac{0}{0}
$$

这不是一个合法数值。求导的关键不是“代入 0”，而是：

1. 先保持 $\Delta x\neq 0$。
2. 化简差商，消去导致 $0/0$ 的因子。
3. 再取极限 $\Delta x\to 0$。

### A.1.3 例子：$f(x)=\frac{1}{x}$

对

$$
f(x)=\frac{1}{x}
$$

在 $x_0$ 处计算差商：

$$
\frac{\Delta f}{\Delta x}
=\frac{\frac{1}{x_0+\Delta x}-\frac{1}{x_0}}{\Delta x}
$$

通分：

$$
\frac{\Delta f}{\Delta x}
=\frac{1}{\Delta x}\cdot
\frac{x_0-(x_0+\Delta x)}{(x_0+\Delta x)x_0}
$$

$$
=\frac{1}{\Delta x}\cdot
\frac{-\Delta x}{(x_0+\Delta x)x_0}
$$

因为在取极限之前 $\Delta x\neq 0$，所以可以约掉：

$$
\frac{\Delta f}{\Delta x}
=\frac{-1}{(x_0+\Delta x)x_0}
$$

令 $\Delta x\to 0$：

$$
f'(x_0)=\lim_{\Delta x\to 0}\frac{-1}{(x_0+\Delta x)x_0}
=-\frac{1}{x_0^2}
$$

所以

$$
\frac{d}{dx}\left(\frac{1}{x}\right)=-\frac{1}{x^2}
$$

### A.1.4 切线方程

如果已知曲线 $y=f(x)$ 在 $x_0$ 处可导，那么点

$$
(x_0,y_0)=(x_0,f(x_0))
$$

处的切线方程是

$$
y-y_0=f'(x_0)(x-x_0)
$$

对 $f(x)=\frac{1}{x}$，因为

$$
y_0=\frac{1}{x_0},\qquad f'(x_0)=-\frac{1}{x_0^2}
$$

所以切线方程为

$$
y-\frac{1}{x_0}
=-\frac{1}{x_0^2}(x-x_0)
$$

## A.2 physical interpretation 物理直观

### A.2.1 平均变化率与瞬时变化率

差商

$$
\frac{\Delta y}{\Delta x}
$$

表示 **average rate of change 平均变化率**。

当 $\Delta x\to 0$ 时，平均变化率的极限变成 **instantaneous rate of change 瞬时变化率**：

$$
\frac{\Delta y}{\Delta x}\to \frac{dy}{dx}
$$

### A.2.2 Pumpkin drop 例子

MIT lecture 用南瓜从楼顶掉落说明速度。

高度函数：

$$
y=400-16t^2
$$

平均速度是

$$
\frac{\Delta y}{\Delta t}
$$

而瞬时速度是

$$
\frac{dy}{dt}
$$

求导：

$$
y'=-32t
$$

当南瓜落地时：

$$
400-16t^2=0
$$

所以

$$
t=5
$$

落地瞬间速度为

$$
y'(5)=-32\cdot 5=-160\ \text{ft/s}
$$

负号表示高度 $y$ 在下降；速度大小是 $160\ \text{ft/s}$。

## A.3 importance to all measurements 对所有测量的重要性

导数不只是在几何图像或物理运动里出现。只要一个量依赖另一个量，就可以问：

$$
\text{一个变量的小变化，会导致另一个变量变化多少？}
$$

这就是测量、误差传播和敏感性分析里的导数直觉。

常见例子：

| 变量 | 导数 | 含义 |
|---|---:|---|
| $q$ = charge 电荷 | $\frac{dq}{dt}$ | current 电流 |
| $s$ = distance 距离 | $\frac{ds}{dt}$ | speed 速度 |
| $T$ = temperature 温度 | $\frac{dT}{dx}$ | temperature gradient 温度梯度 |
| $L$ = measured distance 测量距离 | $\frac{dL}{dh}$ | 高度误差对距离估计的影响 |

MIT lecture 提到 GPS 的例子：如果卫星信号只能把高度 $h$ 测到某个精度，那么我们还要知道这个误差会怎样传到距离 $L$。这类问题本质上就是研究

$$
\frac{\Delta L}{\Delta h}\quad\text{or}\quad \frac{dL}{dh}
$$

所以导数在 economics, political science, finance, physics 等依赖测量和模型敏感性的领域都重要。

# B. How to differentiate any function 如何对任意函数求导

Lecture 1 把 “How to differentiate any function” 放在 Unit 1 的第二个大问题。后面几讲的目标就是建立一套求导工具：先用极限保证定义可用，再推导基本函数和组合规则，最后处理隐函数、反函数、指数、对数和对数求导。

## B.1 limits and continuity 极限与连续

### B.1.1 导数为什么需要极限

导数不是普通代入，而是一个极限：

$$
f'(x_0)=\lim_{x\to x_0}
\frac{f(x)-f(x_0)}{x-x_0}
$$

注意：取极限时默认 $x\neq x_0$。所以在化简过程中可以除以 $x-x_0$，但最后才让 $x$ 靠近 $x_0$。

### B.1.2 连续性

函数 $f$ 在 $x_0$ 处连续的意思是：

$$
\lim_{x\to x_0}f(x)=f(x_0)
$$

直观上：从左右靠近 $x_0$ 时，函数值真的靠近 $f(x_0)$，图像在这里没有断裂。

常见不连续：

- **removable discontinuity 可去间断**：极限存在，但函数值没定义或函数值不等于极限。
- **jump discontinuity 跳跃间断**：左右极限都存在，但不相等。
- **infinite discontinuity 无穷间断**：函数值趋向 $\pm\infty$。
- **oscillatory discontinuity 振荡间断**：靠近一点时反复振荡，极限不存在。

### B.1.3 可导推出连续

定理：

如果 $f$ 在 $x_0$ 处可导，那么 $f$ 在 $x_0$ 处连续。

证明思路：

$$
f(x)-f(x_0)
=\frac{f(x)-f(x_0)}{x-x_0}(x-x_0)
$$

取 $x\to x_0$：

$$
\lim_{x\to x_0}(f(x)-f(x_0))
=f'(x_0)\cdot 0=0
$$

所以

$$
\lim_{x\to x_0}f(x)=f(x_0)
$$

注意反过来不一定成立：连续不一定可导。

### B.1.4 两个重要三角极限

Lecture 2 用单位圆直观说明：

$$
\lim_{\theta\to 0}\frac{\sin\theta}{\theta}=1
$$

$$
\lim_{\theta\to 0}\frac{1-\cos\theta}{\theta}=0
$$

这里 $\theta$ 必须用 **radians 弧度**，不是角度。

这两个极限是推导 $\sin x$ 和 $\cos x$ 导数的基础。

## B.2 differentiation rules 求导规则

### B.2.1 幂函数

对正整数 $n$：

$$
\frac{d}{dx}x^n=nx^{n-1}
$$

直觉来自二项式展开：

$$
(x+\Delta x)^n=x^n+n(\Delta x)x^{n-1}+O((\Delta x)^2)
$$

代入差商后，高阶项在 $\Delta x\to 0$ 时消失，留下

$$
nx^{n-1}
$$

这个公式后来可以推广到任意实数 $r$：

$$
\frac{d}{dx}x^r=rx^{r-1}
$$

### B.2.2 线性规则

如果 $u=u(x)$，$v=v(x)$，$c$ 是常数：

$$
(u+v)'=u'+v'
$$

$$
(cu)'=cu'
$$

例子：

$$
\frac{d}{dx}(x^2+3x^{10})
=2x+30x^9
$$

### B.2.3 三角函数

由三角极限可以推出：

$$
\frac{d}{dx}\sin x=\cos x
$$

$$
\frac{d}{dx}\cos x=-\sin x
$$

进一步：

$$
\frac{d}{dx}\tan x=\sec^2 x
$$

$$
\frac{d}{dx}\sec x=\sec x\tan x
$$

### B.2.4 Product rule 积法则

如果 $u=u(x)$，$v=v(x)$：

$$
(uv)'=u'v+uv'
$$

这条规则不能写成 $u'v'$。乘积求导时，一个因子变，另一个先保持不变，然后反过来再加一次。

例子：

$$
\frac{d}{dx}(x^2\sin x)
=2x\sin x+x^2\cos x
$$

### B.2.5 Quotient rule 商法则

如果 $v\neq 0$：

$$
\left(\frac{u}{v}\right)'
=\frac{u'v-uv'}{v^2}
$$

也可以把

$$
\frac{u}{v}=uv^{-1}
$$

然后用积法则和链式法则推出来。

## B.3 chain rule and higher derivatives 链式法则与高阶导数

### B.3.1 Chain rule 链式法则

如果

$$
y=f(x),\qquad x=g(t)
$$

那么

$$
\frac{dy}{dt}=\frac{dy}{dx}\frac{dx}{dt}
$$

等价写法：

$$
\frac{d}{dt}f(g(t))=f'(g(t))g'(t)
$$

例子：

$$
y=\sin(t^2)
$$

令 $x=t^2$，$y=\sin x$，则

$$
\frac{dy}{dt}
=\frac{dy}{dx}\frac{dx}{dt}
=\cos x\cdot 2t
=2t\cos(t^2)
$$

### B.3.2 函数复合不满足交换律

如果

$$
f(x)=\sin x,\qquad g(x)=x^2
$$

那么

$$
(f\circ g)(x)=f(g(x))=\sin(x^2)
$$

但

$$
(g\circ f)(x)=g(f(x))=(\sin x)^2
$$

一般来说：

$$
f\circ g\neq g\circ f
$$

### B.3.3 Higher derivatives 高阶导数

一阶导数：

$$
f'(x)=\frac{df}{dx}
$$

二阶导数：

$$
f''(x)=\frac{d^2f}{dx^2}
$$

三阶导数：

$$
f'''(x)=\frac{d^3f}{dx^3}
$$

$n$ 阶导数：

$$
f^{(n)}(x)=\frac{d^nf}{dx^n}
$$

例子：

$$
D^n x^n=n!
$$

其中

$$
n!=n(n-1)(n-2)\cdots 2\cdot 1
$$

## B.4 implicit differentiation and inverse functions 隐函数求导与反函数求导

### B.4.1 隐函数求导

有些关系不方便直接写成 $y=f(x)$。例如：

$$
x^2+y^2=1
$$

可以两边同时对 $x$ 求导。注意 $y$ 是 $x$ 的函数，所以对 $y^2$ 求导要用链式法则：

$$
\frac{d}{dx}(y^2)=2y\frac{dy}{dx}
$$

因此

$$
2x+2y\frac{dy}{dx}=0
$$

解得

$$
\frac{dy}{dx}=-\frac{x}{y}
$$

### B.4.2 更一般的隐函数例子

设

$$
y^3+xy^2+1=0
$$

两边求导：

$$
3y^2\frac{dy}{dx}+y^2+2xy\frac{dy}{dx}=0
$$

把含 $\frac{dy}{dx}$ 的项合并：

$$
\left(3y^2+2xy\right)\frac{dy}{dx}=-y^2
$$

所以

$$
\frac{dy}{dx}
=-\frac{y^2}{3y^2+2xy}
$$

### B.4.3 反函数求导

如果

$$
y=f(x)
$$

且 $x=f^{-1}(y)$，那么反函数的导数满足：

$$
\frac{d}{dy}f^{-1}(y)=\frac{1}{\frac{dy}{dx}}
$$

也就是说：反函数图像是原函数关于直线 $y=x$ 的镜像，斜率会变成倒数。

例子：

$$
y=\arctan x
$$

等价于

$$
\tan y=x
$$

两边对 $x$ 求导：

$$
\sec^2 y\cdot \frac{dy}{dx}=1
$$

所以

$$
\frac{dy}{dx}=\frac{1}{\sec^2 y}=\cos^2 y
$$

因为 $\tan y=x$，可由直角三角形得到：

$$
\cos^2 y=\frac{1}{1+x^2}
$$

因此

$$
\frac{d}{dx}\arctan x=\frac{1}{1+x^2}
$$

## B.5 exponentials, logarithms, and logarithmic differentiation 指数、对数与对数求导

### B.5.1 指数函数的导数

对 $a>1$，考虑

$$
\frac{d}{dx}a^x
$$

根据导数定义：

$$
\frac{d}{dx}a^x
=\lim_{\Delta x\to 0}\frac{a^{x+\Delta x}-a^x}{\Delta x}
$$

提出 $a^x$：

$$
=a^x\lim_{\Delta x\to 0}\frac{a^{\Delta x}-1}{\Delta x}
$$

令

$$
M(a)=\lim_{\Delta x\to 0}\frac{a^{\Delta x}-1}{\Delta x}
$$

则

$$
\frac{d}{dx}a^x=M(a)a^x
$$

MIT lecture 选择一个特殊底数 $e$，使得

$$
M(e)=1
$$

于是

$$
\frac{d}{dx}e^x=e^x
$$

### B.5.2 自然对数

自然对数是 $e^x$ 的反函数。

如果

$$
y=e^x
$$

那么

$$
x=\ln y
$$

等价地，如果

$$
w=\ln x
$$

那么

$$
e^w=x
$$

两边对 $x$ 求导：

$$
e^w\frac{dw}{dx}=1
$$

因为 $e^w=x$，所以

$$
\frac{dw}{dx}=\frac{1}{x}
$$

即

$$
\frac{d}{dx}\ln x=\frac{1}{x}
$$

### B.5.3 任意底数指数函数

因为

$$
a=e^{\ln a}
$$

所以

$$
a^x=e^{x\ln a}
$$

用链式法则：

$$
\frac{d}{dx}a^x
=\ln(a)e^{x\ln a}
$$

即

$$
\frac{d}{dx}a^x=(\ln a)a^x
$$

### B.5.4 Logarithmic differentiation 对数求导

对数求导适合处理“变量在底数和指数里都出现”的函数。

核心公式：

如果

$$
u=f(x)
$$

那么

$$
(\ln f)'=\frac{f'}{f}
$$

因此

$$
f'=f(\ln f)'
$$

例子：

$$
f(x)=x^x
$$

先取对数：

$$
\ln f=x\ln x
$$

两边求导：

$$
(\ln f)'=\ln x+1
$$

所以

$$
f'=f(\ln f)'=x^x(\ln x+1)
$$

即

$$
\frac{d}{dx}x^x=x^x(\ln x+1)
$$

## B.6 exam review checklist 复习清单

### B.6.1 通用求导规则

需要熟练掌握：

$$
(u+v)'=u'+v'
$$

$$
(cu)'=cu'
$$

$$
(uv)'=u'v+uv'
$$

$$
\left(\frac{u}{v}\right)'
=\frac{u'v-uv'}{v^2}
$$

$$
\frac{d}{dx}f(u(x))=f'(u(x))u'(x)
$$

### B.6.2 常见函数求导

| 函数 | 导数 |
|---:|---:|
| $x^r$ | $rx^{r-1}$ |
| $\sin x$ | $\cos x$ |
| $\cos x$ | $-\sin x$ |
| $\tan x$ | $\sec^2 x$ |
| $\sec x$ | $\sec x\tan x$ |
| $e^x$ | $e^x$ |
| $a^x$ | $(\ln a)a^x$ |
| $\ln x$ | $\frac{1}{x}$ |
| $\arctan x$ | $\frac{1}{1+x^2}$ |
| $\arcsin x$ | $\frac{1}{\sqrt{1-x^2}}$ |

### B.6.3 解题时先识别结构

拿到一个求导题，先问：

1. 是基本函数吗？
2. 是加减或常数倍吗？
3. 是乘积吗？
4. 是商吗？
5. 是复合函数吗？
6. $y$ 是否被隐含在方程里？
7. 是否是反函数？
8. 是否适合先取 $\ln$？

例子：

$$
\frac{d}{dx}e^{x\arctan x}
$$

这是指数函数加复合函数。令

$$
u=x\arctan x
$$

则

$$
\frac{d}{dx}e^u=e^u u'
$$

而

$$
u'=\arctan x+x\cdot \frac{1}{1+x^2}
$$

所以

$$
\frac{d}{dx}e^{x\arctan x}
=e^{x\arctan x}
\left(
\arctan x+\frac{x}{1+x^2}
\right)
$$

## B.7 这一章最容易混的点

### B.7.1 导数不是 $\frac{0}{0}$

导数来自差商的极限。$\frac{0}{0}$ 只是直接代入时出现的未定式，不是答案。

### B.7.2 切线不是“只碰一次”的线

切线的本质是割线的极限，不是“和曲线只有一个交点”的线。

### B.7.3 可导一定连续，但连续不一定可导

可导比连续更强。尖点、折点、竖直切线处可能连续但不可导。

### B.7.4 链式法则是结构识别，不是死记

只要看到“函数套函数”，就要找内层 $u(x)$ 和外层 $f(u)$：

$$
\frac{d}{dx}f(u(x))=f'(u(x))u'(x)
$$

### B.7.5 对数求导不是只用于 $\ln x$

对数求导用于简化复杂乘积、商、幂指数混合形式，尤其是：

$$
x^x,\qquad (g(x))^{h(x)},\qquad
\frac{(x^2+1)^5\sqrt{x-1}}{e^x\sin x}
$$

先取 $\ln$，把乘积变加法、幂次放下来，再求导。

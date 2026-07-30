---
aliases:
  - MIT 18.01SC Unit 5
  - Exploring the Infinite
  - 无穷与无穷小
tags:
  - math/calculus
  - course/mit-ocw
  - calculus/series
  - calculus/improper-integrals
source: https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/pages/unit-5-exploring-the-infinite/
---

# MIT 18.01SC Unit 5: Exploring the Infinite

> [!abstract] 本章主线
> “无穷”不是一个可代入的数，而是极限过程。本章先处理两个函数同时趋于 0 或无穷的比值，再定义无穷区间及奇点附近的积分，随后把“无限相加”精确定义为部分和的极限，最后用幂级数和 Taylor 公式把函数编码成多项式。

## 阅读地图

- Part A：Session 87–93，L’Hôpital 法则与反常积分
- Part B：Session 94–101，无穷级数、幂级数和 Taylor 级数
- 官网本单元没有 Problem Set；每节可用的 Exercise 已放回对应 Session
- Session 102 的期末考试独立见 [[Final_Exam|Final Exam]]

## 使用极限工具前的判断顺序

1. 直接代入，确认是不是未定式；
2. 先做代数化简、标准极限或等价无穷小；
3. 只有满足条件的 \(0/0\) 或 \(\infty/\infty\) 才用 L’Hôpital；
4. \(0\cdot\infty\)、\(\infty-\infty\)、\(0^0\)、\(1^\infty\)、\(\infty^0\) 要先改写；
5. 得到答案后检查符号、增长阶和定义域。

![[98_attachment/MIT18.01SC/unit05-growth-hierarchy.png|749]]

---

## Part A：L’Hospital’s Rule and Improper Integrals

## Session 87：L’Hôpital’s Rule

### [[Indeterminate Forms and L'Hopital's Rule|洛必达法则]]：未定式与使用条件

设 \(f,g\) 在 \(a\) 的穿孔邻域可导，\(g'(x)\ne0\)，并且

$$
\lim_{x\to a}f(x)=\lim_{x\to a}g(x)=0
$$

或二者绝对值都趋于无穷。若

$$
\lim_{x\to a}\frac{f'(x)}{g'(x)}=L
$$

存在（可含 \(\pm\infty\)），则在相应条件下

$$
\boxed{\lim_{x\to a}\frac{f(x)}{g(x)}=L}.
$$

法则处理的是**比值的极限**，不是函数恒等式；一般不能写 \(f/g=f'/g'\)。

### 为什么成立：Cauchy MVT

先看 \(0/0\)，并暂设 \(f(a)=g(a)=0\)。对靠近 \(a\) 的 \(x\)，Cauchy 平均值定理保证在 \(a,x\) 之间存在 \(c\)：

$$
\frac{f(x)-f(a)}{g(x)-g(a)}
=\frac{f'(c)}{g'(c)}.
$$

左边就是 \(f(x)/g(x)\)。当 \(x\to a\)，夹在二者之间的 \(c\to a\)，所以若导数比趋于 \(L\)，原比值也趋于 \(L\)。

> [!note] 严格性说明
> \(\infty/\infty\)、单侧极限和无穷远极限需要相应版本的 Cauchy MVT 与附加控制；法则的条件不可只凭结果看似正确而省略。

### 例

$$
\lim_{x\to0}\frac{e^x-1}{x}
=\lim_{x\to0}\frac{e^x}{1}=1.
$$

这个极限也正是 \(e^x\) 在 0 的导数；识别导数定义比 L’Hôpital 更能解释其意义。

### 87a–87c：课件例题与证明流程

课件先计算一个本可因式分解的极限：

$$
\lim_{x\to1}\frac{x^{10}-1}{x^2-1}.
$$

直接代入是 $0/0$，分子、分母分别求导后

$$
\lim_{x\to1}\frac{10x^9}{2x}=5.
$$

代数法也给同一答案：约去 $x-1$ 后再代入。两种方法相符说明 L’Hôpital 没有在制造新函数，而是在提取分子与分母的**一阶主导变化**。

证明时应把逻辑顺序写清：

1. 目标是原比值 $f(x)/g(x)$ 的极限；
2. 用 Cauchy MVT 构造中间点 $c_x$，使原比值等于某个导数比；
3. 因 $c_x$ 位于 $a$ 与 $x$ 之间，$x\to a$ 强迫 $c_x\to a$；
4. 只有已知 $f'(t)/g'(t)\to L$，才能把 $t=c_x$ 代入这一极限结论。

这也解释了为何“导数比的极限存在”是定理假设，而不是计算后的可选检查。

### 本地材料与练习

- [[Ses87a_Lecture_Notes.pdf|87a Introduction]]
- [[Ses87b_Lecture_Notes.pdf|87b Elementary Example]]
- [[Ses87c_Lecture_Notes.pdf|87c Why the Rule Works]]
- [[Ses87d_Problems.pdf|87d L’Hôpital Version 1 — misfiled lecture]]
- [[Exercise087_Problems.pdf|Exercise 87]]
- [[Exercise087_Solutions.pdf|Exercise 87 Solutions]]

> [!example]- Exercise 087 完整题解：$\displaystyle\lim_{x\to0}\frac{\sin x}{1-\cos x}$
> **(a) 线性近似。**$\sin x\approx x$、$\cos x\approx1$，分母近似为 $0$，因此线性信息不足以决定结果；这正是需要更高阶近似的信号。
>
> **(b) 二次近似。**$\sin x=x+O(x^3)$，$1-\cos x=x^2/2+O(x^4)$，所以
> $$
> \frac{\sin x}{1-\cos x}
> =\frac{x+O(x^3)}{x^2/2+O(x^4)}
> \sim\frac2x.
> $$
> 因而
> $$
> \lim_{x\to0^+}\frac{\sin x}{1-\cos x}=+\infty,
> \qquad
> \lim_{x\to0^-}\frac{\sin x}{1-\cos x}=-\infty.
> $$
> **结论：两侧极限不存在。**若只写“无穷大”会丢失左右符号。也可用恒等式化为 $\cot(x/2)$ 检查。

> [!question]- 三道自检题与答案
> 1. $\lim_{x\to0}\sin x/x$ 可用 L’Hôpital，结果为 $1$；但若正弦导数本身是靠该极限证明的，就会循环论证。
> 2. $\lim_{x\to2}(x^2-4)/(x-2)$ 可用，但先约分更直接，结果为 $4$。
> 3. 不能写 $f/g=f'/g'$；例如 $x^2/x=x$，而导数比为 $2x$，只有符合条件时二者的极限才由定理联系。

## Session 88：Examples of L’Hôpital’s Rule

### 重复使用

$$
\lim_{x\to0}\frac{e^x-1-x}{x^2}
$$

第一次仍为 \(0/0\)：

$$
=\lim_{x\to0}\frac{e^x-1}{2x}.
$$

第二次仍为 \(0/0\)：

$$
=\lim_{x\to0}\frac{e^x}{2}=\frac12.
$$

每次使用前都重新确认未定式。

### 与 Taylor 近似比较

$$
e^x=1+x+\frac{x^2}{2}+O(x^3),
$$

所以

$$
\frac{e^x-1-x}{x^2}
=\frac12+O(x).
$$

L’Hôpital 给答案，Taylor 还说明误差阶数。

### 扩展到无穷远

$$
\lim_{x\to\infty}\frac{\ln x}{x}
=\lim_{x\to\infty}\frac{1/x}{1}=0.
$$

这说明对数增长慢于线性函数。

### 88a–88d：课件中的三类用法

**一次使用：**

$$
\lim_{x\to0}\frac{\sin5x}{\sin2x}
=\lim_{x\to0}\frac{5\cos5x}{2\cos2x}
=\frac52.
$$

**重复使用：**

$$
\begin{aligned}
\lim_{x\to0}\frac{\cos x-1}{x^2}
&=\lim_{x\to0}\frac{-\sin x}{2x}\\
&=\lim_{x\to0}\frac{-\cos x}{2}=-\frac12.
\end{aligned}
$$

第二次之前，新的比值仍是 $0/0$；这一步确认不可省略。二次近似 $\cos x=1-x^2/2+O(x^4)$ 不仅给出 $-1/2$，还说明被忽略项是 $O(x^2)$。

**无穷远版本：**把 $x\to\infty$ 看成在越来越长区间上比较增长。若分子分母都趋于无穷，且导数比有极限，才可使用相应版本。

### 本地材料与练习

- [[Ses88a_Lecture_Notes.pdf|88a Example]]
- [[Ses88b_Lecture_Notes.pdf|88b Repeating the Rule]]
- [[Ses88c_Lecture_Notes.pdf|88c Comparison with Approximation]]
- [[Ses88d_Lecture_Notes.pdf|88d Extensions]]
- [[Exercise088_Problems.pdf|Exercise 88]]
- [[Exercise088_Solutions.pdf|Exercise 88 Solutions]]

> [!warning] 易错点
> 若第一次求导后的极限已不是未定式，必须停止；不能为了“化简”继续求导。

> [!example]- Exercise 088 完整题解：比较 $e^x$ 与 $x$
> **(a)** $e^x/x$ 为 $\infty/\infty$：
> $$
> \lim_{x\to\infty}\frac{e^x}{x}
> =\lim_{x\to\infty}\frac{e^x}{1}=+\infty.
> $$
> 扩展版允许导数比趋于 $+\infty$，所以原比值也趋于 $+\infty$。
>
> **(b)**
> $$
> \lim_{x\to\infty}\frac{x}{e^x}
> =\lim_{x\to\infty}\frac1{e^x}=0.
> $$
> 两题互为倒数，结论一致地表达“指数增长快于线性增长”。

> [!question]- 三道自检题与答案
> 1. $\lim_{x\to0}(e^x-1-x)/x^2=1/2$，需连续使用两次。
> 2. $\lim_{x\to\infty}x^3/e^x=0$，连续三次后成为 $6/e^x$。
> 3. $\lim_{x\to0}(1+x)/x$ 不是未定式；右极限 $+\infty$、左极限 $-\infty$，不能再求导。

## Session 89：Rates of Growth

### 增长层级

对任意固定 \(p>0\)、\(q>0\)：

$$
\ln x \ll x^p \ll e^{qx}
\qquad(x\to\infty).
$$

符号 \(f\ll g\) 表示 \(f/g\to0\)。

证明 \(\ln x\ll x^p\)：

$$
\lim_{x\to\infty}\frac{\ln x}{x^p}
=\lim_{x\to\infty}\frac{1/x}{px^{p-1}}
=\lim_{x\to\infty}\frac1{px^p}=0.
$$

证明多项式慢于指数：对整数 \(n\)，反复使用 L’Hôpital：

$$
\lim_{x\to\infty}\frac{x^n}{e^{qx}}
=\lim_{x\to\infty}\frac{n!}{q^ne^{qx}}=0.
$$

### 89a–89c：增长与衰减是一件事的两面

课件依次处理：

$$
\lim_{x\to0^+}x\ln x
=\lim_{x\to0^+}\frac{\ln x}{1/x}
=\lim_{x\to0^+}\frac{1/x}{-1/x^2}
=\lim_{x\to0^+}(-x)=0.
$$

虽然 $\ln x\to-\infty$，但 $x$ 趋零更快，所以乘积从负侧趋零。对 $p>0$，

$$
\lim_{x\to\infty}xe^{-px}
=\lim_{x\to\infty}\frac{x}{e^{px}}
=\lim_{x\to\infty}\frac1{pe^{px}}=0.
$$

最后

$$
\lim_{x\to\infty}\frac{\ln x}{x^{1/3}}
=\lim_{x\to\infty}\frac{1/x}{(1/3)x^{-2/3}}
=\lim_{x\to\infty}\frac3{x^{1/3}}=0.
$$

取倒数就把增长层级翻译为衰减层级；“谁增长更快”与“谁的倒数衰减更快”是同一判断。

### 本地材料

- [[Ses89a_Lecture_Notes.pdf|89a Growth of \(\ln x\)]]
- [[Ses89b_Lecture_Notes.pdf|89b Growth of \(e^{px}\)]]
- [[Ses89c_Lecture_Notes.pdf|89c Comparing \(\ln x\) and \(x^{1/3}\)]]

> [!question]- 三问自检
> 1. \(x^{100}/e^x\)？2. \(\ln x/\sqrt x\)？3. \(e^{-x}x^5\)？
>
> 答：都趋于 0；第三个等于 \(x^5/e^x\)。

## Session 90：Advanced Indeterminate Forms

### 幂型未定式

若

$$
y=f(x)^{g(x)}>0,
$$

先取对数：

$$
\ln y=g(x)\ln f(x).
$$

求出右侧极限 \(A\) 后，再由指数连续性得到 \(y\to e^A\)。

### 例：\(0^0\)

$$
\lim_{x\to0^+}x^x.
$$

令 \(y=x^x\)：

$$
\ln y=x\ln x=\frac{\ln x}{1/x}.
$$

这是 \((-\infty)/\infty\)：

$$
\lim_{x\to0^+}\frac{1/x}{-1/x^2}
=\lim_{x\to0^+}(-x)=0.
$$

所以

$$
\boxed{\lim_{x\to0^+}x^x=e^0=1}.
$$

### 其他改写

$$
0\cdot\infty:
\quad fg=\frac f{1/g},
$$

$$
\infty-\infty:
\quad\text{通分、有理化或提取主项}.
$$

### 90b–90c：最重要的反例——Look before you L’Hôp

考虑

$$
\lim_{x\to0}\frac{\sin x}{x^2}.
$$

第一次使用后得到 $\cos x/(2x)$。此时分子趋于 $1$、分母趋于 $0$，已经**不是** $0/0$；若错误地再次求导，会得到 $0$，与真实行为矛盾。由 $\sin x\sim x$：

$$
\lim_{x\to0^+}\frac{\sin x}{x^2}=+\infty,
\qquad
\lim_{x\to0^-}\frac{\sin x}{x^2}=-\infty.
$$

课件的口令是 “Look before you L’Hôp”：每求导一次，都回到直接代入重新分类。对于

$$
\lim_{x\to\infty}\frac{x^5-2x^4+1}{x^4+2},
$$

除以 $x^4$ 或比较最高次项立刻得 $+\infty$，比机械求导四次更清楚。

### 本地材料与练习

- [[Ses90a_Lecture_Notes.pdf|90a \(0^0\)]]
- [[Ses90b_Lecture_Notes.pdf|90b Example]]
- [[Ses90c_Lecture_Notes.pdf|90c Continued]]
- [[Exercise090_Problems.pdf|Exercise 90]]
- [[Exercise090_Solutions.pdf|Exercise 90 Solutions]]

> [!example]- Exercise 090 完整题解：$\displaystyle\lim_{x\to\infty}x^{1/x}$
> 这是 $\infty^0$。令 $y=x^{1/x}>0$，则
> $$
> \ln y=\frac{\ln x}{x}.
> $$
> 右侧为 $\infty/\infty$，所以
> $$
> \lim_{x\to\infty}\frac{\ln x}{x}
> =\lim_{x\to\infty}\frac{1/x}{1}=0.
> $$
> 指数函数连续，故
> $$
> \boxed{\lim_{x\to\infty}x^{1/x}=e^0=1.}
> $$
> 这也说明数列 $\sqrt[n]{n}\to1$。

> [!question]- 三道自检题与答案
> 1. $\lim_{x\to0^+}(1+x)^{1/x}=e$：取对数后研究 $\ln(1+x)/x\to1$。
> 2. $\lim_{x\to\infty}x^2e^{-x}=0$：改写为 $x^2/e^x$。
> 3. $\lim_{x\to0}(1/x-1/\sin x)=0$：先通分为 $(\sin x-x)/(x\sin x)$，再合法使用 L’Hôpital 或三阶近似。

## Session 91：Improper Integrals

### 无穷区间

[[Improper Integral|反常积分]]定义为

$$
\int_a^\infty f(x)dx
=\lim_{b\to\infty}\int_a^bf(x)dx.
$$

极限有限称收敛，否则发散。符号 \(\infty\) 只是极限方向，不能直接代入原函数。

### \(p\)-积分

$$
\int_1^\infty\frac{dx}{x^p}.
$$

若 \(p\ne1\)：

$$
\int_1^b x^{-p}dx
=\frac{b^{1-p}-1}{1-p}.
$$

- \(p>1\)：\(b^{1-p}\to0\)，积分收敛为 \(1/(p-1)\)；
- \(p<1\)：幂次非负，发散；
- \(p=1\)：\(\ln b\to\infty\)，发散。

因此

$$
\boxed{\int_1^\infty x^{-p}dx\text{ 收敛}\iff p>1}.
$$

### 91a–91e：课件例题与“尾部”直觉

最基本的指数尾部（$k>0$）是

$$
\begin{aligned}
\int_0^\infty e^{-kx}dx
&=\lim_{N\to\infty}\left[-\frac1ke^{-kx}\right]_0^N\\
&=\frac1k.
\end{aligned}
$$

$k>0$ 是必要条件；$k=0$ 时被积函数恒为 $1$，$k<0$ 时反而指数增长。物理上 $Ae^{-kt}$ 可描述衰变率，总累计量为 $A/k$。

课件还指出高斯积分

$$
\int_{-\infty}^{\infty}e^{-x^2}dx=\sqrt\pi.
$$

本单元暂不推导精确值，但 Session 92 会用比较证明它至少收敛。与之相对，

$$
\int_1^\infty\frac{dx}{x}
=\lim_{N\to\infty}\ln N=\infty
$$

说明“函数趋于零”仍不足以保证尾面积有限；关键是趋零速度。

![[98_attachment/MIT18.01SC/unit05-improper-integrals.png]]

### 本地材料与练习

- [[Ses91a_Lecture_Notes.pdf|91a Introduction]]
- [[Ses91b_Lecture_Notes.pdf|91b Example 1]]
- [[Ses91c_Lecture_Notes.pdf|91c Example 2]]
- [[Ses91d_Lecture_Notes.pdf|91d Example 3]]
- [[Ses91e_Lecture_Notes.pdf|91e Example 4]]
- [[Exercise091_Problems.pdf|Exercise 91]]
- [[Exercise091_Solutions.pdf|Exercise 91 Solutions]]

> [!question]- 三问自检
> 1. \(\int_1^\infty1/x^2\)？2. \(\int_1^\infty1/x\)？3. 被积函数趋零是否足够？
>
> 答：收敛为 1；发散；不够，调和型尾部反例。

> [!example]- Exercise 091 完整题解：$\displaystyle\int_1^\infty\frac{dx}{(5x+2)^2}$
> 必须先写有限上限：
> $$
> I=\lim_{N\to\infty}\int_1^N(5x+2)^{-2}dx.
> $$
> 令 $u=5x+2$，或直接求原函数 $-1/[5(5x+2)]$：
> $$
> \begin{aligned}
> I
> &=\lim_{N\to\infty}\left[-\frac1{5(5x+2)}\right]_1^N\\
> &=\lim_{N\to\infty}\left(\frac1{35}-\frac1{5(5N+2)}\right)\\
> &=\boxed{\frac1{35}}.
> \end{aligned}
> $$
> 结果为正且小于 $\int_1^\infty dx/(25x^2)=1/25$，数量级合理。

## Session 92：Integral Comparison

设 \(0\le f(x)\le g(x)\) 对充分大的 \(x\) 成立：

- 若 \(\int g\) 收敛，则 \(\int f\) 收敛；
- 若 \(\int f\) 发散，则 \(\int g\) 发散。

方向来自面积大小，不能倒用。

### 极限比较

若 \(f,g>0\) 且

$$
\lim_{x\to\infty}\frac{f(x)}{g(x)}=c,
\qquad0<c<\infty,
$$

则二者同敛散。原因是当 \(x\) 足够大时，\(f\) 被两个正常数倍的 \(g\) 夹住。

例：

$$
\frac1{x^2+3x}\sim\frac1{x^2},
$$

故其从 1 到无穷的积分收敛。

### 92a–92c：三个课件比较

**与 $1/x$ 比较：**

$$
\frac1{\sqrt{x^2+10}}\sim\frac1x,
$$

所以 $\int_0^\infty dx/\sqrt{x^2+10}$ 的尾部发散。比较时从 $1$ 或任意正数开始，避免让 $1/x$ 在 $0$ 的奇点混入“无穷远尾部”问题。

**与 $p$-积分比较：**

$$
\frac1{\sqrt{x^3+3}}\sim\frac1{x^{3/2}},
$$

故 $\int_{10}^\infty dx/\sqrt{x^3+3}$ 收敛，即使原函数难以写出。

**普通比较而非极限比较：**当 $x\ge1$，$x^2\ge x$，所以

$$
0<e^{-x^2}\le e^{-x}.
$$

有限区间 $[0,1]$ 上连续函数积分有限，而 $\int_1^\infty e^{-x}dx$ 收敛，因此利用偶性可知整个高斯积分收敛。

### 本地材料与练习

- [[Ses92a_Lecture_Notes.pdf|92a Limit Comparison]]
- [[Ses92b_Lecture_Notes.pdf|92b Example]]
- [[Ses92c_Lecture_Notes.pdf|92c Example]]
- [[Exercise092_Problems.pdf|Exercise 92]]
- [[Exercise092_Solutions.pdf|Exercise 92 Solutions]]

> [!example]- Exercise 092 完整题解：用极限比较确认 Exercise 091 收敛
> 取
> $$
> f(x)=\frac1{(5x+2)^2},
> \qquad
> g(x)=\frac1{25x^2}.
> $$
> 则
> $$
> \lim_{x\to\infty}\frac{f(x)}{g(x)}
> =\lim_{x\to\infty}\frac{25x^2}{(5x+2)^2}=1.
> $$
> 因 $\int_1^\infty g(x)dx=1/25$ 收敛，极限比较定理给
> $$
> \boxed{\int_1^\infty\frac{dx}{(5x+2)^2}\text{ 收敛}.}
> $$
> 比较判别只给敛散性，不给 Exercise 091 的精确值 $1/35$。

> [!question]- 三道自检题与答案
> 1. $\int_1^\infty dx/\sqrt{x^4+1}$ 与 $1/x^2$ 极限比较，收敛。
> 2. $\int_1^\infty x/(x^2+1)dx$ 与 $1/x$ 极限比较，发散。
> 3. 若 $0\le f\le g$ 且 $\int f$ 收敛，不能推出 $\int g$ 收敛；较大的尾部仍可能发散。

## Session 93：Singularities

### 有限端点奇点

若 \(f\) 在 \(a\) 无界，定义

$$
\int_a^bf(x)dx
=\lim_{\varepsilon\to0^+}\int_{a+\varepsilon}^bf(x)dx.
$$

典型：

$$
\int_0^1x^{-p}dx
\text{ 收敛}\iff p<1.
$$

注意与无穷远的 \(p>1\) 条件正好相反。

### 区间内部奇点

若 \(c\in(a,b)\) 是奇点，必须拆成两个独立反常积分：

$$
\int_a^bf
=\int_a^cf+\int_c^bf.
$$

只有两边都收敛，总积分才收敛。不能让左右无穷“相消”；那是 Cauchy 主值，不是通常反常积分。

### 93a–93d：先找所有坏点，再逐个处理

课件用一个“荒谬答案”警示跨奇点套原函数：

$$
\int_{-1}^{1}\frac{dx}{x^2}
\overset{\text{错误}}=\left[-\frac1x\right]_{-1}^{1}=-2.
$$

被积函数处处非负，面积不可能为负。正确做法是在 $0$ 拆开；任一侧都像 $\int_0^1x^{-2}dx$ 一样发散，所以总积分发散。

端点模型完整结论为

$$
\int_0^1x^{-p}dx
=\begin{cases}
\dfrac1{1-p},&p<1,\\
\text{发散},&p\ge1.
\end{cases}
$$

与无穷远模型恰好相反，因为 $x^{-p}$ 在 $0$ 附近随 $p$ 增大而更严重，在无穷远则随 $p$ 增大而衰减更快。

对

$$
\int_0^\infty\frac{dx}{(x-3)^2},
$$

既有无穷远端，又有内部奇点 $x=3$。必须在 $3$ 拆分；奇点附近与 $1/u^2$ 同型而发散，因此无论远端是否收敛，总积分都发散。

### 本地材料

- [[Ses93a_Lecture_Notes.pdf|93a Singularities]]
- [[Ses93b_Lecture_Notes.pdf|93b Second Kind]]
- [[Ses93c_Lecture_Notes.pdf|93c Overview]]
- [[Ses93d_Lecture_Notes.pdf|93d Example]]

> [!warning] 易错点
> 发现奇点后先拆区间，再分别写极限。跨过奇点直接套原函数会掩盖发散。

> [!question]- 三道自检题与答案
> 1. $\int_0^1x^{-3/4}dx=4$，因为 $3/4<1$。
> 2. $\int_{-1}^{1}dx/x$ 作为通常反常积分发散；对称 Cauchy 主值为 $0$，两者不是同一概念。
> 3. $\int_0^2dx/\sqrt{|x-1|}$ 要在 $1$ 拆开，两侧指数 $p=1/2<1$，所以都收敛。

---

## Part B：Taylor Series

## Session 94：Infinite Series

### 定义

[[Series|无穷级数]]

$$
\sum_{n=1}^{\infty}a_n
$$

不是“把无穷多个数一次加完”，而是部分和序列

$$
S_N=\sum_{n=1}^Na_n
$$

的极限。若 \(S_N\to S\)，称级数收敛到 \(S\)。

### 必要条件

若级数收敛，则

$$
a_n=S_n-S_{n-1}\to S-S=0.
$$

所以 \(a_n\not\to0\) 必发散；但 \(a_n\to0\) 不保证收敛。

### 几何级数

$$
S_N=1+r+\cdots+r^N
=\frac{1-r^{N+1}}{1-r}.
$$

因此

$$
\boxed{\sum_{n=0}^{\infty}r^n=\frac1{1-r},\quad |r|<1}.
$$

有限和公式来自消项：$(1-r)S_N=1-r^{N+1}$。只有 $|r|<1$ 时余项 $r^{N+1}\to0$，才能得到无穷和。$r=1$ 时部分和无界，$r=-1$ 时在 $0,1$ 间振荡，$|r|>1$ 时单项不趋零；因此不能越过适用条件套用 $1/(1-r)$。

### 本地材料与练习

- [[Ses94a_Lecture_Notes.pdf|94a Introduction]]
- [[Ses94b_Lecture_Notes.pdf|94b Divergent Series]]
- [[Ses94c_Lecture_Notes.pdf|94c Notation]]
- [[Ses94d_Lecture_Notes.pdf|94d Examples]]
- [[Exercise094_Problems.pdf|Exercise 94]]
- [[Exercise094_Solutions.pdf|Exercise 94 Solutions]]

![[98_attachment/MIT18.01SC/unit05-series-partial-sums.png]]

> [!example]- Exercise 094 完整题解：归纳证明 $1+1/2+1/4+\cdots=2$
> 令 $S_N=\sum_{n=0}^N2^{-n}$。基例 $S_0=1=(2^1-1)/2^0$。若
> $$
> S_{N-1}=\frac{2^N-1}{2^{N-1}},
> $$
> 则
> $$
> S_N=S_{N-1}+2^{-N}
> =\frac{2(2^N-1)+1}{2^N}
> =\frac{2^{N+1}-1}{2^N}=2-2^{-N}.
> $$
> 归纳法证明了有限和公式；再取极限才得到
> $$
> \boxed{\sum_{n=0}^{\infty}2^{-n}=\lim_{N\to\infty}S_N=2.}
> $$

> [!question]- 三道自检题与答案
> 1. $\sum_{n=0}^\infty(1/3)^n=3/2$。
> 2. $\sum_{n=0}^\infty(-1/2)^n=2/3$，部分和振荡但振幅趋零。
> 3. $a_n\to0$ 只是必要条件；$\sum1/n$ 仍发散。

## Session 95：[[Series Convergence Tests|级数收敛判别]]

对非负项级数：

- \(0\le a_n\le b_n\)，若 \(\sum b_n\) 收敛，则 \(\sum a_n\) 收敛；
- \(0\le b_n\le a_n\)，若 \(\sum b_n\) 发散，则 \(\sum a_n\) 发散。

极限比较：若

$$
\lim_{n\to\infty}\frac{a_n}{b_n}=c,\qquad0<c<\infty,
$$

则二者同敛散。

### 积分判别

若 \(f\) 正、连续、递减且 \(a_n=f(n)\)，则

$$
\sum_{n=N}^{\infty}a_n
$$

与

$$
\int_N^\infty f(x)dx
$$

同敛散。矩形图说明级数与面积互相夹住。

因此

$$
\boxed{\sum_{n=1}^{\infty}\frac1{n^p}\text{ 收敛}\iff p>1}.
$$

### Riemann 和证明与比值判别

对正递减 $f$，单位宽矩形给出积分与级数的上下夹逼。特别地，对调和部分和 $H_N$：

$$
\ln N<H_N<\ln N+1,
$$

所以 $H_N\to\infty$，且增长量级为 $\ln N$。

比值判别对一般符号级数使用绝对值：

$$
L=\lim_{n\to\infty}\left|\frac{a_{n+1}}{a_n}\right|.
$$

$L<1$ 时绝对收敛，$L>1$ 时发散，$L=1$ 无结论。前者因为尾部可被公比 $q<1$ 的几何级数控制；后者因为单项不能趋零。

### 本地材料与练习

- [[Ses95a_Lecture_Notes.pdf|95a Harmonic Series]]
- [[Ses95b_Lecture_Notes.pdf|95b Comparison Tests]]
- [[Ses95c_Lecture_Notes.pdf|95c Examples]]
- [[Exercise095_Problems.pdf|Exercise 95]]
- [[Exercise095_Solutions.pdf|Exercise 95 Solutions]]

> [!example]- Exercise 095 完整题解：比值判别
> **(a)** 原题的 $\sum1/\sqrt[3]n$ 应从 $n=1$ 开始。比值
> $$
> \frac{a_{n+1}}{a_n}=\sqrt[3]{\frac n{n+1}}\to1,
> $$
> 故比值判别无结论；另由 $p=1/3\le1$ 知实际发散。
>
> **(b)** $a_n=(n!)^2/(2n)!$：
> $$
> \frac{a_{n+1}}{a_n}
> =\frac{(n+1)^2}{(2n+1)(2n+2)}
> =\frac{n+1}{2(2n+1)}\to\frac14<1,
> $$
> 故收敛。
>
> **(c)** 对 $a_n=(-3)^n/n!$ 取绝对值：
> $$
> \left|\frac{a_{n+1}}{a_n}\right|=\frac3{n+1}\to0,
> $$
> 故绝对收敛。

> [!question]- 三道自检题与答案
> 1. $\sum n/2^n$ 的比值极限为 $1/2$，收敛。
> 2. $\sum1/n$ 的比值极限为 $1$，无结论而实际发散。
> 3. 比值判别尤其适合阶乘和指数幂；有理式尾部常用比较判别。

## Session 96：Stacking Blocks

把长度相同的砖逐层向外伸。第 \(n\) 块相对下一块最多伸出

$$
\frac1{2n}
$$

而不使上方 \(n\) 块的重心越过支点。总伸出量为

$$
\frac12\sum_{n=1}^{N}\frac1n.
$$

调和级数发散，所以理论上可以伸出任意远，只是增长极慢。

若砖长为 $1$，上方 $n$ 块总质量为 $n$；使它们的共同重心恰落在下一支点上，新增最大偏移为 $1/(2n)$。所以每一项来自力矩平衡，而不是经验猜测。

### [[Harmonic Series Divergence|调和级数发散证明]]：分组法

$$
1+\frac12+
\left(\frac13+\frac14\right)+
\left(\frac15+\cdots+\frac18\right)+\cdots
$$

第 \(k\) 组有 \(2^{k-1}\) 项，每项至少 \(1/2^k\)，所以每组和至少 \(1/2\)。组数无限，部分和无界。

### 本地材料

- [[Ses96a_Lecture_Notes.pdf|96a Preview]]
- [[Ses96b_Lecture_Notes.pdf|96b Stacking Blocks]]

> [!question]- 三问自检
> 1. 项趋零为什么仍可发散？2. 每组下界为何 \(1/2\)？3. 发散是否意味着有限 \(N\) 时无限伸出？
>
> 答：衰减太慢；项数乘最小项；不意味，任意有限块数伸出仍有限。

## Session 97：Power Series

[[Power Series and Radius of Convergence|幂级数]]：

$$
\sum_{n=0}^{\infty}c_n(x-a)^n.
$$

固定 \(x\) 后它是数项级数。通常存在收敛半径 \(R\)：

- \(|x-a|<R\)：绝对收敛；
- \(|x-a|>R\)：发散；
- 端点必须另查。

收敛区间以中心 $a$ 对称，是因为绝对收敛由 $|x-a|$ 控制；端点不在这段论证内，可能两端都收敛、都发散或只收敛一端。

### 比值法求半径

若极限存在：

$$
\left|
\frac{c_{n+1}(x-a)^{n+1}}{c_n(x-a)^n}
\right|
\to L|x-a|.
$$

要求小于 1，故 \(R=1/L\)。

例：

$$
\sum_{n=0}^{\infty}\frac{x^n}{n!}
$$

比值为 \(|x|/(n+1)\to0\)，对所有 \(x\) 收敛，\(R=\infty\)。

### 本地材料与练习

- [[Ses97a_Lecture_Notes.pdf|97a Power Series]]
- [[Ses97b_Lecture_Notes.pdf|97b General Power Series]]
- [[Exercise097_Problems.pdf|Exercise 97]]
- [[Exercise097_Solutions.pdf|Exercise 97 Solutions]]

> [!example]- Exercise 097 完整题解：$\sum_{n=1}^{\infty}x^n/n$
> $$
> \left|\frac{x^{n+1}/(n+1)}{x^n/n}\right|
> =|x|\frac n{n+1}\to|x|.
> $$
> 因而 $|x|<1$ 收敛、$|x|>1$ 发散，
> $$
> \boxed{R=1.}
> $$
> 若进一步求区间：$x=1$ 为调和级数，发散；$x=-1$ 为交错调和级数，条件收敛，所以区间是 $[-1,1)$。

> [!question]- 三道自检题与答案
> 1. $\sum x^n/n!$ 有 $R=\infty$。
> 2. $\sum n!x^n$ 有 $R=0$。
> 3. 求出半径后仍要逐个检查 $a\pm R$。

## Session 98：Taylor’s Series

### 系数从哪里来

若

$$
f(x)=\sum_{n=0}^{\infty}c_n(x-a)^n
$$

可逐项求导，代入 \(x=a\)：

$$
f(a)=c_0,\quad
f'(a)=c_1,\quad
f''(a)=2!c_2,
$$

一般地

$$
c_n=\frac{f^{(n)}(a)}{n!}.
$$

因此 [[Taylor Expansion|Taylor series]] 是

$$
\boxed{
\sum_{n=0}^{\infty}
\frac{f^{(n)}(a)}{n!}(x-a)^n
}.
$$

### Taylor 多项式与余项

$$
f(x)=P_n(x)+R_n(x),
$$

$$
P_n(x)=\sum_{k=0}^{n}\frac{f^{(k)}(a)}{k!}(x-a)^k.
$$

Lagrange 余项：

$$
R_n(x)=\frac{f^{(n+1)}(\xi)}{(n+1)!}(x-a)^{n+1}
$$

其中 \(\xi\) 位于 \(a,x\) 之间。若该区间上

$$
|f^{(n+1)}|\le M,
$$

则

$$
|R_n(x)|\le\frac{M|x-a|^{n+1}}{(n+1)!}.
$$

![[98_attachment/MIT18.01SC/unit05-taylor-error.png]]

### 本地材料与练习

- [[Ses98a_Lecture_Notes.pdf|98a Introduction]]
- [[Ses98b_Lecture_Notes.pdf|98b Taylor Formula]]
- [[Ses98c_Lecture_Notes.pdf|98c Taylor Formula Continued]]
- [[Exercise098_Problems.pdf|Exercise 98]]
- [[Exercise098_Solutions.pdf|Exercise 98 Solutions]]

> [!example]- Exercise 098 完整题解：证明 $e^x$ 处处等于 Taylor 级数
> 固定任意 $x$，取 $d>|x|$。在 $[-d,d]$ 上，所有阶导数仍为 $e^t$，故 $|f^{(N+1)}(t)|\le e^d$。Taylor 不等式给
> $$
> |R_N(x)|\le e^d\frac{|x|^{N+1}}{(N+1)!}
> \le e^d\frac{d^{N+1}}{(N+1)!}.
> $$
> 右端数列相邻比为 $d/(N+2)\to0$，故趋零，夹逼得 $R_N(x)\to0$。由于 $x$ 任意，
> $$
> \boxed{e^x=\sum_{n=0}^{\infty}\frac{x^n}{n!}\quad(x\in\mathbb R).}
> $$
> 正弦、余弦各阶导数绝对值不超过 $1$，可用同一方法证明。

> [!question]- 三道自检题与答案
> 1. $e^x$ 的三次 Maclaurin 多项式是 $1+x+x^2/2+x^3/6$。
> 2. 系数由导数唯一确定，但函数等于级数还需 $R_N\to0$。
> 3. 余项估计所用导数上界必须在连接展开中心与目标点的区间上成立。

## Session 99：Taylor’s Series, Continued

### 常用 Maclaurin 级数

$$
e^x=\sum_{n=0}^{\infty}\frac{x^n}{n!},
$$

$$
\sin x=\sum_{n=0}^{\infty}(-1)^n\frac{x^{2n+1}}{(2n+1)!},
$$

$$
\cos x=\sum_{n=0}^{\infty}(-1)^n\frac{x^{2n}}{(2n)!}.
$$

从几何级数

$$
\frac1{1+x}=1-x+x^2-x^3+\cdots,\qquad |x|<1,
$$

积分得到

$$
\ln(1+x)
=x-\frac{x^2}{2}+\frac{x^3}{3}-\cdots.
$$

### 级数不一定等于原函数

Taylor 系数由所有导数决定，但还需证明余项 \(R_n(x)\to0\)。存在光滑函数所有导数在一点都为零，却不在附近恒为零；所以“可无限求导”本身不够。

### 本地材料

- [[Ses99a_Lecture_Notes.pdf|99a Review]]
- [[Ses99b_Lecture_Notes.pdf|99b Series of \(1/(1+x)\)]]
- [[Ses99c_Lecture_Notes.pdf|99c Series of \(\sin x\)]]

> [!question]- 三道自检题与答案
> 1. \(\sin x\) 的 \(x^5\) 项是什么？  
> 2. \(\ln(1+x)\) 的级数为何只先保证 \(|x|<1\)？  
> 3. “所有阶导数存在”是否自动保证 Taylor 级数等于函数？
>
> 答：\(x^5/5!\)；它从半径为 1 的几何级数逐项积分而来；不保证，还需证明余项趋零。

## Session 100：Operations on Power Series

在共同收敛区间内部，可像多项式一样逐项运算。

### 乘法

Cauchy 乘积：

$$
\left(\sum_{n=0}^{\infty}a_nx^n\right)
\left(\sum_{n=0}^{\infty}b_nx^n\right)
=\sum_{n=0}^{\infty}
\left(\sum_{k=0}^{n}a_kb_{n-k}\right)x^n.
$$

### 求导与积分

$$
\frac d{dx}\sum_{n=0}^{\infty}c_nx^n
=\sum_{n=1}^{\infty}nc_nx^{n-1},
$$

$$
\int\sum_{n=0}^{\infty}c_nx^n dx
=C+\sum_{n=0}^{\infty}\frac{c_n}{n+1}x^{n+1}.
$$

收敛半径保持不变，但端点行为可能改变。

### 代入与误差函数

$$
e^{-x^2}
=\sum_{n=0}^{\infty}\frac{(-1)^nx^{2n}}{n!}.
$$

逐项积分：

$$
\operatorname{erf}(x)
=\frac2{\sqrt\pi}
\sum_{n=0}^{\infty}
\frac{(-1)^nx^{2n+1}}{n!(2n+1)}.
$$

### 本地材料与练习

- [[Ses100a_Lecture_Notes.pdf|100a Multiplication]]
- [[Ses100b_Lecture_Notes.pdf|100b Derivative]]
- [[Ses100c_Lecture_Notes.pdf|100c Integral]]
- [[Ses100d_Lecture_Notes.pdf|100d Substitution]]
- [[Ses100e_Lecture_Notes.pdf|100e Error Function]]
- [[MIT18_01SCF10_ex100prb.pdf|Exercise 100 Problems — misfiled]]
- [[Exercise100_Solutions.pdf|Exercise 100 Solutions]]

> [!example]- Exercise 100 完整题解
> **(1) 展开 \(\arctan(5x)\)。**先求导并使用几何级数：
> $$
> \frac d{dx}\arctan(5x)
> =\frac5{1+25x^2}
> =5\sum_{n=0}^{\infty}(-25x^2)^n
> =\sum_{n=0}^{\infty}(-1)^n5^{2n+1}x^{2n}.
> $$
> 逐项积分：
> $$
> \arctan(5x)
> =C+\sum_{n=0}^{\infty}
> (-1)^n\frac{5^{2n+1}}{2n+1}x^{2n+1}.
> $$
> 代 \(x=0\) 得 \(C=0\)。几何级数要求 \(|25x^2|<1\)，所以初始收敛区间为 \(|x|<1/5\)；端点需另查。
>
> **(2) 近似 \(\int_0^1\sin(x^2)dx\)。**
> $$
> \sin(x^2)
> =\sum_{n=0}^{\infty}
> (-1)^n\frac{x^{4n+2}}{(2n+1)!}.
> $$
> 逐项积分：
> $$
> \int_0^1\sin(x^2)dx
> =\sum_{n=0}^{\infty}
> \frac{(-1)^n}{(4n+3)(2n+1)!}.
> $$
> 取两项：
> $$
> \frac13-\frac1{7\cdot3!}
> =\boxed{\frac{13}{42}\approx0.30952}.
> $$
> 下一项为 \(1/(11\cdot5!)=1/1320\)，交错级数误差不超过这一数；真实值约 \(0.31027\)。

> [!warning] 易错点
> 逐项运算只在收敛半径内部自动安全；端点必须重新检查，不能照搬原级数结论。

> [!question]- 三道自检题与答案
> 1. 幂级数求导后收敛半径怎样？  
> 2. 为什么积分后必须加常数？  
> 3. \(\arctan(5x)\) 的线性项是什么？
>
> 答：半径不变但端点可能改变；不同原函数相差常数；线性项为 \(5x\)。

## Session 101：Conclusion

### 微积分的闭环

1. 导数把函数局部线性化；
2. 积分把局部贡献累计起来；
3. FTC 说明二者互逆；
4. Taylor 级数把局部导数信息组织成全阶多项式近似；
5. 极限决定这些无限过程是否真正收敛。

### 何时选哪件工具

- \(0/0\)、\(\infty/\infty\)：先化简，再考虑 L’Hôpital；
- 无限区间或奇点积分：写极限并做比较；
- 无限求和：研究部分和，而非只看单项；
- 函数近似：写 Taylor 多项式并给余项界；
- 没有初等原函数：定积分、数值法或幂级数仍然可用。

### 本地材料与练习

- [[Ses101a_Lecture_Notes.pdf|101a Finale]]
- [[Exercise101_Problems.pdf|Exercise 101]]
- [[Exercise101_Solutions.pdf|Exercise 101 Solutions]]

> [!example]- Exercise 101 完整题解：双曲正弦串联全课程
> **(a) 描图。**由
> $$
> \sinh x=\frac{e^x-e^{-x}}2
> $$
> 得 \(\sinh(-x)=-\sinh x\)，所以是奇函数。又
> $$
> (\sinh x)'=\cosh x=\frac{e^x+e^{-x}}2>0,
> $$
> 因此无临界点且处处递增。二阶导数为 \(\sinh x\)，仅在 \(x=0\) 为零并变号，所以原点是拐点。两端极限分别为 \(\pm\infty\)。
>
> **(b) 定义反函数。**严格递增且值域为 \(\mathbb R\)，故
> \[
> y=\operatorname{arsinh}x
> \iff x=\sinh y
> \]
> 对所有实数 \(x\) 有唯一定义；图像是 \(y=\sinh x\) 关于 \(y=x\) 的反射。
>
> **(c) 求导。**隐式求导：
> $$
> \cosh y\,\frac{dy}{dx}=1.
> $$
> 由 \(\cosh^2y-\sinh^2y=1\)、\(\cosh y>0\) 以及 \(\sinh y=x\)：
> $$
> \boxed{
> \frac d{dx}\operatorname{arsinh}x
> =\frac1{\sqrt{1+x^2}}
> }.
> $$
>
> **(d) 积分。**令 \(x=au\)（取 \(a>0\)）：
> $$
> \int\frac{dx}{\sqrt{a^2+x^2}}
> =\int\frac{du}{\sqrt{1+u^2}}
> =\boxed{\operatorname{arsinh}(x/a)+C}.
> $$
> 等价的对数形式是
> $$
> \ln\!\left|x+\sqrt{x^2+a^2}\right|+C.
> $$

> [!question]- 三道自检题与答案
> 1. \(\sinh x\) 为什么可逆？  
> 2. \(\operatorname{arsinh}x\) 的定义域？  
> 3. 上述积分中 \(a\) 的符号为何要说明？
>
> 答：导数 \(\cosh x>0\)，所以严格递增；全体实数；提出 \(\sqrt{a^2}=|a|\) 时符号会影响化简。

## 全章易错点总表

| 情形 | 错误做法 | 正确检查 |
|---|---|---|
| L’Hôpital | 不是未定式也求导 | 每次使用前重新代入 |
| 幂型极限 | 把 \(0^0\) 直接算成 1 | 取对数后研究乘积 |
| 反常积分 | 把 \(\infty\) 当端点代入 | 先写有限端点极限 |
| 内部奇点 | 左右发散相消 | 两边必须分别收敛 |
| 无穷级数 | 只因 \(a_n\to0\) 就判收敛 | 研究部分和或使用判别法 |
| 比较判别 | 把充分方向倒用 | 画大小关系并确认基准级数 |
| 幂级数 | 忘记端点 | 半径内部、外部、端点分开 |
| Taylor | 只写多项式不写误差 | 给适用区间与余项界 |

## 本章总结

1. L’Hôpital 是由 Cauchy MVT 支撑的条件性定理；
2. 反常积分和无穷级数都通过有限对象的极限定义；
3. 比较判别把未知对象与 \(p\)-积分、\(p\)-级数等基准比较；
4. 幂级数在收敛区间内像无限多项式；
5. Taylor 系数由导数唯一决定，但等于原函数还需余项趋零；
6. 任何“无穷运算”都必须明确极限对象、收敛条件与误差。

> [!tip] 一遍读懂后的最低验收
> 能说明 L’Hôpital 的证明骨架，判断 \(p\)-积分与 \(p\)-级数，解释调和级数为何发散，求幂级数半径和端点，并用 Taylor 余项给出可验证的近似误差。

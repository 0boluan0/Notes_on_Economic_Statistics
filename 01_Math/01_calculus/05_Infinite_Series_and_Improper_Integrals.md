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
> <!-- bilingual-en:start -->
> Infinity is not a number that can be substituted into a formula; it describes a limiting process. This chapter first studies ratios in which both functions approach zero or both grow without bound, then defines integrals over infinite intervals and near singularities. It next defines an infinite sum as the limit of its partial sums, and finally uses power series and Taylor's formula to represent functions by polynomials.
> <!-- bilingual-en:end -->

## 阅读地图

- Part A：Session 87–93，L’Hôpital 法则与反常积分
- Part B：Session 94–101，无穷级数、幂级数和 Taylor 级数
- 官网本单元没有 Problem Set；每节可用的 Exercise 已放回对应 Session
- Session 102 的期末考试独立见 [[Final_Exam|Final Exam]]

## 使用极限工具前的判断顺序
<!-- bilingual-en:start -->
*What to check before choosing a limit technique*
<!-- bilingual-en:end -->

1. 直接代入，确认是不是未定式；
2. 先做代数化简、标准极限或等价无穷小；
3. 只有满足条件的 \(0/0\) 或 \(\infty/\infty\) 才用 L’Hôpital；
4. \(0\cdot\infty\)、\(\infty-\infty\)、\(0^0\)、\(1^\infty\)、\(\infty^0\) 要先改写；
5. 得到答案后检查符号、增长阶和定义域。
<!-- bilingual-en:start -->

&nbsp;
**1.** Substitute directly and determine whether the result is an indeterminate form;<br>
**2.** First try algebraic simplification, a standard limit, or an equivalent infinitesimal;<br>
**3.** Use L’Hôpital’s rule only for $0/0$ or $\infty/\infty$ forms that satisfy its hypotheses;<br>
**4.** Rewrite $0\cdot\infty$, $\infty-\infty$, $0^0$, $1^\infty$, and $\infty^0$ into a suitable form before proceeding;<br>
**5.** Check the sign, order of growth, and domain of the final result.<br>
<!-- bilingual-en:end -->

![[98_attachment/MIT18.01SC/unit05-growth-hierarchy.png|749]]

---

## Part A：L’Hospital’s Rule and Improper Integrals

## Session 87：L’Hôpital’s Rule

### [[极限与连续#极限存在的检查顺序|洛必达法则]]：未定式与使用条件
<!-- bilingual-en:start -->
*[[极限与连续#极限存在的检查顺序|L’Hôpital’s rule]]: indeterminate forms and hypotheses*
<!-- bilingual-en:end -->

设 \(f,g\) 在 \(a\) 的穿孔邻域可导，\(g'(x)\ne0\)，并且
<!-- bilingual-en:start -->
Let \(f\) and \(g\) be differentiable on a punctured neighborhood of \(a\), with \(g'(x)\ne0\), and suppose that
<!-- bilingual-en:end -->

$$
\lim_{x\to a}f(x)=\lim_{x\to a}g(x)=0
$$

或二者绝对值都趋于无穷。若
<!-- bilingual-en:start -->
or both absolute values tend to infinity. If
<!-- bilingual-en:end -->

$$
\lim_{x\to a}\frac{f'(x)}{g'(x)}=L
$$

存在（可含 \(\pm\infty\)），则在相应条件下
<!-- bilingual-en:start -->
exists (can include \(\pm\infty\)), then under appropriate conditions
<!-- bilingual-en:end -->

$$
\boxed{\lim_{x\to a}\frac{f(x)}{g(x)}=L}.
$$

法则处理的是**比值的极限**，不是函数恒等式；一般不能写 \(f/g=f'/g'\)。
<!-- bilingual-en:start -->
The rule concerns the **limit of a ratio**, not an identity between functions; in general, one may not write $f/g=f'/g'$.
<!-- bilingual-en:end -->

### 为什么成立：Cauchy MVT
<!-- bilingual-en:start -->
*Why it is true: Cauchy MVT*
<!-- bilingual-en:end -->

先看 \(0/0\)，并暂设 \(f(a)=g(a)=0\)。对靠近 \(a\) 的 \(x\)，Cauchy 平均值定理保证在 \(a,x\) 之间存在 \(c\)：
<!-- bilingual-en:start -->
First consider the $0/0$ case and temporarily set $f(a)=g(a)=0$. For $x$ near $a$, the Cauchy mean value theorem guarantees some $c$ between $a$ and $x$ such that
<!-- bilingual-en:end -->

$$
\frac{f(x)-f(a)}{g(x)-g(a)}
=\frac{f'(c)}{g'(c)}.
$$

左边就是 \(f(x)/g(x)\)。当 \(x\to a\)，夹在二者之间的 \(c\to a\)，所以若导数比趋于 \(L\)，原比值也趋于 \(L\)。
<!-- bilingual-en:start -->
The left-hand side is $f(x)/g(x)$. Because $c$ lies between $a$ and $x$, taking $x\to a$ also forces $c\to a$. Hence, if the derivative ratio tends to $L$, so does the original ratio.
<!-- bilingual-en:end -->

> [!note] 严格性说明
> \(\infty/\infty\)、单侧极限和无穷远极限需要相应版本的 Cauchy MVT 与附加控制；法则的条件不可只凭结果看似正确而省略。
> <!-- bilingual-en:start -->
> \(\infty/\infty\), one-sided limits, and infinity limits require the appropriate version of Cauchy MVT and additional control; the conditions of the rule cannot be omitted simply because the results appear to be correct.
> <!-- bilingual-en:end -->

### 例
<!-- bilingual-en:start -->
*Example*
<!-- bilingual-en:end -->

$$
\lim_{x\to0}\frac{e^x-1}{x}
=\lim_{x\to0}\frac{e^x}{1}=1.
$$

这个极限也正是 \(e^x\) 在 0 的导数；识别导数定义比 L’Hôpital 更能解释其意义。
<!-- bilingual-en:start -->
This limit is precisely the derivative of $e^x$ at $0$. Recognizing the definition of a derivative explains its meaning more directly than applying L’Hôpital’s rule.
<!-- bilingual-en:end -->

### 87a–87c：课件例题与证明流程
<!-- bilingual-en:start -->
*87a–87c: Examples and proof strategy from the slides*
<!-- bilingual-en:end -->

课件先计算一个本可因式分解的极限：
<!-- bilingual-en:start -->
The slides begin by evaluating a limit that could also be handled by factorization:
<!-- bilingual-en:end -->

$$
\lim_{x\to1}\frac{x^{10}-1}{x^2-1}.
$$

直接代入是 $0/0$，分子、分母分别求导后
<!-- bilingual-en:start -->
Direct substitution gives $0/0$. Differentiating the numerator and denominator gives
<!-- bilingual-en:end -->

$$
\lim_{x\to1}\frac{10x^9}{2x}=5.
$$

代数法也给同一答案：约去 $x-1$ 后再代入。两种方法相符说明 L’Hôpital 没有在制造新函数，而是在提取分子与分母的**一阶主导变化**。
<!-- bilingual-en:start -->
Algebra gives the same answer: cancel the common factor $x-1$, then substitute. The agreement shows that L’Hôpital’s rule is not inventing a new function; it is extracting the **leading first-order change** in the numerator and denominator.
<!-- bilingual-en:end -->

证明时应把逻辑顺序写清：
<!-- bilingual-en:start -->
The logical sequence should be clearly stated in the proof:
<!-- bilingual-en:end -->

1. 目标是原比值 $f(x)/g(x)$ 的极限；
2. 用 Cauchy MVT 构造中间点 $c_x$，使原比值等于某个导数比；
3. 因 $c_x$ 位于 $a$ 与 $x$ 之间，$x\to a$ 强迫 $c_x\to a$；
4. 只有已知 $f'(t)/g'(t)\to L$，才能把 $t=c_x$ 代入这一极限结论。
<!-- bilingual-en:start -->

&nbsp;
**1.** The target is the limit of the original ratio $f(x)/g(x)$;<br>
**2.** Use the Cauchy MVT to construct an intermediate point $c_x$ at which the original ratio equals a derivative ratio;<br>
**3.** Because $c_x$ lies between $a$ and $x$, $x\to a$ forces $c_x\to a$;<br>
**4.** Only after establishing $f'(t)/g'(t)\to L$ may we apply that limit along the points $t=c_x$.<br>
<!-- bilingual-en:end -->

这也解释了为何“导数比的极限存在”是定理假设，而不是计算后的可选检查。
<!-- bilingual-en:start -->
This explains why existence of the derivative-ratio limit is a hypothesis of the theorem, not an optional check after the calculation.
<!-- bilingual-en:end -->

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
> <!-- bilingual-en:start -->
> **(a) Linear approximation.** Since $\sin x\approx x$ and $\cos x\approx1$, the denominator is approximated by $0$. Linear information is therefore insufficient; this signals that a higher-order approximation is needed.
>
> **(b) Quadratic approximation.** Since $\sin x=x+O(x^3)$ and $1-\cos x=x^2/2+O(x^4)$,
> $$
> \frac{\sin x}{1-\cos x}
> =\frac{x+O(x^3)}{x^2/2+O(x^4)}
> \sim\frac2x.
> $$
> Hence,
> $$
> \lim_{x\to0^+}\frac{\sin x}{1-\cos x}=+\infty,
> \qquad
> \lim_{x\to0^-}\frac{\sin x}{1-\cos x}=-\infty.
> $$
> **Conclusion: the two-sided limit does not exist.** Writing only “infinity” loses the opposite signs of the one-sided limits. The identity $\sin x/(1-\cos x)=\cot(x/2)$ gives the same check.
> <!-- bilingual-en:end -->

> [!question]- 三道自检题与答案
> 1. $\lim_{x\to0}\sin x/x$ 可用 L’Hôpital，结果为 $1$；但若正弦导数本身是靠该极限证明的，就会循环论证。
> 2. $\lim_{x\to2}(x^2-4)/(x-2)$ 可用，但先约分更直接，结果为 $4$。
> 3. 不能写 $f/g=f'/g'$；例如 $x^2/x=x$，而导数比为 $2x$，只有符合条件时二者的极限才由定理联系。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** L’Hôpital’s rule may be used for $\lim_{x\to0}\sin x/x$, giving $1$; however, using it would be circular if the derivative of sine was itself established from this limit.<br>
> **2.** L’Hôpital’s rule applies to $\lim_{x\to2}(x^2-4)/(x-2)$, but cancelling the common factor first is more direct; the limit is $4$.<br>
> **3.** One may not assert that $f/g=f'/g'$. For example, $x^2/x=x$, whereas the derivative ratio is $2x$. L’Hôpital’s rule relates only their limits, and only under its hypotheses.<br>
> <!-- bilingual-en:end -->

## Session 88：Examples of L’Hôpital’s Rule

### 重复使用
<!-- bilingual-en:start -->
*Repeated applications*
<!-- bilingual-en:end -->

$$
\lim_{x\to0}\frac{e^x-1-x}{x^2}
$$

第一次仍为 \(0/0\)：
<!-- bilingual-en:start -->
The first application still leaves the form $0/0$:
<!-- bilingual-en:end -->

$$
=\lim_{x\to0}\frac{e^x-1}{2x}.
$$

第二次仍为 \(0/0\)：
<!-- bilingual-en:start -->
After differentiating once, the new ratio is still $0/0$:
<!-- bilingual-en:end -->

$$
=\lim_{x\to0}\frac{e^x}{2}=\frac12.
$$

每次使用前都重新确认未定式。
<!-- bilingual-en:start -->
Recheck that the expression is indeterminate before every application.
<!-- bilingual-en:end -->

### 与 Taylor 近似比较
<!-- bilingual-en:start -->
*Comparison with a Taylor approximation*
<!-- bilingual-en:end -->

$$
e^x=1+x+\frac{x^2}{2}+O(x^3),
$$

所以
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

$$
\frac{e^x-1-x}{x^2}
=\frac12+O(x).
$$

L’Hôpital 给答案，Taylor 还说明误差阶数。
<!-- bilingual-en:start -->
L’Hôpital gives the answer, and Taylor explains the order of the error.
<!-- bilingual-en:end -->

### 扩展到无穷远
<!-- bilingual-en:start -->
*Extension to limits at infinity*
<!-- bilingual-en:end -->

$$
\lim_{x\to\infty}\frac{\ln x}{x}
=\lim_{x\to\infty}\frac{1/x}{1}=0.
$$

这说明对数增长慢于线性函数。
<!-- bilingual-en:start -->
This means that logarithmic growth is slower than linear growth.
<!-- bilingual-en:end -->

### 88a–88d：课件中的三类用法
<!-- bilingual-en:start -->
*88a–88d: Three uses presented in the slides*
<!-- bilingual-en:end -->

**一次使用：**
<!-- bilingual-en:start -->
**One application:**
<!-- bilingual-en:end -->

$$
\lim_{x\to0}\frac{\sin5x}{\sin2x}
=\lim_{x\to0}\frac{5\cos5x}{2\cos2x}
=\frac52.
$$

**重复使用：**
<!-- bilingual-en:start -->
**Repeated applications:**
<!-- bilingual-en:end -->

$$
\begin{aligned}
\lim_{x\to0}\frac{\cos x-1}{x^2}
&=\lim_{x\to0}\frac{-\sin x}{2x}\\
&=\lim_{x\to0}\frac{-\cos x}{2}=-\frac12.
\end{aligned}
$$

第二次之前，新的比值仍是 $0/0$；这一步确认不可省略。二次近似 $\cos x=1-x^2/2+O(x^4)$ 不仅给出 $-1/2$，还说明被忽略项是 $O(x^2)$。
<!-- bilingual-en:start -->
Before applying the rule a second time, verify that the new ratio is still $0/0$; this check is essential. The quadratic approximation $\cos x=1-x^2/2+O(x^4)$ not only gives $-1/2$, but also shows that the neglected error is $O(x^2)$.
<!-- bilingual-en:end -->

**无穷远版本：**把 $x\to\infty$ 看成在越来越长区间上比较增长。若分子分母都趋于无穷，且导数比有极限，才可使用相应版本。
<!-- bilingual-en:start -->
**At infinity:** Think of $x\to\infty$ as comparing growth over ever larger inputs. The corresponding form of L’Hôpital’s rule applies when both numerator and denominator tend to infinity and the derivative ratio has the required limit.
<!-- bilingual-en:end -->

### 本地材料与练习

- [[Ses88a_Lecture_Notes.pdf|88a Example]]
- [[Ses88b_Lecture_Notes.pdf|88b Repeating the Rule]]
- [[Ses88c_Lecture_Notes.pdf|88c Comparison with Approximation]]
- [[Ses88d_Lecture_Notes.pdf|88d Extensions]]
- [[Exercise088_Problems.pdf|Exercise 88]]
- [[Exercise088_Solutions.pdf|Exercise 88 Solutions]]

> [!warning] 易错点
> 若第一次求导后的极限已不是未定式，必须停止；不能为了“化简”继续求导。
> <!-- bilingual-en:start -->
> If differentiating once removes the indeterminate form, stop. Do not keep differentiating merely to make the expression look simpler.
> <!-- bilingual-en:end -->

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
> <!-- bilingual-en:start -->
> **(a)** The ratio $e^x/x$ has the form $\infty/\infty$:
> $$
> \lim_{x\to\infty}\frac{e^x}{x}
> =\lim_{x\to\infty}\frac{e^x}{1}=+\infty.
> $$
> The extended form of L’Hôpital's rule permits the derivative ratio to tend to $+\infty$, so the original ratio does as well.
>
> **(b)**
> $$
> \lim_{x\to\infty}\frac{x}{e^x}
> =\lim_{x\to\infty}\frac1{e^x}=0.
> $$
> The two ratios are reciprocals and express the same conclusion: exponential growth eventually dominates linear growth.
> <!-- bilingual-en:end -->

> [!question]- 三道自检题与答案
> 1. $\lim_{x\to0}(e^x-1-x)/x^2=1/2$，需连续使用两次。
> 2. $\lim_{x\to\infty}x^3/e^x=0$，连续三次后成为 $6/e^x$。
> 3. $\lim_{x\to0}(1+x)/x$ 不是未定式；右极限 $+\infty$、左极限 $-\infty$，不能再求导。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** Applying L’Hôpital’s rule twice to $\lim_{x\to0}(e^x-1-x)/x^2$ gives $1/2$.<br>
> **2.** Applying it three times to $\lim_{x\to\infty}x^3/e^x$ reduces the ratio to $6/e^x$, whose limit is $0$.<br>
> **3.** $\lim_{x\to0}(1+x)/x$ is not an indeterminate form. Its one-sided limits are $+\infty$ and $-\infty$, so L’Hôpital’s rule does not apply.<br>
> <!-- bilingual-en:end -->

## Session 89：Rates of Growth

### 增长层级
<!-- bilingual-en:start -->
*Growth hierarchy*
<!-- bilingual-en:end -->

对任意固定 \(p>0\)、\(q>0\)：
<!-- bilingual-en:start -->
For any fixed \(p>0\), \(q>0\):
<!-- bilingual-en:end -->

$$
\ln x \ll x^p \ll e^{qx}
\qquad(x\to\infty).
$$

符号 \(f\ll g\) 表示 \(f/g\to0\)。
<!-- bilingual-en:start -->
The symbol \(f\ll g\) indicates \(f/g\to0\).
<!-- bilingual-en:end -->

证明 \(\ln x\ll x^p\)：
<!-- bilingual-en:start -->
To prove \(\ln x\ll x^p\),
<!-- bilingual-en:end -->

$$
\lim_{x\to\infty}\frac{\ln x}{x^p}
=\lim_{x\to\infty}\frac{1/x}{px^{p-1}}
=\lim_{x\to\infty}\frac1{px^p}=0.
$$

证明多项式慢于指数：对整数 \(n\)，反复使用 L’Hôpital：
<!-- bilingual-en:start -->
To prove that polynomial growth is slower than exponential growth, apply L’Hôpital's rule repeatedly for an integer \(n\):
<!-- bilingual-en:end -->

$$
\lim_{x\to\infty}\frac{x^n}{e^{qx}}
=\lim_{x\to\infty}\frac{n!}{q^ne^{qx}}=0.
$$

### 89a–89c：增长与衰减是一件事的两面
<!-- bilingual-en:start -->
*89a-89c: Growth and decay are two sides of the same coin*
<!-- bilingual-en:end -->

课件依次处理：
<!-- bilingual-en:start -->
The slides treat the following cases in order:
<!-- bilingual-en:end -->

$$
\lim_{x\to0^+}x\ln x
=\lim_{x\to0^+}\frac{\ln x}{1/x}
=\lim_{x\to0^+}\frac{1/x}{-1/x^2}
=\lim_{x\to0^+}(-x)=0.
$$

虽然 $\ln x\to-\infty$，但 $x$ 趋零更快，所以乘积从负侧趋零。对 $p>0$，
<!-- bilingual-en:start -->
Although $\ln x\to-\infty$, the factor $x$ tends to zero quickly enough that the product approaches $0$ from below. For $p>0$,
<!-- bilingual-en:end -->

$$
\lim_{x\to\infty}xe^{-px}
=\lim_{x\to\infty}\frac{x}{e^{px}}
=\lim_{x\to\infty}\frac1{pe^{px}}=0.
$$

最后
<!-- bilingual-en:start -->
Finally,
<!-- bilingual-en:end -->

$$
\lim_{x\to\infty}\frac{\ln x}{x^{1/3}}
=\lim_{x\to\infty}\frac{1/x}{(1/3)x^{-2/3}}
=\lim_{x\to\infty}\frac3{x^{1/3}}=0.
$$

取倒数就把增长层级翻译为衰减层级；“谁增长更快”与“谁的倒数衰减更快”是同一判断。
<!-- bilingual-en:start -->
Taking reciprocals converts growth comparisons into decay comparisons: asking which function grows faster is equivalent to asking which reciprocal decays faster.
<!-- bilingual-en:end -->

### 本地材料

- [[Ses89a_Lecture_Notes.pdf|89a Growth of \(\ln x\)]]
- [[Ses89b_Lecture_Notes.pdf|89b Growth of \(e^{px}\)]]
- [[Ses89c_Lecture_Notes.pdf|89c Comparing \(\ln x\) and \(x^{1/3}\)]]

> [!question]- 三问自检
> 1. \(x^{100}/e^x\)？2. \(\ln x/\sqrt x\)？3. \(e^{-x}x^5\)？
>
> 答：都趋于 0；第三个等于 \(x^5/e^x\)。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** \(x^{100}/e^x\)?<br>
> **2.** \(\ln x/\sqrt x\)?<br>
> **3.** \(e^{-x}x^5\)?<br>
> Answer: All tend to 0; the third equals \(x^5/e^x\).
> <!-- bilingual-en:end -->

## Session 90：Advanced Indeterminate Forms

### 幂型未定式
<!-- bilingual-en:start -->
*Indeterminate powers*
<!-- bilingual-en:end -->

若
<!-- bilingual-en:start -->
Suppose
<!-- bilingual-en:end -->

$$
y=f(x)^{g(x)}>0,
$$

先取对数：
<!-- bilingual-en:start -->
Take logarithms first:
<!-- bilingual-en:end -->

$$
\ln y=g(x)\ln f(x).
$$

求出右侧极限 \(A\) 后，再由指数连续性得到 \(y\to e^A\)。
<!-- bilingual-en:start -->
Once the limit on the right is found to be \(A\), continuity of the exponential function gives \(y\to e^A\).
<!-- bilingual-en:end -->

### 例：\(0^0\)
<!-- bilingual-en:start -->
*Example: $0^0$*
<!-- bilingual-en:end -->

$$
\lim_{x\to0^+}x^x.
$$

令 \(y=x^x\)：
<!-- bilingual-en:start -->
Let $y=x^x$. Then
<!-- bilingual-en:end -->

$$
\ln y=x\ln x=\frac{\ln x}{1/x}.
$$

这是 \((-\infty)/\infty\)：
<!-- bilingual-en:start -->
This is \((-\infty)/\infty\):
<!-- bilingual-en:end -->

$$
\lim_{x\to0^+}\frac{1/x}{-1/x^2}
=\lim_{x\to0^+}(-x)=0.
$$

所以
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

$$
\boxed{\lim_{x\to0^+}x^x=e^0=1}.
$$

### 其他改写
<!-- bilingual-en:start -->
*Other reformulations*
<!-- bilingual-en:end -->

$$
0\cdot\infty:
\quad fg=\frac f{1/g},
$$

$$
\infty-\infty:
\quad\text{通分、有理化或提取主项}.
$$

### 90b–90c：最重要的反例——Look before you L’Hôp
<!-- bilingual-en:start -->
*90b–90c: The crucial warning—look before you L’Hôp*
<!-- bilingual-en:end -->

考虑
<!-- bilingual-en:start -->
Consider
<!-- bilingual-en:end -->

$$
\lim_{x\to0}\frac{\sin x}{x^2}.
$$

第一次使用后得到 $\cos x/(2x)$。此时分子趋于 $1$、分母趋于 $0$，已经**不是** $0/0$；若错误地再次求导，会得到 $0$，与真实行为矛盾。由 $\sin x\sim x$：
<!-- bilingual-en:start -->
After one application, the ratio becomes $\cos x/(2x)$. Its numerator tends to $1$ while its denominator tends to $0$, so it is **no longer** a $0/0$ form. Differentiating again would incorrectly produce $0$, contradicting the true behaviour. Since $\sin x\sim x$,
<!-- bilingual-en:end -->

$$
\lim_{x\to0^+}\frac{\sin x}{x^2}=+\infty,
\qquad
\lim_{x\to0^-}\frac{\sin x}{x^2}=-\infty.
$$

课件的口令是 “Look before you L’Hôp”：每求导一次，都回到直接代入重新分类。对于
<!-- bilingual-en:start -->
The rule of thumb in the slides is “Look before you L’Hôp”: after each differentiation, substitute again and reclassify the form before deciding whether another application is valid. For
<!-- bilingual-en:end -->

$$
\lim_{x\to\infty}\frac{x^5-2x^4+1}{x^4+2},
$$

除以 $x^4$ 或比较最高次项立刻得 $+\infty$，比机械求导四次更清楚。
<!-- bilingual-en:start -->
Dividing by $x^4$, or simply comparing the leading terms, gives $+\infty$ immediately. This is clearer than differentiating four times mechanically.
<!-- bilingual-en:end -->

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
> <!-- bilingual-en:start -->
> This has the indeterminate form $\infty^0$. Let $y=x^{1/x}>0$. Then
> $$
> \ln y=\frac{\ln x}{x}.
> $$
> The right-hand side has the form $\infty/\infty$, so
> $$
> \lim_{x\to\infty}\frac{\ln x}{x}
> =\lim_{x\to\infty}\frac{1/x}{1}=0.
> $$
> By continuity of the exponential function,
> $$
> \boxed{\lim_{x\to\infty}x^{1/x}=e^0=1.}
> $$
> Equivalently, the sequence $\sqrt[n]{n}$ tends to $1$.
> <!-- bilingual-en:end -->

> [!question]- 三道自检题与答案
> 1. $\lim_{x\to0^+}(1+x)^{1/x}=e$：取对数后研究 $\ln(1+x)/x\to1$。
> 2. $\lim_{x\to\infty}x^2e^{-x}=0$：改写为 $x^2/e^x$。
> 3. $\lim_{x\to0}(1/x-1/\sin x)=0$：先通分为 $(\sin x-x)/(x\sin x)$，再合法使用 L’Hôpital 或三阶近似。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** $\lim_{x\to0^+}(1+x)^{1/x}=e$: take logarithms and study $\ln(1+x)/x\to1$.<br>
> **2.** $\lim_{x\to\infty}x^2e^{-x}=0$: rewrite it as $x^2/e^x$.<br>
> **3.** $\lim_{x\to0}(1/x-1/\sin x)=0$: first combine the fractions to obtain $(\sin x-x)/(x\sin x)$, then use L’Hôpital's rule when valid or a third-order approximation.<br>
> <!-- bilingual-en:end -->

## Session 91：Improper Integrals

### 无穷区间
<!-- bilingual-en:start -->
*Infinite intervals*
<!-- bilingual-en:end -->

[[反常积分#反常积分|反常积分]]定义为
<!-- bilingual-en:start -->
An [[反常积分#反常积分|improper integral]] over an infinite interval is defined by
<!-- bilingual-en:end -->

$$
\int_a^\infty f(x)dx
=\lim_{b\to\infty}\int_a^bf(x)dx.
$$

极限有限称收敛，否则发散。符号 \(\infty\) 只是极限方向，不能直接代入原函数。
<!-- bilingual-en:start -->
If the finite-endpoint limit exists, the improper integral converges; otherwise it diverges. The symbol \(\infty\) indicates only the direction of the limit and cannot be substituted directly into an antiderivative.
<!-- bilingual-en:end -->

### \(p\)-积分
<!-- bilingual-en:start -->
*$p$-integrals*
<!-- bilingual-en:end -->

$$
\int_1^\infty\frac{dx}{x^p}.
$$

若 \(p\ne1\)：
<!-- bilingual-en:start -->
If \(p\ne1\):
<!-- bilingual-en:end -->

$$
\int_1^b x^{-p}dx
=\frac{b^{1-p}-1}{1-p}.
$$

- \(p>1\)：\(b^{1-p}\to0\)，积分收敛为 \(1/(p-1)\)；
- \(p<1\)：幂次非负，发散；
- \(p=1\)：\(\ln b\to\infty\)，发散。
<!-- bilingual-en:start -->
- If $p>1$, then $b^{1-p}\to0$, and the integral converges to $1/(p-1)$;
- if $p<1$, the nonnegative power of $b$ makes the integral diverge;
- if $p=1$, then $\ln b\to\infty$, so the integral diverges.
<!-- bilingual-en:end -->

因此
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

$$
\boxed{\int_1^\infty x^{-p}dx\text{ 收敛}\iff p>1}.
$$

### 91a–91e：课件例题与“尾部”直觉
<!-- bilingual-en:start -->
*91a–91e: Examples from the slides and the intuition of a “tail”*
<!-- bilingual-en:end -->

最基本的指数尾部（$k>0$）是
<!-- bilingual-en:start -->
The most basic exponential tail ($k>0$) is
<!-- bilingual-en:end -->

$$
\begin{aligned}
\int_0^\infty e^{-kx}dx
&=\lim_{N\to\infty}\left[-\frac1ke^{-kx}\right]_0^N\\
&=\frac1k.
\end{aligned}
$$

$k>0$ 是必要条件；$k=0$ 时被积函数恒为 $1$，$k<0$ 时反而指数增长。物理上 $Ae^{-kt}$ 可描述衰变率，总累计量为 $A/k$。
<!-- bilingual-en:start -->
The condition $k>0$ is necessary: when $k=0$ the integrand is constantly $1$, and when $k<0$ it grows exponentially. In applications, $Ae^{-kt}$ can model a decaying quantity, whose total accumulation is $A/k$.
<!-- bilingual-en:end -->

课件还指出高斯积分
<!-- bilingual-en:start -->
The slides also mention the Gaussian integral
<!-- bilingual-en:end -->

$$
\int_{-\infty}^{\infty}e^{-x^2}dx=\sqrt\pi.
$$

本单元暂不推导精确值，但 Session 92 会用比较证明它至少收敛。与之相对，
<!-- bilingual-en:start -->
This unit does not yet derive the exact value, but Session 92 will at least prove convergence by comparison. In contrast,
<!-- bilingual-en:end -->

$$
\int_1^\infty\frac{dx}{x}
=\lim_{N\to\infty}\ln N=\infty
$$

说明“函数趋于零”仍不足以保证尾面积有限；关键是趋零速度。
<!-- bilingual-en:start -->
This shows that an integrand tending to zero is not enough to make the tail area finite; the rate of decay is decisive.
<!-- bilingual-en:end -->

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
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** What is $\int_1^\infty 1/x^2\,dx$?<br>
> **2.** What happens to $\int_1^\infty 1/x\,dx$?<br>
> **3.** Does an integrand tending to zero guarantee convergence?<br>
> Answer: The first converges to $1$; the second diverges; no—$1/x$ is the standard counterexample.
> <!-- bilingual-en:end -->

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
> <!-- bilingual-en:start -->
> Begin with a finite upper limit:
> $$
> I=\lim_{N\to\infty}\int_1^N(5x+2)^{-2}dx.
> $$
> Let $u=5x+2$, or directly use the antiderivative $-1/[5(5x+2)]$:
> $$
> \begin{aligned}
> I
> &=\lim_{N\to\infty}\left[-\frac1{5(5x+2)}\right]_1^N\\
> &=\lim_{N\to\infty}\left(\frac1{35}-\frac1{5(5N+2)}\right)\\
> &=\boxed{\frac1{35}}.
> \end{aligned}
> $$
> The result is positive and smaller than $\int_1^\infty dx/(25x^2)=1/25$, so its order of magnitude is reasonable.
> <!-- bilingual-en:end -->

## Session 92：Integral Comparison

设 \(0\le f(x)\le g(x)\) 对充分大的 \(x\) 成立：
<!-- bilingual-en:start -->
Suppose $0\le f(x)\le g(x)$ for all sufficiently large $x$:
<!-- bilingual-en:end -->

- 若 \(\int g\) 收敛，则 \(\int f\) 收敛；
- 若 \(\int f\) 发散，则 \(\int g\) 发散。
<!-- bilingual-en:start -->
- If \(\int g\) converges, \(\int f\) converges;
- If \(\int f\) diverges, \(\int g\) diverges.
<!-- bilingual-en:end -->

方向来自面积大小，不能倒用。
<!-- bilingual-en:start -->
The direction of each implication follows from the ordering of the areas and cannot be reversed.
<!-- bilingual-en:end -->

### 极限比较
<!-- bilingual-en:start -->
*Limit comparison*
<!-- bilingual-en:end -->

若 \(f,g>0\) 且
<!-- bilingual-en:start -->
If \(f,g>0\) and
<!-- bilingual-en:end -->

$$
\lim_{x\to\infty}\frac{f(x)}{g(x)}=c,
\qquad0<c<\infty,
$$

则二者同敛散。原因是当 \(x\) 足够大时，\(f\) 被两个正常数倍的 \(g\) 夹住。
<!-- bilingual-en:start -->
the two improper integrals have the same convergence behaviour. For sufficiently large $x$, the function $f$ is bounded between two positive constant multiples of $g$.
<!-- bilingual-en:end -->

例：
<!-- bilingual-en:start -->
Example:
<!-- bilingual-en:end -->

$$
\frac1{x^2+3x}\sim\frac1{x^2},
$$

故其从 1 到无穷的积分收敛。
<!-- bilingual-en:start -->
Therefore, its integral from $1$ to infinity converges.
<!-- bilingual-en:end -->

### 92a–92c：三个课件比较
<!-- bilingual-en:start -->
*92a–92c: Three comparisons from the slides*
<!-- bilingual-en:end -->

**与 $1/x$ 比较：**
<!-- bilingual-en:start -->
**vs. $1/x$:**
<!-- bilingual-en:end -->

$$
\frac1{\sqrt{x^2+10}}\sim\frac1x,
$$

所以 $\int_0^\infty dx/\sqrt{x^2+10}$ 的尾部发散。比较时从 $1$ 或任意正数开始，避免让 $1/x$ 在 $0$ 的奇点混入“无穷远尾部”问题。
<!-- bilingual-en:start -->
Hence the tail of $\int_0^\infty dx/\sqrt{x^2+10}$ diverges. Start the comparison at $1$, or at any other positive number, so that the singularity of $1/x$ at $0$ is not confused with the behaviour at infinity.
<!-- bilingual-en:end -->

**与 $p$-积分比较：**
<!-- bilingual-en:start -->
**Comparison with a $p$-integral:**
<!-- bilingual-en:end -->

$$
\frac1{\sqrt{x^3+3}}\sim\frac1{x^{3/2}},
$$

故 $\int_{10}^\infty dx/\sqrt{x^3+3}$ 收敛，即使原函数难以写出。
<!-- bilingual-en:start -->
Therefore, $\int_{10}^\infty dx/\sqrt{x^3+3}$ converges even though an elementary antiderivative is difficult to write down.
<!-- bilingual-en:end -->

**普通比较而非极限比较：**当 $x\ge1$，$x^2\ge x$，所以
<!-- bilingual-en:start -->
**Direct comparison rather than limit comparison:** When $x\ge1$, $x^2\ge x$, so
<!-- bilingual-en:end -->

$$
0<e^{-x^2}\le e^{-x}.
$$

有限区间 $[0,1]$ 上连续函数积分有限，而 $\int_1^\infty e^{-x}dx$ 收敛，因此利用偶性可知整个高斯积分收敛。
<!-- bilingual-en:start -->
The integral of a continuous function over the finite interval $[0,1]$ is finite, while $\int_1^\infty e^{-x}dx$ converges. By evenness, the full Gaussian integral therefore converges.
<!-- bilingual-en:end -->

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
> <!-- bilingual-en:start -->
> Let
> $$
> f(x)=\frac1{(5x+2)^2},
> \qquad
> g(x)=\frac1{25x^2}.
> $$
> Then
> $$
> \lim_{x\to\infty}\frac{f(x)}{g(x)}
> =\lim_{x\to\infty}\frac{25x^2}{(5x+2)^2}=1.
> $$
> Since $\int_1^\infty g(x)dx=1/25$ converges, the limit-comparison theorem gives
> $$
> \boxed{\int_1^\infty\frac{dx}{(5x+2)^2}\text{ converges}.}
> $$
> Comparison determines convergence, not the exact value $1/35$ found in Exercise 091.
> <!-- bilingual-en:end -->

> [!question]- 三道自检题与答案
> 1. $\int_1^\infty dx/\sqrt{x^4+1}$ 与 $1/x^2$ 极限比较，收敛。
> 2. $\int_1^\infty x/(x^2+1)dx$ 与 $1/x$ 极限比较，发散。
> 3. 若 $0\le f\le g$ 且 $\int f$ 收敛，不能推出 $\int g$ 收敛；较大的尾部仍可能发散。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** Compare $\int_1^\infty dx/\sqrt{x^4+1}$ with $1/x^2$ by limit comparison; it converges.<br>
> **2.** Compare $\int_1^\infty x/(x^2+1)dx$ with $1/x$; it diverges.<br>
> **3.** From $0\le f\le g$ and convergence of $\int f$, one cannot infer convergence of $\int g$; the larger tail may still diverge.<br>
> <!-- bilingual-en:end -->

## Session 93：Singularities

### 有限端点奇点
<!-- bilingual-en:start -->
*A singularity at a finite endpoint*
<!-- bilingual-en:end -->

若 \(f\) 在 \(a\) 无界，定义
<!-- bilingual-en:start -->
If \(f\) is unbounded at \(a\), define
<!-- bilingual-en:end -->

$$
\int_a^bf(x)dx
=\lim_{\varepsilon\to0^+}\int_{a+\varepsilon}^bf(x)dx.
$$

典型：
<!-- bilingual-en:start -->
A standard case is
<!-- bilingual-en:end -->

$$
\int_0^1x^{-p}dx
\text{ 收敛}\iff p<1.
$$

注意与无穷远的 \(p>1\) 条件正好相反。
<!-- bilingual-en:start -->
This is exactly the reverse of the condition $p>1$ for convergence at infinity.
<!-- bilingual-en:end -->

### 区间内部奇点
<!-- bilingual-en:start -->
*A singularity inside the interval*
<!-- bilingual-en:end -->

若 \(c\in(a,b)\) 是奇点，必须拆成两个独立反常积分：
<!-- bilingual-en:start -->
If \(c\in(a,b)\) is a singularity, split the expression into two independent improper integrals:
<!-- bilingual-en:end -->

$$
\int_a^bf
=\int_a^cf+\int_c^bf.
$$

只有两边都收敛，总积分才收敛。不能让左右无穷“相消”；那是 Cauchy 主值，不是通常反常积分。
<!-- bilingual-en:start -->
The original improper integral converges only if both one-sided integrals converge separately. Divergences on the two sides may not be cancelled; doing so defines a Cauchy principal value, not an ordinary improper integral.
<!-- bilingual-en:end -->

### 93a–93d：先找所有坏点，再逐个处理
<!-- bilingual-en:start -->
*93a–93d: Locate every singular point, then handle each one separately*
<!-- bilingual-en:end -->

课件用一个“荒谬答案”警示跨奇点套原函数：
<!-- bilingual-en:start -->
The slides use a deliberately absurd answer to warn against applying a single antiderivative formula across a singularity:
<!-- bilingual-en:end -->

$$
\int_{-1}^{1}\frac{dx}{x^2}
\overset{\text{错误}}=\left[-\frac1x\right]_{-1}^{1}=-2.
$$

被积函数处处非负，面积不可能为负。正确做法是在 $0$ 拆开；任一侧都像 $\int_0^1x^{-2}dx$ 一样发散，所以总积分发散。
<!-- bilingual-en:start -->
The integrand is nonnegative everywhere, so its area cannot be negative. The correct approach is to split the integral at $0$; each side diverges like $\int_0^1x^{-2}dx$, so the entire improper integral diverges.
<!-- bilingual-en:end -->

端点模型完整结论为
<!-- bilingual-en:start -->
The complete conclusion of the endpoint model is
<!-- bilingual-en:end -->

$$
\int_0^1x^{-p}dx
=\begin{cases}
\dfrac1{1-p},&p<1,\\
\text{发散},&p\ge1.
\end{cases}
$$

与无穷远模型恰好相反，因为 $x^{-p}$ 在 $0$ 附近随 $p$ 增大而更严重，在无穷远则随 $p$ 增大而衰减更快。
<!-- bilingual-en:start -->
This is the reverse of the criterion at infinity: increasing $p$ makes the singularity of $x^{-p}$ more severe near $0$, but makes it decay faster as $x\to\infty$.
<!-- bilingual-en:end -->

对
<!-- bilingual-en:start -->
For
<!-- bilingual-en:end -->

$$
\int_0^\infty\frac{dx}{(x-3)^2},
$$

既有无穷远端，又有内部奇点 $x=3$。必须在 $3$ 拆分；奇点附近与 $1/u^2$ 同型而发散，因此无论远端是否收敛，总积分都发散。
<!-- bilingual-en:start -->
there is both an infinite endpoint and an interior singularity at $x=3$. Split the integral at $3$. Near that singularity the integrand behaves like $1/u^2$ and therefore diverges, so the entire integral diverges regardless of what happens at infinity.
<!-- bilingual-en:end -->

### 本地材料

- [[Ses93a_Lecture_Notes.pdf|93a Singularities]]
- [[Ses93b_Lecture_Notes.pdf|93b Second Kind]]
- [[Ses93c_Lecture_Notes.pdf|93c Overview]]
- [[Ses93d_Lecture_Notes.pdf|93d Example]]

> [!warning] 易错点
> 发现奇点后先拆区间，再分别写极限。跨过奇点直接套原函数会掩盖发散。
> <!-- bilingual-en:start -->
> Once a singularity is found, split the interval and write the two limits separately. Evaluating one antiderivative expression across the singularity can conceal divergence.
> <!-- bilingual-en:end -->

> [!question]- 三道自检题与答案
> 1. $\int_0^1x^{-3/4}dx=4$，因为 $3/4<1$。
> 2. $\int_{-1}^{1}dx/x$ 作为通常反常积分发散；对称 Cauchy 主值为 $0$，两者不是同一概念。
> 3. $\int_0^2dx/\sqrt{|x-1|}$ 要在 $1$ 拆开，两侧指数 $p=1/2<1$，所以都收敛。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** $\int_0^1x^{-3/4}\,dx=4$ because $3/4<1$.<br>
> **2.** $\int_{-1}^{1}dx/x$ diverges as an ordinary improper integral. Its symmetric Cauchy principal value is $0$, which is a different concept.<br>
> **3.** Split $\int_0^2dx/\sqrt{|x-1|}$ at $x=1$. On each side the exponent is $p=1/2<1$, so both improper integrals converge.<br>
> <!-- bilingual-en:end -->

---

## Part B：Taylor Series

## Session 94：Infinite Series

### 定义
<!-- bilingual-en:start -->
*Definition*
<!-- bilingual-en:end -->

[[无穷级数与幂级数#数项级数与必要条件|无穷级数]]

$$
\sum_{n=1}^{\infty}a_n
$$

不是“把无穷多个数一次加完”，而是部分和序列
<!-- bilingual-en:start -->
does not mean adding infinitely many numbers all at once. It is defined through the sequence of partial sums
<!-- bilingual-en:end -->

$$
S_N=\sum_{n=1}^Na_n
$$

的极限。若 \(S_N\to S\)，称级数收敛到 \(S\)。
<!-- bilingual-en:start -->
If $S_N\to S$, the series is said to converge to $S$.
<!-- bilingual-en:end -->

### 必要条件
<!-- bilingual-en:start -->
*A necessary condition*
<!-- bilingual-en:end -->

若级数收敛，则
<!-- bilingual-en:start -->
If the series converges, then
<!-- bilingual-en:end -->

$$
a_n=S_n-S_{n-1}\to S-S=0.
$$

所以 \(a_n\not\to0\) 必发散；但 \(a_n\to0\) 不保证收敛。
<!-- bilingual-en:start -->
Therefore, if $a_n$ does not tend to zero, the series must diverge. The converse is false: $a_n\to0$ does not guarantee convergence.
<!-- bilingual-en:end -->

### 几何级数
<!-- bilingual-en:start -->
*Geometric series*
<!-- bilingual-en:end -->

$$
S_N=1+r+\cdots+r^N
=\frac{1-r^{N+1}}{1-r}.
$$

因此
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

$$
\boxed{\sum_{n=0}^{\infty}r^n=\frac1{1-r},\quad |r|<1}.
$$

有限和公式来自消项：$(1-r)S_N=1-r^{N+1}$。只有 $|r|<1$ 时余项 $r^{N+1}\to0$，才能得到无穷和。$r=1$ 时部分和无界，$r=-1$ 时在 $0,1$ 间振荡，$|r|>1$ 时单项不趋零；因此不能越过适用条件套用 $1/(1-r)$。
<!-- bilingual-en:start -->
The finite-sum formula comes from cancellation in $(1-r)S_N=1-r^{N+1}$. An infinite sum follows only when $|r|<1$, because only then does the remainder $r^{N+1}$ tend to zero. At $r=1$ the partial sums are unbounded; at $r=-1$ they oscillate between $0$ and $1$; and when $|r|>1$ the terms do not tend to zero. Thus $1/(1-r)$ must not be used outside $|r|<1$.
<!-- bilingual-en:end -->

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
> <!-- bilingual-en:start -->
> Let $S_N=\sum_{n=0}^N2^{-n}$. The base case is $S_0=1=(2^1-1)/2^0$. Assume
> $$
> S_{N-1}=\frac{2^N-1}{2^{N-1}}.
> $$
> Then
> $$
> S_N=S_{N-1}+2^{-N}
> =\frac{2(2^N-1)+1}{2^N}
> =\frac{2^{N+1}-1}{2^N}=2-2^{-N}.
> $$
> Induction proves the finite-sum formula. Only after taking the limit do we obtain
> $$
> \boxed{\sum_{n=0}^{\infty}2^{-n}=\lim_{N\to\infty}S_N=2.}
> $$
> <!-- bilingual-en:end -->

> [!question]- 三道自检题与答案
> 1. $\sum_{n=0}^\infty(1/3)^n=3/2$。
> 2. $\sum_{n=0}^\infty(-1/2)^n=2/3$，部分和振荡但振幅趋零。
> 3. $a_n\to0$ 只是必要条件；$\sum1/n$ 仍发散。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** $\sum_{n=0}^\infty(1/3)^n=3/2$.<br>
> **2.** $\sum_{n=0}^\infty(-1/2)^n=2/3$; its partial sums oscillate with shrinking amplitude.<br>
> **3.** The condition $a_n\to0$ is necessary but not sufficient; $\sum1/n$ still diverges.<br>
> <!-- bilingual-en:end -->

## Session 95：[[无穷级数与幂级数#判别与误差控制|级数收敛判别]]
<!-- bilingual-en:start -->
*Session 95: [[无穷级数与幂级数#判别与误差控制|tests for series convergence]]*
<!-- bilingual-en:end -->

对非负项级数：
<!-- bilingual-en:start -->
For series with nonnegative terms:
<!-- bilingual-en:end -->

- \(0\le a_n\le b_n\)，若 \(\sum b_n\) 收敛，则 \(\sum a_n\) 收敛；
- \(0\le b_n\le a_n\)，若 \(\sum b_n\) 发散，则 \(\sum a_n\) 发散。
<!-- bilingual-en:start -->
- \(0\le a_n\le b_n\), if \(\sum b_n\) converges, \(\sum a_n\) converges;
- \(0\le b_n\le a_n\), if \(\sum b_n\) diverges, \(\sum a_n\) diverges.
<!-- bilingual-en:end -->

极限比较：若
<!-- bilingual-en:start -->
Limit comparison: if
<!-- bilingual-en:end -->

$$
\lim_{n\to\infty}\frac{a_n}{b_n}=c,\qquad0<c<\infty,
$$

则二者同敛散。
<!-- bilingual-en:start -->
the two series either both converge or both diverge.
<!-- bilingual-en:end -->

### 积分判别
<!-- bilingual-en:start -->
*The integral test*
<!-- bilingual-en:end -->

若 \(f\) 正、连续、递减且 \(a_n=f(n)\)，则
<!-- bilingual-en:start -->
If \(f\) is positive, continuous, decreasing, and \(a_n=f(n)\), then
<!-- bilingual-en:end -->

$$
\sum_{n=N}^{\infty}a_n
$$

与
<!-- bilingual-en:start -->
and
<!-- bilingual-en:end -->

$$
\int_N^\infty f(x)dx
$$

同敛散。矩形图说明级数与面积互相夹住。
<!-- bilingual-en:start -->
have the same convergence behaviour. A rectangle comparison shows how the sum and the integral bound one another.
<!-- bilingual-en:end -->

因此
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

$$
\boxed{\sum_{n=1}^{\infty}\frac1{n^p}\text{ 收敛}\iff p>1}.
$$

### Riemann 和证明与比值判别
<!-- bilingual-en:start -->
*Riemann-sum bounds and the ratio test*
<!-- bilingual-en:end -->

对正递减 $f$，单位宽矩形给出积分与级数的上下夹逼。特别地，对调和部分和 $H_N$：
<!-- bilingual-en:start -->
For a positive decreasing function $f$, unit-width rectangles give upper and lower bounds relating its integral to the corresponding series. In particular, for the harmonic partial sum $H_N$:
<!-- bilingual-en:end -->

$$
\ln N<H_N<\ln N+1,
$$

所以 $H_N\to\infty$，且增长量级为 $\ln N$。
<!-- bilingual-en:start -->
So $H_N\to\infty$, and the growth order is $\ln N$.
<!-- bilingual-en:end -->

比值判别对一般符号级数使用绝对值：
<!-- bilingual-en:start -->
For a series with terms of arbitrary sign, the ratio test uses absolute values:
<!-- bilingual-en:end -->

$$
L=\lim_{n\to\infty}\left|\frac{a_{n+1}}{a_n}\right|.
$$

$L<1$ 时绝对收敛，$L>1$ 时发散，$L=1$ 无结论。前者因为尾部可被公比 $q<1$ 的几何级数控制；后者因为单项不能趋零。
<!-- bilingual-en:start -->
The series converges absolutely when $L<1$, diverges when $L>1$, and the test is inconclusive when $L=1$. In the first case the tail is bounded by a geometric series with ratio $q<1$; in the second, the terms cannot tend to zero.
<!-- bilingual-en:end -->

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
> <!-- bilingual-en:start -->
> **(a)** The original series $\sum1/\sqrt[3]n$ should begin at $n=1$. Its ratio is
> $$
> \frac{a_{n+1}}{a_n}=\sqrt[3]{\frac n{n+1}}\to1.
> $$
> The ratio test is therefore inconclusive; the $p$-series test with $p=1/3\le1$ shows that the series actually diverges.
>
> **(b)** For $a_n=(n!)^2/(2n)!$,
> $$
> \frac{a_{n+1}}{a_n}
> =\frac{(n+1)^2}{(2n+1)(2n+2)}
> =\frac{n+1}{2(2n+1)}\to\frac14<1,
> $$
> so the series converges.
>
> **(c)** For $a_n=(-3)^n/n!$, take absolute values:
> $$
> \left|\frac{a_{n+1}}{a_n}\right|=\frac3{n+1}\to0.
> $$
> Hence the series converges absolutely.
> <!-- bilingual-en:end -->

> [!question]- 三道自检题与答案
> 1. $\sum n/2^n$ 的比值极限为 $1/2$，收敛。
> 2. $\sum1/n$ 的比值极限为 $1$，无结论而实际发散。
> 3. 比值判别尤其适合阶乘和指数幂；有理式尾部常用比较判别。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** For $\sum n/2^n$, the ratio-test limit is $1/2$, so the series converges.<br>
> **2.** For $\sum1/n$, the ratio-test limit is $1$, so the test is inconclusive; the harmonic series nevertheless diverges.<br>
> **3.** The ratio test is especially well suited to factorials and exponential powers. Rational-function tails are often handled more directly by comparison tests.<br>
> <!-- bilingual-en:end -->

## Session 96：Stacking Blocks

把长度相同的砖逐层向外伸。第 \(n\) 块相对下一块最多伸出
<!-- bilingual-en:start -->
Stack identical bricks, shifting each layer outward. Brick \(n\) can extend beyond the brick below it by at most
<!-- bilingual-en:end -->

$$
\frac1{2n}
$$

而不使上方 \(n\) 块的重心越过支点。总伸出量为
<!-- bilingual-en:start -->
without allowing the center of mass of the upper \(n\) blocks to pass the support point. The total overhang is
<!-- bilingual-en:end -->

$$
\frac12\sum_{n=1}^{N}\frac1n.
$$

调和级数发散，所以理论上可以伸出任意远，只是增长极慢。
<!-- bilingual-en:start -->
Because the harmonic series diverges, the overhang can in principle exceed any prescribed distance, although it grows extremely slowly.
<!-- bilingual-en:end -->

若砖长为 $1$，上方 $n$ 块总质量为 $n$；使它们的共同重心恰落在下一支点上，新增最大偏移为 $1/(2n)$。所以每一项来自力矩平衡，而不是经验猜测。
<!-- bilingual-en:start -->
If each brick has length $1$, the upper $n$ bricks have total mass $n$. Placing their combined centre of mass exactly above the next support gives a maximum additional offset of $1/(2n)$. Each term therefore comes from torque balance, not guesswork.
<!-- bilingual-en:end -->

### [[无穷级数与幂级数#数项级数与必要条件|调和级数发散证明]]：分组法
<!-- bilingual-en:start -->
*[[无穷级数与幂级数#数项级数与必要条件|A grouping proof that the harmonic series diverges]]*
<!-- bilingual-en:end -->

$$
1+\frac12+
\left(\frac13+\frac14\right)+
\left(\frac15+\cdots+\frac18\right)+\cdots
$$

第 \(k\) 组有 \(2^{k-1}\) 项，每项至少 \(1/2^k\)，所以每组和至少 \(1/2\)。组数无限，部分和无界。
<!-- bilingual-en:start -->
The $k$th group contains $2^{k-1}$ terms, each at least $1/2^k$, so every group sums to at least $1/2$. Infinitely many such groups make the partial sums unbounded.
<!-- bilingual-en:end -->

### 本地材料

- [[Ses96a_Lecture_Notes.pdf|96a Preview]]
- [[Ses96b_Lecture_Notes.pdf|96b Stacking Blocks]]

> [!question]- 三问自检
> 1. 项趋零为什么仍可发散？2. 每组下界为何 \(1/2\)？3. 发散是否意味着有限 \(N\) 时无限伸出？
>
> 答：衰减太慢；项数乘最小项；不意味，任意有限块数伸出仍有限。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** Why can a series diverge even though its terms tend to zero?<br>
> **2.** Why is each group bounded below by $1/2$?<br>
> **3.** Does divergence mean that a finite stack has infinite overhang?<br>
> Answer: The terms decay too slowly; multiply the number of terms in a group by its smallest term; no, every finite stack has finite overhang.
> <!-- bilingual-en:end -->

## Session 97：Power Series

[[无穷级数与幂级数#幂级数与 Taylor 级数|幂级数]]：

$$
\sum_{n=0}^{\infty}c_n(x-a)^n.
$$

固定 \(x\) 后它是数项级数。通常存在收敛半径 \(R\)：
<!-- bilingual-en:start -->
For each fixed $x$, this becomes a numerical series. A power series has a radius of convergence $R$:
<!-- bilingual-en:end -->

- \(|x-a|<R\)：绝对收敛；
- \(|x-a|>R\)：发散；
- 端点必须另查。
<!-- bilingual-en:start -->
- it converges absolutely when $|x-a|<R$;
- it diverges when $|x-a|>R$;
- the endpoints must be tested separately.
<!-- bilingual-en:end -->

收敛区间以中心 $a$ 对称，是因为绝对收敛由 $|x-a|$ 控制；端点不在这段论证内，可能两端都收敛、都发散或只收敛一端。
<!-- bilingual-en:start -->
The interval of convergence is symmetric about $a$ because absolute convergence is controlled by $|x-a|$. The endpoint cases are not covered by that argument: both may converge, both may diverge, or exactly one may converge.
<!-- bilingual-en:end -->

### 比值法求半径
<!-- bilingual-en:start -->
*Finding the radius with the ratio test*
<!-- bilingual-en:end -->

若极限存在：
<!-- bilingual-en:start -->
If the limit exists:
<!-- bilingual-en:end -->

$$
\left|
\frac{c_{n+1}(x-a)^{n+1}}{c_n(x-a)^n}
\right|
\to L|x-a|.
$$

要求小于 1，故 \(R=1/L\)。
<!-- bilingual-en:start -->
Convergence requires $L|x-a|<1$, so $R=1/L$.
<!-- bilingual-en:end -->

例：
<!-- bilingual-en:start -->
Example:
<!-- bilingual-en:end -->

$$
\sum_{n=0}^{\infty}\frac{x^n}{n!}
$$

比值为 \(|x|/(n+1)\to0\)，对所有 \(x\) 收敛，\(R=\infty\)。
<!-- bilingual-en:start -->
The ratio is $|x|/(n+1)\to0$, so the series converges for every $x$ and $R=\infty$.
<!-- bilingual-en:end -->

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
> <!-- bilingual-en:start -->
> $$
> \left|\frac{x^{n+1}/(n+1)}{x^n/n}\right|
> =|x|\frac n{n+1}\to|x|.
> $$
> Thus the series converges for $|x|<1$ and diverges for $|x|>1$, so
> $$
> \boxed{R=1.}
> $$
> To find the full interval, check the endpoints separately. At $x=1$ the series is harmonic and diverges; at $x=-1$ it is the alternating harmonic series and converges conditionally. The interval of convergence is therefore $[-1,1)$.
> <!-- bilingual-en:end -->

> [!question]- 三道自检题与答案
> 1. $\sum x^n/n!$ 有 $R=\infty$。
> 2. $\sum n!x^n$ 有 $R=0$。
> 3. 求出半径后仍要逐个检查 $a\pm R$。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** $\sum x^n/n!$ has radius $R=\infty$.<br>
> **2.** $\sum n!x^n$ has radius $R=0$.<br>
> **3.** After finding the radius, test the endpoints $a-R$ and $a+R$ separately.<br>
> <!-- bilingual-en:end -->

## Session 98：Taylor’s Series

### 系数从哪里来
<!-- bilingual-en:start -->
*Where do the coefficients come from?*
<!-- bilingual-en:end -->

若
<!-- bilingual-en:start -->
Suppose
<!-- bilingual-en:end -->

$$
f(x)=\sum_{n=0}^{\infty}c_n(x-a)^n
$$

可逐项求导，代入 \(x=a\)：
<!-- bilingual-en:start -->
Differentiate term by term and then set $x=a$:
<!-- bilingual-en:end -->

$$
f(a)=c_0,\quad
f'(a)=c_1,\quad
f''(a)=2!c_2,
$$

一般地
<!-- bilingual-en:start -->
In general,
<!-- bilingual-en:end -->

$$
c_n=\frac{f^{(n)}(a)}{n!}.
$$

因此 [[无穷级数与幂级数#幂级数与 Taylor 级数|Taylor series]] 是
<!-- bilingual-en:start -->
Therefore, the [[无穷级数与幂级数#幂级数与 Taylor 级数|Taylor series]] is
<!-- bilingual-en:end -->

$$
\boxed{
\sum_{n=0}^{\infty}
\frac{f^{(n)}(a)}{n!}(x-a)^n
}.
$$

### Taylor 多项式与余项
<!-- bilingual-en:start -->
*Taylor polynomials and remainder*
<!-- bilingual-en:end -->

$$
f(x)=P_n(x)+R_n(x),
$$

$$
P_n(x)=\sum_{k=0}^{n}\frac{f^{(k)}(a)}{k!}(x-a)^k.
$$

Lagrange 余项：
<!-- bilingual-en:start -->
Lagrange remainder:
<!-- bilingual-en:end -->

$$
R_n(x)=\frac{f^{(n+1)}(\xi)}{(n+1)!}(x-a)^{n+1}
$$

其中 \(\xi\) 位于 \(a,x\) 之间。若该区间上
<!-- bilingual-en:start -->
where $\xi$ lies between $a$ and $x$. If, throughout that interval,
<!-- bilingual-en:end -->

$$
|f^{(n+1)}|\le M,
$$

则
<!-- bilingual-en:start -->
then
<!-- bilingual-en:end -->

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
> <!-- bilingual-en:start -->
> Fix any $x$ and choose $d>|x|$. On $[-d,d]$, every derivative is still $e^t$, so $|f^{(N+1)}(t)|\le e^d$. Taylor's inequality gives
> $$
> |R_N(x)|\le e^d\frac{|x|^{N+1}}{(N+1)!}
> \le e^d\frac{d^{N+1}}{(N+1)!}.
> $$
> The ratio of successive terms on the right is $d/(N+2)\to0$, so that bound tends to zero. By squeezing, $R_N(x)\to0$. Since $x$ was arbitrary,
> $$
> \boxed{e^x=\sum_{n=0}^{\infty}\frac{x^n}{n!}\quad(x\in\mathbb R).}
> $$
> Every derivative of sine and cosine has absolute value at most $1$, so the same argument applies to them.
> <!-- bilingual-en:end -->

> [!question]- 三道自检题与答案
> 1. $e^x$ 的三次 Maclaurin 多项式是 $1+x+x^2/2+x^3/6$。
> 2. 系数由导数唯一确定，但函数等于级数还需 $R_N\to0$。
> 3. 余项估计所用导数上界必须在连接展开中心与目标点的区间上成立。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** The cubic Maclaurin polynomial for $e^x$ is $1+x+x^2/2+x^3/6$.<br>
> **2.** Derivatives determine the coefficients uniquely, but equality between the function and its series also requires $R_N\to0$.<br>
> **3.** The derivative bound used in a remainder estimate must hold on the entire interval joining the expansion center to the target point.<br>
> <!-- bilingual-en:end -->

## Session 99：Taylor’s Series, Continued

### 常用 Maclaurin 级数
<!-- bilingual-en:start -->
*Common Maclaurin series*
<!-- bilingual-en:end -->

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
<!-- bilingual-en:start -->
the geometric series
<!-- bilingual-en:end -->

$$
\frac1{1+x}=1-x+x^2-x^3+\cdots,\qquad |x|<1,
$$

积分得到
<!-- bilingual-en:start -->
Integrating term by term gives
<!-- bilingual-en:end -->

$$
\ln(1+x)
=x-\frac{x^2}{2}+\frac{x^3}{3}-\cdots.
$$

### 级数不一定等于原函数
<!-- bilingual-en:start -->
*A Taylor series need not equal the function it comes from*
<!-- bilingual-en:end -->

Taylor 系数由所有导数决定，但还需证明余项 \(R_n(x)\to0\)。存在光滑函数所有导数在一点都为零，却不在附近恒为零；所以“可无限求导”本身不够。
<!-- bilingual-en:start -->
The Taylor coefficients are determined by the derivatives, but equality with the function also requires proving that the remainder $R_n(x)$ tends to zero. A smooth function can have every derivative equal to zero at one point without being identically zero nearby, so infinite differentiability alone is not enough.
<!-- bilingual-en:end -->

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
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** What is the $x^5$ term in the series for $\sin x$?<br>
> **2.** Why is the series for $\ln(1+x)$ initially guaranteed only for $|x|<1$?<br>
> **3.** Does the existence of derivatives of every order automatically guarantee that a Taylor series equals its function?<br>
> Answer: $x^5/5!$; the logarithm series comes from termwise integration of a geometric series with radius $1$; no, one must also prove that the remainder tends to zero.
> <!-- bilingual-en:end -->

## Session 100：Operations on Power Series

在共同收敛区间内部，可像多项式一样逐项运算。
<!-- bilingual-en:start -->
Within their common interval of convergence, power series may be manipulated term by term like polynomials.
<!-- bilingual-en:end -->

### 乘法
<!-- bilingual-en:start -->
*Multiplication*
<!-- bilingual-en:end -->

Cauchy 乘积：
<!-- bilingual-en:start -->
Cauchy product:
<!-- bilingual-en:end -->

$$
\left(\sum_{n=0}^{\infty}a_nx^n\right)
\left(\sum_{n=0}^{\infty}b_nx^n\right)
=\sum_{n=0}^{\infty}
\left(\sum_{k=0}^{n}a_kb_{n-k}\right)x^n.
$$

### 求导与积分
<!-- bilingual-en:start -->
*Differentiation and integration*
<!-- bilingual-en:end -->

$$
\frac d{dx}\sum_{n=0}^{\infty}c_nx^n
=\sum_{n=1}^{\infty}nc_nx^{n-1},
$$

$$
\int\sum_{n=0}^{\infty}c_nx^n dx
=C+\sum_{n=0}^{\infty}\frac{c_n}{n+1}x^{n+1}.
$$

收敛半径保持不变，但端点行为可能改变。
<!-- bilingual-en:start -->
The radius of convergence remains the same, but the endpoint behavior may change.
<!-- bilingual-en:end -->

### 代入与误差函数
<!-- bilingual-en:start -->
*Substitution and the error function*
<!-- bilingual-en:end -->

$$
e^{-x^2}
=\sum_{n=0}^{\infty}\frac{(-1)^nx^{2n}}{n!}.
$$

逐项积分：
<!-- bilingual-en:start -->
Integrating term by term gives
<!-- bilingual-en:end -->

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
> <!-- bilingual-en:start -->
> **(1) Expand $\arctan(5x)$.** Differentiate first and use the geometric series:
> $$
> \frac d{dx}\arctan(5x)
> =\frac5{1+25x^2}
> =5\sum_{n=0}^{\infty}(-25x^2)^n
> =\sum_{n=0}^{\infty}(-1)^n5^{2n+1}x^{2n}.
> $$
> Integrate term by term:
> $$
> \arctan(5x)
> =C+\sum_{n=0}^{\infty}
> (-1)^n\frac{5^{2n+1}}{2n+1}x^{2n+1}.
> $$
> Substituting $x=0$ gives $C=0$. The geometric series requires $|25x^2|<1$, so the initial interval of convergence is $|x|<1/5$; its endpoints must be checked separately.
>
> **(2) Approximate $\int_0^1\sin(x^2)dx$.**
> $$
> \sin(x^2)
> =\sum_{n=0}^{\infty}
> (-1)^n\frac{x^{4n+2}}{(2n+1)!}.
> $$
> Integrating term by term gives
> $$
> \int_0^1\sin(x^2)dx
> =\sum_{n=0}^{\infty}
> \frac{(-1)^n}{(4n+3)(2n+1)!}.
> $$
> Keeping two terms,
> $$
> \frac13-\frac1{7\cdot3!}
> =\boxed{\frac{13}{42}\approx0.30952}.
> $$
> The next term is $1/(11\cdot5!)=1/1320$, so the alternating-series error is at most this amount; the true value is approximately $0.31027$.
> <!-- bilingual-en:end -->

> [!warning] 易错点
> 逐项运算只在收敛半径内部自动安全；端点必须重新检查，不能照搬原级数结论。
> <!-- bilingual-en:start -->
> Term-by-term operations are automatically valid only inside the radius of convergence. Endpoints must be tested afresh; conclusions for the original series cannot simply be copied to them.
> <!-- bilingual-en:end -->

> [!question]- 三道自检题与答案
> 1. 幂级数求导后收敛半径怎样？  
> 2. 为什么积分后必须加常数？  
> 3. \(\arctan(5x)\) 的线性项是什么？
>
> 答：半径不变但端点可能改变；不同原函数相差常数；线性项为 \(5x\)。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** What happens to the radius of convergence when a power series is differentiated?<br>
> **2.** Why must an arbitrary constant be added after termwise integration?<br>
> **3.** What is the linear term of $\arctan(5x)$?<br>
> Answer: The radius is unchanged, although endpoint behaviour may change; antiderivatives differ by a constant; the linear term is $5x$.
> <!-- bilingual-en:end -->

## Session 101：Conclusion

### 微积分的闭环
<!-- bilingual-en:start -->
*The calculus loop*
<!-- bilingual-en:end -->

1. 导数把函数局部线性化；
2. 积分把局部贡献累计起来；
3. FTC 说明二者互逆；
4. Taylor 级数把局部导数信息组织成全阶多项式近似；
5. 极限决定这些无限过程是否真正收敛。
<!-- bilingual-en:start -->

&nbsp;
**1.** Derivatives linearize functions locally;<br>
**2.** Integrals accumulate local contributions;<br>
**3.** The FTC shows that differentiation and integration are inverse operations;<br>
**4.** Taylor series organize local derivative information into polynomial approximations of all orders;<br>
**5.** Limits determine whether these infinite processes actually converge.<br>
<!-- bilingual-en:end -->

### 何时选哪件工具
<!-- bilingual-en:start -->
*Choosing the right tool*
<!-- bilingual-en:end -->

- \(0/0\)、\(\infty/\infty\)：先化简，再考虑 L’Hôpital；
- 无限区间或奇点积分：写极限并做比较；
- 无限求和：研究部分和，而非只看单项；
- 函数近似：写 Taylor 多项式并给余项界；
- 没有初等原函数：定积分、数值法或幂级数仍然可用。
<!-- bilingual-en:start -->
- \(0/0\), \(\infty/\infty\): simplify before considering L’Hôpital;
- Infinite interval or singular integral: write the relevant limits and use comparison;
- Infinite sum: study partial sums rather than individual terms;
- Function approximation: write a Taylor polynomial and bound the remainder;
- No elementary antiderivative: definite integration, numerical methods, or power series may still work.
<!-- bilingual-en:end -->

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
> <!-- bilingual-en:start -->
> **(a) Sketch the graph.** From
> $$
> \sinh x=\frac{e^x-e^{-x}}2
> $$
> we get $\sinh(-x)=-\sinh x$, so the function is odd. Also,
> $$
> (\sinh x)'=\cosh x=\frac{e^x+e^{-x}}2>0,
> $$
> so it has no critical points and is strictly increasing everywhere. Its second derivative is $\sinh x$, which vanishes and changes sign only at $x=0$, making the origin an inflection point. The limits at the two ends are $\pm\infty$.
>
> **(b) Define the inverse.** Because $\sinh$ is strictly increasing with range $\mathbb R$,
> \[
> y=\operatorname{arsinh}x
> \iff x=\sinh y
> \]
> defines a unique value for every real $x$. Its graph is the reflection of $y=\sinh x$ across $y=x$.
>
> **(c) Differentiate.** Implicit differentiation gives
> $$
> \cosh y\,\frac{dy}{dx}=1.
> $$
> Using $\cosh^2y-\sinh^2y=1$, $\cosh y>0$, and $\sinh y=x$,
> $$
> \boxed{
> \frac d{dx}\operatorname{arsinh}x
> =\frac1{\sqrt{1+x^2}}
> }.
> $$
>
> **(d) Integrate.** Let $x=au$, with $a>0$:
> $$
> \int\frac{dx}{\sqrt{a^2+x^2}}
> =\int\frac{du}{\sqrt{1+u^2}}
> =\boxed{\operatorname{arsinh}(x/a)+C}.
> $$
> The equivalent logarithmic form is
> $$
> \ln\!\left|x+\sqrt{x^2+a^2}\right|+C.
> $$
> <!-- bilingual-en:end -->

> [!question]- 三道自检题与答案
> 1. \(\sinh x\) 为什么可逆？  
> 2. \(\operatorname{arsinh}x\) 的定义域？  
> 3. 上述积分中 \(a\) 的符号为何要说明？
>
> 答：导数 \(\cosh x>0\)，所以严格递增；全体实数；提出 \(\sqrt{a^2}=|a|\) 时符号会影响化简。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** Why is $\sinh x$ invertible?<br>
> **2.** What is the domain of $\operatorname{arsinh}x$?<br>
> **3.** Why must the sign of $a$ be stated in the integral above?<br>
>
> Answer: Its derivative $\cosh x>0$, so it is strictly increasing; all real numbers; the sign affects simplification because $\sqrt{a^2}=|a|$.
> <!-- bilingual-en:end -->

## 全章易错点总表
<!-- bilingual-en:start -->
*Chapter-wide checklist of common mistakes*
<!-- bilingual-en:end -->

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
<!-- bilingual-en:start -->
| Situation | Common mistake | Correct check |
|---|---|---|
| L’Hôpital’s rule | Differentiating when the form is not indeterminate | Substitute again before every application |
| Indeterminate powers | Treating $0^0$ as automatically equal to $1$ | Take logarithms and study the resulting product |
| Improper integrals | Substituting $\infty$ as though it were an endpoint | Begin with a finite-endpoint limit |
| Interior singularities | Cancelling divergences from the two sides | Require the two one-sided integrals to converge separately |
| Infinite series | Inferring convergence solely from $a_n\to0$ | Study partial sums or apply a convergence test |
| Comparison tests | Reversing a one-way implication | Write the size ordering and verify the benchmark series |
| Power series | Forgetting the endpoints | Treat the interior, exterior, and endpoints separately |
| Taylor approximation | Writing a polynomial without an error estimate | State the interval of validity and bound the remainder |
<!-- bilingual-en:end -->

## 本章总结
<!-- bilingual-en:start -->
*Chapter Summary*
<!-- bilingual-en:end -->

1. L’Hôpital 是由 Cauchy MVT 支撑的条件性定理；
2. 反常积分和无穷级数都通过有限对象的极限定义；
3. 比较判别把未知对象与 \(p\)-积分、\(p\)-级数等基准比较；
4. 幂级数在收敛区间内像无限多项式；
5. Taylor 系数由导数唯一决定，但等于原函数还需余项趋零；
6. 任何“无穷运算”都必须明确极限对象、收敛条件与误差。
<!-- bilingual-en:start -->

&nbsp;
**1.** L’Hôpital's rule is a conditional theorem grounded in the Cauchy MVT.<br>
**2.** Improper integrals and infinite series are both defined as limits of finite objects.<br>
**3.** Comparison tests relate an unknown object to benchmarks such as $p$-integrals and $p$-series.<br>
**4.** Within its interval of convergence, a power series behaves like an infinite polynomial.<br>
**5.** Taylor coefficients are uniquely determined by the derivatives, but equality with the underlying function still requires the remainder to tend to zero.<br>
**6.** Every calculation involving infinity must specify what finite object is tending to a limit, the conditions for convergence, and the relevant error.<br>
<!-- bilingual-en:end -->

> [!tip] 一遍读懂后的最低验收
> 能说明 L’Hôpital 的证明骨架，判断 \(p\)-积分与 \(p\)-级数，解释调和级数为何发散，求幂级数半径和端点，并用 Taylor 余项给出可验证的近似误差。
> <!-- bilingual-en:start -->
> At minimum, you should be able to explain the proof strategy behind L’Hôpital's rule, determine convergence of $p$-integrals and $p$-series, explain why the harmonic series diverges, find the radius and endpoint behavior of a power series, and use a Taylor remainder to produce a verifiable error bound.
> <!-- bilingual-en:end -->

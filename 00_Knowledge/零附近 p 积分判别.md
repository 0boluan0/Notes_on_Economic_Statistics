---
aliases:
  - p-integral near zero
  - 端点 p 积分判别
student_os: knowledge-atom
atom_id: CALC-IMP-005
atom_set: improper-integrals
atom_type: theorem
status: source-checked
requires:
  - "[[反常积分]]"
part_of:
  - "[[反常积分.canvas]]"
contrasts_with:
  - "[[无穷远 p 积分判别]]"
---

<!-- bilingual-en:start -->
# The p-integral test near zero
<!-- bilingual-en:end -->
*零附近 p 积分判别*

<!-- bilingual-en:start -->
A curve may become infinitely tall near an endpoint and still enclose finite area there. For example, $1/\sqrt x$ grows without bound as $x\to0^+$, yet the area beneath it from zero to one is $2$. The **p-integral test near zero** tells us which powers have this behaviour. Write the curve as $x^{-p}=1/x^p$, where $p$ is a fixed real exponent and the input satisfies $x>0$.
<!-- bilingual-en:end -->

曲线在端点附近即使无限升高，那一段面积仍可能有限。例如 $1/\sqrt x$ 在 $x\to0^+$ 时无界增长，从零到一的面积却是 $2$。**零附近 p 积分判别**说明哪些幂函数有这种表现。把曲线写成 $x^{-p}=1/x^p$，其中 $p$ 是固定实数指数，输入满足 $x>0$。

<!-- bilingual-en:start -->
First leave out the tiny interval next to zero: integrate from $\varepsilon>0$ to a fixed finite endpoint $b>0$, with $\varepsilon<b$. The [[定积分|definite integral]] on $[\varepsilon,b]$ is ordinary because the function is continuous there. Then let $\varepsilon\to0^+$, meaning that the excluded gap shrinks through positive lengths. The [[反常积分|improper-limit definition]] asks whether these accumulated values approach one finite real number:
<!-- bilingual-en:end -->

先把紧贴零的小区间暂时排除：从 $\varepsilon>0$ 积分到固定的有限端点 $b>0$，并取 $\varepsilon<b$。函数在 $[\varepsilon,b]$ 连续，所以这里是普通[[定积分]]。再令 $\varepsilon\to0^+$，也就是让被排除的间隙从正长度缩小到零。[[反常积分|反常极限定义]]问的是，这些累积值是否趋近同一个有限实数：

$$J_p(b):=\lim_{\varepsilon\to0^+}\int_\varepsilon^b x^{-p}\,dx,\qquad b>0,\ p\in\mathbb R.$$

> [!abstract] The endpoint threshold · 零端点的临界指数
> <!-- bilingual-en:start -->
> This limit is finite exactly when $p<1$. Its value is then $b^{1-p}/(1-p)$. When $p\ge1$, the truncated integrals tend to $+\infty$.
> <!-- bilingual-en:end -->
>
> 当且仅当 $p<1$ 时，上述极限有限，值为 $b^{1-p}/(1-p)$。当 $p\ge1$，截断积分趋于 $+\infty$。
>
> $$\int_0^b x^{-p}\,dx=
> \begin{cases}
> \dfrac{b^{1-p}}{1-p},&p<1,\\[4pt]
> +\infty\quad\text{(divergent)},&p\ge1.
> \end{cases}$$

<!-- bilingual-en:start -->
## Calculate before approaching the endpoint
<!-- bilingual-en:end -->
*先计算有限区间，再逼近端点*

<!-- bilingual-en:start -->
For $p\ne1$, the [[实数幂求导|power derivative]] shows that $x^{1-p}/(1-p)$ is an [[原函数|antiderivative]] of $x^{-p}$ on $x>0$. The finite-interval [[Newton-Leibniz公式|Fundamental Theorem of Calculus]] therefore gives
<!-- bilingual-en:end -->

若 $p\ne1$，由[[实数幂求导]]可知，$x^{1-p}/(1-p)$ 是 $x^{-p}$ 在 $x>0$ 上的[[原函数]]。对有限区间使用 [[Newton-Leibniz公式]]，得到

$$\int_\varepsilon^b x^{-p}\,dx=\frac{b^{1-p}-\varepsilon^{1-p}}{1-p}.$$

<!-- bilingual-en:start -->
If $p<1$, then $1-p>0$ and $\varepsilon^{1-p}\to0$, yielding the finite value in the result. If $p>1$, the negative power $\varepsilon^{1-p}$ tends to $+\infty$. Rewriting the expression as $(\varepsilon^{1-p}-b^{1-p})/(p-1)$ shows that the integral tends to positive infinity.
<!-- bilingual-en:end -->

若 $p<1$，则 $1-p>0$，$\varepsilon^{1-p}\to0$，得到上面的有限值。若 $p>1$，负次幂 $\varepsilon^{1-p}$ 趋于 $+\infty$；把上式写成 $(\varepsilon^{1-p}-b^{1-p})/(p-1)$，就能直接看出积分趋于正无穷。

<!-- bilingual-en:start -->
At $p=1$, use the [[自然对数|natural logarithm]] separately; the power formula divides by zero:
<!-- bilingual-en:end -->

$p=1$ 时必须单独使用[[自然对数]]，不能代入分母为零的幂公式：

$$\int_\varepsilon^b\frac{dx}{x}=\ln b-\ln\varepsilon=\ln(b/\varepsilon)\longrightarrow+\infty.$$

> [!example] An unbounded height with finite area · 高度无界，面积有限
> $$\int_\varepsilon^1 x^{-1/2}\,dx=2(1-\sqrt\varepsilon)\longrightarrow2,$$
> $$\int_\varepsilon^1 x^{-2}\,dx=\frac1\varepsilon-1\longrightarrow+\infty.$$
>
> <!-- bilingual-en:start -->
> In the first case, the area still excluded between zero and $\varepsilon$ is $2\sqrt\varepsilon$, which tends to zero despite the unbounded curve. The same calculation gives $\int_0^1 x^{-3/4}\,dx=4$. A blow-up alone therefore does not decide convergence; its strength matters.
> <!-- bilingual-en:end -->
>
> 第一个例子中，零到 $\varepsilon$ 之间尚未计入的面积为 $2\sqrt\varepsilon$，尽管曲线无界，这个面积仍趋于零。同理，$\int_0^1x^{-3/4}\,dx=4$。所以，仅看见曲线无限升高还不能判断发散，需要看增长强度。

<!-- bilingual-en:start -->
## What if the exponent is zero or negative?
<!-- bilingual-en:end -->
*零指数和负指数并不产生无界奇点*

<!-- bilingual-en:start -->
The formula covers all real $p$, but only $p>0$ makes $x^{-p}$ blow up near zero. For $p=0$, the function is the constant $1$ on $x>0$ and extends continuously by setting its value at zero to $1$. For $p<0$, the exponent $-p$ is positive, and setting the endpoint value to $0$ gives a continuous function on $[0,b]$. In these cases the displayed limit agrees with the ordinary definite integral of that extension; there is no unbounded endpoint singularity. For example, $p=-1$ gives $\int_0^b x\,dx=b^2/2$.
<!-- bilingual-en:end -->

公式覆盖所有实数 $p$，但只有 $p>0$ 才会使 $x^{-p}$ 在零附近无界。若 $p=0$，函数在 $x>0$ 恒等于 $1$，把零点值设为 $1$ 就连续延拓了它。若 $p<0$，指数 $-p$ 为正，把零点值设为 $0$ 后，就在 $[0,b]$ 上连续。这两种情形中的截断极限，等于连续延拓后的普通定积分，并没有无界端点奇点。例如 $p=-1$ 时得到 $\int_0^b x\,dx=b^2/2$。

<!-- bilingual-en:start -->
## Why the two endpoints have opposite conditions
<!-- bilingual-en:end -->
## 为什么两端条件相反

<!-- bilingual-en:start -->
For $0<x<1$, increasing $p$ increases $x^{-p}$, so accumulation near zero becomes more severe. For $x>1$, increasing $p$ decreases $x^{-p}$, so the distant tail decays faster. Thus $p=3/4$ gives a convergent integral from zero to one, but a divergent integral from one to infinity under [[无穷远 p 积分判别|the tail test]]. The number $p=1$ is the dividing line at both ends, with the acceptable inequality reversed.
<!-- bilingual-en:end -->

当 $0<x<1$，增大 $p$ 会增大 $x^{-p}$，零附近的累积更严重；当 $x>1$，增大 $p$ 却会减小 $x^{-p}$，远端衰减更快。因此 $p=3/4$ 在零到一收敛，但按[[无穷远 p 积分判别]]，在一到无穷远发散。两端都以 $p=1$ 为分界，只是收敛所需的不等号方向相反。

<!-- bilingual-en:start -->
Changing the finite upper endpoint from $1$ to any $b>0$ changes the value but not this zero-endpoint threshold: between any two positive endpoints, the integrand is continuous and contributes a finite amount. For a different function, this reasoning requires checking the added finite interval for integrability and other bad points; that is the scope of [[反常积分敛散的局部性|locality of convergence]]. A translated endpoint behaves similarly: $u=x-2$ in [[定积分换元|substitution]] shows that $\int_2^3(x-2)^{-3/4}\,dx=4$ by approaching the singularity at the left endpoint $2$ from the right, within the domain.
<!-- bilingual-en:end -->

把有限上端点从 $1$ 改成任意 $b>0$，会改变数值，却不改变零端点的敛散条件：任意两个正端点之间，被积函数连续，只贡献有限量。对其他函数，必须先检查新增的有限区间是否可积、有没有其他坏点；这才是[[反常积分敛散的局部性]]的适用范围。平移后的端点也可同样处理：通过[[定积分换元]] $u=x-2$，从定义域内右侧逼近奇点 $2$，可得 $\int_2^3(x-2)^{-3/4}\,dx=4$。

> [!warning] An integral can have two separate bad endpoints · 两个问题端点分别检查
> <!-- bilingual-en:start -->
> For $\int_0^\infty x^{-p}\,dx$, split at $1$. Convergence near zero requires $p<1$, while convergence at infinity requires $p>1$. No real $p$ satisfies both, so this whole integral never converges. A test for one endpoint cannot replace the test at the other.
> <!-- bilingual-en:end -->
>
> 对 $\int_0^\infty x^{-p}\,dx$，应在 $1$ 处分开检查：零附近要求 $p<1$，无穷远要求 $p>1$，不存在同时满足两者的实数 $p$，所以整个积分不收敛。检验一个端点，不能代替对另一个端点的检验。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/05_Infinite_Series_and_Improper_Integrals/Ses93a_Lecture_Notes.pdf#page=1|MIT 18.01SC Session 93a, p.1]]: the endpoint examples $1/\sqrt x$, $1/x$ and $1/x^2$. Each is expressed here through its finite truncated integral.
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/05_Infinite_Series_and_Improper_Integrals/Ses93c_Lecture_Notes.pdf#page=1|Session 93c, p.1]]: opposite orderings of the power curves at zero and at infinity.
- [[01_Math/01_calculus/05_Infinite_Series_and_Improper_Integrals.md#有限端点奇点|Permanent course endpoint model]]: the general zero-endpoint criterion and the separate treatment of bad points.
- [OpenStax Calculus 2 §3.7](https://openstax.org/books/calculus-volume-2/pages/3-7-improper-integrals), finite-endpoint definition and Example 3.52: one-sided truncated integrals and a convergent inverse-square-root singularity. The full real-$p$ formula and continuous extensions for $p\le0$ follow from the direct calculation above.
<!-- bilingual-en:end -->

- MIT 93a 第1页支持三种典型端点例子；正文均以有限截断积分表达。
- MIT 93c 第1页支持幂函数在零与无穷远的相反大小顺序。
- 永久课程的有限端点模型保留一般判别及逐坏点处理；OpenStax §3.7 的有限端点定义和例3.52支持单侧截断与可积平方根奇点。全体实数 $p$ 的公式及 $p\le0$ 的连续延拓在正文直接核算。

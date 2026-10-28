---
aliases:
  - Exponential Markov method
  - Chernoff method
  - Exponential moment method
  - 指数矩方法
student_os: knowledge-atom
atom_id: PROB-CONC-007
atom_set: probability-concentration
atom_type: method
status: source-checked
requires:
  - "[[Markov不等式]]"
  - "[[矩母函数]]"
part_of:
  - "[[概率不等式与集中界.canvas]]"
related:
  - "[[相互独立]]"
  - "[[两两独立不推出相互独立]]"
  - "[[乘法Chernoff上界]]"
  - "[[Hoeffding不等式]]"
---

<!-- bilingual-en:start -->
# Exponential Markov method
<!-- bilingual-en:end -->

*指数 Markov 法*

<!-- bilingual-en:start -->
The **exponential Markov method** bounds the probability of a large value by first amplifying large values exponentially, then applying [[Markov不等式|Markov's inequality]] to the transformed variable. A tunable parameter controls the amplification. Trying different parameter values gives a family of valid bounds, from which we can choose the smallest available guarantee.
<!-- bilingual-en:end -->

**指数 Markov 法**先用指数变换放大较大取值，再对变换后的变量使用[[Markov不等式]]，从而限制出现大数值的概率。一个可调参数控制放大的程度。改变参数，就得到一族有效的上界，可以从中选择最小的保证。

<!-- bilingual-en:start -->
Markov says that a nonnegative variable $Y$ satisfies $P(Y\ge b)\le E[Y]/b$ for $b>0$. Here we use $Y=e^{\lambda X}$ and $b=e^{\lambda a}$. The average $E[e^{\lambda X}]$ is the [[矩母函数|moment-generating function]] $M_X(\lambda)$: it averages transformed values using the original probabilities.
<!-- bilingual-en:end -->

Markov 说的是：非负变量 $Y$ 在 $b>0$ 时满足 $P(Y\ge b)\le E[Y]/b$。这里把 $Y$ 取为 $e^{\lambda X}$，把 $b$ 取为 $e^{\lambda a}$。平均数 $E[e^{\lambda X}]$ 就是[[矩母函数]] $M_X(\lambda)$，它按原来的概率权重平均变换后的取值。

<!-- bilingual-en:start -->
## Preserve the event, then apply Markov
<!-- bilingual-en:end -->

*保持事件，再应用 Markov*

<!-- bilingual-en:start -->
Let $a$ be any real threshold and choose $\lambda>0$ such that $M_X(\lambda)<\infty$. The function $x\mapsto e^{\lambda x}$ is increasing, so $X\ge a$ and $e^{\lambda X}\ge e^{\lambda a}$ describe exactly the same outcomes. Since the transformed variable is always positive,
<!-- bilingual-en:end -->

令 $a$ 是任意实阈值，选择使 $M_X(\lambda)<\infty$ 的 $\lambda>0$。函数 $x\mapsto e^{\lambda x}$ 递增，因此 $X\ge a$ 与 $e^{\lambda X}\ge e^{\lambda a}$ 描述完全相同的结果。变换后的变量总为正，于是

$$P(X\ge a)
=P(e^{\lambda X}\ge e^{\lambda a})
\le \frac{E[e^{\lambda X}]}{e^{\lambda a}}
=e^{-\lambda a}M_X(\lambda).$$

<!-- bilingual-en:start -->
The original $X$ may be negative; nonnegativity is supplied by the transformation. Increasing $\lambda$ changes both the numerator and the threshold, so larger $\lambda$ is not automatically better. The task is to balance the growth of $M_X(\lambda)$ against the factor $e^{-\lambda a}$.
<!-- bilingual-en:end -->

原变量 $X$ 可以为负；指数变换已经提供了 Markov 所需的非负性。增大 $\lambda$ 会同时改变分子与阈值，所以参数并非越大越好。需要比较 $M_X(\lambda)$ 的增长与因子 $e^{-\lambda a}$ 共同造成的结果。

> [!example] A parameter you can calculate with · 可以亲手算的参数选择
> <!-- bilingual-en:start -->
> Let $X=1$ with probability $1/4$ and $X=0$ otherwise, and consider $P(X\ge1)$. Since $M_X(\lambda)=3/4+e^\lambda/4$, the bound is $1/4+(3/4)e^{-\lambda}$. Choosing $\lambda=\ln3$ gives $1/2$; choosing $\lambda=\ln9$ gives $1/3$. As $\lambda$ tends to infinity, the bound tends to the exact probability $1/4$, although no finite parameter attains it. Ordinary Markov already gives $E[X]/1=1/4$ here, so the exponential route need not improve every problem.
> <!-- bilingual-en:end -->
>
> 令 $X$ 以 $1/4$ 的概率取一，其余情况取零，考虑 $P(X\ge1)$。由于 $M_X(\lambda)=3/4+e^\lambda/4$，上界为 $1/4+(3/4)e^{-\lambda}$。取 $\lambda=\ln3$ 得到 $1/2$；取 $\lambda=\ln9$ 得到 $1/3$。当 $\lambda$ 趋于无穷，上界趋于真实概率 $1/4$，但没有有限参数能取到它。本例普通 Markov 已经给出 $E[X]/1=1/4$，可见指数路线不必对每个问题都带来改进。

<!-- bilingual-en:start -->
In general, let $D_+=\{\lambda>0:M_X(\lambda)<\infty\}$ be the allowed parameter set. If it is nonempty, all its bounds are valid simultaneously, so
<!-- bilingual-en:end -->

一般地，令 $D_+=\{\lambda>0:M_X(\lambda)<\infty\}$ 为可用参数集合。如果它非空，其中所有参数的界同时成立，因此

$$P(X\ge a)\le\min\left\{1,\inf_{\lambda\in D_+}e^{-\lambda a}M_X(\lambda)\right\}.$$

<!-- bilingual-en:start -->
The **infimum** is the greatest number no larger than every candidate bound; it may be approached without being attained, as in the example. We also retain the universal upper bound one. If $D_+$ is empty, this positive-parameter method produces no finite candidate, but $P(X\ge a)\le1$ still holds.
<!-- bilingual-en:end -->

**下确界**是“不大于每个候选界的数”中最大的一个；它可以被不断逼近而不被取到，正如上例。还要保留通用上界 1。若 $D_+$ 为空，正参数的指数方法就不给出有限候选界，但 $P(X\ge a)\le1$ 仍然成立。

<!-- bilingual-en:start -->
## Why independent sums fit this method
<!-- bilingual-en:end -->

*为什么独立变量之和很适合这条路线*

<!-- bilingual-en:start -->
Suppose $S=X_1+\cdots+X_n$. Exponentiating a sum converts it to a product. If the variables are [[相互独立|mutually independent]], their joint probabilities factor, allowing the expectation of this product to factor too. At parameters where the individual exponential moments are finite,
<!-- bilingual-en:end -->

设 $S=X_1+\cdots+X_n$。和的指数会变成指数的乘积。若各变量[[相互独立]]，联合概率可以分解，乘积的期望也就可以分解。在各项指数矩有限的参数处，

$$M_S(\lambda)
=E\!\left[\prod_{i=1}^n e^{\lambda X_i}\right]
=\prod_{i=1}^n M_{X_i}(\lambda).$$

<!-- bilingual-en:start -->
This is the step that uses independence. [[两两独立不推出相互独立|Pairwise independence]] alone generally cannot justify it. Mutual independence is sufficient; a product equality at a particular parameter is not, by itself, proof of independence.
<!-- bilingual-en:end -->

这正是使用独立性的步骤。仅有[[两两独立不推出相互独立|两两独立]]一般不能保证它成立。相互独立是充分条件；某个参数处碰巧出现乘积等式，本身不能反推变量独立。

<!-- bilingual-en:start -->
For independent terms in $[0,1]$, bounding each exponential average leads to the [[乘法Chernoff上界|multiplicative Chernoff bound]]. For independent terms in known bounded intervals, using their interval lengths leads to [[Hoeffding不等式|Hoeffding's inequality]]. These are useful consequences of the method, with their own assumptions and parameter choices.
<!-- bilingual-en:end -->

对于独立的 $[0,1]$ 加数，逐项控制指数平均可以得到[[乘法Chernoff上界]]；对于处于已知有界区间的独立加数，利用区间长度可以得到[[Hoeffding不等式]]。这些是该方法的实用推论，各有自己的条件和参数选择。

> [!note] A lower tail uses the other direction · 左尾要检查另一方向
> <!-- bilingual-en:start -->
> Apply the same argument to $-X$. For $\lambda>0$ with $M_X(-\lambda)<\infty$, it gives $P(X\le a)\le e^{\lambda a}M_X(-\lambda)$. A finite positive-parameter MGF alone does not supply the negative-parameter condition needed here.
> <!-- bilingual-en:end -->
>
> 对 $-X$ 使用同样论证。若 $\lambda>0$ 且 $M_X(-\lambda)<\infty$，就有 $P(X\le a)\le e^{\lambda a}M_X(-\lambda)$。仅有正参数处的有限矩母函数，并不能代替这里所需的负参数条件。

## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf#page=946|MIT Mathematics for Computer Science, §20.5.6, printed pp. 938–940]]: verifies exponentiation, Markov, factorisation using mutual independence, and parameter selection.
- [MIT 18.S096, Ten Lectures and Forty-Two Open Problems in the Mathematics of Data Science, proof of Theorem 4.3, pp. 57–58](https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-of-data-science-fall-2015/5f0f7205d1cf274e80d77345a7edbf2a_MIT18_S096F15_TenLec.pdf#page=57): independently identifies the Markov and independent-product steps, then optimises the parameter. The zero-or-one example and lower-tail formula are direct derivations.
- [MIT 6.436J Lecture 13, §1.1 and Theorem 2, pp. 2 and 5–6](https://ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/1a592ed184fb4c444547f67c9bcdd8ec_MIT6_436JF18_lec13.pdf): verifies finite-domain restrictions and the MGF of an independent sum.
<!-- bilingual-en:end -->

- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf#page=946|MIT Mathematics for Computer Science，第 20.5.6 节，印刷页 938–940]]：核对指数变换、Markov、用相互独立分解乘积，以及参数选择。
- [MIT 18.S096，Ten Lectures and Forty-Two Open Problems in the Mathematics of Data Science，定理 4.3 的证明，第 57–58 页](https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-of-data-science-fall-2015/5f0f7205d1cf274e80d77345a7edbf2a_MIT18_S096F15_TenLec.pdf#page=57)：独立核对 Markov 和乘积分解各自使用的条件，以及后续优化。零一变量例子和左尾公式为直接推导。
- [MIT 6.436J Lecture 13，第 1.1 节和定理 2，第 2、5–6 页](https://ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/1a592ed184fb4c444547f67c9bcdd8ec_MIT6_436JF18_lec13.pdf)：核对有限定义域和独立和的矩母函数。

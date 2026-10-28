---
student_os: knowledge-atom
atom_id: c6276dc5-7afe-4a2a-bea8-866ed6b6adb8
status: source-checked
aliases:
  - "Law of the unconscious statistician"
  - "Expectation of a function of a random variable"
  - "随机变量函数的期望"
---

<!-- bilingual-en:start -->
# LOTUS: expectation of a function
<!-- bilingual-en:end -->

*LOTUS：怎样计算随机变量函数的期望*

<!-- bilingual-en:start -->
Suppose a random number is recorded, then a rule turns it into a score. To find the average score, apply the rule to each possible number and weight the resulting scores by the original probabilities. This is **LOTUS**, short for the *law of the unconscious statistician*. It lets us compute an expectation without first finding the whole distribution of the transformed score.
<!-- bilingual-en:end -->

假设先记录一个随机数，再按一条规则把它变成分数。要算平均分，就对每个可能的数应用这条规则，再用原来的概率给分数加权。这就是 **LOTUS**，全称为 *law of the unconscious statistician*。它让我们不必先求出变换后分数的完整分布，就能计算期望。

> [!example] Keep the probabilities, change the scores · 保留概率，改变要平均的数
> <!-- bilingual-en:start -->
> Let $X$ take $-2$, $0$ and $2$ with probabilities $1/4$, $1/2$ and $1/4$. A scoring rule squares the recorded number: $g(x)=x^2$, so the score is $Y=g(X)$. The scores are $4$, $0$ and $4$. Their [[期望|expected value]], or probability-weighted average, is $E[Y]=4(1/4)+0(1/2)+4(1/4)=2$.
> <!-- bilingual-en:end -->
>
> 设 $X$ 以 $1/4$、$1/2$、$1/4$ 的概率取 $-2$、$0$、$2$。计分规则是平方：$g(x)=x^2$，于是分数为 $Y=g(X)$，对应分数是 $4$、$0$、$4$。这些分数的[[期望]]，也就是按概率加权的平均数，为 $E[Y]=4(1/4)+0(1/2)+4(1/4)=2$。

<!-- bilingual-en:start -->
The two original values $-2$ and $2$ produce the same score. We could first combine their probabilities to obtain $P(Y=4)=1/2$ and $P(Y=0)=1/2$, then average. LOTUS gives the same answer while leaving those original contributions separate. The function need not be one-to-one or monotone.
<!-- bilingual-en:end -->

原取值 $-2$ 和 $2$ 得到相同分数。也可以先合并它们的概率，求出 $P(Y=4)=1/2$、$P(Y=0)=1/2$，再取平均。LOTUS 保留原来的各项贡献，直接得到相同答案；因此，函数不必一一对应，也不必单调。

<!-- bilingual-en:start -->
## The transformed value and its weight
<!-- bilingual-en:end -->

*变换后的取值，以及它的权重*

<!-- bilingual-en:start -->
For a discrete $X$, let $p_X(x)=P(X=x)$ be its [[概率质量函数|probability mass function]]. The rule is
<!-- bilingual-en:end -->

对于离散的 $X$，令 $p_X(x)=P(X=x)$ 为它的[[概率质量函数]]。计算规则是

$$E[g(X)]=\sum_x g(x)p_X(x).$$

<!-- bilingual-en:start -->
If $X$ has a [[概率密度函数|density]] $f_X$, the corresponding weighted sum becomes an integral. The quantity being averaged is $g(x)$, and the probability weight comes from $f_X(x)\,dx$:
<!-- bilingual-en:end -->

如果 $X$ 有[[概率密度函数|密度]] $f_X$，对应的加权求和就变成积分。被平均的量是 $g(x)$，概率权重由 $f_X(x)\,dx$ 提供：

$$E[g(X)]=\int_{-\infty}^{\infty}g(x)f_X(x)\,dx.$$

<!-- bilingual-en:start -->
For example, let $f_X(x)=3x^2$ for $0<x<1$ and zero elsewhere. This is a density because its integral is one. To find $E[X^2]$, keep the density $3x^2$ and multiply it by the score $x^2$. The [[定积分|definite integral]] then adds these weighted scores:
<!-- bilingual-en:end -->

例如，令 $f_X(x)=3x^2$（$0<x<1$），其余位置为零；它的积分为一，因此是合法密度。求 $E[X^2]$ 时，保留密度 $3x^2$，再乘以分数 $x^2$。[[定积分]]把这些加权分数累积起来：

$$E[X^2]=\int_0^1x^2(3x^2)\,dx=\int_0^1 3x^4\,dx=\frac35.$$

> [!warning] The function does not move through the average · 函数不能直接穿过平均
> <!-- bilingual-en:start -->
> In the first example, $E[X]=0$, so $(E[X])^2=0$, whereas $E[X^2]=2$. Averaging the squared values and squaring the average are different operations. LOTUS tells us which values to average; it does not say $E[g(X)]=g(E[X])$. See [[期望与非线性变换不可交换|expectation and nonlinear transformations]].
> <!-- bilingual-en:end -->
>
> 第一个例子中 $E[X]=0$，因此 $(E[X])^2=0$，但 $E[X^2]=2$。先平方再平均，与先平均再平方，是不同运算。LOTUS 说明应该平均哪些数，并不声称 $E[g(X)]=g(E[X])$；详见[[期望与非线性变换不可交换]]。

<!-- bilingual-en:start -->
## Conditions and the general form
<!-- bilingual-en:end -->

*成立条件与一般写法*

<!-- bilingual-en:start -->
Measurability of $g$ ensures that $g(X)$ is a valid [[随机变量|random variable]] in the model. For a finite real answer, require $E[|g(X)|]<\infty$: the average magnitude of the transformed values must be finite. A finite $E[|X|]$ alone does not guarantee this after squaring or another transformation; see [[矩存在性的使用边界|moment existence conditions]].
<!-- bilingual-en:end -->

函数 $g$ 可测，能够保证 $g(X)$ 在当前模型中仍是合法的[[随机变量]]。若要得到有限实数答案，还需要 $E[|g(X)|]<\infty$，即变换后数值大小的平均有限。仅有 $E[|X|]$ 有限，不能保证平方或其他变换后仍满足条件；见[[矩存在性的使用边界]]。

<!-- bilingual-en:start -->
If $g(X)$ is nonnegative, the same rule also allows the value $+\infty$. If positive and negative contributions both have infinite magnitude, their difference does not define an expectation. These are existence questions, not exceptions to changing the space over which we average.
<!-- bilingual-en:end -->

如果 $g(X)$ 非负，同一计算规则也允许结果为 $+\infty$。如果正负贡献的大小分别累积成无穷，相减便不能定义期望。这些问题关乎期望是否存在，而不是更换平均所用空间的规则失效。

> [!note]- A distribution need not have a density · 没有密度时也可以写出规则
> <!-- bilingual-en:start -->
> Writing $F_X$ for the [[累积分布函数|CDF]], the general expression is $E[g(X)]=\int g(x)\,dF_X(x)$. Here $dF_X$ denotes the probability weights supplied by the entire distribution. It includes point masses and distributions without a density. If the full law of $Y$ is needed for other purposes, use [[随机变量的分布变换|distribution transformation]]; LOTUS only computes the requested average.
> <!-- bilingual-en:end -->
>
> 令 $F_X$ 为[[累积分布函数|CDF]]，一般式为 $E[g(X)]=\int g(x)\,dF_X(x)$。其中 $dF_X$ 表示整个分布提供的概率权重，既能包含单点质量，也适用于没有密度的分布。如果为了其他目的还需要 $Y$ 的完整分布，应使用[[随机变量的分布变换]]；LOTUS 只负责计算这里所需的平均数。

## 来源与核验

<!-- bilingual-en:start -->
- [MIT 6.436J Lecture 8, §2.2 Theorem 1 and §2.3 Theorem 2, pp. 4–7](https://ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/a8b70283ce951b43d64f419b0857e0a3_MIT6_436JF18_lec08.pdf): establishes averaging on the original or transformed distribution, the density formula and existence conditions.
- [[01_Math/08_EC400_MathsCamp/2026_Course_Materials/04_PSI/Lectures/Lecture 2 - Statistics I.pdf#page=12|EC400 PSI Lecture 2, slides 12–13 and 42]]: supplies the course notation, discrete and density forms, and the general measurable-function condition. [[04_PSI Lecture 2 - 随机变量、条件分布与独立性|The permanent course note]] retains the full lecture context. Both worked examples here are direct calculations.
<!-- bilingual-en:end -->

- [MIT 6.436J Lecture 8，第 2.2 节定理 1、第 2.3 节定理 2，第 4–7 页](https://ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/a8b70283ce951b43d64f419b0857e0a3_MIT6_436JF18_lec08.pdf)：证明按原分布或变换后的分布平均等价，并支持密度公式与存在条件。
- [[01_Math/08_EC400_MathsCamp/2026_Course_Materials/04_PSI/Lectures/Lecture 2 - Statistics I.pdf#page=12|EC400 PSI Lecture 2，第 12–13、42 张]]：支持课程符号、离散与密度形式、一般可测函数的条件。[[04_PSI Lecture 2 - 随机变量、条件分布与独立性|永久课程笔记]]保留完整课堂语境。本文两个算例按给定分布直接计算。

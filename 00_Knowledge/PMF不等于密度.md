---
aliases:
  - PMF versus PDF
  - Probability mass versus density
  - PMF is not a density
  - 概率质量与概率密度
student_os: knowledge-atom
atom_id: PROB-RV-003
atom_set: random-variables-distributions-moments
atom_type: distinction
status: source-checked
requires:
  - "[[概率质量函数]]"
  - "[[概率密度函数]]"
related:
  - "[[累积分布函数]]"
part_of:
  - "[[随机变量、分布与矩.canvas]]"
---

<!-- bilingual-en:start -->
# Probability mass and probability density
<!-- bilingual-en:end -->
*概率质量与概率密度*

<!-- bilingual-en:start -->
A **probability mass** is an amount of probability assigned to one value. A **probability density** is a height that must be accumulated over a range before it gives probability. Both can describe a random variable's distribution, but their values answer different questions. Reading a density as a point probability is therefore a change of meaning, not just a notation error.
<!-- bilingual-en:end -->

**概率质量**是分配给某个数值的一份概率；**概率密度**则是需要在一个范围上累积后才能得到概率的高度。两者都可以描述随机变量的分布，但它们的函数值回答的是不同问题。把密度读成单点概率，改变了数值的含义，而不仅仅是弄混符号。

<!-- bilingual-en:start -->
The [[概率质量函数|probability mass function (PMF)]] of a discrete variable $X$ gives $p_X(x)=P(X=x)$ directly. A [[概率密度函数|probability density function (PDF)]] $f_X$ instead gives interval probabilities through an integral, the area under its curve. The letters $p$ and $f$ help distinguish the two here; always check a source's definition because naming conventions vary.
<!-- bilingual-en:end -->

离散变量 $X$ 的[[概率质量函数|概率质量函数（PMF）]]直接给出 $p_X(x)=P(X=x)$。[[概率密度函数|概率密度函数（PDF）]] $f_X$ 则通过积分，也就是曲线下的面积，给出区间概率。这里用 $p$ 和 $f$ 区分二者；不同资料的命名习惯可能不同，应以资料中的定义为准。

> [!example] Read what the number refers to · 先看数值指的是什么
> <!-- bilingual-en:start -->
> In the rain example, $R=1$ records rain and $R=0$ records no rain. A [[Bernoulli分布|Bernoulli]] PMF with $p_R(1)=0.4$ means exactly $P(R=1)=0.4$: the point $1$ carries probability $0.4$.
>
> Now let $T$ be a wait uniformly distributed over $[0,0.1]$ seconds. Its density is $f_T(t)=10$ per second inside that interval. Thus $f_T(0.03)=10$, but $P(T=0.03)=0$. A range has positive probability: $P(0.02<T<0.04)=10\times0.02=0.2$.
> <!-- bilingual-en:end -->
>
> 在下雨的例子里，$R=1$ 表示下雨，$R=0$ 表示没下雨。若 [[Bernoulli分布|Bernoulli]] 的 PMF 满足 $p_R(1)=0.4$，就直接表示 $P(R=1)=0.4$：数值 $1$ 这个点承载了 $0.4$ 的概率。
>
> 再令 $T$ 表示在 $[0,0.1]$ 秒上均匀分布的等待时间。它在区间内的密度是每秒 $10$，所以 $f_T(0.03)=10$，但 $P(T=0.03)=0$。一段区间则可以有正概率：$P(0.02<T<0.04)=10\times0.02=0.2$。

<!-- bilingual-en:start -->
## Sum masses; integrate a density
<!-- bilingual-en:end -->
*点质量相加，密度求积分*

<!-- bilingual-en:start -->
For a discrete variable, the possible values can be listed in a finite or countable set $S$ carrying all the probability. To find the probability of an event, add the masses at the included values. With a density, accumulate over the included range instead. For an interval $(a,b]$, the two rules are
<!-- bilingual-en:end -->

对于离散变量，可以用有限或可数集合 $S$ 列出承载全部概率的数值。求一个事件的概率时，把其中各个数值的点质量相加。有密度时，则在所含范围上累积密度。以区间 $(a,b]$ 为例，两种计算规则分别是

$$\text{PMF:}\quad P(a<X\le b)=\sum_{x\in S:\,a<x\le b}p_X(x).$$

$$\text{PDF:}\quad P(a<X\le b)=\int_a^b f_X(x)\,dx.$$

<!-- bilingual-en:start -->
The same difference appears when checking the whole distribution: the masses must sum to one, whereas the density must integrate to one. A PMF entry lies between zero and one because it is already a probability. A density may exceed one; its units are probability per unit of $X$, such as per second. A density height only becomes a probability after the relevant width has been accounted for.
<!-- bilingual-en:end -->

检查整个分布时也是一样：点质量必须相加为一，密度必须积分为一。PMF 的每一项本来就是概率，所以一定在零与一之间；密度则可以大于一，单位是每一单位 $X$ 所对应的概率，例如“每秒”。只有计入相应宽度之后，密度高度才转化为概率。

<!-- bilingual-en:start -->
This also distinguishes two common plots. In a PMF stem plot, each stem's height is the probability at its value. In a density plot, probability is area under the curve. Drawing a wider stem does not change a discrete probability, while a wider interval under a density curve generally includes more probability.
<!-- bilingual-en:end -->

这也区分了两种常见图形：在 PMF 的竖线图里，每根竖线的高度就是对应数值的概率；在密度图里，概率是曲线下的面积。把 PMF 竖线画得更粗不会改变离散概率，而在密度曲线下选取更宽的区间，通常会包含更多概率。

> [!warning] Zero at every fixed point does not mean no possible outcome · 单点概率为零不等于没有结果
> <!-- bilingual-en:start -->
> A variable with a density has $P(X=x)=0$ for each fixed $x$, but its value must still land somewhere. A point has no width; an interval can have positive width and probability. The distinction between a zero-probability event and an empty event is explained in [[零概率不等于不可能|zero probability does not imply impossibility]].
> <!-- bilingual-en:end -->
>
> 有密度的变量对每个预先指定的 $x$ 都满足 $P(X=x)=0$，但最终仍然会取某个值。单点没有宽度，区间却可以有正的宽度和概率。零概率事件与空事件的区别见[[零概率不等于不可能]]。

<!-- bilingual-en:start -->
## Some distributions need a different representation
<!-- bilingual-en:end -->
*有些分布需要更一般的表示*

<!-- bilingual-en:start -->
PMF and PDF are not an exhaustive either-or choice. A [[混合分布|mixed distribution]] can put some probability at a point and spread the remainder continuously. For example, a payment might be exactly zero with probability $0.8$, and otherwise be uniformly distributed between $0$ and $10$. The point mass alone misses the remaining $0.2$; integrating only the continuous part misses the $0.8$ at zero.
<!-- bilingual-en:end -->

PMF 与 PDF 并不是涵盖所有情形的二选一。[[混合分布]]可以把一部分概率放在单点，另一部分连续地铺开。例如，一笔赔付可能以 $0.8$ 的概率恰好为零，其余情况下在 $0$ 与 $10$ 之间均匀分布。只记录零点质量会遗漏剩下的 $0.2$；只积分连续部分又会遗漏零点的 $0.8$。

<!-- bilingual-en:start -->
There are also distributions with no point masses and no density, as discussed in [[连续CDF不保证密度|a continuous CDF need not have a density]]. The [[累积分布函数|CDF]] works in all these cases: $F_X(x)=P(X\le x)$ always records the accumulated probability. Before using a sum or integral, identify what the supplied function represents and whether it describes the whole distribution.
<!-- bilingual-en:end -->

还有一些分布既没有点质量，也没有密度，见[[连续CDF不保证密度]]。[[累积分布函数|CDF]] 能处理这些情形：$F_X(x)=P(X\le x)$ 始终记录到 $x$ 为止的累积概率。因此，使用求和或积分之前，先确定给出的函数代表什么，以及它是否描述了整个分布。

## 来源与核验

<!-- bilingual-en:start -->
- [MIT 18.05 Probability Reading, Class 4 §2.4 and Class 5, Continuous Random Variables, §4.2](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/mit18_05_s22_probability.pdf): supports point masses, density areas, and units. The rain and waiting-time calculations apply these definitions.
- [MIT 6.436J Lecture 4, §§3–4 and Appendix B](https://ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/9797310bed4c7f5b5d40d007783eec8d_MIT6_436JF18_lec04.pdf): supports PMFs, PDFs, mixed distributions, and the singular example outside those two pure cases.
<!-- bilingual-en:end -->

- [MIT 18.05 Probability Reading，Class 4 第 2.4 节与 Class 5 的 Continuous Random Variables 第 4.2 节](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/mit18_05_s22_probability.pdf)：支持点质量、密度面积与单位。雨天和等待时间的计算应用了这些定义。
- [MIT 6.436J Lecture 4，第 3–4 节与附录 B](https://ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/9797310bed4c7f5b5d40d007783eec8d_MIT6_436JF18_lec04.pdf)：支持 PMF、PDF、混合分布，以及不属于这两种纯情形的奇异分布例子。

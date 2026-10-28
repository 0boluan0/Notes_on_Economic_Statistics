---
aliases:
  - Hoeffding's inequality
  - General Hoeffding bound
  - 霍夫丁不等式
student_os: knowledge-atom
atom_id: PROB-CONC-009
atom_set: probability-concentration
atom_type: theorem
status: source-checked
requires:
  - "[[指数Markov法]]"
  - "[[相互独立]]"
part_of:
  - "[[概率不等式与集中界.canvas]]"
related:
  - "[[乘法Chernoff上界]]"
  - "[[样本均值Hoeffding界]]"
---

<!-- bilingual-en:start -->
# Hoeffding's inequality
<!-- bilingual-en:end -->

*Hoeffding 不等式*

<!-- bilingual-en:start -->
**Hoeffding's inequality** bounds how far a sum of independent bounded measurements can wander from its average. You do not need the full distribution of each measurement. It is enough to know a finite interval that always contains it. Shorter intervals mean less room for any one measurement to move, and this information gives an exponential bound on the total deviation.
<!-- bilingual-en:end -->

**Hoeffding 不等式**控制一批独立、有界的测量值相加后，偏离平均值的概率。你不需要知道每项的完整分布，只要知道一个几乎必然包含其取值的有限区间。区间越短，单项能够波动的余地越小；这些信息会转化成总偏差的指数上界。

<!-- bilingual-en:start -->
Write $X_i$ for measurement $i$ and assume the collection is [[相互独立|mutually independent]], with finite deterministic bounds $a_i\le X_i\le b_i$ almost surely. The total is $S=\sum_iX_i$, and $E[S]$ is its [[期望|expectation]]. Thus $S-E[S]$ is the deviation from the total's own average; individual measurements do not have to have mean zero or a common distribution.
<!-- bilingual-en:end -->

用 $X_i$ 表示第 $i$ 项测量值，假设它们[[相互独立]]，并几乎必然落在有限、确定的界限 $a_i\le X_i\le b_i$ 内。总量为 $S=\sum_iX_i$，其[[期望]]为 $E[S]$。所以 $S-E[S]$ 表示总量相对自己平均值的偏差；不要求每项均值为零，也不要求同分布。

<!-- bilingual-en:start -->
## Measure the width of each possible range
<!-- bilingual-en:end -->

*先计算每个取值区间的宽度*

<!-- bilingual-en:start -->
Let $D=\sum_{i=1}^n(b_i-a_i)^2$: square each interval's width, then add. This is a summary of the bounds we know, not the [[方差|variance]] of the sum. For $D>0$ and a positive deviation threshold $t$, the guarantees are
<!-- bilingual-en:end -->

令 $D=\sum_{i=1}^n(b_i-a_i)^2$：将各区间宽度平方，再求和。这是已知界限的概括量，不是总量的[[方差]]。当 $D>0$、偏差阈值 $t>0$ 时，有

> [!important] One direction or both · 单侧与双侧
> $$P(S-E[S]\ge t)\le e^{-2t^2/D},\qquad
> P(E[S]-S\ge t)\le e^{-2t^2/D}.$$
> $$P(|S-E[S]|\ge t)\le\min\{1,\;2e^{-2t^2/D}\}.$$
> <!-- bilingual-en:start -->
> The first line controls an upward or downward error separately. The second covers either direction. Apply the [[并事件上界|union bound]] to the two tails to obtain the factor two, then cap the result at one because every probability is at most one.
> <!-- bilingual-en:end -->
>
> 第一行分别控制向上或向下的偏差，第二行同时覆盖两个方向。对两尾使用[[并事件上界]]就得到系数二，再利用概率不超过一，把结果截在一以内。

> [!example] Three independently varying measurements · 三项独立测量
> <!-- bilingual-en:start -->
> Suppose $X_1\in[0,1]$, $X_2\in[0,2]$ and $X_3\in[-1,1]$, with means $0.5,1,0$ respectively. The widths are $1,2,2$, so $D=1+4+4=9$, and $E[S]=1.5$. A total of at least $4.5$ is an upward deviation of $t=3$:
> <!-- bilingual-en:end -->
>
> 假设 $X_1\in[0,1]$、$X_2\in[0,2]$、$X_3\in[-1,1]$，均值分别为 $0.5,1,0$。区间宽度分别为 $1,2,2$，所以 $D=1+4+4=9$，且 $E[S]=1.5$。总量至少为 $4.5$ 对应向上偏差 $t=3$：
> $$P(S\ge4.5)\le e^{-2\cdot3^2/9}=e^{-2}\approx0.1353.$$
> <!-- bilingual-en:start -->
> In fact $S\le4$, so this particular event is impossible. Hoeffding's positive ceiling remains valid but misses that extra support information. For a feasible threshold $S\ge3.5$, set $t=2$, giving $e^{-8/9}\approx0.4111$. A bound is a guarantee, not a claim about the event's actual frequency.
> <!-- bilingual-en:end -->
>
> 实际上 $S\le4$，所以第一个事件不可能发生。Hoeffding 给出的正上限仍然正确，但没有利用总量支持的额外信息。对可能发生的阈值 $S\ge3.5$，改取 $t=2$，得到 $e^{-8/9}\approx0.4111$。上界是保证，不是在断言事件的实际频率。

<!-- bilingual-en:start -->
## Why interval length determines the constant
<!-- bilingual-en:end -->

*常数为什么由区间宽度决定*

<!-- bilingual-en:start -->
The proof uses [[指数Markov法|exponential Markov]]: first turn a large deviation into a large exponential, then use independence to multiply exponential averages. Bounding each variable's range controls its exponential average. Optimising the positive exponential parameter produces the term $2t^2/D$.
<!-- bilingual-en:end -->

证明使用[[指数Markov法]]：先把大的偏差转成大的指数值，再借独立性将指数平均拆成乘积。各变量的取值区间控制对应的指数平均，最后优化正的指数参数，得到 $2t^2/D$。

<!-- bilingual-en:start -->
For the familiar symmetric version, assume $E[X_i]=0$ and $|X_i|\le a$ with $a>0$. Each interval $[-a,a]$ has width $2a$, so $D=4na^2$ and
<!-- bilingual-en:end -->

对常见的对称版本，假设 $E[X_i]=0$ 且 $|X_i|\le a$，其中 $a>0$。各区间 $[-a,a]$ 的宽度为 $2a$，所以 $D=4na^2$，从而

$$P\!\left(\left|\sum_iX_i\right|\ge t\right)
\le\min\{1,\;2e^{-t^2/(2na^2)}\}.$$

<!-- bilingual-en:start -->
Subtracting a mean shifts an interval without changing its width. A variable in $[0,1]$ remains in an interval of width one after centring; replacing that interval with the larger $[-1,1]$ throws away information and weakens the constant. The [[样本均值Hoeffding界|sample-mean form]] follows by replacing a sum deviation $t$ with $n$ times an average deviation.
<!-- bilingual-en:end -->

减去均值只平移区间，不改变宽度。一个 $[0,1]$ 变量中心化后，仍位于宽度为一的区间内；若改用更大的 $[-1,1]$，就丢掉了信息，使常数变松。把总和的偏差 $t$ 写成平均偏差的 $n$ 倍，就得到[[样本均值Hoeffding界]]。

<!-- bilingual-en:start -->
If $D=0$, all measurements are constant and every positive deviation has probability zero; do not divide by zero. The intervals must be known bounds on the random variables, not merely the smallest and largest observations in a sample. Dependence needs a different theorem. Since Hoeffding ignores the actual variances and distribution shapes, it need not be the tightest available bound.
<!-- bilingual-en:end -->

若 $D=0$，所有测量值都是常数，任何正偏差的概率均为零，应直接处理，不能除以零。区间必须是随机变量的已知取值界限，不能只拿样本中观察到的最小值和最大值代替。有依赖关系时需要另外的定理。由于 Hoeffding 不使用实际方差和分布形状，所得界未必是可用界中最紧的。

## 来源与核验

<!-- bilingual-en:start -->
- [Hoeffding (1963), Theorem 2, printed p. 16](https://www.cs.rpi.edu/academics/courses/spring06/random/hoefding.pdf#page=5), [DOI](https://doi.org/10.1080/01621459.1963.10500830): independent variables in possibly unequal intervals. Its sample-mean threshold is multiplied by $n$ here to give the sum version; reflection and the union bound give the two-sided version.
- [MIT 18.S096, Theorem 4.3 and its proof, pp. 57–58](https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-of-data-science-fall-2015/5f0f7205d1cf274e80d77345a7edbf2a_MIT18_S096F15_TenLec.pdf#page=57): checks the symmetric constant and exponential argument. The numerical example and degenerate cases above are direct substitutions.
<!-- bilingual-en:end -->

- [Hoeffding（1963），定理 2，印刷页 16](https://www.cs.rpi.edu/academics/courses/spring06/random/hoefding.pdf#page=5)，[DOI](https://doi.org/10.1080/01621459.1963.10500830)：核对独立变量在不同有界区间内的结论。正文将原论文样本均值阈值乘以 $n$，换成总和形式；取相反数并用并事件上界得到双侧结论。
- [MIT 18.S096，定理 4.3 及证明，第 57–58 页](https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-of-data-science-fall-2015/5f0f7205d1cf274e80d77345a7edbf2a_MIT18_S096F15_TenLec.pdf#page=57)：交叉核对对称版本的常数及指数论证。数值例子和退化情形由直接代入核验。

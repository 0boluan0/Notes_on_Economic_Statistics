---
aliases:
  - Hoeffding bound for bounded independent sample means
  - Sample-mean Hoeffding inequality
  - 有界样本均值集中界
student_os: knowledge-atom
atom_id: PROB-CONC-010
atom_set: probability-concentration
atom_type: corollary
status: source-checked
requires:
  - "[[Hoeffding不等式]]"
part_of:
  - "[[概率不等式与集中界.canvas]]"
related:
  - "[[尾界含义]]"
---

<!-- bilingual-en:start -->
# Hoeffding bound for a sample mean
<!-- bilingual-en:end -->

*样本均值的 Hoeffding 界*

<!-- bilingual-en:start -->
A sample average varies from one set of observations to another. The **sample-mean Hoeffding bound** gives a way to choose enough independent, bounded observations so that a large error is unlikely. It answers a practical question: how many observations suffice for a chosen error tolerance and failure probability?
<!-- bilingual-en:end -->

换一批观测，算出来的样本平均也会变化。**样本均值 Hoeffding 界**可以帮助我们选择足够多的独立、有界观测，使大误差出现的概率较小。它回答一个实际问题：给定容许误差和失败概率，多少个观测已经足够？

<!-- bilingual-en:start -->
Let $Y_1,\ldots,Y_n$ be [[相互独立|mutually independent]] observations with known finite bounds $a\le Y_i\le b$ almost surely, where $a<b$. Their average is $\bar Y=n^{-1}\sum_iY_i$. By [[期望线性性|linearity of expectation]], its centre is
<!-- bilingual-en:end -->

设 $Y_1,\ldots,Y_n$ 为[[相互独立]]的观测，几乎必然满足已知有限界限 $a\le Y_i\le b$，其中 $a<b$。平均值为 $\bar Y=n^{-1}\sum_iY_i$。根据[[期望线性性|期望的线性性]]，其中心为

$$m=E[\bar Y]=\frac1n\sum_{i=1}^nE[Y_i].$$

<!-- bilingual-en:start -->
If all observations have the same mean $\mu$, then $m=\mu$. Identical distributions are unnecessary. If their means differ, the guarantee concerns their average mean $m$; it does not turn them into observations from one common population.
<!-- bilingual-en:end -->

若各观测均值同为 $\mu$，则 $m=\mu$；不需要同分布。若各均值不同，保证针对的是这些均值的平均 $m$，不能据此把观测当成来自一个共同总体。

> [!important] Error tolerance · 容许误差
> <!-- bilingual-en:start -->
> For a chosen tolerance $\varepsilon>0$,
> <!-- bilingual-en:end -->
>
> 对选定的容许误差 $\varepsilon>0$，
> $$P(|\bar Y-m|\ge\varepsilon)
> \le\min\left\{1,\;2\exp\!\left(-\frac{2n\varepsilon^2}{(b-a)^2}\right)\right\}.$$
> <!-- bilingual-en:start -->
> This controls deviations of at least $\varepsilon$ in either direction. For just $\bar Y-m\ge\varepsilon$, omit the factor two.
> <!-- bilingual-en:end -->
>
> 这里同时控制两个方向上至少为 $\varepsilon$ 的偏差；若只关心 $\bar Y-m\ge\varepsilon$，去掉系数二即可。

<!-- bilingual-en:start -->
## From the total to the average
<!-- bilingual-en:end -->

*从总量偏差换成平均偏差*

<!-- bilingual-en:start -->
An average error of $\varepsilon$ is a total error of $n\varepsilon$. Apply [[Hoeffding不等式|Hoeffding's inequality for a sum]] with $S=\sum_iY_i$, $t=n\varepsilon$, and $D=n(b-a)^2$. The conversion is
<!-- bilingual-en:end -->

平均误差为 $\varepsilon$，等价于总量误差为 $n\varepsilon$。在[[Hoeffding不等式|总和的 Hoeffding 不等式]]中，令 $S=\sum_iY_i$、$t=n\varepsilon$、$D=n(b-a)^2$，就有

$$\frac{2t^2}{D}
=\frac{2n^2\varepsilon^2}{n(b-a)^2}
=\frac{2n\varepsilon^2}{(b-a)^2}.$$

<!-- bilingual-en:start -->
For fixed interval width and tolerance, the **upper bound** decreases exponentially with $n$. The theorem does not say that the actual error probability equals the bound, or decreases monotonically at every sample size.
<!-- bilingual-en:end -->

固定区间宽度和容许误差时，概率的**上界**随 $n$ 指数下降。定理并没有说实际误差概率等于这个界，也不保证它在每个样本量处都单调下降。

<!-- bilingual-en:start -->
## Choose a sufficient sample size
<!-- bilingual-en:end -->

*选出足够的样本量*

<!-- bilingual-en:start -->
To make the two-sided failure probability at most $\alpha$, where $0<\alpha<1$, solve $2e^{-2n\varepsilon^2/(b-a)^2}\le\alpha$. Taking the [[自然对数|natural logarithm]] gives the sufficient integer sample size
<!-- bilingual-en:end -->

要把双侧失败概率控制在 $\alpha$ 以内（$0<\alpha<1$），解不等式 $2e^{-2n\varepsilon^2/(b-a)^2}\le\alpha$，两边取[[自然对数]]，得到充分的整数样本量

$$n\ge\left\lceil\frac{(b-a)^2\ln(2/\alpha)}{2\varepsilon^2}\right\rceil.$$

<!-- bilingual-en:start -->
The ceiling brackets mean round upward to an integer. This is sufficient, not necessarily the smallest possible sample size. Halving the tolerance multiplies this bound before rounding by four; halving the failure probability adds only $\ln2$ to the logarithm.
<!-- bilingual-en:end -->

上取整符号表示向上取到整数。这是充分样本量，不一定是实际所需的最小值。把容许误差减半，取整前的样本量界变成四倍；把失败概率减半，则只在对数项中增加 $\ln2$。

> [!example] Estimating a success probability · 估计成功概率
> <!-- bilingual-en:start -->
> Suppose independent trials each succeed with probability $p$. Record success as one and failure as zero, so $Y_i\in[0,1]$ and $m=p$. To ensure $P(|\bar Y-p|\ge0.05)\le0.05$, use
> <!-- bilingual-en:end -->
>
> 假设各次试验独立，成功概率都为 $p$。成功记一，失败记零，因此 $Y_i\in[0,1]$、$m=p$。为了保证 $P(|\bar Y-p|\ge0.05)\le0.05$，可取
> $$n\ge\left\lceil\frac{\ln40}{2(0.05)^2}\right\rceil=738.$$
> <!-- bilingual-en:start -->
> With $n=738$, the bound is $2e^{-3.69}\approx0.04994$. The guarantee holds for every $p$ and before seeing the outcomes. It does not assign a probability to a fixed unknown $p$ after observing one dataset.
> <!-- bilingual-en:end -->
>
> $n=738$ 时，上界为 $2e^{-3.69}\approx0.04994$。对任意 $p$，这个保证都在观察结果前成立；它不是在观察到一份数据后，给固定但未知的 $p$ 分配概率。

<!-- bilingual-en:start -->
Centring $Y_i\in[0,1]$ gives $Y_i-E[Y_i]\in[-E[Y_i],1-E[Y_i]]$, an interval still of width one. Keeping that width yields $2e^{-2n\varepsilon^2}$. Keeping only the weaker symmetric bound $|Y_i-E[Y_i]|\le1$ yields $2e^{-n\varepsilon^2/2}$ instead. If $a=b$, every observation is the same constant and any positive error has probability zero. Correlated observations or a sample minimum and maximum used as if they were guaranteed bounds do not meet this theorem's assumptions.
<!-- bilingual-en:end -->

对 $Y_i\in[0,1]$ 中心化后，有 $Y_i-E[Y_i]\in[-E[Y_i],1-E[Y_i]]$，区间宽度仍为一。保留这个宽度，得到 $2e^{-2n\varepsilon^2}$；若只保留较弱的对称条件 $|Y_i-E[Y_i]|\le1$，则只能得到 $2e^{-n\varepsilon^2/2}$。若 $a=b$，每个观测均为同一个常数，任何正误差的概率都为零。相关观测，或者把样本中的最小值、最大值当成保证界限，都不满足本定理条件。

## 来源与核验

<!-- bilingual-en:start -->
- [Hoeffding (1963), Theorems 1–2, printed pp. 15–16](https://www.cs.rpi.edu/academics/courses/spring06/random/hoefding.pdf#page=4), [DOI](https://doi.org/10.1080/01621459.1963.10500830): verifies the independent bounded-average exponent and interval rescaling. The two-sided and sample-size forms above follow by a union bound and algebra; the sample-size number was evaluated directly.
- [MIT 18.S096, Theorem 4.3, pp. 57–58](https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-of-data-science-fall-2015/5f0f7205d1cf274e80d77345a7edbf2a_MIT18_S096F15_TenLec.pdf#page=57): checks the weaker symmetric-range constant in the comparison.
<!-- bilingual-en:end -->

- [Hoeffding（1963），定理 1–2，印刷页 15–16](https://www.cs.rpi.edu/academics/courses/spring06/random/hoefding.pdf#page=4)，[DOI](https://doi.org/10.1080/01621459.1963.10500830)：核对独立有界均值的指数与区间缩放；双侧形式和样本量公式由并事件上界及代数推得，数值样本量直接计算。
- [MIT 18.S096，定理 4.3，第 57–58 页](https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-of-data-science-fall-2015/5f0f7205d1cf274e80d77345a7edbf2a_MIT18_S096F15_TenLec.pdf#page=57)：交叉核对比较中较松的对称区间常数。

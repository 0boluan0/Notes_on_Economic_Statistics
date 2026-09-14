---
aliases:
  - "Hoeffding 不等式用各支持区间长度控制加性偏差"
  - Hoeffding's inequality
  - General Hoeffding bound
  - 霍夫丁不等式
student_os: knowledge-atom
atom_id: PROB-CONC-009
atom_set: probability-concentration
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[指数Markov法]]"
  - "[[相互独立]]"
part_of:
  - "[[概率不等式与集中界.canvas]]"
related:
  - "[[乘法Chernoff上界]]"
  - "[[样本均值Hoeffding界]]"
---

# Hoeffding 不等式用各支持区间长度控制加性偏差
<!-- bilingual-en:start -->
*Hoeffding's inequality controls additive deviations through support-interval lengths*
<!-- bilingual-en:end -->

> [!summary] 一般区间形式
> 设 $X_1,\ldots,X_n$ independent，且 $a_i\le X_i\le b_i$ almost surely。令 $S=\sum_iX_i$ 与
> $$V=\sum_{i=1}^n(b_i-a_i)^2.$$
> 若 $V>0$，则对 $t>0$，
> $$P(S-E[S]\ge t)\le\exp\!\left(-\frac{2t^2}{V}\right),$$
> $$P(|S-E[S]|\ge t)\le2\exp\!\left(-\frac{2t^2}{V}\right).$$
> <!-- bilingual-en:start -->
> Let $X_1,\ldots,X_n$ be independent with $a_i\le X_i\le b_i$ almost surely. Define $S=\sum_iX_i$ and $V=\sum_i(b_i-a_i)^2$. If $V>0$, then for $t>0$,
> $$P(S-E[S]\ge t)\le\exp\!\left(-\frac{2t^2}{V}\right),$$
> $$P(|S-E[S]|\ge t)\le2\exp\!\left(-\frac{2t^2}{V}\right).$$
> <!-- bilingual-en:end -->

一侧公式是核心；对 $S-E[S]$ 与其相反数各用一次，再做 union bound，得到前因子 2 的双侧公式。若 $V=0$，所有 $X_i$ 都是常数，$S-E[S]=0$ almost surely，应直接处理而不是除以 0。
<!-- bilingual-en:start -->
The one-sided inequality is the core result. Applying it to both $S-E[S]$ and its negative, then using a union bound, gives the two-sided factor of two. If $V=0$, every $X_i$ is constant and $S-E[S]=0$ almost surely; this deterministic case should be handled directly rather than by dividing by zero.
<!-- bilingual-en:end -->

一般公式不要求各项同分布或均值为 0；均值已由 $S-E[S]$ 自动中心化。它只保留每个支持区间的长度，不利用更精细的 variance 或 distribution shape，因此在这些信息可用时未必最紧。
<!-- bilingual-en:start -->
The general formula requires neither identical distributions nor zero individual means; centring is already accomplished by $S-E[S]$. It retains only the length of each support interval and ignores finer variances or distributional shape, so it need not be the tightest bound when such information is available.
<!-- bilingual-en:end -->

若 $E[X_i]=0$ 且 $|X_i|\le a$，则可取区间 $[-a,a]$，每个长度为 $2a$，从而

$$
P\!\left(\left|\sum_iX_i\right|\ge t\right)
\le2\exp\!\left(-\frac{t^2}{2na^2}\right).
$$

这正是 symmetric bounded version。它的常数来自区间长度 $2a$；不能把这个 exponent 与 $[0,1]$ 的区间长度 1 混用。
<!-- bilingual-en:start -->
If $E[X_i]=0$ and $|X_i|\le a$, use the interval $[-a,a]$, whose length is $2a$. This gives the displayed symmetric bounded version. Its constant comes from interval length $2a$ and must not be mixed with the length-one constant for variables in $[0,1]$.
<!-- bilingual-en:end -->

> [!question]- 自检
> $n$ 个独立、均值为 0 且 $|X_i|\le a$ 的变量中，为什么 denominator 是 $2na^2$，而不是 $na^2/2$？
> <!-- bilingual-en:start -->
> For $n$ independent, mean-zero variables with $|X_i|\le a$, why is the denominator in the symmetric exponent $2na^2$ rather than $na^2/2$?
> <!-- bilingual-en:end -->
>
> **答案：** 一般 Hoeffding 的 $V$ 使用区间长度平方；$[-a,a]$ 的长度为 $2a$，故 $V=4na^2$，而 $2t^2/V=t^2/(2na^2)$。
> <!-- bilingual-en:start -->
> **Answer:** General Hoeffding uses squared interval lengths. The interval $[-a,a]$ has length $2a$, so $V=4na^2$ and $2t^2/V=t^2/(2na^2)$.
> <!-- bilingual-en:end -->

**继续：** 把 $t$ 写成 $n\varepsilon$，可得到 [[样本均值Hoeffding界]]。
<!-- bilingual-en:start -->
**Continue with:** Substituting $t=n\varepsilon$ yields [[样本均值Hoeffding界|the Hoeffding bound for bounded independent sample means]].
<!-- bilingual-en:end -->

## 来源与核验

- [Hoeffding（1963），定理 2](https://doi.org/10.1080/01621459.1963.10500830)：把原论文的样本均值记号换写为随机和后，核对一般区间长度形式中的单侧指数 $2t^2/\sum_i(b_i-a_i)^2$。
- [MIT OCW 18.S096，定理 4.3](https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-of-data-science-fall-2015/5f0f7205d1cf274e80d77345a7edbf2a_MIT18_S096F15_TenLec.pdf)：核对对称、零均值特例，两侧界的系数，以及指数 Markov 证明。
<!-- bilingual-en:start -->
- [Hoeffding (1963), Theorem 2](https://doi.org/10.1080/01621459.1963.10500830) verifies the one-sided general interval-length exponent $2t^2/\sum_i(b_i-a_i)^2$ after translating from the paper's sample-mean notation.
- [MIT OCW 18.S096, Theorem 4.3](https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-of-data-science-fall-2015/5f0f7205d1cf274e80d77345a7edbf2a_MIT18_S096F15_TenLec.pdf) verifies the symmetric mean-zero form, its two-sided factor, and the exponential-Markov proof.
<!-- bilingual-en:end -->

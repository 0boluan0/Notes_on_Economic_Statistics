---
aliases:
  - "有界独立样本均值的 Hoeffding 界按样本量指数收紧"
  - Hoeffding bound for bounded independent sample means
  - Sample-mean Hoeffding inequality
  - 有界样本均值集中界
student_os: knowledge-atom
atom_id: PROB-CONC-010
atom_set: probability-concentration
atom_type: corollary
status: source-checked
mastery_state: unassessed
requires:
  - "[[Hoeffding不等式]]"
part_of:
  - "[[概率不等式与集中界.canvas]]"
related:
  - "[[尾界含义]]"
---

# 有界独立样本均值的 Hoeffding 界按样本量指数收紧
<!-- bilingual-en:start -->
*Hoeffding's bound for the mean of bounded independent variables tightens exponentially with sample size*
<!-- bilingual-en:end -->

> [!summary] 推论
> 若 $Y_1,\ldots,Y_n$ independent、共同满足 $a\le Y_i\le b$ almost surely 且 $b>a$，令 $\bar Y=n^{-1}\sum_iY_i$，则对 $\varepsilon>0$，
> $$
> P(|\bar Y-E[\bar Y]|\ge\varepsilon)
> \le2\exp\!\left(-\frac{2n\varepsilon^2}{(b-a)^2}\right).
> $$
> 不要求 $Y_i$ 同分布；中心是 $E[\bar Y]=n^{-1}\sum_iE[Y_i]$。
> <!-- bilingual-en:start -->
> If $Y_1,\ldots,Y_n$ are independent, all satisfy $a\le Y_i\le b$ almost surely, and $b>a$, let $\bar Y=n^{-1}\sum_iY_i$. For $\varepsilon>0$,
> $$P(|\bar Y-E[\bar Y]|\ge\varepsilon)\le2\exp\!\left(-\frac{2n\varepsilon^2}{(b-a)^2}\right).$$
> Identical distributions are unnecessary; the centre is $E[\bar Y]=n^{-1}\sum_iE[Y_i]$.
> <!-- bilingual-en:end -->

在一般 Hoeffding 中取 $S=\sum_iY_i$、$t=n\varepsilon$，且每个 interval length 都是 $b-a$，便有

$$
\sum_i(b_i-a_i)^2=n(b-a)^2,
$$

代入后得到上述 exponent。固定容许误差 $\varepsilon$ 时，failure probability 随 $n$ 指数下降。若 $a=b$，每个 $Y_i$ 都等于同一常数 almost surely，样本均值没有随机偏差，应直接处理而不是除以 0。
<!-- bilingual-en:start -->
In general Hoeffding, take $S=\sum_iY_i$ and $t=n\varepsilon$. Every interval has length $b-a$, so the sum of squared lengths is $n(b-a)^2$. Substitution gives the stated exponent. For a fixed tolerance $\varepsilon$, the failure probability decreases exponentially in $n$. If $a=b$, every $Y_i$ equals the same constant almost surely and the sample mean has no random deviation; handle that deterministic case directly instead of dividing by zero.
<!-- bilingual-en:end -->

特别地，若 $0\le Y_i\le1$，则

$$
P(|\bar Y-E[\bar Y]|\ge\varepsilon)
\le2e^{-2n\varepsilon^2}.
$$

对 $0<\alpha<1$，为了让右侧不超过 $\alpha$，充分条件是

$$
n\ge\frac{\log(2/\alpha)}{2\varepsilon^2}.
$$

这是 sample-size guarantee，不是 exact sampling distribution。
<!-- bilingual-en:start -->
For $0\le Y_i\le1$, the bound becomes $2e^{-2n\varepsilon^2}$. For $0<\alpha<1$, to make the right-hand side at most $\alpha$, it is sufficient that $n\ge\log(2/\alpha)/(2\varepsilon^2)$. This is a sample-size guarantee, not an exact sampling distribution.
<!-- bilingual-en:end -->

常数最容易在中心化时混淆。若 $X_i=Y_i-E[Y_i]$ 且 $Y_i\in[0,1]$，那么 $X_i$ 的精确支持区间是 $[-E Y_i,1-E Y_i]$，长度仍为 1，所以一般 Hoeffding 给 $2e^{-2n\varepsilon^2}$。若丢掉这条区间信息，只使用较粗的 $|X_i|\le1$ symmetric version，则只能得到合法但更松的 $2e^{-n\varepsilon^2/2}$。两个公式使用的信息不同，不能互换 exponent。
<!-- bilingual-en:start -->
Constants are especially easy to mix after centring. If $X_i=Y_i-E[Y_i]$ with $Y_i\in[0,1]$, then the exact support interval of $X_i$ is $[-EY_i,1-EY_i]$, still of length one, so general Hoeffding gives $2e^{-2n\varepsilon^2}$. If that interval information is discarded and only the coarser fact $|X_i|\le1$ is used, the symmetric version yields the valid but weaker $2e^{-n\varepsilon^2/2}$. The formulas use different information and their exponents are not interchangeable.
<!-- bilingual-en:end -->

> [!question]- 自检
> 对 $0<\alpha<1$，独立 $[0,1]$ 变量的平均要以至少 $1-\alpha$ 的概率落在 $E[\bar Y]\pm\varepsilon$ 内，Hoeffding 给出怎样的充分样本量？
> <!-- bilingual-en:start -->
> For $0<\alpha<1$, what sufficient sample size does Hoeffding give for the mean of independent $[0,1]$ variables to lie within $E[\bar Y]\pm\varepsilon$ with probability at least $1-\alpha$?
> <!-- bilingual-en:end -->
>
> **答案：** $n\ge\log(2/\alpha)/(2\varepsilon^2)$；最后按整数向上取整。
> <!-- bilingual-en:start -->
> **Answer:** $n\ge\log(2/\alpha)/(2\varepsilon^2)$, rounded up to the next integer.
> <!-- bilingual-en:end -->

## 来源与核验

- [Hoeffding（1963），定理 1–2](https://doi.org/10.1080/01621459.1963.10500830)：核对 $[0,1]$ 独立样本均值的指数 $2n\varepsilon^2$ 以及一般区间长度缩放。
- [MIT OCW 18.S096，定理 4.3](https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-of-data-science-fall-2015/5f0f7205d1cf274e80d77345a7edbf2a_MIT18_S096F15_TenLec.pdf)：交叉核对常数边界比较中刻意保留的、更松的对称条件 $|X_i|\le a$ 版本。
<!-- bilingual-en:start -->
- [Hoeffding (1963), Theorems 1–2](https://doi.org/10.1080/01621459.1963.10500830) verify the $[0,1]$ sample-mean exponent $2n\varepsilon^2$ and the general interval-length scaling.
- [MIT OCW 18.S096, Theorem 4.3](https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-of-data-science-fall-2015/5f0f7205d1cf274e80d77345a7edbf2a_MIT18_S096F15_TenLec.pdf) cross-checks the deliberately looser symmetric $|X_i|\le a$ version used in the constant-boundary comparison.
<!-- bilingual-en:end -->

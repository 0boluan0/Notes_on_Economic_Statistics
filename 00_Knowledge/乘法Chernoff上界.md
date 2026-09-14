---
aliases:
  - "独立的零到一有界变量之和满足乘法 Chernoff 上尾界"
  - Multiplicative Chernoff bound
  - Chernoff upper-tail bound for independent bounded sums
  - 乘法 Chernoff 界
student_os: knowledge-atom
atom_id: PROB-CONC-008
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
  - "[[Hoeffding不等式]]"
---

# 独立的零到一有界变量之和满足乘法 Chernoff 上尾界
<!-- bilingual-en:start -->
*A sum of independent variables taking values in $[0,1]$ satisfies a multiplicative Chernoff upper-tail bound*
<!-- bilingual-en:end -->

> [!summary] 定理
> 设 $T_1,\ldots,T_n$ mutually independent，且 $0\le T_i\le1$ almost surely。令 $T=\sum_iT_i$、$\mu=E[T]$。对 $c\ge1$，
> $$
> P(T\ge c\mu)
> \le \exp\!\left[-(c\ln c-c+1)\mu\right].
> $$
> 各项不必同分布，也不必只取 0 与 1。
> <!-- bilingual-en:start -->
> Let $T_1,\ldots,T_n$ be mutually independent with $0\le T_i\le1$ almost surely, and define $T=\sum_iT_i$ and $\mu=E[T]$. For $c\ge1$,
> $$P(T\ge c\mu)\le \exp\!\left[-(c\ln c-c+1)\mu\right].$$
> The terms need not be identically distributed or restricted to the two values zero and one.
> <!-- bilingual-en:end -->

写 $c=1+\delta$，其中 $\delta\ge0$，就得到等价的相对偏差形式

$$
P(T\ge(1+\delta)\mu)
\le
\left(\frac{e^\delta}{(1+\delta)^{1+\delta}}\right)^\mu
=\exp\!\left(-[(1+\delta)\ln(1+\delta)-\delta]\mu\right).
$$

这里 $\delta$ 是“超过均值的比例”，不是绝对差。$c$ 形式与 $\delta$ 形式是同一条界，不能把一边的 threshold 与另一边的 exponent 拼接。
<!-- bilingual-en:start -->
Writing $c=1+\delta$ with $\delta\ge0$ gives the equivalent relative-deviation form displayed above. The parameter $\delta$ is the proportion above the mean, not an absolute difference. The $c$ and $\delta$ forms are the same bound; a threshold from one form must not be combined with an exponent from another.
<!-- bilingual-en:end -->

对 1000 枚独立公平币，$T$ 为正面数、$\mu=500$。事件 $T\ge600$ 对应 $c=1.2$、$\delta=0.2$，所以

$$
P(T\ge600)
\le \exp[-(1.2\ln1.2-0.2)500]
\approx8.34\times10^{-5}.
$$

这个数字是 upper guarantee，不是 binomial exact probability。
<!-- bilingual-en:start -->
For 1,000 independent fair coins, let $T$ be the number of heads, so $\mu=500$. The event $T\ge600$ corresponds to $c=1.2$ and $\delta=0.2$, giving the displayed bound of approximately $8.34\times10^{-5}$. This number is an upper guarantee, not the exact binomial probability.
<!-- bilingual-en:end -->

只有 pairwise independence 时，联合 MGF 一般不能分解，因而不能照搬此结论；unbounded summands 也不满足本版本。$c=1$ 时右侧为 1，只给平凡界。若 $\mu=0$，由 $T_i\ge0$ 可知每项及其和都几乎必然为 0；此时 $P(T\ge c\mu)=P(T\ge0)=1$，右侧也等于 1，相对偏差写法只剩退化的平凡界。左尾需要另写相应结论，不能通过把 $c<1$ 直接塞进这个上尾定理获得。
<!-- bilingual-en:start -->
Under pairwise independence alone, the joint MGF generally does not factorise, so this theorem cannot simply be reused. Unbounded summands also violate this version. At $c=1$, the right-hand side is one and the bound is trivial. If $\mu=0$, non-negativity forces every summand and their sum to equal zero almost surely; then both $P(T\ge c\mu)=P(T\ge0)$ and the bound equal one, so the relative-deviation form is degenerate and trivial. A lower tail requires its own corresponding statement; it cannot be obtained by inserting $c<1$ into this upper-tail theorem.
<!-- bilingual-en:end -->

> [!question]- 自检
> 若阈值是 $1.3\mu$，应取 $c$ 与 $\delta$ 各多少？
> <!-- bilingual-en:start -->
> If the threshold is $1.3\mu$, what are the corresponding values of $c$ and $\delta$?
> <!-- bilingual-en:end -->
>
> **答案：** $c=1.3$、$\delta=0.3$；指数中的函数可写成 $c\ln c-c+1$ 或 $(1+\delta)\ln(1+\delta)-\delta$。
> <!-- bilingual-en:start -->
> **Answer:** $c=1.3$ and $\delta=0.3$. The exponent function may be written as either $c\ln c-c+1$ or $(1+\delta)\ln(1+\delta)-\delta$.
> <!-- bilingual-en:end -->

## 来源与核验

- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf|MIT Mathematics for Computer Science]] 定理 20.5.1 及第 20.5.3、20.5.6 节：核对 $[0,1]$ 支持、相互独立条件、精确的 $c$ 形式、矩母函数证明与 1,000 次抛硬币计算。
- [MIT OCW 18.S096，定理 4.11](https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-of-data-science-fall-2015/5f0f7205d1cf274e80d77345a7edbf2a_MIT18_S096F15_TenLec.pdf)：交叉核对独立 Bernoulli 和的标准 $\delta$ 参数化。
<!-- bilingual-en:start -->
- Theorem 20.5.1 and Sections 20.5.3 and 20.5.6 of [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf|MIT Mathematics for Computer Science]] verify the $[0,1]$ support, mutual-independence condition, exact $c$-form, MGF proof, and 1,000-coin calculation.
- [MIT OCW 18.S096, Theorem 4.11](https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-of-data-science-fall-2015/5f0f7205d1cf274e80d77345a7edbf2a_MIT18_S096F15_TenLec.pdf) cross-checks the standard $\delta$ parameterisation for independent Bernoulli sums.
<!-- bilingual-en:end -->

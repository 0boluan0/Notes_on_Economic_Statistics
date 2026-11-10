---
aliases:
  - Cauchy principal value versus improper integral
  - Cauchy 主值与反常积分
student_os: knowledge-atom
atom_id: CALC-IMP-010
atom_set: improper-integrals
atom_type: distinction
status: source-checked
requires:
  - "[[反常积分逐坏点拆分]]"
part_of:
  - "[[反常积分.canvas]]"
---

<!-- bilingual-en:start -->
# Cauchy principal value and improper integrals
<!-- bilingual-en:end -->
*柯西主值与反常积分*

<!-- bilingual-en:start -->
A **Cauchy principal value**, written $\operatorname{PV}$, is a specified symmetric way of approaching an integral's singularity. It combines the two truncated contributions **before** taking one limit. An ordinary [[反常积分|improper integral]] instead requires each side to converge separately. These procedures can give different answers to the question “Does this integral exist?”
<!-- bilingual-en:end -->

**柯西主值**记作 $\operatorname{PV}$，是按照规定的对称方式逼近奇点：先把两侧截断后的贡献相加，**再**取一个极限。普通[[反常积分]]则要求每一侧各自收敛。因此，对“这个积分是否存在”，两种计算可能给出不同结论。

<!-- bilingual-en:start -->
## A cancellation that depends on the cutoff
<!-- bilingual-en:end -->
*抵消依赖怎样截断*

<!-- bilingual-en:start -->
Take $f(x)=1/x$ on $[-1,1]$ away from zero. Its antiderivative on either side is $\ln|x|$, where $\ln$ is the [[自然对数|natural logarithm]]; the [[对数函数求导|logarithm derivative]] is $1/x$ there. For independent positive $\varepsilon,\delta<1$,
<!-- bilingual-en:end -->

看 $[-1,1]$ 上、除去零点的 $f(x)=1/x$。在每一侧，它的原函数都是 $\ln|x|$，其中 $\ln$ 表示[[自然对数]]；由[[对数函数求导]]，导数在那里等于 $1/x$。取相互独立的正数 $\varepsilon,\delta<1$，得到

$$\int_{-1}^{-\varepsilon}\frac{dx}{x}=\ln\varepsilon\longrightarrow-\infty,
\qquad
\int_{\delta}^{1}\frac{dx}{x}=-\ln\delta\longrightarrow+\infty.$$

<!-- bilingual-en:start -->
Neither side converges to a finite number, so the ordinary improper integral diverges. But if we impose $\delta=\varepsilon$, the two **finite truncated integrals** add to zero for every cutoff. Taking their combined limit gives
<!-- bilingual-en:end -->

两侧都没有有限极限，所以普通反常积分发散。但如果强制 $\delta=\varepsilon$，那么对于每个截断量，两个**有限截断积分**的和都等于零。因此组合后的极限为

$$\operatorname{PV}\int_{-1}^{1}\frac{dx}{x}
=\lim_{\varepsilon\downarrow0}\left(
\int_{-1}^{-\varepsilon}\frac{dx}{x}
+\int_{\varepsilon}^{1}\frac{dx}{x}\right)=0.$$

> [!example] Unequal cutoffs change the result · 不对称截断会改变结果
> <!-- bilingual-en:start -->
> If the right cutoff is instead $\delta=2\varepsilon$, the combined value is $\ln\varepsilon-\ln(2\varepsilon)=-\ln2$. More generally, $\delta=\lambda\varepsilon$ with a fixed $\lambda>0$ gives $-\ln\lambda$. These are limits under other coordinated rules, not the standard symmetric Cauchy principal value. The variation exposes why independent convergence failed.
> <!-- bilingual-en:end -->
>
> 若改用右截断量 $\delta=2\varepsilon$，组合值就是 $\ln\varepsilon-\ln(2\varepsilon)=-\ln2$。更一般地，固定 $\lambda>0$，令 $\delta=\lambda\varepsilon$，便得到 $-\ln\lambda$。它们是其他协调规则下的极限，不是标准对称柯西主值。结果随着规则改变，说明为什么这里没有独立收敛的普通积分。

<!-- bilingual-en:start -->
## The precise symmetric definitions
<!-- bilingual-en:end -->
*对称截断的明确定义*

<!-- bilingual-en:start -->
Let $a<c<b$ be finite, with $c$ the only possible singularity, and assume all displayed truncated integrals are proper. The principal value at $c$ is
<!-- bilingual-en:end -->

设有限数 $a<c<b$ 中，$c$ 是唯一可能的奇点，并假设下式中的截断积分都是普通积分。关于 $c$ 的主值定义为

$$\operatorname{PV}\int_a^b f(x)\,dx
:=\lim_{\varepsilon\downarrow0}
\left(\int_a^{c-\varepsilon}f(x)\,dx+
\int_{c+\varepsilon}^{b}f(x)\,dx\right),$$

<!-- bilingual-en:start -->
when the limit is finite; take $0<\varepsilon<\min(c-a,b-c)$. The removed distances on the two sides are equal, although the outer endpoints need not be symmetric about $c$. If both ordinary one-sided limits exist, the limit law for a sum shows that this principal value exists and equals their sum. The $1/x$ example proves that the converse is false.
<!-- bilingual-en:end -->

其中极限须为有限值，且取 $0<\varepsilon<\min(c-a,b-c)$。要求相等的是两侧被挖去的距离，外端点却不必关于 $c$ 对称。若两个普通单侧极限都存在，由和的极限定律，这个主值也存在且等于两者之和；$1/x$ 的例子说明反向推论不成立。

<!-- bilingual-en:start -->
For a real function integrable on every finite closed interval of the real line, the standard principal value at infinity is
<!-- bilingual-en:end -->

对于在实线上每个有限闭区间都可积的实函数，无穷远处的标准主值为

$$\operatorname{PV}\int_{-\infty}^{\infty}f(x)\,dx
:=\lim_{R\to\infty}\int_{-R}^{R}f(x)\,dx,$$

<!-- bilingual-en:start -->
again if finite. For example, $f(x)=x/(1+x^2)$ has symmetric integrals zero for every $R$, while its two tails diverge with opposite signs: an antiderivative is $\tfrac12\ln(1+x^2)$. Thus the distinction also occurs without any finite singularity. If finite singularities and infinite tails occur together, specify and justify how each is treated; neither simple displayed definition silently handles all of them.
<!-- bilingual-en:end -->

同样要求极限有限。例如 $f(x)=x/(1+x^2)$ 对每个 $R$ 的对称积分都为零，但两侧尾积分符号相反地发散：其原函数是 $\tfrac12\ln(1+x^2)$。所以即使没有有限奇点，两种对象仍然可能不同。有限奇点与无穷尾端同时出现时，要分别说明并验证处理规则，不能让上面任一简单定义自动包办所有问题。

<!-- bilingual-en:start -->
## Different objects answer different questions
<!-- bilingual-en:end -->
## 两种对象服务于不同问题

<!-- bilingual-en:start -->
Principal values are useful in singular-integral constructions, including the Hilbert transform. Their cancellation is part of the definition, so a model using it must specify that convention. If a question asks for an ordinary improper integral, report divergence when one side fails, even if a principal value exists. Write $\operatorname{PV}$ explicitly when that is the intended quantity.
<!-- bilingual-en:end -->

主值在奇异积分构造中有用途，例如 Hilbert 变换。这里的抵消是定义的一部分，因此使用这种对称性的模型必须明确约定。题目若问普通反常积分，某一侧不收敛就应判发散，即使主值存在也一样；目标若是主值，则明确写出 $\operatorname{PV}$。

<!-- bilingual-en:start -->
Nor does a principal value always exist. For $1/x^2$ on $[-1,1]$, symmetric truncation gives $2/\varepsilon-2\to+\infty$, so there is no finite Cauchy principal value. Subtracting a divergent term to assign a finite remainder would be another regularization, not this definition. In all cases, the calculation is performed with finite truncated numbers; it never treats $+\infty+(-\infty)$ as ordinary arithmetic.
<!-- bilingual-en:end -->

主值也不总存在。对 $[-1,1]$ 上的 $1/x^2$，对称截断给出 $2/\varepsilon-2\to+\infty$，因此没有有限柯西主值。如果另外减去发散项、给余项指定有限值，那是别的正则化，并非这里的定义。所有计算都对有限截断后的数进行，绝不把 $+\infty+(-\infty)$ 当成普通算术。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [NIST DLMF §1.4(v), equations1.4.24–1.4.25](https://dlmf.nist.gov/1.4#v): the equal-distance finite-singularity and symmetric-infinity definitions. The $1/x$, unequal-cutoff and $x/(1+x^2)$ examples, the forward implication and the $1/x^2$ failure are calculated directly above. [DLMF §1.14(v)](https://dlmf.nist.gov/1.14#v) documents the Hilbert-transform application.
- [[01_Math/01_calculus/05_Infinite_Series_and_Improper_Integrals.md#区间内部奇点|Permanent interior-singularity discussion]]: the course convention for independent limits. [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/05_Infinite_Series_and_Improper_Integrals/Ses93a_Lecture_Notes.pdf#page=1|MIT18.01SC Session93a, pp.1–2]] supports the warning against crossing singularities; it is not the source for the principal-value definition.
<!-- bilingual-en:end -->

- [NIST DLMF §1.4(v)，公式1.4.24–1.4.25](https://dlmf.nist.gov/1.4#v)：有限奇点等距离截断与无穷远对称截断的定义。$1/x$、不等截断、$x/(1+x^2)$ 例子、正向推论与 $1/x^2$ 的失败均在正文直接计算。[DLMF §1.14(v)](https://dlmf.nist.gov/1.14#v)支持 Hilbert 变换中的应用。
- [[01_Math/01_calculus/05_Infinite_Series_and_Improper_Integrals.md#区间内部奇点|永久课程内部奇点说明]]：课程各侧独立极限的约定；[[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/05_Infinite_Series_and_Improper_Integrals/Ses93a_Lecture_Notes.pdf#page=1|MIT18.01SC Session93a，第1–2页]]支持跨奇点的警示，不作为主值定义的来源。

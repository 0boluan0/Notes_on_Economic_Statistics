---
aliases:
- Robbins bounds for Stirling's formula
student_os: knowledge-atom
atom_id: MCS-COUNT-024
atom_set: mcs-counting
atom_type: theorem
status: source-checked
requires:
- '[[斯特林公式]]'
related:
- '[[渐近等价]]'
part_of:
- '[[组合计数原理.canvas]]'
---

<!-- bilingual-en:start -->
# Robbins’ factorial bounds
<!-- bilingual-en:end -->

Robbins 阶乘界

<!-- bilingual-en:start -->
[[斯特林公式|Stirling’s formula]] gives a relative approximation to the [[排列|factorial]] $n!=1\cdot2\cdots n$. If a calculation needs a guaranteed upper or lower bound at a particular $n$, we also need a finite error estimate. Robbins’ inequalities supply one for every positive integer $n$.
<!-- bilingual-en:end -->

[[斯特林公式]]给[[排列]]中的阶乘 $n!=1\cdot2\cdots n$ 提供相对近似。如果计算需要指定 $n$ 的可靠上界或下界，还需要有限误差估计。Robbins 不等式对每个正整数 $n$ 都提供这样的估计。

<!-- bilingual-en:start -->
Write $A_n=\sqrt{2\pi n}(n/e)^n$, with $e$ the [[自然对数|natural logarithm]]’s base. Define the **logarithmic error** $\varepsilon_n=\ln(n!/A_n)$. Thus $n!=A_ne^{\varepsilon_n}$ is an exact identity; $\varepsilon_n$ is neither $n!-A_n$ nor the relative error itself.
<!-- bilingual-en:end -->

记 $A_n=\sqrt{2\pi n}(n/e)^n$，其中 $e$ 是[[自然对数]]的底。定义**对数误差** $\varepsilon_n=\ln(n!/A_n)$。于是 $n!=A_ne^{\varepsilon_n}$ 是精确恒等式；$\varepsilon_n$ 既不是 $n!-A_n$，也不是相对误差本身。

> [!summary] Strict finite bounds · 严格有限界
> <!-- bilingual-en:start -->
> Robbins' inequalities give explicit strict lower and upper bounds for every positive integer $n$, rather than only an asymptotic equivalence.
> The logarithmic error lies strictly between $1/(12n+1)$ and $1/(12n)$; exponentiation gives the equivalent bounds on the factorial below.
> <!-- bilingual-en:end -->
> 对每个正整数 $n$，存在余项 $\varepsilon_n$ 使
> $$
> n!=\sqrt{2\pi n}\left(\frac ne\right)^n e^{\varepsilon_n},
> \qquad
> \frac1{12n+1}<\varepsilon_n<\frac1{12n}.
> $$
> 等价地，
> $$
> \sqrt{2\pi n}\left(\frac ne\right)^n e^{1/(12n+1)}
> <n!<
> \sqrt{2\pi n}\left(\frac ne\right)^n e^{1/(12n)}.
> $$

<!-- bilingual-en:start -->
The inequalities include $n=1$ and are strict, not merely eventual statements. Although $0!=1$ is defined, this formula is not applied at $n=0$, where $1/(12n)$ is undefined. No extension to noninteger inputs is asserted here.
<!-- bilingual-en:end -->

这些严格不等式包括 $n=1$，并非只在足够大的 $n$ 成立。虽然定义了 $0!=1$，这里的公式不用于 $n=0$，因为 $1/(12n)$ 无定义。本卡也不主张对非整数输入作推广。

<!-- bilingual-en:start -->
## Reading the error at a fixed size
<!-- bilingual-en:end -->

读取指定规模的误差

<!-- bilingual-en:start -->
Stirling’s formula says that $n!/A_n$ tends to $1$, but supplies no particular finite-size interval. Robbins’ bounds place that ratio between $e^{1/(12n+1)}$ and $e^{1/(12n)}$, giving strict finite-size control.
<!-- bilingual-en:end -->

[[斯特林公式]]只说明把 $n!$ 除以主近似后，比值最终趋于 1；它没有为某个给定 $n$ 说明误差多大。Robbins 界把这个比值直接夹在

$$
e^{1/(12n+1)}\quad\text{与}\quad e^{1/(12n)}
$$
之间，因此可以做有限规模的严格误差控制。

> [!example] Bounding 5! · 夹住 5!
> <!-- bilingual-en:start -->
> With $n=5$, $A_5\approx118.01917$ and $5!=120$. The exact lower endpoint is $A_5e^{1/61}$, approximately $119.96985$; the exact upper endpoint is $A_5e^{1/60}$, approximately $120.00264$. The theorem guarantees the inequalities between the exact expressions. The displayed decimals illustrate their sizes and are not a separate rounding certificate.
> <!-- bilingual-en:end -->
> 取 $n=5$，有 $A_5\approx118.01917$、$5!=120$。精确下端点是 $A_5e^{1/61}$，约为 $119.96985$；精确上端点是 $A_5e^{1/60}$，约为 $120.00264$。定理保证精确表达式之间的不等式；这些小数帮助理解大小，并不另作舍入误差认证。

<!-- bilingual-en:start -->
Since $\varepsilon_n>0$, the uncorrected approximation $A_n$ is smaller than $n!$. Be explicit about the denominator when reporting a relative error:
<!-- bilingual-en:end -->

由于 $\varepsilon_n>0$，未修正的近似 $A_n$ 小于 $n!$。报告相对误差时，要明确分母：

$$
\frac{n!-A_n}{A_n}=e^{\varepsilon_n}-1,
\qquad
\frac{n!-A_n}{n!}=1-e^{-\varepsilon_n}.
$$

<!-- bilingual-en:start -->
For example, the error relative to the true factorial satisfies the following bounds. The last step uses $1-e^{-x}<x$ for $x>0$, which follows by integrating $e^{-t}<1$ over $0<t<x$.
<!-- bilingual-en:end -->

例如，以真实阶乘为分母的误差满足下面的界。最后一步使用 $x>0$ 时的 $1-e^{-x}<x$；在 $0<t<x$ 上积分 $e^{-t}<1$ 即可得到它。

$$
1-e^{-1/(12n+1)}
<\frac{n!-A_n}{n!}
<1-e^{-1/(12n)}
<\frac1{12n}.
$$

<!-- bilingual-en:start -->
## Proof of the finite bounds
<!-- bilingual-en:end -->

有限界的证明

<!-- bilingual-en:start -->
The independently proved Stirling formula gives $\varepsilon_n\to0$. We now control each decrease $\varepsilon_n-\varepsilon_{n+1}$ and add those controls. No factorial is replaced by an approximation inside an exact identity.
<!-- bilingual-en:end -->

独立证明的 Stirling 公式给出 $\varepsilon_n\to0$。现在控制每次下降 $\varepsilon_n-\varepsilon_{n+1}$，再将这些控制相加。推导精确恒等式时，不会把阶乘直接换成近似值。

<!-- bilingual-en:start -->
For $n\ge1$, put $t=1/(2n+1)$. Cancellation and the absolutely convergent [[对数函数幂级数|logarithm series]] yield
<!-- bilingual-en:end -->

对 $n\ge1$，令 $t=1/(2n+1)$。约去公共因子，再使用绝对收敛的[[对数函数幂级数]]，得到

$$
d_n:=\varepsilon_n-\varepsilon_{n+1}
=\left(n+\tfrac12\right)\ln\left(1+\frac1n\right)-1
=\sum_{j=1}^{\infty}\frac{t^{2j}}{2j+1}.
$$

<!-- bilingual-en:start -->
For the upper bound, replace every denominator by $3$ and sum the [[几何级数|geometric series]]. For the lower bound, keep only its first positive term. Both comparisons are strict because further positive terms exist, and the later denominators exceed $3$.
<!-- bilingual-en:end -->

求上界时，把所有分母换成 $3$，再求[[几何级数]]的和；求下界时，只保留第一个正项。因为还有后续正项，而且后续分母大于 $3$，两侧比较都严格成立。

$$
\frac1{3(2n+1)^2}<d_n
<\frac1{12n(n+1)}
=\frac1{12n}-\frac1{12(n+1)}.
$$

<!-- bilingual-en:start -->
The lower estimate also exceeds a difference that will cancel in the sum. All denominators are positive, so cross multiplication proves this using $24n-23>0$:
<!-- bilingual-en:end -->

下界也大于一个相加时会抵消的差。所有分母都为正，交叉相乘后由 $24n-23>0$ 即可验证：

$$
\begin{aligned}
\frac1{3(2n+1)^2}
&>\frac1{12n+1}-\frac1{12n+13}
=\frac{12}{(12n+1)(12n+13)},\\
(12n+1)(12n+13)-36(2n+1)^2&=24n-23>0.
\end{aligned}
$$

<!-- bilingual-en:start -->
Add the bounds from $n$ through $N\ge n$. Each side is a [[望远镜级数|telescoping sum]]: adjacent endpoint terms cancel. This gives
<!-- bilingual-en:end -->

将从 $n$ 到 $N\ge n$ 的各项界相加。两侧是[[望远镜级数]]中的抵消结构：相邻端点项互相抵消。得到

$$
\frac1{12n+1}-\frac1{12(N+1)+1}
<\varepsilon_n-\varepsilon_{N+1}
<\frac1{12n}-\frac1{12(N+1)}.
$$

<!-- bilingual-en:start -->
Let $N\to\infty$ and use $\varepsilon_{N+1}\to0$. The final inequalities remain strict: on each side the sum of the positive gaps already contains a fixed positive gap at its first term, indexed by $n$. Thus $1/(12n+1)<\varepsilon_n<1/(12n)$, and exponentiation proves the factorial bounds.
<!-- bilingual-en:end -->

令 $N\to\infty$，并使用 $\varepsilon_{N+1}\to0$。最终不等式仍严格成立：每一侧的正差之和，都已经包含下标 $n$ 处一个固定的正差。因此 $1/(12n+1)<\varepsilon_n<1/(12n)$，再取指数即得阶乘界。

<!-- bilingual-en:start -->
## Relation to the asymptotic formula
<!-- bilingual-en:end -->

与渐近公式的关系

<!-- bilingual-en:start -->
Both logarithmic endpoints tend to zero, so their exponentials tend to $1$. Robbins’ bounds therefore imply Stirling’s asymptotic formula. The asymptotic statement alone does not recover these particular finite bounds. In the proof above, Stirling’s formula was established separately; it supplies the zero limit needed to recover the error from its consecutive differences.
<!-- bilingual-en:end -->

由于两个余项端点都趋于 0，取指数后两侧都趋于 1，所以 Robbins 界立即推出 Stirling 主渐近式。反向则不成立：知道相对误差趋零，不能恢复这两个具体的有限 $n$ 指数界。

<!-- bilingual-en:start -->
A proof of the stronger estimate must establish the quantitative inequalities, not just the ratio limit.
<!-- bilingual-en:end -->

上面的证明使用已经独立证明的 Stirling 公式提供误差的零极限，再由相邻差恢复具体界。证明更强的估计，必须建立定量不等式，不能只使用比值极限。

> [!question]- Choosing the right statement · 自检
> <!-- bilingual-en:start -->
> Why can’t $n!\sim\sqrt{2\pi n}(n/e)^n$ replace Robbins’ bounds when proving a numerical upper bound at one fixed $n$?
>
> **Answer:** Asymptotic equivalence describes a limiting ratio; Robbins’ bounds give explicit inequalities at every positive integer $n$.
> <!-- bilingual-en:end -->
> 为什么 $n!\sim\sqrt{2\pi n}(n/e)^n$ 不能替代 Robbins 界来证明某个固定 $n$ 的数值上界？
>
> **答案：** 渐近等价只描述 $n\to\infty$ 时的比值极限；Robbins 界才给每个正整数 $n$ 都成立的显式上下界。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- Herbert Robbins, [“A Remark on Stirling’s Formula” (1955), original scan](https://dornsife.usc.edu/sergey-lototsky/wp-content/uploads/sites/211/2024/02/Stirling-Robbins.pdf), printed pp. 26–28, PDF pp. 2–4: strict finite bounds, logarithmic difference series, and telescoping estimates. The lower-step comparison above uses the first series term and the explicitly checked identity $24n-23>0$.
- The permanent course note below records the statement and its relation to the main asymptotic formula. The exact logarithmic error and the two relative-error conventions are distinguished above.
<!-- bilingual-en:end -->

- [[01_Math/07-Mathematics for Computer Science/03_Counting#Stirling 主公式|Stirling 主公式与 Robbins 余项]]：核对余项表达及严格上下界。
- Herbert Robbins, [“A Remark on Stirling's Formula” (1955)](https://doi.org/10.2307/2308012)：原始有限余项界。
- 原文印刷页 26–28（扫描 PDF 第 2–4 页）：严格有限界、对数差级数与抵消估计；本卡下界的逐步比较用第一个级数项及已展开的 $24n-23>0$ 验证。

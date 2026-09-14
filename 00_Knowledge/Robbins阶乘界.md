---
aliases:
  - "Robbins 界把每个正整数阶乘的 Stirling 余项夹在 1/(12n+1) 与 1/(12n) 之间"
  - Robbins bounds for Stirling's formula
student_os: knowledge-atom
atom_id: MCS-COUNT-024
atom_set: mcs-counting
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[斯特林公式]]"
related:
  - "[[渐近等价]]"
part_of:
  - "[[组合计数原理.canvas]]"
---

# Robbins 界把每个正整数阶乘的 Stirling 余项夹在 1/(12n+1) 与 1/(12n) 之间
<!-- bilingual-en:start -->
*Robbins' bounds place the Stirling remainder for every positive integer n between 1/(12n+1) and 1/(12n)*
<!-- bilingual-en:end -->

> [!summary] 原子定理
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
> <!-- bilingual-en:start -->
> Robbins' inequalities give explicit strict lower and upper bounds for every positive integer $n$, rather than only an asymptotic equivalence.
> <!-- bilingual-en:end -->

[[斯特林公式]]只说明把 $n!$ 除以主近似后，比值最终趋于 1；它没有为某个给定 $n$ 说明误差多大。Robbins 界把这个比值直接夹在
$$
e^{1/(12n+1)}\quad\text{与}\quad e^{1/(12n)}
$$
之间，因此可以做有限规模的严格误差控制。

由于两个余项端点都趋于 0，取指数后两侧都趋于 1，所以 Robbins 界立即推出 Stirling 主渐近式。反向则不成立：知道相对误差趋零，不能恢复这两个具体的有限 $n$ 指数界。

> [!question]- 自检
> 为什么 $n!\sim\sqrt{2\pi n}(n/e)^n$ 不能替代 Robbins 界来证明某个固定 $n$ 的数值上界？
>
> **答案：** 渐近等价只描述 $n\to\infty$ 时的比值极限；Robbins 界才给每个正整数 $n$ 都成立的显式上下界。

## 来源与核验

- [[01_Math/07-Mathematics for Computer Science/03_Counting#Stirling 主公式|Stirling 主公式与 Robbins 余项]]：核对余项表达及严格上下界。
- Herbert Robbins, [“A Remark on Stirling's Formula” (1955)](https://doi.org/10.2307/2308012)：原始有限余项界。

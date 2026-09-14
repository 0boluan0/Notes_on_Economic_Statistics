---
aliases:
  - "在单次输出方差固定的简单 i.i.d. Monte Carlo 中，标准误缩小到原来的 1/k 需要把独立样本数增至 k² 倍"
  - "In simple iid Monte Carlo with fixed per-draw variance, reducing standard error to one over k of its former value requires k squared times as many independent draws"
  - "Monte Carlo 平方根律"
  - "Monte Carlo square-root law"
student_os: knowledge-atom
atom_id: PROB-MC-024
atom_set: monte-carlo-methods
atom_type: decision-rule
status: source-checked
mastery_state: unassessed
part_of:
  - "[[Monte Carlo.canvas|Monte Carlo]]"
requires:
  - "[[Monte Carlo均值标准误]]"
leads_to:
  - "[[方差缩减]]"
  - "[[方差缩减比较口径]]"
related:
  - "[[路径数不修模型]]"
---

# 在单次输出方差固定的简单 i.i.d. Monte Carlo 中，标准误缩小到原来的 1/k 需要把独立样本数增至 k² 倍
<!-- bilingual-en:start -->
*In simple iid Monte Carlo with fixed per-draw variance, reducing standard error to one over k of its former value requires k squared times as many independent draws*
<!-- bilingual-en:end -->

> [!summary] 精度按平方根变，计算量按平方变
> 在单次输出方差固定的简单 i.i.d. Monte Carlo 中，均值标准误为 $\sigma/\sqrt N$。因此想把标准误缩小 $k$ 倍，需要把 $N$ 放大到 $k^2N$。
> <!-- bilingual-en:start -->
> With fixed per-draw variance, the standard error of a simple iid Monte Carlo mean is $\sigma/\sqrt N$. A factor-$k$ reduction in standard error therefore requires $k^2N$ draws.
> <!-- bilingual-en:end -->

因为

$$
\frac{SE(\widehat\mu_{k^2N})}{SE(\widehat\mu_N)}
=\frac{\sigma/\sqrt{k^2N}}{\sigma/\sqrt N}
=\frac1k,
$$

所以四倍独立路径只把标准误减半；若想把标准误缩小到原来的十分之一，就需要约百倍路径。这是样本数换算，若单路径成本近似不变，才能同时读成计算成本换算。
<!-- bilingual-en:start -->
Four times as many independent paths only halves the standard error, while a tenfold reduction requires about one hundred times as many paths. This translates into the same factor in total computation only when cost per path remains approximately fixed.
<!-- bilingual-en:end -->

这条律只适用于它所依赖的 i.i.d. 均值基准。相关抽样、MCMC、quasi-Monte Carlo、嵌套模拟或改变单路径方差的方法会改变有效样本量、速率或成本结构。它解释了为什么需要 [[方差缩减]]，却不意味着更多路径能修复模型或代码。
<!-- bilingual-en:start -->
The rule belongs to the iid sample-mean baseline. Dependence, MCMC, quasi-Monte Carlo, nested simulation, or a changed per-path variance can alter the effective sample size, rate, or cost structure. It motivates variance reduction but does not make more paths a remedy for model or code error.
<!-- bilingual-en:end -->

> [!question]- 可核验自检
> 若希望把简单 i.i.d. Monte Carlo 的标准误降为原来的三分之一，其他条件不变时大约需要多少倍样本？
>
> **答案：** 约 9 倍，因为样本数按所需精度倍数的平方增长。
> <!-- bilingual-en:start -->
> About nine times as many draws.
> <!-- bilingual-en:end -->

## 来源与核验

- [[Monte Carlo均值标准误]]：本卡的 $k^2$ 换算由 $SE=\sigma/\sqrt N$ 直接导出。
- Art B. Owen, [*Monte Carlo Theory, Methods and Examples*, Chapter 2, §§2.1–2.2](https://artowen.su.domains/mc/Ch-intro.pdf)：核对简单 Monte Carlo 样本均值的 $N^{-1/2}$ 标准误速率与样本量换算。

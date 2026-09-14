---
aliases:
  - "独立同分布且有限方差时，Monte Carlo 样本均值无偏，方差为 σ²/N，标准误为 σ/√N"
  - "For iid finite-variance draws, the Monte Carlo sample mean is unbiased with variance sigma squared over N and standard error sigma over square root N"
  - "Monte Carlo 抽样误差"
  - "Monte Carlo standard error"
student_os: knowledge-atom
atom_id: PROB-MC-002
atom_set: monte-carlo-methods
atom_type: theorem
status: source-checked
mastery_state: unassessed
part_of:
  - "[[Monte Carlo.canvas|Monte Carlo]]"
requires:
  - "[[Monte Carlo估计]]"
  - "[[期望线性性]]"
  - "[[方差的仿射变换]]"
leads_to:
  - "[[Monte Carlo近似区间]]"
  - "[[Monte Carlo平方根律]]"
related:
  - "[[标准误含义]]"
  - "[[矩存在性的使用边界]]"
  - "[[路径数不修模型]]"
---

# 独立同分布且有限方差时，Monte Carlo 样本均值无偏，方差为 σ²/N，标准误为 σ/√N
<!-- bilingual-en:start -->
*For iid finite-variance draws, the Monte Carlo sample mean is unbiased with variance sigma squared over N and standard error sigma over square root N*
<!-- bilingual-en:end -->

> [!summary] 这是独立同分布样本均值的基准，不是所有模拟的通用误差公式
> 若 $Y_i=g(X_i)$ 独立同分布、$E[Y_i]=\mu$ 且 $\operatorname{Var}(Y_i)=\sigma^2<\infty$，则 Monte Carlo 样本均值对 $\mu$ 无偏，标准误按 $N^{-1/2}$ 下降。
> <!-- bilingual-en:start -->
> If $Y_i=g(X_i)$ are iid with mean $\mu$ and finite variance $\sigma^2$, their Monte Carlo sample mean is unbiased for $\mu$ and its standard error falls at the rate $N^{-1/2}$.
> <!-- bilingual-en:end -->

令

$$
\widehat\mu_N=\frac1N\sum_{i=1}^N Y_i.
$$

线性期望与独立性分别给出

$$
E[\widehat\mu_N]=\mu,
\qquad
\operatorname{Var}(\widehat\mu_N)=\frac{\sigma^2}{N},
\qquad
SE(\widehat\mu_N)=\frac{\sigma}{\sqrt N}.
$$

这三个结论是同一个样本均值定理的精确矩结果，不需要 $Y_i$ 服从正态分布。无偏性只针对已指定的抽样分布与函数 $g$；如果模型、目标量或实现写错，样本均值仍可能稳定地收敛到错误对象，见 [[路径数不修模型]]。
<!-- bilingual-en:start -->
These are exact moment results for one sample-mean theorem and do not require normally distributed outputs. Unbiasedness is relative to the stated sampling law and function $g$; a wrong model, estimand, or implementation can still be estimated very precisely.
<!-- bilingual-en:end -->

实际中如何用样本方差替代未知的 $\sigma^2$，并构造大样本区间，由 [[Monte Carlo近似区间]] 单独承载。路径数与精度的平方根换算见 [[Monte Carlo平方根律]]。相关抽样、MCMC、quasi-Monte Carlo、分层抽样和嵌套估计则可能需要不同的方差公式或有效样本量。
<!-- bilingual-en:start -->
Estimating the unknown variance and constructing a large-sample interval are separate inferential steps. Converting a desired precision gain into a path-count requirement is also kept in its own atom. Correlated sampling, MCMC, quasi-Monte Carlo, stratification, and nested estimation may require different variance formulas or effective sample sizes.
<!-- bilingual-en:end -->

> [!question]- 可核验自检
> 若 $Y_i$ 独立同分布且方差为 9，$N=100$ 时样本均值的理论标准误是多少？
>
> **答案：** $\sigma/\sqrt N=3/10=0.3$。
> <!-- bilingual-en:start -->
> It is $3/\sqrt{100}=0.3$.
> <!-- bilingual-en:end -->

## 来源与核验

- Art B. Owen, [*Monte Carlo Theory, Methods and Examples*, Chapter 2, §§2.1–2.2](https://artowen.su.domains/mc/Ch-intro.pdf)：核对 i.i.d. 样本均值的无偏性、$\sigma^2/N$ 方差与 $\sigma/\sqrt N$ 标准误。
- [MIT 18.05 Reading 6b, §5.2](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/mit18_05_s22_class06-prep-b.pdf)：交叉核对 i.i.d. 样本均值的均值与 $\sigma^2/N$ 方差。

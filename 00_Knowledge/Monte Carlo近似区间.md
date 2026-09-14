---
aliases:
  - "在 i.i.d. 且非退化有限方差下，用样本标准差学生化可构造 Monte Carlo 均值的大样本正态近似区间"
  - "Under iid non-degenerate finite variance, studentising by the sample standard deviation yields a large-sample normal interval for a Monte Carlo mean"
  - "Monte Carlo 置信区间"
  - "Monte Carlo normal interval"
student_os: knowledge-atom
atom_id: PROB-MC-023
atom_set: monte-carlo-methods
atom_type: inference-boundary
status: source-checked
mastery_state: unassessed
part_of:
  - "[[Monte Carlo.canvas|Monte Carlo]]"
requires:
  - "[[Monte Carlo均值标准误]]"
  - "[[经典中心极限定理]]"
  - "[[Slutsky定理]]"
related:
  - "[[标准误含义]]"
  - "[[矩存在性的使用边界]]"
---

# 在 i.i.d. 且非退化有限方差下，用样本标准差学生化可构造 Monte Carlo 均值的大样本正态近似区间
<!-- bilingual-en:start -->
*Under iid non-degenerate finite variance, studentising by the sample standard deviation yields a large-sample normal interval for a Monte Carlo mean*
<!-- bilingual-en:end -->

> [!summary] 这是近似推断，不是有限样本的自动保证
> 当 $Y_1,\ldots,Y_N$ 独立同分布、$E[Y_i]=\mu$ 且 $0<\operatorname{Var}(Y_i)=\sigma^2<\infty$ 时，可用样本标准差 $s_N$ 估计未知的 $\sigma$，再以学生化后的正态极限构造大样本区间。
> <!-- bilingual-en:start -->
> For iid outputs with mean $\mu$ and positive finite variance, estimate the unknown $\sigma$ by the sample standard deviation and use the studentised normal limit to form a large-sample interval.
> <!-- bilingual-en:end -->

令

$$
\widehat\mu_N=\frac1N\sum_{i=1}^N Y_i,
\qquad
s_N^2=\frac1{N-1}\sum_{i=1}^N(Y_i-\widehat\mu_N)^2.
$$

经典 i.i.d. 中心极限定理与 [[Slutsky定理]] 给出

$$
\frac{\sqrt N(\widehat\mu_N-\mu)}{s_N}
\Rightarrow N(0,1).
$$

因此，对置信水平 $1-\alpha$，常用近似区间是

$$
\widehat\mu_N
\pm z_{1-\alpha/2}\frac{s_N}{\sqrt N}.
$$

95% 情形下 $z_{0.975}\approx1.96$。例如，若 $N=10{,}000$、$\widehat\mu_N=5$、$s_N=2$，则估计标准误为 $0.02$，近似 95% 区间为 $[4.9608,5.0392]$。
<!-- bilingual-en:start -->
The usual $1-\alpha$ interval is the estimate plus or minus the corresponding normal quantile times $s_N/\sqrt N$. With $N=10{,}000$, estimate $5$, and sample standard deviation $2$, the estimated standard error is $0.02$ and the approximate 95% interval is $[4.9608,5.0392]$.
<!-- bilingual-en:end -->

厚尾、稀有事件、强偏态或大量零值可以让正态近似很慢；方差不存在时，这个有限方差 CLT 区间没有依据。区间只量化给定模型与实现下的抽样不确定性，不覆盖模型、代码或离散化错误。
<!-- bilingual-en:start -->
Heavy tails, rare events, strong skewness, or many zero outputs can make the normal approximation slow. Infinite variance invalidates this finite-variance CLT route. The interval quantifies sampling uncertainty conditional on the model and implementation; it does not cover model, code, or discretisation error.
<!-- bilingual-en:end -->

> [!question]- 可核验自检
> 为什么把 $\sigma$ 换成 $s_N$ 不只是“把不知道的数随便换掉”？
>
> **答案：** 因为有限二阶矩下 $s_N$ 一致估计 $\sigma$，[[Slutsky定理]] 保证这个替换不改变标准化均值的渐近正态极限。
> <!-- bilingual-en:start -->
> With a finite second moment, $s_N$ consistently estimates $\sigma$, and Slutsky's theorem preserves the asymptotic normal limit after substitution.
> <!-- bilingual-en:end -->

## 来源与核验

- Art B. Owen, [*Monte Carlo Theory, Methods and Examples*, Chapter 2, equations 2.14–2.16](https://artowen.su.domains/mc/Ch-intro.pdf)：核对以 $s_N/\sqrt N$ 构造 Monte Carlo 均值的大样本正态近似区间及其近似性。
- [MIT 18.05 Reading 6b, §§5.2–5.3](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/mit18_05_s22_class06-prep-b.pdf)：交叉核对 i.i.d. 样本均值的正态极限、$\sigma/\sqrt N$ 尺度与 1.96 的 95% 正态覆盖率。

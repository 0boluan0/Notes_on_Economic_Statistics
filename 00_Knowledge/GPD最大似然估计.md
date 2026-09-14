---
aliases:
  - "GPD最大似然估计在声明的可行参数空间内最大化超额样本的似然"
  - GPD maximum likelihood estimation
student_os: knowledge-atom
atom_id: RM-EVT-013
atom_type: method
status: source-checked
part_of:
  - "[[极值理论 EVT 与尾部风险.canvas]]"
requires:
  - "[[广义Pareto分布]]"
  - "[[超阈值法]]"
related:
  - "[[POT阈值选择]]"
  - "[[GPD拟合诊断]]"
  - "[[尾部外推不确定性]]"
---

# GPD最大似然估计在声明的可行参数空间内最大化超额样本的似然
<!-- bilingual-en:start -->
*GPD maximum likelihood estimation maximizes the excess-sample likelihood over a declared feasible parameter space.*
<!-- bilingual-en:end -->

固定阈值 $u$ 后，令 $y_i=L_i-u>0$ 为 $k>0$ 个超额，先以独立、同分布的精确 GPD 为工作模型。最大似然估计选择使这些超额的联合密度最大的参数；它不是任取一组使导数为零的数值。
<!-- bilingual-en:start -->
Fix $u$ and let $y_i=L_i-u>0$ be the $k>0$ excesses. Start with an iid exact GPD working model. Maximum likelihood selects parameters maximizing their joint density; an arbitrary root of the score equations is not enough.
<!-- bilingual-en:end -->

在每个观测都处于密度内部的参数区域，
<!-- bilingual-en:start -->
On the parameter region where every observation lies in the interior of the density support,
<!-- bilingual-en:end -->

$$
\ell(\xi,\beta)=-k\log\beta-
\left(1+\frac1\xi\right)\sum_{i=1}^k\log\left(1+\frac{\xi y_i}{\beta}\right),
\quad \xi\ne0,
$$
$$
\beta>0,\qquad 1+\xi y_i/\beta>0\quad\text{for every }i.
$$

$\xi=0$ 应使用指数分布分支 $\ell(0,\beta)=-k\log\beta-\sum_i y_i/\beta$。若预先固定 $\xi=0$，可直接得 $\hat\beta=\bar y$；这是受限指数模型的估计，不是一般两参数 GPD 的答案。
<!-- bilingual-en:start -->
At $\xi=0$, use the exponential branch $\ell(0,\beta)=-k\log\beta-\sum_i y_i/\beta$. If shape is fixed at zero in advance, $\hat\beta=\bar y$. This solves the restricted exponential model, not the general two-parameter GPD fit.
<!-- bilingual-en:end -->

数值拟合要检查支持约束、目标值、收敛和多起点结果，并说明返回的是受限全局解还是局部极大值。负形状时端点依赖参数；若完全不限制形状，端点附近可出现无界似然，不能许诺任意样本都有有限的全局 MLE。边界模型也不由内部的零梯度条件完整描述。例如 $\xi=-0.25,\beta=2$ 给超额端点 $8$，样本若含 $y=9$，该候选就不可行。
<!-- bilingual-en:start -->
Check support, objective values, convergence, and multiple starting points, and state whether the output is a constrained global solution or a local maximum. Negative shape makes the endpoint parameter-dependent. With unrestricted shape, likelihood can be unbounded near an endpoint, so a finite global MLE is not guaranteed for every sample. Interior zero-score equations do not cover all boundary cases. For example, $\xi=-0.25,\beta=2$ implies endpoint $8$; an observed excess of $9$ makes that candidate infeasible.
<!-- bilingual-en:end -->

常规标准误也有条件：对 iid 精确 GPD，$\xi>-1/2$ 是通常的一致性与渐近正态推断区域；到达或越过该边界，不能机械使用逆信息矩阵的常规正态区间。这不是“$\xi\le-1/2$ 就不存在任何估计”的结论。若 GPD 仅是高阈值近似，还须控制阈值与样本量增长所带来的近似偏差；相关超额的标准误也不能沿用 iid 校准。
<!-- bilingual-en:start -->
Conventional standard errors are conditional too. For iid exact GPD data, $\xi>-1/2$ is the usual region for consistency and asymptotic-normal inference. At or below this boundary, ordinary inverse-information normal intervals cannot be applied mechanically. This does not say that no estimator exists when $\xi\le-1/2$. For an approximate GPD tail, threshold and sample-size growth must also control approximation bias; dependent excesses require more than iid standard-error calibration.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [Haugh，Columbia《Extreme Value Theory》，PDF 第 23 页](https://martin-haugh.github.io/files/QRM/EVT_MasterSlides.pdf#page=23)：核对超额对数似然及每个观测的支持约束；指数分支与不可行例为按密度直接推导。
  <!-- bilingual-en:start -->
  PDF p. 23 supplies the excess log-likelihood and observation-wise support constraints. The exponential branch and infeasible example are derived directly from the density.
  <!-- bilingual-en:end -->
- [McNeil–Frey，《Estimation of tail-related risk measures for heteroscedastic financial time series: an extreme value approach》，作者版 PDF 第 7 页](https://statmath.wu.ac.at/~frey/publications/evt-garch.pdf#page=7)：已目视核验 $\xi>-1/2$ 的负号、iid 精确 GPD 假设，以及仅近似 GPD 时额外的阈值渐近要求。
  <!-- bilingual-en:start -->
  Author-version PDF p. 7 was visually checked for the minus sign in $\xi>-1/2$, the iid exact-GPD assumption, and the additional threshold asymptotics needed for an approximation.
  <!-- bilingual-en:end -->
- [Belzile，EVA 2023《Likelihood-based inference》，“Generalized Pareto distribution”](https://lbelzile.github.io/EVA2023-Rtutorial/content/likelihood.html#generalized-pareto-distribution)：核对端点无界似然、受限搜索、局部数值检查与非正规标准误边界；不把零梯度当成所有边界情形的定义。
  <!-- bilingual-en:start -->
  The GPD section supports endpoint likelihood problems, restricted optimization, numerical checks, and nonregular standard errors. Zero score is not treated as a definition covering every boundary case.
  <!-- bilingual-en:end -->

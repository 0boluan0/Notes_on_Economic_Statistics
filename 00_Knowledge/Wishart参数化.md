---
aliases:
  - "Wishart 符号必须用生成式或期望校准尺度矩阵位置"
  - Wishart notation must be calibrated by its generative form or expectation
  - Wishart scale and precision conventions
  - Wishart 参数化约定
student_os: knowledge-atom
atom_id: STAT-WISH-005
atom_set: wishart-sample-covariance
atom_type: convention-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[Wishart分布]]"
part_of:
  - "[[Wishart 分布与样本协方差推断.canvas]]"
related:
  - "[[样本协方差Wishart律]]"
---

# Wishart 符号必须用生成式或期望校准尺度矩阵位置
<!-- bilingual-en:start -->
*Wishart notation must be calibrated by the generative definition or the expectation before use*
<!-- bilingual-en:end -->

> [!summary] 原子约定
> 单看 $W_p(V,\nu)$ 无法判断 $V$ 是 covariance scale（协方差型尺度）还是 precision（精度矩阵，即逆尺度）。使用公式前，至少核对作者给出的生成式或期望：
> - 若 $Z_i\sim N_p(0,V)$ 且 $W=\sum_iZ_iZ_i^T$，则 $E[W]=\nu V$；
> - 若作者把 $\Omega=V^{-1}$ 当作参数，则同一分布满足 $E[W]=\nu\Omega^{-1}$。
> 另外，参数顺序也不统一：$W_p(V,\nu)$ 与 $W_p(\nu,V)$ 都有人使用。
> <!-- bilingual-en:start -->
> The notation $W_p(V,\nu)$ is not self-interpreting. Identify whether the matrix parameter is a covariance scale or a precision by checking the generating Gaussian vectors or the stated expectation.
> <!-- bilingual-en:end -->

在 covariance-scale 约定下，当 $V\succ0$、$\nu>p-1$ 且 $W\succ0$ 时，非奇异 Wishart 密度的相关部分写成
$$
f(W\mid V,\nu)\propto
|V|^{-\nu/2}|W|^{(\nu-p-1)/2}
\exp\!\left[-\frac12\operatorname{tr}(V^{-1}W)\right].
$$
若改用 inverse-scale（也就是生成 Gaussian 的 precision）参数 $\Omega=V^{-1}$，同一密度则写成
$$
f(W\mid \Omega,\nu)\propto
|\Omega|^{\nu/2}|W|^{(\nu-p-1)/2}
\exp\!\left[-\frac12\operatorname{tr}(\Omega W)\right].
$$
两个式子描述的是同一个分布的两种参数表达；把前一本书的 $V$ 原样代入后一本书以 $\Omega$ 为参数的函数，会把协方差结构整体取逆。

最省错的核对顺序是：

1. 找生成式：Gaussian 向量的协方差究竟是哪一个矩阵；
2. 找期望：应当是 $\nu V$ 还是 $\nu V^{-1}$；
3. 再核对符号和函数调用中自由度、矩阵参数的先后顺序；
4. 最后才调用密度、随机数或逆矩公式。

软件参数名也不能替代定义。有的接口要求 covariance-scale，有的明确允许传入其逆矩阵 `precision_matrix`，而 R 与 SciPy 的文档甚至采用相反的书写顺序。应读函数公式或用一维均值核对，而不是只凭位置或变量名猜测。

> [!question]- 自检
> 一篇文章只写 $W\sim W_p(V,\nu)$。为什么还不能立刻断言 $E[W]=\nu V$？
>
> **答案：** $V$ 可能被作者定义为 precision；必须由生成式、密度中的矩阵位置或作者明确写出的期望确认约定。

## 来源与核验

- [[01_Math/04_多元统计分析/04_多元正态分布The Multivariate Normal Distribution.md#1.6. Wishart 分布|多元统计课程 §1.6.1]]：核对本课程采用的 covariance-scale 生成式。
- [NIST Dataplot, Wishart Random Numbers](https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/wishrand.htm)：核对以多元正态协方差矩阵生成 Wishart 随机矩阵的约定。
- [R `stats::rWishart` documentation](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/rWishart.html)：核对 $W_p(\Sigma,\nu)$ 的尺度在前记号、生成式与 $E[W]=\nu\Sigma$。
- [SciPy `scipy.stats.wishart` documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wishart.html)：核对 $W_p(\nu,\Sigma)$ 的自由度在前记号及密度中的 $|\Sigma|^{-\nu/2}$、$\Sigma^{-1}$ 位置。
- [PyTorch `Wishart` documentation](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.wishart.Wishart)：核对同一接口可分别接收 `covariance_matrix` 或 `precision_matrix`，但二者代表互逆输入。

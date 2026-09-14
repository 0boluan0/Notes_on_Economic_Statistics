---
aliases:
  - "已知正定行协方差 Ω 时多响应 GLS 对每个响应使用同一个加权投影"
  - With known positive-definite row covariance Omega, multivariate GLS uses the same weighted projection for every response
  - multivariate GLS with correlated rows
student_os: knowledge-atom
atom_id: STAT-MVR-014
atom_set: multivariate-linear-regression
atom_type: estimation-method
status: source-checked
mastery_state: unassessed
part_of:
  - "[[多元线性回归.canvas|多元线性回归]]"
requires:
  - "[[多响应线性回归]]"
  - "[[观测行协方差]]"
related:
  - "[[共享设计的逐响应OLS]]"
  - "[[相关行不容独立推断]]"
  - "[[随机效应GLS]]"
---

# 已知正定行协方差 Ω 时多响应 GLS 对每个响应使用同一个加权投影
<!-- bilingual-en:start -->
*With known positive-definite row covariance $\Omega$, multivariate GLS uses the same weighted projection for every response*
<!-- bilingual-en:end -->

> [!summary] 原子方法
> 在可分离误差协方差 $\Sigma\otimes\Omega$ 下，若 $\Omega\succ0$ 已知、$\Sigma\succ0$（可以未知）且 $X$ 满列秩，多响应 GLS 为
> $$
> \hat B_{GLS}=(X^T\Omega^{-1}X)^{-1}X^T\Omega^{-1}Y.
> $$
> 同一个 $\Omega^{-1}$ 加权投影作用于 $Y$ 的每一列，所以仍可逐响应计算；响应协方差 $\Sigma$ 不改变这个可分离模型中的系数点估计。
>
> <!-- bilingual-en:start -->
> With separable covariance $\Sigma\otimes\Omega$, known $\Omega\succ0$, positive-definite but possibly unknown $\Sigma$, and full-column-rank $X$, GLS applies the same $\Omega^{-1}$-weighted projection to every response column.
> <!-- bilingual-en:end -->

把模型左乘 $\Omega^{-1/2}$，得到
$$
\Omega^{-1/2}Y=\Omega^{-1/2}XB+\Omega^{-1/2}U.
$$
变换后的观测行协方差是 $I_n$，对它做普通最小二乘就得到上式。若 $\Omega=cI_n$，常数 $c$ 在正规方程中抵消，GLS 退化为 [[共享设计的逐响应OLS|逐响应 OLS]]；一般 $\Omega$ 下，两者通常不同。

这条公式不授权随意估一个相关矩阵再当作已知。若 $\Omega$ 由同一数据估计，得到的是 feasible GLS，需要计入协方差参数估计与错设风险；若 $X$ 秩亏，则只能用广义逆并解释可估函数。若完整误差协方差不可分离，有效 GLS 还可能在响应方向耦合，不再是这里的逐列形式。

> [!question]- 自检
> 为什么同一观测内的响应相关 $\Sigma$ 不改变这里的 $\hat B_{GLS}$，而观测行相关 $\Omega$ 会改变？
>
> **答案：** 可分离似然的一阶条件在响应方向右乘 $\Sigma^{-1}$，它可消去；$\Omega^{-1}$ 则改变观测行在投影中的权重。

## 来源与核验

- [SciPy `matrix_normal`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.matrix_normal.html)：核对可分离协方差 $\Sigma\otimes\Omega$ 的矩阵正态密度；对 $B$ 求一阶条件得到上述 GLS。
- [[观测行协方差]] 与 [[共享设计的逐响应OLS]]：分别核对 $\Omega$ 的含义，以及 $\Omega\propto I_n$ 时的 OLS 特例。
- [[随机效应GLS]]：给出特定面板协方差结构下 feasible GLS 的具体实例。
<!-- bilingual-en:start -->
- The matrix-normal density yields the GLS normal equation; the row-covariance and OLS atoms fix its assumptions and independent-row special case.
<!-- bilingual-en:end -->

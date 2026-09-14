---
aliases:
  - "非退化 Gaussian copula 没有渐近同尾依赖，而有限自由度 t-copula 具有对称上下尾依赖"
  - "Gaussian and t copula tail dependence"
student_os: knowledge-atom
atom_id: RM-DEP-003
atom_set: dependence-and-copulas
atom_type: model-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[Gaussian Copula]]"
  - "[[t Copula]]"
  - "[[尾部依赖]]"
related:
  - "[[共同违约概率]]"
  - "[[风险蒙特卡洛]]"
  - "[[极值理论]]"
part_of:
  - "[[依赖建模.canvas]]"
---

# 非退化 Gaussian copula 没有渐近同尾依赖，而有限自由度 t-copula 具有对称上下尾依赖
<!-- bilingual-en:start -->
*A non-degenerate Gaussian copula has no asymptotic same-tail dependence, whereas a finite-degree-of-freedom t copula has symmetric upper and lower tail dependence*
<!-- bilingual-en:end -->

> [!summary] 相同相关参数不等于相同联合极端风险
> 对 $-1<\rho<1$ 的二元 [[Gaussian Copula]]，上下 [[尾部依赖]]系数都为 0；具有有限自由度的标准二元 [[t Copula]] 则具有正的、上下对称的尾部依赖。这个差异来自模型族，而不是一个相关参数能够概括的差异。
> <!-- bilingual-en:start -->
> A non-degenerate bivariate Gaussian copula has zero upper and lower tail-dependence coefficients, whereas a finite-degree-of-freedom bivariate t copula has positive symmetric tail dependence.
> <!-- bilingual-en:end -->

## Gaussian copula 的渐近边界

对 $-1<\rho<1$，

$$
\lambda_L=\lambda_U=0.
$$

这是阈值趋向分布端点时的渐近结论，不表示有限阈值下两个极端事件不会共同发生。退化边界 $\rho=1$ 时两个系数为 1，不能混入非退化结论。

## t copula 的尾部系数

自由度 $\nu>0$、相关矩阵参数（更一般地说是形状参数）$-1<\rho<1$ 的标准二元 t copula 满足

$$
\lambda_L=\lambda_U
=2t_{\nu+1}\!\left(-\sqrt{\frac{(\nu+1)(1-\rho)}{1+\rho}}\right),
$$

其中 $t_{\nu+1}$ 是自由度 $\nu+1$ 的一元 Student t 分布函数。有限 $\nu$ 下该值为正；当 $\nu\to\infty$ 时趋向 Gaussian copula 的 0。公式只适用于 [[t Copula|标准 t copula 的共同尺度多元 t 构造]]。
<!-- bilingual-en:start -->
For a standard bivariate t copula with finite degrees of freedom, the common upper and lower tail-dependence coefficient is given by the displayed Student-t formula. It is positive for finite $\nu$ and tends to zero as $\nu$ grows.
<!-- bilingual-en:end -->

当 $\nu=4$、$\rho=0.5$ 时，

$$
\lambda_L=\lambda_U=2t_5(-\sqrt{5/3})\approx0.253170.
$$

即使 $\rho=0$，同一模型仍有 $\lambda_L=\lambda_U\approx0.075587$。共同随机尺度会制造联合极端，因此潜在零相关不等于尾部独立。

t copula 的上下尾系数相同，所以它也不能表达上尾与下尾强度不对称。参数 $\rho$ 控制潜在椭圆结构；只有 $\nu>2$ 时潜在 t 向量的 Pearson 相关存在并等于 $\rho$，更不能把它直接当作任意原始收益的 Pearson 相关。

> [!question]- 自检
> 两个模型的潜在相关参数都为 $0.5$，能否只凭这个数断言它们有相同的联合极端风险？
>
> **答案：** 不能。例如非退化 Gaussian copula 的渐近同尾依赖为 0，而 $\nu=4$ 的 t copula 约为 25.32%。

## 来源与核验

- Masaaki Sibuya (1960), [“Bivariate Extreme Statistics”](https://doi.org/10.1007/BF01682329)：核对非退化二元正态的渐近独立结论。
- Stefano Demarta and Alexander J. McNeil (2005), [“The t Copula and Related Copulas”](https://doi.org/10.1111/j.1751-5823.2005.tb00254.x)：核对标准 t copula 的尾部依赖公式与 Gaussian 极限。
- 作者逐式与数值复核日：2026-09-01；参数范围、退化边界和数值锚点已逐项核对。

---
aliases:
  - "KPSS 统计量由残差部分和与长期方差估计共同决定"
  - KPSS statistic
  - KPSS 统计量
student_os: knowledge-atom
atom_id: TS-UR-016
atom_set: trends-unit-roots-differencing
atom_type: derivation
status: source-checked
mastery_state: unassessed
requires:
  - "[[KPSS检验]]"
related:
  - "[[PP检验]]"
part_of:
  - "[[趋势、单位根与差分.canvas]]"
---

# KPSS 统计量由残差部分和与长期方差估计共同决定
<!-- bilingual-en:start -->
*The KPSS statistic combines residual partial sums with a long-run variance estimate*
<!-- bilingual-en:end -->

> [!summary] 原子推导
> KPSS 先按所选原假设回归常数或“常数加趋势”，取残差 $\widehat e_t$，再累计
> $S_t=\sum_{j=1}^{t}\widehat e_j$。统计量的核心是
> $$
> \operatorname{KPSS}=
> \frac{T^{-2}\sum_{t=1}^{T}S_t^2}{\widehat\omega^2},
> $$
> 其中 $\widehat\omega^2$ 是残差的长期方差估计。

若残差只是围绕固定水平波动，部分和的增长受到控制；若含随机趋势，部分和通常会系统性漂移，使分子变大。分母必须计入残差的短期自相关，而不是只用样本方差。

核函数、带宽或截断滞后决定 $\widehat\omega^2$，从而能改变有限样本结论。报告 KPSS 时应说明水平/趋势规格和长期方差设置；不能只给一个“平稳/非平稳”按钮结果。

> [!question]- 自检
> 为什么把 $\widehat\omega^2$ 换成普通残差方差可能误导？
>
> **答案：** 普通方差忽略跨期协方差；残差有序列相关时，它不等于部分和所需的长期方差。

## 来源与核验

- [Kwiatkowski et al. (1992)](https://doi.org/10.1016/0304-4076(92)90104-Y)：核对残差部分和、长期方差归一化和临界分布。

---
aliases:
  - "用真实均值与正定协方差计算的 Gaussian 平方 Mahalanobis 距离服从卡方分布"
  - "A Gaussian squared Mahalanobis distance based on the true mean and covariance has a chi-square law"
  - "多元正态马氏平方的卡方律"
student_os: knowledge-atom
atom_id: STAT-MVN-005
atom_set: multivariate-normal
atom_type: theorem-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[Gaussian密度与奇异支撑]]"
  - "[[Gaussian仿射闭包]]"
  - "[[正定逆平方根]]"
  - "[[马氏距离]]"
part_of:
  - "[[多元正态分布.canvas]]"
related:
  - "[[联合Gaussian诊断]]"
  - "[[单样本Hotelling T²]]"
  - "[[样本协方差Wishart律]]"
  - "[[正态均值协方差独立]]"
  - "[[Wishart抽样假设]]"
---

# 用真实均值与正定协方差计算的 Gaussian 平方 Mahalanobis 距离服从卡方分布
<!-- bilingual-en:start -->
*A Gaussian squared Mahalanobis distance based on the true mean and positive-definite covariance has a chi-square distribution*
<!-- bilingual-en:end -->

> [!summary] 原子定理
> 若 $X\sim N_p(\mu,\Sigma)$ 且 $\Sigma\succ0$，其中 $\mu,\Sigma$ 是固定的总体参数，则
> $$
> D^2=(X-\mu)^T\Sigma^{-1}(X-\mu)\sim\chi_p^2.
> $$
> <!-- bilingual-en:start -->
> A Gaussian observation measured from its known population centre with its known positive-definite covariance has an exact chi-square squared Mahalanobis distance.
> <!-- bilingual-en:end -->

[[正定逆平方根|白化]]令 $Z=\Sigma^{-1/2}(X-\mu)$。由 Gaussian 仿射闭包，$Z\sim N_p(0,I_p)$；联合 Gaussian 且协方差为单位阵使其分量独立标准正态，于是
$$
D^2=Z^TZ=\sum_{j=1}^pZ_j^2\sim\chi_p^2.
$$
这也说明自由度来自随机变化的独立方向数，而不是从矩阵尺寸机械抄来。
<!-- bilingual-en:start -->
Whitening turns the quadratic form into the sum of squares of $p$ independent standard normals, so its degrees of freedom equal the number of stochastic directions.
<!-- bilingual-en:end -->

若 $\Sigma$ 仅半正定、秩为 $r<p$，普通逆不存在。限定在 Gaussian 支撑上并使用 Moore–Penrose 逆时，
$$
(X-\mu)^T\Sigma^+(X-\mu)\sim\chi_r^2,
$$
但必须把自由度改为 $r$ 并明确这是奇异支撑上的推广。

若用同一批数据估计 $\mu$ 与 $\Sigma$，则只有在 $S$ 可逆时才能定义普通的样本距离
$$
d_i^2=(x_i-\bar x)^TS^{-1}(x_i-\bar x)
$$
它们不再是相互独立的精确 $\chi_p^2$ 变量；中心化和协方差估计都改变其有限样本分布。即使用独立训练样本估计中心与协方差，也只是让新观测与估计量独立；估计量本身仍随机，所以有限样本参考律仍不是已知参数下的 $\chi_p^2$。均值推断使用 [[单样本Hotelling T²|Hotelling 的 F 校准]]；异常诊断也必须区分已知参数、独立训练样本和同样本拟合三种设计，不能共用一条无校正的卡方阈值。

> [!question]- 自检
> 为什么把总体 $\mu,\Sigma$ 换成同一数据算出的 $\bar x,S$ 后，不能继续声称每个 $d_i^2\sim\chi_p^2$？
>
> **答案：** 估计量与各观测共享数据，距离受到中心化和拟合约束；已知参数的白化证明不再适用，有限样本参考律随估计设计而变。

## 来源与核验

- [[01_Math/04_多元统计分析/04_多元正态分布The Multivariate Normal Distribution.md#1.4. 二次型与相关分布|多元统计课程 §1.4]]：核对正定协方差下的白化与 $\chi_p^2$ 结论。
- [[01_Math/04_多元统计分析/04_多元正态分布The Multivariate Normal Distribution.md#1.8.2. 多变量正态性检查|多元统计课程 §1.8.2]]：核对同样本拟合的 Mahalanobis 平方只把 $\chi_p^2$ 作为近似图形参照，不是每个距离的精确独立抽样律。
- [MIT OCW 6.436J, Lecture 14, Definitions 3–4 and Theorem 1](https://ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/ceb423b6a703abd60429367b733f45bf_MIT6_436JF18_lec14.pdf)：核对 Gaussian 平方根构造、仿射闭包和独立标准正态坐标。
- [MIT OCW 18.443, Lecture 4](https://ocw.mit.edu/courses/18-443-statistics-for-applications-fall-2006/da928274a46632ce0d35d2012a1c8e74_lecture4.pdf)：核对谱坐标下非零特征值对应的独立 Gaussian 方向及奇异秩边界。

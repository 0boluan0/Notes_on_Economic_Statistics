---
aliases:
  - "边际正态与样本 Mahalanobis 图都不能单独证成联合 Gaussian"
  - "Marginal normality and sample Mahalanobis plots do not prove joint Gaussianity"
  - "多元正态性诊断边界"
student_os: knowledge-atom
atom_id: STAT-MVN-006
atom_set: multivariate-normal
atom_type: diagnostic-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[联合Gaussian]]"
  - "[[Gaussian马氏平方律]]"
related:
  - "[[联合Gaussian诊断]]"
  - "[[联合高斯独立判据]]"
  - "[[马氏距离不要求正态]]"
part_of:
  - "[[多元正态分布.canvas]]"
---

# 边际正态与样本 Mahalanobis 图都不能单独证成联合 Gaussian
<!-- bilingual-en:start -->
*Marginal normality and sample Mahalanobis plots cannot by themselves establish joint Gaussianity*
<!-- bilingual-en:end -->

> [!summary] 这条边界说什么
> [[联合Gaussian]]要求所有线性组合都服从一元 Gaussian。单变量图只检查坐标方向；用同一数据估计 $\bar x,S$ 后画出的 Mahalanobis Q–Q 图，又不再服从“总体参数已知”时的精确独立卡方抽样律。因此，这两类检查都能反对模型，却不能单独证明模型成立。
> <!-- bilingual-en:start -->
> Joint Gaussianity concerns every linear projection. Marginal plots inspect only coordinate directions, while fitted Mahalanobis plots do not have the exact independent chi-square law that holds with known population parameters.
> <!-- bilingual-en:end -->

## 边际正态为什么不够

每个坐标分别正态，只检查了线性组合 $a^TX$ 中 $a=e_j$ 的少数方向。变量可以各自是标准正态，却通过非 Gaussian 的方式相互依赖；某些其他线性组合便不再正态。一个具体构造见 [[联合高斯独立判据|Gaussian 边际但非联合 Gaussian 的反例]]。

成对散点图也只展示二维投影。维度更高时，弯曲、混合群体或尾部依赖可能只在其他方向出现。

## 样本 Mahalanobis 图为什么不是证明

精确结论

$$
(X-\mu)^T\Sigma^{-1}(X-\mu)\sim\chi_p^2
$$

要求使用固定的真实总体参数。实际诊断通常计算

$$
d_i^2=(x_i-\bar x)^TS^{-1}(x_i-\bar x),
$$

其中每个观测同时参与 $\bar x$ 与 $S$ 的拟合。这些距离受到共同中心化和协方差约束，既不是相互独立，也不能逐点冒充精确 $\chi_p^2$ 变量。$p\ge n$ 或严重共线时，$S$ 还会奇异，普通逆版本根本不存在。

## “没有拒绝”仍不是总体证明

任何有限样本诊断都有检验力边界：小样本可能看不出重要偏离，大样本又可能检出对用途无关的细小偏离。合理结论是“当前证据与该模型相容”或“出现了某类不相容”，而不是“总体已经被证明为 Gaussian”。完整检查流程见 [[联合Gaussian诊断]]。

> [!question]- 自检
> 所有边际 Q–Q 图都接近直线，样本 Mahalanobis 图也大致线性，能否据此宣布总体联合 Gaussian？
>
> **答案：** 不能。这些图提供相容证据，但没有穷尽所有线性方向；样本 Mahalanobis 距离还使用了同样本估计参数。

## 来源与核验

- [MIT OCW 6.436J, Lecture 14, Definition 4](https://ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/ceb423b6a703abd60429367b733f45bf_MIT6_436JF18_lec14.pdf)：核对联合 Gaussian 要求所有线性组合 Gaussian。
- [[Gaussian马氏平方律]]：核对已知总体参数的精确卡方律及同样本估计后的边界。
- [NIST/SEMATECH e-Handbook, Q–Q plots](https://www.itl.nist.gov/div898/handbook/eda/section3/eda33o.htm)：核对 Q–Q 图是分布形状诊断，而不是总体分布证明。

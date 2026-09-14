---
aliases:
  - "类条件 Gaussian 分布使用类别特有正定协方差时，二次项与对数行列式保留并通常产生二次判别边界"
  - "Quadratic discriminant analysis"
  - "QDA class-specific covariance"
student_os: knowledge-atom
atom_id: STAT-DA-006
atom_set: discriminant-analysis
atom_type: model-definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[Gaussian判别得分]]"
related:
  - "[[LDA共享协方差]]"
leads_to:
  - "[[判别协方差估计]]"
part_of:
  - "[[判别分析.canvas|判别分析]]"
---

# 类条件 Gaussian 分布使用类别特有正定协方差时，二次项与对数行列式保留并通常产生二次判别边界
<!-- bilingual-en:start -->
*Gaussian class-conditional distributions with class-specific positive-definite covariances retain quadratic and log-determinant terms and therefore usually produce quadratic decision boundaries*
<!-- bilingual-en:end -->

> [!summary] 模型是什么
> QDA 假定
> $$
> X\mid Y=k\sim N_p(\mu_k,\Sigma_k),
> \qquad \Sigma_k\succ0,
> $$
> 允许每个类别有不同的椭球大小、方向和形状。0–1 损失下仍选择 Gaussian 对数后验得分最大者，只是不再能删去类别共有的协方差项。
> <!-- bilingual-en:start -->
> QDA allows each class to have its own covariance geometry. Its pairwise score differences generally contain quadratic terms in the features.
> <!-- bilingual-en:end -->

## 类别特有协方差保留二次项

类别 $k$ 的得分为

$$
\delta_k^{\mathrm{QDA}}(x)
=\log\pi_k
-\frac12\log|\Sigma_k|
-\frac12(x-\mu_k)^T\Sigma_k^{-1}(x-\mu_k).
$$

比较 $k$ 与 $\ell$ 时，$x^T\Sigma_k^{-1}x$ 与 $x^T\Sigma_\ell^{-1}x$ 一般不能相消，所以边界通常是二次曲面。必须保留“通常”：若 $\Sigma_k=\Sigma_\ell$，QDA 退化为 LDA；某些特殊参数关系也可能让二次项相消或边界退化。

$-\tfrac12\log|\Sigma_k|$ 与距离项必须一起解释。一个类别协方差更大，意味着分布更扩散；它可能在远处给出较慢的距离惩罚，却同时受到较大的体积归一化惩罚。不能只看均值或只看 Mahalanobis 距离决定类别。

## 灵活性以逐类协方差估计为代价

每个 $p\times p$ 对称协方差需要 $p(p+1)/2$ 个参数。QDA 对每个类别各估一套，能表示不同形状，却在小样本或高维时更容易不稳定、奇异或过拟合。具体估计器见 [[判别协方差估计]]，逐类秩上界见 [[判别协方差秩边界]]，稳定化选择见 [[正则化判别]]。

QDA 的概率输出仍依赖 Gaussian 类条件模型、先验与参数估计。边界更弯并不自动更真实；应在完整训练管线外比较泛化损失和概率校准。

> [!question]- 自检
> 两类 QDA 最后估得完全相同的协方差矩阵。决策边界是否仍必然为二次曲线？
>
> **答案：** 不必然；共同二次项会相消，规则退化为 LDA 的线性边界。

## 来源与核验

- [scikit-learn, QDA mathematical formulation](https://scikit-learn.org/stable/modules/lda_qda.html#qda)：核对类别特有协方差、对数行列式与二次得分。
- [Penn State STAT 505, Lesson 10](https://online.stat.psu.edu/stat505/Lesson10)：核对 LDA/QDA 的协方差假设与分类规则。
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, 2nd ed., §4.3：核对 QDA 的参数化与偏差—方差权衡。

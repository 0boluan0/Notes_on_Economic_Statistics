---
aliases:
  - "类条件 Gaussian 分布共享同一正定协方差时，公共二次项相消并产生线性判别边界"
  - "Linear discriminant analysis"
  - "LDA shared covariance"
student_os: knowledge-atom
atom_id: STAT-DA-005
atom_set: discriminant-analysis
atom_type: model-definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[Gaussian判别得分]]"
related:
  - "[[QDA类别协方差]]"
leads_to:
  - "[[判别协方差估计]]"
  - "[[Fisher与LDA]]"
part_of:
  - "[[判别分析.canvas|判别分析]]"
---

# 类条件 Gaussian 分布共享同一正定协方差时，公共二次项相消并产生线性判别边界
<!-- bilingual-en:start -->
*Gaussian class-conditional distributions with one shared positive-definite covariance produce linear discriminant boundaries because their common quadratic term cancels*
<!-- bilingual-en:end -->

> [!summary] 模型是什么
> LDA 的生成模型是
> $$
> X\mid Y=k\sim N_p(\mu_k,\Sigma),
> \qquad \Sigma\succ0,
> $$
> 所有类别共享同一个协方差矩阵。密度本身仍是 Gaussian；“线性”指不同类别得分相等时得到的决策边界是仿射超平面。
> <!-- bilingual-en:start -->
> LDA assumes class-specific means but one shared covariance. Its pairwise decision boundaries are affine hyperplanes; the class densities themselves are not linear.
> <!-- bilingual-en:end -->

## 共同项怎样相消

从[[Gaussian判别得分]]展开并删除所有类别共有的 $-\tfrac12x^T\Sigma^{-1}x$ 与 $-\tfrac12\log|\Sigma|$，得到

$$
\delta_k^{\mathrm{LDA}}(x)
=x^T\Sigma^{-1}\mu_k
-\frac12\mu_k^T\Sigma^{-1}\mu_k
+\log\pi_k.
$$

在 0–1 损失下选择得分最大的类别。类别 $k$ 与 $\ell$ 的边界由 $\delta_k(x)=\delta_\ell(x)$ 给出：

$$
x^T\Sigma^{-1}(\mu_k-\mu_\ell)
-\frac12\left(
\mu_k^T\Sigma^{-1}\mu_k
-\mu_\ell^T\Sigma^{-1}\mu_\ell
\right)
+\log\frac{\pi_k}{\pi_\ell}=0.
$$

这是关于 $x$ 的一次方程。改变先验会移动截距；在共享协方差不变时，它不改变法向量 $\Sigma^{-1}(\mu_k-\mu_\ell)$。在两类问题中，不对称误判成本也只通过成本比移动截距。多类一般损失则必须按[[先验与分类风险]]重新比较各行动的后验风险，决策边界未必仍是这些 LDA 两两超平面。

## 共享协方差是在借强假设降低估计方差

LDA 把所有类别的组内信息合并估计一个 $\Sigma$，参数少于逐类估计协方差的 QDA。每类样本不多时，这种约束可显著降低估计噪声；但若真实类别形状差异很大，线性边界可能产生系统错分。

样本中的“协方差相等检验不显著”不等于证明共享协方差正确，显著也不自动说明 QDA 在新数据上更好。选择应结合领域结构、残差诊断、秩与条件性，以及训练外预测表现。

> [!question]- 自检
> 两类 LDA 使用同一个协方差。把稀有类别的先验调低后，边界仍是线性吗？
>
> **答案：** 仍是线性；先验只改变判别式的常数项，因而平移边界，不引入二次项。

## 来源与核验

- [scikit-learn, LDA mathematical formulation](https://scikit-learn.org/stable/modules/lda_qda.html#lda)：核对共享协方差下的线性对数后验与边界。
- [Penn State STAT 505, Lesson 10](https://online.stat.psu.edu/stat505/Lesson10)：核对线性判别函数、先验与分类规则。
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, 2nd ed., §4.3：核对 LDA 生成模型与决策边界。

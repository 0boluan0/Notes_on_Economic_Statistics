---
aliases:
  - "Gaussian 类条件模型把先验、协方差体积尺度与 Mahalanobis 距离合成对数后验得分，0–1 损失下选择得分最大的类别"
  - "Gaussian discriminant score"
  - "Gaussian discriminant function"
student_os: knowledge-atom
atom_id: STAT-DA-004
atom_set: discriminant-analysis
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[Bayes分类规则]]"
  - "[[Gaussian密度与奇异支撑]]"
  - "[[马氏距离]]"
  - "[[广义方差]]"
leads_to:
  - "[[LDA共享协方差]]"
  - "[[QDA类别协方差]]"
part_of:
  - "[[判别分析.canvas|判别分析]]"
---

# Gaussian 类条件模型把先验、协方差体积尺度与 Mahalanobis 距离合成对数后验得分，0–1 损失下选择得分最大的类别
<!-- bilingual-en:start -->
*A Gaussian class-conditional model combines priors, covariance volume scale, and Mahalanobis distance in a log-posterior score, whose maximum is chosen under zero-one loss*
<!-- bilingual-en:end -->

> [!summary] 得分是什么
> 若
> $$
> X\mid Y=k\sim N_p(\mu_k,\Sigma_k),
> \qquad \Sigma_k\succ0,
> $$
> 则去掉所有类别共有的常数后，类别 $k$ 的对数得分为
> $$
> \delta_k(x)=\log\pi_k-\frac12\log|\Sigma_k|
> -\frac12(x-\mu_k)^T\Sigma_k^{-1}(x-\mu_k).
> $$
> <!-- bilingual-en:start -->
> The Gaussian discriminant score is the class log posterior up to a common constant. It combines prior prevalence, covariance volume, and squared Mahalanobis distance.
> <!-- bilingual-en:end -->

## 三项各自承担不同角色

- $\log\pi_k$ 奖励目标总体中更常见的类别。
- $-\tfrac12\log|\Sigma_k|=-\log\sqrt{|\Sigma_k|}$ 是密度归一化的体积尺度项；同样的中心距离下，分布更扩散的类别不会在所有位置都获得更高密度。
- $-\tfrac12(x-\mu_k)^T\Sigma_k^{-1}(x-\mu_k)$ 惩罚相对该类协方差几何的[[马氏距离|平方 Mahalanobis 距离]]。

在 0–1 损失下，选择 $\delta_k(x)$ 最大的类别等价于选择后验概率最大的类别。若误判成本不对称，仍应先由这些得分归一化得到后验，再按[[先验与分类风险|后验风险]]选择行动；“最大得分”不再是一般答案。

使用对数得分还能避免多个很小密度值直接相乘造成下溢。若需要后验概率，可稳定地对各类得分做 log-sum-exp 归一化，而不是先计算极小的 Gaussian 密度。

## 正定条件不能省略

公式中的 $\Sigma_k^{-1}$ 与 $\log|\Sigma_k|$ 要求类协方差正定。奇异 Gaussian 只在低维仿射支撑上有密度，不能把普通逆与行列式公式强行用于整个 $\mathbb R^p$；应明确转到子空间模型、正则化协方差或其他分布，详见[[Gaussian密度与奇异支撑]]与[[正则化判别]]。

> [!question]- 自检
> 某观察离类别 A 的欧氏均值更近，是否必然有更高的 Gaussian 判别得分？
>
> **答案：** 不必然。得分使用类别自身的 Mahalanobis 距离，还同时包含协方差行列式和类别先验。

## 来源与核验

- [scikit-learn, Linear and Quadratic Discriminant Analysis, mathematical formulation](https://scikit-learn.org/stable/modules/lda_qda.html)：核对 Gaussian 对数后验得分及 LDA/QDA 特例。
- [Penn State STAT 505, Lesson 10](https://online.stat.psu.edu/stat505/Lesson10)：核对先验、类条件密度与判别函数。
- [[Gaussian密度与奇异支撑]]、[[马氏距离]]、[[广义方差]]：复用密度存在、距离与行列式的既有边界。

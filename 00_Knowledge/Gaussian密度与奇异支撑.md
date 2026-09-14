---
aliases:
  - "多元 Gaussian 仅在协方差正定时有全维密度而奇异时落在仿射子空间"
  - "Full-dimensional density and singular support of a multivariate Gaussian"
  - "多元正态密度的正定边界"
student_os: knowledge-atom
atom_id: STAT-MVN-003
atom_set: multivariate-normal
atom_type: theorem-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[联合Gaussian]]"
  - "[[正定二次型的椭球]]"
part_of:
  - "[[多元正态分布.canvas]]"
related:
  - "[[半正定主平方根]]"
  - "[[Gaussian马氏平方律]]"
---

# 多元 Gaussian 仅在协方差正定时有全维密度而奇异时落在仿射子空间
<!-- bilingual-en:start -->
*A multivariate Gaussian has a full-dimensional density only with positive-definite covariance; a singular law lies on an affine subspace*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> 若 $X\sim N_p(\mu,\Sigma)$ 且 $\Sigma\succ0$，则它相对于 $\mathbb R^p$ 上 Lebesgue 测度的密度为
> $$
> f_X(x)=\frac{1}{(2\pi)^{p/2}|\Sigma|^{1/2}}
> \exp\!\left[-\frac12(x-\mu)^T\Sigma^{-1}(x-\mu)\right].
> $$
> 若 $\Sigma\succeq0$ 但 $\operatorname{rank}(\Sigma)=r<p$，分布仍是合法的联合 Gaussian，却集中在 $\mu+\operatorname{range}(\Sigma)$ 上，因而没有 $p$ 维普通密度。
> <!-- bilingual-en:start -->
> Positive definiteness gives the usual full-dimensional Gaussian density. A singular covariance instead concentrates all probability on an affine subspace of dimension equal to its rank.
> <!-- bilingual-en:end -->

写 $\Sigma=Q\Lambda Q^T$。在正定情形，等密度面等价于
$$
(x-\mu)^T\Sigma^{-1}(x-\mu)=c.
$$
它沿特征向量 $q_i$ 的半轴长度是 $\sqrt{c\lambda_i}$：大特征值表示该方向自然变异更大，椭球更长。相关性改变特征向量方向，不能只从边际方差读出椭球朝向。
<!-- bilingual-en:start -->
For a positive-definite covariance, equal-density contours are Mahalanobis ellipsoids. Their principal directions are covariance eigenvectors, with semiaxis lengths proportional to the square roots of the eigenvalues.
<!-- bilingual-en:end -->

在奇异情形，可写 $X=\mu+\Sigma^{1/2}Z$。零特征值方向没有随机变化，所以支撑是低维仿射子空间；此时 $|\Sigma|=0$ 且 $\Sigma^{-1}$ 不存在，不能把伪逆和伪行列式直接塞进上面的 $p$ 维密度公式并继续称其为普通密度。可以在支撑子空间的内在测度上写低维 Gaussian 密度，但必须明确换了参考测度。

> [!question]- 自检
> 若 $X_2=2X_1+1$ 几乎处处且 $(X_1,X_2)$ 联合 Gaussian，它为什么没有二维密度？
>
> **答案：** 所有概率都落在直线 $x_2=2x_1+1$ 上；该直线在二维 Lebesgue 测度下为零测集，所以不存在二维普通密度。

## 来源与核验

- [[01_Math/04_多元统计分析/04_多元正态分布The Multivariate Normal Distribution.md#1.2. 多元正态密度及等密度曲线|多元统计课程 §1.2]]：核对正定协方差下密度与等密度椭球。
- [MIT OCW 6.436J, Lecture 14, Definitions 2–4](https://ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/ceb423b6a703abd60429367b733f45bf_MIT6_436JF18_lec14.pdf)：核对非退化密度与退化 Gaussian 集中在真子空间的区别。
- [MIT OCW 18.443, Lecture 4](https://ocw.mit.edu/courses/18-443-statistics-for-applications-fall-2006/da928274a46632ce0d35d2012a1c8e74_lecture4.pdf)：核对谱平方根构造、主轴方差及奇异 Gaussian 只在低维支撑上有密度。

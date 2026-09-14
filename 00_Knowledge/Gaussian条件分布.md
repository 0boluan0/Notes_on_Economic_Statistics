---
aliases:
  - "分块联合 Gaussian 的条件均值线性且条件协方差为 Schur 补"
  - "Conditional distribution of a partitioned multivariate Gaussian"
  - "多元正态条件分布"
student_os: knowledge-atom
atom_id: STAT-MVN-004
atom_set: multivariate-normal
atom_type: theorem-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[Gaussian仿射闭包]]"
  - "[[联合高斯独立判据]]"
part_of:
  - "[[多元正态分布.canvas]]"
related:
  - "[[Gaussian密度与奇异支撑]]"
---

# 分块联合 Gaussian 的条件均值线性且条件协方差为 Schur 补
<!-- bilingual-en:start -->
*A partitioned jointly Gaussian vector has a linear conditional mean and a Schur-complement conditional covariance*
<!-- bilingual-en:end -->

> [!summary] 原子定理
> 设
> $$
> \begin{pmatrix}X_1\\X_2\end{pmatrix}\sim N\!\left(
> \begin{pmatrix}\mu_1\\\mu_2\end{pmatrix},
> \begin{pmatrix}\Sigma_{11}&\Sigma_{12}\\\Sigma_{21}&\Sigma_{22}\end{pmatrix}
> \right),
> $$
> 且 $\Sigma_{22}$ 可逆。则
> $$
> X_1\mid X_2=x_2\sim N\!\left(
> \mu_1+\Sigma_{12}\Sigma_{22}^{-1}(x_2-\mu_2),
> \Sigma_{11}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}
> \right).
> $$
> <!-- bilingual-en:start -->
> Conditioning a jointly Gaussian block on a nondegenerate Gaussian block preserves Gaussianity. The conditional mean is affine in the observed block, and the conditional covariance is its Schur complement.
> <!-- bilingual-en:end -->

令
$$
R=X_1-\mu_1-\Sigma_{12}\Sigma_{22}^{-1}(X_2-\mu_2).
$$
由仿射闭包，$(R,X_2)$ 联合 Gaussian；直接计算得 $\operatorname{Cov}(R,X_2)=0$，故由 joint-Gaussian 的零协方差独立性，$R$ 与 $X_2$ 独立。这同时解释了为什么条件协方差不依赖具体的 $x_2$。
<!-- bilingual-en:start -->
The residual after the linear adjustment is jointly Gaussian and uncorrelated with the conditioning block, hence independent of it. This yields both the conditional mean and the constant conditional covariance.
<!-- bilingual-en:end -->

若总体协方差正定，主子块 $\Sigma_{22}$ 自动正定，上式可直接使用。若 $\Sigma_{22}$ 奇异，普通逆矩阵公式失效；此时只能在 $X_2$ 的支撑上讨论正则条件分布，并需另行说明广义逆及支撑条件。对支撑之外的 $x_2$，条件分布版本并不由原联合律唯一规定。

二维例子：若 $(X,Y)$ 为相关系数 $\rho=0.6$ 的标准二元正态，则
$$
Y\mid X=x\sim N(0.6x,1-0.6^2).
$$
所以观察 $X=2$ 后，条件均值为 $1.2$，条件方差为 $0.64$。注意这里的 $0.8$ 是条件**标准差**，不是方差。

> [!question]- 自检
> 为什么一般联合分布不能仅凭均值和协方差就套用上述条件均值公式？
>
> **答案：** 线性条件均值、常数条件协方差和残差独立是 joint Gaussian 的特殊结构；一般分布的同一组前两阶矩不能决定条件分布。

## 来源与核验

- [[01_Math/04_多元统计分析/04_多元正态分布The Multivariate Normal Distribution.md#1.3.4. 条件分布|多元统计课程 §1.3.4]]：核对分块条件均值、条件协方差及二维特例。
- [MIT OCW 6.436J, Lecture 14, Theorem 1(g)](https://ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/ceb423b6a703abd60429367b733f45bf_MIT6_436JF18_lec14.pdf)：核对可逆条件块下的线性条件期望、独立残差与 Schur 补。
- [MIT OCW 1.010, Brief Notes 7](https://ocw.mit.edu/courses/1-010-uncertainty-in-engineering-fall-2008/7d700f8750ca621ee14e370eeedaf000_notes_07.pdf)：核对条件协方差不依赖观测值，以及该结论的 joint-normal 边界。
- [Penn State STAT 505, Lesson 6](https://online.stat.psu.edu/stat505/Lesson06)：核对分块多元正态条件分布和回归式解释。

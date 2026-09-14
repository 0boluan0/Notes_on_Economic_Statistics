---
aliases:
  - "随机向量联合 Gaussian 当且仅当每个线性组合都是一元 Gaussian"
  - "Multivariate Gaussian linear-combination characterisation"
  - "多元正态的线性组合定义"
  - "Multivariate Normal Distribution"
  - "MVN"
  - "多元正态"
  - "Gaussian向量刻画"
student_os: knowledge-atom
atom_id: STAT-MVN-001
atom_set: multivariate-normal
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[协方差矩阵半正定性]]"
  - "[[半正定主平方根]]"
part_of:
  - "[[多元正态分布.canvas]]"
related:
  - "[[Gaussian仿射闭包]]"
  - "[[Gaussian密度与奇异支撑]]"
  - "[[联合高斯独立判据]]"
  - "[[Gaussian过程均值协方差决定性]]"
---

# 随机向量联合 Gaussian 当且仅当每个线性组合都是一元 Gaussian
<!-- bilingual-en:start -->
*A random vector is jointly Gaussian if and only if every linear combination is univariate Gaussian*
<!-- bilingual-en:end -->

> [!summary] 原子定义
> 对实随机向量 $X\in\mathbb R^p$，下列说法等价：
> 1. 对每个 $a\in\mathbb R^p$，$a^TX$ 都是一元 Gaussian；
> 2. 存在 $\mu\in\mathbb R^p$、矩阵 $B\in\mathbb R^{p\times r}$ 和独立标准正态向量 $Z\sim N_r(0,I_r)$，使
>    $$X=\mu+BZ.$$
> 此时 $E[X]=\mu$、$\operatorname{Cov}(X)=BB^T=: \Sigma\succeq0$，记作 $X\sim N_p(\mu,\Sigma)$。
> <!-- bilingual-en:start -->
> A real random vector is jointly Gaussian exactly when every scalar linear projection is Gaussian; equivalently, it is an affine image of a vector of independent standard normals. Its covariance may be singular.
> <!-- bilingual-en:end -->

这里把常数也算作方差为 0 的退化 Gaussian，所以 $a=0$ 不会成为定义的例外。反过来，任意对称半正定 $\Sigma$ 都能取 $B=\Sigma^{1/2}$ 构造 $N_p(\mu,\Sigma)$；因此存在条件是 $\Sigma\succeq0$，不是必须 $\Sigma\succ0$。
<!-- bilingual-en:start -->
Constants are treated as degenerate Gaussians with variance zero. Every positive-semidefinite covariance is admissible because a square root supplies the constructive representation.
<!-- bilingual-en:end -->

$\mu$ 与 $\Sigma$ 唯一决定分布，因为特征函数为
$$
\varphi_X(t)=\exp\!\left(i t^T\mu-\frac12t^T\Sigma t\right).
$$
生成因子 $B$ 却不唯一：只要 $BB^T=\Sigma$，不同 $B$ 构造的是同一个 Gaussian 律。

“每个分量各自正态”只检查了 $a=e_j$ 的少数方向，远弱于“每个 $a$”。例如，[[联合高斯独立判据|Gaussian 边际但非联合 Gaussian 的反例]]中，两分量都标准正态，却有某些线性组合不是正态。
<!-- bilingual-en:start -->
Normal coordinates alone are insufficient because they test only the coordinate projections. Joint Gaussianity requires all directions, not merely all marginals.
<!-- bilingual-en:end -->

> [!question]- 自检
> 若 $X_1$ 与 $X_2$ 各自都服从正态，能否据此写 $(X_1,X_2)^T\sim N_2(\mu,\Sigma)$？
>
> **答案：** 不能。还要确认所有线性组合 $a_1X_1+a_2X_2$ 都正态；边际正态不能排除非 Gaussian 的联合依赖。

## 来源与核验

- [[01_Math/04_多元统计分析/04_多元正态分布The Multivariate Normal Distribution.md#1.3. 多元正态分布的性质|多元统计课程 §1.3]]：核对线性组合刻画和参数记号。
- [MIT OCW 6.436J, Lecture 14, Definitions 2–4 and Theorem 1](https://ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/ceb423b6a703abd60429367b733f45bf_MIT6_436JF18_lec14.pdf)：核对密度、仿射生成与全体线性组合三种定义的等价边界，以及退化 Gaussian 约定。
- [MIT OCW 18.443, Lecture 4](https://ocw.mit.edu/courses/18-443-statistics-for-applications-fall-2006/da928274a46632ce0d35d2012a1c8e74_lecture4.pdf)：核对半正定协方差的平方根构造、参数决定性与奇异情形。

---
aliases:
  - "在满列秩、零条件均值和球形误差条件下，OLS 系数协方差由 σ²(X'X)⁻¹ 给出，并可用残差自由度估计"
  - Classical OLS covariance estimation
  - OLS 系数协方差矩阵
student_os: knowledge-atom
atom_id: ECON-OLS-016
atom_set: regression-inference
atom_type: proposition
status: source-checked
mastery_state: unassessed
part_of:
  - "[[回归推断.canvas|回归推断]]"
requires:
  - "[[满列秩与OLS唯一性]]"
  - "[[经典 Gauss–Markov 定理]]"
leads_to:
  - "[[回归t检验]]"
  - "[[线性组合推断]]"
---

# 在满列秩、零条件均值和球形误差条件下，OLS 系数协方差由 σ²(X'X)⁻¹ 给出，并可用残差自由度估计
<!-- bilingual-en:start -->
*Under full column rank, zero conditional mean, and spherical errors, the OLS coefficient covariance is σ²(X'X)⁻¹ and can be estimated using residual degrees of freedom*
<!-- bilingual-en:end -->

> [!summary] 原子命题
> 在线性模型 $y=X\beta+u$ 中，若 $X$ 满列秩、$E(u\mid X)=0$ 且 $\operatorname{Var}(u\mid X)=\sigma^2I$，则
> $$
> \operatorname{Var}(\hat\beta\mid X)=\sigma^2(X'X)^{-1}.
> $$
> 未知的 $\sigma^2$ 用残差平方和除以残差自由度估计；协方差矩阵对角元的平方根才是各系数的标准误。
>
> <!-- bilingual-en:start -->
> For $y=X\beta+u$, full column rank, $E(u\mid X)=0$, and $\operatorname{Var}(u\mid X)=\sigma^2I$ give $\operatorname{Var}(\hat\beta\mid X)=\sigma^2(X'X)^{-1}$. Estimate the unknown error variance from the residual sum of squares and residual degrees of freedom; coefficient standard errors are the square roots of the diagonal covariance entries.
> <!-- bilingual-en:end -->

由
$$
\hat\beta-\beta=(X'X)^{-1}X'u
$$
先得到一般条件协方差
$$
\operatorname{Var}(\hat\beta\mid X)
=(X'X)^{-1}X'\operatorname{Var}(u\mid X)X(X'X)^{-1}.
$$
只有把误差协方差进一步设为 $\sigma^2I$，中间矩阵才缩成 $\sigma^2X'X$。因此经典公式不是 OLS 的代数恒等式，而是数据误差结构进入推断的结果。
<!-- bilingual-en:start -->
The general conditional covariance follows from $\hat\beta-\beta=(X'X)^{-1}X'u$. It reduces to the classical expression only when the error covariance is $\sigma^2I$; the formula is therefore a sampling-model result, not an algebraic identity of least squares.
<!-- bilingual-en:end -->

令 $p=\operatorname{rank}(X)$，在满列秩模型中也就是实际估计的参数数目，截距若存在也算一个参数。经典残差方差估计量是
$$
\hat\sigma^2=\frac{\hat u'\hat u}{n-p}=\frac{RSS}{n-p},
$$
从而
$$
\widehat{\operatorname{Var}}(\hat\beta\mid X)
=\hat\sigma^2(X'X)^{-1}.
$$
写成 $n-p$ 可以避免两种常见记号冲突：有的教材用 $k$ 表示包含截距的参数数，有的用 $k$ 表示不含截距的解释变量数。
<!-- bilingual-en:start -->
Writing the denominator as $n-p$, where $p=\operatorname{rank}(X)$ is the number of estimated parameters, avoids ambiguity over whether a symbol such as $k$ includes the intercept.
<!-- bilingual-en:end -->

$\hat\sigma$ 描述观测方程的噪声尺度；$se(\hat\beta_j)$ 描述某个系数估计量的抽样不确定性。后者还受 $X$ 的尺度、变异和共线性影响，所以两者不能互换。若异方差或相关性使球形误差条件失效，系数仍可能是 OLS，但这张卡上的经典协方差估计就不再是正确口径，转到 [[标准误口径匹配]]。

> [!question]- 自检
> 一个含截距和三个解释变量的模型使用 $n=100$ 个观测。经典 $\hat\sigma^2$ 的分母是多少？为什么不是 99？
>
> **答案：** $p=4$，所以分母是 $100-4=96$。拟合过程使用了四个参数自由度，不是只估计一个样本均值。

## 来源与核验

- [[02_Economy/01_Econometrics/03_多元线性回归.md#2.2. 随机误差项方差的估计|本地课程：多元回归误差方差估计]]：核对 $RSS/(n-k-1)$；改写为 $RSS/(n-p)$ 统一截距记号。
- [[02_Economy/01_Econometrics/05_多元回归模型的矩阵表达.md#2.3. OLS 估计量的方差协方差矩阵|本地课程：OLS 方差协方差矩阵]] 与 [[02_Economy/01_Econometrics/05_多元回归模型的矩阵表达.md#3.1. 有限样本性质|有限样本推导]]：核对一般协方差传播与球形误差下的化简。
- [MIT OpenCourseWare 14.310x, Lecture 17](https://ocw.mit.edu/courses/14-310x-data-analysis-for-social-scientists-spring-2023/mit14_310x_s23_week08_lec17.pdf)：交叉核验回归抽样分布、误差方差估计和系数标准误。
<!-- bilingual-en:start -->
- The local matrix-regression sections verify the covariance derivation and residual degrees of freedom; MIT OCW 14.310x Lecture 17 provides an independent course-level check.
<!-- bilingual-en:end -->

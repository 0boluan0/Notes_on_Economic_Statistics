---
aliases:
  - "Conditional distribution of a partitioned multivariate Gaussian"
  - "多元正态条件分布"
  - "Conditional Gaussian distribution"
student_os: knowledge-atom
atom_id: STAT-MVN-004
atom_set: multivariate-normal
atom_type: theorem-boundary
status: source-checked
requires:
  - "[[Gaussian仿射闭包]]"
  - "[[联合高斯独立判据]]"
part_of:
  - "[[多元正态分布.canvas]]"
related:
  - "[[Gaussian密度与奇异支撑]]"
---

<!-- bilingual-en:start -->
# Conditional Gaussian distributions

A conditional distribution describes the uncertainty that remains after another quantity is observed. For a [[联合Gaussian|jointly Gaussian]] vector, observing some coordinates leaves a Gaussian distribution for the remaining coordinates. The observation shifts the remaining mean; the conditional covariance is the same for every observed value in the formula below.
<!-- bilingual-en:end -->

*Gaussian 条件分布*

条件分布描述得知另一个量以后，剩余的不确定性。对于[[联合Gaussian|联合 Gaussian]] 向量，观察其中一部分坐标后，其余坐标的条件分布仍是 Gaussian。观测值会移动剩余部分的均值，而下述公式中的条件协方差不随具体观测值改变。

> [!example]
> <!-- bilingual-en:start -->
> Let $Z_1,Z_2$ be independent [[正态分布|standard normals]] and set
> $$X=Z_1,\qquad Y=0.6Z_1+0.8Z_2.$$
> Both $X$ and $Y$ have variance $1$, and their [[相关系数|correlation]] is $0.6$. Once $X=x$ is observed, the first term of $Y$ is fixed at $0.6x$; the independent noise $0.8Z_2$ remains. Therefore
> $$Y\mid X=x\sim N(0.6x,0.64).$$
> At $x=2$, the conditional mean is $1.2$, the conditional variance is $0.64$, and the conditional [[标准差|standard deviation]] is $0.8$. The zero-probability meaning of conditioning on an exact value is explained below.
> <!-- bilingual-en:end -->
>
> 设 $Z_1,Z_2$ 是独立[[正态分布|标准正态]]，令
> $$X=Z_1,\qquad Y=0.6Z_1+0.8Z_2.$$
> $X,Y$ 的方差均为 $1$，[[相关系数]]为 $0.6$。观察到 $X=x$ 后，$Y$ 的第一项固定为 $0.6x$，独立噪声 $0.8Z_2$ 仍然存在。因此
> $$Y\mid X=x\sim N(0.6x,0.64).$$
> 当 $x=2$ 时，条件均值为 $1.2$，条件方差为 $0.64$，条件[[标准差]]为 $0.8$。对一个概率为零的精确观测值如何理解“条件”，下文会说明。

<!-- bilingual-en:start -->
## Two blocks of coordinates

Partition one jointly Gaussian vector into $X_1\in\mathbb R^p$ and $X_2\in\mathbb R^q$. The subscripts label **blocks**, each of which may have several coordinates. Partition its [[均值向量|mean vector]] and [[协方差矩阵|covariance matrix]] in the same way:
$$
\begin{pmatrix}X_1\\X_2\end{pmatrix}\sim N\!\left(
\begin{pmatrix}\mu_1\\\mu_2\end{pmatrix},
\begin{pmatrix}\Sigma_{11}&\Sigma_{12}\\\Sigma_{21}&\Sigma_{22}\end{pmatrix}
\right).
$$
Here $\Sigma_{11}$ describes variation within $X_1$, $\Sigma_{22}$ within $X_2$, and the $p\times q$ matrix $\Sigma_{12}=\Sigma_{21}^T$ contains cross-covariances. Assume $\Sigma_{22}$ is invertible, so its [[逆矩阵|inverse]] exists. For a covariance matrix this is equivalent to being [[正定矩阵|positive definite]].
<!-- bilingual-en:end -->

*把坐标分成两块*

把一个联合 Gaussian 向量分成 $X_1\in\mathbb R^p$ 和 $X_2\in\mathbb R^q$。下标标记的是**坐标块**，每块可以含多个坐标。相应地划分[[均值向量]]与[[协方差矩阵]]：
$$
\begin{pmatrix}X_1\\X_2\end{pmatrix}\sim N\!\left(
\begin{pmatrix}\mu_1\\\mu_2\end{pmatrix},
\begin{pmatrix}\Sigma_{11}&\Sigma_{12}\\\Sigma_{21}&\Sigma_{22}\end{pmatrix}
\right).
$$
$\Sigma_{11}$ 描述 $X_1$ 内部的变化，$\Sigma_{22}$ 描述 $X_2$ 内部的变化，$p\times q$ 矩阵 $\Sigma_{12}=\Sigma_{21}^T$ 则记录跨块协方差。假设 $\Sigma_{22}$ 可逆，因而存在[[逆矩阵]]。对协方差矩阵而言，可逆等价于[[正定矩阵|正定]]。

> [!important]
> <!-- bilingual-en:start -->
> A Gaussian version of the conditional distribution is
> $$
> X_1\mid X_2=x_2\sim N_p\!\left(
> \mu_1+\Sigma_{12}\Sigma_{22}^{-1}(x_2-\mu_2),\quad
> \Sigma_{11}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}
> \right).
> $$
> The mean equals the original mean plus an adjustment for how far the observation lies from $\mu_2$. The $p\times q$ matrix $K=\Sigma_{12}\Sigma_{22}^{-1}$ converts that observed deviation into a $p$-coordinate adjustment. The covariance matrix
> $$C=\Sigma_{11}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}$$
> is called the **Schur complement of $\Sigma_{22}$** in the block covariance matrix. It is positive semidefinite and may be singular.
> <!-- bilingual-en:end -->
>
> 条件分布的一个 Gaussian 版本为
> $$
> X_1\mid X_2=x_2\sim N_p\!\left(
> \mu_1+\Sigma_{12}\Sigma_{22}^{-1}(x_2-\mu_2),\quad
> \Sigma_{11}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}
> \right).
> $$
> 条件均值等于原均值，加上针对观测值偏离 $\mu_2$ 程度的调整。$p\times q$ 矩阵 $K=\Sigma_{12}\Sigma_{22}^{-1}$ 把观测偏差转成 $p$ 个坐标的均值调整。协方差矩阵
> $$C=\Sigma_{11}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}$$
> 称为这个分块协方差矩阵中 **$\Sigma_{22}$ 的 Schur 补**。它是半正定的，也可能奇异。

<!-- bilingual-en:start -->
## Why the remaining noise is independent

Subtract the predicted part and call what remains the residual:
$$R=X_1-\mu_1-K(X_2-\mu_2).$$
By [[Gaussian仿射闭包|Gaussian affine closure]], $(R,X_2)$ is jointly Gaussian. Its cross-covariance is
$$\operatorname{Cov}(R,X_2)=\Sigma_{12}-K\Sigma_{22}=0.$$
For jointly Gaussian blocks, [[联合高斯独立判据|zero cross-covariance implies independence]]. Also $E[R]=0$ and $\operatorname{Cov}(R)=C$. Rewriting
$$X_1=\mu_1+K(X_2-\mu_2)+R$$
now explains the formula: after fixing $X_2=x_2$, only the independent Gaussian residual $R$ remains random. Its covariance $C$ does not change with $x_2$.

Evaluating the mean formula at the random block $X_2$ gives the random [[条件期望|conditional expectation]] $E[X_1\mid X_2]$. Evaluating it at a specified value $x_2$ gives a numerical vector.
<!-- bilingual-en:end -->

*为什么剩余噪声与观测块独立*

减去预测部分，把剩余量称为残差：
$$R=X_1-\mu_1-K(X_2-\mu_2).$$
由 [[Gaussian仿射闭包|Gaussian 仿射闭包]]，$(R,X_2)$ 仍联合 Gaussian。它们的跨块协方差为
$$\operatorname{Cov}(R,X_2)=\Sigma_{12}-K\Sigma_{22}=0.$$
对联合 Gaussian 块，[[联合高斯独立判据|跨块协方差为零推出独立]]。并且 $E[R]=0$、$\operatorname{Cov}(R)=C$。把原式写成
$$X_1=\mu_1+K(X_2-\mu_2)+R$$
就能看懂条件分布公式：固定 $X_2=x_2$ 后，只有独立的 Gaussian 残差 $R$ 仍随机。它的协方差 $C$ 不随 $x_2$ 改变。

在条件均值公式中代入随机块 $X_2$，得到随机的[[条件期望]] $E[X_1\mid X_2]$；代入指定观测值 $x_2$，得到数值向量。

<!-- bilingual-en:start -->
## What the invertibility condition does

A positive-definite full covariance guarantees that $\Sigma_{22}$ is positive definite. The full covariance need not be positive definite, however: only the conditioning block must be invertible for this formula. For example, if $X_1=X_2=Z\sim N(0,1)$, then $\Sigma_{22}=1$ and $C=0$. After observing $X_2=x_2$, $X_1$ equals $x_2$ with certainty. This is a degenerate Gaussian conditional distribution.

If $\Sigma_{22}$ is singular, the displayed ordinary-inverse formula cannot be used. For instance, a conditioning block $(Z,Z)$ only takes values on a line. A generalized-inverse extension must account for that [[Gaussian密度与奇异支撑|support]]; it is a separate version of the result.
<!-- bilingual-en:end -->

*可逆条件管的是哪一块*

整个协方差正定会保证 $\Sigma_{22}$ 正定。不过，使用这个公式并不要求整个协方差正定，只要求用来做条件的块可逆。例如，若 $X_1=X_2=Z\sim N(0,1)$，则 $\Sigma_{22}=1$ 而 $C=0$。观察到 $X_2=x_2$ 后，$X_1$ 必定等于 $x_2$，这是退化的 Gaussian 条件分布。

若 $\Sigma_{22}$ 奇异，上述普通逆矩阵公式就不能使用。例如，条件块 $(Z,Z)$ 只能在一条直线上取值。广义逆版本必须同时处理这种[[Gaussian密度与奇异支撑|支撑]]限制，属于需要另行说明条件的推广。

> [!note]
> <!-- bilingual-en:start -->
> Because a nondegenerate Gaussian block assigns probability zero to each exact value, $X_1\mid X_2=x_2$ is not defined by dividing by $P(X_2=x_2)$. The formula specifies a **regular conditional distribution**: a probability distribution indexed by the observed value. It is a valid Gaussian choice for every $x_2$ when $\Sigma_{22}$ is invertible. Conditional distributions are determined by the joint law only up to sets of $x_2$ having probability zero under the distribution of $X_2$. Other versions can differ on such sets, even within the support; values outside a singular conditioning support are likewise not fixed by the joint law.
> <!-- bilingual-en:end -->
>
> 非退化 Gaussian 块在每个精确取值上的概率都为零，所以 $X_1\mid X_2=x_2$ 不是通过除以 $P(X_2=x_2)$ 来定义的。公式指定了一个**正则条件分布**：以观测值为索引的一族概率分布。当 $\Sigma_{22}$ 可逆时，公式为每个 $x_2$ 都给出了一个合法的 Gaussian 选择。联合分布只在“除去 $X_2$ 分布下的零概率取值集”这一意义上确定条件分布；其他版本可以在这样的集合上不同，即使这些点位于支撑内部。奇异条件块支撑之外的取值，同样不能由联合分布唯一规定。

> [!warning]
> <!-- bilingual-en:start -->
> The affine conditional mean and observation-independent conditional covariance come from joint Gaussian structure. For a general joint distribution, the same means and covariances do not determine the conditional distribution, and an uncorrelated residual need not be independent.
> <!-- bilingual-en:end -->
>
> 条件均值的仿射形式、条件协方差不依赖观测值，都来自联合 Gaussian 的特殊结构。对一般联合分布，同一组均值和协方差不能决定条件分布；不相关的残差也未必独立。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/04_多元统计分析/04_多元正态分布The Multivariate Normal Distribution.md#1.3.4. 条件分布|Multivariate statistics course, §1.3.4]]: block conditional mean and covariance formulas.
- [MIT 6.436J Lecture 14, pp. 5, 7–8, Theorem 1(g)](https://ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/ceb423b6a703abd60429367b733f45bf_MIT6_436JF18_lec14.pdf): invertible conditioning covariance, independent residual, and its covariance.
- [MIT 1.010 Brief Notes 7, p. 2](https://ocw.mit.edu/courses/1-010-uncertainty-in-engineering-fall-2008/7d700f8750ca621ee14e370eeedaf000_notes_07.pdf) and [Penn State STAT 505, Lesson 6, §6.1](https://online.stat.psu.edu/stat505/Lesson06): conditional Gaussian parameters and bivariate variance. The $0.6/0.8$ example is calculated directly.
- [Durrett, Probability: Theory and Examples, §4.1.3, printed pp. 214–215](https://services.math.duke.edu/~rtd/PTE/PTE5_011119.pdf): regular conditional distributions as probability kernels and almost-everywhere versions.
<!-- bilingual-en:end -->

- [[01_Math/04_多元统计分析/04_多元正态分布The Multivariate Normal Distribution.md#1.3.4. 条件分布|多元统计课程 §1.3.4]]：分块条件均值与协方差公式。
- [MIT 6.436J Lecture 14，第 5、7–8 页，Theorem 1(g)](https://ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/ceb423b6a703abd60429367b733f45bf_MIT6_436JF18_lec14.pdf)：可逆条件协方差、独立残差及其协方差。
- [MIT 1.010 Brief Notes 7，第 2 页](https://ocw.mit.edu/courses/1-010-uncertainty-in-engineering-fall-2008/7d700f8750ca621ee14e370eeedaf000_notes_07.pdf) 与 [Penn State STAT 505, Lesson 6, §6.1](https://online.stat.psu.edu/stat505/Lesson06)：Gaussian 条件参数和二维条件方差。$0.6/0.8$ 例子由正文直接计算。
- [Durrett, Probability: Theory and Examples，§4.1.3，印刷第 214–215 页](https://services.math.duke.edu/~rtd/PTE/PTE5_011119.pdf)：以概率核表达的正则条件分布及几乎处处版本。

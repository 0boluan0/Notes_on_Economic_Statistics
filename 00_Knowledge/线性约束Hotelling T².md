---
aliases:
  - Hotelling test for linear restrictions
  - 线性变换均值向量的 Hotelling T²
  - 线性约束的 Hotelling 检验
student_os: knowledge-atom
atom_id: STAT-HOT-003
atom_set: hotelling-mean-inference
atom_type: reduction-theorem
status: source-checked
requires:
  - "[[单样本Hotelling T²]]"
  - "[[Gaussian仿射闭包]]"
  - "[[矩阵秩]]"
part_of:
  - "[[Hotelling T² 与多元均值推断.canvas]]"
related:
  - "[[Hotelling投影极值]]"
  - "[[配对Hotelling T²]]"
  - "[[pooled Hotelling T²]]"
---

<!-- bilingual-en:start -->
# Hotelling tests for linear restrictions
<!-- bilingual-en:end -->

*线性约束的 Hotelling 检验*

<!-- bilingual-en:start -->
Sometimes a question concerns only a few differences or weighted averages of several population means. For example, asking whether four instruments have the same mean reading does not specify what that common reading should be. A test of **linear restrictions** keeps the combinations that express the question and applies the [[单样本Hotelling T²|one-sample Hotelling test]] to those transformed observations.
<!-- bilingual-en:end -->

有时，我们只关心多个总体均值之间的几个差异或加权平均。例如，问四台仪器的平均读数是否相同，并没有规定这个共同读数必须是多少。**线性约束检验**先保留真正表达问题的组合，再对变换后的观测使用[[单样本Hotelling T²|单样本 Hotelling 检验]]。

<!-- bilingual-en:start -->
## Write the question as a transformation

Let $X_i=(X_{i1},\ldots,X_{ip})^T$ be the $p$ measurements on observation $i$, and let $\mu=E[X_i]$ be their [[均值向量|population mean vector]]. A fixed real $m\times p$ matrix $D$ lists one set of weights per row. The [[矩阵乘法|matrix product]] $Y_i=DX_i$ therefore produces $m$ combinations from each original observation. The corresponding population means are $E[Y_i]=D\mu$.

The hypothesis is $H_0:D\mu=\psi_0$, where the fixed $m$-vector $\psi_0$ lists the proposed values of those combinations. For equality of four means, choose
<!-- bilingual-en:end -->

*把问题写成变换*

设 $X_i=(X_{i1},\ldots,X_{ip})^T$ 是第 $i$ 次观测的 $p$ 个测量值，$\mu=E[X_i]$ 是它们的[[均值向量|总体均值向量]]。固定实矩阵 $D$ 的尺寸为 $m\times p$，每一行列出一组权重。[[矩阵乘法]] $Y_i=DX_i$ 因而把每次原始观测转换为 $m$ 个组合，组合的总体均值为 $E[Y_i]=D\mu$。

原假设写成 $H_0:D\mu=\psi_0$，固定的 $m$ 维向量 $\psi_0$ 列出这些组合应有的值。若要检验四个均值相等，可以取

$$
D=\begin{pmatrix}
1&0&0&-1\\
0&1&0&-1\\
0&0&1&-1
\end{pmatrix},\qquad \psi_0=\begin{pmatrix}0\\0\\0\end{pmatrix}.
$$

<!-- bilingual-en:start -->
Then $DX_i=(X_{i1}-X_{i4},X_{i2}-X_{i4},X_{i3}-X_{i4})^T$. All three means vanish exactly when $\mu_1=\mu_2=\mu_3=\mu_4$. A common mean of $100$ is just as compatible with this hypothesis as a common mean of $0$. Here $m=3$ counts the independent restrictions, while $p=4$ counts the original measurements.
<!-- bilingual-en:end -->

此时 $DX_i=(X_{i1}-X_{i4},X_{i2}-X_{i4},X_{i3}-X_{i4})^T$。这三个差值的总体均值都为零，恰好等价于 $\mu_1=\mu_2=\mu_3=\mu_4$。共同均值为 $100$ 和共同均值为 $0$ 都符合原假设。这里 $m=3$ 数的是独立约束，$p=4$ 数的是原始测量变量。

<!-- bilingual-en:start -->
## Transform the covariance as well as the mean

For $n\ge2$ complete observations, the [[样本均值向量|sample mean]] and [[样本协方差矩阵|sample covariance]] are
<!-- bilingual-en:end -->

*均值和协方差一起变换*

对 $n\ge2$ 个完整观测，[[样本均值向量]]与[[样本协方差矩阵]]为

$$
\bar X=\frac1n\sum_{i=1}^nX_i,\qquad
S=\frac1{n-1}\sum_{i=1}^n(X_i-\bar X)(X_i-\bar X)^T.
$$

<!-- bilingual-en:start -->
The transformed mean is $\bar Y=D\bar X$. Since $Y_i-\bar Y=D(X_i-\bar X)$, its sample covariance is
<!-- bilingual-en:end -->

变换后的样本均值为 $\bar Y=D\bar X$。由 $Y_i-\bar Y=D(X_i-\bar X)$，变换后的样本协方差为

$$
S_Y=\frac1{n-1}\sum_{i=1}^n(Y_i-\bar Y)(Y_i-\bar Y)^T=DSD^T.
$$

<!-- bilingual-en:start -->
Here $D^T$ is the [[矩阵转置|transpose]]. $DSD^T$ estimates the covariance of **one transformed observation**; the estimated covariance of its sample mean is $DSD^T/n$. That distinction explains the factor $n$ in the statistic below. These sample identities are algebraic. Writing $\Sigma=\operatorname{Cov}(X_i)$ for the population covariance, the population version is the [[协方差仿射变换|covariance transformation rule]], $\operatorname{Cov}(DX_i)=D\Sigma D^T$.
<!-- bilingual-en:end -->

其中 $D^T$ 是[[矩阵转置|转置]]。$DSD^T$ 估计**一个变换后观测**的协方差，样本均值的协方差估计则是 $DSD^T/n$。这一区别解释了下式中的因子 $n$。以上样本恒等式是代数事实；以 $\Sigma=\operatorname{Cov}(X_i)$ 表示总体协方差，总体版本是[[协方差仿射变换]] $\operatorname{Cov}(DX_i)=D\Sigma D^T$。

> [!important] Exact calibration in the transformed dimension · 在变换后的维数上精确校准
> <!-- bilingual-en:start -->
> Suppose $X_1,\ldots,X_n$ are [[独立同分布|independent and identically distributed]] $N_p(\mu,\Sigma)$, with [[正定矩阵|positive-definite]] $\Sigma$. Choose $D$ and $\psi_0$ before examining the test data. Require $1\le m\le p$, $D$ to have full row [[矩阵秩|rank]] $m$, and $n>m$. Then $DSD^T$ is invertible with probability one, and under $H_0:D\mu=\psi_0$,
> <!-- bilingual-en:end -->
>
> 设 $X_1,\ldots,X_n$ [[独立同分布]]于 $N_p(\mu,\Sigma)$，且 $\Sigma$ [[正定矩阵|正定]]。在查看检验数据前确定 $D$ 和 $\psi_0$，要求 $1\le m\le p$、$D$ 满行[[矩阵秩|秩]] $m$，并且 $n>m$。此时 $DSD^T$ 以概率一可逆，且在 $H_0:D\mu=\psi_0$ 下，
>
> $$
> T_D^2=n(D\bar X-\psi_0)^T(DSD^T)^{-1}(D\bar X-\psi_0),\qquad
> \boxed{\frac{n-m}{m(n-1)}T_D^2\sim F_{m,n-m}.}
> $$
>
> <!-- bilingual-en:start -->
> The [[F分布|F distribution]] has numerator degrees of freedom $m$ and denominator degrees of freedom $n-m$. For a significance level $0<\alpha<1$ selected in advance, reject when the scaled statistic exceeds its cumulative $1-\alpha$ quantile. This tests the whole vector of restrictions jointly.
> <!-- bilingual-en:end -->
>
> [[F分布|F 分布]]的分子自由度为 $m$，分母自由度为 $n-m$。预先选择显著性水平 $0<\alpha<1$，缩放后的统计量超过累积概率为 $1-\alpha$ 的分位数时拒绝原假设。这是在联合检验整组约束。

<!-- bilingual-en:start -->
Why does the theorem apply? [[Gaussian仿射闭包|Gaussian affine closure]] gives $Y_i\sim N_m(D\mu,D\Sigma D^T)$, and applying the same fixed transformation separately preserves independence across observations. Full row rank means no nonzero vector $a\in\mathbb R^m$ satisfies $D^Ta=0$. Consequently
$$a^TD\Sigma D^Ta=(D^Ta)^T\Sigma(D^Ta)>0\quad(a\ne0).$$
Thus the transformed covariance is positive definite, and the ordinary $m$-dimensional one-sample theorem applies. The rows of $D$ are **[[线性无关|linearly independent]] weights**; this does not say the output coordinates are statistically independent.
<!-- bilingual-en:end -->

为什么能用这个定理？[[Gaussian仿射闭包]]给出 $Y_i\sim N_m(D\mu,D\Sigma D^T)$，对每个观测分别施加同一个固定变换，仍保留观测之间的独立性。满行秩意味着不存在满足 $D^Ta=0$ 的非零向量 $a\in\mathbb R^m$。于是
$$a^TD\Sigma D^Ta=(D^Ta)^T\Sigma(D^Ta)>0\quad(a\ne0).$$
所以变换后的协方差正定，可以使用普通的 $m$ 维单样本定理。$D$ 的行是**[[线性无关|线性无关]]的权重**，这不表示各输出坐标在概率意义上相互独立。

> [!example] Equal means can have correlated differences · 均值相等问题中的差值也会相关
> <!-- bilingual-en:start -->
> In the four-instrument example, suppose $n=10$, $\bar X=(10,11,12,10)^T$ and $S=I_4$, where $I_4$ has ones on the diagonal and zeros elsewhere. Then $D\bar X=(0,1,2)^T$ and
> <!-- bilingual-en:end -->
>
> 在四台仪器的例子中，设 $n=10$、$\bar X=(10,11,12,10)^T$、$S=I_4$，其中 $I_4$ 的对角为 $1$、其余元素为 $0$。于是 $D\bar X=(0,1,2)^T$，并且
>
> $$
> DSD^T=\begin{pmatrix}2&1&1\\1&2&1\\1&1&2\end{pmatrix},\qquad
> (DSD^T)^{-1}=\frac14\begin{pmatrix}3&-1&-1\\-1&3&-1\\-1&-1&3\end{pmatrix}.
> $$
>
> <!-- bilingual-en:start -->
> Each difference contains the fourth measurement, which creates the off-diagonal covariance $1$. Keeping these terms gives $T_D^2=10(11/4)=27.5$ and scaled statistic $7(27.5)/27=385/54\approx7.130$, to compare with $F_{3,7}$. Treating the three differences as uncorrelated would use the wrong covariance.
> <!-- bilingual-en:end -->
>
> 每个差值都包含第四个测量，因此产生非对角协方差 $1$。保留这些项得到 $T_D^2=10(11/4)=27.5$，缩放统计量为 $7(27.5)/27=385/54\approx7.130$，应与 $F_{3,7}$ 比较。把三个差值当成不相关变量，会用错协方差。

<!-- bilingual-en:start -->
## Redundancy and equivalent ways of writing the question

If $A$ is an invertible $m\times m$ matrix, replacing $(D,\psi_0)$ with $(AD,A\psi_0)$ expresses the same hypothesis. Writing $v=D\bar X-\psi_0$ and $V=DSD^T$, the [[乘积的逆矩阵|inverse product rule]] gives
$$n(Av)^T(AVA^T)^{-1}(Av)=nv^TV^{-1}v.$$
So rescaling or recombining the restrictions leaves $T_D^2$ unchanged when the right-hand side changes with them. The statistic depends on the hypothesis, not a particular basis for its [[行空间|row space]].

If $D$ has rank $r<m$, first check consistency. The rows $(1,-1)$ and $(2,-2)$ with targets $(0,1)^T$ demand both $\mu_1-\mu_2=0$ and $2(\mu_1-\mu_2)=1$: no mean vector satisfies them. Targets $(3,6)^T$ are consistent and reduce to the single restriction $\mu_1-\mu_2=3$. More generally, keep a basis of $r\ge1$ independent rows with their corresponding targets, check the discarded equations follow, and use $n>r$ and $F_{r,n-r}$. A $5\times8$ matrix of rank $3$ supplies three restrictions, not five. If $r=0$, the system is either inconsistent or imposes no restriction; there is no zero-degree Hotelling test.
<!-- bilingual-en:end -->

*冗余约束与同一问题的不同写法*

若 $A$ 是可逆的 $m\times m$ 矩阵，把 $(D,\psi_0)$ 同时改写为 $(AD,A\psi_0)$，表达的仍是同一个原假设。记 $v=D\bar X-\psi_0$、$V=DSD^T$，由[[乘积的逆矩阵|乘积求逆规则]]可得
$$n(Av)^T(AVA^T)^{-1}(Av)=nv^TV^{-1}v.$$
因此，只要右端也同步变换，对约束缩放或重新组合都不会改变 $T_D^2$。统计量依赖的是原假设，而不是约束[[行空间]]的某一组特定基。

若 $D$ 的秩为 $r<m$，先检查约束是否一致。例如，行 $(1,-1)$、$(2,-2)$ 配上目标 $(0,1)^T$，同时要求 $\mu_1-\mu_2=0$ 和 $2(\mu_1-\mu_2)=1$，不存在满足条件的均值向量。目标改为 $(3,6)^T$ 就相容，可以化为一个约束 $\mu_1-\mu_2=3$。一般地，保留 $r\ge1$ 个线性无关行及其对应目标，确认被删方程可由它们推出，再用 $n>r$ 与 $F_{r,n-r}$。一个秩为 $3$ 的 $5\times8$ 矩阵提供三个约束，而不是五个。若 $r=0$，约束要么不相容，要么完全没有限制；不能使用零自由度的 Hotelling 检验。

> [!warning] The chosen question determines the scope · 结论的范围由所选问题决定
> <!-- bilingual-en:start -->
> Only $n>m$ is needed under the stated model. For instance, $p=4,n=4,m=3$ makes the original $S$ singular by the [[样本协方差秩边界|sample-covariance rank bound]], while $DSD^T$ is invertible with probability one. No inverse of $S$ is required. An actually singular transformed covariance still cannot be inserted into the classical formula by merely replacing its inverse with a pseudoinverse.
>
> The exact theorem also holds more generally whenever the transformed observations themselves are iid Gaussian with positive-definite covariance, even if the full input covariance is singular. That transformed-model condition must be established. Selecting a favorable $D$ from the same data changes the calibration, and a test of $D\mu$ says nothing automatically about mean directions outside the chosen row space.
> <!-- bilingual-en:end -->
>
> 在上述模型下，只需 $n>m$。例如 $p=4,n=4,m=3$ 时，原始 $S$ 由[[样本协方差秩边界]]可知必然奇异，但 $DSD^T$ 以概率一可逆，无须求 $S^{-1}$。若实际变换后的协方差仍奇异，也不能仅把逆改成伪逆，就继续使用经典校准。
>
> 更一般地，只要变换后的观测本身 iid Gaussian 且协方差正定，精确定理就适用，即使完整输入协方差奇异；必须另外确认这个变换后模型。利用同一批数据挑选有利的 $D$ 会改变校准；检验 $D\mu$ 也不会自动回答所选行空间之外的均值方向问题。

## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/04_多元统计分析/05_ 总体平均向量的推论.md#1.3. 线性变换下的均值向量检验|Multivariate statistics course, Chapter 5 §1.3]] and [[01_Math/04_多元统计分析/06_比较多个均值向量comparisons of multivariate mean vectors.md#1.3. 参数线性变换检验|Chapter 6 §1.3]]: course scope and notation for transformed means. $DSD^T$ is the observation covariance estimate; the mean covariance estimate is $DSD^T/n$.
- [MIT 6.436J Lecture 14, pp. 3–6, Theorem 1(c)](https://ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/ceb423b6a703abd60429367b733f45bf_MIT6_436JF18_lec14.pdf): Gaussian affine closure and transformed mean/covariance. Positive definiteness for full row rank is proved above.
- [NIST/SEMATECH, Hotelling's T-squared](https://www.itl.nist.gov/div898/handbook/pmc/section5/pmc543.htm): one-sample F scaling, applied to the $m$-dimensional transformed data. The displayed transpose and exact Gaussian assumptions are explicit here.
- [Penn State STAT 505, Lesson 7, §§7.1.1–7.1.4](https://online.stat.psu.edu/stat505/Lesson07): one-sample calibration and linear-combination inference. Redundant-constraint reduction, invariance and numerical examples follow by the displayed algebra.
<!-- bilingual-en:end -->

- 多元统计课程第 5 章 §1.3 与第 6 章 §1.3 核对课程范围与变换均值记号。$DSD^T$ 是观测协方差估计，均值协方差估计为 $DSD^T/n$。
- MIT 第 3–6 页定理 1(c) 核对 Gaussian 仿射闭包及变换后的均值、协方差；满行秩时的正定性由正文证明。
- NIST 核对单样本 F 缩放，再用于 $m$ 维变换数据；本文显式保留转置和精确 Gaussian 条件。
- Penn State §7.1.1–§7.1.4 核对单样本校准和线性组合推断。冗余约束化简、换基不变性与数值例子由正文代数核验。

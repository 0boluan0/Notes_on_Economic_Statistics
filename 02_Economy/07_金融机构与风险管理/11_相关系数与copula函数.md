
# 1. 相关与协方差的定义与计算
<!-- bilingual-en:start -->
*1. Definition and Calculation of Correlation and Covariance*
<!-- bilingual-en:end -->

## 1.1 相关系数与协方差定义
<!-- bilingual-en:start -->
*1.1 Definitions of the Correlation Coefficient and Covariance*
<!-- bilingual-en:end -->

三种常用的[[相关度量比较|相关度量]]回答不同问题：

- **[[相关系数|Pearson相关系数]]**
  $$
  \rho_P=\operatorname{Corr}(X,Y)
  =\frac{\operatorname{Cov}(X,Y)}{\sigma_X\sigma_Y}
  $$
  衡量有限二阶矩下的**线性**共同变化；任一变量方差为0时无定义，对离群值敏感，也可能漏掉强非线性关系。
- **[[Spearman秩相关|Spearman秩相关系数]]**是样本秩的Pearson相关。连续边际下可写为$\rho_S=\operatorname{Corr}(F_X(X),F_Y(Y))$，衡量单调一致性，并对各变量的严格递增变换不变；若只对一个变量作严格递减变换，符号会反转。存在并列秩时必须采用约定的平均秩和并列修正。
- **[[Kendall秩相关|Kendall秩相关系数]]**在连续、无并列情形下为
  $$
  \tau=P\big((X-X')(Y-Y')>0\big)-P\big((X-X')(Y-Y')<0\big),
  $$
  其中$(X',Y')$是独立同分布副本。它比较一致对与不一致对；存在并列时要明确使用$\tau_b$等修正版。

三者都只是依赖关系的投影，单独等于0都不能一般性地证明独立。独立且相应矩存在会推出这些相关度量为0；反方向只有在额外分布条件下才成立，例如联合正态中Pearson相关为0可推出独立。
<!-- bilingual-en:start -->
The three common [[相关度量比较|dependence measures]] answer different questions:

- **[[相关系数|Pearson correlation]]**, $\rho_P=\operatorname{Cov}(X,Y)/(\sigma_X\sigma_Y)$, measures linear co-movement when second moments are finite. It is undefined if either variance is zero, sensitive to outliers, and can miss strong nonlinear dependence.
- **[[Spearman秩相关|Spearman rank correlation]]** is Pearson correlation applied to sample ranks. With continuous marginals it can be written as $\rho_S=\operatorname{Corr}(F_X(X),F_Y(Y))$. It measures monotone association and is invariant to strictly increasing transformations; ties require a stated ranking and tie convention.
- **[[Kendall秩相关|Kendall rank correlation]]** is the probability of concordance minus the probability of discordance for two independent copies. Ties require a version such as $\tau_b$.

All three are projections of dependence. A value of zero does not generally prove independence. Independence implies zero correlation when the relevant quantities exist; the converse needs extra assumptions, such as joint normality for zero Pearson correlation.
<!-- bilingual-en:end -->

## 1.2 EWMA更新协方差和相关系数
<!-- bilingual-en:start -->
*1.2 EWMA updates covariance and correlation coefficient*
<!-- bilingual-en:end -->

[[EWMA波动率|EWMA]] 必须先说明收益口径。若向量 $u_{n-1}$ 是零条件均值收益或已经去均值的创新，并对整个矩阵使用同一个 $0<\lambda<1$，则
$$
\Sigma_n=\lambda\Sigma_{n-1}+(1-\lambda)u_{n-1}u_{n-1}^{\mathsf T}.
$$
因此两变量协方差与方差分别为
$$
\operatorname{Cov}_{XY,n}=\lambda\operatorname{Cov}_{XY,n-1}+(1-\lambda)x_{n-1}y_{n-1},
\qquad
\sigma^2_{X,n}=\lambda\sigma^2_{X,n-1}+(1-\lambda)x_{n-1}^2.
$$
若 $\Sigma_{n-1}$ 为半正定矩阵，同一 $\lambda$ 的矩阵递推仍保持半正定；对每个资产对随意使用不同衰减因子则不保证这一性质。相关矩阵的合法性见 [[协方差矩阵半正定性]]。
<!-- bilingual-en:start -->
If $u_{n-1}$ contains zero-conditional-mean returns or demeaned innovations, EWMA updates the whole covariance matrix with one common decay factor: $\Sigma_n=\lambda\Sigma_{n-1}+(1-\lambda)u_{n-1}u_{n-1}^{\mathsf T}$. A positive-semidefinite starting matrix remains positive semidefinite under this update. Arbitrary pair-specific decay factors do not guarantee a valid covariance matrix.
<!-- bilingual-en:end -->

>[!question] 
>**模拟考题：**假设在第 $n-1$ 日，资产 $X$ 和 $Y$ 的相关系数估计值为 0.6，波动率估计分别为 1%和 2%（即 $\sigma_{X,n-1}=1\%, \sigma_{Y,n-1}=2\%$）。已知当日协方差 $\mathrm{Cov}_{n-1}=0.6 \times 0.01 \times 0.02 = 0.00012$。若第 $n-1$ 日 $X$ 的收益为 0.5%（即 $x_{n-1}=0.5\%$），$Y$ 的收益为 2.5%（$y_{n-1}=2.5\%$），使用指数加权移动平均法（EWMA，衰减因子 $\lambda=0.95$）计算第 $n$ 日更新的协方差和相关系数。

>[!example] 
> **详细解答：**EWMA 下协方差的更新公式为：
> $$
> \mathrm{Cov}_{n} = \lambda \,\mathrm{Cov}_{n-1} + (1-\lambda)\,x_{n-1}\,y_{n-1} \,,
> $$ 
> 而方差更新类似：$\sigma^2_{X,n} = \lambda\,\sigma^2_{X,n-1} + (1-\lambda)\,x_{n-1}^2$（$Y$ 同理）。将数值代入：
> - $\sigma^2_{X,n} = 0.95 \times (0.01)^2 + 0.05 \times (0.005)^2 = 0.00009625$，则$\sigma_{X,n} = \sqrt{0.00009625} \approx 0.00981$（即0.981%）。
> - $\sigma^2_{Y,n} = 0.95 \times (0.02)^2 + 0.05 \times (0.025)^2 = 0.00041125$，则$\sigma_{Y,n} = \sqrt{0.00041125} \approx 0.02028$（即2.028%）。
> - 协方差更新：$\mathrm{Cov}_{n} = 0.95 \times 0.00012 + 0.05 \times (0.005 \times 0.025) = 0.00012025$。
> 
> 因此，第 $n$ 日的相关系数为：
> $$
> \rho_n \;=\; \frac{\mathrm{Cov}_n}{\sigma_{X,n}\,\sigma_{Y,n}} \;=\; \frac{0.00012025}{(0.00981)\,(0.02028)} \;\approx\; 0.6044 \,. 
> $$
> 相比上一日的相关0.6略有上升。

## 1.3独立性与不相关的区别

独立意味着联合分布可以分解为边际分布的乘积；不相关只说明某一种共同变化测度为0，因此条件更弱。复习时应分别检查[[相关度量比较|相关度量]]、相关矩阵合法性、[[尾部依赖]]与[[Copula分解|Sklar 分解]]，不能用一个相关系数替代完整的联合分布。
<!-- bilingual-en:start -->
Independence means that the joint distribution factorizes into its marginals. Zero correlation only sets one dependence summary to zero and is therefore weaker. A complete review separates the chosen dependence measure, matrix validity, tail dependence, and the Copula decomposition instead of treating one coefficient as the whole joint distribution.
<!-- bilingual-en:end -->

## 协方差矩阵的正定性条件


在多变量情形下，所有随机变量的协方差构成对称矩阵$\Omega$。有效的协方差矩阵必须是**半正定**（positive semidefinite, PSD）的，即对任意向量$w$：
$$ 
w^T\,\Omega\,w \;\ge\; 0 \,. 
$$ 
若对每个非零$w$都有严格不等式$w^T\Omega w>0$，矩阵才是**正定**（positive definite, PD）。PSD允许某个非零线性组合方差为0；PD不允许。内部一致性只要求PSD，否则某个组合会得到负方差。

等价地，实对称矩阵PSD当且仅当全部特征值非负，也当且仅当全部**主子式**非负。PD要求全部特征值严格为正；按Sylvester判据，也等价于全部**顺序主子式**严格为正。不要把PSD与PD的判据混写。**例如：**
$$ 
\Omega = \begin{pmatrix}
1 & 0 & 0.9\\[6pt]
0 & 1 & 0.9\\[6pt]
0.9 & 0.9 & 1
\end{pmatrix} \,,
$$ 
这是一个对角元素为1、部分相关系数为0.9的相关矩阵假设。直观来看，第一变量与第二变量不相关，但都与第三变量高度正相关。然而该矩阵并非正定矩阵。计算其行列式：
$$ 
\det(\Omega) = 1 \cdot \det\begin{pmatrix}1 & 0.9\\ 0.9 & 1\end{pmatrix} - 0 + 0.9 \cdot \det\begin{pmatrix}0 & 1\\ 0.9 & 0.9\end{pmatrix} \,,
$$ 
$$ 
\det(\Omega) = 1(1 - 0.9^2) + 0.9(0 - 0.9) = 1(1 - 0.81) - 0.9^2 = 0.19 - 0.81 = -0.62 \,<\, 0 \,. 
$$ 
由于行列式为负，该矩阵存在负特征值，不满足半正定条件。因此这个“相关矩阵”不具备内部一致性，实际上不可能是某组随机变量的相关矩阵。在风险管理中，若估计矩阵不是PSD，需要在保留对角线和业务约束的前提下做最近PSD等修正；若下游算法要求标准Cholesky，还可能需要进一步得到PD矩阵。

**模拟考题：**判断以下相关矩阵是否满足正定要求，并给出理由：
$$ 
\Omega = \begin{pmatrix}
1 & 0 & 0.9\\
0 & 1 & 0.9\\
0.9 & 0.9 & 1
\end{pmatrix} \,. $$
<!-- bilingual-en:start -->
Under EWMA, covariance is updated by combining the previous covariance with the latest cross-product of returns, while each variance is updated in the same way using the latest squared return. In the worked example, the previous correlation is 0.6, the volatilities are 1% and 2%, the latest returns are 0.5% and 2.5%, and the decay factor is 0.95. The updated variances are 0.00009625 and 0.00041125, giving volatilities of approximately 0.00981 and 0.02028. The updated covariance is 0.00012025, so the new correlation is approximately 0.6044, slightly above the previous value of 0.6.

Independence is stronger than zero correlation: independent variables are uncorrelated when their moments exist, but uncorrelated variables need not be independent unless additional distributional assumptions, such as joint normality, apply.

A valid covariance or correlation matrix must be symmetric and positive semidefinite because every linear-combination variance must satisfy $w^T\Omega w\ge0$. Positive definite means strict positivity for every nonzero $w$. A symmetric matrix is PSD exactly when all eigenvalues, or equivalently all principal minors, are non-negative; Sylvester's leading-principal-minor test with strict inequalities is for PD. For the displayed matrix, the three second-order principal minors are $1$, $0.19$, and $0.19$, but the full determinant is $-0.62$. The negative determinant implies a negative eigenvalue, so the matrix is not PSD. A downstream standard Cholesky routine may require a PD repair rather than merely a singular PSD approximation.
<!-- bilingual-en:end -->

详细解答：检验$\Omega$是否PSD，可以计算特征值或全部主子式。上面矩阵的一阶主子式均为1，三个二阶主子式分别为$1$、$0.19$和$0.19$，但三阶行列式为$-0.62$。因此$\Omega$存在负特征值，不是PSD，不能作为有效的相关矩阵；结论是**不满足半正定要求**。
<!-- bilingual-en:start -->
Detailed answer: the first-order principal minors are all 1 and the three second-order principal minors are $1$, $0.19$, and $0.19$, whereas the full determinant is $-0.62$. The matrix therefore has a negative eigenvalue and would imply a negative variance for some linear combination. It **fails** the positive-semidefiniteness requirement.
<!-- bilingual-en:end -->

## 多元正态分布与相关系数的生成机制
<!-- bilingual-en:start -->
*Generating Correlated Multivariate Normal Variables*
<!-- bilingual-en:end -->

别看
<!-- bilingual-en:start -->
Skip this section.
<!-- bilingual-en:end -->

在**[[多元正态分布.canvas|多元正态分布]]**中，[[Gaussian仿射闭包|任意线性组合仍为正态]]，而[[Gaussian条件分布|条件分布仍为正态]]。例如，若 $(V_1, V_2)$ 服从二维正态分布，$V_2$ 在给定 $V_1=v_1$ 条件下仍是正态，其条件均值和标准差为：
$$ 
E[V_2 \mid V_1 = v_1] = \mu_2 + \rho\,\frac{\sigma_2}{\sigma_1}\, (v_1 - \mu_1)\,, \qquad 
\sqrt{\mathrm{Var}(V_2 \mid V_1 = v_1)} = \sigma_2\,\sqrt{\,1-\rho^2\,} \,,
$$ 
其中 $\mu_i, \sigma_i$ 是 $V_i$ 的均值和标准差，$\rho$ 是相关系数。这表明在联合正态中，相关使得一个变量对另一个的条件期望是线性函数，条件方差为常数。
<!-- bilingual-en:start -->
In a **[[多元正态分布.canvas|multivariate normal distribution]]**, [[Gaussian仿射闭包|every linear combination is normal]], and [[Gaussian条件分布|conditioning preserves Gaussianity]] under the stated covariance-block conditions. For example, if $(V_1, V_2)$ is bivariate normal, then the conditional distribution of $V_2$ given $V_1=v_1$ is normal with the mean and standard deviation shown above. Here, $\mu_i$ and $\sigma_i$ are the mean and standard deviation of $V_i$, and $\rho$ is the correlation coefficient. Thus, under joint normality, one variable's conditional mean is a linear function of the other variable, while its conditional variance is constant.
<!-- bilingual-en:end -->

**相关系数的生成机制：**对于正态分布，我们可以通过线性变换方便地“制造”出指定的相关性。例如，要生成**两**个相关系数为 $\rho$ 的标准正态随机变量 $X, Y$，可以按以下步骤：
1. 先生成两个独立标准正态变量 $Z_1, Z_2 \sim N(0,1)$；
2. 定义 
$$
X = Z_1,\qquad 
Y = \rho\,Z_1 + \sqrt{\,1-\rho^2\,}\;Z_2\,.
$$ 
由此构造的 $(X, Y)$ 均为标准正态且相关系数为 $\rho$。这是因为 $E(X)=E(Y)=0,\ \mathrm{Var}(Y) = \rho^2 + (1-\rho^2)=1$，且 
$$
Cov(X,Y) = E(XY) = E[\rho Z_1^2 + \sqrt{1-\rho^2} Z_1 Z_2] = \rho\,E(Z_1^2) + 0 = \rho\,,
$$ 
从而 $Corr(X,Y)=\rho$。
<!-- bilingual-en:start -->
**Generating a prescribed correlation:** With normal variables, a linear transformation can create any valid target correlation. To generate **two** standard normal variables $X$ and $Y$ with correlation $\rho$:
**1.** Generate two independent standard normal variables $Z_1, Z_2 \sim N(0,1)$.<br>
**2.** Define $X$ and $Y$ as shown above.<br>
Both constructed variables are standard normal. Moreover, $E(X)=E(Y)=0,\ \mathrm{Var}(Y) = \rho^2 + (1-\rho^2)=1$, and the displayed covariance calculation gives $\mathrm{Cov}(X,Y)=\rho$. Therefore, $\mathrm{Corr}(X,Y)=\rho$.
<!-- bilingual-en:end -->

一般地，对于$n$维正态分布，可以使用**Cholesky分解**。若目标协方差矩阵$\Sigma$对称PD，则存在唯一的正对角下三角矩阵$A$使$AA^T=\Sigma$。生成独立标准正态向量$Z=(Z_1,\dots,Z_n)^T$并令$X=AZ$，便得到协方差为$\Sigma$的正态向量。任意$AA^T$只能保证PSD，不自动保证PD；若$\Sigma$只是奇异PSD，标准正对角Cholesky不适用，需要允许零对角的广义/主元分解或其他矩阵平方根。
<!-- bilingual-en:start -->
More generally, if the target covariance matrix $\Sigma$ is symmetric PD, it has a unique lower-triangular Cholesky factor $A$ with a positive diagonal and $AA^T=\Sigma$. For an independent standard normal vector $Z$, the vector $X=AZ$ is normal with covariance $\Sigma$. Any product $AA^T$ is PSD, not automatically PD. A singular PSD matrix needs a generalized or pivoted factorization, or another matrix square root, rather than the standard positive-diagonal Cholesky factor.
<!-- bilingual-en:end -->

**模拟考题：**假设我们需要模拟两个相关的标准正态随机变量，目标相关系数为 $\rho=0.5$。请给出一种可行的模拟方法（要求利用独立正态变量来构造）。
<!-- bilingual-en:start -->
Suppose we need to simulate two correlated standard normal random variables with the target correlation coefficient being $\rho=0.5$.Please provide a feasible simulation method (which requires independent normal variables to be constructed).
<!-- bilingual-en:end -->

**详细解答：**方法之一是使用线性组合构造法。首先生成 $Z_1, Z_2 \sim N(0,1)$，且相互独立。然后令：
$$ 
X = Z_1,\qquad 
Y = 0.5\,Z_1 + \sqrt{1-0.5^2}\,Z_2 = 0.5\,Z_1 + \sqrt{0.75}\,Z_2 \,. 
$$ 
这样得到的 $X, Y$ 均为标准正态随机变量。由于 $Y$ 包含了 $Z_1$ 的成分，两者之间的相关系数为 $0.5$。验证：$\mathrm{Cov}(X,Y) = 0.5\,\mathrm{Var}(Z_1) = 0.5$，标准差均为1，因此相关系数 $=0.5$。这种构造方法可以推广到任意 $\rho$ 值（$-1 \le \rho \le 1$）。
<!-- bilingual-en:start -->
**Detailed answer:** Generate independent variables $Z_1, Z_2 \sim N(0,1)$ and define $X$ and $Y$ as shown above. Both $X$ and $Y$ are standard normal. Because $Y$ contains the component $0.5Z_1$, $\mathrm{Cov}(X,Y) = 0.5\,\mathrm{Var}(Z_1) = 0.5$; both standard deviations are 1, so the correlation is $0.5$. The same construction works for any $\rho$ in $-1 \le \rho \le 1$.
<!-- bilingual-en:end -->

# 2. 因子模型
<!-- bilingual-en:start -->
*2. Factor Models*
<!-- bilingual-en:end -->

当涉及 $N$ 个随机变量（如 $N$ 个资产收益）时，直接估计两两之间的相关系数有 $\frac{N(N-1)}{2}$ 个参数，随着 $N$ 增大变得非常繁琐。**[[公共因子模型|因子模型（Factor Model）]]**假设变量的相关结构由少数几个共同因子驱动，从而减少需估计的参数数量。
<!-- bilingual-en:start -->
With $N$ random variables, such as $N$ asset returns, estimating every pairwise correlation requires $\frac{N(N-1)}{2}$ parameters. This quickly becomes unwieldy as $N$ grows. A **[[公共因子模型|factor model]]** assumes that a small number of common factors drive most of the dependence, greatly reducing the number of parameters that must be estimated.
<!-- bilingual-en:end -->

## 2.1 单因子模型
<!-- bilingual-en:start -->
*2.1 One-Factor Model*
<!-- bilingual-en:end -->

**单因子模型：**假设存在一个公共因子 $F$，以及每个变量各自的独立特异因素 $Z_i$。令 $U_i$ 表示标准化后的第 $i$ 个变量（均值0，方差1，例如资产收益的标准化），模型表示为：
$$
U_i = a_i\,F \;+\; \sqrt{\,1 - a_i^2\,}\;Z_i \,, \qquad i=1,2,\dots,N,
$$ 
其中 $F \sim N(0,1)$，各 $Z_i \sim N(0,1)$ 彼此独立且与 $F$ 独立，$a_i$ 是第 $i$ 个变量对公共因子的加载系数（$-1 \le a_i \le 1$）。在该模型下，任意两变量的相关系数可由因子加载计算得出：
$$
Corr(U_i, U_j) = Cov(U_i, U_j) = a_i a_j \,,
$$ 
因为 $Cov(U_i, U_j) = a_i a_j\,Var(F) + 0 = a_i a_j$（公共因子部分贡献相关，特异部分独立无协方差）。单因子模型将原本 $N(N-1)/2$ 个相关参数简化为 $N$ 个因子加载参数 $\{a_i\}$。
<!-- bilingual-en:start -->
**One-factor model:** Assume there is one common factor $F$ and an independent idiosyncratic factor $Z_i$ for each variable. Let $U_i$ be the standardized $i$th variable, with mean 0 and variance 1. The model is given above, where $F \sim N(0,1)$, the $Z_i \sim N(0,1)$ are mutually independent and independent of $F$, and $a_i$ is variable $i$'s loading on the common factor, with $-1 \le a_i \le 1$. For any two variables, the correlation is the product of their loadings because $Cov(U_i, U_j) = a_i a_j\,Var(F) + 0 = a_i a_j$: the common factor creates covariance, whereas the idiosyncratic components do not. The model therefore replaces $N(N-1)/2$ pairwise correlations with $N$ loadings $\{a_i\}$.
<!-- bilingual-en:end -->

## 2.2 多因子模型
<!-- bilingual-en:start -->
*2.2 Multi-Factor Model*
<!-- bilingual-en:end -->

**多因子模型：**可以推广到 $M$ 个因子。假设有因子 $F_1,\dots,F_M$ 彼此独立且均为 $N(0,1)$，每个变量 $U_i$ 有对应的加载向量 $(a_{i1}, a_{i2}, \dots, a_{iM})$，则：
$$
U_i = a_{i1}F_1 + a_{i2}F_2 + \cdots + a_{iM}F_M \;+\; \sqrt{\,1 - \sum_{m=1}^M a_{im}^2\,}\;Z_i \,.
$$ 
在保证 $1 - \sum_{m}a_{im}^2 \ge 0$ 的前提下，每个 $U_i$ 方差仍为1。任意两变量的相关系数是各自对公共因子加载的**逐因子乘积之和**：
$$
Corr(U_i, U_j) = \sum_{m=1}^M a_{im}\,a_{jm} \,. 
$$ 
例如，在两因子模型下 $Corr(U_i, U_j) = a_{i1}a_{j1} + a_{i2}a_{j2}$。单因子模型是 $M=1$ 的特例。
<!-- bilingual-en:start -->
**Multi-factor model:** The one-factor model extends naturally to $M$ factors. Suppose $F_1,\dots,F_M$ are mutually independent $N(0,1)$ factors, and variable $U_i$ has loading vector $(a_{i1}, a_{i2}, \dots, a_{iM})$. Provided that $1 - \sum_{m}a_{im}^2 \ge 0$, each $U_i$ retains unit variance. The correlation between any two variables is the **sum of the pairwise products of their loadings on each common factor**. Thus, in a two-factor model, $Corr(U_i, U_j) = a_{i1}a_{j1} + a_{i2}a_{j2}$; the one-factor model is the special case $M=1$.
<!-- bilingual-en:end -->

# 3. Gaussian Copula 建模
<!-- bilingual-en:start -->
*3. Gaussian Copula Modeling*
<!-- bilingual-en:end -->

在处理非正态变量时，[[Copula分解|Sklar 定理]]把边际分布与依赖结构分开。[[Gaussian Copula]]通过多元标准正态潜变量构造依赖，但不会把原变量的边际强行改成正态。连续边际下，其[[Copula拟合|拟合]]与模拟步骤如下：
<!-- bilingual-en:start -->
For non-normal variables, [[Copula分解|Sklar's theorem]] separates marginal distributions from dependence. A [[Gaussian Copula]] supplies dependence through multivariate standard-normal latent variables without forcing the original marginals to be normal. With continuous marginals, [[Copula拟合|fitting]] and simulation proceed as follows:
<!-- bilingual-en:end -->

1. **边际分布估计：**首先针对每个变量估计其边际分布 $F_{V_i}(v)$（累积分布函数），例如通过历史数据拟合得到。
2. **概率积分变换（PIT）：**对连续边际，$Q_i=F_{V_i}(V_i)$服从$U(0,1)$，再令$Z_i=\Phi^{-1}(Q_i)$，便得到标准正态边际。若$V_i$离散，$F_{V_i}(V_i)$一般不服从连续均匀分布，Copula在跳点之间也不唯一；此时须明确使用随机化PIT、潜变量阈值模型或其他离散Copula约定，不能直接声称$Z_i$标准正态。
3. **估计依赖：**用变换后的$Z_i$估计一个合法的相关矩阵$R$，并检查$R$至少PSD；需要标准Cholesky时还要PD。$R$是Gaussian-Copula潜变量的相关参数，不等于原变量在任意边际下的Pearson相关。
4. **模拟并逆变换：**先抽取$Z\sim N(0,R)$，再令$Q_i=\Phi(Z_i)$和$V_i=F_{V_i}^{-1}(Q_i)$。于是
   $$
   P(V_1\le v_1,\dots,V_n\le v_n)
   =\Phi_R\!\left(\Phi^{-1}(F_{V_1}(v_1)),\dots,\Phi^{-1}(F_{V_n}(v_n))\right),
   $$
   其中$\Phi_R$是相关矩阵为$R$的多元标准正态CDF。
<!-- bilingual-en:start -->

&nbsp;
**1.** **Estimate the marginal distributions:** Estimate each variable's marginal cumulative distribution function $F_{V_i}(v)$, for example from historical data.<br>
**2.** **Apply the probability-integral transform (PIT):** For a continuous marginal, $Q_i=F_{V_i}(V_i)$ is uniform and $Z_i=\Phi^{-1}(Q_i)$ is standard normal. For a discrete marginal, the ordinary PIT is not continuous uniform and the Copula is non-unique between jumps; use a stated randomized PIT, latent-threshold model, or another discrete convention.<br>
**3.** **Estimate dependence:** Estimate a valid latent correlation matrix $R$ from the transformed $Z_i$. It must be PSD, and standard Cholesky requires PD. This latent parameter is not generally the original variables' Pearson correlation.<br>
**4.** **Simulate and invert:** Draw $Z\sim N(0,R)$, set $Q_i=\Phi(Z_i)$, and return to the original scale with $V_i=F_{V_i}^{-1}(Q_i)$. The displayed multivariate-normal CDF then defines the joint distribution while preserving the chosen marginals.<br>
<!-- bilingual-en:end -->

简单说，连续数据拟合时先做$V_i\to Q_i=F_i(V_i)\to Z_i=\Phi^{-1}(Q_i)$，模拟时按$Z\sim N(0,R)\to Q_i=\Phi(Z_i)\to V_i=F_i^{-1}(Q_i)$反向走完整条链。离散变量不能未经处理套用连续PIT结论。
<!-- bilingual-en:start -->
In short, continuous-data fitting follows $V_i\to Q_i=F_i(V_i)\to Z_i=\Phi^{-1}(Q_i)$, whereas simulation follows the reverse chain $Z\sim N(0,R)\to Q_i=\Phi(Z_i)\to V_i=F_i^{-1}(Q_i)$. Discrete variables need an explicit treatment rather than the continuous-PIT claim.
<!-- bilingual-en:end -->

**模拟考题：**有两个边际连续但非正态的风险因子$V_1$和$V_2$，我们希望用Gaussian Copula建立联合分布。请写出拟合依赖并模拟新样本的基本步骤；若边际离散，还需说明什么会改变。
<!-- bilingual-en:start -->
**Practice question:** Two risk factors $V_1$ and $V_2$ have continuous, non-normal marginals. State the steps for fitting a Gaussian Copula and simulating new observations, and explain what changes for discrete marginals.
<!-- bilingual-en:end -->

**详细解答：**可以按照以下步骤：
1. **确定边际分布：**分别确定 $V_1$ 和 $V_2$ 的边际分布 $F_{V_1}(x)$ 和 $F_{V_2}(y)$（例如通过数据拟合出各自的分布类型和参数）。
2. **转换到标准正态空间：**令$q_i=F_{V_i}(v_i)$、$z_i=\Phi^{-1}(q_i)$。连续且模型正确时，$q_i$为均匀边际、$z_i$为标准正态边际。
3. **估计相关结构：**从$(z_1,z_2)$估计$\rho$并检查相关矩阵合法；$\rho$是潜在正态空间参数。
4. **模拟：**抽取相关系数为$\rho$的$(Z_1,Z_2)$，再令$V_i=F_{V_i}^{-1}(\Phi(Z_i))$。联合CDF为
   $$
   P(V_1\le x,V_2\le y)=\Phi_{2,\rho}\!\left(\Phi^{-1}(F_{V_1}(x)),\Phi^{-1}(F_{V_2}(y))\right).
   $$
   离散边际的普通PIT不均匀且Copula不唯一，必须另行声明随机化或潜变量约定。
<!-- bilingual-en:start -->
**Detailed answer:**
**1.** **Determine the marginals:** Estimate $F_{V_1}(x)$ and $F_{V_2}(y)$, including their distributional forms and parameters.<br>
**2.** **Transform to normal space:** Set $q_i=F_{V_i}(v_i)$ and $z_i=\Phi^{-1}(q_i)$. Under continuous, correctly specified marginals, the transformed observations have standard-normal marginals.<br>
**3.** **Estimate dependence:** Estimate $\rho$ from $(z_1,z_2)$ and check that the resulting matrix is valid. This is a latent-normal parameter.<br>
**4.** **Simulate:** Draw a bivariate standard-normal vector with correlation $\rho$ and set $V_i=F_{V_i}^{-1}(\Phi(Z_i))$. The displayed bivariate-normal CDF gives the joint law. For discrete marginals, the ordinary PIT is not uniform and the Copula is non-unique, so a randomized or latent-variable convention must be stated.
<!-- bilingual-en:end -->

## 3.1Copula 函数的定义与代数表达
<!-- bilingual-en:start -->
*3.1 Definition and Algebraic Form of a Copula*
<!-- bilingual-en:end -->
[[Copula|Copula函数]]描述边际分布之外的依赖结构。[[Copula分解|Sklar 定理]]说明：任意二维联合分布函数 $F_{X,Y}$ 都存在一个Copula $C$，使得
$$
F_{X,Y}(x,y) = C\!\big(F_X(x),\;F_Y(y)\big)\,,
$$
其中 $C$ 的两个边际都是 $U(0,1)$。若 $F_X,F_Y$ 都连续，则 $C$ 在 $[0,1]^2$ 上唯一；若存在离散或混合边际，$C$ 只在 $\operatorname{Ran}(F_X)\times\operatorname{Ran}(F_Y)$ 上由联合分布确定，向整个单位方形的延拓一般不唯一。因此“联合分布已确定”不等于“离散边际下Copula表示唯一”，估计与解释时必须说明约定。
<!-- bilingual-en:start -->
A [[Copula|Copula function]] describes dependence separately from the marginal distributions. [[Copula分解|Sklar's theorem]] states that for any bivariate joint CDF $F_{X,Y}$ there exists a Copula $C$ such that the displayed decomposition holds, with uniform marginals for $C$. If both marginals are continuous, $C$ is unique on $[0,1]^2$. With a discrete or mixed marginal, it is determined only on $\operatorname{Ran}(F_X)\times\operatorname{Ran}(F_Y)$ and its extension to the whole unit square is generally non-unique. Estimation and interpretation must therefore state the convention used for discrete data.
<!-- bilingual-en:end -->

对于[[Gaussian Copula]]而言，有显式的代数表达形式。以二维为例，假设 $X$ 和 $Y$ 边际分布分别为 $G_1(x)$ 和 $G_2(y)$。高斯Copula下的联合分布函数为：
$$ 
F_{X,Y}(x,y) \;=\; \Phi_{2,\rho}\!\Big(\Phi^{-1}\big(G_1(x)\big)\,,\;\Phi^{-1}\big(G_2(y)\big)\Big)\,,
$$ 
其中 $\Phi^{-1}$ 是标准正态分布的反函数，$\Phi_{2,\rho}$ 表示相关系数为 $\rho$ 的二维正态分布的累积函数。等式右边其实就是Copula函数：
$$ 
C(u_1, u_2) = \Phi_{2,\rho}\!\big(\Phi^{-1}(u_1),\; \Phi^{-1}(u_2)\big)\,, \qquad 0 \le u_1,u_2 \le 1\,.
$$ 
可以看出，Copula函数将边际分布的概率值 $(u_1,u_2)$ 通过正态分位数映射，再代入相关正态分布的CDF，从而得到联合概率。对任意给定的 $\rho$，Gaussian Copula 都保证 $F_X$ 和 $F_Y$ 保持各自不变，仅通过 $\rho$ 来影响变量间的关联形式。
<!-- bilingual-en:start -->
The [[Gaussian Copula]] has an explicit algebraic form. In two dimensions, suppose $X$ and $Y$ have marginal CDFs $G_1(x)$ and $G_2(y)$. Their Gaussian-Copula joint CDF is the expression shown above, where $\Phi^{-1}$ is the standard normal quantile function and $\Phi_{2,\rho}$ is the bivariate standard normal CDF with correlation $\rho$. Thus the Copula maps marginal probabilities $(u_1,u_2)$ into normal quantiles and evaluates their joint normal probability. For any fixed $\rho$, the marginals $F_X$ and $F_Y$ remain unchanged; $\rho$ affects only their dependence.
<!-- bilingual-en:end -->

**模拟考题：**设 $V_1$ 和 $V_2$ 的边际分布函数分别为 $G_1(v_1)$ 和 $G_2(v_2)$。请写出高斯 Copula 下它们联合分布函数的表达式，并指出其中的 Copula 函数形式。
<!-- bilingual-en:start -->
**Practice question:** Let the marginal distribution functions of $V_1$ and $V_2$ be $G_1(v_1)$ and $G_2(v_2)$. Write their joint distribution under a Gaussian Copula and identify the Copula function.
<!-- bilingual-en:end -->

**详细解答：**高斯Copula下的联合分布由边际分布和标准正态Copula组成：
$$ 
F_{V_1,V_2}(v_1, v_2) = \Phi_{2,\rho}\Big(\,\Phi^{-1}\!\big(G_1(v_1)\big)\,,\;\Phi^{-1}\!\big(G_2(v_2)\big)\Big)\,. 
$$ 
其中 $\Phi_{2,\rho}$ 是参数为 $\rho$ 的二维标准正态分布函数，$\Phi^{-1}$ 将边际分布概率映射为正态值。这一定义可等价于Copula函数：
$$ 
C_{\rho}(u_1, u_2) = \Phi_{2,\rho}\!\big(\Phi^{-1}(u_1),\,\Phi^{-1}(u_2)\big)\,,
$$ 
使得 $F_{V_1,V_2}(v_1,v_2) = C_{\rho}\big(G_1(v_1),\,G_2(v_2)\big)$。
<!-- bilingual-en:start -->
**Detailed answer:** Under a Gaussian Copula, the joint distribution combines the two marginals with a standard normal Copula. Here, $\Phi_{2,\rho}$ is the bivariate standard normal CDF with correlation parameter $\rho$, and $\Phi^{-1}$ maps each marginal probability to a normal quantile. Equivalently, the Copula is $C_\rho(u_1,u_2)=\Phi_{2,\rho}(\Phi^{-1}(u_1),\Phi^{-1}(u_2))$, so that $F_{V_1,V_2}(v_1,v_2) = C_{\rho}\big(G_1(v_1),\,G_2(v_2)\big)$.
<!-- bilingual-en:end -->

## 3.2Copula 在信贷组合违约率建模中的应用
<!-- bilingual-en:start -->
*3.2 Using a Copula to Model Credit-Portfolio Default Rates*
<!-- bilingual-en:end -->


Copula 方法在信贷风险中广泛用于构建**贷款组合违约分布**。最经典的是 **单因子高斯 Copula 模型**，如新巴塞尔协议中的资产组合模型。其思想是：假设每个借款人 $i$ 有一个潜在的标准正态变量 $U_i$（可视为资产价值标准化指标），并引入一个公共因子 $F \sim N(0,1)$ 表征宏观经济状况，设：
$$ 
U_i = \sqrt{\rho}\;F + \sqrt{\,1-\rho\,}\;Z_i \,,
$$ 
其中 $\rho$ 是同质的**潜在资产相关参数**，$Z_i\sim N(0,1)$ 是借款人 $i$ 的独立特有风险。在这个单因子结构中，任意两家不同公司 $i,j$ 的潜变量相关系数 $\operatorname{Corr}(U_i,U_j)$ 均为 $\rho$；它不是观测违约指标的普通Pearson相关。
<!-- bilingual-en:start -->
Copulas are widely used in credit risk to construct the **distribution of defaults in a loan portfolio**. The standard example is the **one-factor Gaussian Copula**, including the asset-value model underlying Basel capital formulas. Each borrower $i$ is assigned a latent standard normal variable $U_i$, interpreted as a standardized asset-value index. A common factor $F \sim N(0,1)$ represents macroeconomic conditions, while $Z_i \sim N(0,1)$ captures borrower-specific risk. Under the homogeneous specification shown above, the loading is $\sqrt{\rho}$, so any two latent variables $U_i$ and $U_j$ have correlation $\rho$.
<!-- bilingual-en:end -->

将违约事件与 $U_i$ 挂钩：设第 $i$ 个借款人的年度违约概率（PD）为 $p_i$。在模型中，这等价于定义一个违约临界值 $\theta_i = \Phi^{-1}(p_i)$，并假定：
$$ 
\text{若 } U_i < \theta_i \text{，则发生违约。}
$$ 
在此框架下，可以计算组合违约的分布。例如，对于大型均质组合（所有贷款PD相同为 $p$），**条件违约概率**（给定因子 $F=f$）为：
$$ 
P(\text{违约}|F=f) = \Phi\!\Big(\frac{\theta - \sqrt{\rho}\,f}{\sqrt{\,1-\rho\,}}\Big)\,,
$$ 
其中 $\theta = \Phi^{-1}(p)$。这表示在公共因子水准 $f$ 下，各贷款违约概率会随之变化：如果经济因子 $f$ 很低（不景气），条件违约率会上升，反之下降。由于 $F$ 本身服从 $N(0,1)$，可以进一步推导无条件的违约分布函数，即**组合违约率**（违约占比） $DR$ 的分布。事实上，当组合贷款数目 $M$ 很大时，$DR$ 近似等于给定 $F$ 时的违约概率，因此：
$$ 
P(DR \le x) = P\!\Big(\Phi\Big(\frac{\theta - \sqrt{\rho}\,F}{\sqrt{\,1-\rho\,}}\Big) \le x\Big)\,. 
$$ 
通过对 $F$ 积分（或等价变换），可得违约率 $DR$ 的分布形式（这就是 Vasicek 分布）。利用该分布，我们能够求出高置信水平下的极端违约情景等。
<!-- bilingual-en:start -->
Default is linked to the latent variable $U_i$. If borrower $i$ has annual probability of default (PD) $p_i$, define the threshold $\theta_i = \Phi^{-1}(p_i)$ and treat the borrower as defaulting when its latent variable falls below that threshold. For a large homogeneous portfolio with common PD $p$, the **conditional probability of default** given $F=f$ is the expression shown above, where $\theta = \Phi^{-1}(p)$. A low value of the common factor represents adverse economic conditions and raises conditional default probability; a high value lowers it. As the number of loans $M$ becomes large, idiosyncratic risk diversifies away and the realized default rate $DR$ converges to that conditional probability. Integrating over $F$, or applying an equivalent change of variables, gives the Vasicek distribution for $DR$ and hence high-confidence default-rate quantiles.
<!-- bilingual-en:end -->

**模拟考题：**假设有两个公司，年违约概率均为2%（即 $p=0.02$）。利用单因子高斯Copula模型，并设两家公司潜变量的相关参数 $\rho=0.1$，求它们在同一年内**同时违约**的概率。
<!-- bilingual-en:start -->
**Practice question:** Two companies each have an annual default probability of 2%, so $p=0.02$. Under a one-factor Gaussian Copula with latent-variable correlation $\rho=0.1$, calculate the probability that both companies **default in the same year**.
<!-- bilingual-en:end -->

**详细解答：**两家公司同时违约的概率可以通过Copula计算，即：
$$ 
P(\text{两家公司都违约}) = C_{\rho}(p,\;p) \;=\; \Phi_{2,\;0.1}\Big(\Phi^{-1}(0.02),\;\Phi^{-1}(0.02)\Big)\,. 
$$ 
将违约概率转换为正态临界值：$\Phi^{-1}(0.02)\approx-2.05374891$。于是
$$ 
P(\text{同时违约})=\Phi_{2,0.1}(-2.05374891,-2.05374891).
$$ 
这个值需要通过二维正态积分计算：
$$
\Phi_{2,0.1}(-2.05374891,-2.05374891)\approx0.000687984=0.0687984\%.
$$
独立时同时违约概率为 $0.02\times0.02=0.0004=0.04\%$。在这个参数设定下，正的潜变量相关性把共同违约概率从0.04%提高到约0.0688%；这是一年、两家公司、给定高斯Copula下的结果，不应外推成任意组合的固定倍数。
<!-- bilingual-en:start -->
**Detailed answer:** Joint default occurs when both latent variables fall below $\Phi^{-1}(0.02)\approx-2.05374891$. Numerical evaluation gives $\Phi_{2,0.1}(-2.05374891,-2.05374891)\approx0.000687984$, or $0.0687984\%$. Under independence the probability is $0.02\times0.02=0.0004$, or $0.04\%$. For this one-year, two-borrower Gaussian-Copula example, positive latent correlation therefore raises joint-default probability from 0.04% to about 0.0688%; it is not a universal multiplier for other portfolios.
<!-- bilingual-en:end -->

## 3.3最坏违约率计算与 VaR 推导
<!-- bilingual-en:start -->
*3.3 Worst-Case Default Rate and the Derivation of VaR*
<!-- bilingual-en:end -->


在信贷组合风险管理中，**最坏违约率**（Worst Case Default Rate, **WCDR**）通常是组合违约率分布在置信水平 $\alpha$ 下的**高分位点**，不是绝对最大值；模型认为违约率超过它的概率为 $1-\alpha$。99.9%是常见的监管置信水平之一，但具体资本口径还必须遵守适用规则。单因子高斯Copula给出该分位点的解析式。
<!-- bilingual-en:start -->
In credit-portfolio risk management, the **worst-case default rate (WCDR)** is a high quantile of the portfolio default-rate distribution. It is not an absolute maximum; it is the default rate that is exceeded only with probability $1-\alpha$ at confidence level $\alpha$. Regulators often use 99.9%, and the resulting WCDR feeds into credit-risk capital calculations. The one-factor Gaussian Copula yields a closed-form expression.
<!-- bilingual-en:end -->

对于大型均质组合（违约概率均为 $p$，相关系数 $\rho$），一年期违约率 $DR$ 在模型下满足： 
$$ 
DR = \Phi\Big(\frac{\Phi^{-1}(p) - \sqrt{\rho}\,F}{\sqrt{\,1-\rho\,}}\Big)\,,
$$ 
其中 $F \sim N(0,1)$。条件违约率随 $F$ 下降而上升，因此置信水平 $\alpha$ 的违约率分位点对应**不利因子状态** $F=-z_\alpha$，其中 $z_\alpha=\Phi^{-1}(\alpha)$；绝不能把 $F=+z_\alpha$ 说成坏状态。设 $\theta=\Phi^{-1}(p)$，则
$$ 
x_\alpha = \Phi\!\Big(\frac{\theta + \sqrt{\rho}\,z_\alpha}{\sqrt{\,1-\rho\,}}\Big) \,,
$$ 
这给出了WCDR的计算公式。
<!-- bilingual-en:start -->
For a large homogeneous portfolio with common default probability $p$ and correlation coefficient $\rho$, the one-year default rate $DR$ is driven by the common factor $F \sim N(0,1)$. At confidence level $\alpha$, such as $\alpha=99.9\%$, define the default-rate quantile as $x_\alpha = \text{WCDR}( \alpha)$ and the standard normal quantile as $z_\alpha = \Phi^{-1}(\alpha)$. Because worse economic states correspond to low values of $F$, the adverse state is $F=-z_\alpha$, not $F=+z_\alpha$. Let $\theta = \Phi^{-1}(p)$. Then
$$
x_\alpha
=
\Phi\!\left(
\frac{\Phi^{-1}(p)+\sqrt{\rho}\,\Phi^{-1}(\alpha)}
{\sqrt{1-\rho}}
\right).
$$
This is the WCDR formula.
<!-- bilingual-en:end -->

计算**风险价值（VaR）**需要将WCDR转换为实际损失金额。例如，当有 $M$ 笔贷款总额 $L$，每笔敞口相同且违约损失率（损失率 = 1-回收率，即LGD）为 $\lambda$，则有：
$$ 
\text{VaR}_{\alpha} = L \times \lambda \times x_\alpha \,,
$$ 
这是模型中的一年损失**分位点**，不是一年内的绝对最大损失。若资本定义为非预期损失，还要按适用口径从该损失分位点中扣除预期损失；不能仅凭这条教学公式声称满足监管资本要求。
<!-- bilingual-en:start -->
To compute **value at risk (VaR)**, convert WCDR into a loss amount. If a homogeneous portfolio has total exposure $L$ and loss given default $\lambda=1-\text{recovery rate}$, then the confidence-level loss is WCDR multiplied by total exposure and LGD. In other words, the one-year loss quantile at confidence level $\alpha$ equals the worst-case default fraction times the portfolio exposure times the loss rate.
<!-- bilingual-en:end -->

**模拟考题：**某银行持有价值\$100百万的均质零售贷款组合，每笔贷款的年违约概率为2%，平均回收率为60%（故LGD为 $\lambda=40\%$）。假设单因子高斯Copula的潜在资产相关参数为 $\rho=0.1$。请计算该组合一年期的**99.9%违约率分位点**以及**99.9%损失VaR**。
<!-- bilingual-en:start -->
**Practice question:** A bank holds a homogeneous retail-loan portfolio worth \$100 million. Each loan has a 2% annual probability of default, and the average recovery rate is 60%, so $\lambda=40\%$. Assume a one-factor Gaussian Copula with latent asset-correlation parameter $\rho=0.1$. Calculate the portfolio's one-year **99.9th-percentile default rate** and its **99.9% loss VaR**.
<!-- bilingual-en:end -->

**详细解答：**参数为 $p=0.02$、LGD $\lambda=40\%$、$\alpha=0.999$、$\Phi^{-1}(0.999)\approx3.09023231$、$\Phi^{-1}(0.02)\approx-2.05374891$。应用WCDR公式：
$$ 
x_{99.9\%}=\Phi\!\left(
\frac{-2.05374891+\sqrt{0.1}\times3.09023231}{\sqrt{0.9}}
\right).
$$ 
计算分步如下：
- $\sqrt{0.1}\times3.09023231\approx0.97721726$，分子约为 $-1.07653165$。
- $\sqrt{0.9}\approx0.94868330$，标准化参数约为 $-1.13476400$。
- $\Phi(-1.13476400)\approx0.12823711$。
<!-- bilingual-en:start -->
**Detailed answer:** The parameters are $p=0.02$, LGD $\lambda=40\%$, $\alpha=0.999$, $\Phi^{-1}(0.999)\approx3.09023231$, and $\Phi^{-1}(0.02)\approx-2.05374891$. Then $\sqrt{0.1}\times3.09023231\approx0.97721726$, the numerator is about $-1.07653165$, and division by $\sqrt{0.9}\approx0.94868330$ gives $-1.13476400$. Hence $\Phi(-1.13476400)\approx0.12823711$.
<!-- bilingual-en:end -->

因此，**99.9%违约率分位点**为 $x_{99.9\%}\approx0.128237=12.8237\%$；模型给出的超越概率为0.1%。对应的一年**99.9%损失VaR**为
$$ 
\text{VaR}_{99.9\%}
=100\,\text{百万}\times12.8237\%\times40\%
\approx5.12948\,\text{百万美元}.
$$ 
换言之，该模型下的一年99.9%损失分位点约为\$512.95万，占组合的5.12948%；它不是绝对最大可能损失，也尚未扣除预期损失。
<!-- bilingual-en:start -->
Therefore, the **99.9th-percentile default rate** is $x_{99.9\%}\approx0.128237=12.8237\%$, with model-implied exceedance probability 0.1%. Multiplying by \$100 million and a 40% LGD gives a one-year **99.9% loss VaR** of approximately \$5.12948 million, or 5.12948% of portfolio value. It is not an absolute maximum and has not been reduced by expected loss.
<!-- bilingual-en:end -->

## Copula相关性与尾部风险
<!-- bilingual-en:start -->
*Copula Correlation and Tail Risk*
<!-- bilingual-en:end -->
[[Gaussian Copula]]模型用潜在正态相关矩阵描述依赖，但这不等于可由原变量的Pearson相关系数完整刻画。它在描述[[尾部依赖|尾部依赖（tail dependence）]]方面存在局限；Gaussian 与 t 模型的具体比较见 [[Gaussian与t Copula尾部]]。上尾依赖系数定义为
$$ 
\lambda_U = \lim_{q \to 1^-} P\big(Y > F_Y^{-1}(q) \,\big|\, X > F_X^{-1}(q)\big) \,,
$$ 
表示当 $X$ 处于极高分位时 $Y$ 也极端偏大的概率（下尾类似定义）。
<!-- bilingual-en:start -->
The Gaussian Copula uses a latent-normal correlation matrix to parameterize dependence; this is not the same as saying that the original variables' Pearson correlations fully determine their joint law. It is restrictive when modeling **tail dependence**, the tendency for variables to become extreme together. Upper-tail dependence is the limiting conditional probability that $Y$ is also extremely high given that $X$ is increasingly extreme; lower-tail dependence is defined analogously.
<!-- bilingual-en:end -->

对于非退化的二维高斯Copula（$-1<\rho<1$），上、下尾依赖系数都为0。这**不表示有限阈值下的共同极端事件不可能或一定很少**；它只表示当阈值趋向分布端点时，条件共同极端概率趋于0。即使潜在相关较高，高斯依赖仍可能低估危机中观察到的极端损失聚集。
<!-- bilingual-en:start -->
For a non-degenerate bivariate Gaussian Copula with $-1<\rho<1$, both upper- and lower-tail dependence coefficients are zero. This does not mean finite-threshold joint extremes are impossible or necessarily rare; it means that their limiting conditional probability vanishes as the threshold moves to the endpoint. Even with high latent correlation, Gaussian dependence can therefore understate the clustering of rare losses observed when many assets fall together during a crisis.
<!-- bilingual-en:end -->

**尾部风险**包括多个金融变量同时进入不利尾部的风险。假设某组合平均年违约概率为1%，十年中观察到一年违约率为3%；这个事实本身并不能证明高斯Copula错误，更不能说任意 $\rho$ 都无法产生该结果。正确做法是把该观测与给定 $p$、$\rho$、组合粒度和样本期下的预测分布比较；只有当尾部事件系统性地比模型预测更频繁或更严重，才构成模型尾部拟合不足的证据。
<!-- bilingual-en:start -->
**Tail risk** includes the risk that several financial variables enter adverse tails together. Suppose a portfolio has an average annual PD of 1% and records a 3% default rate in one year out of ten. That observation alone neither rejects a Gaussian Copula nor proves that no value of $\rho$ can produce it. It must be compared with the predictive distribution conditional on $p$, $\rho$, portfolio granularity, and sample length. Systematically more frequent or severe tail events than the model predicts would be evidence of deficient tail fit.
<!-- bilingual-en:end -->

一个候选替代是有限自由度的[[t Copula|$t$-Copula（Student's $t$ Copula）]]。标准构造不是简单地把高斯单因子模型中的公共因子改成 $t$ 而继续保留正态特有项；这种混合一般不产生标准 $t$-Copula。正确的椭圆 $t$-Copula 构造为：先取 $Z\sim N(0,R)$ 与独立的 $W\sim\chi^2_\nu$，令
$$
T=\frac{Z}{\sqrt{W/\nu}},\qquad U_i=t_\nu(T_i),\qquad X_i=F_i^{-1}(U_i),
$$
其中 $t_\nu$ 表示自由度为 $\nu$ 的一元Student-$t$分布函数。所有分量共享随机尺度 $W$，从而产生共同尾部。二维、$-1<\rho<1$、有限 $\nu$ 时，其对称尾依赖系数为
$$
\lambda_L=\lambda_U
=2t_{\nu+1}\!\left(-\sqrt{\frac{(\nu+1)(1-\rho)}{1+\rho}}\right)>0.
$$
自由度越小通常尾部越厚，但是否改善拟合必须用边际、整体依赖与尾部的样本外诊断分别验证。
<!-- bilingual-en:start -->
One candidate alternative is a finite-degrees-of-freedom [[t Copula|Student's $t$ Copula]]. Merely replacing the common Gaussian factor by a $t$ variable while retaining normal idiosyncratic terms does not generally create a standard $t$ Copula. The elliptical construction draws $Z\sim N(0,R)$ and an independent $W\sim\chi^2_\nu$, sets $T=Z/\sqrt{W/\nu}$, then $U_i=t_\nu(T_i)$ and $X_i=F_i^{-1}(U_i)$, where $t_\nu$ denotes the univariate Student-$t$ CDF. The common random scale $W$ creates joint tail behavior. In the bivariate case with finite $\nu$ and $-1<\rho<1$, the symmetric coefficient is $\lambda_L=\lambda_U=2t_{\nu+1}(-\sqrt{(\nu+1)(1-\rho)/(1+\rho)})>0$. Lower degrees of freedom usually mean heavier tails, but improvement must be demonstrated by separate out-of-sample checks of marginals, overall dependence, and tails.
<!-- bilingual-en:end -->

**模拟考题：**为何单因子高斯Copula模型可能低估信用组合的尾部风险？什么是尾部相关性？举例说明采用厚尾Copula（如 $t$-Copula）如何改进对尾部共同违约事件的拟合。
<!-- bilingual-en:start -->
**Practice question:** Why can a one-factor Gaussian Copula underestimate the tail risk of a credit portfolio? What is tail dependence? Explain how a heavy-tailed Copula, such as a $t$-Copula, can fit clustered default events more effectively.
<!-- bilingual-en:end -->

**详细解答：**单因子高斯Copula假定公共因子 $F$ 为正态，从而各违约事件通过对 $F$ 的共同线性暴露产生依赖。它在非退化情形下渐近尾部独立；若独立的样本外证据显示共同极端违约比给定参数下的预测分布更频繁或更严重，才说明模型低估了尾部风险。
<!-- bilingual-en:start -->
**Detailed answer:** A one-factor Gaussian Copula assumes a normal common factor $F$, so dependence among defaults arises through shared linear exposure to that factor. The non-degenerate model is asymptotically tail-independent. If independent out-of-sample evidence shows that joint extreme defaults occur more frequently or severely than the fitted predictive distribution allows, the model is understating tail risk.
<!-- bilingual-en:end -->

**尾部相关性**指变量在极端尾部同时发生极端变动的相关程度。非退化二维高斯Copula的尾部相关性为0（$-1<\rho<1$），意味着例如 $P(X$ 极端下跌 $\land Y$ 极端下跌$)$相对于单边极端事件的条件概率趋于零。现实中金融资产可能呈现更强的尾部共动，例如市场崩盘时多数资产一起下跌、经济萧条时多家公司一同违约。
<!-- bilingual-en:start -->
**Tail dependence** measures the limiting tendency of variables to enter the same extreme tail together. For a non-degenerate bivariate Gaussian Copula it is zero when $-1<\rho<1$. Thus the conditional probability of an extreme fall in $Y$, given an increasingly extreme fall in $X$, tends to zero as the threshold moves to the endpoint. Financial data may show stronger tail co-movement: many assets fall together in a crash, and many firms default together in a recession.
<!-- bilingual-en:end -->

采用有限自由度的 $t$-Copula 可以给出正的对称尾依赖。应使用上面的共同尺度多元 $t$ 构造，而不是只把公共因子换成 $t$、特有项仍保留正态。它可提高多个潜变量同时进入下尾的概率，因此是共同违约建模的候选；但“更厚尾”不自动等于“更正确”，仍须分别检验边际、依赖结构、尾部覆盖和样本外稳定性。1%平均PD与偶尔3%违约率只能作为待检验的尾部观测，不能单独识别Copula族或参数。
<!-- bilingual-en:start -->
A finite-$\nu$ $t$ Copula has positive symmetric tail dependence. It must use the common-scale multivariate-$t$ construction above, rather than a $t$ common factor mixed with normal idiosyncratic terms. This can raise the probability that several latent variables enter the lower tail together, making it a candidate for joint-default modeling. Heavier tails are not automatically a better model: marginals, dependence, tail coverage, and out-of-sample stability must be validated separately. An average PD of 1% with an occasional 3% default year is an observation to test, not enough by itself to identify the Copula family or parameters.
<!-- bilingual-en:end -->

## 边际 PD 与 Copula 依赖参数的极大似然估计
<!-- bilingual-en:start -->
*Maximum Likelihood Estimation of Marginal PD and the Copula Dependence Parameter*
<!-- bilingual-en:end -->
单因子 Vasicek 信贷模型同时包含边际**违约概率** $PD$ 与 Gaussian Copula 的**潜在资产相关参数** $\rho$；前者控制单名边际违约率，后者控制依赖。给定历史数据，可以用**极大似然估计（MLE）**联合估计这两个整体模型参数。
<!-- bilingual-en:start -->
The one-factor Vasicek credit model contains both a marginal **probability of default** $PD$ and the Gaussian-Copula **latent asset-correlation parameter** $\rho$. The former controls the single-name default margin and the latter controls dependence. Given historical observations, the two full-model parameters can be estimated jointly by **maximum likelihood estimation (MLE)**.
<!-- bilingual-en:end -->

以渐近均质组合为例，记 $p=PD$、$a=\Phi^{-1}(p)$，并假定 $0<\rho<1$。模型给出
$$
DR=\Phi\!\left(\frac{a-\sqrt{\rho}\,F}{\sqrt{1-\rho}}\right),
\qquad F\sim N(0,1).
$$
由于 $DR$ 随 $F$ 单调递减，对 $0<x<1$，Vasicek分布的CDF为
$$
G(x)=P(DR\le x)
=\Phi\!\left(\frac{\sqrt{1-\rho}\,\Phi^{-1}(x)-\Phi^{-1}(p)}{\sqrt{\rho}}\right).
$$
令括号内为 $h(x)$，其密度为
$$
g(x)=\phi\!\big(h(x)\big)
\sqrt{\frac{1-\rho}{\rho}}
\frac{1}{\phi\!\big(\Phi^{-1}(x)\big)}.
$$
观测违约率对应的系统因子反解是
$$
F_t=\frac{\Phi^{-1}(p)-\sqrt{1-\rho}\,\Phi^{-1}(DR_t)}{\sqrt{\rho}}.
$$
因此可用 $\ell(p,\rho)=\sum_t\log g(DR_t;p,\rho)$ 标定参数。这个连续、渐近Vasicek特例还有更直接的变换：
$$
Y_t=\Phi^{-1}(DR_t)\sim N\!\left(
\frac{\Phi^{-1}(p)}{\sqrt{1-\rho}},\frac{\rho}{1-\rho}
\right).
$$
若年度观测独立，则令 $\hat\mu=\bar Y$、$\hat v=T^{-1}\sum_t(Y_t-\bar Y)^2$，闭式MLE为
$$
\hat\rho=\frac{\hat v}{1+\hat v},
\qquad
\hat p=\Phi\!\big(\hat\mu\sqrt{1-\hat\rho}\big).
$$
对有限组合的离散违约数、年份相关、异质敞口或更一般Copula，这个闭式解不再适用，必须写出相应似然并做数值估计。
<!-- bilingual-en:start -->
For an asymptotic homogeneous portfolio, write $p=PD$, $a=\Phi^{-1}(p)$, and assume $0<\rho<1$. The default rate is $DR=\Phi((a-\sqrt{\rho}F)/\sqrt{1-\rho})$ with $F\sim N(0,1)$. Since $DR$ decreases in $F$, its CDF is $G(x)=\Phi((\sqrt{1-\rho}\,\Phi^{-1}(x)-\Phi^{-1}(p))/\sqrt{\rho})$. If the bracketed expression is $h(x)$, the density is $g(x)=\phi(h(x))\sqrt{(1-\rho)/\rho}/\phi(\Phi^{-1}(x))$, and the factor inversion is $F_t=[\Phi^{-1}(p)-\sqrt{1-\rho}\,\Phi^{-1}(DR_t)]/\sqrt{\rho}$. Thus one may maximize $\ell(p,\rho)=\sum_t\log g(DR_t;p,\rho)$.

This continuous asymptotic Vasicek case also has a closed-form transformed-normal MLE. Since $Y_t=\Phi^{-1}(DR_t)\sim N(\Phi^{-1}(p)/\sqrt{1-\rho},\rho/(1-\rho))$, independent annual observations give $\hat\mu=\bar Y$, $\hat v=T^{-1}\sum_t(Y_t-\bar Y)^2$, $\hat\rho=\hat v/(1+\hat v)$, and $\hat p=\Phi(\hat\mu\sqrt{1-\hat\rho})$. Finite-portfolio default counts, serial dependence, heterogeneous exposures, or more general Copulas require the appropriate likelihood and usually numerical estimation.
<!-- bilingual-en:end -->

估计后不能只看样本内似然。应按照[[Copula验证]]分别检验边际分布、整体依赖、上下尾覆盖、参数稳定性与样本外表现；较高似然不等于联合模型正确，也不等于资本模型已获批准。
<!-- bilingual-en:start -->
After estimation, in-sample likelihood is not enough. [[Copula验证|Copula validation]] should separately assess marginal fit, overall dependence, upper- and lower-tail coverage, parameter stability, and out-of-sample performance. A higher likelihood neither proves that the joint model is correct nor constitutes regulatory model approval.
<!-- bilingual-en:end -->

**模拟考题：**给定过去5年的某贷款组合违约率数据：$ \{2.1\%,\;0.5\%,\;1.4\%,\;3.0\%,\;0.8\%\}$，试说明如何利用极大似然估计来推断单因子 Vasicek 信贷模型的边际违约概率 $PD$ 与 Gaussian Copula 依赖参数 $\rho$。简单描述估计步骤并指出计算中涉及的关键公式。
<!-- bilingual-en:start -->
**Practice question:** The annual default rates of a loan portfolio over the past five years are $ \{2.1\%,\;0.5\%,\;1.4\%,\;3.0\%,\;0.8\%\}$. Explain how maximum likelihood can estimate the marginal PD and Gaussian-Copula dependence parameter $\rho$ of the one-factor Vasicek credit model. State the estimation steps and the key formulas.
<!-- bilingual-en:end -->

**详细解答：**先声明这是连续、渐近均质Vasicek近似，并暂时把五个年度观测视为相互独立。CDF与因子反解分别为
$$
G(x)=\Phi\!\left(
\frac{\sqrt{1-\rho}\,\Phi^{-1}(x)-\Phi^{-1}(p)}{\sqrt{\rho}}
\right),
\qquad
F_t=\frac{\Phi^{-1}(p)-\sqrt{1-\rho}\,\Phi^{-1}(DR_t)}{\sqrt{\rho}}.
$$
可以把每个 $DR_t$ 代入上述密度 $g$，最大化
$$
\ell(p,\rho)=\sum_{t=1}^{5}\log g(DR_t;p,\rho),
$$
约束为 $0<p<1$、$0<\rho<1$。在这个特例中，更简便的是计算 $Y_t=\Phi^{-1}(DR_t)$，再用
$$
\hat\mu=\bar Y,\qquad
\hat v=\frac{1}{5}\sum_{t=1}^{5}(Y_t-\bar Y)^2,\qquad
\hat\rho=\frac{\hat v}{1+\hat v},\qquad
\hat p=\Phi\!\big(\hat\mu\sqrt{1-\hat\rho}\big).
$$
这里方差除以5才是正态似然的MLE；除以4得到的是无偏样本方差，不是本题的MLE。把估计值代回WCDR公式可得到模型分位点，但实际应用还要检查年度相关、组合有限粒度、参数不确定性和样本外尾部覆盖。
<!-- bilingual-en:start -->
**Detailed answer:** State first that this is the continuous asymptotic-homogeneous Vasicek approximation and provisionally treat the five annual observations as independent. The CDF and factor inversion are $G(x)=\Phi((\sqrt{1-\rho}\,\Phi^{-1}(x)-\Phi^{-1}(p))/\sqrt{\rho})$ and $F_t=[\Phi^{-1}(p)-\sqrt{1-\rho}\,\Phi^{-1}(DR_t)]/\sqrt{\rho}$. One route is to maximize $\ell(p,\rho)=\sum_{t=1}^{5}\log g(DR_t;p,\rho)$ subject to $0<p<1$ and $0<\rho<1$.

For this special case, transform $Y_t=\Phi^{-1}(DR_t)$ and use $\hat\mu=\bar Y$, $\hat v=5^{-1}\sum_t(Y_t-\bar Y)^2$, $\hat\rho=\hat v/(1+\hat v)$, and $\hat p=\Phi(\hat\mu\sqrt{1-\hat\rho})$. Dividing by five gives the normal-likelihood MLE variance; dividing by four gives the unbiased sample variance, not the MLE. Fitted parameters can be inserted into the WCDR formula, but application still requires checks for serial dependence, finite-portfolio granularity, parameter uncertainty, and out-of-sample tail coverage.
<!-- bilingual-en:end -->

以上步骤概括了利用 MLE 联合标定整体单因子信贷模型参数的过程。在实际计算中应使用软件，以确保计算精度和搜索效率。
<!-- bilingual-en:start -->
These steps summarize how MLE jointly calibrates the full one-factor credit model. In an actual application, numerical software should be used both for accuracy and for efficient constrained optimization.
<!-- bilingual-en:end -->

# 作业
<!-- bilingual-en:start -->
*Homework*
<!-- bilingual-en:end -->

## 11.6

>[!question] 
>资产X和Y当前的日波动率分别为1.0%和1.2%，上个交易日收盘价分别为30美元和50美元，收益相关系数为0.5。采用[[GARCH一步方差预测|GARCH(1,1)]]式更新，令 $\alpha=0.04$、$\beta=0.94$；协方差递推的 $\omega=0.000001$，两个方差递推的 $\omega=0.000003$。若今天收盘价分别为31美元和51美元，最新相关系数是多少？
><!-- bilingual-en:start -->
>Assume that the current daily volatilities of assets X and Y are 1.0% and 1.2%, their previous closing prices were \$30 and \$50, and their return correlation was 0.5. Use a [[GARCH一步方差预测|GARCH(1,1)]]-style update with $\alpha=0.04$ and $\beta=0.94$, taking $\omega=0.000001$ for covariance and $\omega=0.000003$ for each variance. If today's closing prices are \$31 and \$51, what is the updated correlation estimate?
><!-- bilingual-en:end -->

逻辑:
1. 老方差和今日平方收益推新方差
2. 老协方差和今日收益乘积推新协方差
3. 新协方差除以新波动率乘积得到新相关系数
<!-- bilingual-en:start -->
Logic:
**1.** Update each variance from the previous volatility and today's return.<br>
**2.** Update covariance from the previous covariance and today's cross-product of returns.<br>
**3.** Divide the updated covariance by the product of the updated volatilities.<br>
<!-- bilingual-en:end -->

- 资产$X$昨收$30$美元，今收$31$美元
    $$
    r_X = \ln\left(\frac{31}{30}\right) \approx 0.03279
    $$
- 资产$Y$昨收$50$美元，今收$51$美元
    $$
    r_Y = \ln\left(\frac{51}{50}\right) \approx 0.01980
    $$
<!-- bilingual-en:start -->
- Asset $X$ closed at \$30 yesterday and \$31 today.
- Asset $Y$ closed at \$50 yesterday and \$51 today.
<!-- bilingual-en:end -->

对$X$（参数$\omega=0.000003,\ \alpha=0.04,\ \beta=0.94$，昨日$\sigma_{X,\text{old}}=1\%=0.01$）：
$$
\sigma_{X,\text{new}}^2 = 0.000003 + 0.04 \times (0.03279)^2 + 0.94 \times (0.01)^2
$$
$$
\sigma_{X,\text{new}} = \sqrt{0.000140} \approx 0.01183 = 1.18\%
$$
对$Y$（昨日$\sigma_{Y,\text{old}}=1.2\%=0.012$）：
$$
\sigma_{Y,\text{new}}^2 = 0.000003 + 0.04 \times (0.0198)^2 + 0.94 \times (0.012)^2
$$
$$
\sigma_{Y,\text{new}} = \sqrt{0.00015404} \approx 0.01241 = 1.24\%
$$
<!-- bilingual-en:start -->
For $X$, use $\omega=0.000003,\ \alpha=0.04,\ \beta=0.94$ and yesterday's volatility $\sigma_{X,\text{old}}=1\%=0.01$. For $Y$, yesterday's volatility is $\sigma_{Y,\text{old}}=1.2\%=0.012$.
<!-- bilingual-en:end -->

 **相关系数的GARCH(1,1)估计**
<!-- bilingual-en:start -->
**GARCH(1,1) Correlation Update**
<!-- bilingual-en:end -->

- 参数$\omega=0.000001,\ \alpha=0.04,\ \beta=0.94$ 
- 昨日相关系数$\rho_{XY,\text{old}}=0.5$
- $r_X=0.03279,\ r_Y=0.01980$
<!-- bilingual-en:start -->
- Parameters: $\omega=0.000001,\ \alpha=0.04,\ \beta=0.94$.
- Previous correlation: $\rho_{XY,\text{old}}=0.5$.
- Returns: $r_X=0.03279,\ r_Y=0.01980$.
<!-- bilingual-en:end -->

协方差估计更新（类GARCH）：
$$
\text{cov}_{\text{new}} = \omega + \alpha r_X r_Y + \beta\, \text{cov}_{\text{old}}
$$
昨日协方差：
<!-- bilingual-en:start -->
Update the covariance with the GARCH-style recursion shown above. The previous covariance is the previous correlation multiplied by the two previous volatilities.
<!-- bilingual-en:end -->

$$
\text{cov}_{\text{old}} = \rho_{XY,\text{old}} \times \sigma_{X,\text{old}} \times \sigma_{Y,\text{old}} = 0.5 \times 0.01 \times 0.012 = 0.00006
$$
新协方差：
$$
\text{cov}_{\text{new}} = 0.000001 + 0.00002597 + 0.0000564 \approx 0.00008337
$$
 **最新相关系数**
$$
\rho_{XY,\text{new}} = \frac{\text{cov}_{\text{new}}}{\sigma_{X,\text{new}} \times \sigma_{Y,\text{new}}}
= \frac{0.00008337}{0.01183 \times 0.01241} \approx 0.568
$$
<!-- bilingual-en:start -->
The new covariance uses the stated covariance intercept $\omega=0.000001$. This gives $0.000001+0.04(0.03279)(0.01980)+0.94(0.00006)\approx0.00008337$. Combining it with the updated volatilities shown above, approximately 0.01183 and 0.01241, gives an updated correlation of about **0.568**.
<!-- bilingual-en:end -->

## 11.9

>[!question] 
>假定你有3个相互独立的标准正态变量 $z_1,z_2,z_3$。请用Cholesky分解构造具有指定相关矩阵的三元正态变量 $\epsilon_1,\epsilon_2,\epsilon_3$。
> <!-- bilingual-en:start -->
> Assume that $z_1,z_2,z_3$ are mutually independent standard normal variables. Use a Cholesky decomposition to construct trivariate normal variables $\epsilon_1,\epsilon_2,\epsilon_3$.
> <!-- bilingual-en:end -->
请写出 $\epsilon_1,\epsilon_2,\epsilon_3$ 关于 $z_1,z_2,z_3$ 和三个两两相关系数的表达式。
<!-- bilingual-en:start -->
Express them in terms of $z_1,z_2,z_3$ and the three pairwise correlations.
<!-- bilingual-en:end -->

- $z_1, z_2, z_3$：互相独立的标准正态随机变量 
- 希望构造相关的三元正态变量$(\epsilon_1, \epsilon_2, \epsilon_3)$，使其协方差矩阵为$\Sigma$（由你给定的相关系数决定）
<!-- bilingual-en:start -->
- $z_1, z_2, z_3$ are mutually independent standard normal random variables.
- We want to construct a correlated trivariate normal vector $(\epsilon_1, \epsilon_2, \epsilon_3)$ with covariance matrix $\Sigma$, determined by the specified correlations.
<!-- bilingual-en:end -->

 **Cholesky分解方法**
<!-- bilingual-en:start -->
**Cholesky Decomposition**
<!-- bilingual-en:end -->

若相关矩阵 $\Sigma$ 为正定矩阵，则存在唯一的正对角下三角Cholesky因子 $L$，使得
$$
\Sigma = L L^\top
$$
奇异的半正定矩阵仍可写成平方根分解，但标准正对角Cholesky不适用，通常要用广义或带主元分解。
<!-- bilingual-en:start -->
Any positive-definite covariance matrix $\Sigma$ can be factorized as shown above, where $L$ is a lower-triangular matrix. A positive-semidefinite matrix may require a generalized or pivoted factorization if it is singular.
<!-- bilingual-en:end -->

设
$$
\begin{pmatrix}
\epsilon_1 \\
\epsilon_2 \\
\epsilon_3
\end{pmatrix}
= L
\begin{pmatrix}
z_1 \\
z_2 \\
z_3
\end{pmatrix}
$$
<!-- bilingual-en:start -->
Let
<!-- bilingual-en:end -->

 **一般三元正态相关结构（协方差矩阵）**
<!-- bilingual-en:start -->
**General Trivariate Normal Correlation Structure**
<!-- bilingual-en:end -->

假设三变量的相关系数分别为 $\rho_{12},\rho_{13},\rho_{23}$，则
<!-- bilingual-en:start -->
If the three pairwise correlations are $\rho_{12},\rho_{13},\rho_{23}$, then the correlation matrix is the one shown above.
<!-- bilingual-en:end -->

$$
\Sigma = \begin{pmatrix}
1 & \rho_{12} & \rho_{13} \\
\rho_{12} & 1 & \rho_{23} \\
\rho_{13} & \rho_{23} & 1
\end{pmatrix}
$$

 **Cholesky分解下的$L$矩阵表达式**
<!-- bilingual-en:start -->
**Expression for the Cholesky Factor $L$**
<!-- bilingual-en:end -->

在 $|\rho_{12}|<1$ 且整个矩阵正定时，$L$ 为
$$
L = \begin{pmatrix}
1 & 0 & 0 \\
\rho_{12} & \sqrt{1-\rho_{12}^2} & 0 \\
\rho_{13} & \frac{ \rho_{23} - \rho_{12}\rho_{13} }{ \sqrt{1-\rho_{12}^2} } & \sqrt{ 1-\rho_{13}^2 - \left( \frac{ \rho_{23} - \rho_{12}\rho_{13} }{ \sqrt{1-\rho_{12}^2} } \right)^2 }
\end{pmatrix}
$$
<!-- bilingual-en:start -->
The displayed formula assumes $|\rho_{12}|<1$ and that the full matrix is positive definite, so every denominator is nonzero and the final radicand is positive. It is obtained by solving $\Sigma=LL^\top$ row by row. Singular boundary cases require a generalized or pivoted factorization.
<!-- bilingual-en:end -->

 **$\epsilon_1,\epsilon_2,\epsilon_3$的具体表达式**
<!-- bilingual-en:start -->
**Explicit Expressions for $\epsilon_1,\epsilon_2,\epsilon_3$**
<!-- bilingual-en:end -->

根据$\epsilon = L z$，逐项写出：
<!-- bilingual-en:start -->
Using $\epsilon = L z$, write out each component as shown above.
<!-- bilingual-en:end -->

$$\begin{aligned} \epsilon_1 &= z_1 \\ \epsilon_2 &= \rho_{12}z_1 + \sqrt{1-\rho_{12}^2}z_2 \\ \epsilon_3 &= \rho_{13}z_1 + \frac{ \rho_{23} - \rho_{12}\rho_{13} }{ \sqrt{1-\rho_{12}^2} } z_2 + \sqrt{ 1-\rho_{13}^2 - \left( \frac{ \rho_{23} - \rho_{12}\rho_{13} }{ \sqrt{1-\rho_{12}^2} } \right)^2 } z_3 \end{aligned}$$

## 11.14

>[!question] 
>假设银行持有一个非常大的均质贷款组合，每笔贷款的年违约概率为1.5%，回收率为30%。银行采用单因子高斯Copula，潜在资产相关参数为0.2。请用Vasicek模型估计组合违约率的99.5%分位点。
><!-- bilingual-en:start -->
>Assume a bank holds a very large homogeneous portfolio of loans. Each loan has an annual probability of default of 1.5% and a recovery rate of 30%. The bank uses a one-factor Gaussian Copula with latent asset-correlation parameter 0.2. Use the Vasicek model to estimate the 99.5th percentile of the portfolio default rate.
><!-- bilingual-en:end -->

- 贷款数目很大（$n\to\infty$，可认为“连续”）
- 每笔贷款年违约概率 $p=1.5\% = 0.015$
- 违约时回收率为30%（计算违约率分位点时不用；计算损失分位点时才需要）
- 潜在资产相关参数 $\rho=0.2$
- 计算贷款池**总体违约率的99.5%分位点**
<!-- bilingual-en:start -->
- The number of loans is very large, so the portfolio can be treated as asymptotically granular.
- Each loan has annual default probability $p=1.5\% = 0.015$.
- Recovery at default is 30%; it is irrelevant to the default-rate quantile itself but would matter for a loss quantile.
- The latent asset-correlation parameter is $\rho=0.2$.
- We need the **99.5th percentile of the portfolio-wide default rate**.
<!-- bilingual-en:end -->

 **Vasicek单因子模型公式**
<!-- bilingual-en:start -->
**Vasicek One-Factor Formula**
<!-- bilingual-en:end -->

对于大量贷款池，设
<!-- bilingual-en:start -->
For a very large loan portfolio, let
<!-- bilingual-en:end -->

- 单笔贷款年违约概率 $p$
- 单因子潜在资产相关参数 $\rho$
- 组合违约率的$q$分位点为$L_q$
<!-- bilingual-en:start -->
- $p$ be the annual default probability of each loan.
- $\rho$ be the one-factor latent asset-correlation parameter.
- $L_q$ be the $q$th quantile of the portfolio default rate.
<!-- bilingual-en:end -->

**分位点计算公式为：**
<!-- bilingual-en:start -->
**The quantile formula is:**
<!-- bilingual-en:end -->

$$
L_q = \Phi\!\left(
\frac{\Phi^{-1}(p)+\sqrt{\rho}\,\Phi^{-1}(q)}{\sqrt{1-\rho}}
\right).
$$

- $\Phi$：标准正态分布函数
- $\Phi^{-1}$：标准正态分布分位点函数
<!-- bilingual-en:start -->
- $\Phi$ is the standard normal cumulative distribution function.
- $\Phi^{-1}$ is the standard normal quantile function.
<!-- bilingual-en:end -->

 **代入本题参数计算**
<!-- bilingual-en:start -->
**Substitute the Given Parameters**
<!-- bilingual-en:end -->

- $p=0.015$
- $\rho=0.2$
- $q=0.995$

**计算各分位点**
<!-- bilingual-en:start -->
**Calculate the Required Quantiles**
<!-- bilingual-en:end -->

- $\Phi^{-1}(0.015)\approx -2.17009038$
- $\Phi^{-1}(0.995)\approx 2.57582930$
- $\sqrt{\rho}=\sqrt{0.2}\approx 0.44721360$
- $\sqrt{1-\rho}=\sqrt{0.8}\approx 0.89442719$
    
代入公式：
$$
L_{0.995}=\Phi\!\left(
\frac{-2.17009038+0.44721360\times2.57582930}{0.89442719}
\right)
\approx\Phi(-1.13832015).
$$
- $0.44721360\times2.57582930\approx1.15194588$
- 分子约为 $-1.01814449$
- 标准化参数约为 $-1.13832015$
<!-- bilingual-en:start -->
Substitute into the formula:
- $0.44721360\times2.57582930\approx1.15194588$.
- The numerator is approximately $-1.01814449$.
- The standardized argument is approximately $-1.13832015$.
<!-- bilingual-en:end -->



查标准正态分布表：
<!-- bilingual-en:start -->
Using a standard normal table:
<!-- bilingual-en:end -->

- $\Phi(-1.13832015)\approx0.12749341$
    
**最终答案**
<!-- bilingual-en:start -->
**Final Answer**
<!-- bilingual-en:end -->

- 99.5%置信度下的贷款组合违约率分位点为
    $$
    \boxed{12.74934\%}
    $$
<!-- bilingual-en:start -->
- At the 99.5% confidence level, the portfolio default-rate quantile is $\boxed{12.74934\%}$. The 30% recovery rate would matter for a loss quantile, but not for this default-rate quantile.
<!-- bilingual-en:end -->

## 11.15

>[!question] 
>如果过去10年间一个消费贷款组合的违约率为1%、9%、2%、3%、5%、1%、6%、7%、4%和1%。Vasicek模型中参数的最大似然估计是多少?
><!-- bilingual-en:start -->
>If a consumer-loan portfolio recorded annual default rates of 1%, 9%, 2%, 3%, 5%, 1%, 6%, 7%, 4%, and 1% over the past ten years, what are the maximum-likelihood estimates of the Vasicek-model parameters?
><!-- bilingual-en:end -->

 **1. 违约率数据**
<!-- bilingual-en:start -->
**1. Default-Rate Data**
<!-- bilingual-en:end -->

10年违约率序列：
$$
1\%,\ 9\%,\ 2\%,\ 3\%,\ 5\%,\ 1\%,\ 6\%,\ 7\%,\ 4\%,\ 1\%
$$
即：
$$
0.01,\ 0.09,\ 0.02,\ 0.03,\ 0.05,\ 0.01,\ 0.06,\ 0.07,\ 0.04,\ 0.01
$$
<!-- bilingual-en:start -->
The ten annual default rates are shown above, first as percentages and then as decimal proportions.
<!-- bilingual-en:end -->

**2. 取正态分位点 $Y_t = \Phi^{-1}(l_t)$**
<!-- bilingual-en:start -->
**2. Transform to Normal Quantiles: $Y_t = \Phi^{-1}(l_t)$**
<!-- bilingual-en:end -->

查表或用软件可得（下表保留6位小数；计算时使用未舍入值）：
<!-- bilingual-en:start -->
The values can be obtained from a standard normal table or software. Six decimals are displayed below, while the calculations use unrounded values.
<!-- bilingual-en:end -->

|**$l_t$**|**$Y_t = \Phi^{-1}(l_t)$**|
|---|---|
|0.01|$-2.326348$|
|0.09|$-1.340755$|
|0.02|$-2.053749$|
|0.03|$-1.880794$|
|0.05|$-1.644854$|
|0.01|$-2.326348$|
|0.06|$-1.554774$|
|0.07|$-1.475791$|
|0.04|$-1.750686$|
|0.01|$-2.326348$|

 **3. 计算均值和方差**
<!-- bilingual-en:start -->
**3. Calculate the Mean and Variance**
<!-- bilingual-en:end -->

正态似然下，均值与方差的MLE分别为
$$
\hat\mu=\bar Y=-1.86804455,
\qquad
\sum_{t=1}^{10}(Y_t-\bar Y)^2=1.25839741,
$$
$$
\hat v_{\mathrm{MLE}}
=\frac{1.25839741}{10}
=0.125839741.
$$
这里必须除以 $T=10$；除以9得到无偏样本方差，但不是正态模型的方差MLE。

 **4. 映射回 Vasicek 参数**
<!-- bilingual-en:start -->
Under the normal likelihood, the MLEs of the transformed mean and variance are $\hat\mu=\bar Y=-1.86804455$ and $\hat v_{\mathrm{MLE}}=1.25839741/10=0.125839741$. The variance must divide by $T=10$; division by nine gives the unbiased sample variance, not the normal-model variance MLE.

**4. Map Back to the Vasicek Parameters**
<!-- bilingual-en:end -->

由于
$$
E[Y]=\frac{\Phi^{-1}(p)}{\sqrt{1-\rho}},
\qquad
\operatorname{Var}(Y)=\frac{\rho}{1-\rho},
$$
所以
$$
\hat\rho=\frac{\hat v}{1+\hat v}
=\frac{0.125839741}{1.125839741}
=\boxed{0.11177412},
$$
$$
\hat p=\Phi\!\left(\hat\mu\sqrt{1-\hat\rho}\right)
=\Phi(-1.76055234\ldots)
=\boxed{0.03915710}.
$$
 **5. 最终结论**
<!-- bilingual-en:start -->
Because $E[Y]=\Phi^{-1}(p)/\sqrt{1-\rho}$ and $\operatorname{Var}(Y)=\rho/(1-\rho)$, the exact transformed-normal MLEs are $\hat\rho=\hat v/(1+\hat v)=\boxed{0.11177412}$ and $\hat p=\Phi(\hat\mu\sqrt{1-\hat\rho})=\boxed{0.03915710}$.

**5. Final Conclusion**
<!-- bilingual-en:end -->

- **Vasicek相关参数MLE：** $\boxed{\hat\rho=0.11177412}$
- **长期无条件违约概率MLE：** $\boxed{\hat p=3.915710\%}$

这些估计依赖连续、渐近组合近似以及年度观测相互独立的假设；有限组合、序列相关或参数不确定性需要改写似然或另行处理。
<!-- bilingual-en:start -->
- **Vasicek correlation-parameter MLE:** $\boxed{\hat\rho=0.11177412}$.
- **Long-run unconditional default-probability MLE:** $\boxed{\hat p=3.915710\%}$.

These estimates rely on the continuous asymptotic portfolio approximation and independent annual observations. A finite portfolio, serial dependence, or parameter uncertainty requires a different likelihood or additional treatment.
<!-- bilingual-en:end -->

## 11.16

>[!question] 
>上个交易日收盘时，资产X价格为300美元、日波动率为1.3%，资产Y价格为8美元、日波动率为1.5%，两者收益相关系数为0.8。今天X收于298美元，Y仍收于8美元。请计算：(a) $\lambda=0.94$ 的EWMA更新；(b) $\omega=0.000002$、$\alpha=0.04$、$\beta=0.94$ 的题设GARCH(1,1)式更新。分别求X、Y的新波动率和新相关系数，并说明实际建模中两个资产是否必须使用相同参数。
><!-- bilingual-en:start -->
>At the end of the previous trading day, asset X had a price of \$300 and daily volatility of 1.3%; today it closes at \$298. Asset Y had a price of \$8, daily volatility of 1.5%, and return correlation of 0.8 with X; today it again closes at \$8. Calculate the updated volatilities of X and Y and their updated correlation using (a) an EWMA model with $\lambda=0.94$ and (b) a GARCH(1,1)-style model with $\omega=0.000002$, $\alpha=0.04$, and $\beta=0.94$. In practice, should X and Y use the same parameter values?
><!-- bilingual-en:end -->

 **1. 收益率计算**
<!-- bilingual-en:start -->
**1. Calculate Returns**
<!-- bilingual-en:end -->

- $r_X = \ln\left(\frac{298}{300}\right) \approx -0.006688988$
- $r_Y = \ln\left(\frac{8}{8}\right) = 0$   

 **2. EWMA模型**
<!-- bilingual-en:start -->
**2. EWMA Model**
<!-- bilingual-en:end -->

 **波动率更新公式**
<!-- bilingual-en:start -->
**Volatility-Update Formula**
<!-- bilingual-en:end -->

$$
\sigma_{\text{new}}^2 = \lambda \sigma_{\text{old}}^2 + (1-\lambda) r_{\text{new}}^2
$$

**$X$：**
    $$
    \sigma_{X,\text{new}}^2 = 0.94(0.013)^2+0.06(-0.006688988)^2=0.000161544554
    $$
    $$  
    \sigma_{X,\text{new}} = \sqrt{0.000161544554} \approx 0.01271002 = 1.271002\%
    $$


**$Y$：**
$$ 
    \sigma_{Y,\text{new}}^2 = 0.0002115 + 0 = 0.0002115  
    $$
    $$
    \sigma_{Y,\text{new}} = \sqrt{0.0002115} \approx 0.01454304 = 1.454304\%
    $$

 **相关系数(EWMA-协方差)**
<!-- bilingual-en:start -->
**Correlation Coefficient (EWMA Covariance Update)**
<!-- bilingual-en:end -->

- 上日协方差：$0.8\times0.013\times0.015=0.000156$
- $r_X r_Y = -0.006688988 \times 0 = 0$
- $0.94\times0.000156 = 0.00014664$
    $$
    \text{cov}_{\text{new}} = 0.00014664
    $$
    $$
    \rho_{\text{new}} = \frac{0.00014664}{0.01271002\times0.01454304} \approx \boxed{0.793325}
    $$
<!-- bilingual-en:start -->
- Previous covariance: $0.8\times0.013\times0.015=0.000156$.
- Today's return cross-product: $r_X r_Y = -0.006688988 \times 0 = 0$.
- Decayed previous covariance: $0.94\times0.000156 = 0.00014664$.
- With the updated volatilities, the EWMA correlation is $\boxed{0.793325}$.
<!-- bilingual-en:end -->

 **3. GARCH(1,1)模型**
<!-- bilingual-en:start -->
**3. GARCH(1,1) Model**
<!-- bilingual-en:end -->

 **波动率递推**
<!-- bilingual-en:start -->
**Volatility Recursion**
<!-- bilingual-en:end -->

$$
\sigma_{\text{new}}^2 = \omega + \alpha r_{\text{new}}^2 + \beta \sigma_{\text{old}}^2
$$
**$X$：**

    $$
    \sigma_{X,\text{new}}^2 = 0.000002 + 0.04(-0.006688988)^2 + 0.94(0.013)^2 = 0.000162649702
    $$
    
    $$
    \sigma_{X,\text{new}} = \sqrt{0.000162649702} \approx 0.01275342 = 1.275342\%
    $$

**$Y$：**

    $$ 
    \sigma_{Y,\text{new}}^2 = 0.000002 + 0 + 0.0002115 = 0.0002135
    $$
    $$
    \sigma_{Y,\text{new}} = \sqrt{0.0002135} \approx 0.01461164 = 1.461164\%
    $$

 **相关系数(GARCH协方差法)**
<!-- bilingual-en:start -->
**Correlation Coefficient (GARCH Covariance Update)**
<!-- bilingual-en:end -->

- $\text{cov}_{\text{old}}=0.000156$
    
- $r_X r_Y=0$
    
- $\beta\text{cov}_{\text{old}}=0.94\times0.000156=0.00014664$
    
    $$
    \text{cov}_{\text{new}} = 0.000002 + 0 + 0.00014664 = 0.00014864
    $$
    $$
    \rho_{\text{new}} = \frac{0.00014864}{0.01275342\times0.01461164} \approx \boxed{0.797646}
    $$

 **4. 参数是否必须相同？模型边界是什么？**
<!-- bilingual-en:start -->
**4. Must the Parameters Be the Same, and What Is the Model Boundary?**
<!-- bilingual-en:end -->

- $\omega$参与决定长期方差或协方差水平；$\alpha$和$\beta$控制对新冲击与旧状态的响应。X与Y不必使用相同标量参数，实际中应按数据与模型结构联合或分别校准。
- 本题把同一标量递推逐对用于方差和协方差，只是教学近似。对多资产矩阵逐对独立套用这类公式，不保证更新后的协方差矩阵半正定，甚至可能产生绝对值大于1的“相关系数”。实际应用应使用能保持矩阵合法性的多变量波动模型或整矩阵更新，并检查PSD、参数稳定性和样本外表现。
<!-- bilingual-en:start -->
- $\omega$ helps determine the long-run variance or covariance level, while $\alpha$ and $\beta$ control responsiveness to new shocks and persistence. X and Y need not use identical scalar parameters; they should be calibrated consistently with the chosen model and data.
- Applying separate scalar recursions pair by pair, as this exercise does, is only a teaching approximation. It does not guarantee that a multi-asset covariance matrix remains positive semidefinite and can even imply an invalid correlation. Real applications require a multivariate specification or whole-matrix update that preserves validity, followed by PSD, stability, and out-of-sample checks. Under the exercise convention, the GARCH-style updated correlation is $\boxed{0.797646}$.
<!-- bilingual-en:end -->


## 11.19

数值积分与矩阵运算可由软件实现，并应保留可复现的参数、算法和精度设置。
<!-- bilingual-en:start -->
Numerical integration and matrix operations can be implemented in software; the parameters, algorithm, and precision settings should be retained for reproducibility.
<!-- bilingual-en:end -->

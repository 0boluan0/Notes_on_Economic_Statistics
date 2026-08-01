
# 1. 相关与协方差的定义与计算
<!-- bilingual-en:start -->
*1. Definition and Calculation of Correlation and Covariance*
<!-- bilingual-en:end -->

## 1.1 相关系数与协方差定义
<!-- bilingual-en:start -->
*1.1 Definitions of the Correlation Coefficient and Covariance*
<!-- bilingual-en:end -->

[[相关性、Copula 与尾部依赖|相关系数]]

## 1.2 EWMA更新协方差和相关系数
<!-- bilingual-en:start -->
*1.2 EWMA updates covariance and correlation coefficient*
<!-- bilingual-en:end -->

[[波动率度量：历史、实现与隐含波动率|EWMA]] 下协方差的更新公式为：
$$
\mathrm{Cov}_{n} = \lambda \,\mathrm{Cov}_{n-1} + (1-\lambda)\,x_{n-1}\,y_{n-1} \, $$ 方差更新公式：$\sigma^2_{X,n} = \lambda\,\sigma^2_{X,n-1} + (1-\lambda)\,x_{n-1}^2$

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

概率统计内容，~~略~~ 简述：本节可按相关矩阵的正定性、尾部相关定义与计算、以及 Copula 的基本性质（Sklar 定理）三方面复习要点。

## 协方差矩阵的正定性条件


在多变量情形下，所有随机变量的协方差构成**协方差矩阵** $\Omega$。要成为有效的协方差矩阵，$\Omega$ 必须是**正定或半正定**的，即满足对任意非零向量 $w$：
$$ 
w^T\,\Omega\,w \;\ge\; 0 \,. 
$$ 
这是协方差矩阵的**内部一致性**条件——否则计算得出的组合方差将出现负值等不合理情况。

检验协方差矩阵正定性的常用方法之一是检查其特征值或主子式：所有特征值均非负（主子式均为非负）是半正定矩阵的充要条件。**例如：** 
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
由于行列式为负，该矩阵存在负特征值，不满足半正定条件。因此这个“相关矩阵”不具备内部一致性，实际上不可能是某组随机变量的协方差矩阵。在风险管理中，若计算得到的相关矩阵不正定，需要进行调整（如降秩近似或调整相关系数）以修正为最接近的正定矩阵。

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

A valid covariance or correlation matrix must be positive semidefinite, because the variance of every linear combination must be non-negative: $w^T\,\Omega\,w \ge 0$. For the displayed matrix, each diagonal entry is 1 and each relevant $2\times2$ principal minor is 0.19, but the full determinant is $-0.62$. The negative determinant implies a negative eigenvalue, so the matrix is not positive semidefinite and cannot be a valid correlation matrix. In practice, an invalid estimated matrix must be adjusted, for example through a nearest-positive-semidefinite or lower-rank approximation.
<!-- bilingual-en:end -->

详细解答：检验 $\Omega$ 的正定性，可以计算其特征值或主子式。上面矩阵的一阶和二阶主子式均为非负（对角元为1，任意 $2\times2$ 子矩阵行列式$=1-0.9^2=0.19$），但三阶行列式计算得到 $-0.62$。因为出现了负的行列式（即负特征值），$\Omega$ 不是半正定矩阵。因此该矩阵不能作为有效的相关矩阵（它会导致某线性组合的方差为负，这是不可能的）。因此结论是**不满足**正定条件。
<!-- bilingual-en:start -->
Detailed answer: positive semidefiniteness can be checked through eigenvalues or principal minors. Although the first- and second-order principal minors are non-negative, the full determinant is $-0.62$. The matrix therefore has a negative eigenvalue and cannot be a valid covariance or correlation matrix: it would imply a negative variance for some linear combination. The correct conclusion is that the matrix **fails** the positive-semidefiniteness requirement.
<!-- bilingual-en:end -->

## 多元正态分布与相关系数的生成机制
<!-- bilingual-en:start -->
*Generating Correlated Multivariate Normal Variables*
<!-- bilingual-en:end -->

别看
<!-- bilingual-en:start -->
Skip this section.
<!-- bilingual-en:end -->

在**[[多元正态分布|多元正态分布]]**中，相关性的一个重要性质是：**任意线性组合**的分布仍为正态，且条件分布是正态分布。例如，若 $(V_1, V_2)$ 服从二维正态分布，$V_2$ 在给定 $V_1=v_1$ 条件下仍是正态，其条件均值和标准差为：
$$ 
E[V_2 \mid V_1 = v_1] = \mu_2 + \rho\,\frac{\sigma_2}{\sigma_1}\, (v_1 - \mu_1)\,, \qquad 
\sqrt{\mathrm{Var}(V_2 \mid V_1 = v_1)} = \sigma_2\,\sqrt{\,1-\rho^2\,} \,,
$$ 
其中 $\mu_i, \sigma_i$ 是 $V_i$ 的均值和标准差，$\rho$ 是相关系数。这表明在联合正态中，相关使得一个变量对另一个的条件期望是线性函数，条件方差为常数。
<!-- bilingual-en:start -->
In a **[[多元正态分布|multivariate normal distribution]]**, every linear combination is normally distributed, and every conditional distribution is also normal. For example, if $(V_1, V_2)$ is bivariate normal, then the conditional distribution of $V_2$ given $V_1=v_1$ is normal with the mean and standard deviation shown above. Here, $\mu_i$ and $\sigma_i$ are the mean and standard deviation of $V_i$, and $\rho$ is the correlation coefficient. Thus, under joint normality, one variable's conditional mean is a linear function of the other variable, while its conditional variance is constant.
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

一般地，对于 $n$ 维正态分布，可以使用**科列斯基分解**法：设希望生成协方差矩阵为 $\Sigma$（$n\times n$）的正态向量。先生成 $n$ 维独立标准正态向量 $Z = (Z_1,\dots,Z_n)^T$。令 $A$ 为 $\Sigma$ 的科列斯基下三角矩阵（满足 $A A^T = \Sigma$），则随机向量 $X = A\,Z$ 即服从协方差为 $\Sigma$ 的 $n$ 维正态分布。此方法确保生成的相关结构满足正定性要求，因为 $\Sigma = AA^T$ 天生正定。
<!-- bilingual-en:start -->
More generally, an $n$-dimensional normal vector can be generated by **Cholesky decomposition**. Suppose the target covariance matrix is $\Sigma$ of size $n\times n$. Generate an independent standard normal vector $Z = (Z_1,\dots,Z_n)^T$, and let $A$ be the lower-triangular Cholesky factor satisfying $A A^T = \Sigma$. Then $X = A\,Z$ is multivariate normal with covariance matrix $\Sigma$. This construction also makes the consistency requirement explicit: a Cholesky factor exists only when the target matrix has the required positive-semidefinite structure.
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

当涉及 $N$ 个随机变量（如 $N$ 个资产收益）时，直接估计两两之间的相关系数有 $\frac{N(N-1)}{2}$ 个参数，随着 $N$ 增大变得非常繁琐。**因子模型（[[因子分析|Factor]] Model）**假设变量的相关结构由少数几个共同因子驱动，从而减少需估计的参数数量。
<!-- bilingual-en:start -->
With $N$ random variables, such as $N$ asset returns, estimating every pairwise correlation requires $\frac{N(N-1)}{2}$ parameters. This quickly becomes unwieldy as $N$ grows. A **factor model ([[因子分析|factor]] model)** assumes that a small number of common factors drive most of the dependence, greatly reducing the number of parameters that must be estimated.
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

在处理非正态分布的变量时，我们需要一种灵活的方法来定义它们之间的相关结构，而不改变各自的边际分布。这正是 **Copula** 方法的核心思想。**Gaussian Copula** 是 Copula 函数的一种特殊类别，用于通过正态分布构造相关性。其建模的基本步骤如下：
<!-- bilingual-en:start -->
When variables have non-normal marginal distributions, a Copula lets us model their dependence without changing those marginals. This is the central purpose of a **Copula**. A **Gaussian Copula** imposes dependence through a multivariate normal latent space. Its basic construction proceeds as follows:
<!-- bilingual-en:end -->

1. **边际分布估计：**首先针对每个变量估计其边际分布 $F_{V_i}(v)$（累积分布函数），例如通过历史数据拟合得到。
2. **分位数映射：**将每个原始变量 $V_i$ 映射到对应的标准正态变量 $U_i$。具体做法是利用**分位数对分位数**映射：令 
   $$u_i = \Phi^{-1}\!\big(F_{V_i}(v_i)\big)\,,$$ 
   其中 $\Phi^{-1}$ 是标准正态分布的反函数（把边际分布概率映射为对应的正态分位数）。如此得到的新变量 $U_1,\dots,U_n$ 均服从标准正态分布。
3. **施加相关结构：**假定 $(U_1,\dots,U_n)$ 服从某一 $n$ 元**多元正态分布**，并根据需要设定它们之间的相关矩阵（如通过单因子模型或历史估计得到）。在 Gaussian Copula 模型中，我们通常直接设定这些 $U$ 变量的线性相关系数矩阵为我们期望的相关结构。
4. **构建联合分布：**由于 $U$ 的联合分布已由上述步骤确定，利用逆映射可得到原变量 $V$ 的联合分布，即 Copula 联合分布。形式上，对于任意一组取值 $(v_1,\dots,v_n)$：
   $$ 
   P(V_1 \le v_1, \dots, V_n \le v_n) \;=\; P\!\big(U_1 \le \Phi^{-1}(F_{V_1}(v_1)),\,\dots,\,U_n \le \Phi^{-1}(F_{V_n}(v_n))\big)\,. 
   $$ 
   右侧概率可通过已知的 $U$ 联合正态分布计算（如多元正态CDF计算），这定义了原变量的Copula联合分布。
<!-- bilingual-en:start -->

&nbsp;
**1.** **Estimate the marginal distributions:** Estimate each variable's marginal cumulative distribution function $F_{V_i}(v)$, for example from historical data.<br>
**2.** **Map quantiles into normal space:** Transform each observation $V_i$ into a standard normal latent variable $U_i$ using the probability-integral and inverse-normal transformations:<br>
   $$u_i = \Phi^{-1}\!\big(F_{V_i}(v_i)\big)\,.$$
   Each transformed variable is standard normal.
**3.** **Impose a dependence structure:** Assume $(U_1,\dots,U_n)$ follows an $n$-dimensional **multivariate normal distribution** with a chosen correlation matrix, estimated historically or specified through a factor model.<br>
**4.** **Construct the joint distribution:** The multivariate normal law determines joint probabilities in latent space. Mapping those probabilities back through the marginal distributions defines the Copula-based joint distribution of the original variables $(V_1,\dots,V_n)$.<br>
<!-- bilingual-en:end -->

简单来说，Gaussian Copula 先把各变量各自“正态化”（变为$U(0,1)$的概率再映射到标准正态），然后假设这些正态化后的变量服从一个多元正态（相关由Copula参数决定），最后通过逆变换回到原始变量空间，从而为原变量施加所需的相关关系。
<!-- bilingual-en:start -->
In short, a Gaussian Copula first converts each original variable into a uniform probability and then into a standard normal quantile. It models the transformed variables jointly with a multivariate normal distribution whose correlation matrix contains the Copula parameters. Finally, inverse marginal transformations return the simulated values to their original scales. The marginals remain unchanged; only the dependence structure is supplied by the Copula.
<!-- bilingual-en:end -->

**模拟考题：**有两种非正态分布的风险因子 $V_1$ 和 $V_2$，我们希望用高斯 Copula 来建立它们的联合分布关系。请写出使用 Gaussian Copula 建立相关结构的基本步骤。
<!-- bilingual-en:start -->
**Practice question:** Two risk factors $V_1$ and $V_2$ have non-normal marginal distributions. We want to use a Gaussian Copula to model their joint distribution. State the basic steps used to construct the dependence structure.
<!-- bilingual-en:end -->

**详细解答：**可以按照以下步骤：
1. **确定边际分布：**分别确定 $V_1$ 和 $V_2$ 的边际分布 $F_{V_1}(x)$ 和 $F_{V_2}(y)$（例如通过数据拟合出各自的分布类型和参数）。
2. **转换到标准正态空间：**将观测值 $v_1, v_2$ 转换为对应的标准正态分位数：
   $$u_1 = \Phi^{-1}(F_{V_1}(v_1)), \qquad u_2 = \Phi^{-1}(F_{V_2}(v_2)),$$ 
   这样得到的 $U_1, U_2$ 均服从 $N(0,1)$ 分布。
3. **假设正态相关结构：**在 $U_1, U_2$ 空间引入相关假设，即设 $(U_1, U_2)$ 服从相关系数为 $\rho$ 的二维正态分布 $N(0,1)$。参数 $\rho$ 被称为**Copula相关系数**。
4. **得到联合分布：**通过 $U$ 空间的二维正态分布，可以得到任意联合事件的概率。例如：
   $$P(V_1 \le x,\; V_2 \le y) = P\!\big(U_1 \le \Phi^{-1}(F_{V_1}(x)),\; U_2 \le \Phi^{-1}(F_{V_2}(y))\big)\,,$$ 
   后者可用标准正态Copula的分布函数计算，从而定义了 $(V_1, V_2)$ 的联合分布。
<!-- bilingual-en:start -->
**Detailed answer:**
**1.** **Determine the marginals:** Estimate $F_{V_1}(x)$ and $F_{V_2}(y)$, including their distributional forms and parameters.<br>
**2.** **Transform to standard normal space:** Map observations $v_1, v_2$ to<br>
   $$u_1 = \Phi^{-1}(F_{V_1}(v_1)), \qquad u_2 = \Phi^{-1}(F_{V_2}(v_2)).$$
   The transformed variables $U_1, U_2$ are standard normal.
**3.** **Specify normal dependence:** Let $(U_1, U_2)$ be bivariate normal with correlation $\rho$. This $\rho$ is the **Copula correlation parameter**.<br>
**4.** **Recover the joint distribution:** Compute joint probabilities in latent normal space:<br>
   $$P(V_1 \le x,\; V_2 \le y) = P\!\big(U_1 \le \Phi^{-1}(F_{V_1}(x)),\; U_2 \le \Phi^{-1}(F_{V_2}(y))\big).$$
   The bivariate normal CDF on the right defines the joint distribution of $(V_1,V_2)$ while preserving both marginals.
<!-- bilingual-en:end -->

## 3.1Copula 函数的定义与代数表达
<!-- bilingual-en:start -->
*3.1 Definition and Algebraic Form of a Copula*
<!-- bilingual-en:end -->
**Copula函数**是描述多维随机变量**相关结构**的函数，它将各维边际分布拼接成联合分布，同时保证边际分布保持不变。Copula 概念的数学基础是 **Sklar 定理**：任何多维分布函数 $F_{X,Y}(x,y)$ 都可表示为其边际分布和一个Copula的组合：
$$
F_{X,Y}(x,y) = C\!\big(F_X(x),\;F_Y(y)\big)\,,
$$
其中 $C(u,v)$ 即为一二维Copula函数（满足边际为$U(0,1)$分布的联结函数）。
<!-- bilingual-en:start -->
A **Copula function** describes the dependence structure of a multivariate distribution. It combines marginal distributions into a joint distribution while leaving each marginal unchanged. Its mathematical foundation is **Sklar's theorem**: any multivariate distribution function $F_{X,Y}(x,y)$ can be represented by its marginals and a Copula. Here, $C(u,v)$ is a bivariate Copula whose own marginals are uniform on $U(0,1)$.
<!-- bilingual-en:end -->

对于**Gaussian Copula**而言，有显式的代数表达形式。以二维为例，假设 $X$ 和 $Y$ 边际分布分别为 $G_1(x)$ 和 $G_2(y)$。高斯Copula下的联合分布函数为：
$$ 
F_{X,Y}(x,y) \;=\; \Phi_{2,\rho}\!\Big(\Phi^{-1}\big(G_1(x)\big)\,,\;\Phi^{-1}\big(G_2(y)\big)\Big)\,,
$$ 
其中 $\Phi^{-1}$ 是标准正态分布的反函数，$\Phi_{2,\rho}$ 表示相关系数为 $\rho$ 的二维正态分布的累积函数。等式右边其实就是Copula函数：
$$ 
C(u_1, u_2) = \Phi_{2,\rho}\!\big(\Phi^{-1}(u_1),\; \Phi^{-1}(u_2)\big)\,, \qquad 0 \le u_1,u_2 \le 1\,.
$$ 
可以看出，Copula函数将边际分布的概率值 $(u_1,u_2)$ 通过正态分位数映射，再代入相关正态分布的CDF，从而得到联合概率。对任意给定的 $\rho$，Gaussian Copula 都保证 $F_X$ 和 $F_Y$ 保持各自不变，仅通过 $\rho$ 来影响变量间的关联形式。
<!-- bilingual-en:start -->
The **Gaussian Copula** has an explicit algebraic form. In two dimensions, suppose $X$ and $Y$ have marginal CDFs $G_1(x)$ and $G_2(y)$. Their Gaussian-Copula joint CDF is the expression shown above, where $\Phi^{-1}$ is the standard normal quantile function and $\Phi_{2,\rho}$ is the bivariate standard normal CDF with correlation $\rho$. Thus the Copula maps marginal probabilities $(u_1,u_2)$ into normal quantiles and evaluates their joint normal probability. For any fixed $\rho$, the marginals $F_X$ and $F_Y$ remain unchanged; $\rho$ affects only their dependence.
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
其中相关参数 $\rho$ 被视为所有贷款对公共因子的同质相关性，$Z_i \sim N(0,1)$ 是借款人 $i$ 独立的特有风险。给定单因子结构，任意两家公司 $i,j$ 的 **Copula相关系数**（对应 $U_i, U_j$ 间的相关系数）均为 $\rho$。
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

**模拟考题：**假设有两个公司，年违约概率均为 2%（即 $p=0.02$）。利用单因子高斯 Copula 模型，并设两家公司之间的 Copula相关系数 $\rho=0.1$，求它们在同一年内**同时违约**的概率。
<!-- bilingual-en:start -->
**Practice question:** Two companies each have an annual default probability of 2%, so $p=0.02$. Under a one-factor Gaussian Copula with latent-variable correlation $\rho=0.1$, calculate the probability that both companies **default in the same year**.
<!-- bilingual-en:end -->

**详细解答：**两家公司同时违约的概率可以通过Copula计算，即：
$$ 
P(\text{两家公司都违约}) = C_{\rho}(p,\;p) \;=\; \Phi_{2,\;0.1}\Big(\Phi^{-1}(0.02),\;\Phi^{-1}(0.02)\Big)\,. 
$$ 
我们需要将违约概率转换为对应的正态临界值：$\Phi^{-1}(0.02) \approx -2.0537$。于是：
$$ 
P(\text{同时违约}) = \Phi_{2,\;0.1}(-2.0537,\; -2.0537) \,.
$$ 
这个值需要通过二维正态积分计算。近似计算可得到约 $7.4\times 10^{-4}$，即0.074%的概率。相比于独立情形下 $0.02 \times 0.02 = 0.0004$（0.04%）的同时违约概率，考虑相关性后同时违约的概率有所提高（从0.04%增至0.074%）。这体现了正相关违约风险中，相关性使极端共同违约事件更可能发生。
<!-- bilingual-en:start -->
**Detailed answer:** Joint default occurs when both latent variables fall below the threshold $\Phi^{-1}(0.02) \approx -2.0537$. Therefore, the joint-default probability is the bivariate normal lower-tail probability shown above. Numerical integration gives approximately $7.4\times 10^{-4}$, or 0.074%. Under independence, the probability would be $0.02 \times 0.02 = 0.0004$, or 0.04%. Positive dependence therefore raises the probability of simultaneous default from about 0.04% to 0.074%.
<!-- bilingual-en:end -->

## 3.3最坏违约率计算与 VaR 推导
<!-- bilingual-en:start -->
*3.3 Worst-Case Default Rate and the Derivation of VaR*
<!-- bilingual-en:end -->


在信贷组合风险管理中，我们关注高置信水平下的**最坏违约率**（Worst Case Default Rate, **WCDR**），即在给定置信度下组合违约率可能达到的最大值。通常监管设定99.9%的置信度（即极端情景），相应的WCDR用于计算信用风险资本。利用单因子高斯Copula模型，可以推导WCDR的解析形式。
<!-- bilingual-en:start -->
In credit-portfolio risk management, the **worst-case default rate (WCDR)** is a high quantile of the portfolio default-rate distribution. It is not an absolute maximum; it is the default rate that is exceeded only with probability $1-\alpha$ at confidence level $\alpha$. Regulators often use 99.9%, and the resulting WCDR feeds into credit-risk capital calculations. The one-factor Gaussian Copula yields a closed-form expression.
<!-- bilingual-en:end -->

对于大型均质组合（违约概率均为 $p$，相关系数 $\rho$），一年期违约率 $DR$ 在模型下满足： 
$$ 
DR = \Phi\Big(\frac{\Phi^{-1}(p) - \sqrt{\rho}\,F}{\sqrt{\,1-\rho\,}}\Big)\,,
$$ 
其中 $F \sim N(0,1)$。要得到置信度 $\alpha$（如$\alpha=99.9\%$）对应的违约率分位点 $x_\alpha = \text{WCDR}( \alpha)$，相当于取 $F$ 在高端分位数 $z_\alpha = \Phi^{-1}(\alpha)$（如3.0902对应99.9%）处的情形，因为最极端违约发生在公共因子最差的情况下。设 $\theta = \Phi^{-1}(p)$ 为单个贷款违约临界值，则：
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
表示在置信度 $\alpha$ 下，一年内最大损失 = 最坏违约比例 $\times$ 敞口总额 $\times$ 损失率。
<!-- bilingual-en:start -->
To compute **value at risk (VaR)**, convert WCDR into a loss amount. If a homogeneous portfolio has total exposure $L$ and loss given default $\lambda=1-\text{recovery rate}$, then the confidence-level loss is WCDR multiplied by total exposure and LGD. In other words, the one-year loss quantile at confidence level $\alpha$ equals the worst-case default fraction times the portfolio exposure times the loss rate.
<!-- bilingual-en:end -->

**模拟考题：**某银行持有价值 \$100 百万的均质零售贷款组合，每笔贷款的年违约概率为 2%，平均回收率为 60%（故单笔损失率 $\lambda=40\%$）。假设贷款之间的Copula相关系数为 $\rho=0.1$（单因子高斯Copula模型）。请计算该组合一年期的 **99.9%最坏违约率** 以及 **99.9%置信水平下的损失VaR**。
<!-- bilingual-en:start -->
**Practice question:** A bank holds a homogeneous retail-loan portfolio worth \$100 million. Each loan has a 2% annual probability of default, and the average recovery rate is 60%, so $\lambda=40\%$. Assume a one-factor Gaussian Copula with correlation $\rho=0.1$. Calculate the portfolio's one-year **99.9% worst-case default rate** and its **loss VaR at the 99.9% confidence level**.
<!-- bilingual-en:end -->

**详细解答：**首先确定参数：$p=0.02$，回收率$=60\%$，$\lambda=40\%$，置信水平$\alpha=99.9\%$，$\Phi^{-1}(0.999) \approx 3.0902$，$\Phi^{-1}(0.02) = \theta \approx -2.0537$。应用WCDR公式：
$$ 
x_{99.9\%} = \Phi\!\Big(\frac{-2.0537 + \sqrt{0.1}\times 3.0902}{\sqrt{1-0.1}}\Big)\,. 
$$ 
计算分步如下：
- $\sqrt{0.1} \times 3.0902 \approx 0.9773$，与 $-2.0537$ 相加得 $-1.0764$。
- $\sqrt{1-0.1} = \sqrt{0.9} \approx 0.9487$。
- 分数值为 $-1.0764/0.9487 \approx -1.1349$。
- 最后取标准正态CDF：$\Phi(-1.1349) = 0.1282$。
<!-- bilingual-en:start -->
**Detailed answer:** The parameters are $p=0.02$, recovery $=60\%$, $\lambda=40\%$, and confidence level $\alpha=99.9\%$. Also, $\Phi^{-1}(0.999) \approx 3.0902$ and $\Phi^{-1}(0.02) = \theta \approx -2.0537$. Applying the WCDR formula:
- $\sqrt{0.1} \times 3.0902 \approx 0.9773$, and adding this to $-2.0537$ gives $-1.0764$.
- $\sqrt{1-0.1} = \sqrt{0.9} \approx 0.9487$.
- The argument of the standard normal CDF is $-1.0764/0.9487 \approx -1.1349$.
- Finally, $\Phi(-1.1349) = 0.1282$.
<!-- bilingual-en:end -->

因此 **99.9%最坏违约率** $x_{99.9\%} \approx 12.8\%$。这意味着我们有99.9%的把握违约率不会超过12.8%。对应的信用组合 **99.9% VaR**（一年期损失）为：
$$ 
\text{VaR}_{99.9\%} = 100\,\text{百万} \times 12.8\% \times 40\% = 5.12\,\text{百万美元}\,. 
$$ 
换言之，在极端情况下该组合一年内最大可能损失约\$512万，占组合的5.12%。
<!-- bilingual-en:start -->
Therefore, the **99.9% worst-case default rate** is $x_{99.9\%} \approx 12.8\%$: according to the model, the portfolio default rate exceeds 12.8% with only 0.1% probability. Multiplying by \$100 million and a 40% LGD gives a one-year **99.9% VaR** of about \$5.12 million, or 5.12% of portfolio value.
<!-- bilingual-en:end -->

## Copula相关性与尾部风险
<!-- bilingual-en:start -->
*Copula Correlation and Tail Risk*
<!-- bilingual-en:end -->
高斯Copula模型假设资产间关联完全由线性相关系数 $\rho$ 控制。这种假设在描述**尾部关联性（Tail Dependence）**方面存在局限。**尾部相关**通常指在极端情况下（如一段分布尾部）变量同时发生极端变动的倾向。用定量描述，比如**上尾相关**系数可定义为：
$$ 
\lambda_U = \lim_{q \to 1^-} P\big(Y > F_Y^{-1}(q) \,\big|\, X > F_X^{-1}(q)\big) \,,
$$ 
表示当 $X$ 处于极高分位时 $Y$ 也极端偏大的概率（下尾类似定义）。
<!-- bilingual-en:start -->
The Gaussian Copula assumes that dependence is fully characterized by a linear correlation coefficient $\rho$. This is restrictive when modeling **tail dependence**, the tendency for variables to become extreme together. For example, upper-tail dependence measures the limiting conditional probability that $Y$ is also extremely high given that $X$ is extremely high; lower-tail dependence is defined analogously.
<!-- bilingual-en:end -->

对于高斯Copula，当相关系数 $\rho < 1$ 时，上尾相关和下尾相关系数实际上都为0。这意味着在Gaussian Copula模型中，**极端事件很少同时发生**：即使相关较高，发生罕见极端损失的情况下，另一风险因素同时极端的不概率趋于零。这与某些金融现象（如危机中多资产同步暴跌）不符。
<!-- bilingual-en:start -->
For a Gaussian Copula with $\rho < 1$, both upper- and lower-tail dependence coefficients are zero. This does not mean ordinary joint extremes are impossible; it means that their limiting conditional probability vanishes as the threshold moves farther into the tail. Even with high linear correlation, Gaussian dependence can therefore understate the clustering of rare losses observed when many assets fall together during a crisis.
<!-- bilingual-en:end -->

**尾部风险**是指金融资产在分布尾部发生共振（同时极端变化）的风险。高斯Copula由于尾部独立，往往低估了这种风险。例如，假设某组合年均违约概率 $PD=1\%$，在10年中有一年违约率达到3%。在高斯Copula单因子模型下，无论选择何种 $\rho$，都难以给予“一年出现3倍于平均违约的事件”以足够概率质量——因为正态因子很难产生如此厚尾的联合违约事件。这表明模型对尾部共灾的刻画不足。
<!-- bilingual-en:start -->
**Tail risk** here is the risk that several financial variables move into adverse distributional tails together. A Gaussian Copula may understate this risk because it is asymptotically tail-independent. For example, suppose a portfolio has average annual PD of 1% but records a 3% default rate in one year out of ten. A thin-tailed one-factor Gaussian model may assign too little probability to such clustered default outcomes, indicating that the model does not adequately capture common tail shocks.
<!-- bilingual-en:end -->

解决方案是采用具有更强尾部相关性的 Copula 模型，例如**$t$-Copula（学生t Copula）**。具体做法如：让单因子模型中的公共因子 $F$ 服从自由度较低的$t$分布（而非正态），或者直接使用多元$t$分布作为Copula基础。学生$t$分布相比正态有更肥厚的尾部，因而$t$-Copula 能产生**正的尾部相关性**：在极端情景下，多个风险变量**同时处于极端**的概率不再接近于0。这提高了模型对系统性尾部事件的捕捉能力。
<!-- bilingual-en:start -->
One remedy is a Copula with stronger tail dependence, such as a **$t$-Copula (Student's $t$ Copula)**. The common factor $F$ can be modeled with a low-degrees-of-freedom $t$ distribution rather than a normal distribution, or the Copula can be built directly from a multivariate $t$ distribution. Because Student's $t$ has heavier tails, the model assigns more probability to extreme common-factor realizations and hence to several risks becoming extreme together. This improves sensitivity to systemic tail events.
<!-- bilingual-en:end -->

**模拟考题：**为何单因子高斯Copula模型可能低估信用组合的尾部风险？什么是尾部相关性？举例说明采用厚尾Copula（如 $t$-Copula）如何改进对尾部共同违约事件的拟合。
<!-- bilingual-en:start -->
**Practice question:** Why can a one-factor Gaussian Copula underestimate the tail risk of a credit portfolio? What is tail dependence? Explain how a heavy-tailed Copula, such as a $t$-Copula, can fit clustered default events more effectively.
<!-- bilingual-en:end -->

**详细解答：**单因子高斯Copula假定公共因子 $F$ 为正态，从而各违约事件的关联主要体现在共同响应 $F$ 的线性部分。此模型下，极端尾部事件（例如大多数债务人在同一年违约）出现的概率非常低。一旦观察到比模型预测更频繁的极端事件，说明模型低估了尾部风险。
<!-- bilingual-en:start -->
**Detailed answer:** A one-factor Gaussian Copula assumes that the common factor $F$ is normal, so dependence among defaults arises through borrowers' shared linear exposure to $F$. Large common shocks, and therefore years in which many borrowers default together, receive very little probability. If extreme default years occur more frequently than the model predicts, the model is understating tail risk.
<!-- bilingual-en:end -->

**尾部相关性**指变量在极端尾部同时发生极端变动的相关程度。高斯Copula的尾部相关性为0（在 $\rho < 1$ 时），意味着例如 $P(X$ 极端下跌 $\land Y$ 极端下跌$)$相对于单边极端事件的条件概率趋于零。现实中金融资产往往存在尾部相关，例如市场崩盘时多数资产一起下跌、经济萧条时多家公司一同违约。
<!-- bilingual-en:start -->
**Tail dependence** measures the limiting tendency of variables to enter the same extreme tail together. For a Gaussian Copula, it is zero whenever $\rho < 1$. Thus the conditional probability of an extreme fall in $Y$, given an increasingly extreme fall in $X$, tends to zero as the threshold moves into the tail. Financial data often show stronger tail co-movement: many assets fall together in a crash, and many firms default together in a recession.
<!-- bilingual-en:end -->

采用更重尾的 Copula 可以缓解这一问题。比如**学生t-Copula**：令单因子模型中公共因子 $F$ 服从自由度$\nu$较低的$t$分布。$t$分布尾部衰减比正态慢，意味着 $F$ 有更大概率取极端值。这将导致多个 $U_i = \sqrt{\rho}F + \sqrt{1-\rho}Z_i$ 同时极端低的概率增加，即贷款的联合违约更容易发生。换言之，$t$-Copula 模型赋予组合违约分布更厚的尾部，使模型可以解释“PD=1%但偶尔违约率达3%”此类现象。总之，引入尾部相关性更强的Copula（通过选择厚尾分布的因子）能更好地拟合数据中观察到的尾部共倒现象，提高风险度量对极端情景的敏感度。
<!-- bilingual-en:start -->
A heavier-tailed Copula can mitigate this problem. In a **Student's $t$ Copula**, for example, the common factor $F$ has a $t$ distribution with relatively few degrees of freedom $\nu$. Its tails decay more slowly than normal tails, so extreme values of $F$ occur more often. Consequently, several latent variables $U_i = \sqrt{\rho}F + \sqrt{1-\rho}Z_i$ are more likely to be extremely low at the same time, increasing joint-default probability. The resulting portfolio default distribution has a heavier tail and can better accommodate observations such as an average PD of 1% accompanied by occasional 3% default years.
<!-- bilingual-en:end -->

## Copula 参数的极大似然估计方法
<!-- bilingual-en:start -->
*Maximum Likelihood Estimation of Copula Parameters*
<!-- bilingual-en:end -->
Copula模型通常包含需要估计的参数，例如单因子高斯Copula模型中的**违约概率** $PD$ 和**相关系数** $\rho$。给定历史数据，我们可以使用**极大似然估计（MLE）**来估计这些参数。
<!-- bilingual-en:start -->
Copula models contain parameters that must be estimated, such as the **probability of default** $PD$ and **correlation coefficient** $\rho$ in a one-factor Gaussian Copula. Given historical observations, these parameters can be estimated by **maximum likelihood estimation (MLE)**.
<!-- bilingual-en:end -->

以违约率数据为例：假设我们观测到 $T$ 年中每年的组合违约率 $DR_1, DR_2, \dots, DR_T$。在单因子模型假设下，这些违约率服从一个由 $(PD,\;\rho)$ 参数决定的分布（即 Vasicek 分布）。记 $G(DR)$ 为违约率的累计分布函数（CDF），$g(DR)$ 为相应的概率密度函数（PDF）。MLE 方法步骤如下：
<!-- bilingual-en:start -->
Suppose we observe annual portfolio default rates $DR_1, DR_2, \dots, DR_T$ over $T$ years. Under the one-factor model, they follow a Vasicek distribution determined by $(PD,\;\rho)$. Let $G(DR)$ denote the CDF and $g(DR)$ the corresponding PDF. The MLE procedure is:
<!-- bilingual-en:end -->

1. **初始猜测：**先对 $PD$ 和 $\rho$ 选取一个初始猜测值（例如 $PD$ 可用历史平均违约率，$\rho$ 可从资产相关性经验取值）。
2. **构建似然函数：**写出 **对数似然函数** $\ell(PD,\rho) = \sum_{t=1}^T \ln\! \big[g(DR_t; PD,\rho)\big]$。其中 $g(DR_t; PD,\rho)$ 可以通过微分 Copula分布函数得到，其形式较复杂（略去推导）。关键是 $g(DR)$ 会包含 $PD$ 和 $\rho$ 非线性组合，比如模型推导出的密度一般形如：
   $$ 
   g(DR) = \frac{1}{\sqrt{2\pi(1-\rho)}} \exp\!\Big\{-\frac{1}{2(1-\rho)}\Big[\Phi^{-1}(DR) - \Phi^{-1}([[信用风险：PD、LGD、EAD 与评级迁移|PD]])\sqrt{\rho}\Big]^2\Big\} \times \frac{1}{DR'(F)} \,,
   $$ 
   其中最后一项是从违约率分布的隐函数中求导的雅可比项。这一密度函数对应 $DR$ 在单因子模型下的分布（无需学生完整记忆公式，理解其随参数变化即可）。
3. **最大化：**通过数学优化方法，找到使对数似然 $\ell(PD,\rho)$ 最大的参数值 $(\hat{PD}, \hat{\rho})$。通常需要借助数值算法迭代搜索，因为对数似然对参数的一阶条件方程一般无法解析解出。
4. **结果检验：**得到 MLE 参数后，可检验拟合优度，或利用它们计算感兴趣的风险量（如$99.9\%$违约率分位值）。
<!-- bilingual-en:start -->

&nbsp;
**1.** **Choose starting values:** Use an initial value for $PD$, such as the historical average default rate, and a plausible starting value for $\rho$.<br>
**2.** **Construct the likelihood:** Write the **log-likelihood** $\ell(PD,\rho) = \sum_{t=1}^T \ln\! \big[g(DR_t; PD,\rho)\big]$. The density $g(DR_t; PD,\rho)$ follows from differentiating the Vasicek CDF and includes a Jacobian term from the transformation between the systematic factor and the observed default rate.<br>
**3.** **Maximize it:** Numerically search for $(\hat{PD}, \hat{\rho})$ that maximizes $\ell(PD,\rho)$. Closed-form first-order conditions are generally unavailable.<br>
**4.** **Check and use the result:** Assess goodness of fit and use the estimates to calculate quantities such as the 99.9% default-rate quantile.<br>
<!-- bilingual-en:end -->

实际操作中，可以使用软件对历史违约率序列进行MLE拟合。例如，某信用卡组合过去10年数据，通过MLE得到估计 $PD \approx 1.34\%$，$\rho \approx 0.11$。基于此，我们可以绘制出模型拟合的违约率分布，并从中读出99.9%分位违约率约10.4%【对应之前WCDR公式的结果】。
<!-- bilingual-en:start -->
In practice, software is used to fit the MLE to a historical default-rate series. For example, a ten-year credit-card portfolio sample may produce estimates of $PD \approx 1.34\%$ and $\rho \approx 0.11$. The fitted default-rate distribution can then be plotted, and its 99.9th percentile—about 10.4% in this illustration—can be read or calculated using the WCDR formula.
<!-- bilingual-en:end -->

**模拟考题：**给定过去5年的某贷款组合违约率数据：$ \{2.1\%,\;0.5\%,\;1.4\%,\;3.0\%,\;0.8\%\}$，试说明如何利用极大似然估计来推断单因子Copula模型的违约概率 $PD$ 以及相关参数 $\rho$。简单描述估计步骤并指出计算中涉及的关键公式。
<!-- bilingual-en:start -->
**Practice question:** The annual default rates of a loan portfolio over the past five years are $ \{2.1\%,\;0.5\%,\;1.4\%,\;3.0\%,\;0.8\%\}$. Explain how maximum likelihood can be used to estimate the one-factor Copula parameters $PD$ and $\rho$. State the estimation steps and the key formulas.
<!-- bilingual-en:end -->

**详细解答：**首先构建单因子高斯Copula下违约率的分布模型。其CDF可以表示为：
$$ 
P(DR \le x) = \Phi\!\Big(\frac{\Phi^{-1}(x) - \Phi^{-1}(PD)}{\sqrt{\rho}}\Big)\,,
$$ 
据此可推导PDF（略去繁琐推导）。估计步骤如下：
1. **初值选取：**用历史违约率的均值作为初始 $PD$（如上述数据平均违约率约$1.56\%$），相关性 $\rho$ 初始可取一个小值（比如0.1）。
2. **构建似然：**假设每年违约率独立（年份之间近似独立），则总似然 $L(PD,\rho) = \prod_{t=1}^5 g(DR_t; PD,\rho)$。取对数：
   $$ 
   \ell(PD,\rho) = \sum_{t=1}^5 \ln g(DR_t; PD,\rho)\,. 
   $$ 
   需要将每个观测违约率代入模型PDF $g(DR; PD,\rho)$。这涉及例如将 $DR_t$ 反算为对应因子 $F_t$ 的值：$F_t = \frac{\Phi^{-1}(DR_t) - \Phi^{-1}(PD)}{\sqrt{1-\rho}}$，然后代入正态密度计算。
3. **优化求解：**通过数值方法调整 $PD$ 和 $\rho$，反复计算 $\ell(PD,\rho)$，使其最大化。可以采用梯度上升法或内置优化算法。最终得到的参数即为 $\hat{PD}, \hat{\rho}$。
4. **结果与应用：**将估计的参数代入模型，即得到该组合的违约分布。可以进一步算出$99.9\%$分位违约率用于风险计算。比如（假设）估计结果 $\hat{PD}=1.5\%, \hat{\rho}=0.12$，则99.9%违约率 $\approx \Phi\!\Big(\frac{\Phi^{-1}(0.015)+\sqrt{0.12}\times 3.09}{\sqrt{0.88}}\Big)$，可得相应极端违约水平，用以评估所需经济资本等。
<!-- bilingual-en:start -->
**Detailed answer:** First specify the Vasicek distribution implied by the one-factor Gaussian Copula and derive its PDF from the CDF shown above.
**1.** **Choose starting values:** Use the sample mean default rate, approximately $1.56\%$, as the initial $PD$, and start $\rho$ at a small positive value such as 0.1.<br>
**2.** **Construct the likelihood:** If annual observations are treated as independent, use $L(PD,\rho) = \prod_{t=1}^5 g(DR_t; PD,\rho)$ and its log form. Under the stated model, the consistent inversion is $F_t=[\Phi^{-1}(PD)-\sqrt{1-\rho}\,\Phi^{-1}(DR_t)]/\sqrt{\rho}$; this corrects the inconsistent inversion printed in the source text. The normal factor density and the transformation Jacobian together determine $g(DR_t;PD,\rho)$.<br>
**3.** **Optimize numerically:** Vary $PD$ and $\rho$ to maximize $\ell(PD,\rho)$, using a constrained optimizer so that both parameters remain in their admissible ranges.<br>
**4.** **Apply the estimates:** Substitute $\hat{PD}$ and $\hat{\rho}$ into the fitted default-rate distribution. For illustration, if $\hat{PD}=1.5\%$ and $\hat{\rho}=0.12$, the 99.9% default-rate quantile is $\Phi\!\big((\Phi^{-1}(0.015)+\sqrt{0.12}\times 3.09)/\sqrt{0.88}\big)$ and can be used to assess economic capital.<br>
<!-- bilingual-en:end -->

以上步骤概括了利用MLE标定Copula模型参数的过程。在实际计算中应使用软件，以确保计算精度和搜索效率。
<!-- bilingual-en:start -->
These steps summarize how MLE calibrates a Copula model. In an actual application, numerical software should be used both for accuracy and for efficient constrained optimization.
<!-- bilingual-en:end -->

# 作业
<!-- bilingual-en:start -->
*Homework*
<!-- bilingual-en:end -->

## 11.6

>[!question] 
>假定资产X和Y的当前日波动率分别为1.0%和1.2%，上个交易日结束时资产价格分别为30美元和50美元，资产回报的相关系数为0.5。在这里我们采用GARCH(1，1)模型来计算更新相关系数及波动率，[[条件异方差：ARCH 与 GARCH|GARCH]](1，1)模型中的参数估计为a=0.04及B=0.94，在相关系数估计中采用w=0.000 001，在波动率估计中采用w=0.000003，假如在今天交易结束时，资产价格分别为31美元和51美元，相关系数的最新估计为多少?
><!-- bilingual-en:start -->
>Assume that the current daily volatilities of assets X and Y are 1.0% and 1.2%, their previous closing prices were \$30 and \$50, and the correlation of their returns was 0.5. Use a [[条件异方差：ARCH 与 GARCH|GARCH]](1,1)-style update with $a=0.04$ and $B=0.94$, taking $w=0.000001$ for covariance and $w=0.000003$ for each variance. If today's closing prices are \$31 and \$51, what is the updated correlation estimate?
><!-- bilingual-en:end -->

逻辑:
1. 老波动率和今日波动率推新波动率
2. 老协方差和今日波动率推新协方差
3. 新协方差除以新波动率得到新相关系数
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
\text{cov}_{\text{new}} = \omega + \alpha, r_X r_Y + \beta, \text{cov}_{\text{old}}
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
\text{cov}_{\text{new}} = 0.000003 + 0.00002596 + 0.0000564 = 0.00008536
$$
 **最新相关系数**
$$
\rho_{XY,\text{new}} = \frac{\text{cov}_{\text{new}}}{\sigma_{X,\text{new}} \times \sigma_{Y,\text{new}}}
= \frac{0.00008536}{0.01175 \times 0.01233} \approx \frac{0.00008536}{0.00014487} \approx 0.589
$$
<!-- bilingual-en:start -->
The new covariance should use the stated covariance intercept $\omega=0.000001$. This gives $0.000001+0.04(0.03279)(0.01980)+0.94(0.00006)\approx0.00008337$. Combining it with the updated volatilities shown above, approximately 0.01183 and 0.01241, gives an updated correlation of about **0.568**. The source's values 0.00008536 and 0.589 mix the variance intercept with the covariance recursion and also use different volatility denominators, so they are not internally consistent.
<!-- bilingual-en:end -->

## 11.9

>[!question] 
>假定你有3个相互独立并服从正态分布的变量z1、z2、z3，你想将这3组变量由cholesky分解来产生服从三元正态分布的随机变量“$\epsilon$ 1 、$\epsilon$ 2、$\epsilon$ 3
> <!-- bilingual-en:start -->
> Assume that $z_1,z_2,z_3$ are mutually independent standard normal variables. Use a Cholesky decomposition to construct trivariate normal variables $\epsilon_1,\epsilon_2,\epsilon_3$.
> <!-- bilingual-en:end -->
请求出由z1、z2、z3及变量之间的相关系数组成的$\epsilon$ 1、$\epsilon$ 2、$\epsilon$ 3的表达式。
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

任何协方差矩阵$\Sigma$都可以Cholesky分解为
$$
\Sigma = L L^\top
$$
其中$L$是下三角矩阵。
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

假设三变量的相关系数分别为$\rho_{12},,\rho_{13},,\rho_{23}$，则
<!-- bilingual-en:start -->
If the three pairwise correlations are $\rho_{12},,\rho_{13},,\rho_{23}$, then the correlation matrix is the one shown above.
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

$L$具体如下（可以记公式）：
$$
L = \begin{pmatrix}
1 & 0 & 0 \\
\rho_{12} & \sqrt{1-\rho_{12}^2} & 0 \\
\rho_{13} & \frac{ \rho_{23} - \rho_{12}\rho_{13} }{ \sqrt{1-\rho_{12}^2} } & \sqrt{ 1-\rho_{13}^2 - \left( \frac{ \rho_{23} - \rho_{12}\rho_{13} }{ \sqrt{1-\rho_{12}^2} } \right)^2 }
\end{pmatrix}
$$
<!-- bilingual-en:start -->
The lower-triangular factor $L$ is shown above. This formula may be memorized, but it is more important to understand that it is obtained by solving $\Sigma=LL^\top$ row by row.
<!-- bilingual-en:end -->

 **$\epsilon_1,,\epsilon_2,,\epsilon_3$的具体表达式**
<!-- bilingual-en:start -->
**Explicit Expressions for $\epsilon_1,,\epsilon_2,,\epsilon_3$**
<!-- bilingual-en:end -->

根据$\epsilon = L z$，逐项写出：
<!-- bilingual-en:start -->
Using $\epsilon = L z$, write out each component as shown above.
<!-- bilingual-en:end -->

$$\begin{aligned} \epsilon_1 &= z_1 \\ \epsilon_2 &= \rho_{12}z_1 + \sqrt{1-\rho_{12}^2}z_2 \\ \epsilon_3 &= \rho_{13}z_1 + \frac{ \rho_{23} - \rho_{12}\rho_{13} }{ \sqrt{1-\rho_{12}^2} } z_2 + \sqrt{ 1-\rho_{13}^2 - \left( \frac{ \rho_{23} - \rho_{12}\rho_{13} }{ \sqrt{1-\rho_{12}^2} } \right)^2 } z_3 \end{aligned}$$

## 11.14

>[!question] 
>假定银行有一笔大数量的贷款，每笔贷款每年的违约概率为1.5%，违约时的回收率为30%，银行采用高斯copula来模拟违约时间。请使用vasicek模型来估计99.5%置信度下的违约率。假设Copula相关系数为0.2。
><!-- bilingual-en:start -->
>Assume a bank holds a very large portfolio of loans. Each loan has an annual probability of default of 1.5% and a recovery rate of 30%. The bank models default dependence with a Gaussian Copula. Use the Vasicek model to estimate the portfolio default rate at the 99.5% confidence level, assuming a Copula correlation of 0.2.
><!-- bilingual-en:end -->

- 贷款数目很大（$n\to\infty$，可认为“连续”）
- 每笔贷款年违约概率 $p=1.5\% = 0.015$
- 违约时回收率 $=30\%$（其实计算违约率时不用）
- Copula相关系数 $\rho=0.2$
- 计算“99.5%置信度下的违约率”（即，极端情况下的贷款池**总体违约率99.5%分位点**）
<!-- bilingual-en:start -->
- The number of loans is very large, so the portfolio can be treated as asymptotically granular.
- Each loan has annual default probability $p=1.5\% = 0.015$.
- Recovery at default is $=30\%$; it is irrelevant when calculating the default-rate quantile itself.
- The Copula correlation is $\rho=0.2$.
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
- 单因子Copula相关系数 $\rho$
- 组合违约率的$q$分位点为$L_q$
<!-- bilingual-en:start -->
- $p$ be the annual default probability of each loan.
- $\rho$ be the one-factor Copula correlation.
- $L_q$ be the $q$th quantile of the portfolio default rate.
<!-- bilingual-en:end -->

**分位点计算公式为：**
<!-- bilingual-en:start -->
**The quantile formula is:**
<!-- bilingual-en:end -->

$$

L_q = \Phi\left( \frac{ \Phi^{-1}(p) + \sqrt{\rho}, \Phi^{-1}(q) }{ \sqrt{1-\rho} } \right)

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

- $\Phi^{-1}(0.015)\approx -2.17$
- $\Phi^{-1}(0.995)\approx 2.58$
- $\sqrt{\rho}=\sqrt{0.2}\approx 0.447$
- $\sqrt{1-\rho}=\sqrt{0.8}\approx 0.894$
    
代入公式：
$$
L_{0.995} = \Phi\left(
\frac{-2.17 + 0.447\times 2.58}{0.894}
\right)
$$
- $0.447 \times 2.58 \approx 1.153$
- $-2.17 + 1.153 = -1.017$    
- $-1.017 / 0.894 \approx -1.138$
<!-- bilingual-en:start -->
Substitute into the formula:
- $0.447 \times 2.58 \approx 1.153$.
- $-2.17 + 1.153 = -1.017$.
- $-1.017 / 0.894 \approx -1.138$.
<!-- bilingual-en:end -->



查标准正态分布表：
<!-- bilingual-en:start -->
Using a standard normal table:
<!-- bilingual-en:end -->

- $\Phi(-1.138)\approx 0.127$
    
**最终答案**
<!-- bilingual-en:start -->
**Final Answer**
<!-- bilingual-en:end -->

- 99.5%置信度下的贷款组合违约率为
    $$
    \boxed{12.7\%}
    $$
<!-- bilingual-en:start -->
- At the 99.5% confidence level, the portfolio default rate is
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

查表/用Python可得（四舍五入保留三位小数）：
<!-- bilingual-en:start -->
The values can be obtained from a standard normal table or with Python and are rounded to three decimal places.
<!-- bilingual-en:end -->

|**$l_t$**|**$Y_t = \Phi^{-1}(l_t)$**|
|---|---|
|0.01|$-2.326$|
|0.09|$-1.340$|
|0.02|$-2.054$|
|0.03|$-1.881$|
|0.05|$-1.645$|
|0.01|$-2.326$|
|0.06|$-1.555$|
|0.07|$-1.475$|
|0.04|$-1.751$|
|0.01|$-2.326$|

 **3. 计算均值和方差**
<!-- bilingual-en:start -->
**3. Calculate the Mean and Variance**
<!-- bilingual-en:end -->

**均值：**
$$
\bar{Y} = \frac{1}{10} \sum_{t=1}^{10} Y_t = \frac{-2.326-1.340-2.054-1.881-1.645-2.326-1.555-1.475-1.751-2.326}{10}
$$
$$
\bar{Y} = \frac{-18.679}{10} = -1.868
$$
**方差：**
$$
s_Y^2 = \frac{1}{9}\sum_{t=1}^{10}(Y_t-\bar{Y})^2
$$
$$
s_Y^2 = \frac{1.260}{9} = 0.140
$$
 **4. 得到 Vasicek 模型最大似然参数**
<!-- bilingual-en:start -->
**Mean:** Use the average of the ten transformed observations shown above.

**Variance:** Distinguish the maximum-likelihood variance, which divides by 10, from the unbiased sample variance, which divides by 9.

**4. Obtain the Vasicek-Model Parameter Estimates**
<!-- bilingual-en:end -->

 **(1) 系统相关性参数**
$$
\boxed{\hat{\rho} = 0.140}
$$
<!-- bilingual-en:start -->
**(1) Systematic-Correlation Parameter**
<!-- bilingual-en:end -->

 **(2) 长期违约率参数**
$$
\hat{p} = \Phi(\bar{Y}) = \Phi(-1.868) \approx 0.0309 = 3.1\%
$$
 **5. 最终结论**
<!-- bilingual-en:start -->
**(2) Long-Run Default-Probability Parameter**

**5. Final Conclusion**
<!-- bilingual-en:end -->

- **Vasicek模型相关性参数最大似然估计值为** $\boxed{0.14}$
- **长期平均违约率最大似然估计为约** $\boxed{3.1\%}$
<!-- bilingual-en:start -->
- The shortcut in the source reports a Vasicek correlation estimate of $\boxed{0.14}$.
- It also reports a long-run default probability of approximately $\boxed{3.1\%}$.

These are not the exact MLEs under the stated Vasicek transformation. If $Y_t=\Phi^{-1}(l_t)$, then $\operatorname{Var}(Y)=\rho/(1-\rho)$ and $E[Y]=\Phi^{-1}(p)/\sqrt{1-\rho}$. Using the MLE variance $1.2584/10\approx0.12584$ gives $\hat\rho\approx0.1118$ and $\hat p\approx3.92\%$. Using the unbiased variance $1.2584/9\approx0.13982$ instead gives $\hat\rho\approx0.1227$ and $\hat p\approx4.01\%$. Thus 0.14 and 3.1% result from treating the transformed variance directly as $\rho$ and ignoring the scaling in the transformed mean.
<!-- bilingual-en:end -->

## 11.16

>[!question] 
>假定在上个交易日结束时某资产X的价格为300美元，价格波动率为每天1.3%，今天X的价格在交易结束时为298美元，假定在上个交易日结束时资产Y的价格为8美元，价格波动率为每天1.5%。Y的价格与X的价格的相关系数为0.8。今天在交易结束时Y的价格同昨天相同，即8美元。请求出最新的X价格及Y价格的波动率及相关系数，在计算中请采用:(a)EWMA模型，参数为$\lambda$=0.94:(b)GARCH(1，1)模型，其中模型参数如$\omega$=0.000 002、a=0.04及 $\beta$-0.94。在实践中，对于X和Y的。参数是否相同?
><!-- bilingual-en:start -->
>At the end of the previous trading day, asset X had a price of \$300 and daily volatility of 1.3%; today it closes at \$298. Asset Y had a price of \$8, daily volatility of 1.5%, and return correlation of 0.8 with X; today it again closes at \$8. Calculate the updated volatilities of X and Y and their updated correlation using (a) an EWMA model with $\lambda$=0.94 and (b) a GARCH(1,1) model with $\omega$=0.000002, $a=0.04$, and $\beta=0.94$. In practice, should X and Y use the same parameter values?
><!-- bilingual-en:end -->

 **1. 收益率计算**
<!-- bilingual-en:start -->
**1. Calculate Returns**
<!-- bilingual-en:end -->

- $r_X = \ln\left(\frac{298}{300}\right) \approx -0.006684$
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
    \sigma_{X,\text{new}}^2 = 0.00015886 + 0.000002682 = 0.00016154
    $$
    $$  
    \sigma_{X,\text{new}} = \sqrt{0.00016154} \approx 0.01271 = 1.27\%
    $$


**$Y$：**
$$ 
    \sigma_{Y,\text{new}}^2 = 0.0002115 + 0 = 0.0002115  
    $$
    $$
    \sigma_{Y,\text{new}} = \sqrt{0.0002115} \approx 0.01454 = 1.45\%
    $$

 **相关系数(EWMA-协方差)**
<!-- bilingual-en:start -->
**Correlation Coefficient (EWMA Covariance Update)**
<!-- bilingual-en:end -->

- 上日协方差：$0.8\times0.013\times0.015=0.000156$
- $r_X r_Y = -0.006684 \times 0 = 0$
- $0.94\times0.000156 = 0.00014664$
    $$
    \text{cov}_{\text{new}} = 0.00014664
    $$
    $$
    \rho_{\text{new}} = \frac{0.00014664}{0.01271\times0.01454} = \frac{0.00014664}{0.00018477} \approx 0.794
    $$
<!-- bilingual-en:start -->
- Previous covariance: $0.8\times0.013\times0.015=0.000156$.
- Today's return cross-product: $r_X r_Y = -0.006684 \times 0 = 0$.
- Decayed previous covariance: $0.94\times0.000156 = 0.00014664$.
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
    \sigma_{X,\text{new}}^2 = 0.000002 + 0.000001788 + 0.00015886 = 0.000162648
    $$
    
    $$
    \sigma_{X,\text{new}} = \sqrt{0.000162648} \approx 0.01276 = 1.28\%
    $$

**$Y$：**

    $$ 
    \sigma_{Y,\text{new}}^2 = 0.000002 + 0 + 0.0002115 = 0.0002135
    $$
    $$
    \sigma_{Y,\text{new}} = \sqrt{0.0002135} \approx 0.01461 = 1.46\%
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
    \rho_{\text{new}} = \frac{0.00014864}{0.01276\times0.01461} = \frac{0.00014864}{0.00018648} \approx 0.797
    $$

 **4. $\omega$参数是否相同？**
<!-- bilingual-en:start -->
**4. Should the $\omega$ Parameters Be the Same?**
<!-- bilingual-en:end -->

- $\omega$参数是控制模型长期均值（长期方差/协方差）的参数。
- **在实际建模时，$X$和$Y$各自的$\omega$参数通常要**单独校准（不一定相同），以贴合不同资产的历史波动水平。**
- 只有在教学或简化题目时，才会人为设置为相同。
<!-- bilingual-en:start -->
- The $\omega$ parameter helps determine the model's long-run variance or covariance level.
- In practice, the $\omega$ parameters for $X$ and $Y$ are normally calibrated **separately**, because different assets have different long-run volatility levels.
- Equal values are generally imposed only in teaching examples or deliberately simplified models.
<!-- bilingual-en:end -->


## 11.19

计算由软件实现数值积分与矩阵运算得到（可用脚本重现）。~~电脑算的,略.~~
<!-- bilingual-en:start -->
The numerical integration and matrix operations can be performed in software and reproduced with a script. ~~The omitted computation was delegated to a computer.~~
<!-- bilingual-en:end -->

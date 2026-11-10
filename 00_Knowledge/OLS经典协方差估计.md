---
aliases:
  - Classical OLS covariance estimation
  - OLS 系数协方差矩阵
student_os: knowledge-atom
atom_id: ECON-OLS-016
atom_set: regression-inference
atom_type: proposition
status: source-checked
part_of:
  - "[[回归推断.canvas|回归推断]]"
requires:
  - "[[满列秩与OLS唯一性]]"
  - "[[经典 Gauss–Markov 定理]]"
leads_to:
  - "[[回归t检验]]"
  - "[[线性组合推断]]"
---

<!-- bilingual-en:start -->
# Classical OLS covariance estimation
<!-- bilingual-en:end -->

*OLS 经典协方差估计*

<!-- bilingual-en:start -->
Fitting a regression gives coefficient estimates. To say how much those estimates would vary across repeated samples, we also need their **[[协方差矩阵|covariance matrix]]**: its diagonal entries describe each coefficient's variance, and its off-diagonal entries describe how estimation errors move together. The classical OLS estimate combines two ingredients: the noise remaining after fitting and the information supplied by the design matrix.
<!-- bilingual-en:end -->

拟合回归得到系数估计值。要说明重复抽样时这些估计值会波动多大，还需要它们的**[[协方差矩阵]]**：对角元描述各系数的方差，非对角元描述估计误差如何一起变动。经典 OLS 估计把两种信息结合起来：拟合后剩余的噪声，以及设计矩阵提供的信息。

<!-- bilingual-en:start -->
## Begin with a fitted line
<!-- bilingual-en:end -->

*先看一条拟合直线*

<!-- bilingual-en:start -->
Observe outcomes $1,3,2$ at predictor values $t=0,1,2$. Fit an intercept and slope, $y_i\approx\beta_0+\beta_1t_i$, by [[普通最小二乘|ordinary least squares]], which minimises the sum of squared residuals. The fitted line is $\hat y_i=3/2+t_i/2$. Its predictions are $3/2,2,5/2$, leaving [[回归残差|residuals]] $-1/2,1,-1/2$ and residual sum of squares $RSS=3/2$.
<!-- bilingual-en:end -->

在解释变量 $t=0,1,2$ 处，观测结果分别是 $1,3,2$。用[[普通最小二乘]]拟合含截距、斜率的直线 $y_i\approx\beta_0+\beta_1t_i$，即最小化残差平方和，得到 $\hat y_i=3/2+t_i/2$。预测值为 $3/2,2,5/2$，[[回归残差]]为 $-1/2,1,-1/2$，残差平方和为 $RSS=3/2$。

<!-- bilingual-en:start -->
The residuals show the unexplained variation in this sample. They do not directly give the slope's uncertainty: the spread and scale of the predictor values also matter. The calculation below shows how both enter, and which assumptions justify using it.
<!-- bilingual-en:end -->

残差显示这次样本里未被拟合解释的变动，但不能直接当作斜率的不确定性，因为解释变量的分散程度与计量尺度也有影响。下面的计算把这两方面放在一起，并说明它依赖哪些假设。

<!-- bilingual-en:start -->
## The covariance being estimated
<!-- bilingual-en:end -->

*到底在估计哪个协方差？*

<!-- bilingual-en:start -->
Write the real model $y=X\beta+u$. There are $n$ observations and $k\ge1$ coefficients, so $X$ is $n\times k$. Each row holds one observation's regression terms; an intercept column counts among the $k$ columns. Assume [[满列秩与OLS唯一性|full column rank]], so $G=X^TX$ is invertible. Here $T$ denotes [[矩阵转置|transpose]]. Fix $X$, and assume $E(u\mid X)=0$ and $\operatorname{Cov}(u\mid X)=\sigma^2I_n$, with $0<\sigma^2<\infty$. This means equal finite conditional error variances and zero conditional covariances between distinct errors.
<!-- bilingual-en:end -->

写出实数模型 $y=X\beta+u$。共有 $n$ 个观测、$k\ge1$ 个系数，所以 $X$ 是 $n\times k$；每行放一个观测的回归项，截距列若存在，也计入这 $k$ 列。假设[[满列秩与OLS唯一性|满列秩]]，于是 $G=X^TX$ 可逆，其中 $T$ 表示[[矩阵转置|转置]]。固定 $X$ 后，假设 $E(u\mid X)=0$、$\operatorname{Cov}(u\mid X)=\sigma^2I_n$，且 $0<\sigma^2<\infty$，即误差的条件方差相同且有限，不同误差的条件协方差为零。

<!-- bilingual-en:start -->
The coefficient error is a linear transformation of the model errors: $\hat\beta-\beta=G^{-1}X^Tu$. Applying [[协方差仿射变换|covariance propagation]] gives
<!-- bilingual-en:end -->

系数估计误差是模型误差的线性变换：$\hat\beta-\beta=G^{-1}X^Tu$。按[[协方差仿射变换|协方差的线性传播]]，有

$$
V:=\operatorname{Cov}(\hat\beta\mid X)
=G^{-1}X^T(\sigma^2I_n)XG^{-1}
=\sigma^2G^{-1}.
$$

<!-- bilingual-en:start -->
$V$ describes repeated samples with this design, not the variation among the entries of one observed coefficient vector. If $X$ is random, the formula is conditional on almost every full-rank realisation satisfying the moment conditions. It is not automatically an unconditional covariance formula.
<!-- bilingual-en:end -->

$V$ 描述使用这个设计重复抽样时的波动，不是某次系数向量内部各元素的差异。若 $X$ 随机，公式是对几乎每个满足矩条件的满列秩实现作条件表述，不能自动当作无条件协方差公式。

<!-- bilingual-en:start -->
## Why divide the residual sum by n minus k?
<!-- bilingual-en:end -->

*为什么残差平方和要除以 n−k？*

<!-- bilingual-en:start -->
The unknown error variance $\sigma^2$ must be estimated. The fitted residual vector is $\hat u=y-X\hat\beta=My$, where
<!-- bilingual-en:end -->

未知的误差方差 $\sigma^2$ 需要估计。拟合残差向量为 $\hat u=y-X\hat\beta=My$，其中

$$M=I_n-XG^{-1}X^T.$$

<!-- bilingual-en:start -->
$M$ is the [[正交投影|orthogonal projection]] that removes the part explained by the columns of $X$. It leaves an $n-k$ dimensional residual space. Because $MX=0$, the residual is also $Mu$; because projecting twice changes nothing, $M^TM=M$. Hence $RSS=\hat u^T\hat u=u^TMu$.
<!-- bilingual-en:end -->

$M$ 是去掉 $X$ 各列所能解释部分的[[正交投影]]，留下一个 $n-k$ 维的残差空间。由于 $MX=0$，残差也等于 $Mu$；投影两次不会继续改变结果，所以 $M^TM=M$，从而 $RSS=\hat u^T\hat u=u^TMu$。

<!-- bilingual-en:start -->
Expand the quadratic form as $\sum_{i,j}M_{ij}u_i u_j$. Conditional means are zero, so $E(u_i u_j\mid X)$ is zero for $i\ne j$ and equals $\sigma^2$ for $i=j$. Only the diagonal terms remain. Their sum is the [[迹与特征值总和|trace]] $\operatorname{tr}M=n-k$: a projection has one eigenvalue $1$ for each direction it retains and $0$ for each it removes. Therefore
<!-- bilingual-en:end -->

把二次型展开为 $\sum_{i,j}M_{ij}u_i u_j$。条件均值为零，所以当 $i\ne j$ 时 $E(u_i u_j\mid X)=0$，当 $i=j$ 时等于 $\sigma^2$，只剩下对角项。对角元之和就是[[迹与特征值总和|迹]] $\operatorname{tr}M=n-k$：投影每保留一个方向就对应一个特征值 $1$，每去掉一个方向对应一个 $0$。因此

$$E(RSS\mid X)=\sigma^2\operatorname{tr}M=(n-k)\sigma^2.$$

> [!important] Estimate the error variance, then propagate it · 先估误差方差，再传播到系数
> <!-- bilingual-en:start -->
> If **$n>k$**, define
> <!-- bilingual-en:end -->
>
> 当 **$n>k$** 时，定义
>
> $$s^2=\frac{RSS}{n-k},\qquad\widehat V=s^2(X^TX)^{-1}.$$
>
> <!-- bilingual-en:start -->
> Under the stated conditional moment assumptions, $E(s^2\mid X)=\sigma^2$ and $E(\widehat V\mid X)=V$. These are [[估计量无偏性|unbiasedness]] statements over repeated samples. They do not mean the estimates equal their targets in this sample, and they require no normality assumption.
> <!-- bilingual-en:end -->
>
> 在上述条件矩假设下，$E(s^2\mid X)=\sigma^2$、$E(\widehat V\mid X)=V$。这些是针对重复抽样的[[估计量无偏性|无偏性]]结论，不表示这次样本的估计恰好等于真值，也不要求正态性。

<!-- bilingual-en:start -->
The denominator counts observations minus fitted coefficients. Here $k$ includes the intercept; a book that counts only nonconstant predictors would write $n-k-1$ using its different meaning of $k$. With $100$ observations, an intercept and three other predictors, there are four coefficients, so the denominator is $100-4=96$, not $99$. At $n=k$, a full-rank fit has $M=0$ and $RSS=0$ for every outcome. The expression $RSS/(n-k)$ is then undefined; a perfect fit supplies no residual information about the noise variance.
<!-- bilingual-en:end -->

分母数的是观测数减去拟合系数数目。本卡 $k$ 包含截距；若教材只把非常数解释变量计入 $k$，就会按它自己的记号写成 $n-k-1$。例如，$100$ 个观测、一个截距加三个解释变量，共四个系数，分母是 $100-4=96$，不是 $99$。当 $n=k$ 时，满秩拟合对每个结果都有 $M=0$、$RSS=0$；此时 $RSS/(n-k)$ 无法定义，完美拟合没有提供可用来估计噪声方差的残差信息。

<!-- bilingual-en:start -->
## Complete the three-observation calculation
<!-- bilingual-en:end -->

*把三个观测的例子算完*

<!-- bilingual-en:start -->
For the opening example, the first column of $X$ supplies the intercept and the second contains $0,1,2$. In coefficient order $(\beta_0,\beta_1)$,
<!-- bilingual-en:end -->

开篇例子中，$X$ 第一列提供截距，第二列放 $0,1,2$。按系数顺序 $(\beta_0,\beta_1)$，

$$
X=\begin{bmatrix}1&0\\1&1\\1&2\end{bmatrix},\qquad
G=\begin{bmatrix}3&3\\3&5\end{bmatrix},\qquad
G^{-1}=\begin{bmatrix}5/6&-1/2\\-1/2&1/2\end{bmatrix}.
$$

<!-- bilingual-en:start -->
There is one residual degree of freedom, so $s^2=(3/2)/(3-2)=3/2$. The estimated coefficient covariance is
<!-- bilingual-en:end -->

剩余一个残差自由度，所以 $s^2=(3/2)/(3-2)=3/2$，系数协方差的估计为

$$
\widehat V=\begin{bmatrix}5/4&-3/4\\-3/4&3/4\end{bmatrix},\qquad
se(\hat\beta_0)=\frac{\sqrt5}{2},\qquad se(\hat\beta_1)=\frac{\sqrt3}{2}.
$$

<!-- bilingual-en:start -->
A reported [[标准误含义|standard error]] estimates the sampling standard deviation: take the square root of the relevant diagonal entry. The off-diagonal $-3/4$ estimates the covariance between intercept and slope errors; covariance can be negative. For the uncertainty of a coefficient combination, the off-diagonal entries also matter, as in [[线性组合推断|linear-combination inference]].
<!-- bilingual-en:end -->

报告的[[标准误含义|标准误]]估计的是抽样标准差，要对相应对角元开平方。非对角元 $-3/4$ 估计截距误差与斜率误差之间的协方差，协方差可以为负。若要计算系数组合的不确定性，非对角元也参与计算，见[[线性组合推断]]。

<!-- bilingual-en:start -->
Here $s=\sqrt{3/2}$ estimates the noise scale of individual outcomes, while the two coefficient standard errors measure different estimation uncertainties. Predictor units, spread, and near-collinearity affect $G^{-1}$ and thus affect those uncertainties. Unbiasedness of $s^2$ or $\widehat V$ does not imply unbiasedness after taking square roots.
<!-- bilingual-en:end -->

这里 $s=\sqrt{3/2}$ 估计单个观测方程的噪声尺度，两个系数标准误则衡量各自的估计不确定性。解释变量的单位、分散程度及近共线性会影响 $G^{-1}$，从而影响这些不确定性。$s^2$ 或 $\widehat V$ 无偏，并不意味着开平方后仍然无偏。

<!-- bilingual-en:start -->
## When the error covariance changes
<!-- bilingual-en:end -->

*误差协方差改变时怎么办？*

<!-- bilingual-en:start -->
For a general finite conditional error covariance $\Omega=\operatorname{Cov}(u\mid X)$, covariance propagation instead gives
<!-- bilingual-en:end -->

若一般的有限条件误差协方差为 $\Omega=\operatorname{Cov}(u\mid X)$，协方差传播给出

$$\operatorname{Cov}(\hat\beta\mid X)=G^{-1}X^T\Omega XG^{-1}.$$

<!-- bilingual-en:start -->
Spherical covariance $\Omega=\sigma^2I_n$ is a sufficient condition for both the classical simplification and the residual-variance correction above. Other covariance structures can coincide in special designs; failure of spherical covariance does not prove every classical-looking formula wrong. But the guarantee is lost, and $RSS/(n-k)$ no longer generally estimates one common error variance. Check the actual structure through [[标准误口径匹配|covariance-estimator matching]]; [[协方差失效下的OLS|OLS under nonspherical errors]] explains why the fitted coefficients and their efficiency are separate issues.
<!-- bilingual-en:end -->

球形协方差 $\Omega=\sigma^2I_n$ 足以保证前面的经典化简与残差方差修正。其他协方差结构在特殊设计中也可能出现相同结果，不能因球形条件失效就断言每个经典形式都必然错误。但原有保证已不再成立，$RSS/(n-k)$ 通常也不再是在估计一个共同误差方差。应通过[[标准误口径匹配]]核对真实结构；[[协方差失效下的OLS]]进一步区分拟合系数与效率问题。

<!-- bilingual-en:start -->
Finally, this card's moment calculation does not provide an exact test distribution. Under the additional joint Gaussian model and $n>k$, $s^2$ supports the classical exact t and F calibration described in [[BLUE正态性边界|normality and OLS inference]]. Under mere moment assumptions, $s^2$ can even be zero with positive probability, despite $\sigma^2>0$.
<!-- bilingual-en:end -->

最后，本卡的矩计算本身不提供精确检验分布。额外假设联合 Gaussian 且 $n>k$ 后，$s^2$ 才在这套经典构造中支持精确 t 与 F 校准，详见[[BLUE正态性边界]]。仅有矩假设时，即使 $\sigma^2>0$，$s^2$ 也可能以正概率等于零。

## 来源与核验

<!-- bilingual-en:start -->
- Bruce E. Hansen, [*Econometrics*, §§4.4, 4.6–4.8 and 4.10; PDF pp. 56–57, 59–62, 64–65](https://users.ssc.wisc.edu/~xshi28/Econometrics_Hansen.pdf#page=56): conditional covariance propagation, residual projection, $RSS/(n-k)$ and conditional unbiasedness, and standard errors. The source often reports the covariance of $\sqrt n(\hat\beta-\beta)$; this card consistently uses the unscaled coefficient covariance.
- [MIT OCW 14.310x, Lecture 17, pp. 53–56](https://ocw.mit.edu/courses/14-310x-data-analysis-for-social-scientists-spring-2023/mit14_310x_s23_week08_lec17.pdf#page=53): independently checks the simple-regression denominator $n-2$ and the added normality step for inference. The three-observation arithmetic is worked out directly here.
- [[02_Economy/01_Econometrics/03_多元线性回归.md#2.2. 随机误差项方差的估计|The permanent course error-variance section]] uses $n-k-1$ with $k$ excluding the intercept. [[02_Economy/01_Econometrics/05_多元回归模型的矩阵表达.md#2.3. OLS 估计量的方差协方差矩阵|The matrix-covariance section]] and [[02_Economy/01_Econometrics/05_多元回归模型的矩阵表达.md#3.1. 有限样本性质|its finite-sample derivation]] support the general sandwich and spherical special case.
<!-- bilingual-en:end -->

- Hansen《Econometrics》第 4.4、4.6–4.8、4.10 节（PDF pp. 56–57、59–62、64–65）核对条件协方差传播、残差投影、$RSS/(n-k)$ 及条件无偏性、标准误定义。原文常用 $\sqrt n(\hat\beta-\beta)$ 的协方差；本卡始终使用未经该缩放的系数协方差。
- MIT 14.310x 第 17 讲 pp. 53–56 独立核对简单回归分母 $n-2$ 与推断中额外加入正态性的步骤；三个观测的数值由正文直接计算。
- [[02_Economy/01_Econometrics/03_多元线性回归.md#2.2. 随机误差项方差的估计|永久课程误差方差部分]]使用不含截距的 $k$，所以写 $n-k-1$；[[02_Economy/01_Econometrics/05_多元回归模型的矩阵表达.md#2.3. OLS 估计量的方差协方差矩阵|矩阵协方差部分]]与[[02_Economy/01_Econometrics/05_多元回归模型的矩阵表达.md#3.1. 有限样本性质|有限样本推导]]核对一般夹心公式及球形特例。

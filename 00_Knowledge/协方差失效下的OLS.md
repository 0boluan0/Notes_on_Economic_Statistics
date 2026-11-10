---
aliases:
- OLS under nonspherical errors
- OLS with heteroskedastic or correlated errors
- 非球形误差下的最小二乘
student_os: knowledge-atom
atom_id: ECON-ERR-003
atom_set: regression-error-covariance
atom_type: boundary
status: source-checked
requires:
- '[[异方差]]'
- '[[误差序列相关]]'
- '[[零条件均值无偏性]]'
related:
- '[[经典 Gauss–Markov 定理]]'
- '[[标准误口径匹配]]'
leads_to:
- '[[异方差稳健协方差]]'
- '[[HAC协方差]]'
- '[[广义最小二乘]]'
part_of:
- '[[异方差与自相关.canvas|异方差与自相关]]'
---

<!-- bilingual-en:start -->
# OLS with nonspherical errors
<!-- bilingual-en:end -->

*非球形误差下的 OLS*

<!-- bilingual-en:start -->
A regression can still estimate the right average relationship while reporting the wrong uncertainty around it. [[普通最小二乘|Ordinary least squares (OLS)]] chooses coefficients by minimizing squared residuals. Whether those coefficients are centered on the target, how variable they are, and whether another estimator is more precise are three separate questions. Unequal error variances or correlated errors do not answer all three at once.
<!-- bilingual-en:end -->

回归仍可能估计正确的平均关系，却给出错误的不确定性。[[普通最小二乘|OLS]] 按残差平方和最小来选择系数。系数的抽样中心是否正确、抽样波动多大、其他方法是否更精确，是三个不同问题；误差方差不等或误差相关，不能一次回答全部问题。

> [!example] Two noisy measurements · 两次精度不同的测量
> <!-- bilingual-en:start -->
> Two measurements have the same unknown mean $\beta$: $Y_1=\beta+u_1$ and $Y_2=\beta+u_2$. Suppose both errors have mean zero, are uncorrelated, and have variances $1$ and $4$. Intercept-only OLS is their average, $\hat\beta=(Y_1+Y_2)/2$. It is unbiased, but its variance is $(1+4)/4=5/4$. Giving the more precise measurement weight $4/5$ and the other weight $1/5$ is also unbiased, with variance $4/5$. Different precision changes uncertainty and can change efficiency without creating bias.
> <!-- bilingual-en:end -->
>
> 两次测量都以未知的 $\beta$ 为均值，误差均值为零、互不相关，方差分别为 $1$ 与 $4$。只有截距的 OLS 就是两者平均，仍然无偏，但方差为 $5/4$。把权重改成 $4/5$ 与 $1/5$，仍然无偏，方差却降为 $4/5$。精度不同会影响不确定性，并可能影响效率，不等于自动产生偏误。

<!-- bilingual-en:start -->
## What remains true with a general covariance
<!-- bilingual-en:end -->

*一般协方差下仍然成立的结论*

<!-- bilingual-en:start -->
Write the real linear model as $y=X\beta+u$. Here $y,u$ have $n\ge1$ entries, $X$ is an $n\times k$ design matrix, and $\beta$ has $k\ge1$ coefficients. Work at a fixed design, or condition on the entire random design $X$. Assume its columns are [[线性无关|linearly independent]], the errors have finite conditional second moments, and $E(u\mid X)=0$. The last condition means that knowing the full design leaves no systematic mean error; it is stronger than merely checking each observation against its own predictor in dependent data.
<!-- bilingual-en:end -->

写实数线性模型 $y=X\beta+u$，其中 $y,u$ 有 $n\ge1$ 个元素，$X$ 是 $n\times k$ 设计矩阵，$\beta$ 有 $k\ge1$ 个系数。以下固定设计，或给定完整随机设计 $X$。假设设计列[[线性无关]]、误差有有限条件二阶矩，且 $E(u\mid X)=0$。最后一个条件表示知道整个设计后，误差仍没有系统性均值；在依赖数据中，它比只对每个观测自己的预测变量检查零均值更强。

<!-- bilingual-en:start -->
Let $L=(X^TX)^{-1}X^T$, so $LX=I_k$, where $I_k$ is the $k\times k$ identity. OLS gives $\hat\beta=Ly=\beta+Lu$. Consequently [[零条件均值无偏性|zero conditional mean gives conditional unbiasedness]]: $E(\hat\beta\mid X)=\beta$. No equal-variance, independence or Gaussian assumption is needed for this implication.
<!-- bilingual-en:end -->

令 $L=(X^TX)^{-1}X^T$，则 $LX=I_k$，其中 $I_k$ 是 $k\times k$ 单位矩阵。因此 $\hat\beta=Ly=\beta+Lu$，由[[零条件均值无偏性]]得到 $E(\hat\beta\mid X)=\beta$。这一步不需要同方差、独立或 Gaussian 假设。

<!-- bilingual-en:start -->
The [[协方差矩阵|error covariance matrix]] $\Omega=\operatorname{Cov}(u\mid X)$ has variances on its diagonal and pairwise covariances off the diagonal. It is symmetric positive semidefinite. The [[协方差仿射变换|covariance transformation rule]] gives the exact coefficient covariance:
<!-- bilingual-en:end -->

误差[[协方差矩阵]] $\Omega=\operatorname{Cov}(u\mid X)$ 的对角元是各误差方差，非对角元是两两协方差；它对称且半正定。由[[协方差仿射变换]]，系数协方差恰为

$$\operatorname{Cov}(\hat\beta\mid X)=L\Omega L^T=(X^TX)^{-1}X^T\Omega X(X^TX)^{-1}.$$

<!-- bilingual-en:start -->
The [[标准误含义|standard error]] of a coefficient is the square root of the corresponding diagonal entry. Software must estimate this matrix from the available observations. [[异方差|Heteroskedasticity]] means the error variances need not agree; [[误差序列相关|serial correlation]] concerns dependence across time. Neither label specifies the entire matrix or the direction in which a particular standard error changes.
<!-- bilingual-en:end -->

某个系数的[[标准误含义|标准误]]是相应对角元的平方根；软件还需要从观测估计这个矩阵。[[异方差]]表示误差方差不必相同，[[误差序列相关]]涉及跨时间的相关。任一标签都不足以确定整个矩阵，也不能确定某个标准误朝哪个方向变化。

<!-- bilingual-en:start -->
## Spherical errors are sufficient, not necessary
<!-- bilingual-en:end -->

*球形误差是充分条件，不是必要条件*

<!-- bilingual-en:start -->
If $\Omega=\sigma^2I_n$ with $\sigma^2>0$, the covariance simplifies to $\sigma^2(X^TX)^{-1}$. This spherical-error assumption also supplies the classical [[经典 Gauss–Markov 定理|Gauss–Markov guarantee]] that OLS is BLUE: best, in variance, among linear unbiased estimators. Outside that assumption the guarantee cannot simply be invoked; OLS is not automatically inefficient, and the covariance need not automatically change numerically.
<!-- bilingual-en:end -->

若 $\Omega=\sigma^2I_n$ 且 $\sigma^2>0$，系数协方差化简为 $\sigma^2(X^TX)^{-1}$。这种球形误差假设还提供经典[[经典 Gauss–Markov 定理|Gauss–Markov 保证]]：OLS 是线性无偏估计量中方差最小的 BLUE。离开此假设就不能直接调用这项保证，但不等于 OLS 必然无效率，也不等于协方差的数值必然变化。

<!-- bilingual-en:start -->
For a particular $X$ and a proposed scalar $\sigma^2$, equality with that covariance formula is exactly the condition $X^T(\Omega-\sigma^2I_n)X=0$. It follows by multiplying the two sides of the covariance equality by $X^TX$. Only the covariance seen through this design matters to that equality; the full $\Omega$ need not be spherical. This equality alone does not establish efficiency or justify a residual estimate of $\sigma^2$.
<!-- bilingual-en:end -->

对特定 $X$ 和选定标量 $\sigma^2$，系数协方差恰好等于该公式的条件是 $X^T(\Omega-\sigma^2I_n)X=0$，只需在协方差等式两边分别乘 $X^TX$ 即可得到。这个等式只考察该设计所涉及的协方差部分，不要求整个 $\Omega$ 球形。等式本身既不能证明效率，也不能证明用残差估计 $\sigma^2$ 正确。

> [!example] A shared shock that cancels · 相互抵消的共同冲击
> <!-- bilingual-en:start -->
> Take three measurements $Y_1=\beta+e_1+Z$, $Y_2=\beta+e_2-Z$, $Y_3=\beta+e_3$. The four variables $e_1,e_2,e_3,Z$ are independent, mean zero, variance one. Then $X=\mathbf1=(1,1,1)^T$ and
> $$\Omega=\begin{bmatrix}2&-1&0\\-1&2&0\\0&0&1\end{bmatrix}.$$
> The average cancels $Z$, so $\operatorname{Var}(\bar Y)=1/3$, exactly the value from the reference formula with $\sigma^2=1$, despite unequal variances and correlation. OLS is even BLUE here: any linear unbiased estimate has weights summing to one and variance $\sum_iw_i^2+(w_1-w_2)^2\ge1/3$, attained by equal weights. The lower bound follows from $\sum_i(w_i-1/3)^2\ge0$ and $\sum_iw_i=1$.
>
> However, the usual residual variance $s^2=\sum_i(Y_i-\bar Y)^2/(3-1)$ has expectation $2$: the expected numerator is $\sum_i\operatorname{Var}(Y_i)-3\operatorname{Var}(\bar Y)=5-1=4$. Thus $s^2/3$ is not an unbiased estimate of the actual $1/3$. Matching a known-scalar formula is not the same as validating the usual estimated standard error.
> <!-- bilingual-en:end -->
>
> 四个独立、零均值、单位方差变量按上述方式进入三次测量。共同冲击 $Z$ 在平均值中抵消，故 $\operatorname{Var}(\bar Y)=1/3$；虽然误差异方差且相关，仍与参考标量 $\sigma^2=1$ 的经典公式一致。任何线性无偏估计的权重之和为一，方差为 $\sum_iw_i^2+(w_1-w_2)^2\ge1/3$，由 $\sum_i(w_i-1/3)^2\ge0$ 和 $\sum_iw_i=1$ 得到该下界，等权平均达到它，所以这里 OLS 仍是 BLUE。
>
> 然而，通常的残差方差 $s^2$ 的期望是 $2$：分子期望为 $5-1=4$，再除以 $2$。因此 $s^2/3$ 并不是实际方差 $1/3$ 的无偏估计。已知标量公式数值相同，与实际残差标准误估计有效，是不同结论。

<!-- bilingual-en:start -->
There is also no universal direction of error from ignoring correlation. For two errors of variance one and covariance $\rho$, the average has variance $(1+\rho)/2$; the independence calculation gives $1/2$. Positive and negative $\rho$ move the true variance in opposite directions. Other regressors combine errors with different weights, so even the sign of an error correlation alone is not a general rule for coefficient uncertainty.
<!-- bilingual-en:end -->

忽略相关也没有统一的误差方向。两个单位方差、协方差为 $\rho$ 的误差，其平均方差是 $(1+\rho)/2$，而独立计算给出 $1/2$。正负 $\rho$ 让实际方差朝不同方向变化。其他设计会用不同权重组合误差，因此仅凭误差相关的正负，也不能普遍判断系数不确定性。

<!-- bilingual-en:start -->
## Unbiasedness does not settle large-sample inference
<!-- bilingual-en:end -->

*无偏不解决大样本推断*

<!-- bilingual-en:start -->
[[估计量一致性|Consistency]] asks whether estimation error becomes small in probability as the sample grows. Zero conditional mean alone does not ensure this. If every measurement is $Y_i=\beta+Z$, with one shared $Z$ equally likely to be $-1$ or $1$, then the OLS average is unbiased for every $n$, yet its error has absolute value one for every $n$. More copies do not provide more independent information. [[OLS一致性条件|OLS consistency]] additionally requires suitable design and error-average limits.
<!-- bilingual-en:end -->

[[估计量一致性|一致性]]问样本增长后，估计误差是否依概率变小。零条件均值本身不够：若所有测量都是 $Y_i=\beta+Z$，同一个 $Z$ 等概率取 $-1$ 或 $1$，则任何 $n$ 下平均值都无偏，但误差绝对值始终为一。增加副本没有增加独立信息；[[OLS一致性条件]]还要求合适的设计与误差平均极限。

<!-- bilingual-en:start -->
Changing to [[异方差稳健协方差|HC]], [[HAC协方差|HAC]] or [[聚类标准误边界|cluster-robust inference]] changes uncertainty estimation under different assumptions; it does not refit the OLS coefficients. [[广义最小二乘|GLS]] instead changes the weighting when a covariance model is available. Exact Gaussian t/F laws and asymptotic approximations each need their own conditions. None of these methods repairs a wrong conditional mean. A lagged outcome with serially correlated errors can be correlated with the current error; omitted dynamics or confounders can also invalidate the zero-mean premise. Unchanged coefficients after changing standard errors therefore do not certify the model.
<!-- bilingual-en:end -->

改用[[异方差稳健协方差|HC]]、[[HAC协方差|HAC]] 或[[聚类标准误边界|聚类稳健推断]]，是在不同假设下改变不确定性估计，不会重新拟合 OLS 系数。[[广义最小二乘|GLS]] 则在有协方差模型时改变权重。精确 Gaussian t/F 分布与大样本近似各需自己的条件。这些方法都不能修复错误条件均值：误差序列相关时，滞后结果可能与当前误差相关；遗漏动态或混杂因素也可能破坏零均值前提。所以切换标准误后系数不变，不能证明模型正确。

## 来源与核验

<!-- bilingual-en:start -->
- [Cameron and Miller, *A Practitioner’s Guide to Cluster-Robust Inference* (2015), PDF pp.3–8](https://cameron.econ.ucdavis.edu/research/Cameron_Miller_JHR_2015_February.pdf): general OLS covariance and the distinction between coefficient precision and estimating that precision.
- [CMU 36-401, Lectures24–25, §§3,6–7, PDF pp.12–13,24–27](https://www.stat.cmu.edu/~cshalizi/mreg/15/lectures/24/lecture-24--25.pdf): zero-mean model, linear-unbiased variance comparison, GLS and mean-specification boundary. Covariance is only positive semidefinite in general; invertibility is an additional GLS condition. Typographical errors in intermediate formulas are not used.
- [MIT14.32, *Serial Correlation*, PDF pp.1–2](https://ocw.mit.edu/courses/14-32-econometrics-spring-2007/79b4f5053aa4b0b555869b2bd8d372b5_rec_4_27.pdf): course scope for robust inference versus covariance modeling. Its unqualified consistency and loss-of-BLUE wording is not treated as a theorem without further conditions.
- The covariance equality criterion and all measurement examples are derived explicitly above; [[02_Economy/01_Econometrics/08_自相关.md#3. 忽略自相关的后果|the permanent course consequences section]] supplies the local notation and scope.
<!-- bilingual-en:end -->

Cameron–Miller 核对一般协方差与精度估计的区别；CMU 指定页核对零均值模型、线性无偏比较、GLS 与均值错设边界，不沿用其中一般协方差必正定等不严谨措辞。MIT 用于课程路线，不把未限定的一致性或必失 BLUE 说法当作一般定理。协方差相等判据与全部测量算例在正文直接推导；永久课程只确定本地语境。

---
aliases:
  - AR(1) quasi-differencing
  - AR(1) 准差分
student_os: knowledge-atom
atom_id: ECON-ERR-014
atom_set: regression-error-covariance
atom_type: transformation
status: source-checked
requires:
  - "[[误差序列相关]]"
  - "[[广义最小二乘]]"
related:
  - "[[可行GLS]]"
  - "[[Durbin-Watson检验]]"
leads_to:
  - "[[误差协方差决策]]"
part_of:
  - "[[异方差与自相关.canvas|异方差与自相关]]"
---

<!-- bilingual-en:start -->
# AR(1) quasi-differencing
<!-- bilingual-en:end -->

*AR(1) 准差分*

<!-- bilingual-en:start -->
Suppose half of yesterday's regression error carries into today. An error of $2$ yesterday followed by $1$ today need not contain a new shock today: the $1$ may be entirely inherited. **Quasi-differencing subtracts the carried-over fraction of the previous equation**, leaving the new shock. It transforms the response and every explanatory column together, so that they still describe the same regression coefficients.
<!-- bilingual-en:end -->

假设昨天回归误差的一半会延续到今天。昨天误差为 $2$、今天为 $1$，并不一定意味着今天有新冲击：这个 $1$ 可能完全是延续下来的。**准差分从今天的方程中减去前一期方程的相应比例**，留下新冲击。因变量和全部解释变量列必须一起变换，才能继续描述原来的回归系数。

<!-- bilingual-en:start -->
This is a specific way to handle [[误差序列相关|serial correlation in regression errors]]. It implements [[广义最小二乘|generalized least squares (GLS)]] when the stated covariance model is correct and its correlation parameter is known. Estimating that parameter adds a separate [[可行GLS|feasible-GLS]] step.
<!-- bilingual-en:end -->

这是处理[[误差序列相关]]的一种特定变换。当声明的协方差模型正确、相关参数已知时，它实现[[广义最小二乘|GLS]]；估计相关参数则另外引入[[可行GLS]]步骤。

<!-- bilingual-en:start -->
## Subtract a fraction of the whole equation
<!-- bilingual-en:end -->

*减去前一期完整方程的一部分*

<!-- bilingual-en:start -->
Take $T\ge2$ equally spaced observations. Let $y_t$ be the response at time $t$, $x_t$ a column of $p\ge1$ explanatory values, and $\beta$ the $p$ unknown coefficients. A constant regressor, if present, is one entry of $x_t$. The $T\times p$ design matrix $X$ has row $x_t^T$. An **AR(1) error model** means that each error $u_t$ contains $\rho$ times the preceding error and a new innovation $\varepsilon_t$:
<!-- bilingual-en:end -->

取 $T\ge2$ 个等间隔观测。$y_t$ 是第 $t$ 期因变量，$x_t$ 是含 $p\ge1$ 个解释值的列向量，$\beta$ 是对应的 $p$ 个未知系数；若含截距，常数 $1$ 就是 $x_t$ 的一个分量。设计矩阵 $X$ 有 $T$ 行、$p$ 列，第 $t$ 行为 $x_t^T$。**AR(1) 误差模型**表示每期误差 $u_t$ 由上一期误差的 $\rho$ 倍与新的创新 $\varepsilon_t$ 组成：

$$y_t=x_t^T\beta+u_t,\qquad u_t=\rho u_{t-1}+\varepsilon_t\quad(t\ge2).$$

<!-- bilingual-en:start -->
For now, $\rho$ is a fixed known number with $|\rho|<1$. Multiply the previous regression equation by $\rho$ and subtract it. Define the starred quantities as the transformed data, not new parameters:
<!-- bilingual-en:end -->

先把 $\rho$ 看作固定已知、满足 $|\rho|<1$ 的数。将前一期回归方程乘以 $\rho$ 后相减；星号标记变换后的数据，并不表示另换一组参数：

$$y_t^*=y_t-\rho y_{t-1},\qquad x_t^*=x_t-\rho x_{t-1},\qquad y_t^*=(x_t^*)^T\beta+\varepsilon_t\quad(t\ge2).$$

<!-- bilingual-en:start -->
The cancellation is exact: $u_t-\rho u_{t-1}=\varepsilon_t$. With an original intercept $\beta_0$ and other regressors $z_t$, the same identity reads
<!-- bilingual-en:end -->

这里恰好消去 $u_t-\rho u_{t-1}=\varepsilon_t$。若把原截距单独写成 $\beta_0$，其余解释变量写成 $z_t$，同一恒等式就是

$$y_t-\rho y_{t-1}=(1-\rho)\beta_0+(z_t-\rho z_{t-1})^T\gamma+\varepsilon_t,$$

<!-- bilingual-en:start -->
where $\gamma$ contains the other coefficients. The original constant column becomes $1-\rho$, not $1$. If software instead fits a fresh unit intercept on these later rows, its coefficient estimates $(1-\rho)\beta_0$ and must be divided by $1-\rho$ to recover $\beta_0$. Passing the explicitly transformed constant column avoids that reinterpretation; do not add another automatic constant to it.
<!-- bilingual-en:end -->

其中 $\gamma$ 是其余系数。原常数列变成 $1-\rho$，不是 $1$。若软件对这些后续行另加一列常数 $1$，它的系数对应 $(1-\rho)\beta_0$，要除以 $1-\rho$ 才还原原截距。直接传入已变换的常数列，可以保持原参数含义；此时不要再自动添加另一列常数。

<!-- bilingual-en:start -->
## Why the first observation needs a different transformation
<!-- bilingual-en:end -->

*首期为何要单独处理*

<!-- bilingual-en:start -->
The algebra alone says nothing about the covariance of the innovations. For the usual stationary GLS construction, assume $E[u_1]=E[\varepsilon_t]=0$, the innovations are pairwise uncorrelated with common finite variance $\tau^2>0$, and each future innovation is uncorrelated with $u_1$. Initialise the first error with $\operatorname{Var}(u_1)=\tau^2/(1-\rho^2)$. These are stationary **second-moment** conditions: they make the error variance constant and the covariance depend only on the time gap. They do not alone assert independence or a Gaussian distribution.
<!-- bilingual-en:end -->

代数相减本身并没有说明创新的协方差。通常的平稳 GLS 构造还假设 $E[u_1]=E[\varepsilon_t]=0$；各创新两两不相关、具有相同的有限方差 $\tau^2>0$，且后续创新都与 $u_1$ 不相关。首期按 $\operatorname{Var}(u_1)=\tau^2/(1-\rho^2)$ 初始化。这是平稳的**二阶矩**条件：误差方差不随时间变化，协方差只取决于时间间隔；它们本身没有断言独立或正态。

<!-- bilingual-en:start -->
The recursion then gives the error [[协方差矩阵|covariance matrix]] $\Omega$, whose $(t,s)$ entry is
<!-- bilingual-en:end -->

由递推得到误差[[协方差矩阵]] $\Omega$，其第 $(t,s)$ 个元素为

$$\Omega_{ts}=\operatorname{Cov}(u_t,u_s)=\frac{\tau^2}{1-\rho^2}\rho^{|t-s|}.$$

<!-- bilingual-en:start -->
Here $\tau^2$ is the **innovation variance**; it differs from the original error variance $\tau^2/(1-\rho^2)$. There is no observed period $0$ to subtract from period $1$. The Prais–Winsten transformation instead rescales the entire first row:
<!-- bilingual-en:end -->

$\tau^2$ 是**创新方差**，不同于原误差方差 $\tau^2/(1-\rho^2)$。首期没有已观察到的第 $0$ 期可供相减，因此 Prais–Winsten 对完整首行作缩放：

$$y_1^*=\sqrt{1-\rho^2}\,y_1,\qquad x_1^*=\sqrt{1-\rho^2}\,x_1,\qquad u_1^*=\sqrt{1-\rho^2}\,u_1.$$

<!-- bilingual-en:start -->
Its error variance is now $\tau^2$, matching all later innovations. It is also uncorrelated with them. Writing all row operations as the invertible matrix $D$, we obtain $y^*=Dy$, $X^*=DX$, and $\operatorname{Cov}(Du)=\tau^2I_T$, where $I_T$ is the identity matrix. This makes the errors uncorrelated with equal variance; dividing further by $\tau$ would give identity covariance. Zero covariance does not generally imply independence. If the entire error vector is additionally [[联合Gaussian|jointly Gaussian]], the transformed errors are independent Gaussian variables.
<!-- bilingual-en:end -->

首行误差方差现在为 $\tau^2$，与后面的创新相同，而且与它们不相关。将全部行操作记作可逆矩阵 $D$，便有 $y^*=Dy$、$X^*=DX$ 和 $\operatorname{Cov}(Du)=\tau^2I_T$，其中 $I_T$ 是单位矩阵。这使误差不相关且等方差；再除以 $\tau$ 才得到单位协方差。不相关通常不等于独立；若整个误差向量另外满足[[联合Gaussian|联合正态]]，变换后的误差才可由此推出相互独立且正态。

> [!example] Keeping the first row changes the fit · 保留首行会改变拟合
> <!-- bilingual-en:start -->
> Consider an intercept-only model $y_t=\mu+u_t$, with observed responses $(10,12,11)$ and known $\rho=1/2$. The transformed response and constant column are
> <!-- bilingual-en:end -->
>
> 仅含截距的模型为 $y_t=\mu+u_t$，观测为 $(10,12,11)$，已知 $\rho=1/2$。变换后的因变量与常数列如下：
>
> | $t$ | $y_t^*$ | $x_t^*$ |
> | --- | --- | --- |
> | $1$ | $5\sqrt3$ | $\sqrt3/2$ |
> | $2$ | $7$ | $1/2$ |
> | $3$ | $5$ | $1/2$ |
>
> <!-- bilingual-en:start -->
> With a single transformed column, [[普通最小二乘|least squares]] estimates $\mu$ as $\sum x_t^*y_t^*/\sum(x_t^*)^2$. Prais–Winsten uses all three rows, giving $13.5/1.25=10.8$. Cochrane–Orcutt uses only the last two rows, giving $6/0.5=12$. Neither calculation adds a new unit-intercept column. The difference is the information retained, not a change in the definition of $\mu$.
> <!-- bilingual-en:end -->
>
> 单个变换列的[[普通最小二乘|最小二乘]]系数为 $\sum x_t^*y_t^*/\sum(x_t^*)^2$。Prais–Winsten 用全部三行，得 $13.5/1.25=10.8$；Cochrane–Orcutt 只用后两行，得 $6/0.5=12$。两次都不另加一列常数 $1$。区别是保留的信息不同，并不是 $\mu$ 的定义变了。

<!-- bilingual-en:start -->
Cochrane–Orcutt drops the first **transformed equation**, not every use of the original first observation: $y_1$ still enters $y_2-\rho y_1$. It is not the full-sample GLS transformation. Dropping a row can also reduce the design rank. Prais–Winsten retains the first equation under the specified stationary initial covariance; if the initial error has a different variance or is fixed at zero, its first-row rule must be reconsidered. Merely requiring $|\rho|<1$ does not establish that initial covariance.
<!-- bilingual-en:end -->

Cochrane–Orcutt 丢掉的是第一条**变换后方程**，而不是完全不用原始首个观测：$y_1$ 仍出现在 $y_2-\rho y_1$ 中。它不是全部样本的完整 GLS 变换；删除一行还可能降低设计矩阵的秩。Prais–Winsten 按指定的平稳初始协方差保留首行；如果初始误差方差不同、或固定为零，就必须重新考虑首行规则。仅有 $|\rho|<1$ 并不能证明初始协方差符合要求。

<!-- bilingual-en:start -->
## Covariance correction still needs a valid regression model
<!-- bilingual-en:end -->

*协方差变换仍需要有效的回归条件*

<!-- bilingual-en:start -->
For the finite-sample GLS claim, take $X$ as fixed with [[线性无关|linearly independent columns]], and impose the zero-mean and covariance conditions above. Equivalently, for random regressors we need the corresponding assumptions **conditional on the whole design $X$**, including [[零条件均值无偏性|zero conditional mean]] $E[u\mid X]=0$, meaning no remaining systematic error once the entire design is given. Because known $D$ is invertible, $DX$ keeps full column rank. OLS of $Dy$ on $DX$ then estimates the original $\beta$, is unbiased in this conditional model, and has the GLS variance. Normality is unnecessary for that conclusion; exact classical Gaussian [[回归t检验|$t$]]/[[嵌套模型F检验|$F$]] reference laws need the extra distributional assumptions and positive residual degrees of freedom $T-p$.
<!-- bilingual-en:end -->

有限样本 GLS 结论先把 $X$ 当作固定且各列[[线性无关]]，并要求前述零均值与协方差条件。如果解释变量随机，则相应假设要在**给定整个设计矩阵 $X$** 后成立，包括[[零条件均值无偏性|零条件均值]] $E[u\mid X]=0$：知道全部设计以后，误差仍无系统偏向。已知 $D$ 可逆，所以 $DX$ 保持满列秩。将 $Dy$ 对 $DX$ 做 OLS，估计的仍是原 $\beta$，在该条件模型下无偏，并具有 GLS 方差。这些结论不要求正态；精确的经典正态 [[回归t检验|$t$]]/[[嵌套模型F检验|$F$]] 参考分布还需要附加分布假设，以及正的残差自由度 $T-p$。

<!-- bilingual-en:start -->
This distinction matters with lagged responses. In the simple stationary example $y_t=u_t$, using $y_{t-1}$ as a regressor gives $\operatorname{Cov}(y_{t-1},u_t)=\rho\tau^2/(1-\rho^2)$, nonzero when $\rho\ne0$. Thus a zero coefficient in the written equation does not make its error exogenous. A dynamic regression may still admit a useful transformed equation, but its regressor–innovation orthogonality and time-series limit arguments must be checked separately. Fixed-design finite-sample claims and consistency of the initial OLS fit cannot simply be imported.
<!-- bilingual-en:end -->

含滞后因变量时尤其要区分这些条件。简单平稳例子 $y_t=u_t$ 中，如果把 $y_{t-1}$ 当作解释变量，则 $\operatorname{Cov}(y_{t-1},u_t)=\rho\tau^2/(1-\rho^2)$，在 $\rho\ne0$ 时不为零。因此，方程中写了一个零系数，不会让误差自动外生。动态回归仍可能有有用的变换方程，但必须另查解释变量与创新的正交条件和时间序列极限论证；不能直接套用固定设计的有限样本结论，也不能直接假定初始 OLS 一致。

<!-- bilingual-en:start -->
## When rho is estimated
<!-- bilingual-en:end -->

*当相关参数需要估计时*

<!-- bilingual-en:start -->
A common feasible procedure starts with an OLS fit justified by the relevant [[OLS一致性条件|consistency conditions]], forms original-scale residuals $\hat u_t=y_t-x_t^T\hat\beta^{(0)}$, and regresses them on their lag. Here the superscript $(0)$ marks the initial coefficient fit. A no-intercept lag regression gives, when its denominator is positive,
<!-- bilingual-en:end -->

常见可行步骤是在相应[[OLS一致性条件]]有依据时先做 OLS，构造原尺度残差 $\hat u_t=y_t-x_t^T\hat\beta^{(0)}$，再将残差对其滞后值回归。这里上标 $(0)$ 标记初始系数估计。不含截距的残差滞后回归在分母为正时给出

$$\hat\rho=\frac{\sum_{t=2}^T\hat u_t\hat u_{t-1}}{\sum_{t=2}^T\hat u_{t-1}^2}.$$

<!-- bilingual-en:start -->
Substitute $\hat\rho$ into every row transformation and refit. Iteration reconstructs $\hat u_t=y_t-x_t^T\hat\beta$ from the updated coefficients on the **original scale**, then updates $\hat\rho$. For $t\ge2$, the transformed regression residuals instead estimate $u_t-\hat\rho u_{t-1}$; using them as if they were the original $u_t$ changes the procedure.
<!-- bilingual-en:end -->

将 $\hat\rho$ 代入全部行变换并重新拟合。迭代时要用更新后的系数回到**原尺度**，重建 $\hat u_t=y_t-x_t^T\hat\beta$ 后再更新 $\hat\rho$。对 $t\ge2$，变换回归的残差估计的是 $u_t-\hat\rho u_{t-1}$；把它直接当成原误差 $u_t$，就换了算法。

<!-- bilingual-en:start -->
The estimated transformation is random and uses the responses. It does not inherit known-$\rho$ finite-sample unbiasedness or exact Gaussian reference laws automatically. The usual argument for [[可行GLS|FGLS consistency and first-order equivalence]] checks a credible covariance model, a consistent correlation estimate, and regularity controlling the transformed design and regressor–error cross-products; numerical iteration stopping is not that proof. Check remaining transformed-error patterns and the adequacy of the mean equation. Omitted trends, seasonality or higher-order dynamics may require a different model, not another round of the same transformation.
<!-- bilingual-en:end -->

估计出来的变换是随机的，还使用了因变量数据，不能自动继承已知 $\rho$ 时的有限样本无偏性或精确正态参考分布。[[可行GLS|FGLS 的一致性与一阶等价]]的通常论证要核对可信的协方差模型、相关参数的一致估计，以及对变换后设计和解释变量—误差交叉和的正则控制；算法迭代停止不是这些条件的证明。还要检查变换后误差形态和均值方程是否合适。遗漏趋势、季节或更高阶动态，可能需要另一种模型，而非继续重复原变换。

> [!warning] Quasi-differences are not ordinary first differences · 准差分不等于普通一阶差分
> <!-- bilingual-en:start -->
> At $\rho=0$, the full Prais–Winsten transformation is the identity; Cochrane–Orcutt still discards an otherwise usable equation. At $\rho=1$, the later-row algebra becomes ordinary first differences, but the stationary finite-variance model with $\tau^2>0$ no longer supplies the first-row construction. At either $\rho=1$ or $-1$, its square-root factor vanishes and $D$ is singular. Values outside $(-1,1)$ also lie outside this stationary causal covariance model. Near $\rho=1$, the later-row constant $1-\rho$ is small; near either boundary the first-row factor is small, so finite-sample sensitivity deserves attention. An estimate outside the allowed range calls for an explicit constrained model or another analysis, not a silent claim that the same GLS proof applies.
> <!-- bilingual-en:end -->
>
> $\rho=0$ 时，完整 Prais–Winsten 变换就是原样保留数据；Cochrane–Orcutt 仍会丢掉一条原本可用的方程。$\rho=1$ 时，后续行代数退化为普通一阶差分，但创新方差 $\tau^2>0$ 的平稳有限方差模型已不能提供上述首行构造。在 $\rho=1$ 或 $-1$，首行根号因子为零，$D$ 奇异；$(-1,1)$ 外也不属于这里的平稳因果协方差模型。接近 $1$ 时，后续行常数 $1-\rho$ 很小；接近任一边界时，首行因子很小，都值得留意有限样本敏感性。估计值越界时，应明确采用受约束模型或另作分析，不能默认为同一个 GLS 证明仍适用。

## 来源与核验

<!-- bilingual-en:start -->
- MIT 14.32, [*Serial Correlation*, pp. 1–3](https://ocw.mit.edu/courses/14-32-econometrics-spring-2007/79b4f5053aa4b0b555869b2bd8d372b5_rec_4_27.pdf): the AR(1) subtraction, transformed intercept, residual-based correlation estimate, and original-scale residuals for iteration. The card states the exogeneity, moment and asymptotic qualifications explicitly.
- Stata, [`prais — Prais–Winsten and Cochrane–Orcutt regression`, pp. 3–5 and 10–11](https://www.stata.com/manuals/tsprais.pdf): the retained/scaled first row versus the omitted first equation, whole-column transformation and plug-in handling of rho. Innovation variance and stationary initialisation are made explicit here; numerical examples and covariance checks are direct calculations.
- Bruce E. Hansen, [*Econometrics* (2010), §4.3, printed pp. 47–48, and §7.1, pp. 104–107](https://users.ssc.wisc.edu/~xshi28/Econometrics_Hansen.pdf): conditioning on the whole design for unbiasedness and the distinction between infeasible and feasible weighting. The heteroskedastic FGLS example in §7.1 is not treated as a proof for arbitrary dynamic regressors.
- [[02_Economy/01_Econometrics/08_自相关.md#5.2. 科克伦–奥克特 CO 迭代估计|Permanent course notes: CO iteration]]: course notation and the original-scale updating order.
<!-- bilingual-en:end -->

- MIT 14.32《Serial Correlation》第 1–3 页：核对 AR(1) 相减、截距变换、残差估计相关参数与迭代时重建原尺度残差；正文补足外生性、矩及渐近条件。
- Stata `prais` 独立手册第 3–5、10–11 页：核对保留并缩放首行与丢首行的区别、整列变换和代入估计相关参数的口径。创新方差与平稳初始化在正文中明确，算例与协方差由直接计算核验。
- Hansen《Econometrics》(2010) 第 4.3 节、印刷第 47–48 页，以及第 7.1 节、第 104–107 页：核对给定整个设计的无偏性条件与已知/估计权重的区别；不把其中异方差例子的 FGLS 结果直接搬到任意动态回归。
- [[02_Economy/01_Econometrics/08_自相关.md#5.2. 科克伦–奥克特 CO 迭代估计|永久课程笔记：CO 迭代]]：保留课程符号与原尺度更新顺序。

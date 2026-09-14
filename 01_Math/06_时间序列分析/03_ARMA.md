# ARMA：从创新表示到可检验预测
<!-- bilingual-en:start -->
*ARMA: from innovation representation to testable forecasts*
<!-- bilingual-en:end -->

ARMA 的核心不是背诵 ACF/PACF 形状，而是回答一条完整问题链：我们正在对哪个平稳对象建模？本期新信息如何进入序列？这个表示是否稳定、唯一且可恢复？样本证据能否排除明显遗漏？最后，它能否在真实预测时点胜过简单基准？[[ARMA 模型：识别、估计、诊断与预测.canvas|主题 Canvas]] 展示全局关系；本章把这些关系连成一条可连续阅读和实际执行的路径。原始课堂展开、例题与作业仍保留在[[01_Math/06_时间序列分析/03_平稳时间序列模型|课堂记录]]中。
<!-- bilingual-en:start -->
ARMA is not a list of ACF/PACF patterns. It is a chain of questions: what stationary target is being modelled, how does new information enter, is the representation stable and recoverable, do the diagnostics reveal omitted structure, and does the fitted procedure forecast honestly at real historical origins? The topic Canvas shows the global map; this chapter supplies the continuous argument.
<!-- bilingual-en:end -->

## 1. 先确定建模对象
<!-- bilingual-en:start -->
*1. Start by fixing the modelling target*
<!-- bilingual-en:end -->

经典 ARMA 描述的是均值和自协方差不随日历时间改变的二阶过程，即[[宽平稳定义|宽平稳对象]]。这并不要求每条样本路径看起来水平，也不保证从一条路径就能可靠恢复总体矩；[[平稳性、遍历性与谱.canvas|平稳性、遍历性与谱]]单独处理这些前置边界。进入 ARMA 前至少要说明：观测频率、是否去除了确定性项、是否做了变换或差分，以及模型针对水平、变化率还是季节差分。
<!-- bilingual-en:start -->
Classical ARMA describes a covariance-stationary target whose mean and autocovariance do not depend on calendar time. This neither requires every realised path to look flat nor guarantees that one path consistently reveals the population moments. State the frequency, deterministic adjustments, transformations, and differencing before fitting a dynamic model.
<!-- bilingual-en:end -->

对零均值宽平稳序列，[[自协方差与ACF|自协方差与 ACF]]为
$$\gamma(h)=\operatorname{Cov}(X_t,X_{t-h}),\qquad
\rho(h)=\frac{\gamma(h)}{\gamma(0)}.$$
ARMA 用少量参数生成整条 $\gamma(h)$；样本 ACF 只是这条总体结构的有噪声估计。
<!-- bilingual-en:start -->
For a zero-mean covariance-stationary series, the autocovariance and ACF depend only on the lag. ARMA uses a small number of parameters to generate that entire sequence; the sample ACF is only a noisy estimate of the population object.
<!-- bilingual-en:end -->

## 2. 创新不是残差，白噪声也不是“完全随机”
<!-- bilingual-en:start -->
*2. An innovation is not a residual, and white noise is not complete randomness*
<!-- bilingual-en:end -->

记 $\mathcal H_{t-1}$ 为常数项与过去观测张成的闭线性空间；[[创新]]是本期观测减去它在这个空间上的一步线性预测：
$$\varepsilon_t=X_t-P_{\mathcal H_{t-1}}X_t.$$
它与过去的闭线性空间正交，是“这一期真正新增了什么”的二阶版本。模型拟合后得到的 residual 依赖估计参数和初值，只是潜在创新的估计，不能与真实 innovation 互换。
<!-- bilingual-en:start -->
An innovation is the current observation minus its one-step affine linear projection on the observed past. It is orthogonal to the constant and linear past. A fitted residual depends on estimated parameters and initialisation, so it is an estimate of the latent innovation rather than the same object.
<!-- bilingual-en:end -->

本章的基本驱动通常假定为[[白噪声二阶定义|二阶白噪声]]：
$$E(\varepsilon_t)=0,\qquad \operatorname{Var}(\varepsilon_t)=\sigma^2,\qquad
\operatorname{Cov}(\varepsilon_t,\varepsilon_{t-h})=0\quad(h\ne0).$$
这只排除跨期线性相关。若还要把 ARMA 递推解释为完整条件均值，需要创新相对于预测信息集是 MDS、独立，或有 joint Gaussian 等足够条件。ARCH 创新可以是 MDS 和白噪声，却因条件方差依赖过去而不是 i.i.d.；这正说明“均值难预测”与“风险恒定”是两件事。
<!-- bilingual-en:start -->
Second-order white noise rules out serial linear correlation, not every form of dependence. Interpreting an ARMA recursion as the full conditional mean needs an MDS, independence, joint Gaussianity, or another sufficient condition. ARCH innovations may be both white noise and an MDS while remaining non-i.i.d. because their conditional variance changes.
<!-- bilingual-en:end -->

## 3. 同一个代数骨架里的 AR、MA 与 ARMA
<!-- bilingual-en:start -->
*3. AR, MA, and ARMA on one algebraic backbone*
<!-- bilingual-en:end -->

[[滞后算子约定|滞后算子]]满足 $BX_t=X_{t-1}$。本库固定
$$\phi(B)=1-\phi_1B-\cdots-\phi_pB^p,\qquad
\theta(B)=1+\theta_1B+\cdots+\theta_qB^q.$$
不同教材可能给 MA 多项式使用减号；根、软件参数与手算公式必须按当前声明的多项式解释，不能脱离符号约定记口诀。
<!-- bilingual-en:start -->
The lag operator satisfies $BX_t=X_{t-1}$. This vault uses minus signs in the AR polynomial and plus signs in the MA polynomial. Other conventions are legitimate, but roots and reported coefficients must always be interpreted under the convention actually written.
<!-- bilingual-en:end -->

[[AR(p)模型]]写成
$$X_t-\mu=\sum_{i=1}^p\phi_i(X_{t-i}-\mu)+\varepsilon_t.$$
它直接记住最近 $p$ 个观测，但一次冲击会通过观测递归传播更久。若 AR(1) 的 $\phi=0.6$，一个单位创新对本期、下一期、下两期的作用依次为 $1,0.6,0.36$；“一阶”不等于“冲击只活一期”。
<!-- bilingual-en:start -->
An AR($p$) uses the previous $p$ observations directly, while a shock may propagate much longer through the recursion. In AR(1) with $\phi=0.6$, a unit innovation has successive effects $1,0.6,0.36,\ldots$.
<!-- bilingual-en:end -->

[[MA(q)模型]]写成
$$X_t=\mu+\varepsilon_t+\theta_1\varepsilon_{t-1}+\cdots+\theta_q\varepsilon_{t-q}.$$
它记住最近 $q$ 个潜在创新，冲击的直接寿命有限。以 MA(1) 为例，
$$\gamma(0)=\sigma^2(1+\theta^2),\quad
\gamma(1)=\sigma^2\theta,\quad
\gamma(h)=0\ (|h|>1).$$
“moving average” 是历史名称，不是对最近几个观测值取滑动平均。
<!-- bilingual-en:start -->
An MA($q$) is a finite filter of latent innovations, not a rolling average of observed values. Its direct shock memory is finite, which makes its population autocovariance vanish beyond lag $q$.
<!-- bilingual-en:end -->

[[ARMA(p,q)模型]]把两者合在一起：
$$\phi(B)(X_t-\mu)=\theta(B)\varepsilon_t.$$
$p$ 控制观测递归，$q$ 控制最近创新怎样直接进入；$p=q=0$ 得白噪声，$q=0$ 得 AR，$p=0$ 得 MA。这个方程还没有自动保证因果、可逆、参数唯一或预测解释成立，这些是下一层检查。
<!-- bilingual-en:start -->
ARMA combines observation recursion with finite innovation memory. The equation alone does not guarantee causality, invertibility, a unique parameterisation, or a full conditional-mean interpretation; those are separate claims.
<!-- bilingual-en:end -->

## 4. 三个不能混在一起的表示条件
<!-- bilingual-en:start -->
*4. Three representation conditions that answer different questions*
<!-- bilingual-en:end -->

第一，[[AR因果根条件|因果性]]问能否只用当前与过去创新稳定地生成本期值。对上述 $\phi(z)$ 约定，所有多项式零点必须在单位圆外。AR(1) 的零点是 $1/\phi$，递推特征值是 $\phi$；所以“多项式零点在外”与“伴随矩阵特征值在内”都等价于 $|\phi|<1$，对象不同但没有矛盾。
<!-- bilingual-en:start -->
Causality asks whether current and past innovations generate the present through a stable one-sided filter. Lag-polynomial zeros lie outside the unit circle, whereas the reciprocal companion eigenvalues lie inside it. In AR(1), both statements reduce to $|\phi|<1$.
<!-- bilingual-en:end -->

第二，[[MA可逆性|可逆性]]问能否从当前与过去观测稳定地恢复创新。对 $\theta(z)=1+\theta_1z+\cdots+\theta_qz^q$，零点也需在单位圆外。不可逆有限 MA 仍然存在且宽平稳；可逆性选择一个可恢复、二阶意义下唯一的创新参数化，而不是有限 MA 的存在条件。
<!-- bilingual-en:start -->
Invertibility asks whether current and past observations recover the innovations through a stable filter. A finite MA exists and is covariance-stationary even when it is noninvertible; invertibility is a normalisation for recoverability and second-order uniqueness.
<!-- bilingual-en:end -->

第三，[[ARMA公共因子|最小性]]问 $\phi$ 与 $\theta$ 是否互素。即使两组根都在单位圆外，
$$(1-aB)X_t=(1-aB)\varepsilon_t$$
仍可约成白噪声；未约分的 ARMA(1,1) 参数只是冗余表示。接近抵消的根虽可形式识别，也会造成平坦似然、巨大标准误和不稳定解释。
<!-- bilingual-en:start -->
Minimality asks whether the AR and MA polynomials are coprime. Causal and invertible factors can still cancel, leaving redundant parameters. Near cancellation can be formally identified while producing a flat likelihood and unstable estimates.
<!-- bilingual-en:end -->

## 5. 两个无限表示与 Wold 边界
<!-- bilingual-en:start -->
*5. Two infinite representations and the Wold boundary*
<!-- bilingual-en:end -->

因果性给出[[ARMA无限MA表示|无限 MA 冲击响应]]：
$$X_t-\mu=\frac{\theta(B)}{\phi(B)}\varepsilon_t
=\sum_{j=0}^{\infty}\psi_j\varepsilon_{t-j}.$$
有限阶因果 ARMA 的 $\psi_j$ 几何衰减，既绝对可和又平方可和；$\psi_j$ 也是一单位创新在 $j$ 期后的动态作用。
<!-- bilingual-en:start -->
Causality yields a one-sided infinite-MA representation. For a finite-order causal ARMA, the impulse-response coefficients decay geometrically and are square summable, giving a finite-variance mean-square limit.
<!-- bilingual-en:end -->

可逆性给出[[ARMA无限AR表示|无限 AR 逆滤波]]：
$$\varepsilon_t=\frac{\phi(B)}{\theta(B)}(X_t-\mu).$$
它不是把 $X_t$ 重新宣布成有限阶 AR，而是说恢复今天的创新可能要用一段无限长的观测历史。有限样本中的初值处理正因此不可避免。
<!-- bilingual-en:start -->
Invertibility yields an infinite-AR inverse filter for the innovation. It does not turn the observed process into a finite-order AR; it explains why finite-sample estimation must handle the unavailable pre-sample history.
<!-- bilingual-en:end -->

[[Wold分解边界|Wold 分解]]把 ARMA 放进更大的图景：一般宽平稳过程可分为可由无限过去线性预测的确定性部分，以及由一步创新驱动的纯非确定部分。只有后者必有单边创新表示；有限阶 ARMA 又只是其中传递函数为有理式的一小类。宽平稳不等于“必然是某个有限 ARMA”。
<!-- bilingual-en:start -->
Wold decomposition separates a covariance-stationary process into a linearly deterministic part and a purely nondeterministic innovation-driven part. A finite ARMA is only a rational, finite-parameter subclass of the latter; stationarity does not imply a finite ARMA order.
<!-- bilingual-en:end -->

## 6. ACF、PACF 与 Yule–Walker：从总体规律到候选
<!-- bilingual-en:start -->
*6. ACF, PACF, and Yule–Walker: from population structure to candidates*
<!-- bilingual-en:end -->

[[偏自相关函数|$k$ 阶 PACF]]是把 $X_t$ 对 $X_{t-1},\ldots,X_{t-k}$ 作线性投影时最后一个总体系数。它先剔除中间滞后的线性传递：AR(1) 可以有 $\rho(2)=\phi^2\ne0$，但控制 $X_{t-1}$ 后 $X_{t-2}$ 不再增加线性信息，因此 $\alpha(2)=0$。
<!-- bilingual-en:start -->
The lag-$k$ PACF is the final coefficient in the population linear projection on lags one through $k$. It removes dependence transmitted through the intervening lags, which is why AR(1) can have nonzero lag-two autocorrelation but zero lag-two partial autocorrelation.
<!-- bilingual-en:end -->

在因果、可逆且最小的理想总体模型中：

| 模型 | 总体 ACF | 总体 PACF |
|---|---|---|
| AR($p$) | 拖尾 | $p$ 阶后截尾 |
| MA($q$) | $q$ 阶后截尾 | 拖尾 |
| ARMA($p,q$) | 通常拖尾 | 通常拖尾 |

<!-- bilingual-en:start -->
For ideal causal, invertible, minimal population models, an AR($p$) has a tailing ACF and a PACF cutoff after $p$; an MA($q$) has the reverse pattern; a mixed ARMA generally has two tails.
<!-- bilingual-en:end -->

[[ACF-PACF阶数识别]]的关键词是“提出候选”。样本图有抽样误差，近单位根、季节性、结构突变、预处理和混合模型都会模糊形状。一个孤立尖峰既不证明真实阶数，也不能替代估计和诊断。
<!-- bilingual-en:start -->
ACF/PACF patterns propose candidates. Sampling error, near-unit roots, seasonality, breaks, preprocessing, and mixed dynamics can all blur the textbook shapes. A single spike is not a model certificate.
<!-- bilingual-en:end -->

对纯 AR($p$)，[[Yule-Walker方程]]把模型参数与总体自协方差连接起来：
$$\gamma(k)=\sum_{i=1}^p\phi_i\gamma(k-i),\qquad k\ge1,$$
而 $k=0$ 的方程另含创新方差。用样本自协方差替代得到矩估计，不是一般 ARMA 的精确 MLE；含 MA 项时低阶方程还会出现创新交叉项。
<!-- bilingual-en:start -->
For a pure AR($p$), the Yule–Walker equations link the AR coefficients to an autocovariance recursion. Replacing population moments with sample moments gives a method-of-moments estimator, not a general exact likelihood for MA or ARMA models.
<!-- bilingual-en:end -->

## 7. 估计、选阶和诊断必须形成闭环
<!-- bilingual-en:start -->
*7. Estimation, order selection, and diagnosis must form a loop*
<!-- bilingual-en:end -->

AR 的条件最小二乘可在给定最初 $p$ 个观测后用 OLS；含 MA 项时，创新要按候选参数递归重建，问题变成非线性优化。[[ARMA似然初值处理]]区分三种不能混称的做法：CSS 给定样本前误差后最小化平方和；条件 Gaussian likelihood 在同样初值条件下加入分布假设；精确 Gaussian likelihood 则用状态空间方法计入初始状态不确定性。不同目标、有效样本和常数项口径下的 likelihood 不应直接混排。
<!-- bilingual-en:start -->
Conditional least squares for a pure AR can be OLS after conditioning on initial observations. MA terms make the innovations latent and the objective nonlinear. CSS, conditional Gaussian likelihood, and exact Gaussian likelihood treat the pre-sample state differently and must not be compared as though they were one likelihood.
<!-- bilingual-en:end -->

[[Box-Jenkins流程]]是循环，不是一次性菜单：
$$\text{明确并平稳化对象}\to\text{提出少量候选}\to\text{估计}
\to\text{诊断}\to\text{必要时重设}.$$
[[ARMA信息准则|AIC、AICc 与 BIC]]只在相同响应、有效样本、变换、似然定义和参数计数下比较候选。较小值表示相对偏好，不证明模型为真；不同 $d$ 改变响应数据，不能让 AIC 独自选择差分阶数。
<!-- bilingual-en:start -->
Box–Jenkins iterates among defining and stationarising the target, proposing a small candidate set, estimating, diagnosing, and respecifying. Information criteria rank comparable fits; a smaller score is not proof of truth, and AIC should not choose among likelihoods based on different differenced responses.
<!-- bilingual-en:end -->

[[Ljung-Box检验]]联合检验前 $\ell$ 个自相关是否为零：
$$Q^*=n(n+2)\sum_{k=1}^{\ell}\frac{r_k^2}{n-k}.$$
对拟合残差，参考自由度必须按实际模型和软件实现调整；常规非季节 ARMA 常用 $\ell-(p+q)$，但不能机械地永远减同一个数。[[ARMA残差诊断]]还要求看残差路径、均值、ACF、异常点、分布尾部与结构变化。未拒绝只表示在这些滞后和检验力下没有检测到剩余线性相关。
<!-- bilingual-en:start -->
Ljung–Box is a joint test across selected lags. Fitted residuals require a model-specific degrees-of-freedom adjustment. A non-rejection means that the test did not detect remaining linear correlation at those lags and with that power; it does not prove i.i.d., Gaussian, homoscedastic errors or a true model.
<!-- bilingual-en:end -->

即使原残差 ACF 合格，仍应检查绝对值或平方残差。[[波动率聚集]]表现为方向的线性相关很弱，而波动幅度持续相关；这说明均值方程可能尚可，但条件方差需要进入[[条件异方差：ARCH 与 GARCH.canvas|ARCH/GARCH]]，而不是继续往均值方程里盲目加滞后。
<!-- bilingual-en:start -->
An adequate residual ACF does not end the diagnosis. Persistent dependence in absolute or squared residuals signals volatility clustering: the mean equation may be adequate while the conditional variance still needs an ARCH/GARCH model.
<!-- bilingual-en:end -->

## 8. ARIMA 是更换对象后再用 ARMA
<!-- bilingual-en:start -->
*8. ARIMA applies ARMA after changing the target*
<!-- bilingual-en:end -->

[[ARIMA模型|ARIMA($p,d,q$)]]写成
$$\phi(B)(1-B)^dX_t=c+\theta(B)\varepsilon_t.$$
它对 $W_t=(1-B)^dX_t$ 拟合 ARMA($p,q$)，而不是假装原水平已经平稳。$d$ 决定对象，$p,q$ 决定差分对象的短期动态。如何区分确定性趋势和单位根、以及怎样解释差分，见[[趋势、单位根与差分.canvas|趋势、单位根与差分]]。
<!-- bilingual-en:start -->
ARIMA fits ARMA dynamics to the $d$-times differenced target. The differencing order chooses the response; the AR and MA orders describe its short-run dependence. Deterministic trends, unit roots, and the meaning of differencing belong to the separate trend-and-unit-root path.
<!-- bilingual-en:end -->

差分遵循[[最小差分原则]]。对白噪声多差一次，$\Delta\varepsilon_t=\varepsilon_t-\varepsilon_{t-1}$ 会产生 $\rho(1)=-1/2$，并在本库符号下得到 $\theta=-1$、MA 根位于单位圆的不可逆边界。差分不是越多越稳；它可能主动制造你随后又试图用 ARMA 解释的动态。
<!-- bilingual-en:start -->
Use the smallest defensible differencing order. Differencing white noise once creates lag-one autocorrelation $-1/2$ and an MA root on the noninvertible boundary under this vault's sign convention. Differencing can manufacture the very dynamics that a later ARMA fit then tries to explain.
<!-- bilingual-en:end -->

## 9. 从点预测到诚实的区间和评估
<!-- bilingual-en:start -->
*9. From point forecasts to honest intervals and evaluation*
<!-- bilingual-en:end -->

[[ARMA多步预测]]向前递推时使用已观测的过去值与 fitted innovations，并把预测起点之后尚未发生的创新在线性投影中置零。若创新是 MDS，这给出平方损失下的条件均值；若只有二阶白噪声，它最稳妥地只称为最佳线性预测。
<!-- bilingual-en:start -->
Recursive ARMA forecasting uses observed past values and fitted past innovations, while future innovations have zero linear projection. With MDS innovations this is the conditional mean; with second-order white noise alone it is only the best linear predictor.
<!-- bilingual-en:end -->

[[AR(1)多步预测]]把长期行为算得最清楚。对
$$X_t-\mu=\phi(X_{t-1}-\mu)+\varepsilon_t,\qquad |\phi|<1,$$
有
$$\widehat X_{t+h|t}=\mu+\phi^h(X_t-\mu),$$
$$\operatorname{MSE}(e_{t+h|t})=\sigma^2\sum_{j=0}^{h-1}\phi^{2j}
=\sigma^2\frac{1-\phi^{2h}}{1-\phi^2}.$$
远期点预测回到 $\mu$，误差却扩大到过程的无条件方差；这不是未来变得更稳定，而是今天的信息逐渐失去预测力。
<!-- bilingual-en:start -->
For stable AR(1), the point forecast reverts to the mean while the forecast-error variance grows toward the unconditional process variance. The future is not becoming more stable; current information is losing predictive power.
<!-- bilingual-en:end -->

[[ARMA预测区间]]还需要分布或近似。已知参数且创新 jointly Gaussian 时，可写
$$\widehat X_{t+h|t}\pm z_{1-\alpha/2}\sqrt{\operatorname{Var}(e_{t+h|t})}.$$
实际 plug-in 区间通常只含未来创新不确定性，忽略参数估计和模型变化；厚尾、偏态或条件异方差又会破坏对称 Gaussian 形状。区间是否可信，最终要按 horizon 检查实际覆盖率。
<!-- bilingual-en:start -->
A Gaussian prediction interval needs a Gaussian forecast-error distribution or an explicit approximation. Plug-in formulas often include only future-innovation uncertainty, omitting parameter and model uncertainty. Heavy tails, skewness, or conditional heteroskedasticity can invalidate symmetric intervals, so coverage must be checked by horizon.
<!-- bilingual-en:end -->

[[滚动起点评估]]在每个历史预测起点只使用当时可得的信息，重新执行变换、选阶与估计，并分别评价所需 horizon。expanding window 保留全部旧数据，fixed rolling window 丢弃最早数据以适应变化但增加估计方差。validation origins 用来选模型；最终 test period 在选择结束前保持未看。所有比较都服从[[预测时点与信息集]]，不能把随机打乱的未来数据偷偷放回过去。
<!-- bilingual-en:start -->
Rolling-origin evaluation repeats the full fitting pipeline using only information available at each historical origin and scores each decision-relevant horizon. Expanding and fixed rolling windows trade information against adaptation. Use validation origins for selection and preserve a final untouched period for honest assessment.
<!-- bilingual-en:end -->

## 10. 一条可以直接执行的 ARMA 路径
<!-- bilingual-en:start -->
*10. An executable ARMA workflow*
<!-- bilingual-en:end -->

1. 写清目标、频率、预测期限与损失；画原序列，记录变换、异常点和可能断点。
2. 确定要建模的平稳对象；若差分，只用最小合理阶数并保留尺度解释。
3. 查看 ACF/PACF 与领域机制，提出少量相邻的 $p,q$ 候选，不从单个尖峰宣布答案。
4. 对每个候选说明符号约定和估计方法，检查因果性、可逆性与 AR/MA 互素性。
5. 只在可比 likelihood 问题中使用 AIC/AICc/BIC；随后检查 residual ACF、Ljung–Box、绝对/平方残差、异常点和稳定性。
6. 诊断失败就回到设定，而不是继续预测；诊断尚可的候选才进入 rolling-origin。
7. 按实际 horizon 与基准比较点损失和区间覆盖；选择完成后，只在 untouched test 上报告一次最终表现。

<!-- bilingual-en:start -->
**1.** Define the target, frequency, horizons, and loss; inspect transformations, outliers, and breaks.<br>
**2.** Choose a stationary modelling target and use only defensible differencing.<br>
**3.** Use ACF/PACF and mechanism to propose a small neighbouring candidate set.<br>
**4.** State the sign convention and estimator; check causality, invertibility, and coprimeness.<br>
**5.** Compare only compatible likelihoods, then diagnose residual linear and magnitude dependence.<br>
**6.** Respecify after diagnostic failure; only adequate candidates proceed to rolling-origin evaluation.<br>
**7.** Compare decision-relevant horizons and interval coverage, then report one untouched final test.
<!-- bilingual-en:end -->

如果只能记住一句话：**ARMA 不是从 ACF 图里读出的真模型，而是一套从平稳对象、创新表示、可识别参数化到时间顺序预测证据的可反驳工作流。**
<!-- bilingual-en:start -->
If only one sentence remains: **ARMA is not a true model read from an ACF plot; it is a falsifiable workflow from a stationary target and innovation representation to an identified parameterisation and chronological forecast evidence.**
<!-- bilingual-en:end -->

## 来源与核验

- [[01_Math/06_时间序列分析/lecture.pdf#page=73|课程讲义 pp. 73–132, 257–259]]：支持课程顺序、白噪声与 MDS、ARMA 定义、根、ACF/PACF、估计、Box–Jenkins、预测和 ARIMA。
- [MIT OCW 18.S096, Time Series Analysis I](https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/1926c83ecd7ea700f7cb63914c6d7c0f_MIT18_S096F13_lecnote8.pdf)：支持 Wold、innovation、AR/MA/ARMA、根、Yule–Walker、ARIMA 与估计边界。
- [Hyndman & Athanasopoulos, FPP3 Chapter 9](https://otexts.com/fpp3/arima.html)：支持 ARIMA 建模、ACF/PACF、信息准则与诊断工作流。
- [FPP3 §5.4](https://otexts.com/fpp3/diagnostics.html)、[§5.5](https://otexts.com/fpp3/prediction-intervals.html) 与 [§5.10](https://otexts.com/fpp3/tscv.html)：分别支持残差诊断、预测区间与 rolling-origin 评估。
- [R `stats::arima` documentation](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/arima.html) 与 [statsmodels `acorr_ljungbox`](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_ljungbox.html)：核对 CSS/exact likelihood 初始化及 Ljung–Box 的实现边界。

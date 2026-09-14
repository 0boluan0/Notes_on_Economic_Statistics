# VAR：从联合预测到可识别的冲击与有边界的 Granger 关系
<!-- bilingual-en:start -->
*VAR: from joint forecasting to identified shocks and carefully bounded Granger relations*
<!-- bilingual-en:end -->

VAR 的学习主线不是“把几条回归放在一起”。真正的顺序是：先用共同滞后建立一个简约型联合预测系统；再检查规格、估计条件与稳定性；把多期预测误差写成创新的传播；最后才决定问题是否需要结构识别。只有已识别冲击才能承担结构 IRF 和结构 FEVD 的经济名称，而 Granger 因果始终回答信息增量问题。[[VAR、脉冲响应与 Granger 因果.canvas|主题 Canvas]] 展示全局关系；本章把这些关系排成一条连续推理路径。单方程的外部输入、ADL 与 intervention analysis 先读 [[01_Math/06_时间序列分析/08_动态回归与干预|动态回归与干预]]；课堂板书和作业仍保留在[[01_Math/06_时间序列分析/05_多方程模型Multi-equation Time Series Models|课堂记录]]中。
<!-- bilingual-en:start -->
The learning path is not “put several regressions together.” Build a reduced-form joint forecasting system from common lags, check specification and stability, express multi-step forecast errors as propagated innovations, and only then ask whether the problem needs structural identification. Economic names belong to identified structural shocks, while Granger causality remains a statement about incremental information. The Canvas supplies the global map; this chapter supplies the continuous argument. Study single-equation external inputs, ADLs, and intervention analysis first in the [[01_Math/06_时间序列分析/08_动态回归与干预|dynamic-regression and intervention path]]; the original board work and exercises remain in the [[01_Math/06_时间序列分析/05_多方程模型Multi-equation Time Series Models|course record]].
<!-- bilingual-en:end -->

## 1. 从单变量记忆转向系统反馈
<!-- bilingual-en:start -->
*1. From univariate memory to system feedback*
<!-- bilingual-en:end -->

单变量 [[AR(p)模型|AR($p$)]] 只让一条序列的过去预测自身。若利率、通胀和产出互相提供预测信息，逐条建立彼此隔离的 AR 会遗漏跨变量反馈。[[VAR(p)模型|VAR($p$)]] 把它们堆成 $K$ 维向量 $y_t$，并让每个分量使用全部系统变量的前 $p$ 期：
$$
y_t=c+A_1y_{t-1}+\cdots+A_py_{t-p}+Dx_t+u_t.
$$
$A_i$ 的第 $(r,s)$ 个元素描述：在控制其余已列滞后后，变量 $s$ 的第 $i$ 阶滞后怎样进入变量 $r$ 的线性预测。$x_t$ 可容纳趋势、季节项或真正外生的控制；它们不是 VAR 动态系数的一部分。
<!-- bilingual-en:start -->
A univariate AR uses a series' own past. A VAR stacks mutually informative variables into a vector and lets every equation use the same set of system lags. Entry $(r,s)$ of $A_i$ measures how lag $i$ of variable $s$ enters the linear prediction of variable $r$, conditional on the other listed lags. Deterministic and genuinely exogenous terms may be added separately.
<!-- bilingual-en:end -->

这个方程首先是**简约型**动态。若只假定二阶矩，$u_t$ 是相对于系统过去的最佳线性预测误差；若另有 $E(u_t\mid\mathcal F_{t-1})=0$，它才同时是完整信息集下的条件均值误差。把所有变量共同建模并不自动解决遗漏变量、测量误差或政策预期引起的内生性，更不自动决定同期作用方向。
<!-- bilingual-en:start -->
This is initially a reduced-form dynamic model. Under second-order assumptions, $u_t$ is a linear projection error; under the stronger martingale-difference condition it is also a conditional-mean error. Joint modelling does not by itself remove omitted-variable, measurement-error, or policy-anticipation problems, nor does it identify contemporaneous directions.
<!-- bilingual-en:end -->

若各变量为 $I(1)$ 且存在协整，机械逐条差分会删除误差修正通道。应先用[[协整与差分边界|整合与协整判断]]决定保留水平 VAR、使用差分 VAR，还是通过 [[VAR到VECM重参数化|VAR–VECM 重参数化]]显式保留长期约束。
<!-- bilingual-en:start -->
When variables are integrated and cointegrated, mechanical componentwise differencing can delete the error-correction channel. The choice among a level VAR, differenced VAR, and VECM must follow the integration properties and the inferential objective.
<!-- bilingual-en:end -->

## 2. 模型越宽，规格选择越重要
<!-- bilingual-en:start -->
*2. Wider systems make specification more consequential*
<!-- bilingual-en:end -->

一个 $K$ 变量 VAR($p$) 有 [[VAR参数量|$K^2p$ 个动态系数]]。若每条方程另有 $d$ 个确定或外生项，条件均值参数再增加 $Kd$ 个；高斯似然还需估计 $\Sigma_u$ 的 $K(K+1)/2$ 个自由元素。三变量 VAR(4) 仅滞后矩阵就有 $3^2\times4=36$ 个系数，所以“再加一个变量看看”会扩展每一条方程，而不是只多一个斜率。
<!-- bilingual-en:start -->
A $K$-variable VAR($p$) has $K^2p$ dynamic coefficients, plus $Kd$ deterministic or exogenous mean parameters and, under Gaussian likelihood, $K(K+1)/2$ innovation-covariance parameters. Adding a variable expands every equation, so dimensionality grows quickly.
<!-- bilingual-en:end -->

[[VAR规格选择]]必须从任务开始。联合预测关心样本外误差；Granger 检验关心比较信息集；结构响应还要求变量集足以使冲击名称和排除限制可信。AIC、AICc、BIC/SC 可在相同样本与同一因变量系统的可比候选间排序，但不能代替经济目标、数据频率、单位根与协整判断。阶数选定后仍应复用 [[ARMA残差诊断|残差序列相关诊断]]、稳定根检查和相邻规格敏感性分析。
<!-- bilingual-en:start -->
Specification starts from the task. Forecasting, Granger testing, and structural analysis need different information sets and validation. Information criteria rank comparable candidates; they do not replace the objective, data frequency, integration analysis, residual diagnostics, stability checks, or sensitivity to neighboring specifications.
<!-- bilingual-en:end -->

在每条简约型方程使用同一组滞后回归量时，[[VAR逐方程OLS|逐方程 OLS]] 与共同回归量下的多元最小二乘给出相同的系数点估计。跨方程创新同期相关不会单独造成系数偏误，但估计仍需要创新对过去回归量正交、设计矩阵有足够秩，以及保证样本矩收敛的动态正则条件；[[OLS一致性条件]]与[[样本正交与总体外生性|样本残差正交不证明总体外生]]仍然适用。
<!-- bilingual-en:start -->
With identical lagged regressors in every reduced-form equation, equation-by-equation OLS gives the same coefficient estimates as multivariate least squares. Contemporaneous cross-equation correlation does not alone bias those coefficients, but orthogonality to past regressors, adequate rank, and dynamic regularity are still required.
<!-- bilingual-en:end -->

> [!example] 一个最小二变量系统
> 考虑
> $$
> \begin{pmatrix}y_t\\x_t\end{pmatrix}
> =
> \begin{pmatrix}0.5&0.2\\0&0.4\end{pmatrix}
> \begin{pmatrix}y_{t-1}\\x_{t-1}\end{pmatrix}
> +u_t.
> $$
> $x$ 的上一期会进入 $y$ 的预测，而 $y$ 的上一期不进入 $x$ 的预测。这只是给定规格中的预测方向；尚不能把 $u_{xt}$ 命名为对 $x$ 的外生经济干预。

<!-- bilingual-en:start -->
> [!example] A minimal bivariate system
> In the displayed VAR, lagged $x$ enters the prediction of $y$, whereas lagged $y$ does not enter the prediction of $x$. This is a predictive direction in the stated specification; it does not identify $u_{xt}$ as an external economic intervention on $x$.
<!-- bilingual-en:end -->

## 3. 伴随形式把高阶 VAR 变成一个状态系统
<!-- bilingual-en:start -->
*3. Companion form turns a higher-order VAR into one state system*
<!-- bilingual-en:end -->

把 $y_t,\ldots,y_{t-p+1}$ 堆成 $Kp$ 维状态 $Y_t$，[[VAR伴随形式]]写成
$$
Y_t=FY_{t-1}+U_t.
$$
这不是另一个模型，而是同一系统的扩维表达。它让 $F^h$ 同时编码所有滞后矩阵在 $h$ 期内的组合传播，并把稳定性、预测、VMA 和状态协方差放到同一个线性系统框架中。
<!-- bilingual-en:start -->
Stacking the current vector and its $p-1$ lags creates the first-order state equation $Y_t=FY_{t-1}+U_t$. Companion form is a re-expression of the same model; powers of $F$ collect all lag interactions across a horizon and unify stability, forecasting, VMA propagation, and state covariance.
<!-- bilingual-en:end -->

[[VAR稳定根条件|稳定性]]有两个等价但方向相反的口径：
$$
\rho(F)<1
\quad\Longleftrightarrow\quad
\det\!\left(I_K-A_1z-\cdots-A_pz^p\right)\ne0
\ \text{对所有 }|z|\le1.
$$
前者说伴随矩阵特征值在单位圆内；后者说滞后多项式的根在单位圆外。软件若报告 inverse roots，必须先确认输出对象再判断“内”还是“外”。这一条件复用了[[离散系统谱稳定性|离散线性系统的谱稳定性]]，但 [[非正规矩阵瞬态|非正规矩阵]] 仍可能在最终衰减前出现较大的有限期响应。
<!-- bilingual-en:start -->
Stability can be stated as companion eigenvalues inside the unit circle or lag-polynomial roots outside it. The objects are reciprocals, so software output must be identified before interpreting “inside” and “outside.” Asymptotic spectral stability also does not preclude large finite-horizon transients from nonnormal dynamics.
<!-- bilingual-en:end -->

在稳定性与有限创新二阶矩下，[[VAR因果VMA表示|唯一因果平稳解]]为
$$
y_t-\mu=\sum_{h=0}^{\infty}\Phi_hu_{t-h},
\qquad \Phi_0=I_K,
$$
其中
$$
\Phi_h=\sum_{i=1}^{\min(p,h)}A_i\Phi_{h-i}.
$$
只有 VAR(1) 才可写 $\Phi_h=A_1^h$。这条递推是后续预测、IRF 与 FEVD 的共同发动机，也是多变量版本的 [[ARMA无限MA表示|无限 MA 展开]]。
<!-- bilingual-en:start -->
Under stability and finite second moments, the unique causal stationary solution is a VMA with recursively generated matrices $\Phi_h$. Only a VAR(1) has $\Phi_h=A_1^h$; higher-order systems require the full recursion or companion powers. The same sequence drives forecasts, impulse responses, and variance decompositions.
<!-- bilingual-en:end -->

稳定性保证任意有限初值的影响最终消失，却不表示从任意固定初值启动的早期样本已经平稳。只有按平稳分布初始化时，过程才从第一期起拥有时间不变的均值与协方差。
<!-- bilingual-en:start -->
Stability makes a finite initial-condition effect vanish; it does not make the early path from an arbitrary fixed start stationary. Stationarity from the first date requires initialization from the stationary distribution.
<!-- bilingual-en:end -->

## 4. 预测误差与无条件协方差是两个不同对象
<!-- bilingual-en:start -->
*4. Forecast-error and unconditional covariance are different objects*
<!-- bilingual-en:end -->

给定时点 $t$ 的信息，[[VAR多步预测误差|$h$ 步预测误差]]只由尚未观察到的 $u_{t+1},\ldots,u_{t+h}$ 构成：
$$
y_{t+h}-\widehat y^L_{t+h\mid t}
=\sum_{i=0}^{h-1}\Phi_i u_{t+h-i}.
$$
若创新跨期不相关且 $\operatorname{Cov}(u_t)=\Sigma_u$，总预测误差协方差为
$$
\Sigma_h=\sum_{i=0}^{h-1}\Phi_i\Sigma_u\Phi_i'.
$$
这个对象完全由简约型 VAR 决定，不需要先命名结构冲击。结构识别只在进一步问“每一个经济冲击贡献多少”时进入。
<!-- bilingual-en:start -->
An $h$-step forecast error contains only innovations that arrive after the forecast origin. Its covariance is the finite sum above and is fully determined by the reduced form. Identification becomes necessary only when total uncertainty is to be allocated to named economic shocks.
<!-- bilingual-en:end -->

[[VAR无条件协方差|同期无条件协方差]]则把无限多期过去创新对当前值的方差贡献相加。对 VAR(1)，
$$
\Omega=A\Omega A'+\Sigma_u
=\sum_{i=0}^{\infty}A^i\Sigma_u(A')^i.
$$
它解离散 Lyapunov 方程。一般不能写成
$$
(I-A)^{-1}\Sigma_u(I-A')^{-1},
$$
因为后者把长期累计乘数放在协方差两侧，包含不同滞后间的交叉项；它不是同一期 $y_t$ 的方差。稳定时 $\Sigma_h\to\Omega$，但有限期预测误差与无条件方差的含义仍不能混称。
<!-- bilingual-en:start -->
The contemporaneous unconditional covariance sums the effects of the entire innovation history and solves a discrete Lyapunov equation. Sandwiching $\Sigma_u$ between long-run multipliers generally adds cross-lag terms and is not the same object. Although the forecast-error covariance converges to $\Omega$ in a stable system, their finite-horizon interpretations remain distinct.
<!-- bilingual-en:end -->

## 5. 创新是预测新息，不是已经命名的经济冲击
<!-- bilingual-en:start -->
*5. Innovations are forecasting news, not already named economic shocks*
<!-- bilingual-en:end -->

[[简约型VAR创新]]定义了相对于过去的当期不可预测部分。不同方程的创新可以同期相关：
$$
\operatorname{Cov}(u_t)=\Sigma_u
$$
不必为对角矩阵。共同新闻、未建模的同期传导和时间聚合都可能让多个方程在同一期一起预测失误。这不妨碍简约型联合预测，却触发了[[简约型创新不是结构冲击|创新与结构冲击的边界]]。
<!-- bilingual-en:start -->
A reduced-form innovation is news relative to the past, and its covariance need not be diagonal. Common news, contemporaneous transmission, and temporal aggregation may create same-period forecast errors across equations. This is compatible with reduced-form forecasting but prevents automatic structural naming.
<!-- bilingual-en:end -->

假定
$$
u_t=B\varepsilon_t,\qquad E(\varepsilon_t\varepsilon_t')=I_K.
$$
数据给出 $\Sigma_u=BB'$。但对任意正交矩阵 $Q$，$(BQ)(BQ)'=BB'$，所以相同的简约型拟合兼容多组冲击方向。[[结构VAR]]正是在简约型动态上增加当期关系和可解释冲击的模型；它不是把残差列名改成“需求”“供给”“政策”。
<!-- bilingual-en:start -->
If $u_t=B\varepsilon_t$, the reduced form reveals $BB'$, not a unique $B$: every orthogonal rotation $BQ$ produces the same covariance. An SVAR augments the reduced-form dynamics with a contemporaneous map and interpretable shocks; it is not created by relabelling residual columns.
<!-- bilingual-en:end -->

在单位冲击方差的 $B$ 参数化下，$B$ 有 $K^2$ 个元素，$\Sigma_u$ 只提供 $K(K+1)/2$ 个独立矩。[[SVAR识别条件|常见精确识别]]至少还需
$$
\frac{K(K-1)}2
$$
个独立、有效的限制。但这是阶条件，不是最终证明：限制还需满足相应秩条件，并且经济上可信。数量不足通常欠识别；数量更多可能过度识别；数量刚好也可能因为限制依赖或全局多解而未被唯一识别。
<!-- bilingual-en:start -->
With unit-variance shocks, the familiar impact-matrix parameterization needs at least $K(K-1)/2$ additional independent restrictions for exact identification. This counting condition is not a proof: rank, local or global uniqueness, and economic credibility still matter.
<!-- bilingual-en:end -->

## 6. 四类识别方案回答不同问题
<!-- bilingual-en:start -->
*6. Four identification strategies answer different questions*
<!-- bilingual-en:end -->

[[Cholesky递归识别]]取
$$
\Sigma_u=PP',\qquad P\ \text{为正对角下三角矩阵},
$$
并令 $u_t=P\varepsilon_t$。在这个方向下，$P_{ij}=0$（$j>i$）表示第 $i$ 个变量当期不响应排在其后的第 $j$ 个冲击。[[Cholesky 正定判据|分解的存在唯一性]]是线性代数事实；变量顺序所代表的同期零限制是否可信却是经济问题。换顺序会换结构 IRF 和 FEVD。
<!-- bilingual-en:start -->
Recursive identification uses a lower-triangular Cholesky impact matrix for a chosen ordering. The algebraic factor is unique for a positive-definite covariance, but the ordering imposes directed contemporaneous exclusions. Changing the order changes the structural responses and FEVDs.
<!-- bilingual-en:end -->

[[长期识别限制]]不把当期影响设为零，而约束结构冲击的累计效应：
$$
\Theta(1)=\left(I-\sum_{i=1}^pA_i\right)^{-1}B.
$$
若建模变量是原水平的一阶差分，差分响应的累计和才对应原水平的长期变化。稳定水平变量的远期点响应本来就趋于零；把响应总和设为零是更强、含义不同的限制。单位根或协整系统还需在 VECM 或永久—暂时分解中重写长期对象。
<!-- bilingual-en:start -->
Long-run restrictions constrain cumulative structural effects rather than impact coefficients. A cumulative response in a differenced variable maps to a level effect; for a stationary level, a zero response sum is stronger than the already vanishing distant-horizon point response. Integrated systems require an explicitly appropriate long-run representation.
<!-- bilingual-en:end -->

[[符号限制集合识别]]只要求若干响应在指定期限为正、负或非负。它通常留下多个满足条件的正交旋转，因此结果是一个可接受结构集合。逐期限中位数未必来自同一个旋转；抽样不确定性与识别集合不确定性都应保留。额外零、叙事或幅度限制可能缩小集合，但不能因为软件只画出一条线就宣布点识别。
<!-- bilingual-en:start -->
Sign restrictions usually leave a set of admissible rotations. Pointwise median responses need not come from one common structural model, so sampling uncertainty and identification uncertainty must both remain visible. Extra restrictions may shrink the set, but a single plotted curve is not evidence of point identification.
<!-- bilingual-en:end -->

[[Proxy SVAR]]用外部工具 $z_t$ 与简约型创新的共变动来识别一个目标冲击的影响方向：
$$
E(z_tu_t)=b_1E(z_t\varepsilon_{1t})
$$
在工具只关联目标冲击时与 $b_1$ 平行。这种方法可以只识别一个目标冲击，不必命名整个正交补空间；但[[外部工具识别条件]]要求工具对目标冲击相关、对其他冲击外生，还要处理尺度、弱工具与所用传播方法的可恢复性或动态排除条件。
<!-- bilingual-en:start -->
A Proxy SVAR uses the covariance between an external instrument and reduced-form innovations to identify a target shock's impact direction up to scale. It may identify only that target shock, but validity requires relevance, exogeneity to the other shocks, an economic normalization, weak-instrument diagnostics, and the recoverability or dynamic exclusions required by the propagation method.
<!-- bilingual-en:end -->

识别方案不是可互换的技术按钮。递归限制诉诸当期时序，长期限制诉诸永久与暂时效应，符号限制诉诸响应方向，外部工具诉诸外生信号。选择哪一种，应由目标冲击的制度含义决定，而不是由哪一种最容易给出漂亮曲线决定。
<!-- bilingual-en:start -->
Identification schemes are not interchangeable buttons. Recursive restrictions invoke contemporaneous timing, long-run restrictions invoke permanent effects, sign restrictions invoke response directions, and external instruments invoke outside variation. The economic meaning of the target shock should determine the choice.
<!-- bilingual-en:end -->

## 7. IRF、GIRF 与 FEVD 不回答同一个问题
<!-- bilingual-en:start -->
*7. IRF, GIRF, and FEVD answer different questions*
<!-- bilingual-en:end -->

识别出 $u_t=B\varepsilon_t$ 后，[[结构脉冲响应]]由
$$
\Theta_h=\Phi_hB
$$
给出。元素 $\theta_{ij,h}$ 是第 $j$ 个已识别、已定标结构冲击在期限 $h$ 对变量 $i$ 的**点响应**。累计响应 $\sum_{s=0}^h\theta_{ij,s}$ 是另一对象。报告必须写明冲击是一单位还是一标准差、变量是水平/对数/差分/增长率、是否累计、期限以及抽样区间或识别集合。
<!-- bilingual-en:start -->
Once $u_t=B\varepsilon_t$ is identified, structural responses are $\Theta_h=\Phi_hB$. An element is a horizon-specific point response to an identified and scaled shock; its cumulative sum is another object. Shock scale, variable units and transformations, cumulative convention, horizon, and uncertainty must accompany the graph.
<!-- bilingual-en:end -->

回到前面的二变量 VAR，若一个可辩护的递归识别给出
$$
P=
\begin{pmatrix}1&0\\0.3&0.95\end{pmatrix},
$$
第二个结构冲击的当期影响为 $(0,0.95)'$。随后
$$
\Theta_1e_2=A_1Pe_2=(0.19,0.38)',\qquad
\Theta_2e_2=A_1^2Pe_2=(0.171,0.152)'.
$$
$x$ 的冲击起初不影响 $y$，却通过 $x_{t-1}$ 进入 $y_t$，所以 $y$ 的响应在下一期才出现。这个解释成立的前提不是矩阵乘法本身，而是 $P$ 的变量顺序和冲击名称确实可信。
<!-- bilingual-en:start -->
In the numerical example, a recursively identified second shock has impact vector $(0,0.95)'$, then responses $(0.19,0.38)'$ and $(0.171,0.152)'$. The effect on $y$ appears with a lag through the cross-lag coefficient. The economic interpretation depends on the credibility of the ordering, not on the matrix multiplication alone.
<!-- bilingual-en:end -->

[[广义脉冲响应]]则对某个简约型创新的条件变化取平均。在线性 VAR 中，
$$
\operatorname{GIRF}_j(h)
=\Phi_h\Sigma_u e_j\,\sigma_{jj}^{-1/2}.
$$
它把与 $u_{jt}$ 同期相关的其他创新的平均伴随变化纳入，因此不需要 Cholesky 排序。[[广义脉冲响应边界|排序不变不等于结构识别]]：若 $u_{jt}$ 本身混合多个经济冲击，GIRF 只是沿这个混合方向传播，不能因不依赖排序就自动叫作结构因果响应。
<!-- bilingual-en:start -->
A GIRF averages the conditional movement associated with one reduced-form innovation and is invariant to Cholesky ordering in a linear VAR. Ordering invariance is not structural identification: if the innovation mixes economic shocks, the GIRF propagates that same mixture.
<!-- bilingual-en:end -->

[[预测误差方差分解|FEVD]]在给定期限 $h$ 下，把变量 $i$ 的预测误差方差分摊给已识别、按所用口径正交的冲击：
$$
\omega_{ij}(h)=
\frac{\sum_{s=0}^{h-1}(e_i'\Theta_se_j)^2}
{\sum_{s=0}^{h-1}\sum_{k=1}^K(e_i'\Theta_se_k)^2}.
$$
它回答“在这个期限和识别方案下，预测错误有多少来自冲击 $j$”，不是无条件的变量方差来源表。份额会随期限和 Cholesky 排序或其他识别方案改变。广义 FEVD 可避免递归排序，但原始份额在相关创新下未必加总为 1，且仍不自动成为结构冲击贡献。
<!-- bilingual-en:start -->
FEVD allocates a variable's horizon-specific forecast-error variance to identified orthogonal shocks. It is not an unconditional table of variance sources; shares change with the horizon and identifying scheme. Generalized FEVD avoids recursive ordering but may require normalization and does not automatically become a structural-shock decomposition.
<!-- bilingual-en:end -->

## 8. Granger 因果检验的是信息增量
<!-- bilingual-en:start -->
*8. Granger causality tests incremental information*
<!-- bilingual-en:end -->

[[Granger因果]]比较两个信息集。令 $\mathcal F_t$ 包含 $x$ 的历史，$\mathcal F_t^{-x}$ 删除这部分历史而保留其他比较信息。$x$ 不 Granger 导致 $y$ 要求对所有相关期限 $h$ 和事件 $A$，
$$
P\{y_{t+h}\in A\mid\mathcal F_t\}
=P\{y_{t+h}\in A\mid\mathcal F_t^{-x}\}.
$$
若加入 $x$ 的过去会改变未来 $y$ 的条件分布，$x$ 就含有增量预测信息。常见线性 VAR 检验只落地到条件均值，不是完整分布定义的自动检验。
<!-- bilingual-en:start -->
Granger causality compares a full information set with one that excludes the history of $x$. Noncausality means that future conditional distributions are unchanged. A standard linear VAR test implements only the conditional-mean version within the selected finite specification.
<!-- bilingual-en:end -->

在二变量 VAR($p$) 的 $y$ 方程
$$
y_t=c+\sum_{i=1}^pa_i y_{t-i}
+\sum_{i=1}^pb_i x_{t-i}+u_{yt}
$$
中，[[VAR Granger检验|非因果原假设]]是
$$
H_0:b_1=\cdots=b_p=0.
$$
它必须作联合 Wald、F 或 LR 检验，不能只看一个滞后。拒绝表示在给定变量、滞后、样本和线性模型下存在增量预测信息；未拒绝只表示证据不足。协整 VECM 中还可能有误差修正通道，需按[[弱外生与Granger非因果|短期与长期限制]]一起判断。
<!-- bilingual-en:start -->
In a linear VAR, noncausality is a joint zero restriction on every relevant lag coefficient in the target equation. Rejection indicates incremental predictive content in the stated specification; failure to reject is not proof of absence. In a VECM, lagged differences and error-correction terms may both transmit predictive information.
<!-- bilingual-en:end -->

[[Granger因果边界]]必须伴随结论：遗漏共同驱动、时间聚合、错误差分、非线性、结构突变或新增控制变量都会改变信息集与结果。Granger 关系不是对 $x$ 作外生干预的反事实效应，也不识别同期共同冲击。合格表述是“在所列信息集与规格下拒绝从 $x$ 到 $y$ 的 Granger 非因果”，而不是“证明 $x$ 导致 $y$”。
<!-- bilingual-en:start -->
Every conclusion inherits the information set and specification. Omitted common drivers, temporal aggregation, transformations, nonlinearities, breaks, and added controls can change the relation. Report rejection or non-rejection of a stated noncausality null; do not translate it into proof of a structural intervention effect.
<!-- bilingual-en:end -->

## 9. 把整条路径变成可执行工作流
<!-- bilingual-en:start -->
*9. Turn the chain into an executable workflow*
<!-- bilingual-en:end -->

[[VAR实证流程]]可以压缩成六个必须按顺序回答的问题：

1. **目标是什么？** 联合预测、描述动态、Granger 检验，还是结构冲击的因果路径？
2. **哪些变量和变换属于信息集？** 写清频率、样本、确定项、外生项、单位根、协整与候选滞后。
3. **简约型是否可估且足够？** 检查正交性、秩、参数量、残差序列相关、稳定根、异方差、异常值与结构变化。
4. **任务真的需要识别吗？** 联合预测和总预测误差协方差可停在简约型；结构 IRF、结构 FEVD 与历史冲击分解不能。
5. **限制为什么可信？** 把递归时序、长期效应、符号方向或外部工具有效性写成可反驳的制度与理论论证。
6. **输出到底是什么对象？** 区分总预测误差、结构 IRF、GIRF、FEVD 与 Granger 关系，并报告各自的期限、尺度、不确定性和解释边界。

<!-- bilingual-en:start -->
An operational VAR workflow asks, in order: what the objective is; what belongs in the information set; whether the reduced form is estimable and diagnostically adequate; whether identification is actually required; why the restrictions are credible; and exactly which object—forecast error, structural IRF, GIRF, FEVD, or Granger relation—is being reported with its horizon, scale, uncertainty, and boundary.
<!-- bilingual-en:end -->

> [!attention] 课程讲义中的四处需要主动纠偏
> 1. [[01_Math/06_时间序列分析/lecture.pdf#page=221|p. 221]] 的“简约型右侧没有当期变量，所以没有内生性”只能排除同期联立回归量；遗漏变量、政策预期与创新—滞后回归量正交仍需检查。
> 2. [[01_Math/06_时间序列分析/lecture.pdf#page=223|p. 223]] 把同期无条件协方差写成长期乘数夹住 $\Sigma_u$，一般不对；正确对象是 Lyapunov 级数。
> 3. [[01_Math/06_时间序列分析/lecture.pdf#page=236|p. 236]] 的二变量 FEVD 第二项下标混写；跨冲击贡献必须使用对应的 $\phi_{12}$ 响应。
> 4. [[01_Math/06_时间序列分析/lecture.pdf#page=240|p. 240]] 说识别重要的是限制数量而非内容；数量和秩只是统计识别的必要部分，经济解释还取决于限制内容是否可信。

<!-- bilingual-en:start -->
> [!attention] Four corrections when reading the lecture slides
> The absence of contemporaneous right-hand-side variables removes one simultaneity problem, not every source of endogeneity. The displayed unconditional-covariance sandwich is generally not the contemporaneous covariance; the Lyapunov sum is. The bivariate FEVD slide mixes an impulse-response index in its cross-shock term. Finally, the number and rank of restrictions are only part of identification: the economic content of those restrictions determines whether the named shocks are credible.
<!-- bilingual-en:end -->

## 10. 来源与核验
<!-- bilingual-en:start -->
*10. Sources and verification*
<!-- bilingual-en:end -->

- [[01_Math/06_时间序列分析/lecture.pdf#page=219|课程讲义 pp. 219–240]]：支持课程中的二变量结构式、简约化、稳定性、估计、IRF、FEVD、Granger 检验与 SVAR 引入；上节已逐项标明需要纠正的公式和解释。
- [Lütkepohl (2005), *New Introduction to Multiple Time Series Analysis*](https://doi.org/10.1007/978-3-540-27752-1)，第 2–4、9 章：支持有限阶 VAR、估计、规格检查、预测与结构模型的系统框架。
- [Kilian & Lütkepohl (2017), *Structural Vector Autoregressive Analysis*](https://doi.org/10.1017/9781108164818)：支持简约型—结构型区分、识别、IRF、FEVD 与推断边界。
- [Stock & Watson (2001), *Vector Autoregressions*](https://www.aeaweb.org/articles?id=10.1257/jep.15.4.101)：支持 VAR 在数据描述、预测、结构推断和政策分析中的任务差异。
- [Granger (1969)](https://doi.org/10.2307/1912791) 与 [Granger (1988)](https://doi.org/10.1016/0304-4076(88)90045-0)：支持信息集定义、线性检验及预测因果的解释边界。
- [Pesaran & Shin (1998)](https://doi.org/10.1016/S0165-1765(97)00214-0)：支持 GIRF 的条件响应、线性闭式与排序不变性。
- [Blanchard & Quah (1989)](https://www.nber.org/papers/w2737)、[Fry & Pagan (2011)](https://www.aeaweb.org/articles?id=10.1257/jel.49.4.938) 与 [Stock & Watson (2018)](https://www.nber.org/papers/w24216)：分别支持长期限制、符号限制与外部工具识别的含义和边界。

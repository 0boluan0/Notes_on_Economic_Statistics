---
aliases:
  - "ETS 似然把平滑参数与初始状态共同作为估计对象"
  - ETS likelihood and initial states
  - ETS initialization and AICc
  - ETS 初始状态估计
student_os: knowledge-atom
atom_id: TS-ETS-007
atom_set: exponential-smoothing-ets
atom_type: estimation-principle
status: source-checked
mastery_state: unassessed
requires:
  - "[[ETS三轴模型]]"
related:
  - "[[ETS创新残差]]"
  - "[[ARMA信息准则]]"
  - "[[滚动起点评估]]"
part_of:
  - "[[指数平滑与 ETS.canvas]]"
---

# ETS 似然把平滑参数与初始状态共同作为估计对象
<!-- bilingual-en:start -->
*ETS likelihood treats smoothing parameters and initial states as joint estimation objects*
<!-- bilingual-en:end -->

> [!summary] 原子原则
> 现代 ETS 拟合通常不是先随手固定 $\ell_0,b_0,s_0,\ldots$ 再只估计 $\alpha,\beta,\gamma,\phi$；而是在误差模型给定后，用 likelihood 联合估计平滑参数与受约束的初始状态。信息准则的复杂度计数必须反映这些自由量。
> <!-- bilingual-en:start -->
> Modern ETS estimation usually optimises smoothing parameters and constrained initial states jointly under the chosen error model. Information-criterion parameter counts must include the freely estimated states as well as model parameters.
> <!-- bilingual-en:end -->

对季节 ETS，估计对象可包含
$$
(\alpha,\beta,\gamma,\phi;\ \ell_0,b_0,s_0,s_{-1},\ldots,s_{-m+1}).
$$
季节初值还受归一化限制：加法季节指标和约为零，乘法季节指标和约为 $m$。因此写出 $m$ 个数不等于有 $m$ 个独立自由参数；确切计数要服从模型约束与软件的 likelihood 定义。

在 Gaussian additive-error ETS 中，最大化条件 likelihood 与最小化一步平方误差可得到相同参数；multiplicative-error likelihood 还含随状态变化的尺度项，通常不再等价于只最小化原尺度 SSE。初始化方案、参数约束与优化目标不同，软件即使都标为 “Holt–Winters” 也可能给出不同拟合。

ETS 的 AICc 只有在候选使用同一响应变量、同一有效样本、同一变换及 Jacobian 口径，并从可比的 observed-data likelihood 计算时才可排序，而且 $k$ 应计入自由初始状态与创新方差。AIC 本身并不禁止比较非嵌套模型族；真正的门槛是 likelihood 是否针对同一数据问题、是否保留相同常数与初始化口径。FPP3 特别提醒，其 ETS 与 ARIMA 实现采用不同 likelihood 计算，因此软件报告的两族 AICc 不能直接决胜。跨族选择应回到 [[滚动起点评估|rolling-origin 预测证据]]。

传统方法参数化常令 $\alpha,\beta^*,\gamma^*,\phi$ 位于 $(0,1)$；在 innovations 参数化中，$\beta=\alpha\beta^*$、$\gamma=(1-\alpha)\gamma^*$，所以传统限制相应变成 $0<\beta<\alpha$ 与 $0<\gamma<1-\alpha$。状态空间 admissibility region 又可能更宽；具体边界依模型与实现。不能看到软件给出超出传统区间的参数就自动判错，也不能忽略它使用了哪套 admissibility 与数值约束。
<!-- bilingual-en:start -->
Seasonal initial states obey normalisations, so their displayed count is not automatically their number of free parameters. Gaussian additive-error likelihood coincides with SSE minimisation under the corresponding setup, whereas multiplicative-error likelihood generally does not. AICc comparisons require the same response, effective sample, transformation/Jacobian, retained likelihood constants, and initialisation convention. AIC does not forbid non-nested families in principle, but the ETS and ARIMA likelihoods reported by the FPP3 implementations are not directly comparable; cross-family selection should therefore use chronological forecast evaluation.
<!-- bilingual-en:end -->

> [!question]- 自检
> 两个 ETS 候选使用不同长度的有效样本，软件各自报告 AICc。能否直接选较小者？
>
> **答案：** 不能。样本与 likelihood 问题不相同，数值没有共同比较基准；先统一目标、样本、变换与估计口径。

## 来源与核验

- [Hyndman & Athanasopoulos, FPP3 §8.6](https://otexts.com/fpp3/ets-estimation.html)：核对 likelihood 联合估计平滑参数/初始状态、AICc 公式与参数计数、传统与 admissible 约束。
- [Hyndman et al. (2002), §4](https://www.monash.edu/business/ebs/research/publications/ebs/a_state_space_framework_for_automatic_forecasting_using_exponential_smoothing_methods.pdf)：核对条件 likelihood、初始状态约束与自动选模框架。
- [[ARMA信息准则]]：复用共同样本、likelihood 与信息准则解释的通用边界。

---
aliases:
  - "波动过滤EVT对标准化残差拟合尾部再按预测条件位置与尺度重标度"
  - Volatility-filtered EVT
  - Conditional EVT
student_os: knowledge-atom
atom_id: RM-EVT-016
atom_type: method
status: source-checked
part_of:
  - "[[极值理论 EVT 与尾部风险.canvas]]"
requires:
  - "[[条件尺度与标准化冲击]]"
  - "[[POT分位数]]"
  - "[[POT尾部ES]]"
related:
  - "[[过滤历史模拟]]"
  - "[[GARCH残差双层诊断]]"
  - "[[极端聚集处理]]"
---

# 波动过滤EVT对标准化残差拟合尾部再按预测条件位置与尺度重标度
<!-- bilingual-en:start -->
*Volatility-filtered EVT fits a tail model to standardized residuals and rescales it using the forecast conditional location and scale.*
<!-- bilingual-en:end -->

波动过滤 EVT 是估计条件尾部风险的两阶段方法：先用时间序列模型解释损失的条件均值与波动，再把 POT 用于标准化残差的高损失一端。它复用[[过滤历史模拟]]的标准化思想，但残差尾部使用参数化 GPD 外推，而不是只重抽历史残差。
<!-- bilingual-en:start -->
Volatility-filtered EVT estimates conditional tail risk in two stages: a time-series model describes the conditional mean and volatility of losses, then POT models the high-loss tail of standardized residuals. It shares the standardization idea of [[过滤历史模拟|filtered historical simulation]], but extrapolates the residual tail with a parametric GPD instead of only resampling historical residuals.
<!-- bilingual-en:end -->

令 $\mathcal F_t$ 是截至 $t$ 的信息。在模型
<!-- bilingual-en:start -->
Let $\mathcal F_t$ contain information available through time $t$. In the model
<!-- bilingual-en:end -->

$$
L_{t+1}=\mu_{t+1\mid t}+\sigma_{t+1\mid t}Z_{t+1},
\qquad \sigma_{t+1\mid t}>0,
$$

位置和尺度须由过去信息确定；并假设 $Z_{t+1}$ 独立于 $\mathcal F_t$，具有跨时稳定的分布。若称 $\sigma$ 为条件标准差，还要求 $E[Z]=0,\operatorname{Var}(Z)=1$。由历史拟合得到 $\hat z_i=(L_i-\hat\mu_i)/\hat\sigma_i$，先检查[[GARCH残差双层诊断|残差及其尺度动态]]和尾部聚集，再拟合残差 POT。采用收益数据时，要先统一正损失方向，不能把大额正收益尾当成损失尾。
<!-- bilingual-en:start -->
Location and scale must depend only on past information. Assume that $Z_{t+1}$ is independent of $\mathcal F_t$ and has a time-stable distribution. Calling $\sigma$ a conditional standard deviation additionally requires $E[Z]=0$ and $\operatorname{Var}(Z)=1$. Construct fitted historical residuals $\hat z_i=(L_i-\hat\mu_i)/\hat\sigma_i$, check [[GARCH残差双层诊断|residual and scale dynamics]] and tail clustering, then fit their POT tail. With return data, establish the positive-loss direction first; the large-gain tail is not the loss tail.
<!-- bilingual-en:end -->

若残差的目标分位和有限 ES 分别为 $q_\alpha^Z$、$ES_\alpha^Z$，则一步条件风险为
<!-- bilingual-en:start -->
If the residual model supplies target quantile $q_\alpha^Z$ and finite $ES_\alpha^Z$, one-step conditional risk is
<!-- bilingual-en:end -->

$$
q_{\alpha,t+1\mid t}^{L}=\mu_{t+1\mid t}+\sigma_{t+1\mid t}q_\alpha^Z,
\qquad
ES_{\alpha,t+1\mid t}^{L}=\mu_{t+1\mid t}+\sigma_{t+1\mid t}ES_\alpha^Z.
$$

这是条件在已知位置和正尺度下的仿射变换，实际用估计与预测值代入。例如残差模型给 $q_\alpha^Z=3,ES_\alpha^Z=4$，且下一期位置为 $0.1$ 万元、尺度为 $2$ 万元，则相应条件 VaR 为 $6.1$ 万元、ES 为 $8.1$ 万元；残差风险数本身没有损失金额单位。
<!-- bilingual-en:start -->
These are affine transformations conditional on the known location and positive scale; implementation substitutes fitted and forecast values. If the residual model gives $q_\alpha^Z=3$ and $ES_\alpha^Z=4$, with next-period location $0.1$ and scale $2$ in CNY 10,000, conditional VaR is $6.1$ and ES is $8.1$ in those units. Residual risk numbers themselves are dimensionless.
<!-- bilingual-en:end -->

过滤不自动得到 iid 残差，也不消除断点、尾部误设或预测尺度误差。若把标准化残差的 GPD 当作精确延伸到端点的尾模型，单位有限方差与[[广义Pareto矩存在条件]]必须相容；正尾形状 $\xi\ge1/2$ 就与该精确有限方差设定冲突。多日风险还涉及未来尺度沿路径变化，不能把这条一步公式直接按平方根时间缩放。
<!-- bilingual-en:start -->
Filtering does not automatically produce iid residuals or eliminate breaks, tail misspecification, or scale-forecast error. If the residual GPD is treated as an exact tail extending to its endpoint, unit finite variance must be compatible with [[广义Pareto矩存在条件|the GPD moment conditions]]; upper-tail shape $\xi\ge1/2$ conflicts with that exact finite-variance specification. Multi-day risk involves evolving future scales, so the one-step formula does not justify square-root-of-time scaling.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [McNeil–Frey，作者版 PDF 第 4–8 页，式 (1)–(3)](https://statmath.wu.ac.at/~frey/publications/evt-garch.pdf#page=4)：支持过去可测位置/尺度、iid 单位方差创新、标准化残差与一步 VaR/ES 重标度。已读取第 4–8 页并目视第 5 页的两条重标度公式；数字例为独立代入。
  <!-- bilingual-en:start -->
  Author-version pp. 4–8, equations (1)–(3), support past-measurable location and scale, iid unit-variance innovations, standardized residuals, and one-step VaR/ES rescaling. The formulas on p. 5 were also visually checked; the example is independently evaluated.
  <!-- bilingual-en:end -->
- [同文 PDF 第 9、17–18 页](https://statmath.wu.ac.at/~frey/publications/evt-garch.pdf#page=17)：支持原序列依赖与过滤区别，以及多日模拟相对于平方根时间缩放的边界；有限方差相容性由其单位方差假设和 GPD 矩条件联合核验。
  <!-- bilingual-en:start -->
  These pages support the distinction between raw-series dependence and filtering, and the multi-day scaling limitation. Finite-variance compatibility is checked by combining the model's unit-variance assumption with the GPD moment conditions.
  <!-- bilingual-en:end -->

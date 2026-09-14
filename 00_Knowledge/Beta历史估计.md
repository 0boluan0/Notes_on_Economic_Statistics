---
aliases:
  - 历史 Beta 是样本回归估计而不是资产的永久常数
  - Historical beta estimation
  - Beta regression boundary
student_os: knowledge-atom
atom_id: INV-CAPM-007
atom_set: capm-systematic-risk
atom_type: estimation-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[Beta与总波动]]"
  - "[[市场代理联合检验]]"
part_of:
  - "[[CAPM、系统风险与资本成本.canvas]]"
leads_to:
  - "[[权益Beta杠杆调整]]"
---

# 历史 Beta 是样本回归估计而不是资产的永久常数
<!-- bilingual-en:start -->
*Historical beta is a sample regression estimate, not a permanent asset constant*
<!-- bilingual-en:end -->

> [!summary] 原子估计
> 常见历史 beta 来自时间序列回归
> $$R_{i,t}-R_{f,t}=\alpha_i+\beta_i\bigl(R_{m,t}-R_{f,t}\bigr)+\varepsilon_{i,t}.$$
> OLS 斜率估计的是指定市场代理、观测频率和样本窗口下的线性协方差关系。它带有抽样误差，而且只描述过去这段数据；业务、经营杠杆、融资结构或市场制度变化后，未来 beta 可以改变。
> <!-- bilingual-en:start -->
> Historical beta is commonly estimated from $R_{i,t}-R_{f,t}=\alpha_i+\beta_i(R_{m,t}-R_{f,t})+\varepsilon_{i,t}$. The OLS slope estimates a linear covariance relation for a specified market proxy, return frequency, and sample window. It has sampling error and describes that historical sample only; changes in business mix, operating leverage, financing, or market structure can change future beta.
> <!-- bilingual-en:end -->

估计前要统一复权收益、币种、无风险收益周期和交易日。交易不活跃时，资产价格可能在市场变化之后才更新，日频同步回归会把共同变化错开，常把 beta 向零拉；简单增加小数位或报告较小标准误并不能修复这个数据生成问题。
<!-- bilingual-en:start -->
Returns must use consistent corporate-action adjustments, currency, risk-free intervals, and trading dates. With infrequent trading, an asset price may update after the market moves, so a synchronous daily regression misaligns common variation and often biases beta toward zero. More decimal places or a small conventional standard error do not repair that data-generating problem.
<!-- bilingual-en:end -->

实务估计应同时报告样本期、频率、代理指数、点估计和不确定性，并用滚动窗口或结构分段检查稳定性。若目标是未来项目资本成本，还应与行业可比公司的去杠杆 beta、业务变化和目标资本结构交叉核对；历史回归不是自动的预测器。
<!-- bilingual-en:start -->
Practical estimates should report the window, frequency, proxy, point estimate, and uncertainty, with rolling or regime checks for stability. For a future project's cost of capital, cross-check against unlevered comparable-company betas, business changes, and target capital structure. A historical regression is not an automatic forecast.
<!-- bilingual-en:end -->

> [!question]- 自检
> 某股票五年周频 beta 的标准误很小，为什么仍不能说未来十年 beta 已经确定？
>
> **答案：** 标准误只在给定回归与样本稳定假设下描述估计噪声；它不覆盖未来业务、杠杆、市场代理和制度环境改变造成的参数变化。

## 来源与核验

- [Jensen (1968), “The Performance of Mutual Funds”](https://doi.org/10.1111/j.1540-6261.1968.tb00815.x)：核对超额收益回归、beta 估计与抽样显著性。
- [Scholes & Williams (1977), “Estimating Betas from Nonsynchronous Data”](https://doi.org/10.1016/0304-405X(77)90041-1)：核对非同步交易造成的 beta 偏误及其不是普通精度问题。
- [[02_Economy/06_证券投资学/11_风险资产的定价.md#3. 如何计算 $\beta$|课程 beta 部分]]：核对课程回归口径；本卡补足代理、频率、窗口和结构变化边界。

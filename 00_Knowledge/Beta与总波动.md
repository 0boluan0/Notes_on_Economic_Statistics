---
aliases:
  - Beta 衡量资产相对市场组合的协方差暴露而不是总波动
  - Beta versus total volatility
  - Market beta
student_os: knowledge-atom
atom_id: INV-CAPM-004
atom_set: capm-systematic-risk
atom_type: definition-distinction
status: source-checked
mastery_state: unassessed
requires:
  - "[[特有风险定价边界]]"
part_of:
  - "[[CAPM、系统风险与资本成本.canvas]]"
leads_to:
  - "[[证券市场线]]"
  - "[[Beta历史估计]]"
---

# Beta 衡量资产相对市场组合的协方差暴露而不是总波动
<!-- bilingual-en:start -->
*Beta measures an asset's covariance exposure to the market portfolio, not its total volatility*
<!-- bilingual-en:end -->

> [!summary] 原子定义
> 在单期无风险收益于期初给定且 $\operatorname{Var}(R_m)>0$ 时，资产相对市场的 beta 为
> $$\beta_i=\frac{\operatorname{Cov}(R_i,R_m)}{\operatorname{Var}(R_m)}.$$
> 它是资产超额收益对市场超额收益最佳线性投影的斜率：市场超额收益变化一个单位时，线性拟合值平均变化多少。除非另外假定条件期望确为线性，不能把这个投影斜率直接说成完整的条件收益关系。它也不是“资产会跌多少”的概率，更不是资产自身标准差。
> <!-- bilingual-en:start -->
> With the one-period risk-free return fixed at the start of the period and $\operatorname{Var}(R_m)>0$, an asset's market beta is $\beta_i=\operatorname{Cov}(R_i,R_m)/\operatorname{Var}(R_m)$. It is the slope in the best linear projection of asset excess return on market excess return: the fitted linear value changes by beta for a one-unit change in market excess return. Without the additional assumption that the conditional mean itself is linear, this projection slope is not the whole conditional-return relation. It is neither a loss probability nor the asset's own standard deviation.
> <!-- bilingual-en:end -->

在线性市场模型 $R_i-R_f=\alpha_i+\beta_i(R_m-R_f)+\varepsilon_i$ 中，若残差与市场正交，则
$$
\operatorname{Var}(R_i)=\beta_i^2\operatorname{Var}(R_m)+\operatorname{Var}(\varepsilon_i).
$$
第一项是该模型中的市场共同波动，第二项是残余波动。因此低 beta 与高总波动可以同时成立；高 beta 也不保证每个市场下跌日都下跌更多。
<!-- bilingual-en:start -->
In the linear market model, if the residual is orthogonal to the market, total variance separates into $\beta_i^2\operatorname{Var}(R_m)+\operatorname{Var}(\varepsilon_i)$. The first term is market-related variation in that model and the second is residual variation. Low beta can therefore coexist with high total volatility, and high beta does not force a larger loss on every market-down day.
<!-- bilingual-en:end -->

组合 beta 具有线性聚合性质：若组合收益 $R_p=\sum_iw_iR_i$，使用同一个市场基准与同一时期口径，则 $\beta_p=\sum_iw_i\beta_i$。这让 beta 可用于组合风险归因，但前提是各估计的市场代理、频率和样本口径相容。
<!-- bilingual-en:start -->
Portfolio beta aggregates linearly: if $R_p=\sum_iw_iR_i$, then $\beta_p=\sum_iw_i\beta_i$ under the same market benchmark and period convention. This supports portfolio attribution only when the component estimates use compatible proxies, frequencies, and samples.
<!-- bilingual-en:end -->

> [!question]- 自检
> 股票 A 的波动率为 50%、beta 为 0.6；股票 B 的波动率为 25%、beta 为 1.3。哪只股票的 CAPM 系统风险更高？
>
> **答案：** B 的 beta 更高，因此按 CAPM 它对市场组合的协方差暴露更大；A 的高总波动可能主要来自残余风险。

## 来源与核验

- [Sharpe (1964), “Capital Asset Prices”](https://doi.org/10.1111/j.1540-6261.1964.tb02865.x)：核对资产对市场组合风险的边际贡献与均衡风险价格。
- [Jensen (1968), “The Performance of Mutual Funds”](https://doi.org/10.1111/j.1540-6261.1968.tb00815.x)：核对市场模型回归、beta 风险参数与残差口径。
- [[02_Economy/06_证券投资学/11_风险资产的定价.md#3. 如何计算 $\beta$|课程 beta 计算部分]]：核对课程公式；本卡把“市场变动 1%”限定为线性预测而非逐期机械变动。

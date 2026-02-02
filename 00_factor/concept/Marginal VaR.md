---
aliases:
- 边际VaR
tags:
- concept
---


边际VaR定义为组合VaR对某资产头寸的变化率，直观上是**组合VaR对单个资产头寸的偏导数**。边际VaR表示在当前组合中，若第$i$项资产持仓增加一微小单位，组合VaR增加多少。公式上，资产$i$的边际VaR = $\partial \text{VaR}_{\text{组合}} / \partial w_i$（$w_i$为资产权重或金额）。边际VaR反映了每增加一元资产$i$所带来的风险增量。

  边际VaR与资产在组合中的**系统性风险贡献**有关。对于高度分散的组合，在正态参数法框架下，可以证明资产$i$的边际VaR与其在组合中的Beta系数成正比——Beta越高（与组合高度正相关，波动贡献大），边际VaR越大；反之，如果某资产与组合低相关甚至负相关，增加它反而可能降低组合风险，此时边际VaR可以为负值（意味着增持该资产会降低总VaR）。因此，边际VaR提供了调整组合的指引：增加边际VaR小甚至为负的资产有助于降低整体风险。

## 相关链接

- 基础风险度量：[[VaR]]
- VaR分解：[[00_factor/concept/Incremental VaR|递增VaR]], [[00_factor/concept/Component VaR|成分VaR]]
- 应用：用于[[00_factor/concept/Component VaR|成分VaR]]的计算（成分VaR = 持仓 × 边际VaR）
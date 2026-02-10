---
aliases:
- 成分VaR
- Component VaR
tags:
- concept
---
成分VaR又称风险贡献度，是将组合总VaR划分到各组成资产的一种分摊，使各部分成分VaR之和等于整体VaR。成分VaR旨在回答：“组合总 VaR 中有多少是由资产 $i$ 贡献的？”根据风险分配理论，若风险度量满足正齐次性，可应用**欧拉分解法（Euler’s Theorem）**：对于组合风险 $V$，如果对任意 $\lambda>0$ 有 $V(\lambda \mathbf{x}) = \lambda V(\mathbf{x})$（线性齐次），则
$$
V(\mathbf{x}) = \sum_{i=1}^{N} x_i \frac{\partial V}{\partial x_i}(\mathbf{x}).
$$
其中 $x_i \frac{\partial V}{\partial x_i}$ 可以解释为第 $i$ 项的风险成分。套用于 VaR，若将组合各资产头寸 $w_i$ 均放大 $\lambda$ 倍，VaR 也放大 $\lambda$ 倍（VaR 的一阶齐次性在正态等模型下成立），则有：
$$
\text{VaR}_{\text{组合}} = \sum_{i} w_i \frac{\partial \text{VaR}}{\partial w_i}.
$$
右侧每一项正是资产 $i$ 的持仓规模乘以其边际 VaR，定义为资产 $i$ 的**成分 VaR**。这一定义下，各资产成分 VaR 之和正好等于组合 VaR，实现了 VaR 的可加分解。

  **性质：**成分VaR具有如下特性：(1) 对大型组合，成分VaR与单一资产被视作“小幅变化”时的递增VaR非常接近，即成分VaR近似等于将该资产从组合中移除一点点的VaR差异；(2) 将所有资产的成分VaR相加，正好得到组合总VaR（由欧拉定理保证）。

应用方面，成分VaR被广泛用于**风险归因和风险预算**。例如，若某资产成分VaR占比很高，说明它对整体风险贡献突出，风险管理者可能考虑减仓该资产以降低总体VaR。反之，成分VaR占比低且收益尚可的资产有增加配置的空间。边际VaR/成分VaR还是优化投资组合时的重要依据，通过调节持仓使各资产的收益与其风险贡献相匹配，可以实现风险调整后的收益最大化。

## 相关链接

- 基础风险度量：[[VaR]]
- VaR分解：[[Marginal VaR|边际VaR]], [[Incremental VaR|递增VaR]]

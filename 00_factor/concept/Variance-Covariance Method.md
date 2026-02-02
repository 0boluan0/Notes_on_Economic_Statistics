---
aliases:
- 参数法
- 方差-协方差法
- Parametric Method
- Variance
- Variance-Covariance Method
tags:
- 风险管理
- VaR计算
- concept
---
方差-协方差法是计算VaR的参数方法，假定资产或投资组合收益服从某种已知分布（通常假设正态分布），利用收益的期望和方差-协方差等参数直接计算VaR。

## 基本假设

假设资产收益服从正态分布（或其他已知分布），利用参数化方法计算VaR。

## 计算公式

在正态分布假设下，若已知组合在单位时间的预期收益μ和标准差σ，则α置信水平下的一天VaR为：

$$\text{VaR}_{\alpha,1\text{天}} = -(\mu + z_{\alpha}\sigma)$$

其中z_α是标准正态分布的α分位点：
- α = 99%时，z_0.99 ≈ 2.33
- α = 95%时，z_0.95 ≈ 1.645

当假设μ ≈ 0时，VaR近似为：$\text{VaR}_{\alpha} \approx z_{\alpha}\sigma$

## 持有期扩展

若持有期为N天且每天损失独立，则N天VaR可按平方根规则调整：

$$\text{VaR}_{\alpha,N\text{天}} \approx z_{\alpha}\sigma\sqrt{N}$$

## 组合VaR计算

对于多资产组合，利用协方差矩阵计算组合方差：

$$\sigma_p^2 = \sum_{i,j} w_i w_j \sigma_{ij} = \mathbf{w}^T \Sigma \mathbf{w}$$

其中：
- w_i：资产i的权重
- σ_ij：资产i和j的协方差
- Σ：协方差矩阵

## 优点

- **计算快速简便**：只需估计期望、方差等参数即可算出VaR
- **容易理解和实现**：基于成熟的统计模型有解析解
- **适合线性投资组合**：对于仅包含线性金融产品的组合，计算结果明确可靠

## 缺点

- **分布假设局限**：通常假设收益正态分布，实际金融资产可能呈现厚尾、偏态
- **非线性风险低估**：对包含期权等非线性衍生品的组合，需近似处理（如Delta-Gamma方法）
- **相关性假设简化**：通常用历史协方差矩阵，假定相关关系稳定

## 改进方法

1. **Delta-Gamma方法**：考虑期权价格对标的资产的一阶和二阶导数
2. **Cornish-Fisher展开**：考虑分布的偏度和峰度
3. **使用其他分布假设**：如t分布、混合正态分布等

相关链接: [[VaR]], [[00_factor/concept/Historical Simulation Method|历史模拟法]], [[00_factor/concept/Monte Carlo Simulation Method|蒙特卡罗模拟方法]], [[Delta]], [[Gamma]]

## source_notes

- [[12_VAR风险#2.1 方差-协方差法（正态分布假设法）]]

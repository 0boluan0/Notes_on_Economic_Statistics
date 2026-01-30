---
aliases:
- Macaulay Duration
- 马考利久期
tags:
- concept
---
马考利久期（Macaulay Duration）定义为**现金流现值加权的平均到期时间**：

$$

D_M = \frac{ \sum_{t=1}^{n} t \cdot \frac{CF_t}{(1+y)^t} }{ \sum_{t=1}^{n} \frac{CF_t}{(1+y)^t} }

$$

- $CF_t$：第$t$期的现金流（付息/还本）
- $y$：到期收益率（即市场利率）
- $n$：期数

## 相关链接

- 一般久期：[[久期]]
- 修正久期：[[修正久期]] = 马考利久期 / (1 + y/m)
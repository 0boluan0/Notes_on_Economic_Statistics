---
aliases:
- Gamma
tags:
- concept
---
>[!note] **一、定义**
>
> **Gamma**（$\Gamma$）是期权定价理论中的一个希腊字母参数，是衡量期权头寸**Delta变动速度**的指标。
>
> - $\Delta$ 衡量的是期权价格对标的资产价格变化的敏感度；    
> - $\Gamma$ 衡量的是**Delta自身对标的资产价格变化的敏感度**。
>
> 简而言之，$\Gamma$ 是**二阶敏感性**。
>
> 当 Gamma 的绝对值很小时， Delta 变化缓 慢，这时为保证 Delta 中性所做的交易调整并不需 要太频繁;但是当 Gamma 的绝对值很大时，交易 组合的 Delta 对于基础资产的价格就变得很敏感， 此时 在任意一段时间内，对一个 Delta 中性的交 易组合不做调整会非常危险
>
> ---
>
# **二、公式定义**

假设$S$为标的资产价格，$C$为看涨期权价格，则：

- **Delta：**
    $$
\Delta = \frac{\partial C}{\partial S}
    $$
- **Gamma：**
    $$
\Gamma = \frac{\partial^2 C}{\partial S^2} = \frac{\partial \Delta}{\partial S}
    $$

即$\Gamma$是$\Delta$对$S$的导数。

# **三、经济意义**

- $\Gamma$描述了**Delta对标的价格变化的敏感度**。
- $\Gamma$大，说明$S$变动时，$\Delta$会迅速变化（风险敞口波动大）。
- $\Gamma$小，说明$S$变动时，$\Delta$变化不大（风险较平滑）。

# **四、举例说明**

假设你持有一个看涨期权：

- $当前\Delta = 0.5，\Gamma = 0.1$
- 若标的价格$S$上涨$1$元，则$\Delta$将增加$0.1$，变为$0.6$
- 若$S$再上涨$1$元，$\Delta$又增加$0.1$，变为$0.7$

所以，**Gamma反映了Delta随标的价格的变化速度**。

# **五、Gamma中性的含义**

- **Gamma中性**：$通过期权和现货等多种组合，使得\Gamma=0，即对标的资产价格的二阶变动也不敏感。$
- 这种策略可以让组合在标的资产出现大幅波动时也能较好地对冲风险。

## 相关链接

- 其他希腊字母：[[Delta]], [[Vega]], [[Theta]], [[Rho]]
- 应用：Gamma中性策略

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM ""
WHERE (
  contains(file.path, "01_Math/") OR
  contains(file.path, "02_Economy/") OR
  contains(file.path, "03_Computer_Science/")
) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```

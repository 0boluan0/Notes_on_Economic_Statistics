---
aliases:
- 广义自回归条件异方差
- GARCH
tags:
- concept
---
ARCH的N要取得比较大.所以发明了GARCH模型,在保持对条件异方差性建模能力的同时，用更少的参数捕捉长期的波动影响。

> [!note] GARCH(p,q)模型
> $$
> \begin{cases}
> \varepsilon_t = \nu_t \sqrt{h_t} \\
> h_t = \alpha_0 + \sum_{i=1}^{q} \alpha_i \varepsilon_{t-i}^2 + \sum_{j=1}^{p} \beta_j h_{t-j}
> \end{cases}
> $$
> **其中** $h_t$ 依赖于 $q$ 阶误差平方和 $p$ 阶自身滞后。要求参数满足 $\alpha_0 > 0$，$\alpha_i \ge 0$（$i=1,\ldots,q$），$\beta_j \ge 0$（$j=1,\ldots,p$），且保证平稳性的约束 $\sum_{i=1}^q \alpha_i + \sum_{j=1}^p \beta_j < 1$。

GARCH模型通常能够用更少的滞后项达到与高阶ARCH模型同样的效果 。因此，GARCH模型在刻画长记忆的波动性时更**参数节省**（parsimonious）。

条件方差：$\mathbb{E}(\varepsilon_t^2 \mid \mathcal{F}_{t-1}) = h_t = \alpha_0 + \sum_{i=1}^q \alpha_i \varepsilon_{t-i}^2 + \sum_{j=1}^p \beta_j h_{t-j}$。
如果过程是平稳的，即满足 $\sum_{i=1}^q \alpha_i + \sum_{j=1}^p \beta_j < 1$，那么整个序列的无条件二阶矩（长期平均方差）存在，记作：$\mathbb{E}(\varepsilon_t^2) = \frac{\alpha_0}{1 - \sum_{i=1}^q \alpha_i - \sum_{j=1}^p \beta_j}$。

实证分析中最常用的就是GARCH(1,1)

## 相关链接

- 基础模型：[[ARCH]]
- 扩展模型：[[TARCH]], [[EGARCH]]
- 相关概念：[[EWMA]], [[Volatility Clustering|波动聚集]], [[Conditional Heteroskedasticity|条件异方差]]

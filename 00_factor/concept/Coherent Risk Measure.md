---
aliases:
- 一致性风险度量
- Coherent
- Coherent Risk Measure
tags:
- 金融
- concept
---
- 风险度量就是用数学函数$\varrho(\cdot)$，把资产组合（或损失、收益的分布）映射为一个**衡量风险的数值**。
    
- Artzner等人提出：**“合理”的风险度量模型，必须同时满足以下四个公理**，也就是“一致性”要求。

1. **单调性（Monotonicity）**
    如果$X \leq Y$，那么$\varrho(X) \geq \varrho(Y)$。
    —— 损失越大，风险度量值也越大。
    
2. **次可加性（Subadditivity）**
    对任意$X, Y$，$\varrho(X+Y) \leq \varrho(X) + \varrho(Y)$。
    —— 组合风险不会超过各自单独风险之和，体现分散投资有益。
    
3. **正齐次性（Positive Homogeneity）**
$对任意h>0，\varrho(hX) = h\varrho(X)。$
    —— 投资规模变$h$倍，风险也变$h$倍。
    
4. **平移不变性（Translational Invariance）**
$对任意常数K，\varrho(X+K) = \varrho(X) - K。$
    —— 账户里直接加/减现金，只会减少/增加风险值$K$，风险结构本质不变。

VaR不是一致度量标准.标准差和方差也不是.

>[!example] VaR不满足次可加性的例子
>![[Pasted image 20250627161319.png]]

CVaR(ES)是一致度量标准

## 相关链接

- 具体风险度量：[[VaR]], [[ES]]
- 性质：VaR不满足次可加性，不是一致度量标准；ES满足全部四个公理，是一致度量标准
---
aliases:
- 隐含波动率
- Implied Volatility
tags:
- concept
---
**概念：**隐含波动率（Implied Volatility, IV）是从期权等衍生品的市场价格中反推出的波动率。因为在期权定价模型（例如Black–Scholes公式）中，期权价格是标的资产价格$S$、执行价$K$、无风险利率$r$、到期时间$T$和波动率$\sigma$等因素的函数，其中只有波动率无法直接从市场观察到。隐含波动率就是指使定价模型的理论价格等于市场实际期权价格所对应的波动率值。

**求解方法：**根据期权的实际市场价格，利用定价公式反推波动率：
- 将已知的标的资产现价$S$、行权价$K$、剩余到期时间$T$、无风险利率$r$等参数代入期权定价公式（例如Black–Scholes公式），使模型价格等于市场期权价格。由于定价公式中波动率$\sigma$是未知的，我们通过数值方法调整$\sigma$，直到模型价格与市场价格一致。此时对应的$\sigma$即为该期权的隐含波动率。
- 由于期权价格与波动率之间通常是单调关系（波动率高则期权价格高），因此对应每个期权的市场价格，存在唯一的隐含波动率解。

**隐含波动率的意义：**隐含波动率反映了市场参与者对未来波动性的预期程度。不同执行价、不同到期日的期权往往有不同的隐含波动率，从而形成**波动率微笑（微笑曲线）**或**波动率期限结构**等特征。市场上著名的波动率指数VIX就是基于标普500指数期权隐含波动率计算得到的指数，被称为市场恐慌指数。

## 相关链接

- 其他波动率度量：[[Historical Volatility|历史波动率]], [[Realized Volatility|已实现波动率]]
- 相关概念：波动率微笑，VIX指数

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

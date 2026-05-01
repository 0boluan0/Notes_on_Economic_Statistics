---
aliases:
- Implied Volatility
- IV
- implied vol
- 隐含波动率
tags:
- concept
- 金融风险
- 期权
- 波动建模
---
# Implied Volatility

## 先记一句话

隐含波动率就是：**把期权市场价格代回定价模型，反推出市场正在定价的未来波动率**。

它是 forward-looking volatility。

## 它是什么

期权价格由多个变量决定：

- 标的价格 $S$；
- 行权价 $K$；
- 到期时间 $T$；
- 无风险利率 $r$；
- 波动率 $\sigma$。

市场价格已知时，反求让模型价格等于市场价格的 $\sigma$，就是 implied volatility。

## 它解决什么判断

隐含波动率回答：

> 市场现在愿意为未来不确定性支付多少价格？

它常用于：

- 比较市场预期和历史波动；
- 构造 volatility smile / term structure；
- 作为期权风险管理和 VaR 的输入。

## 常见误区

- IV 不是未来真实波动率的保证，而是市场价格隐含出来的波动率。
- 不同行权价和期限通常有不同 IV。
- IV 依赖定价模型；模型错，反推出的 IV 也只是该模型下的翻译。

## 来自课程位置

- [[10_波动率|金融风险管理 10：隐含波动率]]
- [[08_操作员如何管理风险暴露|金融风险管理 08：期权风险暴露]]

## 关联卡片

- [[Historical Volatility]]
- [[Realized Volatility]]
- [[Vega]]
- [[Delta-Gamma Approximation]]
- [[Option Greeks-hub]]

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

---
aliases:
- Yield to Maturity
- YTM
- 到期收益率
tags:
- concept
- finance
---
# Yield to Maturity

## 一句话记忆

YTM 是让债券未来现金流现值等于当前价格的折现率。

## 它是什么

Yield to Maturity 是满足下式的 $r$：

$$
P=\sum_{t=1}^{T}\frac{C_t}{(1+r)^t}+\frac{F}{(1+r)^T}
$$

它是给定债券价格、票息和本金后反推出的承诺收益率。

## 解决什么判断

- 当前债券价格隐含的收益率是多少。
- 债券价格与市场利率为什么反向变动。
- 不同债券能否用同一收益率口径比较。

## 最小例子

若一张一年期债券价格为 98，到期支付 100 且无票息：

$$
98=\frac{100}{1+YTM}
$$

所以 $YTM=2.04\%$。

## 易混点

- YTM 不是票面利率；票面利率决定票息，YTM 由市场价格反推。
- YTM 假设持有至到期且现金流按同样收益率再投资。
- 持有期回报会受卖出价格影响，不一定等于买入时的 YTM。
- 长期债券价格对利率更敏感，这一部分要接 [[duration|Duration]]。

## 来自课程位置

- [[06_债券和股票估价|03_债券与股票估值]]
- [[06_债券和股票估价#三.债券的收益率|02_债券定价]]

## 关联卡片

- [[Bond Valuation Model]]
- [[Present Value]]
- [[duration|Duration]]
- [[Modified Duration]]
- [[Interest Rate Risk Management-hub]]

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

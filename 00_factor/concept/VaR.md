---
aliases:
- VaR
- Value at Risk
- 风险价值
- 在险价值
tags:
- concept
- 金融风险
- VaR
---
# VaR

## 先记一句话

VaR 就是：**在给定置信水平和持有期下，损失通常不会超过的分位数金额**。

它回答：

> 未来一段时间，在正常市场条件下，最坏的那一小部分之前会亏多少？

## 它是什么

若损失变量为 $L$，置信水平为 $\alpha$，则
$$
VaR_\alpha
$$
是损失分布的 $\alpha$ 分位数：
$$
P(L\leq VaR_\alpha)=\alpha.
$$

例如 1 日 99% VaR = 200 万，意思是：

> 在模型假设下，未来 1 天损失超过 200 万的概率约为 1%。

## 它解决什么判断

VaR 用来把不同头寸、市场变量和组合风险压缩成一个可报告的损失金额。

它常用于：

- 日常风险限额；
- 市场风险资本；
- 投资组合风险比较；
- 回测模型是否低估风险。

## 三个必须一起说的参数

说 VaR 时不能只说金额，必须带：

- 置信水平：95%、99%、99.9%；
- 持有期：1 天、10 天、1 年；
- 损失口径：绝对损失、相对基准、还是 P&L。

## 它的边界

VaR 不告诉你超过 VaR 后会亏多惨。

所以在尾部风险上，[[ES]] 更完整。

VaR 也可能不满足次可加性，因此不是最稳健的一致性风险度量。

## 常见误区

- VaR 不是最大可能损失；它只是分位数。
- 99% VaR 不代表 99% 不亏钱，而是 99% 情况下损失不超过该数。
- VaR 的结果强依赖分布假设、历史窗口、波动率和相关性。
- 危机时相关性和波动率会变，平时估出来的 VaR 可能失效。

## 来自课程位置

- [[12_VAR风险|金融风险管理 12：VaR 风险]]
- [[14_VaR参数法和模拟法|金融风险管理 14：VaR 参数法和模拟法]]
- [[22_情景分析和压力测试|金融风险管理 22：VaR 与压力测试]]

## 关联卡片

- [[ES]]
- [[Variance-Covariance Method]]
- [[Historical Simulation Method]]
- [[Monte Carlo Simulation Method]]
- [[Marginal VaR]]
- [[Incremental VaR]]
- [[Component VaR]]
- [[Backtesting]]
- [[Stressed VaR]]

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

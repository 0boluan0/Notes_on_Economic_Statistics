---
aliases:
- ES
- Expected Shortfall
- CVaR
- Conditional VaR
- 预期损失
- 预期亏损
- 条件VaR
tags:
- concept
- 金融风险
- VaR
---
# ES

## 先记一句话

ES 就是：**已经超过 VaR 以后，尾部平均会亏多少**。

VaR 看分位点，ES 看分位点之外的尾部均值。

## 它是什么

若损失变量为 $L$，则
$$
ES_\alpha=E[L\mid L\geq VaR_\alpha].
$$

例如 99% ES 关心的是最坏 1% 情形下的平均损失。

因此通常：
$$
ES_\alpha\geq VaR_\alpha.
$$

## 它解决什么判断

ES 用来回答 VaR 没回答的问题：

> 如果真的进入尾部，平均会亏多惨？

它比 VaR 更适合描述极端损失。

## 正态情形

若损失
$$
L\sim N(\mu,\sigma^2),
$$
则
$$
ES_\alpha=\mu+\sigma\frac{\phi(z_\alpha)}{1-\alpha}.
$$

其中 $z_\alpha$ 是标准正态 $\alpha$ 分位点，$\phi$ 是标准正态密度。

## 和 VaR 的分工

- [[VaR]]：给出尾部入口的分位点。
- ES：给出进入尾部后的平均损失。

监管和压力测试更偏好 ES，因为它更关注尾部厚度，并且更符合相干风险度量要求。

## 常见误区

- ES 不是另一个置信水平下的 VaR。
- ES 需要估计尾部均值，对数据和模型要求更高。
- 正态公式只在正态损失假设下成立；厚尾分布下 ES 会更大。

## 来自课程位置

- [[22_情景分析和压力测试#VaR与预期亏损（ES）的关系|金融风险管理 22：VaR 与 ES]]

## 关联卡片

- [[VaR]]
- [[Coherent Risk Measure]]
- [[Spectral Risk Measure]]
- [[EVT VaR Calculation]]
- [[Stress Testing]]

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

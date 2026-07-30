---
aliases:
- Historical Volatility
- HV
- historical volatility
- 历史波动率
tags:
  - concept
  - 金融风险
  - 波动建模
---
# Historical Volatility

## 先记一句话

历史波动率就是：**用过去收益率的样本标准差估计未来波动率**。

它是最直观的 backward-looking volatility。

## 它是什么

先计算对数收益率：
$$
u_t=\ln\frac{S_t}{S_{t-1}}.
$$

再在一个历史窗口内计算标准差：
$$
s=\sqrt{\frac{1}{n-1}\sum_{t=1}^{n}(u_t-\bar u)^2}.
$$

日波动率年化时常用：
$$
\sigma_{\text{annual}}=s\sqrt{252}.
$$

## 它解决什么判断

历史波动率回答：

> 过去一段时间实际波动有多大？

它常作为 VaR 参数法、波动率比较、风险监控的基础输入。

## 常见误区

- 历史波动率默认过去代表未来，波动突变时反应慢。
- 窗口越长越平滑，但越迟钝；窗口越短越敏感，但更噪。
- 年化平方根规则隐含独立同分布或近似条件。

## 来自课程位置

- [[10_波动率|金融风险管理 10：历史波动率]]

## 关联卡片

- [[Implied Volatility]]
- [[Realized Volatility]]
- [[EWMA]]
- [[GARCH]]
- [[VaR]]


## 最小例子

把 **Historical Volatility** 放在最简单的可计算情形中：先给定定义所需的最小输入，再按定义计算输出；若关键关系不成立，就不能把该对象归入本概念。这个检查也能帮助区分相近概念。
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

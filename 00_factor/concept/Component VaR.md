---
aliases:
- Component VaR
- component Value at Risk
- risk contribution
- 成分VaR
- 风险贡献度
tags:
- concept
- 金融风险
- VaR
---

# Component VaR

## 先记一句话

成分 VaR 就是：**把组合总 VaR 分摊到每个头寸上，说明谁贡献了多少风险**。

它回答：

> 总 VaR 里，哪一部分是资产 $i$ 贡献的？

## 它是什么

在可微且一阶齐次的风险度量下，可以用 Euler 分解：
$$
VaR(w)=\sum_i w_i\frac{\partial VaR(w)}{\partial w_i}.
$$

其中
$$
w_i\frac{\partial VaR(w)}{\partial w_i}
$$
就是资产 $i$ 的 component VaR。

## 它解决什么判断

Component VaR 用于：

- 风险归因；
- 风险预算；
- 识别组合里最占风险资本的头寸；
- 调仓时决定先减哪里。

## 和边际/递增 VaR 的分工

- [[Marginal VaR]]：每增加一点的风险变化率。
- [[Incremental VaR]]：加入或删除一笔交易后的 VaR 差额。
- Component VaR：总 VaR 的风险归因。

## 常见误区

- Component VaR 不等于单资产 VaR。
- Component VaR 可以很大，即使资产自身波动不最大，因为它和组合高度同向。
- 分解成立依赖齐次性和局部可微等条件。

## 来自课程位置

- [[12_VAR风险|金融风险管理 12：VaR 分解]]

## 关联卡片

- [[VaR]]
- [[Marginal VaR]]
- [[Incremental VaR]]
- [[Position]]

## 边界与相关概念

本卡片只保留该概念的定义、记号与最小直觉；具体估计步骤、证明或完整应用应放在对应的 `procedure`、`proof` 或 `framework` 卡片中。

相关卡片：[[Marginal VaR]]、[[Incremental VaR]]、[[12_VAR风险]]、[[VaR]]、[[Position]]。

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

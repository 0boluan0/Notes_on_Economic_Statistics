---
aliases:
- Unsystematic Risk
- Idiosyncratic Risk
- Specific Risk
- 非系统风险
- 特质风险
tags:
- concept
- finance
- risk-management
---
# Unsystematic Risk

## 一句话记忆

Unsystematic risk 是由单个公司、行业或资产特有因素引起、可通过充分分散投资降低的风险。

## 它是什么

在单因子收益模型

$$
R_i-R_f=\alpha_i+\beta_i(R_m-R_f)+\varepsilon_i
$$

中，$\varepsilon_i$ 所代表的特质波动通常被视为非系统风险，其方差可写为 $\sigma^2_{\varepsilon_i}$。它不同于影响整个市场的 [[Systematic Risk]]。

## 最小例子

某公司产品召回导致该公司股价下跌，而市场整体未发生同方向变化，这是非系统风险。

## 易混点

- “可分散”不是说单一持仓没有风险，而是说组合中可通过持有不完全相关的资产降低其影响。
- 分散化主要消除特质风险，不能消除 [[Systematic Risk]]。
- 行业冲击可能同时影响多家公司，是否可分散取决于组合范围和风险定义。
>
## 关联卡片
- [[CAPM Estimation]]
- [[Systematic Risk]]
- 分散化投资（本仓库暂未单独建立对应卡片）

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

---
aliases:
- Bond and Stock Valuation
- 债券与股票估值
tags:
- procedure
---
# Bond and Stock Valuation

## 适用场景

对固定收益和权益资产进行公允估值、投资比较或敏感性分析。

## 所需数据/条件

- 现金流结构（票息、分红、终值）
- 折现率或收益率曲线
- 增长假设与估值口径（股利、自由现金流、估值倍数）

## 估值步骤

### 步骤 1：识别资产类型与现金流口径

- 债券：票息 + 到期本金。
- 股票：股利、FCFE/FCFF 或估值倍数。

### 步骤 2：确定折现率

- 债券：到期收益率或期限结构。
- 股票：权益资本成本（CAPM 等）。

### 步骤 3：计算债券现值

$$
P = \sum_{t=1}^{n} \frac{C_t}{(1+r)^t} + \frac{M}{(1+r)^n}
$$

### 步骤 4：计算股票价值

- 股利贴现模型（DDM）：
$$
P_0 = \frac{D_1}{r-g}
$$
- 或使用相对估值倍数（P/E、P/B）。

### 步骤 5：做敏感性分析

- 对 $r$、$g$、现金流假设进行区间测试。

## 输出物

- 估值表与关键假设说明。

## 常见误区

- 折现率与现金流口径不匹配。
- 忽略增长可持续性与资本结构变化。

## 相关链接

- [[Bond Valuation Model|债券估价模型]]
- [[Stock Valuation Model|股票估价模型]]
- [[CAPM|资本资产定价模型]]
- [[Price-to-Earnings Ratio|市盈率]]

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
